"""Block-FP8 Gate+SiLU and Down kernels for SM100 TP8 decode."""

import functools

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from .arch import cuda_arch

EXPERTS, TOPK, HIDDEN, I_LOC, Q = 256, 8, 3072, 192, 8
NPAIR = Q * TOPK
# The MMA's M padding of the expert input and activation rows.
ROW_BM = 16


# Each rank keeps its scale table at the granularity its slice is block-uniform at
# (`iquestrt.blockfp8.shard_scale`): Gate/Up per (64 rows, 128 K), Down per (128 rows, 64 K).
KB = 128
GATE_ROWS = 64
DOWN_COLS = 64
# The Gate's CTA: 16 features of one routed pair, whichever launch shape (the verify
# covers its live rows, every draft depth one row), so fewer rows mean fewer CTAs, not
# other tiles.
GATE_BN = 16
DOWN_BN = 16


GATE_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace {

constexpr int EXPERTS = 256, TOPK = 8, HIDDEN = 3072, I_LOC = 192, ROW_BM = 16;
constexpr int KB = 128, KBLOCKS = HIDDEN / KB, GATE_ROWS = 64;
// The CTA's tile: its 16 gate rows, then its 16 up rows; warp 0 computes the gate half,
// warp 1 the up half.
constexpr int BN = 16, TILE_ROWS = 2 * BN, THREADS = 64;
// Twelve groups of two 128-K blocks; group g is K chunk g / 2 of split g % 2. While a
// warp computes one group, the next SLOTS - 1 are in flight.
constexpr int BPG = 2, GROUPS = KBLOCKS / BPG, SLOTS = 5;
constexpr int SLOT = TILE_ROWS * BPG * KB;
constexpr int X_OFF = SLOTS * SLOT;               // the pair's activation row
constexpr int S_OFF = X_OFF + HIDDEN;             // [gate, up] x 24 block scales
constexpr int L_OFF = S_OFF + 2 * KBLOCKS * 4;    // the live sums, [gate, up] x 16
constexpr int SMEM = L_OFF + TILE_ROWS * 4;

__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(s), "l"(src), "r"(valid ? 16 : 0) : "memory");
}

__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool valid) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
               :: "r"(s), "l"(src), "r"(valid ? 4 : 0) : "memory");
}

__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }

__device__ __forceinline__ unsigned lds32(const uint8_t* p) {
  return *reinterpret_cast<const unsigned*>(p);
}

__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const uint8_t* p) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(p));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

// The instruction TileLang's T.gemm emits for these E4M3 tiles: D = A * B + D, with
// A 16x32 row-major, B 32x8 column-major (PTX ISA, mma.m16n8k32 8-bit fragments).
__device__ __forceinline__ void mma_e4m3(float (&d)[4], const unsigned (&a)[4],
                                         const unsigned (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Byte offset in a slot of 16-byte chunk c of tile row r in the group's block j. The chunk
// index is XORed with the row, so ldmatrix's eight rows of one chunk hit distinct banks.
__device__ __forceinline__ int slot_offset(int j, int r, int c) {
  return ((j * TILE_ROWS + r) * 8 + (c ^ (r & 7))) * 16;
}

__device__ __forceinline__ int chunk_of(int g) { return (g >> 1) + (g & 1) * (GROUPS / 2); }

// Group g of the CTA's tile into `slot`: four 16-byte copies a thread, zeros for an
// expert id out of range.
__device__ __forceinline__ void issue(uint8_t* slot, const uint8_t* w_e, int tile, int g,
                                      bool valid, int tid) {
  const int k0 = chunk_of(g) * BPG * KB;
#pragma unroll
  for (int i = 0; i < TILE_ROWS * BPG * 8 / THREADS; ++i) {
    const int q = tid + i * THREADS;
    const int c = q & 7, r = (q >> 3) & (TILE_ROWS - 1), j = q >> 8;
    const int wrow = r < BN ? tile * BN + r : I_LOC + tile * BN + (r - BN);
    cp_async16(slot + slot_offset(j, r, c),
               w_e + static_cast<int64_t>(wrow) * HIDDEN + k0 + j * KB + c * 16, valid);
  }
}

// Grid (I_LOC / BN, rows * TOPK): 16 features of one routed pair per CTA.
__global__ void __launch_bounds__(THREADS, 1) moe_gate(
    const uint8_t* __restrict__ x, const uint8_t* __restrict__ w,
    const float* __restrict__ w_scale, const int* __restrict__ sel,
    uint8_t* __restrict__ act) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int tile = blockIdx.x, pair = blockIdx.y, row = pair / TOPK;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  // Every input but the weights comes from the Router or before it.
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory");
  const int expert = sel[pair];
  const bool valid = expert >= 0 && expert < EXPERTS;
  const uint8_t* w_e = w + static_cast<int64_t>(valid ? expert : 0) * 2 * I_LOC * HIDDEN;
  // The first copy group: group 0, the activation row and the tile's block scales.
  issue(smem, w_e, tile, 0, valid, tid);
#pragma unroll
  for (int i = 0; i < HIDDEN / 16 / THREADS; ++i)
    cp_async16(smem + X_OFF + (tid + i * THREADS) * 16,
               x + static_cast<int64_t>(row) * HIDDEN + (tid + i * THREADS) * 16, true);
  if (tid < 2 * KBLOCKS) {
    const int half = tid / KBLOCKS, b = tid % KBLOCKS;
    const int group = (half * I_LOC + tile * BN) / GATE_ROWS;
    cp_async4(smem + S_OFF + tid * 4,
              w_scale + (static_cast<int64_t>(valid ? expert : 0) * (2 * I_LOC / GATE_ROWS)
                         + group) * KBLOCKS + b, valid);
  }
  commit();
#pragma unroll
  for (int g = 1; g < SLOTS - 1; ++g) {
    issue(smem + g * SLOT, w_e, tile, g, valid, tid);
    commit();
  }
  // Fragments: lane = 4 * group + t; ldmatrix's lane l addresses row l % 8 (+ 8 for the
  // odd matrices) of chunk 2m (+ 1 for the upper two); B column group (only column 0
  // carries the row, so the other groups pass zeros).
  const int group = lane >> 2, t = lane & 3, mat = lane >> 3;
  const int mrow = warp * BN + (lane & 7) + (mat & 1) * 8, khalf = mat >> 1;
  const float* scale = reinterpret_cast<const float*>(smem + S_OFF) + warp * KBLOCKS;
  float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
  for (int g = 0; g < GROUPS; ++g) {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(SLOTS - 2) : "memory");
    // Group g has landed for every thread, and every warp is past group g - 1.
    __syncthreads();
    if (g + SLOTS - 1 < GROUPS)
      issue(smem + ((g + SLOTS - 1) % SLOTS) * SLOT, w_e, tile, g + SLOTS - 1, valid, tid);
    commit();
    const uint8_t* slot = smem + (g % SLOTS) * SLOT;
    const int split = g & 1;
#pragma unroll
    for (int j = 0; j < BPG; ++j) {
      const int b = chunk_of(g) * BPG + j;
      float part[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        unsigned a[4];
        ldsm_x4(a, slot + slot_offset(j, mrow, 2 * m + khalf));
        const uint8_t* xb = smem + X_OFF + b * KB + m * 32 + t * 4;
        const unsigned bf[2] = {group == 0 ? lds32(xb) : 0u, group == 0 ? lds32(xb + 16) : 0u};
        mma_e4m3(part, a, bf);
      }
      const float s = scale[b];
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[split][i] = fmaf(part[i], s, acc[split][i]);
    }
  }
  float* live = reinterpret_cast<float*>(smem + L_OFF);
  if (t == 0) {
    live[warp * BN + group] = acc[0][0] + acc[1][0];
    live[warp * BN + group + 8] = acc[0][2] + acc[1][2];
  }
  __syncthreads();
  if (tid < BN) {
    const float gate = live[tid], up = live[BN + tid];
    const float v = gate / (1.0f + expf(gate * -1.0f)) * up;
    act[(static_cast<int64_t>(pair) * ROW_BM + row) * I_LOC + tile * BN + tid] =
        static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
  }
}

}  // namespace

void moe_gate_launch(int64_t x, int64_t w, int64_t w_scale, int64_t sel, int64_t act,
                     int64_t rows) {
  TORCH_CHECK(rows >= 1 && rows <= 8, "the Gate covers 1..8 rows, got ", rows);
  static const bool sized = [] {
    TORCH_CHECK(cudaFuncSetAttribute(moe_gate, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     SMEM) == cudaSuccess, "MoE Gate shared memory");
    return true;
  }();
  (void)sized;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(I_LOC / BN, static_cast<unsigned>(rows * TOPK), 1);
  config.blockDim = dim3(THREADS, 1, 1);
  config.dynamicSmemBytes = SMEM;
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &config, moe_gate, reinterpret_cast<const uint8_t*>(x),
      reinterpret_cast<const uint8_t*>(w), reinterpret_cast<const float*>(w_scale),
      reinterpret_cast<const int*>(sel), reinterpret_cast<uint8_t*>(act));
  TORCH_CHECK(err == cudaSuccess, "MoE Gate launch: ", cudaGetErrorString(err));
}
"""


DOWN_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int EXPERTS = 256, TOPK = 8, HIDDEN = 3072, I_LOC = 192, ROW_BM = 16;
constexpr int BN = 16, KB = 128, COLS = 64, GROUPS = I_LOC / COLS;
// 208-byte shared rows put the eight MMA row groups' 4-byte reads on distinct banks. The
// 16 bytes past each row's 192 hold the fold's parts and scales (`pad`), so a CTA takes
// 28160 B and eight CTAs share an SM.
constexpr int ROW_STRIDE = I_LOC + 16;
constexpr int SCALES = GROUPS * BN;     // the first scale's `pad` index
constexpr int THREADS = 32 * TOPK;      // warp s computes routed slot s

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(dst));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(s), "l"(src) : "memory");
}

__device__ __forceinline__ unsigned lds32(const uint8_t* p) {
  return *reinterpret_cast<const unsigned*>(p);
}

// The instruction TileLang's T.gemm emits for these E4M3 tiles: D = A * B + D, with
// A 16x32 row-major, B 32x8 column-major (PTX ISA, mma.m16n8k32 8-bit fragments).
__device__ __forceinline__ void mma_e4m3(float (&d)[4], const unsigned (&a)[4],
                                         const unsigned (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// Float i of a tile's padding, four per row: the part of group g and feature f at
// g * BN + f, the scale of group g at SCALES + g.
__device__ __forceinline__ float* pad(uint8_t* tile, int i) {
  return reinterpret_cast<float*>(tile + (i >> 2) * ROW_STRIDE + I_LOC) + (i & 3);
}

// Grid (HIDDEN / BN, rows): 16 hidden features of one query row per CTA. `sel` is not
// __restrict__, so its read after the wait stays a second read.
__global__ void __launch_bounds__(THREADS) moe_down(
    const uint8_t* __restrict__ act, const uint8_t* __restrict__ w,
    const float* __restrict__ w_scale, const int* sel,
    const float* __restrict__ wt, __nv_bfloat16* __restrict__ out) {
  __shared__ __align__(16) uint8_t tile_s[TOPK][BN * ROW_STRIDE];
  __shared__ __align__(16) uint8_t act_s[TOPK][I_LOC];

  const int tile = blockIdx.x, row = blockIdx.y;
  const int slot = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int pair = row * TOPK + slot;
  // Before the wait: the weights are static, and the id only chooses the tile to fetch;
  // it is confirmed after the wait (the module's PDL chain).
  const int expert = __ldg(sel + pair);
  const bool valid = expert >= 0 && expert < EXPERTS;
  uint8_t* tile_w = tile_s[slot];
  // The expert's 16 x 192 tile is one contiguous 3 KB run of w[expert].
  if (valid) {
    const uint8_t* src = w + (static_cast<int64_t>(expert) * HIDDEN + tile * BN) * I_LOC;
    for (int i = lane; i < BN * I_LOC / 16; i += 32)
      cp_async16(tile_w + (i / (I_LOC / 16)) * ROW_STRIDE + (i % (I_LOC / 16)) * 16,
                 src + i * 16);
  } else {
    for (int i = lane; i < BN * I_LOC / 16; i += 32)
      *reinterpret_cast<uint4*>(tile_w + (i / (I_LOC / 16)) * ROW_STRIDE
                                + (i % (I_LOC / 16)) * 16) = make_uint4(0u, 0u, 0u, 0u);
  }
  // After it: the pair's activation, the Gate's output, in row `row` of its 16-row block.
  asm volatile("griddepcontrol.wait;\n" ::: "memory");
  if (lane < I_LOC / 16)
    cp_async16(act_s[slot] + lane * 16,
               act + (static_cast<int64_t>(pair) * ROW_BM + row) * I_LOC + lane * 16);
  const int routed = sel[pair];
  // The scales under the copies.
  if (lane < GROUPS) {
    const float s = valid ? w_scale[(static_cast<int64_t>(expert) * (HIDDEN / KB)
                                     + tile * BN / KB) * GROUPS + lane] : 0.f;
    *pad(tile_w, SCALES + lane) = s * wt[pair];
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::: "memory");
  __syncwarp();
  // Fragments: lane = 4 * group + t; A rows group and group + 8, B column group (only
  // column 0 carries the row, so the other groups pass zeros), D column 0 at t == 0.
  const int group = lane >> 2, t = lane & 3;
  const uint8_t* a_lo = tile_w + group * ROW_STRIDE;
  const uint8_t* a_hi = tile_w + (group + 8) * ROW_STRIDE;
#pragma unroll
  for (int g = 0; g < GROUPS; ++g) {
    float d[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int ki = 0; ki < 2; ++ki) {
      const int k = g * COLS + ki * 32 + t * 4;
      const unsigned a[4] = {lds32(a_lo + k), lds32(a_hi + k), lds32(a_lo + k + 16),
                             lds32(a_hi + k + 16)};
      const unsigned b[2] = {group == 0 ? lds32(act_s[slot] + k) : 0u,
                             group == 0 ? lds32(act_s[slot] + k + 16) : 0u};
      mma_e4m3(d, a, b);
    }
    if (t == 0) {
      *pad(tile_w, g * BN + group) = d[0];
      *pad(tile_w, g * BN + group + 8) = d[2];
    }
  }
  // The tile must be the routed expert's before anything is stored.
  if (routed != expert) __trap();
  __syncthreads();
  if (threadIdx.x < BN) {
    // The fold: slots in order, groups in order, one fma each.
    const int f = threadIdx.x;
    float total = 0.f;
#pragma unroll
    for (int s = 0; s < TOPK; ++s)
#pragma unroll
      for (int g = 0; g < GROUPS; ++g)
        total = fmaf(*pad(tile_s[s], g * BN + f), *pad(tile_s[s], SCALES + g), total);
    out[row * HIDDEN + tile * BN + f] = __float2bfloat16_rn(total);
  }
}

}  // namespace

void moe_down_launch(int64_t act, int64_t w, int64_t w_scale, int64_t sel, int64_t wt,
                     int64_t out, int64_t rows) {
  TORCH_CHECK(rows >= 1 && rows <= 8, "the Down covers 1..8 rows, got ", rows);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(HIDDEN / BN, static_cast<unsigned>(rows), 1);
  config.blockDim = dim3(THREADS, 1, 1);
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &config, moe_down, reinterpret_cast<const uint8_t*>(act),
      reinterpret_cast<const uint8_t*>(w), reinterpret_cast<const float*>(w_scale),
      reinterpret_cast<const int*>(sel), reinterpret_cast<const float*>(wt),
      reinterpret_cast<__nv_bfloat16*>(out));
  TORCH_CHECK(err == cudaSuccess, "MoE Down launch: ", cudaGetErrorString(err));
}
"""

_DOWN_CPP = ("void moe_down_launch(int64_t act, int64_t w, int64_t w_scale, int64_t sel, "
             "int64_t wt, int64_t out, int64_t rows);")
_GATE_CPP = ("void moe_gate_launch(int64_t x, int64_t w, int64_t w_scale, int64_t sel, "
             "int64_t act, int64_t rows);")
_EXTENSIONS = {}


def _extension(name: str, cpp: str, cuda: str, function: str, arch: str = "10.0"):
    """Compile once per process, on rank 0 first, then let the others load from cache, as
    `sampler._ext` does: ranks building at once contend on torch's build lock. `arch`
    is the CUDA architecture for this build (tcgen05 needs "10.0a")."""
    if name not in _EXTENSIONS:
        def build():
            with cuda_arch(arch):
                return load_inline(name=name, cpp_sources=cpp, cuda_sources=cuda,
                                   functions=[function], extra_cuda_cflags=["-O3"],
                                   verbose=False)

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            if dist.get_rank() == 0:
                _EXTENSIONS[name] = build()
            dist.barrier()
            if dist.get_rank() != 0:
                _EXTENSIONS[name] = build()
            dist.barrier()
        else:
            _EXTENSIONS[name] = build()
    return _EXTENSIONS[name]


def _check_operands(kernel: str, operands):
    for name, t, dtype, shape in operands:
        if t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous():
            raise ValueError(f"the {kernel}'s {name} must be contiguous {dtype}{list(shape)}, "
                             f"got {t.dtype}{list(t.shape)}")


def _gate_edit(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"the Gate's source must hold {old!r} once")
    return source.replace(old, new)


# The Gate from seven rows (`build_gate_silu_block`, SEVEN AND EIGHT ROWS TAKE ANOTHER
# BUILD): the group loop rolled by two, four ring slots, six CTAs an SM.
GATE_WIDE_CUDA = _gate_edit(_gate_edit(_gate_edit(
    GATE_CUDA,
    "#pragma unroll\n  for (int g = 0; g < GROUPS; ++g) {",
    "#pragma unroll 2\n  for (int g = 0; g < GROUPS; ++g) {"),
    "GROUPS = KBLOCKS / BPG, SLOTS = 5;", "GROUPS = KBLOCKS / BPG, SLOTS = 4;"),
    "__launch_bounds__(THREADS, 1)", "__launch_bounds__(THREADS, 6)")
GATE_WIDE_ROWS = 7     # the fewest rows that launch it


def gate_build(rows: int) -> tuple[str, str]:
    """The Gate's (module name, CUDA source) for a launch over `rows` rows."""
    if not 1 <= rows <= Q:
        raise ValueError(f"the block Gate covers 1..{Q} rows, got {rows}")
    if rows < GATE_WIDE_ROWS:
        return "iquestrt_moe_gate", GATE_CUDA
    return "iquestrt_moe_gate_wide", GATE_WIDE_CUDA


class GateBlock:
    """The Gate over the first `rows` query rows: `(x_fp8, w_fp8, w_scale, sel, act)`."""

    def __init__(self, rows: int):
        self.rows = rows
        name, source = gate_build(rows)
        self._ext = _extension(name, _GATE_CPP, source, "moe_gate_launch")

    def __call__(self, x_fp8, w_fp8, w_scale, sel, act):
        _check_operands("Gate", (
            ("x_fp8", x_fp8, torch.float8_e4m3fn, (ROW_BM, HIDDEN)),
            ("w_fp8", w_fp8, torch.float8_e4m3fn, (EXPERTS, 2 * I_LOC, HIDDEN)),
            ("w_scale", w_scale, torch.float32, (EXPERTS, 2 * I_LOC // GATE_ROWS, HIDDEN // KB)),
            ("sel", sel, torch.int32, (NPAIR,)),
            ("act", act, torch.float8_e4m3fn, (NPAIR, ROW_BM, I_LOC)),
        ))
        self._ext.moe_gate_launch(x_fp8.data_ptr(), w_fp8.data_ptr(), w_scale.data_ptr(),
                                  sel.data_ptr(), act.data_ptr(), self.rows)


@functools.lru_cache(maxsize=None)
def build_gate_silu_block(rows: int = Q):
    """Block-FP8 Gate/Up and SiLU*Up for the routed pairs of the first `rows` query rows."""
    if not 1 <= rows <= Q:
        raise ValueError(f"the block Gate covers 1..{Q} rows, got {rows}")
    return GateBlock(rows), I_LOC // GATE_BN * rows * TOPK


class DownBlock:
    """The Down over the first `rows` query rows: `(act_fp8, w_fp8, w_scale, sel, wt, out)`."""

    def __init__(self, rows: int):
        self.rows = rows
        self._ext = _extension("iquestrt_moe_down", _DOWN_CPP, DOWN_CUDA, "moe_down_launch")

    def __call__(self, act_fp8, w_fp8, w_scale, sel, wt, out):
        _check_operands("Down", (
            ("act_fp8", act_fp8, torch.float8_e4m3fn, (NPAIR, ROW_BM, I_LOC)),
            ("w_fp8", w_fp8, torch.float8_e4m3fn, (EXPERTS, HIDDEN, I_LOC)),
            ("w_scale", w_scale, torch.float32, (EXPERTS, HIDDEN // KB, I_LOC // DOWN_COLS)),
            ("sel", sel, torch.int32, (NPAIR,)),
            ("wt", wt, torch.float32, (NPAIR,)),
            ("out", out, torch.bfloat16, (ROW_BM, HIDDEN)),
        ))
        self._ext.moe_down_launch(act_fp8.data_ptr(), w_fp8.data_ptr(), w_scale.data_ptr(),
                                  sel.data_ptr(), wt.data_ptr(), out.data_ptr(), self.rows)


@functools.lru_cache(maxsize=None)
def build_down_block(rows: int = Q):
    """Block-FP8 Down over each of the first `rows` query rows' eight experts."""
    if not 1 <= rows <= Q:
        raise ValueError(f"the block Down covers 1..{Q} rows, got {rows}")
    return DownBlock(rows), HIDDEN // DOWN_BN * rows
