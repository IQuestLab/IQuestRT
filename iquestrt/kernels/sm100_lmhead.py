"""Recursive-draft LM head with row-scaled E4M3 weights and FP32 logits."""

import functools

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from .arch import cuda_arch

# Declarations only -- `load_inline(functions=...)` generates the pybind module.
_CPP = r"""
#include <torch/extension.h>
void draft_lmhead(torch::Tensor x, torch::Tensor w, torch::Tensor w_scale, torch::Tensor out);
"""

# K_DIM and N_DIM are substituted per shape by `_build`.
_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>

namespace {

constexpr int K = K_DIM, N = N_DIM;
constexpr int ROWS = 16;                    // vocabulary rows a CTA: one m16 MMA
constexpr int STAGES = 3, TILE = 256;       // cp.async ring depth; bytes of K a stage holds
constexpr int TILES = K / TILE;
constexpr int KB_TILE = TILE / 32;          // k32 blocks a tile
constexpr int CPT = ROWS * TILE / 16 / 32;  // 16-byte chunks a lane copies a tile
constexpr int RCH = TILE / 16;              // chunks a row of a tile
constexpr int RPI = 32 / RCH;               // rows one copy instruction covers
constexpr int STAGE_BYTES = ROWS * TILE;
constexpr int X_BYTES = 2 * K;              // the row in FP16
constexpr int SMEM = X_BYTES + STAGES * STAGE_BYTES;
constexpr int XC = K / 8 / 32;              // 8-value activation chunks a lane converts
static_assert(K % (TILE * STAGES) == 0 && N % ROWS == 0 && SMEM <= 48 * 1024, "shape");

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int PENDING>
__device__ __forceinline__ void cp_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(PENDING) : "memory");
}
// Orders the warp's shared-memory accesses; the compiler moves none across it.
__device__ __forceinline__ void warp_sync() { asm volatile("bar.warp.sync -1;" ::: "memory"); }
__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}
__device__ __forceinline__ void sts128(uint32_t addr, uint32_t a, uint32_t b, uint32_t c,
                                       uint32_t d) {
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(addr), "r"(a), "r"(b), "r"(c),
               "r"(d) : "memory");
}
// Two E4M3, the low or the high 16 bits of `a`, to two FP16: exact.
__device__ __forceinline__ uint32_t unpack_lo(uint32_t a) {
  uint32_t r;
  asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.rn.f16x2.e4m3x2 %0, l; }" : "=r"(r) : "r"(a));
  return r;
}
__device__ __forceinline__ uint32_t unpack_hi(uint32_t a) {
  uint32_t r;
  asm("{ .reg .b16 l, h; mov.b32 {l, h}, %1; cvt.rn.f16x2.e4m3x2 %0, h; }" : "=r"(r) : "r"(a));
  return r;
}
// Two packed BF16 to two E4M3, rounded to nearest and saturated to +-448, then to FP16.
__device__ __forceinline__ uint32_t cast_unpack(uint32_t bf16x2) {
  const float lo = __uint_as_float(bf16x2 << 16), hi = __uint_as_float(bf16x2 & 0xFFFF0000u);
  uint32_t r;
  asm("{ .reg .b16 e; cvt.rn.satfinite.e4m3x2.f32 e, %2, %1; cvt.rn.f16x2.e4m3x2 %0, e; }"
      : "=r"(r) : "f"(lo), "f"(hi));
  return r;
}
__device__ __forceinline__ void hmma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                     uint32_t b1, float c0, float c1, float c2, float c3) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
      "{%8,%9}, {%10,%11,%12,%13};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c0), "f"(c1),
        "f"(c2), "f"(c3));
}

__global__ void __launch_bounds__(32) draft_lmhead_kernel(
    const uint16_t* __restrict__ x, const uint8_t* __restrict__ w,
    const float* __restrict__ w_scale, float* __restrict__ out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x, g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.x * ROWS;
  const float scale0 = __ldg(w_scale + row0 + g), scale1 = __ldg(w_scale + row0 + g + 8);
  const uint32_t xs = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  const uint32_t ws = xs + X_BYTES;

  // The weight tiles: lane copies chunk c = lane % RCH of rows lane / RCH + i * RPI. Chunk
  // c of row r is stored at position c ^ (r & 7).
  const int cr = lane / RCH, cc = lane % RCH;
  const uint8_t* wsrc = w + static_cast<size_t>(row0 + cr) * K + cc * 16;
  uint32_t cdst[CPT];
#pragma unroll
  for (int i = 0; i < CPT; ++i) {
    const int r = cr + i * RPI;
    cdst[i] = ws + r * TILE + ((cc ^ (r & 7)) << 4);
  }
  auto issue = [&](int tile, int stage) {
    const uint8_t* src = wsrc + tile * TILE;
#pragma unroll
    for (int i = 0; i < CPT; ++i) cp_async16(cdst[i] + stage * STAGE_BYTES, src + i * RPI * K);
    cp_commit();
  };
#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) issue(s, s);

  // The row in FP16, in chunks of 8 values while the first tiles are in flight. Chunk c
  // holds lane c % 4's B registers of k32 block c / 4: the low HMMA's pair, then the
  // high's (k 4t, 4t+1, 16+4t, 16+4t+1, 4t+2, 4t+3, 16+4t+2, 16+4t+3 of the block), so
  // one 16-byte load gives a lane both pairs. Lane l converts chunks j * 32 + l.
  {
    const uint2* xq = reinterpret_cast<const uint2*>(x) + (lane >> 2) * 8 + (lane & 3);
    uint2 lo[XC], hi[XC];
#pragma unroll
    for (int j = 0; j < XC; ++j) {
      lo[j] = __ldg(xq + j * 64);      // k 4t..4t+3
      hi[j] = __ldg(xq + j * 64 + 4);  // k 16+4t..16+4t+3
    }
#pragma unroll
    for (int j = 0; j < XC; ++j)
      sts128(xs + (j * 32 + lane) * 16, cast_unpack(lo[j].x), cast_unpack(hi[j].x),
             cast_unpack(lo[j].y), cast_unpack(hi[j].y));
  }

  // ldmatrix x4: lanes 0-7 address rows 0-7 of a block's first 16 bytes, 8-15 rows 8-15,
  // 16-23 rows 0-7 of the second 16 bytes and 24-31 rows 8-15, giving the E4M3 A
  // registers of m16n8k32. Chunk 2*kk + h of row r sits at (2*kk + h) ^ (r & 7)
  // = 2*((kk & 3) ^ p) + (h ^ q) + 2*(kk & 4), with r & 7 = 2p + q.
  const int lr = (lane & 7) + ((lane >> 3) & 1) * 8, lh = lane >> 4;
  const int p = (lane & 7) >> 1, q = lane & 1;
  uint32_t la[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) la[j] = ws + lr * TILE + ((2 * (j ^ p) + (lh ^ q)) << 4);
  const uint32_t xb = xs + t * 16;
  float acc0 = 0.0f, acc2 = 0.0f;
#pragma unroll 1
  for (int base = 0; base < TILES; base += STAGES) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      const int tile = base + s;
      warp_sync();  // every lane is done reading the stage refilled next
      if (tile + STAGES - 1 < TILES) issue(tile + STAGES - 1, (s + STAGES - 1) % STAGES);
      else cp_commit();
      cp_wait<STAGES - 1>();
      warp_sync();  // and every lane's part of this tile has landed
      uint32_t a[KB_TILE][4];
      uint4 b[KB_TILE];
#pragma unroll
      for (int kk = 0; kk < KB_TILE; ++kk) {
        ldsm_x4(la[kk & 3] + (kk & 4) * 32 + s * STAGE_BYTES, a[kk]);
        b[kk] = lds128(xb + (tile * KB_TILE + kk) * 64);
      }
#pragma unroll
      for (int kk = 0; kk < KB_TILE; ++kk) {
        float d[4];
        const uint32_t alo[4] = {unpack_lo(a[kk][0]), unpack_lo(a[kk][1]),
                                 unpack_lo(a[kk][2]), unpack_lo(a[kk][3])};
        hmma(d, alo, b[kk].x, b[kk].y, 0.0f, 0.0f, 0.0f, 0.0f);
        const uint32_t ahi[4] = {unpack_hi(a[kk][0]), unpack_hi(a[kk][1]),
                                 unpack_hi(a[kk][2]), unpack_hi(a[kk][3])};
        hmma(d, ahi, b[kk].z, b[kk].w, d[0], d[1], d[2], d[3]);
        acc0 = acc0 + d[0];
        acc2 = acc2 + d[2];
      }
    }
  }
  if (t == 0) {
    out[row0 + g] = acc0 * scale0;
    out[row0 + g + 8] = acc2 * scale1;
  }
}

}  // namespace

void draft_lmhead(torch::Tensor x, torch::Tensor w, torch::Tensor w_scale, torch::Tensor out) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.numel() == K &&
              x.is_contiguous() && reinterpret_cast<uintptr_t>(x.data_ptr()) % 8 == 0,
              "draft LM head: x must be one contiguous, 8-byte aligned BF16 row of ", K);
  TORCH_CHECK(w.device() == x.device() && w.scalar_type() == at::kFloat8_e4m3fn &&
              w.dim() == 2 && w.size(0) == N && w.size(1) == K && w.is_contiguous() &&
              reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "draft LM head: w must be a contiguous, 16-byte aligned E4M3 [", N, ", ", K, "]");
  TORCH_CHECK(w_scale.device() == x.device() && w_scale.scalar_type() == at::kFloat &&
              w_scale.numel() == N && w_scale.is_contiguous(),
              "draft LM head: w_scale must be ", N, " contiguous fp32 row scales");
  TORCH_CHECK(out.device() == x.device() && out.scalar_type() == at::kFloat &&
              out.numel() == N && out.is_contiguous(),
              "draft LM head: out must be ", N, " contiguous fp32 logits");
  const c10::cuda::CUDAGuard guard(x.device());
  draft_lmhead_kernel<<<N / ROWS, 32, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint16_t*>(x.data_ptr()),
      reinterpret_cast<const uint8_t*>(w.data_ptr()), w_scale.data_ptr<float>(),
      out.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
"""


def _build(k: int, n: int):
    with cuda_arch():
        return load_inline(
            name=f"iquestrt_draft_lmhead_k{k}_n{n}",
            cpp_sources=_CPP,
            cuda_sources=_CUDA.replace("K_DIM", str(k)).replace("N_DIM", str(n)),
            functions=["draft_lmhead"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )


@functools.lru_cache(maxsize=None)
def build_draft_lmhead_fp8(k: int, n: int):
    """Q1 ``LMHead -> fp32 logits`` with row-scaled E4M3 weight storage, as
    ``fn(x, w, w_scale, out)``: one BF16 row ``x`` (1, k), the E4M3 copy ``w`` (n, k),
    its fp32 row scales ``w_scale`` (n,) and the fp32 logits ``out`` (1, n)."""
    if k % 768 or n % 16 or 2 * k + 3 * 16 * 256 > 48 * 1024:
        raise ValueError(
            f"the draft LM head needs K a multiple of 768 up to 18432 and N of 16; "
            f"got N={n} K={k}")
    if torch.cuda.get_device_capability() != (10, 0):
        raise RuntimeError("the draft LM head extension is built for compute capability 10.0")
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if dist.get_rank() == 0:
            ext = _build(k, n)
        dist.barrier()
        if dist.get_rank() != 0:
            ext = _build(k, n)
        dist.barrier()
    else:
        ext = _build(k, n)
    return ext.draft_lmhead
