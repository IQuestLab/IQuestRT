"""Long-context full attention on SM100 tensor cores.

TMA fills a four-tile K/V ring; separate warps issue MMAs and update online softmax.
Splits read only live positions. Include the zero-value sink once and combine
FP32 partials without changing their scaling convention."""

import functools

import torch

from .sm100_moe_fp8 import _check_operands, _extension

Q, HEADS, DIM = 8, 6, 128
TILE = 64
# A row's reference max moves when a tile's max passes it by more than this (base 2).
RESCALE_TAU = 8.0

STREAM_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int Q = 8, HEADS = 6, DIM = 128, BK = 64, THREADS = 160, STAGES = 4;
// Shared memory, all in the 128-byte swizzle that TMA writes and the MMAs read: Q as
// [2 k-blocks][64 rows][128 B], two P buffers of [64 rows][128 B], then the ring, a
// stage holding K then V as [2 k-blocks][64 positions][128 B].
constexpr int Q_OFF = 0, P_OFF = 16384, KV_OFF = 32768, STAGE_BYTES = 32768;
constexpr int BAR_OFF = KV_OFF + STAGES * STAGE_BYTES;
constexpr int SMEM = BAR_OFF + 256;
// TMEM: two S buffers of 64 columns, then O's 128.
constexpr uint32_t TMEM_COLS = 256, S_COL = 0, O_COL = 128;
constexpr float NEG_INF = -3.0e38f, NEG_GUARD = -1.5e38f, TAU = RESCALE_TAU_F;
constexpr float LOG2E = 1.4426950408889634f, LN2 = 0.6931471805599453f;

// full / vfull: a stage's K / V landed; empty: the PV that read it completed; s_full /
// s_free: an S buffer was written / read by the three softmax warps; p_full: a P buffer
// was written; pv_done: the PV that read a P buffer completed.
struct Bars {
  uint64_t full[STAGES], empty[STAGES], vfull[STAGES];
  uint64_t q_full, s_full[2], s_free[2], p_full[2], pv_done[2];
  uint32_t tmem;
};
static_assert(sizeof(Bars) <= 256, "the barriers fit");

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(smem_u32(bar)), "r"(count)
               : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
               :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity) {
  asm volatile("{\n.reg .pred p;\nLAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
               "@!p bra LAB_WAIT;\n}\n" :: "r"(smem_u32(bar)), "r"(parity) : "memory");
}
__device__ __forceinline__ void tma_2d(void* dst, const CUtensorMap* map, uint64_t* bar,
                                       int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
               " [%0], [%1, {%3, %4}], [%2];\n"
               :: "r"(smem_u32(dst)), "l"(map), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
               : "memory");
}
__device__ __forceinline__ void prefetch_map(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];\n" :: "l"(reinterpret_cast<uint64_t>(map)) : "memory");
}
// The SM100 shared-memory matrix descriptor of a 128-byte-swizzled operand.
__device__ __forceinline__ uint64_t sw128(uint32_t addr, uint32_t lbo, uint32_t sbo) {
  return (uint64_t)((addr & 0x3FFFFu) >> 4) | ((uint64_t)(lbo >> 4) << 16) |
         ((uint64_t)(sbo >> 4) << 32) | (1ull << 46) | (2ull << 61);
}
__device__ __forceinline__ void umma(uint32_t d, uint64_t a, uint64_t b, uint32_t idesc,
                                     uint32_t accumulate) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
               "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n}\n"
               :: "r"(d), "l"(a), "l"(b), "r"(idesc), "r"(accumulate) : "memory");
}
__device__ __forceinline__ void umma_commit(uint64_t* bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n"
               :: "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void tc_before() {
  asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
}
__device__ __forceinline__ void tc_after() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}
__device__ __forceinline__ void tmem_wait_ld() {
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void tmem_wait_st() {
  asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
}
TMEM_ACCESS
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// The kind::f16 instruction descriptor: F32 D, BF16 A and B, K-major A, B K-major (0) or
// MN-major (1).
constexpr uint32_t idesc(int m, int n, int b_mn) {
  return (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)b_mn << 16) |
         ((uint32_t)(n >> 3) << 17) | ((uint32_t)(m >> 4) << 24);
}
constexpr uint32_t IDESC_QK = idesc(64, 64, 0), IDESC_PV = idesc(64, 128, 1);

__global__ void __launch_bounds__(THREADS, 1) attn_stream(
    const __grid_constant__ CUtensorMap qmap, const __grid_constant__ CUtensorMap kmap,
    const __grid_constant__ CUtensorMap vmap, const __nv_bfloat16* __restrict__ sink,
    const int64_t* __restrict__ qpos, const float* __restrict__ scale,
    float* __restrict__ pm, float* __restrict__ pl, float* __restrict__ pacc, int n_split) {
  extern __shared__ __align__(1024) char smem[];
  Bars& bar = *reinterpret_cast<Bars*>(smem + BAR_OFF);
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; ++i) {
      mbar_init(&bar.full[i], 1);
      mbar_init(&bar.vfull[i], 1);
      mbar_init(&bar.empty[i], 1);
    }
    mbar_init(&bar.q_full, 1);
    for (int b = 0; b < 2; ++b) {
      mbar_init(&bar.s_full[b], 1);
      mbar_init(&bar.pv_done[b], 1);
      mbar_init(&bar.s_free[b], 3);
      mbar_init(&bar.p_full[b], 3);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (warp == 3) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
                 :: "r"(smem_u32(&bar.tmem)), "r"(TMEM_COLS));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n");
  }
  if (warp == 4) {
    if (lane == 0) {
      prefetch_map(&qmap);
      prefetch_map(&kmap);
      prefetch_map(&vmap);
    }
    // P's rows 48-63 feed only TMEM lanes nobody reads; they stay zero.
    for (int i = lane; i < 2 * 16 * 8; i += 32) {
      const int b = i >> 7, r = 48 + ((i >> 3) & 15), c = i & 7;
      *reinterpret_cast<uint4*>(smem + P_OFF + b * 8192 + r * 128 + c * 16) =
          make_uint4(0, 0, 0, 0);
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
  }
  tc_before();
  __syncthreads();
  tc_after();
  const uint32_t tmem = bar.tmem;
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  const int split = blockIdx.x;
  const int qp0 = (int)qpos[0];
  const int tiles = (qp0 + Q + BK - 1) / BK;
  const int per = (tiles + n_split - 1) / n_split;
  const int t0 = split * per;
  const int nt = max(min(t0 + per, tiles) - t0, 0);

  if (warp == 4) {
    if (lane == 0) {
      mbar_expect_tx(&bar.q_full, 16384);
      tma_2d(smem + Q_OFF, &qmap, &bar.q_full, 0, 0);
      tma_2d(smem + Q_OFF + 8192, &qmap, &bar.q_full, 64, 0);
      for (int t = 0; t < nt; ++t) {
        const int s = t % STAGES;
        if (t >= STAGES) mbar_wait(&bar.empty[s], ((t / STAGES) - 1) & 1);
        char* st = smem + KV_OFF + s * STAGE_BYTES;
        const int pos = (t0 + t) * BK;
        mbar_expect_tx(&bar.full[s], STAGE_BYTES / 2);
        mbar_expect_tx(&bar.vfull[s], STAGE_BYTES / 2);
        tma_2d(st, &kmap, &bar.full[s], 0, pos);
        tma_2d(st + 8192, &kmap, &bar.full[s], 64, pos);
        tma_2d(st + 16384, &vmap, &bar.vfull[s], 0, pos);
        tma_2d(st + 24576, &vmap, &bar.vfull[s], 64, pos);
      }
    }
    __syncwarp();
  } else if (warp == 3) {
    if (lane == 0) {
      const uint32_t qa = smem_u32(smem + Q_OFF);
      mbar_wait(&bar.q_full, 0);
      for (int t = 0; t <= nt; ++t) {
        if (t < nt) {
          // S_t = Q K_t^T: 8 k-steps of 16 dims, four to a 128-byte row of a k-block.
          const int s = t % STAGES, b = t & 1;
          mbar_wait(&bar.full[s], (t / STAGES) & 1);
          if (t >= 2) mbar_wait(&bar.s_free[b], ((t >> 1) - 1) & 1);
          tc_after();
          const uint32_t kb = smem_u32(smem + KV_OFF + s * STAGE_BYTES);
#pragma unroll
          for (int j = 0; j < 8; ++j)
            umma(tmem + S_COL + 64 * b, sw128(qa + (j >> 2) * 8192 + (j & 3) * 32, 16, 1024),
                 sw128(kb + (j >> 2) * 8192 + (j & 3) * 32, 16, 1024), IDESC_QK, j > 0);
          umma_commit(&bar.s_full[b]);
        }
        if (t >= 1) {
          // O += P_u V_u: 4 k-steps of 16 positions; V is the MN-major operand, its two
          // 64-dim blocks 8192 bytes apart, its 8-position groups 1024.
          const int u = t - 1, s = u % STAGES, b = u & 1;
          mbar_wait(&bar.p_full[b], (u >> 1) & 1);
          mbar_wait(&bar.vfull[s], (u / STAGES) & 1);
          tc_after();
          const uint32_t vb = smem_u32(smem + KV_OFF + s * STAGE_BYTES + 16384);
          const uint32_t pb = smem_u32(smem + P_OFF + b * 8192);
#pragma unroll
          for (int j = 0; j < 4; ++j)
            umma(tmem + O_COL, sw128(pb + j * 32, 16, 1024), sw128(vb + j * 2048, 8192, 1024),
                 IDESC_PV, (u > 0 || j > 0) ? 1u : 0u);
          umma_commit(&bar.pv_done[b]);
          umma_commit(&bar.empty[s]);
        }
      }
      // The last commit's arrival lands before the CTA retires.
      if (nt > 0) mbar_wait(&bar.empty[(nt - 1) % STAGES], ((nt - 1) / STAGES) & 1);
    }
    __syncwarp();
    // The softmax warps have loaded O: TMEM is free.
    asm volatile("bar.sync 1, 128;\n" ::: "memory");
    tc_after();
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                 :: "r"(tmem), "r"(TMEM_COLS));
  } else {
    const int row = 16 * warp + (lane & 15), half = lane >> 4;
    const int qi = row / HEADS;
    const float sl2 = scale[0] * LOG2E;
    const uint32_t tl = tmem + ((uint32_t)(32 * warp) << 16);
    char* prow = smem + P_OFF + row * 128;
    float m = NEG_INF, l = 0.f;
    if (split == 0) {
      // The sink: one more logit, q. sink_k, with no value, folded before the first tile.
      mbar_wait(&bar.q_full, 0);
      const char* qrow = smem + Q_OFF + half * 8192 + row * 128;
      const uint4* sk = reinterpret_cast<const uint4*>(sink + 64 * half);
      float d = 0.f;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const uint4 qv = *reinterpret_cast<const uint4*>(qrow + ((c ^ (row & 7)) << 4));
        const uint4 sv = sk[c];
        const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w}, sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&qw[i]));
          const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&sw[i]));
          d += a.x * b.x + a.y * b.y;
        }
      }
      d += __shfl_xor_sync(0xffffffffu, d, 16);
      m = d * sl2;
      // The row's two threads add their sums at the end: the sink's 2^0 counts once.
      l = half ? 0.f : 1.f;
    }
    for (int t = 0; t < nt; ++t) {
      const int b = t & 1;
      mbar_wait(&bar.s_full[b], (t >> 1) & 1);
      tc_after();
      uint32_t s[32];
      tmem_ld_x32(tl + S_COL + 64 * b, s);
      tmem_wait_ld();
      tc_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&bar.s_free[b]);
      // Only the tile holding the step's positions masks: masked logits become -inf.
      const int p0 = (t0 + t) * BK + 32 * half;
      if ((t0 + t) * BK + BK - 1 > qp0) {
#pragma unroll
        for (int j = 0; j < 32; ++j)
          if (p0 + j > qp0 + qi) s[j] = __float_as_uint(-INFINITY);
      }
      float mx = __uint_as_float(s[0]);
#pragma unroll
      for (int j = 1; j < 32; ++j) mx = fmaxf(mx, __uint_as_float(s[j]));
      mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 16));
      // The tile's max in base-2 units (scale > 0 keeps where the max is).
      const float tmx = mx == -INFINITY ? NEG_INF : mx * sl2;
      // The reference max moves only past TAU; O then rescales by 2^(m - tmx). The guard
      // keeps the sentinel out of every exponential while a row has seen nothing.
      const bool move = tmx > m + TAU;
      const bool rescale = move && m > NEG_GUARD;
      if (__any_sync(0xffffffffu, rescale)) {
        const float a = rescale ? ex2(m - tmx) : 1.f;
        if (t > 0) {
          // O holds PV_0.. PV_{t-1}; the last of them must have landed.
          mbar_wait(&bar.pv_done[(t - 1) & 1], ((t - 1) >> 1) & 1);
          tc_after();
          uint32_t o[64];
          tmem_ld_x64(tl + O_COL, o);
          tmem_wait_ld();
#pragma unroll
          for (int j = 0; j < 64; ++j) o[j] = __float_as_uint(__uint_as_float(o[j]) * a);
          tmem_st_x64(tl + O_COL, o);
          tmem_wait_st();
        }
        l *= a;
      }
      if (move) m = tmx;
      // P buffer b was read by PV_{t-2}.
      if (t >= 2) mbar_wait(&bar.pv_done[b], ((t >> 1) - 1) & 1);
      const bool seen = m > NEG_GUARD;
      float rs = 0.f;
      uint32_t pk[16];
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float e0 = seen ? ex2(fmaf(__uint_as_float(s[2 * j]), sl2, -m)) : 0.f;
        const float e1 = seen ? ex2(fmaf(__uint_as_float(s[2 * j + 1]), sl2, -m)) : 0.f;
        rs += e0 + e1;
        pk[j] = pack_bf16(e0, e1);
      }
      l += rs;
      char* pb = prow + b * 8192;
#pragma unroll
      for (int c = 0; c < 4; ++c)
        *reinterpret_cast<uint4*>(pb + (((4 * half + c) ^ (row & 7)) << 4)) =
            make_uint4(pk[4 * c], pk[4 * c + 1], pk[4 * c + 2], pk[4 * c + 3]);
      asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
      tc_before();
      __syncwarp();
      if (lane == 0) mbar_arrive(&bar.p_full[b]);
    }
    float acc[64];
    if (nt > 0) {
      mbar_wait(&bar.pv_done[(nt - 1) & 1], ((nt - 1) >> 1) & 1);
      tc_after();
      uint32_t o[64];
      tmem_ld_x64(tl + O_COL, o);
      tmem_wait_ld();
#pragma unroll
      for (int j = 0; j < 64; ++j) acc[j] = __uint_as_float(o[j]);
    } else {
#pragma unroll
      for (int j = 0; j < 64; ++j) acc[j] = 0.f;
    }
    // O is loaded: warp 3 may free TMEM.
    tc_before();
    asm volatile("bar.arrive 1, 128;\n" ::: "memory");
    l += __shfl_xor_sync(0xffffffffu, l, 16);
    // Partials in natural-log units at [query, head, split]; row = 6 query + head.
    const int64_t o = (int64_t)row * n_split + split;
    if (half == 0) {
      pm[o] = m > NEG_GUARD ? m * LN2 : NEG_INF;
      pl[o] = l;
    }
    // The warp's 16 rows of O through the idle ring (16-byte chunks XOR-swizzled by row),
    // then one 512-byte row a store.
    char* stage = smem + KV_OFF + warp * 8192;
    const int r16 = lane & 15;
#pragma unroll
    for (int j = 0; j < 16; ++j)
      *reinterpret_cast<float4*>(stage + r16 * 512 + (((16 * half + j) ^ (r16 & 7)) << 4)) =
          make_float4(acc[4 * j], acc[4 * j + 1], acc[4 * j + 2], acc[4 * j + 3]);
    __syncwarp();
#pragma unroll 4
    for (int r = 0; r < 16; ++r) {
      const float4 x = *reinterpret_cast<const float4*>(stage + r * 512 + ((lane ^ (r & 7)) << 4));
      *reinterpret_cast<float4*>(pacc + ((int64_t)(16 * warp + r) * n_split + split) * DIM +
                                 4 * lane) = x;
    }
  }
}

PFN_cuTensorMapEncodeTiled_v12000 encode_tiled() {
  static const PFN_cuTensorMapEncodeTiled_v12000 fn = [] {
    void* f = nullptr;
    cudaDriverEntryPointQueryResult found;
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &f, 12000,
                                                 cudaEnableDefault, &found) == cudaSuccess &&
                found == cudaDriverEntryPointSuccess, "cuTensorMapEncodeTiled not found");
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(f);
  }();
  return fn;
}

// [rows, 128] bf16 with 256-byte rows, read in boxes of 64 columns x 64 rows; rows past
// the last read as zeros.
CUtensorMap rows_map(int64_t ptr, int64_t rows) {
  CUtensorMap m;
  cuuint64_t dims[2] = {DIM, static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {DIM * 2};
  cuuint32_t box[2] = {64, 64};
  cuuint32_t elem_strides[2] = {1, 1};
  const CUresult err = encode_tiled()(
      &m, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, reinterpret_cast<void*>(ptr), dims, strides,
      box, elem_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(err == CUDA_SUCCESS, "streaming attention tensor map: error ", err);
  return m;
}

}  // namespace

void attn_stream_launch(int64_t q, int64_t k, int64_t v, int64_t kv, int64_t sink,
                        int64_t qpos, int64_t scale, int64_t pm, int64_t pl, int64_t pacc,
                        int64_t n_split) {
  static const bool sized = [] {
    TORCH_CHECK(cudaFuncSetAttribute(attn_stream, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     SMEM) == cudaSuccess, "streaming attention shared memory");
    return true;
  }();
  (void)sized;
  const CUtensorMap qm = rows_map(q, Q * HEADS), km = rows_map(k, kv), vm = rows_map(v, kv);
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(n_split), 1, 1);
  config.blockDim = dim3(THREADS, 1, 1);
  config.dynamicSmemBytes = SMEM;
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &config, attn_stream, qm, km, vm, reinterpret_cast<const __nv_bfloat16*>(sink),
      reinterpret_cast<const int64_t*>(qpos), reinterpret_cast<const float*>(scale),
      reinterpret_cast<float*>(pm), reinterpret_cast<float*>(pl), reinterpret_cast<float*>(pacc),
      static_cast<int>(n_split));
  TORCH_CHECK(err == cudaSuccess, "streaming attention launch: ", cudaGetErrorString(err));
}
"""

_STREAM_CPP = ("void attn_stream_launch(int64_t q, int64_t k, int64_t v, int64_t kv, "
               "int64_t sink, int64_t qpos, int64_t scale, int64_t pm, int64_t pl, "
               "int64_t pacc, int64_t n_split);")


def _tmem_access() -> str:
    """The tcgen05.ld / tcgen05.st forms the kernel uses, `16x32bx2`: thread t of a warp
    takes TMEM lane t % 16 of the warp's quarter, the second half-warp `n` columns on."""
    def regs(n, first):
        return ", ".join(f"%{first + i}" for i in range(n))

    def load(n):
        outs = ", ".join(f'"=r"(r[{i}])' for i in range(n))
        return (f"__device__ __forceinline__ void tmem_ld_x{n}(uint32_t taddr, "
                f"uint32_t (&r)[{n}]) {{\n"
                f'  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x{n}.b32 {{{regs(n, 0)}}}, '
                f'[%{n}], {n};\\n" : {outs} : "r"(taddr));\n}}\n')

    def store(n):
        ins = ", ".join(f'"r"(r[{i}])' for i in range(n))
        return (f"__device__ __forceinline__ void tmem_st_x{n}(uint32_t taddr, "
                f"const uint32_t (&r)[{n}]) {{\n"
                f'  asm volatile("tcgen05.st.sync.aligned.16x32bx2.x{n}.b32 [%0], {n}, '
                f'{{{regs(n, 1)}}};\\n" :: "r"(taddr), {ins} : "memory");\n}}\n')

    return load(32) + load(64) + store(64)


class StreamAttention:
    """The launch of the streaming read at `n_split` splits."""

    def __init__(self, n_split: int):
        self.n_split = n_split
        cuda = (STREAM_CUDA.replace("TMEM_ACCESS", _tmem_access())
.replace("RESCALE_TAU_F", f"{RESCALE_TAU!r}f"))
        self._ext = _extension("iquestrt_attention_stream_v2", _STREAM_CPP, cuda,
                               "attn_stream_launch", arch="10.0a")

    def __call__(self, q, k, v, sink_k, qpos, scale, pm, pl, pacc):
        kv, ns = k.shape[0], self.n_split
        bf16, f32 = torch.bfloat16, torch.float32
        _check_operands("streaming attention", (
            ("q", q, bf16, (Q, HEADS, DIM)), ("k", k, bf16, (kv, 1, DIM)),
            ("v", v, bf16, (kv, 1, DIM)), ("sink_k", sink_k, bf16, (1, DIM)),
            ("qpos", qpos, torch.int64, (Q,)), ("scale", scale, f32, (1,)),
            ("pm", pm, f32, (Q, HEADS, ns)), ("pl", pl, f32, (Q, HEADS, ns)),
            ("pacc", pacc, f32, (Q, HEADS, ns, DIM))))
        if kv % TILE:
            raise ValueError(f"the streaming read needs a multiple of {TILE} positions, "
                             f"got {kv}")
        self._ext.attn_stream_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), kv,
                                     sink_k.data_ptr(), qpos.data_ptr(), scale.data_ptr(),
                                     pm.data_ptr(), pl.data_ptr(), pacc.data_ptr(), ns)


@functools.lru_cache(maxsize=None)
def build_stream(n_split: int) -> StreamAttention:
    return StreamAttention(n_split)


# The combine. A lane holds its slice of every split in
# registers, and the compact32 map gives it two splits a group past 128 splits.
COMBINE_SPLIT_RANGE = (130, 160)

COMBINE_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {

constexpr int Q = 8, HEADS = 6, DIM = 128, NS = N_SPLIT, ITEMS = 2, GROUPS = 4, PACK = 2;
constexpr int CHAINS = 4;
static_assert(CHAINS == 4, "the walk's tree adds four chains");
constexpr float NEG_INF = -3.0e38f;
constexpr unsigned FULL = 0xffffffffu;

// Volatile asm: NVVM keeps every load whole and in order; ptxas still schedules them.
__device__ __forceinline__ float ld_cg(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];\n" : "=f"(v) : "l"(p));
  return v;
}

// Warp w of CTA (head, query, z) merges dims [32 (PACK z + w), +32) of one row.
__global__ void __launch_bounds__(32 * PACK, 1) attn_stream_combine(
    const float* __restrict__ pm, const float* __restrict__ pl,
    const float* __restrict__ pacc, __nv_bfloat16* __restrict__ out) {
  __shared__ __align__(16) float wsm[PACK][NS];
  cudaTriggerProgrammaticLaunchCompletion();
  cudaGridDependencySynchronize();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int row = blockIdx.y * HEADS + blockIdx.x;
  const int d = (blockIdx.z * PACK + warp) * 32 + lane;
  const float* pmr = pm + row * NS;
  const float* plr = pl + row * NS;
  const float* pa = pacc + (int64_t)row * NS * DIM + d;
  // Lane t owns splits (t + 32 g) * 2 + u, as the compact32 combine's lane does.
  float m[GROUPS][ITEMS], l[GROUPS][ITEMS], p[NS];
#pragma unroll
  for (int g = 0; g < GROUPS; ++g)
#pragma unroll
    for (int u = 0; u < ITEMS; ++u) {
      const int s = (lane + g * 32) * ITEMS + u;
      m[g][u] = s < NS ? ld_cg(pmr + s) : NEG_INF;
      l[g][u] = s < NS ? ld_cg(plr + s) : 0.f;
    }
#pragma unroll
  for (int s = 0; s < NS; ++s) p[s] = ld_cg(pa + s * DIM);

  // The row max: per group, groups (0, 2) and (1, 3), then the xor-16.. 1 butterfly.
  float vmax[GROUPS];
#pragma unroll
  for (int g = 0; g < GROUPS; ++g) {
    vmax[g] = -INFINITY;
#pragma unroll
    for (int u = 0; u < ITEMS; ++u) vmax[g] = fmaxf(vmax[g], m[g][u]);
  }
  float mx = fmaxf(fmaxf(vmax[0], vmax[2]), fmaxf(vmax[1], vmax[3]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, o));

  // The weights and the row sum, each pl * w rounded before it joins its group's sum.
  float w[GROUPS][ITEMS], vsum[GROUPS];
#pragma unroll
  for (int g = 0; g < GROUPS; ++g) {
    vsum[g] = 0.f;
#pragma unroll
    for (int u = 0; u < ITEMS; ++u) {
      const int s = (lane + g * 32) * ITEMS + u;
      w[g][u] = s < NS ? expf(m[g][u] - mx) : 0.f;
      vsum[g] = vsum[g] + (s < NS ? __fmul_rn(l[g][u], w[g][u]) : 0.f);
      if (s < NS) wsm[warp][s] = w[g][u];
    }
  }
  float ls = 0.f + ((vsum[0] + vsum[2]) + (vsum[1] + vsum[3]));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) ls = ls + __shfl_xor_sync(FULL, ls, o);
  __syncwarp();

  // CHAINS interleaved FMA chains, split s into chain s % CHAINS, then their FP32 tree.
  float c[CHAINS];
#pragma unroll
  for (int j = 0; j < CHAINS; ++j) c[j] = 0.f;
#pragma unroll
  for (int s = 0; s < NS; ++s) c[s % CHAINS] = fmaf(wsm[warp][s], p[s], c[s % CHAINS]);
  const float acc = (c[0] + c[1]) + (c[2] + c[3]);
  out[row * DIM + d] = __float2bfloat16_rn(acc / ls);
}

}  // namespace

void attn_stream_combine_launch(int64_t pm, int64_t pl, int64_t pacc, int64_t out) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(HEADS, Q, 4 / PACK);
  config.blockDim = dim3(32 * PACK, 1, 1);
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &config, attn_stream_combine, reinterpret_cast<const float*>(pm),
      reinterpret_cast<const float*>(pl), reinterpret_cast<const float*>(pacc),
      reinterpret_cast<__nv_bfloat16*>(out));
  TORCH_CHECK(err == cudaSuccess, "streaming combine launch: ", cudaGetErrorString(err));
}
"""

_COMBINE_CPP = ("void attn_stream_combine_launch(int64_t pm, int64_t pl, int64_t pacc, "
                "int64_t out);")


class StreamCombine:
    """The launch of the full read's combine at `n_split` splits."""

    def __init__(self, n_split: int):
        lo, hi = COMBINE_SPLIT_RANGE
        if not lo <= n_split <= hi or n_split % 2:
            raise ValueError(f"the streaming combine takes an even {lo}-{hi} splits, "
                             f"got {n_split}")
        self.n_split = n_split
        self._ext = _extension(f"iquestrt_attention_stream_combine_{n_split}_v2", _COMBINE_CPP,
                               COMBINE_CUDA.replace("N_SPLIT", str(n_split)),
                               "attn_stream_combine_launch")

    def __call__(self, pm, pl, pacc, out):
        ns, f32 = self.n_split, torch.float32
        _check_operands("streaming combine", (
            ("pm", pm, f32, (Q, HEADS, ns)), ("pl", pl, f32, (Q, HEADS, ns)),
            ("pacc", pacc, f32, (Q, HEADS, ns, DIM)),
            ("out", out, torch.bfloat16, (Q, HEADS, DIM))))
        self._ext.attn_stream_combine_launch(pm.data_ptr(), pl.data_ptr(), pacc.data_ptr(),
                                             out.data_ptr())


@functools.lru_cache(maxsize=None)
def build_stream_combine(n_split: int) -> StreamCombine:
    return StreamCombine(n_split)
