"""The MoE router of one Q=8 step in ONE launch: GEMV, top-8, softmax, expert sort."""

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from .arch import cuda_arch

HIDDEN = 3072
EXPERTS = 256
TOPK = 8
Q = 8
KSPLIT = 12
BM = 16

# One row's top 8 and their softmax from its 256 logits, on one warp.
# The finisher below selects through `select_row`. The main
# stack's attention tail, which folds the logits into its own all-reduce
# (`q8/nanpoll.py`), ranks through `rank_keys` and stores through `store_top`, and
# selects through `select_row` where a tie leaves the ranking unsettled.
SELECT = r"""
// ------------------------------------------------------------------ top-8 selection
namespace iquestrt_select {

constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr float kSelectedMask = -3.0e38f;

// Monotone for finite fp32.  Canonicalising signed zero keeps the floating comparison
// rule under which -0 and +0 tie and the lower expert id wins.
__device__ __forceinline__ unsigned ordered_key(float value) {
  unsigned bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) == 0) bits = 0;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// Row `row`'s routing, one warp: lane l holds the logits of experts 4l..4l+3 in v[0..3]
// and of 128+4l..128+4l+3 in v[4..7]. Each lane sorts
// its eight candidates once by DESC(value)/ASC(expert), so its list head is its own max
// with the lower id winning a tie, and each top-k round takes the head with the largest
// ordered key and, among those, the smallest id, then drops it from its owner's list.
// Lanes 0..7 store the row's pairs when `store` holds.
__device__ __forceinline__ void select_row(
    float (&v)[8], int row, int* __restrict__ sel_out, float* __restrict__ wt_out,
    bool store) {
  const int lane = threadIdx.x & 31;
  constexpr unsigned mask = 0xffffffffu;
  int e[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    e[j] = (j < 4) ? lane * 4 + j : 128 + lane * 4 + (j - 4);

#define IQUESTRT_CE(a, b)                                                        \
  {                                                                            \
    const bool sw = (v[b] > v[a]) || (v[b] == v[a] && e[b] < e[a]);            \
    const float tv = sw ? v[b] : v[a];                                         \
    const float uv = sw ? v[a] : v[b];                                         \
    const int te = sw ? e[b] : e[a];                                           \
    const int ue = sw ? e[a] : e[b];                                           \
    v[a] = tv; v[b] = uv; e[a] = te; e[b] = ue;                                \
  }
  IQUESTRT_CE(0, 1) IQUESTRT_CE(2, 3) IQUESTRT_CE(4, 5) IQUESTRT_CE(6, 7)
  IQUESTRT_CE(0, 2) IQUESTRT_CE(1, 3) IQUESTRT_CE(4, 6) IQUESTRT_CE(5, 7)
  IQUESTRT_CE(1, 2) IQUESTRT_CE(5, 6)
  IQUESTRT_CE(0, 4) IQUESTRT_CE(1, 5) IQUESTRT_CE(2, 6) IQUESTRT_CE(3, 7)
  IQUESTRT_CE(2, 4) IQUESTRT_CE(3, 5)
  IQUESTRT_CE(1, 2) IQUESTRT_CE(3, 4) IQUESTRT_CE(5, 6)
#undef IQUESTRT_CE

  // Lane t keeps round t's pair. The softmax and the ranks advance with the rounds: every
  // lane adds round t's exp to the sum in round order, which is the descending-value order
  // torch sums in; lane t counts the earlier winners below its id, and each earlier lane
  // counts round t's winner if it is below its own.
  int selected_id = kExperts, rank = 0;
  float top_value = 0.0f, sum = 0.0f, probability = 0.0f;
  int won[kTopK];
#pragma unroll
  for (int t = 0; t < kTopK; ++t) {
    const unsigned key = ordered_key(v[0]);
    const unsigned max_key = __reduce_max_sync(mask, key);
    const int candidate_id = key == max_key ? e[0] : kExperts;
    const int winner_id = __reduce_min_sync(mask, candidate_id);
    const int owner_lane = winner_id < 128 ? winner_id / 4 : (winner_id - 128) / 4;
    const float winner_value = __shfl_sync(mask, v[0], owner_lane);
    if (t == 0) top_value = winner_value;
    const float p = expf(__fsub_rn(winner_value, top_value));
    sum = __fadd_rn(sum, p);
    int below = 0;
#pragma unroll
    for (int s = 0; s < t; ++s) below += won[s] < winner_id;
    if (lane < t) rank += winner_id < selected_id;
    if (lane == t) {
      selected_id = winner_id;
      probability = p;
      rank = below;
    }
    won[t] = winner_id;
    if (lane == owner_lane) {
#pragma unroll
      for (int k = 0; k < 7; ++k) {
        v[k] = v[k + 1];
        e[k] = e[k + 1];
      }
      v[7] = kSelectedMask;
      e[7] = kExperts;
    }
  }
  if (lane < kTopK) probability = probability / sum;

  // Lane t < 8 stores its pair at its id's rank among the row's eight distinct ids.
  if (store && lane < kTopK) {
    sel_out[row * kTopK + rank] = selected_id;
    wt_out[row * kTopK + rank] = probability;
  }
}

// The router fold's form (`q8/nanpoll.py`): select_row's rounds on
// ordered keys with one reduction each, and the softmax apart, so the rounds can run
// before the values' positive scale is known. `rank_keys` sorts lane l's keys of experts
// 4l..4l+3 and 128+4l..128+4l+3 by select_row's network; round t's largest key is
// `top[t]`, and the lane holding it takes its head and writes its id to `ids_s[t]`, eight
// ints of the warp's shared memory. A round that finds its key at one lane has
// select_row's winner. One that finds it at two takes two heads, so `tied` holds and the
// row must select with select_row instead. Lane t < 8 gets round t's `id` and its `rank`
// among the eight; every lane gets `ninth`, the largest key left.
__device__ __forceinline__ void rank_keys(unsigned (&k)[8], int* ids_s, unsigned (&top)[kTopK],
                                          unsigned& ninth, int& id, int& rank, bool& tied) {
  const int lane = threadIdx.x & 31;
  constexpr unsigned mask = 0xffffffffu;
  int e[8];
#pragma unroll
  for (int j = 0; j < 8; ++j)
    e[j] = (j < 4) ? lane * 4 + j : 128 + lane * 4 + (j - 4);

#define IQUESTRT_CE(a, b)                                                        \
  {                                                                            \
    const bool sw = (k[b] > k[a]) || (k[b] == k[a] && e[b] < e[a]);            \
    const unsigned tk = sw ? k[b] : k[a];                                      \
    const unsigned uk = sw ? k[a] : k[b];                                      \
    const int te = sw ? e[b] : e[a];                                           \
    const int ue = sw ? e[a] : e[b];                                           \
    k[a] = tk; k[b] = uk; e[a] = te; e[b] = ue;                                \
  }
  IQUESTRT_CE(0, 1) IQUESTRT_CE(2, 3) IQUESTRT_CE(4, 5) IQUESTRT_CE(6, 7)
  IQUESTRT_CE(0, 2) IQUESTRT_CE(1, 3) IQUESTRT_CE(4, 6) IQUESTRT_CE(5, 7)
  IQUESTRT_CE(1, 2) IQUESTRT_CE(5, 6)
  IQUESTRT_CE(0, 4) IQUESTRT_CE(1, 5) IQUESTRT_CE(2, 6) IQUESTRT_CE(3, 7)
  IQUESTRT_CE(2, 4) IQUESTRT_CE(3, 5)
  IQUESTRT_CE(1, 2) IQUESTRT_CE(3, 4) IQUESTRT_CE(5, 6)
#undef IQUESTRT_CE

  // A taken list ends in key 0, below every value's.
  int took = 0;
#pragma unroll
  for (int t = 0; t < kTopK; ++t) {
    const unsigned max_key = __reduce_max_sync(mask, k[0]);
    top[t] = max_key;
    const bool take = k[0] == max_key;
    if (take) ids_s[t] = e[0];
    took += take;
#pragma unroll
    for (int j = 0; j < 7; ++j) {
      k[j] = take ? k[j + 1] : k[j];
      e[j] = take ? e[j + 1] : e[j];
    }
    k[7] = take ? 0u : k[7];
    e[7] = take ? kExperts : e[7];
  }
  ninth = __reduce_max_sync(mask, k[0]);
  tied = __reduce_add_sync(mask, took) != kTopK;
  __syncwarp();
  const int4 lo = *reinterpret_cast<const int4*>(ids_s);
  const int4 hi = *reinterpret_cast<const int4*>(ids_s + 4);
  const int ids[kTopK] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
  id = kExperts;
#pragma unroll
  for (int s = 0; s < kTopK; ++s)
    if (lane == s) id = ids[s];
  rank = 0;
#pragma unroll
  for (int s = 0; s < kTopK; ++s) rank += ids[s] < id;
}

// ordered_key's inverse, which reads -0 as +0.
__device__ __forceinline__ float key_value(unsigned key) {
  return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

// The softmax of the row's eight values in round order, stored at the ranks of their ids
// `id` (lane t < 8: round t's) as select_row stores it. Lane t < 8 takes round t's exp
// alone, and every lane sums the eight from shuffles in round order: select_row's
// operations on the same values, one exp a lane where every lane computed all eight
// (`q8/nanpoll.py`).
__device__ __forceinline__ void store_top(const float (&value)[kTopK], int id, int rank,
                                          int row, int* __restrict__ sel_out,
                                          float* __restrict__ wt_out) {
  const int lane = threadIdx.x & 31;
  float own = value[0];
#pragma unroll
  for (int t = 1; t < kTopK; ++t)
    if (lane == t) own = value[t];
  const float probability = expf(__fsub_rn(own, value[0]));
  float sum = 0.0f;
#pragma unroll
  for (int t = 0; t < kTopK; ++t) sum = __fadd_rn(sum, __shfl_sync(0xffffffffu, probability, t));
  if (lane < kTopK) {
    sel_out[row * kTopK + rank] = id;
    wt_out[row * kTopK + rank] = probability / sum;
  }
}

}  // namespace iquestrt_select
"""

_CPP = r"""
#include <torch/extension.h>

void router_q8_launch(
    int64_t x_pad, int64_t w, int64_t parts, int64_t sel, int64_t wt, int64_t device_index);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

constexpr int kHidden = 3072;
constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kSplit = 12;
constexpr int kKTile = kHidden / kSplit;
constexpr int kFirstPacks = 128 / 8;
constexpr int kSecondPacks = kKTile / 8 - kFirstPacks;
constexpr int kBM = 16;
constexpr int kNTile = 8;
constexpr int kBlocks = kNTile * kSplit;
constexpr int kThreads = 128;
constexpr int kSmemBytes = 48 * 1024;
static_assert(kHidden % kSplit == 0, "router split-K must divide hidden");
static_assert(kKTile >= 128 && kKTile <= 256 && kKTile % 8 == 0,
              "router split-K tile must be 128..256 and k8 aligned");
static_assert(kNTile == 8, "CTA (r, 0) finishes row r of the eight");

__device__ __forceinline__ unsigned smem_address(const void* ptr) {
  return static_cast<unsigned>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* global) {
  const unsigned address = smem_address(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
               : : "r"(address), "l"(global));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" : :);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  if constexpr (N == 0) asm volatile("cp.async.wait_all;\n" : :);
  else asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

__device__ __forceinline__ void ldmatrix_x1(const void* smem, unsigned* out) {
  const unsigned address = smem_address(smem);
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
               : "=r"(out[0]) : "r"(address));
}

__device__ __forceinline__ void ldmatrix_x2(const void* smem, unsigned* out) {
  const unsigned address = smem_address(smem);
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(out[0]), "=r"(out[1]) : "r"(address));
}

__device__ __forceinline__ void iquestrt_bf16_mma_k8(
    const unsigned int* a, const unsigned int* b, float* c) {
  asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(b[0]));
}

__device__ __forceinline__ uint64_t bf16x8_to_e4m3x8(const uint4& packed) {
  const auto* values = reinterpret_cast<const __nv_bfloat16*>(&packed);
  uint64_t result = 0;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    const __nv_fp8_storage_t encoded = __nv_cvt_float_to_fp8(
        __bfloat162float(values[index]), __NV_SATFINITE, __NV_E4M3);
    result |= static_cast<uint64_t>(encoded) << (8 * index);
  }
  return result;
}

__device__ __forceinline__ void store_optional_e4m3x8(
    uint8_t* output, int offset, const uint4& packed) {
  if (output != nullptr)
    *reinterpret_cast<uint64_t*>(output + offset) = bf16x8_to_e4m3x8(packed);
}

// Exact XOR maps emitted by the production TileLang BF16-k8 builder.  The cp.async
// stores and ldmatrix consumers must use the same transform.
__device__ __forceinline__ int x_store_index(int tid, int i) {
  return (((tid & 15) >> 3) * 1024 + i * 512 + (tid >> 4) * 64
          + (((tid >> 6) + ((tid & 7) >> 2)) & 1) * 32
          + ((((tid & 63) >> 5) + ((tid & 3) >> 1)) & 1) * 16
          + ((((tid & 31) >> 4) + (tid & 1)) & 1) * 8);
}

__device__ __forceinline__ int w_store_index(int tid, int i) {
  return (((tid & 15) >> 3) * 2048 + i * 512 + (tid >> 4) * 64
          + (((tid >> 6) + ((tid & 7) >> 2)) & 1) * 32
          + ((((tid & 63) >> 5) + ((tid & 3) >> 1)) & 1) * 16
          + ((((tid & 31) >> 4) + (tid & 1)) & 1) * 8);
}

__device__ __forceinline__ int x_ldmatrix_index(int tid, int kk) {
  return ((kk >> 3) * 1024 + ((tid & 15) >> 3) * 512
          + (((tid & 15) * 64
              + ((((tid & 7) >> 2) + ((kk & 7) >> 2)) & 1) * 32
              + ((((tid & 3) >> 1) + ((kk & 3) >> 1)) & 1) * 16
              + (((tid & 1) + (kk & 1)) & 1) * 8) & 511));
}

__device__ __forceinline__ int w_ldmatrix_index(int tid, int kk) {
  return ((kk >> 3) * 2048 + (tid >> 5) * 512 + (tid & 7) * 64
          + ((((kk & 7) >> 2) + ((tid & 7) >> 2)) & 1) * 32
          + ((((tid & 3) >> 1) + ((kk & 3) >> 1)) & 1) * 16
          + (((tid & 1) + (kk & 1)) & 1) * 8);
}

""" + SELECT + r"""
constexpr long long kReadLimit = 1ll << 22;  // polls of a row's partials before a trap

// One relaxed gpu-scope read of 16 bytes; the asm keeps it inside the poll loop.
__device__ __forceinline__ float4 ld_relaxed_f4(const float* p) {
  float4 v;
  asm volatile("ld.relaxed.gpu.global.v4.f32 {%0, %1, %2, %3}, [%4];\n"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(p) : "memory");
  return v;
}

__device__ __forceinline__ bool armed4(const float4& v) {
  return isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w);
}

// Row `row`'s finish, one warp. Lane l owns experts 4l..4l+3 and 128+4l..128+4l+3: it
// polls their 12 split partials until none is NaN, hands the row to warp 5 to re-arm
// (`rearm_row`), adds the splits in order and selects (`iquestrt_select::select_row`). A
// `warm` pass reads once and stores nothing: it only brings the code in.
__device__ __forceinline__ void finish_row_polled(
    const float* __restrict__ parts, int row,
    int* __restrict__ sel_out, float* __restrict__ wt_out, bool warm) {
  const int lane = threadIdx.x & 31;
  constexpr unsigned mask = 0xffffffffu;
  float v[8];
  float4 lo[kSplit], hi[kSplit];
  // Bit 2 sp: split sp's experts 4l.. still read NaN; bit 2 sp + 1: its experts 128+4l..
  unsigned pending = (1u << (2 * kSplit)) - 1u;
  for (long long n = 0;; ++n) {
    const unsigned todo = pending;
#pragma unroll
    for (int sp = 0; sp < kSplit; ++sp) {
      const float* base = parts + static_cast<size_t>(sp) * kBM * kExperts + row * kExperts;
      if (todo & (1u << (2 * sp))) lo[sp] = ld_relaxed_f4(base + lane * 4);
      if (todo & (2u << (2 * sp))) hi[sp] = ld_relaxed_f4(base + 128 + lane * 4);
    }
#pragma unroll
    for (int sp = 0; sp < kSplit; ++sp) {
      if ((todo & (1u << (2 * sp))) && !armed4(lo[sp])) pending &= ~(1u << (2 * sp));
      if ((todo & (2u << (2 * sp))) && !armed4(hi[sp])) pending &= ~(2u << (2 * sp));
    }
    if (warm || __all_sync(mask, pending == 0u)) break;
    if (n == kReadLimit) __trap();
  }
  // The row is read: warp 5, waiting at named barrier 2, re-arms it.
  if (!warm) asm volatile("bar.arrive 2, 64;" ::: "memory");
  v[0] = lo[0].x; v[1] = lo[0].y; v[2] = lo[0].z; v[3] = lo[0].w;
  v[4] = hi[0].x; v[5] = hi[0].y; v[6] = hi[0].z; v[7] = hi[0].w;
#pragma unroll
  for (int sp = 1; sp < kSplit; ++sp) {
    v[0] = __fadd_rn(v[0], lo[sp].x); v[1] = __fadd_rn(v[1], lo[sp].y);
    v[2] = __fadd_rn(v[2], lo[sp].z); v[3] = __fadd_rn(v[3], lo[sp].w);
    v[4] = __fadd_rn(v[4], hi[sp].x); v[5] = __fadd_rn(v[5], hi[sp].y);
    v[6] = __fadd_rn(v[6], hi[sp].z); v[7] = __fadd_rn(v[7], hi[sp].w);
  }
  iquestrt_select::select_row(v, row, sel_out, wt_out, !warm);
}

// Row `row`'s partials back to qNaN: warp 5 of CTA (row, 0), once warp 4 has read them.
__device__ __forceinline__ void rearm_row(float* __restrict__ parts, int row) {
  const int lane = threadIdx.x & 31;
  const float nan = __int_as_float(0x7fc00000);
  const float4 armed = make_float4(nan, nan, nan, nan);
#pragma unroll
  for (int sp = 0; sp < kSplit; ++sp) {
    float* base = parts + static_cast<size_t>(sp) * kBM * kExperts + row * kExperts;
    *reinterpret_cast<float4*>(base + lane * 4) = armed;
    *reinterpret_cast<float4*>(base + 128 + lane * 4) = armed;
  }
}

// The lower 128 threads of each CTA are the BF16 HMMA producer and sync on named barrier
// 1. The upper four warps meet only the PDL fence, except in CTA (r, 0), whose warp 4
// finishes row r while the producer runs and whose warp 5 re-arms the row once warp 4 has
// read it (named barrier 2).
__device__ __forceinline__ void q8_compute_sync() {
  asm volatile("bar.sync 1, 128;" : : : "memory");
}

template <bool WaitPDL = false, bool TriggerPDL = false,
          bool CopyXPad = true, bool WeightPrefetch = false>
__global__ __launch_bounds__(256, 1) void router_q8_fused_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    float* __restrict__ parts,
    int* __restrict__ sel,
    float* __restrict__ wt,
    __nv_bfloat16* __restrict__ x_pad,
    uint8_t* __restrict__ x_pad_fp8) {
  extern __shared__ __align__(1024) unsigned char dynamic_smem[];
  auto* x_s = reinterpret_cast<__nv_bfloat16*>(dynamic_smem);
  auto* w_s = reinterpret_cast<__nv_bfloat16*>(dynamic_smem + 16384);
  const int tid = threadIdx.x;
  const int bx = blockIdx.x;
  const int bs = blockIdx.y;

  static_assert(!WeightPrefetch || WaitPDL,
                "early Router weight prefetch requires a PDL fence");
  if constexpr (WeightPrefetch) {
    if (tid < kThreads) {
      // Router weights are immutable and independent of the AR producer.  Split them
      // into two committed groups before the PDL fence so their global-memory latency
      // overlaps the producer's cross-rank tail.  Input reads stay below wait.
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int expert = bx * 32 + i * 8 + (tid >> 4);
        const int col = bs * kKTile + (tid & 15) * 8;
        cp_async_16(w_s + w_store_index(tid, i),
                    w + expert * kHidden + col);
      }
      cp_async_commit();
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int expert = bx * 32 + i * 8 + (tid >> 4);
        const int col = bs * kKTile + 128 + (tid & 15) * 8;
        if ((tid & 15) < kSecondPacks)
          cp_async_16(w_s + 4096 + w_store_index(tid, i),
                      w + expert * kHidden + col);
      }
      cp_async_commit();
    }
  }

  // Row bx's finisher waits below, after its warm pass.
  const bool finisher = bs == 0 && (tid >> 5) == 4;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (WaitPDL)
    if (!finisher) asm volatile("griddepcontrol.wait;" ::: "memory");
  // The Gate reads `sel` after its own wait, which waits for this whole grid: releasing
  // it here moves only its launch.
  if constexpr (TriggerPDL)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif

  if (tid < kThreads) {
    const int linear_block = bs * kNTile + bx;
    const int linear_thread = linear_block * kThreads + tid;
    const auto* x4 = reinterpret_cast<const uint4*>(x);
    auto* p4 = reinterpret_cast<uint4*>(x_pad);
    for (int i = linear_thread; i < 8 * kHidden / 8;
         i += kBlocks * kThreads) {
      const uint4 packed = x4[i];
      if constexpr (CopyXPad) {
        p4[i] = packed;
        store_optional_e4m3x8(x_pad_fp8, i * 8, packed);
      }
    }

    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    unsigned a_pair[2];
    unsigned b_pair[1];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int row = min(i * 8 + (tid >> 4), 7);
      const int col = bs * kKTile + (tid & 15) * 8;
      cp_async_16(x_s + x_store_index(tid, i),
                  x + row * kHidden + col);
    }
    if constexpr (!WeightPrefetch) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int expert = bx * 32 + i * 8 + (tid >> 4);
        const int col = bs * kKTile + (tid & 15) * 8;
        cp_async_16(w_s + w_store_index(tid, i),
                    w + expert * kHidden + col);
      }
    }
    cp_async_commit();
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int row = min(i * 8 + (tid >> 4), 7);
      const int col = bs * kKTile + 128 + (tid & 15) * 8;
      if ((tid & 15) < kSecondPacks)
        cp_async_16(x_s + 2048 + x_store_index(tid, i),
                    x + row * kHidden + col);
    }
    if constexpr (!WeightPrefetch) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int expert = bx * 32 + i * 8 + (tid >> 4);
        const int col = bs * kKTile + 128 + (tid & 15) * 8;
        if ((tid & 15) < kSecondPacks)
          cp_async_16(w_s + 4096 + w_store_index(tid, i),
                      w + expert * kHidden + col);
      }
    }
    cp_async_commit();
    cp_async_wait<1>();
    q8_compute_sync();
#pragma unroll
    for (int kk = 0; kk < kFirstPacks; ++kk) {
      ldmatrix_x2(x_s + x_ldmatrix_index(tid, kk), a_pair);
      ldmatrix_x1(w_s + w_ldmatrix_index(tid, kk), b_pair);
      iquestrt_bf16_mma_k8(a_pair, b_pair, acc);
    }
    cp_async_wait<0>();
    q8_compute_sync();
#pragma unroll
    for (int kk = 0; kk < kSecondPacks; ++kk) {
      ldmatrix_x2(x_s + 2048 + x_ldmatrix_index(tid, kk), a_pair);
      ldmatrix_x1(w_s + 4096 + w_ldmatrix_index(tid, kk), b_pair);
      iquestrt_bf16_mma_k8(a_pair, b_pair, acc);
    }

    const int lane = tid & 31;
    const int warp_n = tid >> 5;
    // acc[0..1] hold row lane / 4's two adjacent experts (acc[2..3] the padding rows):
    // one 8-byte store, so the four lanes of a row fill its 32-byte sector at once.
    *reinterpret_cast<float2*>(parts + static_cast<size_t>(bs) * kBM * kExperts
                               + (lane / 4) * kExperts + bx * 32 + warp_n * 8
                               + 2 * (lane & 3)) = make_float2(acc[0], acc[1]);
  }
  // One copy of the finish, run twice: warm before the fence, then for real.
  if (finisher) {
#pragma unroll 1
    for (int pass = 0; pass < 2; ++pass) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
      if constexpr (WaitPDL)
        if (pass == 1) asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
      finish_row_polled(parts, bx, sel, wt, pass == 0);
    }
  } else if (bs == 0 && (tid >> 5) == 5) {
    asm volatile("bar.sync 2, 64;" ::: "memory");
    rearm_row(parts, bx);
  }
}

void router_q8_launch(
    int64_t x_pad, int64_t w, int64_t parts, int64_t sel, int64_t wt, int64_t device_index) {
  TORCH_CHECK(device_index >= 0, "router needs a concrete CUDA device");
  TORCH_CHECK((x_pad | w | parts) % 16 == 0,
              "router vector path needs 16-byte aligned operands");
  const c10::cuda::CUDAGuard device_guard(
      static_cast<c10::DeviceIndex>(device_index));
  // The AR tail already wrote the normalised rows into x_pad[:8]: the input and the
  // widening destination are one buffer, and this instantiation compiles the copy out.
  auto kernel = router_q8_fused_kernel<true, true, false, true>;
  cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes);
  TORCH_CHECK(err == cudaSuccess, "router dynamic-smem attribute: ",
              cudaGetErrorString(err));
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kNTile, kSplit, 1);
  config.blockDim = 256;
  config.dynamicSmemBytes = kSmemBytes;
  config.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  auto* xp = reinterpret_cast<__nv_bfloat16*>(x_pad);
  err = cudaLaunchKernelEx(
      &config, kernel, xp, reinterpret_cast<const __nv_bfloat16*>(w),
      reinterpret_cast<float*>(parts), reinterpret_cast<int*>(sel),
      reinterpret_cast<float*>(wt), xp, static_cast<uint8_t*>(nullptr));
  TORCH_CHECK(err == cudaSuccess, "router launch: ", cudaGetErrorString(err));
}
"""


_EXT = None


def _build():
    if torch.cuda.get_device_capability(torch.cuda.current_device()) != (10, 0):
        raise RuntimeError("the router extension is built for compute capability 10.0")
    with cuda_arch():
        return load_inline(
            name="iquestrt_router_q8_sm100",
            cpp_sources=_CPP,
            cuda_sources=_CUDA,
            functions=["router_q8_launch"],
            extra_cuda_cflags=["-O3", "-lineinfo"],
            verbose=False,
        )


def _ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        device_id = torch.cuda.current_device()
        rank = dist.get_rank()
        rank0_error = None
        status_device = torch.device("cuda", device_id)
        if rank == 0:
            try:
                _EXT = _build()
            except Exception as exc:  # noqa: BLE001 - propagate after collective status
                rank0_error = exc
        rank0_ok = torch.tensor(
            int(rank0_error is None), dtype=torch.int32, device=status_device)
        dist.broadcast(rank0_ok, src=0)
        if not int(rank0_ok.item()):
            if rank0_error is not None:
                raise RuntimeError("rank 0 failed to build fused router extension") \
                    from rank0_error
            raise RuntimeError("rank 0 failed to build fused router extension")

        local_error = None
        if rank != 0:
            try:
                _EXT = _build()
            except Exception as exc:  # noqa: BLE001 - make every rank leave together
                local_error = exc
        local_ok = torch.tensor(
            int(local_error is None), dtype=torch.int32, device=status_device)
        dist.all_reduce(local_ok, op=dist.ReduceOp.MIN)
        if not int(local_ok.item()):
            if local_error is not None:
                raise RuntimeError("a TP rank failed to load fused router extension") \
                    from local_error
            raise RuntimeError("a TP rank failed to load fused router extension")
    else:
        _EXT = _build()
    return _EXT


class RouterSelect:
    """x_pad[:Q] -> (sel [Q*k] int32, ascending within each row; wt [Q*k] fp32).

    Calls share this object's `parts` workspace, which each call leaves armed for the
    next, so they must be serialised on one stream, which one
    captured step is."""

    def __init__(self, n_experts: int, top_k: int, hidden: int, *, device, ws,
                 q_len: int):
        if (hidden, n_experts, top_k, q_len) != (HIDDEN, EXPERTS, TOPK, Q):
            raise ValueError(
                f"the router is compiled for H={HIDDEN}, E={EXPERTS}, top_k={TOPK}, "
                f"Q={Q}; got H={hidden}, E={n_experts}, top_k={top_k}, Q={q_len}")
        target = torch.device(device)
        index = torch.cuda.current_device() if target.index is None else target.index
        self.device = torch.device("cuda", index)
        # [Q*topk], row-major: `MoeDecodeKernels` reads it as one flat list of
        # (row, expert) pairs and derives the row as `p // topk`.
        self.sel = ws.slot(("router", "sel"), (q_len * top_k,), torch.int32)
        self.wt = ws.slot(("router", "wt"), (q_len * top_k,), torch.float32)
        self.parts = ws.slot(("router", "parts"), (KSPLIT, BM, n_experts),
                             torch.float32)
        # Armed once; every launch re-arms the partials it reads.
        self.parts.fill_(float("nan"))
        with torch.cuda.device(index):
            self._extension = _ext()

    def __call__(self, x_pad: torch.Tensor, weight: torch.Tensor):
        for tensor, dtype, shape, name in (
            (x_pad, torch.bfloat16, (BM, HIDDEN), "x_pad"),
            (weight, torch.bfloat16, (EXPERTS, HIDDEN), "weight"),
        ):
            if (tensor.device != self.device or tensor.dtype != dtype
                    or tuple(tensor.shape) != shape or not tensor.is_contiguous()):
                raise ValueError(
                    f"router {name} expected contiguous {dtype} {shape} on "
                    f"{self.device}, got {tensor.dtype} {tuple(tensor.shape)} on "
                    f"{tensor.device}")
        self._extension.router_q8_launch(
            x_pad.data_ptr(), weight.data_ptr(), self.parts.data_ptr(),
            self.sel.data_ptr(), self.wt.data_ptr(), self.device.index)
        return self.sel, self.wt
