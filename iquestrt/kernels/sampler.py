"""Vocabulary-parallel top-k/top-p sampling and speculative rejection.

All ranks share candidates and Philox counters. Accept d when u*q(d)<p(d), draw
rejections from max(0,p-q), and draw a bonus when all drafts pass. This preserves
the optimized target's distribution. Never reuse a random draw across positions."""

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from .arch import cuda_arch

# iquest_q1's generation_config.json; `Engine` refuses a checkpoint with others.
TEMPERATURE, TOP_K, TOP_P = 1.0, 20, 0.95
ROWS = 8            # the most rows one call draws: the verify's live rows
THREADS = 1024
PER_THREAD = 20     # logits per thread: a rank's slice holds 1024..20480
WORLD = 8           # the most ranks the exchange stages

SAMPLE, DRAFT, VERIFY = 0, 1, 2

# Declarations only -- `load_inline(functions=...)` generates the pybind module, and a
# hand-written PYBIND11_MODULE collides with it (symptom: build succeeds, attributes
# missing).
_CPP = r"""
#include <torch/extension.h>
#include <vector>
std::vector<int64_t> sp_alloc(int64_t nbytes);
int64_t sp_open(std::vector<int64_t> handle_words);
int64_t sp_register(std::vector<int64_t> bufs);
void sample_launch(int64_t logits, int64_t ld, int64_t rows, int64_t v_local,
                   int64_t mode, int64_t depth, int64_t d_bufs, int64_t gens, int64_t world, int64_t rank, int64_t seed,
                   int64_t pos, int64_t q_ids, int64_t q_probs, int64_t q_logits,
                   int64_t p_ids, int64_t p_probs, int64_t cand, int64_t dtok,
                   int64_t hidx, int64_t q, int64_t out, int64_t row_acc,
                   int64_t row_tok, int64_t done, int64_t n_draft, double top_p);
"""

_CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <cstring>

#define CK(x) do { cudaError_t e = (x); TORCH_CHECK(e == cudaSuccess, \
    "CUDA: ", cudaGetErrorString(e), " at ", __FILE__, ":", __LINE__); } while (0)

std::vector<int64_t> sp_alloc(int64_t nbytes) {
  void* p = nullptr;
  CK(cudaMalloc(&p, nbytes));
  CK(cudaMemset(p, 0, nbytes));
  cudaIpcMemHandle_t h;
  CK(cudaIpcGetMemHandle(&h, p));
  std::vector<int64_t> out;
  out.push_back(reinterpret_cast<int64_t>(p));
  const int64_t words = (sizeof(h) + 7) / 8;
  std::vector<int64_t> tmp(words, 0);
  std::memcpy(tmp.data(), &h, sizeof(h));
  for (auto w : tmp) out.push_back(w);
  return out;
}

int64_t sp_open(std::vector<int64_t> handle_words) {
  cudaIpcMemHandle_t h;
  std::vector<int64_t> tmp(handle_words.begin(), handle_words.end());
  std::memcpy(&h, tmp.data(), sizeof(h));
  void* p = nullptr;
  CK(cudaIpcOpenMemHandle(&p, h, cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(p);
}

int64_t sp_register(std::vector<int64_t> bufs) {
  void** d_bufs = nullptr;
  const int world = (int)bufs.size();
  TORCH_CHECK(world <= 8, "the exchange stages 8 ranks at most");
  CK(cudaMalloc(&d_bufs, 8 * sizeof(void*)));
  void* hb[8];
  for (int i = 0; i < world; ++i) hb[i] = reinterpret_cast<void*>(bufs[i]);
  CK(cudaMemcpy(d_bufs, hb, world * sizeof(void*), cudaMemcpyHostToDevice));
  return reinterpret_cast<int64_t>(d_bufs);
}

#define SP_TOPK 20
#define SP_ROWS 8
#define SP_PER 20
#define SP_WORLD 8
#define SP_WORDS (2 * SP_TOPK)   // LL words one rank stages per row
#define SP_LIST (SP_TOPK * 32)
#define SP_CAND (SP_TOPK * SP_PER)
#define SP_ID_MASK 0x7FFFFFFFLL
#define SP_FULL 0xFFFFFFFFu
#define SP_SAMPLE 0
#define SP_DRAFT 1
#define SP_VERIFY 2

// The order-preserving u32 map of the fp32 bits.
__device__ __forceinline__ unsigned sp_key(float v) {
  const unsigned b = __float_as_uint(v);
  return b ^ ((unsigned)((int)b >> 31) | 0x80000000u);   // negative: ~b; else b | 2^31
}

// key << 31 | (ID_MASK - gid): larger values first, equal ones by the lower id.
// Every packed key is >= 0, so -1 marks "none".
__device__ __forceinline__ long long sp_pack(unsigned key, long long gid) {
  return ((long long)key << 31) | (SP_ID_MASK - gid);
}

__device__ __forceinline__ float sp_value(long long p) {
  const unsigned key = (unsigned)((unsigned long long)p >> 31);
  return __uint_as_float((key & 0x80000000u) ? (key & 0x7FFFFFFFu) : ~key);
}

__device__ __forceinline__ long long sp_gid(long long p) {
  return SP_ID_MASK - (p & SP_ID_MASK);
}

__device__ __forceinline__ long long sp_wmax(long long v) {
  for (int o = 16; o; o >>= 1) {
    const long long w = __shfl_xor_sync(SP_FULL, v, o);
    if (w > v) v = w;
  }
  return v;
}

__device__ __forceinline__ float sp_wsum(float v) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(SP_FULL, v, o);
  return v;
}

// Inclusive prefix sum over lanes 0..lane, and suffix sum over lane..31.
__device__ __forceinline__ float sp_scan(float v, int lane) {
  for (int o = 1; o < 32; o <<= 1) {
    const float t = __shfl_up_sync(SP_FULL, v, o);
    if (lane >= o) v += t;
  }
  return v;
}

__device__ __forceinline__ float sp_rscan(float v, int lane) {
  for (int o = 1; o < 32; o <<= 1) {
    const float t = __shfl_down_sync(SP_FULL, v, o);
    if (lane + o < 32) v += t;
  }
  return v;
}

// Inverse CDF in lane order: the first lane whose cumulative weight exceeds
// u * total; the last positive one if rounding leaves none. Some weight is > 0.
__device__ __forceinline__ int sp_pick(float w, float u, int lane) {
  const float cdf = sp_scan(w, lane);
  const float total = __shfl_sync(SP_FULL, cdf, 31);
  const unsigned hit = __ballot_sync(SP_FULL, w > 0.f && u * total < cdf);
  const unsigned any = __ballot_sync(SP_FULL, w > 0.f);
  return hit ? __ffs(hit) - 1 : 31 - __clz(any);
}

// Philox4x32-10 (Salmon et al., SC'11) of counter (pos, role, 0) under key `seed`.
__device__ __forceinline__ uint4 sp_philox(unsigned long long seed,
                                           unsigned long long pos, unsigned role) {
  uint4 c = make_uint4((unsigned)pos, (unsigned)(pos >> 32), role, 0u);
  unsigned k0 = (unsigned)seed, k1 = (unsigned)(seed >> 32);
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    const unsigned hi0 = __umulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    const unsigned hi1 = __umulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = make_uint4(hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0);
    k0 += 0x9E3779B9u;
    k1 += 0xBB67AE85u;
  }
  return c;
}

__device__ __forceinline__ float sp_unit(unsigned x) {
  return (float)(x >> 8) * 5.9604644775390625e-08f;      // top 24 bits * 2^-24
}

// An asynchronous copy of 8 or 4 bytes from global into shared memory: no register
// holds the value until the thread waits for its copies (sp_cp_wait).
__device__ __forceinline__ void sp_cp8(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;"
               :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void sp_cp4(void* dst, const void* src) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
               :: "r"((unsigned)__cvta_generic_to_shared(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void sp_cp_wait() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// Warp-wide bitonic networks over one packed key per lane. Keys are distinct apart
// from -1 ("none"; every packed key is >= 0), so each network has one result. A
// compare-exchange with lane ^ j keeps the larger key iff keep_max.
__device__ __forceinline__ long long sp_cx(long long v, int j, bool keep_max) {
  const long long w = __shfl_xor_sync(SP_FULL, v, j);
  return keep_max == (w > v) ? w : v;
}

// Sorts the warp's 32 keys, descending in lane order if desc, else ascending.
__device__ __forceinline__ long long sp_sort32(long long v, int lane, bool desc) {
#pragma unroll
  for (int k = 2; k <= 32; k <<= 1)
#pragma unroll
    for (int j = k >> 1; j; j >>= 1)
      v = sp_cx(v, j, ((lane & j) == 0) == (((lane & k) == 0) == desc));
  return v;
}

// Sorts a bitonic sequence of 32 keys, one per lane, descending if desc. Lane by lane,
// the larger of a descending and an ascending sequence is one: the 32 largest of the
// 64 (Batcher's half-cleaner), which is how two sorted sequences merge below.
__device__ __forceinline__ long long sp_merge32(long long v, int lane, bool desc) {
#pragma unroll
  for (int j = 16; j; j >>= 1) v = sp_cx(v, j, ((lane & j) == 0) == desc);
  return v;
}

__device__ __forceinline__ long long sp_max(long long a, long long b) { return a > b ? a : b; }

// The 20 largest of n keys in shared memory, descending in lanes 0..19 of the calling
// warp: 32-key chunks, each sorted ascending and merged into the 32 largest so far.
__device__ __forceinline__ long long sp_top20(const long long* keys, int n, int lane) {
  long long top = sp_sort32(lane < n ? keys[lane] : -1, lane, true);
  for (int c = 32; c < n; c += 32)
    top = sp_merge32(sp_max(top, sp_sort32(c + lane < n ? keys[c + lane] : -1, lane, false)),
                     lane, true);
  return top;
}

__global__ void __launch_bounds__(1024) sample_kernel(
    const float* __restrict__ logits, long long ld, int rows, int v_local, int mode,
    int depth, unsigned long long** bufs, int* gens, int world, int rank,
    const long long* __restrict__ seed_p, long long* pos_p,
    long long* q_ids, float* q_probs, float* q_logits, long long* p_ids, float* p_probs,
    const long long* __restrict__ cand, long long* dtok, long long* hidx, int q,
    long long* out, int* row_acc, long long* row_tok, int* done, int n_draft,
    float top_p) {
  const int r = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int nthr = blockDim.x;
  __shared__ long long s_wmax[32];
  __shared__ long long s_list[SP_LIST];
  __shared__ long long s_cand[SP_CAND];
  __shared__ long long s_bound;
  __shared__ int s_nlist, s_nextra;
  __shared__ long long s_seed, s_pos, s_arow[SP_ROWS], s_qid[SP_TOPK];
  __shared__ float s_qp[SP_TOPK];

  // After step 1, warp 0 stores this rank's 20 to every rank (step 2) while warp 1
  // polls for every rank's and runs steps 3-5. Their reads of global memory issue
  // first, so the latency hides under step 1: warp 0's lane p < world takes rank p's
  // staging, warp 1's lane 0 this rank's, and both lanes 0 this row's call count
  // (warp 0 advances it after step 1). Read where they were used, each was a
  // dependent load on the way, the staging pointers one per peer between the
  // exchange's volatile stores, which they might alias.
  unsigned long long* stage = nullptr;
  int calls = 0;
  if (warp < 2) {
    if (lane == 0) calls = gens[r];
    if (warp == 0 ? lane < world : lane == 0) stage = bufs[warp == 0 ? lane : rank];
  }
  // What steps 4-5 read -- the seed, pos, the accept row and this row's draft
  // distribution -- holds still for the call, so warp 1 copies it
  // into shared memory now and waits for it after the poll: each was a load, or for
  // the draft distribution a chain of them, after the selection.
  if (warp == 1) {
    if (lane == 0) sp_cp8(&s_seed, seed_p);
    if (lane == 1) sp_cp8(&s_pos, pos_p);
    if (mode == SP_VERIFY) {
      if (lane <= n_draft) sp_cp8(&s_arow[lane], cand + lane);   // cand[0..K]
      if (r < n_draft && lane < SP_TOPK) {
        sp_cp8(&s_qid[lane], q_ids + (size_t)r * SP_TOPK + lane);
        sp_cp4(&s_qp[lane], q_probs + (size_t)r * SP_TOPK + lane);
      }
    }
  }

  // 1. This rank's 20 largest, without 20 sequential extractions. Every thread holds
  // at least one logit (v_local >= blockDim), keys are distinct, and:
  //   T_w, the 20th largest of the 32 warp maxima, is at most the 20th largest
  //   thread maximum T_t, so every thread maximum >= T_t is among those >= T_w,
  //   which sit in exactly 20 warps: at most SP_LIST = 20 x 32;
  //   the 20 largest keys are >= T_t, and keys >= T_t sit in exactly the 20 threads
  //   whose maxima are >= T_t: at most SP_CAND = 20 x SP_PER.
  // Ranking the thread maxima for T_t puts the 20 largest, in order, at the head of
  // s_cand. When they are the only keys >= T_t, they are the answer; otherwise the
  // other keys >= T_t follow them, at most
  // 19 x 19 (T_t's own thread has no other key >= T_t), and warps 0 and 1 each sort
  // the list (sp_top20).
  const float* row = logits + (size_t)r * (size_t)ld;
  const long long gid0 = (long long)rank * v_local;
  // Five clamped 16-byte loads per thread, issued together: thread t holds the 4-wide
  // chunks t, t + blockDim,..., so logit k sits at local id sp_id(k) (v_local % 4 ==
  // 0, rows 16-byte aligned). The thread's maximum compares u32 keys in id order (ties
  // keep the lower id, as the packed order does), and only it is packed.
  unsigned key[SP_PER];
  const float4* row4 = reinterpret_cast<const float4*>(row);
  const int last4 = v_local / 4 - 1;
#pragma unroll
  for (int c = 0; c < SP_PER / 4; ++c) {
    const float4 f = row4[min(tid + c * nthr, last4)];
    key[4 * c] = sp_key(f.x);
    key[4 * c + 1] = sp_key(f.y);
    key[4 * c + 2] = sp_key(f.z);
    key[4 * c + 3] = sp_key(f.w);
  }
#define sp_id(k) (4 * (tid + ((k) >> 2) * nthr) + ((k) & 3))
  unsigned live = 0, kmax = 0;
  int imax = 0;
  if (sp_id(SP_PER - 1) < v_local) {
    // Every chunk in range, as for most threads: no bounds to test.
    live = (1u << SP_PER) - 1u;
    kmax = key[0];
#pragma unroll
    for (int k = 1; k < SP_PER; ++k)
      if (key[k] > kmax) { kmax = key[k]; imax = k; }
  } else {
#pragma unroll
    for (int k = 0; k < SP_PER; ++k) {
      if (sp_id(k) < v_local) {
        live |= 1u << k;
        if (k == 0 || key[k] > kmax) { kmax = key[k]; imax = k; }
      }
    }
  }
  const long long tmax = sp_pack(kmax, gid0 + sp_id(imax));
  const long long wmax = sp_wmax(tmax);
  if (lane == 0) s_wmax[warp] = wmax;
  if (tid == 0) {
    s_nlist = 0;
    s_nextra = 0;
  }
  __syncthreads();
  if (warp == 0) {
    const long long mine = s_wmax[lane];
    int rank_w = 0;
    for (int j = 0; j < 32; ++j) rank_w += s_wmax[j] > mine;
    if (rank_w == SP_TOPK - 1) s_bound = mine;
  }
  __syncthreads();
  const long long t_w = s_bound;
  if (tmax >= t_w) s_list[atomicAdd(&s_nlist, 1)] = tmax;
  __syncthreads();
  const int nlist = s_nlist;
  if (tid < nlist) {
    const long long mine = s_list[tid];
    int rank_t = 0;
    for (int j = 0; j < nlist; ++j) rank_t += s_list[j] > mine;
    if (rank_t < SP_TOPK) s_cand[rank_t] = mine;
    if (rank_t == SP_TOPK - 1) s_bound = mine;
  }
  __syncthreads();
  const long long t_t = s_bound;
  if (tmax >= t_t) {
    // These 20 threads sit in up to 20 warps, each running this code for one lane, so
    // its length is the cost. A key is >= T_t only if its u32 key is >= T_t's, and
    // most threads have one such key, their maximum, already in s_cand: counting those
    // is the short way. A thread with more resolves a tie with T_t's key by id and
    // appends the others.
    const unsigned tkey = (unsigned)((unsigned long long)t_t >> 31);
    int n = 0;
#pragma unroll
    for (int k = 0; k < SP_PER; ++k) n += (int)((live >> k & 1u) & (key[k] >= tkey));
    if (n > 1) {
      const int tid_at = (int)(SP_ID_MASK - (t_t & SP_ID_MASK) - gid0);   // T_t's local id
      unsigned hit = 0;
#pragma unroll
      for (int k = 0; k < SP_PER; ++k)
        hit |= (unsigned)((key[k] > tkey) | ((key[k] == tkey) & (sp_id(k) <= tid_at))) << k;
      hit &= live & ~(1u << imax);                     // the maximum is in s_cand already
      int at = SP_TOPK + atomicAdd(&s_nextra, __popc(hit));
#pragma unroll
      for (int k = 0; k < SP_PER; ++k)
        if (hit >> k & 1u) s_cand[at++] = sp_pack(key[k], gid0 + sp_id(k));
    }
  }
  __syncthreads();
  if (warp > 1) return;
  const int ncand = SP_TOPK + s_nextra;
  const long long sel = ncand == SP_TOPK ? (lane < SP_TOPK ? s_cand[lane] : -1)
                                         : sp_top20(s_cand, ncand, lane);
  const long long mine = lane < SP_TOPK ? sel : -1;   // this rank's 20, descending
  const unsigned gen = (unsigned)__shfl_sync(SP_FULL, calls, 0) + 1u;
  const size_t base = (size_t)((gen & 1u) * SP_ROWS + r) * world * SP_WORDS;

  // 2. Every rank's 20 to every rank, in NCCL's LL form: each 8-byte word carries half
  // a key and the call's generation, so a word is valid once its generation matches,
  // with no fence and no flag. The staging half alternates with the generation's
  // parity.
  if (warp == 0) {
    if (lane == 0) gens[r] = (int)gen;   // this row's call count; only this CTA moves it
    const unsigned long long tag = (unsigned long long)gen << 32;
    const unsigned long long lo = tag | ((unsigned long long)mine & 0xFFFFFFFFull);
    const unsigned long long hi = tag | ((unsigned long long)mine >> 32);
    for (int p = 0; p < world; ++p) {
      unsigned long long* const peer = reinterpret_cast<unsigned long long*>(
          __shfl_sync(SP_FULL, reinterpret_cast<unsigned long long>(stage), p));
      if (lane < SP_TOPK) {
        volatile unsigned long long* dst = peer + base + (size_t)rank * SP_WORDS + 2 * lane;
        dst[0] = lo;
        dst[1] = hi;
      }
    }
    return;
  }

  // 3. (Warp 1.) The vocabulary's 20 largest of the world x 20 keys, each rank's 20
  // descending. Lane i takes element i of every even rank's list and element 31 - i of
  // every odd rank's (-1 past the 20th, and for ranks past world), so in lane order a
  // descending list meets an ascending one, and three levels of merges leave the 20
  // largest descending in lanes 0..19. This rank's own 20 come from registers; the
  // others' are polled from its staging, all of a lane's words in one pass, so a key
  // costs one round trip rather than one per word.
  unsigned long long* const own = reinterpret_cast<unsigned long long*>(
      __shfl_sync(SP_FULL, reinterpret_cast<unsigned long long>(stage), 0));
  const long long mine_rev = __shfl_sync(SP_FULL, mine, 31 - lane);
  long long v[SP_WORLD];
  unsigned pend = 0;
#pragma unroll
  for (int p = 0; p < SP_WORLD; ++p) {
    v[p] = p == rank ? ((p & 1) ? mine_rev : mine) : -1;
    if (p < world && p != rank && ((p & 1) ? 31 - lane : lane) < SP_TOPK) pend |= 1u << p;
  }
  while (pend) {
    // Every load of a pass issues before any is compared.
    unsigned long long w0[SP_WORLD], w1[SP_WORLD];
#pragma unroll
    for (int p = 0; p < SP_WORLD; ++p) {
      if (pend >> p & 1u) {
        volatile unsigned long long* src =
            own + base + (size_t)p * SP_WORDS + 2 * ((p & 1) ? 31 - lane : lane);
        w0[p] = src[0];
        w1[p] = src[1];
      }
    }
#pragma unroll
    for (int p = 0; p < SP_WORLD; ++p) {
      if ((pend >> p & 1u) && (w0[p] >> 32) == gen && (w1[p] >> 32) == gen) {
        v[p] = (long long)((w0[p] & 0xFFFFFFFFull) | (w1[p] << 32));
        pend &= ~(1u << p);
      }
    }
  }
  // Lanes leave the poll on different passes. Without this the compiler ran every
  // later shuffle on its diverged path. Exactly one in the kernel: with two in a row
  // ptxas (CUDA 13.0) kept neither, and with a second in the verify's finalizer
  // neither again. It also shows each lane's entry copies to the whole warp.
  sp_cp_wait();
  __syncwarp();
  const long long pos0 = s_pos;
  const unsigned long long seed = (unsigned long long)s_seed;
  const uint4 u = sp_philox(seed,
                            mode == SP_VERIFY ? pos0 + 1 + r
                                              : (mode == SP_SAMPLE ? pos0 : pos0 + 1 + depth),
                            mode == SP_DRAFT ? 1u + depth : 0u);
  const long long m01 = sp_merge32(sp_max(v[0], v[1]), lane, true);
  const long long m23 = sp_merge32(sp_max(v[2], v[3]), lane, false);
  const long long m45 = sp_merge32(sp_max(v[4], v[5]), lane, true);
  const long long m67 = sp_merge32(sp_max(v[6], v[7]), lane, false);
  const long long m03 = sp_merge32(sp_max(m01, m23), lane, true);
  const long long m47 = sp_merge32(sp_max(m45, m67), lane, false);
  const long long all = sp_merge32(sp_max(m03, m47), lane, true);
  const long long top = lane < SP_TOPK ? all : -1;

  // 4. The served distribution over them.
  const bool in = lane < SP_TOPK && top >= 0;
  const float x = in ? sp_value(top) : 0.f;
  const long long gid = in ? sp_gid(top) : -1;
  const float x0 = __shfl_sync(SP_FULL, x, 0);
  const float e = in ? expf(x - x0) : 0.f;
  const float p = e / sp_wsum(e);
  const float tail = sp_rscan(p, lane);        // own mass + every smaller one's
  const bool keep = in && (lane == 0 || tail > 1.f - top_p);
  const float ek = keep ? e : 0.f;
  const float prob = ek / sp_wsum(ek);

  // 5. The draw.
  if (mode != SP_VERIFY) {
    const long long tok = __shfl_sync(SP_FULL, gid, sp_pick(prob, sp_unit(u.y), lane));
    if (lane == 0) out[0] = tok;
    if (mode == SP_DRAFT && lane < SP_TOPK) {
      q_ids[depth * SP_TOPK + lane] = gid;
      q_probs[depth * SP_TOPK + lane] = prob;
      q_logits[depth * SP_TOPK + lane] = x;
    }
    return;
  }

  // The verify: row r decides position pos0 + 1 + r.
  if (lane < SP_TOPK) {
    p_ids[r * SP_TOPK + lane] = gid;
    p_probs[r * SP_TOPK + lane] = prob;
  }
  const long long d = r < n_draft ? s_arow[r + 1] : -1;
  int acc = 0;
  long long tok;
  if (d >= 0) {
    float q_mine = 0.f, q_d = 0.f;
#pragma unroll
    for (int j = 0; j < SP_TOPK; ++j) {
      const long long id = s_qid[j];
      if (in && id == gid) q_mine = s_qp[j];
      if (id == d) q_d = s_qp[j];
    }
    const float p_d = sp_wsum(gid == d ? prob : 0.f);
    acc = sp_unit(u.x) * q_d < p_d;
    if (acc) {
      tok = d;
    } else {
      // The residual max(0, p - q) is zero only if p <= q everywhere, which a
      // rejection rules out up to rounding; then draw from p.
      const float res = fmaxf(prob - q_mine, 0.f);
      const int w = sp_wsum(res) > 0.f ? sp_pick(res, sp_unit(u.y), lane)
                                       : sp_pick(prob, sp_unit(u.y), lane);
      tok = __shfl_sync(SP_FULL, gid, w);
    }
  } else {
    tok = __shfl_sync(SP_FULL, gid, sp_pick(prob, sp_unit(u.y), lane));
  }
  bool last = false;
  int n = 0;
  long long y = 0;
  if (lane == 0) {
    row_acc[r] = acc;
    row_tok[r] = tok;
    __threadfence();
    last = atomicAdd(done, 1) == rows - 1;
    if (last) {
      // The last row to finish: after its acquire fence it reads every row's slot at
      // once, the accepted prefix no longer a chain of dependent loads.
      __threadfence();
      int a[SP_ROWS];
      long long t[SP_ROWS];
#pragma unroll
      for (int i = 0; i < SP_ROWS; ++i) {
        a[i] = i < rows ? ((volatile int*)row_acc)[i] : 0;
        t[i] = i < rows ? ((volatile long long*)row_tok)[i] : 0;
      }
      // n: the first row that did not accept; the last row never does (its bonus draw).
      n = rows - 1;
#pragma unroll
      for (int i = SP_ROWS - 2; i >= 0; --i)
        if (i < rows - 1 && !a[i]) n = i;
      y = t[0];
#pragma unroll
      for (int i = 1; i < SP_ROWS; ++i)
        if (i == n) y = t[i];
    }
  }
  if (__shfl_sync(SP_FULL, (int)last, 0)) {
    // The step's outputs, a lane each. Only lane 0 read the rows, after its fence: the
    // other lanes would need a barrier behind it, and a second __syncwarp costs the
    // poll's reconvergence (above).
    n = __shfl_sync(SP_FULL, n, 0);
    y = __shfl_sync(SP_FULL, y, 0);
    if (lane < q) {
      // Depth 0's row i is logical row r, seed row first (position.seed_first_logical).
      const int lr = lane == 0 ? q - 1 : (lane == q - 1 ? 0 : lane);
      dtok[lane] = lr < n ? s_arow[lr + 1] : y;
      hidx[lane] = lr < n ? lr : n;
    }
    if (lane == 0) {
      out[0] = n;
      out[1] = y;
      *pos_p = pos0 + 1 + n;
      *done = 0;
    }
  }
}

void sample_launch(int64_t logits, int64_t ld, int64_t rows, int64_t v_local,
                   int64_t mode, int64_t depth, int64_t d_bufs, int64_t gens, int64_t world, int64_t rank, int64_t seed,
                   int64_t pos, int64_t q_ids, int64_t q_probs, int64_t q_logits,
                   int64_t p_ids, int64_t p_probs, int64_t cand, int64_t dtok,
                   int64_t hidx, int64_t q, int64_t out, int64_t row_acc,
                   int64_t row_tok, int64_t done, int64_t n_draft, double top_p) {
  auto s = at::cuda::getCurrentCUDAStream();
  sample_kernel<<<(int)rows, 1024, 0, s>>>(
      reinterpret_cast<const float*>(logits), ld, (int)rows, (int)v_local, (int)mode,
      (int)depth, reinterpret_cast<unsigned long long**>(d_bufs),
      reinterpret_cast<int*>(gens),
      (int)world, (int)rank, reinterpret_cast<const long long*>(seed),
      reinterpret_cast<long long*>(pos), reinterpret_cast<long long*>(q_ids),
      reinterpret_cast<float*>(q_probs), reinterpret_cast<float*>(q_logits),
      reinterpret_cast<long long*>(p_ids),
      reinterpret_cast<float*>(p_probs), reinterpret_cast<const long long*>(cand),
      reinterpret_cast<long long*>(dtok), reinterpret_cast<long long*>(hidx), (int)q,
      reinterpret_cast<long long*>(out), reinterpret_cast<int*>(row_acc),
      reinterpret_cast<long long*>(row_tok), reinterpret_cast<int*>(done),
      (int)n_draft, (float)top_p);
  CK(cudaGetLastError());
}
"""

_EXT = None


def _build():
    with cuda_arch():
        return load_inline(name="iquestrt_sampler", cpp_sources=_CPP, cuda_sources=_CUDA,
                           functions=["sp_alloc", "sp_open", "sp_register", "sample_launch"],
                           extra_cuda_cflags=["-O3"], verbose=False)


def _ext():
    """Compile on rank 0 first, then let the others load from cache.

    All ranks calling `load_inline` at once contend on torch's build lock. Serialising
    it makes the first call cost one compile instead of four."""
    global _EXT
    if _EXT is not None:
        return _EXT
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        if dist.get_rank() == 0:
            _EXT = _build()
        dist.barrier()
        if dist.get_rank() != 0:
            _EXT = _build()          # cache hit now
        dist.barrier()
    else:
        _EXT = _build()
    return _EXT


def _check(name, t, dtype, shape=None):
    if t.dtype != dtype or t.stride(-1) != 1 or (shape is not None and t.shape != shape):
        raise ValueError(f"Sampler: {name} needs unit-stride {dtype}"
                         + (f"{list(shape)}" if shape is not None else "")
                         + f", got {t.dtype}{list(t.shape)} stride {t.stride()}")


class Sampler:
    """iquest_q1's served distribution over vocab-parallel fp32 logits, drawn on the device.

    `world == 1` is allowed and stages the exchange locally, so the kernel is testable
    on one GPU. Calls that use the same row must run in the same order on every rank;
    the Engine's step does."""

    def __init__(self, v_local: int, n_draft: int):
        if not THREADS <= v_local <= THREADS * PER_THREAD or v_local % 4:
            raise ValueError(f"Sampler: a rank's slice holds {THREADS}.."
                             f"{THREADS * PER_THREAD} logits, a multiple of 4; got "
                             f"{v_local}")
        if not 0 < n_draft < ROWS:
            raise ValueError(f"Sampler: the verify draws {ROWS} rows at most, so at most "
                             f"{ROWS - 1} drafts; got {n_draft}")
        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        if self.world > WORLD:
            raise ValueError(f"Sampler: the exchange stages {WORLD} ranks, got {self.world}")
        self.v_local, self.n_draft = v_local, n_draft
        ext = _ext()
        # Two halves x rows x ranks x 40 LL words of 8 bytes, zeroed: generation 0
        # matches no call.
        stage = 2 * ROWS * self.world * 2 * TOP_K * 8
        bufs = self._share(ext, stage) if self.world > 1 else [ext.sp_alloc(stage)[0]]
        self.d_bufs = ext.sp_register(bufs)
        i32, i64, f32 = torch.int32, torch.int64, torch.float32
        self.gens = torch.zeros(ROWS, dtype=i32, device="cuda")
        self.done = torch.zeros(1, dtype=i32, device="cuda")
        self.row_acc = torch.zeros(ROWS, dtype=i32, device="cuda")
        self.row_tok = torch.zeros(ROWS, dtype=i64, device="cuda")
        # The request's state: the seed, the position of the last committed token, and
        # each draft depth's distribution, read by the next verify.
        self.seed = torch.zeros(1, dtype=i64, device="cuda")
        self.pos = torch.zeros(1, dtype=i64, device="cuda")
        self.q_ids = torch.full((n_draft, TOP_K), -1, dtype=i64, device="cuda")
        self.q_probs = torch.zeros((n_draft, TOP_K), dtype=f32, device="cuda")
        # Each depth's 20 served logits, largest first, for the stop rule.
        self.q_logits = torch.zeros((n_draft, TOP_K), dtype=f32, device="cuda")
        # The verify rows' distributions; nothing reads them back.
        self.p_ids = torch.full((ROWS, TOP_K), -1, dtype=i64, device="cuda")
        self.p_probs = torch.zeros((ROWS, TOP_K), dtype=f32, device="cuda")

    def _share(self, ext, nbytes: int) -> list[int]:
        got = ext.sp_alloc(nbytes)
        mine, handle = got[0], got[1:]
        gathered = [None] * self.world
        dist.all_gather_object(gathered, handle)
        return [mine if i == self.rank else ext.sp_open(list(h))
                for i, h in enumerate(gathered)]

    def start(self, seed: int, pos: int) -> None:
        """A new request: draws keyed by `seed`, its first token at position `pos`."""
        if not 0 <= seed < 1 << 64:
            raise ValueError(f"Sampler: the seed is 64-bit unsigned, got {seed}")
        self.seed.fill_(seed - (1 << 64) if seed >= 1 << 63 else seed)
        self.pos.fill_(pos)
        self.q_ids.fill_(-1)
        self.q_probs.zero_()

    def state(self) -> tuple[torch.Tensor,...]:
        """The tensors a step reads and writes across steps (capture saves them)."""
        return self.seed, self.pos, self.q_ids, self.q_probs

    def sample(self, logits: torch.Tensor, out: torch.Tensor) -> None:
        """The token at position `pos`, into out[0]."""
        self._launch(logits, 1, SAMPLE, 0, out)

    def draft(self, logits: torch.Tensor, depth: int, out: torch.Tensor) -> None:
        """Draft depth `depth`'s token for position pos+1+depth, into out[0]; its
        distribution into q_ids[depth]/q_probs[depth], its 20 logits into
        q_logits[depth]."""
        if not 0 <= depth < self.n_draft:
            raise ValueError(f"Sampler: depth {depth} outside 0..{self.n_draft - 1}")
        self._launch(logits, 1, DRAFT, depth, out)

    def verify(self, logits: torch.Tensor, cand: torch.Tensor, dtok: torch.Tensor,
               hidx: torch.Tensor, result: torch.Tensor, drafts: int | None = None) -> None:
        """The rejection rule over rows 0..K of `logits`, K = `drafts` (default
        `n_draft`), -> result = (n, y).

        `cand` is the accept row (`Engine.verify_rows`): cand[r+1] is row r's draft,
        negative for none. Writes depth 0's rows into `dtok`/`hidx` (q wide) and
        advances `pos` by 1 + n."""
        k = self.n_draft if drafts is None else drafts
        if not 0 < k <= self.n_draft:
            raise ValueError(f"Sampler: a verify reads 1..{self.n_draft} drafts, got {k}")
        q = cand.shape[0]
        if q <= k:
            raise ValueError(f"Sampler: {k} drafts need {k + 1} candidates, got {q}")
        for name, t in (("cand", cand), ("dtok", dtok), ("hidx", hidx)):
            _check(name, t, torch.int64, (q,))
        _check("result", result, torch.int64, (2,))
        self._launch(logits, k + 1, VERIFY, 0, result, cand, dtok, hidx, n_draft=k)

    def _launch(self, logits, rows, mode, depth, out, cand=None, dtok=None, hidx=None,
                n_draft=None):
        _check("logits", logits, torch.float32)
        if logits.data_ptr() % 16 or logits.stride(0) % 4:
            raise ValueError("Sampler: logits rows must be 16-byte aligned")
        if logits.dim() != 2 or logits.shape[0] < rows or logits.shape[1] != self.v_local:
            raise ValueError(f"Sampler: logits need [>={rows}, {self.v_local}], got "
                             f"{list(logits.shape)}")
        _check("out", out, torch.int64)
        _ext().sample_launch(
            logits.data_ptr(), logits.stride(0), rows, self.v_local, mode, depth,
            self.d_bufs, self.gens.data_ptr(), self.world, self.rank,
            self.seed.data_ptr(), self.pos.data_ptr(), self.q_ids.data_ptr(),
            self.q_probs.data_ptr(), self.q_logits.data_ptr(), self.p_ids.data_ptr(),
            self.p_probs.data_ptr(),
            cand.data_ptr() if cand is not None else 0,
            dtok.data_ptr() if dtok is not None else 0,
            hidx.data_ptr() if hidx is not None else 0,
            cand.shape[0] if cand is not None else 0, out.data_ptr(),
            self.row_acc.data_ptr(), self.row_tok.data_ptr(), self.done.data_ptr(),
            self.n_draft if n_draft is None else n_draft, TOP_P)
