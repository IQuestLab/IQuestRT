"""NVLS all-reduce fused with the attention residual and two RMSNorms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from .arch import cuda_arch
from .tail import TailKernels

HIDDEN = 3072
THREADS = 384
WORLD = 8
Q = 8

_CPP = r"""
void ar_tail_direct1bar_launch(
    int64_t mc, int64_t residual, int64_t w_ao, int64_t w_ffn,
    int64_t scale, int64_t eps, int64_t h_out, int64_t hn_out,
    int64_t hn32_out, int64_t flag_ptrs, int64_t rank, int64_t world,
    int64_t q);
void ar_tail_direct1bar_h_inv_xpad_fp8_pdl_overlap_local_launch(
    int64_t mc, int64_t residual, int64_t w_ao, int64_t w_ffn,
    int64_t scale, int64_t eps, int64_t h_out, int64_t inv_out,
    int64_t x_pad, int64_t x_pad_fp8, int64_t flag_ptrs,
    int64_t rank, int64_t world, int64_t q);
void mlp_tail_norm_overlap_local_launch(
    int64_t mc, int64_t residual, int64_t scale, int64_t w_norm,
    int64_t eps, int64_t out, int64_t norm_out, int64_t flag_ptrs,
    int64_t rank, int64_t world, int64_t q);
void mtp_eh_ar_norm_head0_pdl_launch(
    int64_t mc, int64_t w_norm, int64_t eps, int64_t raw_out,
    int64_t norm_out, int64_t flag_ptrs, int64_t rank, int64_t world,
    int64_t q);
void ar_tail_sandwich_h32_launch(
    int64_t mc, int64_t residual32, int64_t w_ao, int64_t w_ffn,
    int64_t scale, int64_t eps, int64_t h32_out, int64_t hn_out,
    int64_t flag_ptrs, int64_t rank, int64_t world, int64_t q);
void ar_tail_h32_inv_xpad_fp8_launch(
    int64_t mc, int64_t residual, int64_t w_ao, int64_t w_ffn,
    int64_t scale, int64_t eps, int64_t h32_out, int64_t inv_out,
    int64_t x_pad, int64_t x_pad_fp8, int64_t flag_ptrs,
    int64_t rank, int64_t world, int64_t q);
"""

_REDUCE = r"""
// Threads 0..127 reproduce TileLang's AllReduce<SumOp, 128>.  Threads 128..383
// participate in the NVLS load and then retire.  Named barriers let the first group
// retain the exact 128-lane reduction association without waiting for retired threads.
__device__ __forceinline__ void sync_group_128(int group) {
  // Barrier 0 belongs to __syncthreads(); rows use named barriers 1..3.
  const int barrier = group + 1;
  asm volatile("bar.sync %0, 128;" :: "r"(barrier) : "memory");
}

// TileLang's association: lane l adds lane
// l ^ 64, then lane l ^ 32's such sum, then the XOR tree 16..1 within its warp.  One
// shared-memory round gives the same bits as TileLang's two: lane l reads its three
// partners and forms (x + x[l ^ 64]) + (x[l ^ 32] + x[l ^ 96]), the second sum being
// the one lane l ^ 32 formed.  The leading barrier keeps this call's stores behind the
// previous call's loads, so consecutive calls need no barrier between them.
__device__ __forceinline__ float reduce_group_128(
    float x, float* red, int group, int lane) {
  sync_group_128(group);
  red[lane] = x;
  sync_group_128(group);
  x = (x + red[lane ^ 64]) + (red[lane ^ 32] + red[lane ^ 96]);
#pragma unroll
  for (int off = 16; off; off >>= 1) {
    x += __shfl_xor_sync(0xffffffffu, x, off);
  }
  return x;
}
"""

_CUDA = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <atomic>
#include <cuda/atomic>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

constexpr int H = 3072;
constexpr int NT = 128;
constexpr int DIRECT_NT = 384;

// PyTorch's BF16 multimem path: the NVSwitch accumulator is FP32 and the four
// registers contain eight BF16 results, including the production BF16 rounding point.
struct U4 { unsigned int x, y, z, w; };

__device__ __forceinline__ uint64_t ar_bf16x8_to_e4m3x8(
    const U4& packed) {
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

// Keep the rank-local residual load physically before the remote peer
// wait.  A plain C++ load is semantically movable because the signal loop does not
// alias the residual; explicit PTX makes materialisation of the intended overlap an
// inspectable property rather than an optimiser heuristic.
__device__ __forceinline__ U4 ld_global_u4(const void* ptr) {
  U4 v;
  asm volatile(
      "ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(ptr)
      : "memory");
  return v;
}

__device__ __forceinline__ U4 mm_ld_reduce_bf16x8(
    const __nv_bfloat16* mc) {
  U4 v;
  asm volatile(
      "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
      "{%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(mc)
      : "memory");
  return v;
}

// signal_cas, put_signal, wait_signal and sync_remote_blocks are adapted from PyTorch
// (torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.h),
// Copyright (c) 2016- Facebook, Inc and its contributors, BSD-3-Clause.
// See LICENSE-PyTorch.txt in this directory.
template <std::memory_order Sem>
__device__ __forceinline__ unsigned int signal_cas(
    unsigned int* addr, unsigned int compare, unsigned int val) {
  ::cuda::atomic_ref<unsigned int, ::cuda::thread_scope_system> ref(*addr);
  ref.compare_exchange_strong(compare, val, ::cuda::std::memory_order(Sem));
  return compare;
}

template <std::memory_order Sem>
__device__ __forceinline__ void put_signal(unsigned int* addr) {
  while (signal_cas<Sem>(addr, 0, 1) != 0) { }
}

template <std::memory_order Sem>
__device__ __forceinline__ void wait_signal(unsigned int* addr) {
  while (signal_cas<Sem>(addr, 1, 0) != 1) { }
}

template <bool prev_access, bool next_access>
__device__ __forceinline__ void sync_remote_blocks(
    unsigned int** signal_pads, int rank, int world) {
  if (threadIdx.x < world) {
    const int peer = threadIdx.x;
    if constexpr (prev_access) {
      put_signal<std::memory_order_release>(
          signal_pads[peer] + blockIdx.x * world + rank);
    } else {
      put_signal<std::memory_order_relaxed>(
          signal_pads[peer] + blockIdx.x * world + rank);
    }
    if constexpr (next_access) {
      wait_signal<std::memory_order_acquire>(
          signal_pads[rank] + blockIdx.x * world + peer);
    } else {
      wait_signal<std::memory_order_relaxed>(
          signal_pads[rank] + blockIdx.x * world + peer);
    }
  }
}

""" + _REDUCE + r"""
// Fp32Residual (the recursive draft's fp32_residual_connection): the residual comes in
// as fp32 and h = residual + BF16(BF16(norm) * a_scale) stays fp32 -- written to h32_out
// and normalised by the second RMSNorm without a BF16 rounding.
template <bool Fp32Residual = false>
__global__ __launch_bounds__(DIRECT_NT, 1) void ar_tail_direct1bar_impl(
    const __nv_bfloat16* mc,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ w_ao,
    const float* __restrict__ w_ffn,
    const float* __restrict__ scale,
    const float* __restrict__ eps,
    __nv_bfloat16* __restrict__ h_out,
    __nv_bfloat16* __restrict__ hn_out,
    float* __restrict__ hn32_out,
    unsigned int** signal_pads,
    int rank, int world, int q,
    const float* __restrict__ residual32,
    float* __restrict__ h32_out) {
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= q) return;

  __shared__ __align__(16) __nv_bfloat16 ao_s[H];
  __shared__ __align__(16) __nv_bfloat16 h_s[H];
  __shared__ __align__(16) float h32_s[Fp32Residual ? H : 4];
  __shared__ float red[NT];

  // The sole cross-rank barrier.  The following CTA barrier prevents any NVLS read
  // from being issued before all eight ranks have published the current input.
  sync_remote_blocks<false, true>(signal_pads, rank, world);
  __syncthreads();

  // Every rank redundantly reads the complete switch-reduced row.  Storing the packed
  // registers in shared memory preserves multimem_all_reduce_'s BF16 result rounding.
  const __nv_bfloat16* mc_row = mc + static_cast<size_t>(row) * H;
  constexpr int N8 = H / 8;
  for (int i = tid; i < N8; i += DIRECT_NT) {
    const U4 v = mm_ld_reduce_bf16x8(mc_row + 8 * i);
    *reinterpret_cast<U4*>(ao_s + 8 * i) = v;
  }
  __syncthreads();

  if (tid >= NT) return;
  const int lane = tid;

  // First RMSNorm.  The local lane-major summation and 128-thread XOR tree match
  // TileLang's 128-thread reduce_sum.
  float sq[24];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const U4 u = *reinterpret_cast<const U4*>(ao_s + i * 1024 + lane * 8);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(b[j]);
      sq[i * 8 + 2 * j] = f.x * f.x;
      sq[i * 8 + 2 * j + 1] = f.y * f.y;
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int rv = 0; rv < 24; ++rv) {
    sum += sq[(rv % 3) * 8 + rv / 3];
  }
  sum = reduce_group_128(sum, red, 0, lane);
  const float inv = rsqrtf(sum / 3072.0f + eps[0]);

  // Preserve all three reference rounding points:
  // BF16(norm) -> BF16(* a_scale) -> BF16(+ residual).
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int base = i * 512 + lane * 4;
    const float4 w = *reinterpret_cast<const float4*>(w_ao + base);
    const uint2 au = *reinterpret_cast<const uint2*>(ao_s + base);
    const __nv_bfloat162* ab = reinterpret_cast<const __nv_bfloat162*>(&au);
    float4 av;
    reinterpret_cast<float2*>(&av)[0] = __bfloat1622float2(ab[0]);
    reinterpret_cast<float2*>(&av)[1] = __bfloat1622float2(ab[1]);
    const float4 norm = make_float4(
        w.x * (av.x * inv), w.y * (av.y * inv),
        w.z * (av.z * inv), w.w * (av.w * inv));

    uint2 nv;
    reinterpret_cast<__nv_bfloat162*>(&nv)[0] =
        __float22bfloat162_rn(reinterpret_cast<const float2*>(&norm)[0]);
    reinterpret_cast<__nv_bfloat162*>(&nv)[1] =
        __float22bfloat162_rn(reinterpret_cast<const float2*>(&norm)[1]);
    const __nv_bfloat162* nvb = reinterpret_cast<const __nv_bfloat162*>(&nv);
    float4 nvf;
    reinterpret_cast<float2*>(&nvf)[0] = __bfloat1622float2(nvb[0]);
    reinterpret_cast<float2*>(&nvf)[1] = __bfloat1622float2(nvb[1]);

    const float s = scale[0];
    const float4 scaled = make_float4(
        nvf.x * s, nvf.y * s, nvf.z * s, nvf.w * s);
    uint2 sv;
    reinterpret_cast<__nv_bfloat162*>(&sv)[0] =
        __float22bfloat162_rn(reinterpret_cast<const float2*>(&scaled)[0]);
    reinterpret_cast<__nv_bfloat162*>(&sv)[1] =
        __float22bfloat162_rn(reinterpret_cast<const float2*>(&scaled)[1]);
    const __nv_bfloat162* svb = reinterpret_cast<const __nv_bfloat162*>(&sv);
    float4 svf;
    reinterpret_cast<float2*>(&svf)[0] = __bfloat1622float2(svb[0]);
    reinterpret_cast<float2*>(&svf)[1] = __bfloat1622float2(svb[1]);

    if constexpr (Fp32Residual) {
      const float4 rf = *reinterpret_cast<const float4*>(
          residual32 + static_cast<size_t>(row) * H + base);
      const float4 added = make_float4(
          svf.x + rf.x, svf.y + rf.y, svf.z + rf.z, svf.w + rf.w);
      *reinterpret_cast<float4*>(h32_s + base) = added;
      *reinterpret_cast<float4*>(
          h32_out + static_cast<size_t>(row) * H + base) = added;
    } else {
      const uint2 ru = *reinterpret_cast<const uint2*>(
          residual + static_cast<size_t>(row) * H + base);
      const __nv_bfloat162* rb = reinterpret_cast<const __nv_bfloat162*>(&ru);
      float4 rf;
      reinterpret_cast<float2*>(&rf)[0] = __bfloat1622float2(rb[0]);
      reinterpret_cast<float2*>(&rf)[1] = __bfloat1622float2(rb[1]);
      const float4 added = make_float4(
          svf.x + rf.x, svf.y + rf.y, svf.z + rf.z, svf.w + rf.w);

      uint2 hv;
      reinterpret_cast<__nv_bfloat162*>(&hv)[0] =
          __float22bfloat162_rn(reinterpret_cast<const float2*>(&added)[0]);
      reinterpret_cast<__nv_bfloat162*>(&hv)[1] =
          __float22bfloat162_rn(reinterpret_cast<const float2*>(&added)[1]);
      *reinterpret_cast<uint2*>(h_s + base) = hv;
      *reinterpret_cast<uint2*>(
          h_out + static_cast<size_t>(row) * H + base) = hv;
    }
  }
  sync_group_128(0);

  // Dependent second RMSNorm, with the same local and block reduction order.
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    if constexpr (Fp32Residual) {
      const float4 lo = *reinterpret_cast<const float4*>(h32_s + i * 1024 + lane * 8);
      const float4 hi = *reinterpret_cast<const float4*>(h32_s + i * 1024 + lane * 8 + 4);
      const float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) sq[i * 8 + j] = v[j] * v[j];
    } else {
      const U4 u = *reinterpret_cast<const U4*>(h_s + i * 1024 + lane * 8);
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(b[j]);
        sq[i * 8 + 2 * j] = f.x * f.x;
        sq[i * 8 + 2 * j + 1] = f.y * f.y;
      }
    }
  }
  sum = 0.0f;
#pragma unroll
  for (int rv = 0; rv < 24; ++rv) {
    sum += sq[(rv % 3) * 8 + rv / 3];
  }
  sum = reduce_group_128(sum, red, 0, lane);
  const float inv2 = rsqrtf(sum / 3072.0f + eps[0]);

#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int base = i * 512 + lane * 4;
    const float4 w = *reinterpret_cast<const float4*>(w_ffn + base);
    float4 hf;
    if constexpr (Fp32Residual) {
      hf = *reinterpret_cast<const float4*>(h32_s + base);
    } else {
      const uint2 hu = *reinterpret_cast<const uint2*>(h_s + base);
      const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&hu);
      reinterpret_cast<float2*>(&hf)[0] = __bfloat1622float2(hb[0]);
      reinterpret_cast<float2*>(&hf)[1] = __bfloat1622float2(hb[1]);
    }
    const float4 norm = make_float4(
        w.x * (hf.x * inv2), w.y * (hf.y * inv2),
        w.z * (hf.z * inv2), w.w * (hf.w * inv2));

    uint2 hv;
    reinterpret_cast<__nv_bfloat162*>(&hv)[0] =
        __float22bfloat162_rn(reinterpret_cast<const float2*>(&norm)[0]);
    reinterpret_cast<__nv_bfloat162*>(&hv)[1] =
        __float22bfloat162_rn(reinterpret_cast<const float2*>(&norm)[1]);
    *reinterpret_cast<uint2*>(
        hn_out + static_cast<size_t>(row) * H + base) = hv;

    // The fp32 router operand is optional.  When present it must widen the rounded
    // BF16 result, not expose the unrounded accumulator.
    if (hn32_out != nullptr) {
      const __nv_bfloat162* hvb = reinterpret_cast<const __nv_bfloat162*>(&hv);
      const float2 lo = __bfloat1622float2(hvb[0]);
      const float2 hi = __bfloat1622float2(hvb[1]);
      *reinterpret_cast<float4*>(
          hn32_out + static_cast<size_t>(row) * H + base) =
          make_float4(lo.x, lo.y, hi.x, hi.y);
    }
  }
}

// Producer half for the SM100 inv-router chain.  It preserves the NVLS barrier, first
// RMSNorm and every BF16 rounding point of ar_tail_direct1bar, but stops after h and the
// second norm's inverse RMS.  The dependent 96-CTA router applies w_ffn while consuming
// them, so writing hn here would be dead work.  PDL releases the dependent grid only
// after all h stores have been issued; its griddepcontrol.wait supplies the
// global-memory fence.
template <bool WaitPDL = false, bool OverlapLocal = false, bool EmitXPad = false,
          bool EarlyPDLTrigger = false>
__global__ __launch_bounds__(DIRECT_NT, 1) void ar_tail_direct1bar_h_inv(
    const __nv_bfloat16* mc,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ w_ao,
    const float* __restrict__ scale,
    const float* __restrict__ eps,
    __nv_bfloat16* __restrict__ h_out,
    float* __restrict__ inv_out,
    unsigned int** signal_pads,
    int rank, int world, int q,
    const float* __restrict__ w_ffn,
    __nv_bfloat16* __restrict__ x_pad,
    uint8_t* __restrict__ x_pad_fp8) {
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= q) return;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (WaitPDL)
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  __shared__ __align__(16) __nv_bfloat16 ao_s[H];
  __shared__ float red[NT];
  // OverlapLocal staging: the eleven non-signal warps load rank-local rows into
  // dynamic shared memory while warp 0 waits for peers.
  extern __shared__ __align__(16) unsigned char local_stage[];
  auto* residual_stage = reinterpret_cast<__nv_bfloat16*>(local_stage);
  auto* w_ao_stage = reinterpret_cast<float*>(residual_stage + H);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  // The Q8 dual-output consumer can schedule its CTAs and issue independent Router
  // weight loads while this producer is still waiting for its peers.  Its own
  // griddepcontrol.wait remains the completion and visibility fence before any x_pad
  // access.  Other h-only instantiations retain the tail trigger below.
  if constexpr (EarlyPDLTrigger)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  sync_remote_blocks<false, true>(signal_pads, rank, world);
  if constexpr (OverlapLocal) {
    // Lanes 0..7 own the system-scope signal protocol and the rest of warp 0 stays
    // parked with them.  Only tid>=32 may issue these local loads.  This code is after
    // griddepcontrol.wait and touches neither the OProj output nor the NVLS multicast
    // pointer; producer data remains protected by the PDL wait and the peer barrier.
    if (tid >= 32) {
      constexpr int LOADERS = DIRECT_NT - 32;
      const size_t row_base = static_cast<size_t>(row) * H;
      for (int pack = tid - 32; pack < H / 8; pack += LOADERS) {
        *reinterpret_cast<U4*>(residual_stage + pack * 8) =
            ld_global_u4(residual + row_base + pack * 8);
      }
      for (int pack = tid - 32; pack < H / 4; pack += LOADERS) {
        *reinterpret_cast<float4*>(w_ao_stage + pack * 4) =
            *reinterpret_cast<const float4*>(w_ao + pack * 4);
      }
    }
  }
  __syncthreads();
  const __nv_bfloat16* mc_row = mc + static_cast<size_t>(row) * H;
  constexpr int N8 = H / 8;
  for (int i = tid; i < N8; i += DIRECT_NT) {
    const U4 v = mm_ld_reduce_bf16x8(mc_row + 8 * i);
    *reinterpret_cast<U4*>(ao_s + 8 * i) = v;
  }
  __syncthreads();

  if (tid < NT) {
    const int lane = tid;
    float sq[24];
    U4 h_bits[3];
#pragma unroll
    for (int segment = 0; segment < 3; ++segment) {
      const U4 u = *reinterpret_cast<const U4*>(
          ao_s + segment * 1024 + lane * 8);
      const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 f = __bfloat1622float2(b[pair]);
        sq[segment * 8 + 2 * pair] = f.x * f.x;
        sq[segment * 8 + 2 * pair + 1] = f.y * f.y;
      }
    }
    float sum = 0.0f;
#pragma unroll
    for (int rv = 0; rv < 24; ++rv)
      sum += sq[(rv % 3) * 8 + rv / 3];
    sum = reduce_group_128(sum, red, 0, lane);
    const float inv = rsqrtf(sum / 3072.0f + eps[0]);

    const size_t row_base = static_cast<size_t>(row) * H;
    const float s = scale[0];
#pragma unroll
    for (int segment = 0; segment < 3; ++segment) {
      const int base = segment * 1024 + lane * 8;
      const U4 au = *reinterpret_cast<const U4*>(ao_s + base);
      const U4 ru = *reinterpret_cast<const U4*>(
          (OverlapLocal ? residual_stage : residual + row_base) + base);
      const __nv_bfloat162* ab = reinterpret_cast<const __nv_bfloat162*>(&au);
      const __nv_bfloat162* rb = reinterpret_cast<const __nv_bfloat162*>(&ru);
      U4 hv;
      __nv_bfloat162* hb = reinterpret_cast<__nv_bfloat162*>(&hv);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const float4 w = *reinterpret_cast<const float4*>(
            (OverlapLocal ? w_ao_stage : w_ao) + base + half * 4);
        const float2 a0 = __bfloat1622float2(ab[half * 2]);
        const float2 a1 = __bfloat1622float2(ab[half * 2 + 1]);
        const float4 norm = make_float4(
            w.x * (a0.x * inv), w.y * (a0.y * inv),
            w.z * (a1.x * inv), w.w * (a1.y * inv));
        uint2 rounded_norm;
        reinterpret_cast<__nv_bfloat162*>(&rounded_norm)[0] =
            __float22bfloat162_rn(reinterpret_cast<const float2*>(&norm)[0]);
        reinterpret_cast<__nv_bfloat162*>(&rounded_norm)[1] =
            __float22bfloat162_rn(reinterpret_cast<const float2*>(&norm)[1]);
        const __nv_bfloat162* nb =
            reinterpret_cast<const __nv_bfloat162*>(&rounded_norm);
        const float2 n0 = __bfloat1622float2(nb[0]);
        const float2 n1 = __bfloat1622float2(nb[1]);
        const float4 scaled = make_float4(
            n0.x * s, n0.y * s, n1.x * s, n1.y * s);
        uint2 rounded_scaled;
        reinterpret_cast<__nv_bfloat162*>(&rounded_scaled)[0] =
            __float22bfloat162_rn(reinterpret_cast<const float2*>(&scaled)[0]);
        reinterpret_cast<__nv_bfloat162*>(&rounded_scaled)[1] =
            __float22bfloat162_rn(reinterpret_cast<const float2*>(&scaled)[1]);
        const __nv_bfloat162* sb =
            reinterpret_cast<const __nv_bfloat162*>(&rounded_scaled);
        const float2 sf0 = __bfloat1622float2(sb[0]);
        const float2 sf1 = __bfloat1622float2(sb[1]);
        const float2 r0 = __bfloat1622float2(rb[half * 2]);
        const float2 r1 = __bfloat1622float2(rb[half * 2 + 1]);
        hb[half * 2] = __float22bfloat162_rn(
            make_float2(sf0.x + r0.x, sf0.y + r0.y));
        hb[half * 2 + 1] = __float22bfloat162_rn(
            make_float2(sf1.x + r1.x, sf1.y + r1.y));
      }
      h_bits[segment] = hv;
      *reinterpret_cast<U4*>(h_out + row_base + base) = hv;
    }

    // The production tail immediately reduces the rounded BF16 h row a second time.
    // Emit only that scalar and leave the 96-CTA router to apply w_ffn to its own K
    // slice.  This preserves the exact 128-lane association while avoiding both the
    // 18 KiB hn materialisation and 96 redundant full-row reductions.
#pragma unroll
    for (int segment = 0; segment < 3; ++segment) {
      const __nv_bfloat162* hb =
          reinterpret_cast<const __nv_bfloat162*>(&h_bits[segment]);
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const float2 f = __bfloat1622float2(hb[pair]);
        sq[segment * 8 + 2 * pair] = f.x * f.x;
        sq[segment * 8 + 2 * pair + 1] = f.y * f.y;
      }
    }
    sum = 0.0f;
#pragma unroll
    for (int rv = 0; rv < 24; ++rv)
      sum += sq[(rv % 3) * 8 + rv / 3];
    sum = reduce_group_128(sum, red, 0, lane);
    const float inv2 = rsqrtf(sum / 3072.0f + eps[0]);
    if (lane == 0) inv_out[row] = inv2;

    // The 128 exact norm lanes still own the rounded h vectors
    // used to form inv2.  Apply the second RMSNorm once here and publish exactly the
    // three live BF16 rows consumed by both the router and MoE.
    if constexpr (EmitXPad) {
#pragma unroll
      for (int segment = 0; segment < 3; ++segment) {
        const int base = segment * 1024 + lane * 8;
        const __nv_bfloat162* hb =
            reinterpret_cast<const __nv_bfloat162*>(&h_bits[segment]);
        U4 normalized;
        __nv_bfloat162* out =
            reinterpret_cast<__nv_bfloat162*>(&normalized);
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const float4 nw = *reinterpret_cast<const float4*>(
              w_ffn + base + half * 4);
          const float2 h0 = __bfloat1622float2(hb[half * 2]);
          const float2 h1 = __bfloat1622float2(hb[half * 2 + 1]);
          const float4 norm = make_float4(
              nw.x * (h0.x * inv2), nw.y * (h0.y * inv2),
              nw.z * (h1.x * inv2), nw.w * (h1.y * inv2));
          out[half * 2] = __float22bfloat162_rn(
              reinterpret_cast<const float2*>(&norm)[0]);
          out[half * 2 + 1] = __float22bfloat162_rn(
              reinterpret_cast<const float2*>(&norm)[1]);
        }
        *reinterpret_cast<U4*>(x_pad + row_base + base) = normalized;
        if (x_pad_fp8 != nullptr) {
          *reinterpret_cast<uint64_t*>(x_pad_fp8 + row_base + base) =
              ar_bf16x8_to_e4m3x8(normalized);
        }
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (!EarlyPDLTrigger)
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// The next prologue otherwise recomputes this RMSNorm independently in every QKV CTA.
// This variant leaves the same BF16 residual output visible to the model and also emits
// its exact attention-normalized value once, while the reduced MoE values are live.
template <bool OverlapLocal>
__global__ __launch_bounds__(DIRECT_NT, 1) void mlp_tail_norm_direct1bar_impl(
    const __nv_bfloat16* mc,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ scale,
    const float* __restrict__ w_norm,
    const float* __restrict__ eps,
    __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ norm_out,
    unsigned int** signal_pads,
    int rank, int world, int q) {
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= q) return;
  __shared__ __align__(16) __nv_bfloat16 h_s[H];
  __shared__ float red[NT];
  extern __shared__ __align__(16) unsigned char local_stage[];
  auto* residual_stage = reinterpret_cast<__nv_bfloat16*>(local_stage);
  auto* norm_stage = reinterpret_cast<float*>(residual_stage + H);

  const int base8 = tid * 8;
  const size_t row_base = static_cast<size_t>(row) * H;
  U4 residual_bits;

  sync_remote_blocks<false, true>(signal_pads, rank, world);
  if constexpr (OverlapLocal) {
    // Warp 0 owns the eight system-scope signal lanes above.  The other eleven warps
    // are independent and can cover its wait with rank-local loads.  Staging the whole
    // rows lets warp 0 consume its own vectors after the existing CTA barrier, without
    // delaying the first signal or changing the one-barrier protocol.
    if (tid >= 32) {
      constexpr int LOADERS = DIRECT_NT - 32;
      for (int pack = tid - 32; pack < H / 8; pack += LOADERS) {
        *reinterpret_cast<U4*>(residual_stage + pack * 8) =
            ld_global_u4(residual + row_base + pack * 8);
      }
      for (int pack = tid - 32; pack < H / 4; pack += LOADERS) {
        *reinterpret_cast<float4*>(norm_stage + pack * 4) =
            *reinterpret_cast<const float4*>(w_norm + pack * 4);
      }
    }
  }
  __syncthreads();

  const U4 reduced = mm_ld_reduce_bf16x8(mc + row_base + base8);
  const __nv_bfloat162* rb = reinterpret_cast<const __nv_bfloat162*>(&reduced);
  if constexpr (OverlapLocal) {
    residual_bits = *reinterpret_cast<const U4*>(residual_stage + base8);
  } else {
    residual_bits = *reinterpret_cast<const U4*>(residual + row_base + base8);
  }
  const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&residual_bits);
  U4 result;
  __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&result);
  const float s = scale[0];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const float2 m = __bfloat1622float2(rb[pair]);
    const __nv_bfloat162 scaled = __float22bfloat162_rn(
        make_float2(m.x * s, m.y * s));
    const float2 sf = __bfloat1622float2(scaled);
    const float2 h = __bfloat1622float2(hb[pair]);
    ob[pair] = __float22bfloat162_rn(make_float2(h.x + sf.x, h.y + sf.y));
  }
  *reinterpret_cast<U4*>(h_s + base8) = result;
  *reinterpret_cast<U4*>(out + row_base + base8) = result;
  __syncthreads();

  if (tid >= NT) return;
  const int lane = tid;
  float sq[24];
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    const U4 u = *reinterpret_cast<const U4*>(h_s + i * 1024 + lane * 8);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(b[j]);
      sq[i * 8 + 2 * j] = f.x * f.x;
      sq[i * 8 + 2 * j + 1] = f.y * f.y;
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int rv = 0; rv < 24; ++rv) {
    sum += sq[(rv % 3) * 8 + rv / 3];
  }
  sum = reduce_group_128(sum, red, 0, lane);
  const float inv = rsqrtf(sum / 3072.0f + eps[0]);

#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const int base = i * 512 + lane * 4;
    const float4 w = *reinterpret_cast<const float4*>(
        (OverlapLocal ? norm_stage : w_norm) + base);
    const uint2 hu = *reinterpret_cast<const uint2*>(h_s + base);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(&hu);
    float4 hf;
    reinterpret_cast<float2*>(&hf)[0] = __bfloat1622float2(h2[0]);
    reinterpret_cast<float2*>(&hf)[1] = __bfloat1622float2(h2[1]);
    const float4 norm = make_float4(
        w.x * (hf.x * inv), w.y * (hf.y * inv),
        w.z * (hf.z * inv), w.w * (hf.w * inv));
    uint2 nv;
    reinterpret_cast<__nv_bfloat162*>(&nv)[0] =
        __float22bfloat162_rn(reinterpret_cast<const float2*>(&norm)[0]);
    reinterpret_cast<__nv_bfloat162*>(&nv)[1] =
        __float22bfloat162_rn(reinterpret_cast<const float2*>(&norm)[1]);
    *reinterpret_cast<uint2*>(norm_out + row_base + base) = nv;
  }
}

void ar_tail_direct1bar_launch(
    int64_t mc, int64_t residual, int64_t w_ao, int64_t w_ffn,
    int64_t scale, int64_t eps, int64_t h_out, int64_t hn_out,
    int64_t hn32_out, int64_t flag_ptrs, int64_t rank, int64_t world,
    int64_t q) {
  TORCH_CHECK(mc != 0, "multicast pointer is null; NVLS is unavailable");
  TORCH_CHECK(flag_ptrs != 0, "symmetric-memory signal pointer table is null");
  TORCH_CHECK(world == 8, "ar_tail supports world=8, got ", world);
  TORCH_CHECK(rank >= 0 && rank < world, "invalid rank ", rank);
  TORCH_CHECK(q == 8, "ar_tail is Q8-only, got ", q);
  TORCH_CHECK((mc & 15) == 0, "multicast input must be 16-byte aligned");
  TORCH_CHECK((residual & 7) == 0, "residual must be 8-byte aligned");
  TORCH_CHECK((w_ao & 15) == 0 && (w_ffn & 15) == 0,
              "RMSNorm weights must be 16-byte aligned");
  TORCH_CHECK((h_out & 7) == 0 && (hn_out & 7) == 0,
              "BF16 outputs must be 8-byte aligned");
  TORCH_CHECK(hn32_out == 0 || (hn32_out & 15) == 0,
              "optional fp32 output must be 16-byte aligned");

  auto stream = at::cuda::getCurrentCUDAStream();
  ar_tail_direct1bar_impl<false><<<static_cast<int>(q), DIRECT_NT, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<const float*>(w_ao),
      reinterpret_cast<const float*>(w_ffn),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<const float*>(eps),
      reinterpret_cast<__nv_bfloat16*>(h_out),
      reinterpret_cast<__nv_bfloat16*>(hn_out),
      reinterpret_cast<float*>(hn32_out),
      reinterpret_cast<unsigned int**>(flag_ptrs),
      static_cast<int>(rank), static_cast<int>(world), static_cast<int>(q),
      nullptr, nullptr);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ar_tail_sandwich_h32_launch(
    int64_t mc, int64_t residual32, int64_t w_ao, int64_t w_ffn,
    int64_t scale, int64_t eps, int64_t h32_out, int64_t hn_out,
    int64_t flag_ptrs, int64_t rank, int64_t world, int64_t q) {
  TORCH_CHECK(mc != 0 && flag_ptrs != 0 && residual32 != 0 && h32_out != 0,
              "fp32-residual sandwich tail pointers must be non-null");
  TORCH_CHECK(world == 8 && rank >= 0 && rank < world,
              "fp32-residual sandwich tail requires a valid TP8 rank");
  TORCH_CHECK(q == 8, "fp32-residual sandwich tail is Q8-only, got ", q);
  TORCH_CHECK((mc | residual32 | w_ao | w_ffn | h32_out) % 16 == 0 && (hn_out & 7) == 0,
              "fp32-residual sandwich tail operands must be aligned");
  auto stream = at::cuda::getCurrentCUDAStream();
  ar_tail_direct1bar_impl<true><<<static_cast<int>(q), DIRECT_NT, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(mc), nullptr,
      reinterpret_cast<const float*>(w_ao),
      reinterpret_cast<const float*>(w_ffn),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<const float*>(eps),
      nullptr,
      reinterpret_cast<__nv_bfloat16*>(hn_out),
      nullptr,
      reinterpret_cast<unsigned int**>(flag_ptrs),
      static_cast<int>(rank), static_cast<int>(world), static_cast<int>(q),
      reinterpret_cast<const float*>(residual32),
      reinterpret_cast<float*>(h32_out));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void ar_tail_direct1bar_h_inv_xpad_fp8_pdl_overlap_local_launch(
    int64_t mc, int64_t residual, int64_t w_ao, int64_t w_ffn,
    int64_t scale, int64_t eps, int64_t h_out, int64_t inv_out,
    int64_t x_pad, int64_t x_pad_fp8, int64_t flag_ptrs,
    int64_t rank, int64_t world, int64_t q) {
  TORCH_CHECK(mc != 0 && inv_out != 0 && x_pad != 0 && x_pad_fp8 != 0
              && flag_ptrs != 0,
              "dual-output AR-tail pointers must be non-null");
  TORCH_CHECK(world == 8 && rank >= 0 && rank < world,
              "dual-output AR-tail requires a valid TP8 rank");
  TORCH_CHECK(q == 8, "dual-output AR-tail is Q8-only, got ", q);
  TORCH_CHECK((mc | residual | w_ao | w_ffn | h_out | x_pad | x_pad_fp8) % 16 == 0,
              "dual-output AR-tail vector operands must be 16-byte aligned");
  TORCH_CHECK((inv_out & 3) == 0,
              "dual-output AR-tail inv must be 4-byte aligned");

  auto stream = at::cuda::getCurrentCUDAStream();
  const auto kernel = ar_tail_direct1bar_h_inv<true, true, true, true>;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(q), 1, 1);
  config.blockDim = dim3(DIRECT_NT, 1, 1);
  config.dynamicSmemBytes = H * sizeof(__nv_bfloat16) + H * sizeof(float);
  config.stream = stream;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &config, kernel,
      reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<const float*>(w_ao),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<const float*>(eps),
      reinterpret_cast<__nv_bfloat16*>(h_out),
      reinterpret_cast<float*>(inv_out),
      reinterpret_cast<unsigned int**>(flag_ptrs),
      static_cast<int>(rank), static_cast<int>(world), static_cast<int>(q),
      reinterpret_cast<const float*>(w_ffn),
      reinterpret_cast<__nv_bfloat16*>(x_pad),
      reinterpret_cast<uint8_t*>(x_pad_fp8));
  TORCH_CHECK(err == cudaSuccess, "dual-output AR-tail launch: ",
              cudaGetErrorString(err));
}

void mlp_tail_norm_overlap_local_launch(
    int64_t mc, int64_t residual, int64_t scale, int64_t w_norm,
    int64_t eps, int64_t out, int64_t norm_out, int64_t flag_ptrs,
    int64_t rank, int64_t world, int64_t q) {
  TORCH_CHECK(mc != 0 && flag_ptrs != 0, "NVLS pointers must be non-null");
  TORCH_CHECK(world == 8 && rank >= 0 && rank < world,
              "MLP AR-tail local overlap requires a valid TP8 rank");
  // The overlap schedule is row-local: grid.x is Q, every CTA stages exactly
  // one 3072-element residual row plus the shared RMSNorm weight, and the
  // signal-pad index is already blockIdx.x.
  TORCH_CHECK(q == 8, "MLP AR-tail local overlap is Q8-only, got ", q);
  TORCH_CHECK((mc & 15) == 0 && (residual & 15) == 0 && (out & 15) == 0
              && (norm_out & 15) == 0 && (w_norm & 15) == 0,
              "MLP AR-tail local overlap vector operands must be 16-byte aligned");
  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr int dynamic_smem = H * (sizeof(__nv_bfloat16) + sizeof(float));
  mlp_tail_norm_direct1bar_impl<true>
      <<<static_cast<int>(q), DIRECT_NT, dynamic_smem, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<const float*>(w_norm),
      reinterpret_cast<const float*>(eps),
      reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<__nv_bfloat16*>(norm_out),
      reinterpret_cast<unsigned int**>(flag_ptrs),
      static_cast<int>(rank), static_cast<int>(world), static_cast<int>(q));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// The recursive draft's attention tail under fp32_residual_connection.  It is the
// ar_tail_direct1bar_h_inv<true, true, true, true> schedule -- PDL wait and early
// release, rank-local staging behind the peer wait, the first RMSNorm and its BF16
// rounding points -- except that h = residual + BF16(BF16(norm) * a_scale) stays fp32:
// it is written to h32_out, and the second RMSNorm's inverse and the Router/MoE inputs
// are computed from it without a BF16 rounding.
__global__ __launch_bounds__(DIRECT_NT, 1) void ar_tail_h32_inv_xpad_fp8(
    const __nv_bfloat16* mc,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ w_ao,
    const float* __restrict__ scale,
    const float* __restrict__ eps,
    float* __restrict__ h32_out,
    float* __restrict__ inv_out,
    unsigned int** signal_pads,
    int rank, int world, int q,
    const float* __restrict__ w_ffn,
    __nv_bfloat16* __restrict__ x_pad,
    uint8_t* __restrict__ x_pad_fp8) {
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= q) return;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  __shared__ __align__(16) __nv_bfloat16 ao_s[H];
  __shared__ float red[NT];
  extern __shared__ __align__(16) unsigned char draft_stage[];
  __nv_bfloat16* residual_rows = reinterpret_cast<__nv_bfloat16*>(draft_stage);
  float* w_ao_rows = reinterpret_cast<float*>(residual_rows + H);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  sync_remote_blocks<false, true>(signal_pads, rank, world);
  const size_t row_base = static_cast<size_t>(row) * H;
  if (tid >= 32) {
    constexpr int LOADERS = DIRECT_NT - 32;
    for (int pack = tid - 32; pack < H / 8; pack += LOADERS)
      *reinterpret_cast<U4*>(residual_rows + pack * 8) =
          ld_global_u4(residual + row_base + pack * 8);
    for (int pack = tid - 32; pack < H / 4; pack += LOADERS)
      *reinterpret_cast<float4*>(w_ao_rows + pack * 4) =
          *reinterpret_cast<const float4*>(w_ao + pack * 4);
  }
  __syncthreads();
  for (int i = tid; i < H / 8; i += DIRECT_NT)
    *reinterpret_cast<U4*>(ao_s + 8 * i) = mm_ld_reduce_bf16x8(mc + row_base + 8 * i);
  __syncthreads();
  if (tid >= NT) return;

  const int lane = tid;
  float sq[24];
#pragma unroll
  for (int segment = 0; segment < 3; ++segment) {
    const U4 u = *reinterpret_cast<const U4*>(ao_s + segment * 1024 + lane * 8);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 f = __bfloat1622float2(b[pair]);
      sq[segment * 8 + 2 * pair] = f.x * f.x;
      sq[segment * 8 + 2 * pair + 1] = f.y * f.y;
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int rv = 0; rv < 24; ++rv)
    sum += sq[(rv % 3) * 8 + rv / 3];
  sum = reduce_group_128(sum, red, 0, lane);
  const float inv = rsqrtf(sum / 3072.0f + eps[0]);

  const float s = scale[0];
  float h32[3][8];
#pragma unroll
  for (int segment = 0; segment < 3; ++segment) {
    const int base = segment * 1024 + lane * 8;
    const U4 au = *reinterpret_cast<const U4*>(ao_s + base);
    const U4 ru = *reinterpret_cast<const U4*>(residual_rows + base);
    const __nv_bfloat162* ab = reinterpret_cast<const __nv_bfloat162*>(&au);
    const __nv_bfloat162* rb = reinterpret_cast<const __nv_bfloat162*>(&ru);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 a = __bfloat1622float2(ab[pair]);
      const float2 w = *reinterpret_cast<const float2*>(w_ao_rows + base + 2 * pair);
      const __nv_bfloat162 rounded_norm =
          __float22bfloat162_rn(make_float2(w.x * (a.x * inv), w.y * (a.y * inv)));
      const float2 n = __bfloat1622float2(rounded_norm);
      const float2 sf = __bfloat1622float2(
          __float22bfloat162_rn(make_float2(n.x * s, n.y * s)));
      const float2 r = __bfloat1622float2(rb[pair]);
      h32[segment][2 * pair] = sf.x + r.x;
      h32[segment][2 * pair + 1] = sf.y + r.y;
    }
    float* out = h32_out + row_base + base;
    *reinterpret_cast<float4*>(out) =
        make_float4(h32[segment][0], h32[segment][1], h32[segment][2], h32[segment][3]);
    *reinterpret_cast<float4*>(out + 4) =
        make_float4(h32[segment][4], h32[segment][5], h32[segment][6], h32[segment][7]);
  }

#pragma unroll
  for (int segment = 0; segment < 3; ++segment)
#pragma unroll
    for (int k = 0; k < 8; ++k) sq[segment * 8 + k] = h32[segment][k] * h32[segment][k];
  sum = 0.0f;
#pragma unroll
  for (int rv = 0; rv < 24; ++rv)
    sum += sq[(rv % 3) * 8 + rv / 3];
  sum = reduce_group_128(sum, red, 0, lane);
  const float inv2 = rsqrtf(sum / 3072.0f + eps[0]);
  if (lane == 0) inv_out[row] = inv2;

#pragma unroll
  for (int segment = 0; segment < 3; ++segment) {
    const int base = segment * 1024 + lane * 8;
    U4 normalized;
    __nv_bfloat162* nb = reinterpret_cast<__nv_bfloat162*>(&normalized);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 w = *reinterpret_cast<const float2*>(w_ffn + base + 2 * pair);
      nb[pair] = __float22bfloat162_rn(make_float2(
          w.x * (h32[segment][2 * pair] * inv2),
          w.y * (h32[segment][2 * pair + 1] * inv2)));
    }
    *reinterpret_cast<U4*>(x_pad + row_base + base) = normalized;
    *reinterpret_cast<uint64_t*>(x_pad_fp8 + row_base + base) =
        ar_bf16x8_to_e4m3x8(normalized);
  }
}

void ar_tail_h32_inv_xpad_fp8_launch(
    int64_t mc, int64_t residual, int64_t w_ao, int64_t w_ffn,
    int64_t scale, int64_t eps, int64_t h32_out, int64_t inv_out,
    int64_t x_pad, int64_t x_pad_fp8, int64_t flag_ptrs,
    int64_t rank, int64_t world, int64_t q) {
  TORCH_CHECK(mc != 0 && h32_out != 0 && inv_out != 0 && x_pad != 0 &&
              x_pad_fp8 != 0 && flag_ptrs != 0,
              "fp32-residual attention tail pointers must be non-null");
  TORCH_CHECK(world == 8 && rank >= 0 && rank < world,
              "fp32-residual attention tail requires a valid TP8 rank");
  TORCH_CHECK(q == 8, "fp32-residual attention tail is Q8-only, got ", q);
  TORCH_CHECK((mc | residual | w_ao | w_ffn | h32_out | x_pad | x_pad_fp8) % 16 == 0,
              "fp32-residual attention tail vector operands must be 16-byte aligned");
  constexpr size_t stage_bytes = H * (sizeof(__nv_bfloat16) + sizeof(float));
  cudaLaunchConfig_t launch{};
  launch.gridDim = dim3(static_cast<unsigned>(q), 1, 1);
  launch.blockDim = dim3(DIRECT_NT, 1, 1);
  launch.dynamicSmemBytes = stage_bytes;
  launch.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute pdl{};
  pdl.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  pdl.val.programmaticStreamSerializationAllowed = 1;
  launch.attrs = &pdl;
  launch.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &launch, ar_tail_h32_inv_xpad_fp8,
      reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<const float*>(w_ao),
      reinterpret_cast<const float*>(scale),
      reinterpret_cast<const float*>(eps),
      reinterpret_cast<float*>(h32_out),
      reinterpret_cast<float*>(inv_out),
      reinterpret_cast<unsigned int**>(flag_ptrs),
      static_cast<int>(rank), static_cast<int>(world), static_cast<int>(q),
      reinterpret_cast<const float*>(w_ffn),
      reinterpret_cast<__nv_bfloat16*>(x_pad),
      reinterpret_cast<uint8_t*>(x_pad_fp8));
  TORCH_CHECK(err == cudaSuccess, "fp32-residual attention tail launch: ",
              cudaGetErrorString(err));
}

// MTP's row-parallel eh_proj immediately feeds the decoder layer's attention
// RMSNorm.  The producer is the SM100 TileLang GEMV with a PDL trigger; this
// consumer waits for that trigger, performs the normal one-barrier NVLS reduce,
// and materialises the exact 128-lane RMSNorm result for prenorm QKV.  Head 0 is
// sandwich-normalised and therefore also needs the raw reduced hidden as its
// residual base.  Head 1's residual base is the normalised tensor, so writing a
// second raw copy would be dead traffic.
template <bool NeedRaw>
__global__ __launch_bounds__(NT, 1) void mtp_eh_ar_norm_pdl(
    const __nv_bfloat16* mc,
    const float* __restrict__ w_norm,
    const float* __restrict__ eps,
    __nv_bfloat16* __restrict__ raw_out,
    __nv_bfloat16* __restrict__ norm_out,
    unsigned int** signal_pads,
    int rank, int world, int q) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x;
  if (row >= q) return;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  sync_remote_blocks<false, true>(signal_pads, rank, world);
  __syncthreads();

  const size_t row_base = static_cast<size_t>(row) * H;
  U4 reduced_bits[3];
  float sq[24];
#pragma unroll
  for (int segment = 0; segment < 3; ++segment) {
    const int base = segment * 1024 + lane * 8;
    const U4 reduced = mm_ld_reduce_bf16x8(mc + row_base + base);
    reduced_bits[segment] = reduced;
    if constexpr (NeedRaw)
      *reinterpret_cast<U4*>(raw_out + row_base + base) = reduced;
    const __nv_bfloat162* values =
        reinterpret_cast<const __nv_bfloat162*>(&reduced);
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const float2 v = __bfloat1622float2(values[pair]);
      sq[segment * 8 + pair * 2] = v.x * v.x;
      sq[segment * 8 + pair * 2 + 1] = v.y * v.y;
    }
  }

  float sum = 0.0f;
#pragma unroll
  for (int rv = 0; rv < 24; ++rv)
    sum += sq[(rv % 3) * 8 + rv / 3];
  __shared__ float red[NT];
  sum = reduce_group_128(sum, red, 0, lane);
  const float inv = rsqrtf(sum / 3072.0f + eps[0]);

#pragma unroll
  for (int segment = 0; segment < 3; ++segment) {
    const int base = segment * 1024 + lane * 8;
    const __nv_bfloat162* values =
        reinterpret_cast<const __nv_bfloat162*>(&reduced_bits[segment]);
    U4 norm_bits;
    __nv_bfloat162* out = reinterpret_cast<__nv_bfloat162*>(&norm_bits);
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const float4 w = *reinterpret_cast<const float4*>(
          w_norm + base + half * 4);
      const float2 h0 = __bfloat1622float2(values[half * 2]);
      const float2 h1 = __bfloat1622float2(values[half * 2 + 1]);
      const float4 n = make_float4(
          w.x * (h0.x * inv), w.y * (h0.y * inv),
          w.z * (h1.x * inv), w.w * (h1.y * inv));
      out[half * 2] =
          __float22bfloat162_rn(reinterpret_cast<const float2*>(&n)[0]);
      out[half * 2 + 1] =
          __float22bfloat162_rn(reinterpret_cast<const float2*>(&n)[1]);
    }
    *reinterpret_cast<U4*>(norm_out + row_base + base) = norm_bits;
  }
}

template <bool NeedRaw>
void launch_mtp_eh_ar_norm(
    int64_t mc, int64_t w_norm, int64_t eps, int64_t raw_out,
    int64_t norm_out, int64_t flag_ptrs, int64_t rank, int64_t world,
    int64_t q) {
  TORCH_CHECK(mc != 0 && w_norm != 0 && eps != 0 && norm_out != 0 &&
              flag_ptrs != 0, "MTP eh AR-norm NVLS pointers must be non-null");
  TORCH_CHECK(world == 8 && rank >= 0 && rank < world,
              "MTP eh AR-norm requires a valid TP8 rank");
  TORCH_CHECK(q == 8, "MTP eh AR-norm is Q8-only, got ", q);
  TORCH_CHECK(!NeedRaw || raw_out != 0,
              "MTP sandwich head must materialise raw reduced hidden");
  TORCH_CHECK((mc & 15) == 0 && (w_norm & 15) == 0 &&
              (norm_out & 15) == 0 && (!NeedRaw || (raw_out & 15) == 0),
              "MTP eh AR-norm vector operands must be 16-byte aligned");
  auto stream = at::cuda::getCurrentCUDAStream();
  auto kernel = mtp_eh_ar_norm_pdl<NeedRaw>;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(q), 1, 1);
  config.blockDim = dim3(NT, 1, 1);
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  config.attrs = &attr;
  config.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(
      &config, kernel,
      reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const float*>(w_norm),
      reinterpret_cast<const float*>(eps),
      reinterpret_cast<__nv_bfloat16*>(raw_out),
      reinterpret_cast<__nv_bfloat16*>(norm_out),
      reinterpret_cast<unsigned int**>(flag_ptrs),
      static_cast<int>(rank), static_cast<int>(world), static_cast<int>(q));
  TORCH_CHECK(err == cudaSuccess, "MTP eh AR-norm PDL launch: ",
              cudaGetErrorString(err));
}

void mtp_eh_ar_norm_head0_pdl_launch(
    int64_t mc, int64_t w_norm, int64_t eps, int64_t raw_out,
    int64_t norm_out, int64_t flag_ptrs, int64_t rank, int64_t world,
    int64_t q) {
  launch_mtp_eh_ar_norm<true>(mc, w_norm, eps, raw_out, norm_out,
                              flag_ptrs, rank, world, q);
}

"""

_EXT: Any | None = None
_EXT_FAILED: str | None = None


def _build():
    """Build the NVLS image."""
    with cuda_arch():
        return load_inline(
            name="iquestrt_multimem_ar_tail_v16_fp32_residual",
            cpp_sources=_CPP,
            cuda_sources=_CUDA,
            functions=["ar_tail_direct1bar_launch",
                       "ar_tail_direct1bar_h_inv_xpad_fp8_pdl_overlap_local_launch",
                       "mlp_tail_norm_overlap_local_launch",
                       "mtp_eh_ar_norm_head0_pdl_launch",
                       "ar_tail_sandwich_h32_launch",
                       "ar_tail_h32_inv_xpad_fp8_launch"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )


def _ext():
    """Compile on rank 0, then let the other ranks take the extension-cache hit.

    Build failures are first broadcast as data. A bare rank-0 exception would otherwise
    strand the other seven ranks forever in a barrier, hiding the compiler diagnostic
    behind what looks like an NVLS deadlock."""
    global _EXT, _EXT_FAILED
    if _EXT is not None:
        return _EXT
    if _EXT_FAILED is not None:
        raise RuntimeError(_EXT_FAILED)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        rank = dist.get_rank()
        root_error = None
        if rank == 0:
            try:
                _EXT = _build()
            except Exception as exc:  # noqa: BLE001 - broadcast before every rank raises
                root_error = f"{type(exc).__name__}: {exc}"
        root_status = [root_error]
        dist.broadcast_object_list(root_status, src=0)
        if root_status[0] is not None:
            _EXT = None
            _EXT_FAILED = (
                "rank 0 failed to build iquestrt_multimem_ar_tail: " + root_status[0]
            )
            raise RuntimeError(_EXT_FAILED)

        local_error = None
        if rank != 0:
            try:
                _EXT = _build()
            except Exception as exc:  # noqa: BLE001 - aggregate cache-load failures
                local_error = f"rank {rank}: {type(exc).__name__}: {exc}"
        ok = torch.tensor(
            int(local_error is None), dtype=torch.int32,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not int(ok.item()):
            errors: list[str | None] = [None] * dist.get_world_size()
            dist.all_gather_object(errors, local_error)
            detail = "; ".join(error for error in errors if error is not None)
            _EXT = None
            _EXT_FAILED = (
                "a nonzero rank failed to load iquestrt_multimem_ar_tail: " + detail
            )
            raise RuntimeError(_EXT_FAILED)
    else:
        _EXT = _build()
    return _EXT


def _check_cuda_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    numel: int | None = None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if numel is not None and tensor.numel() != numel:
        raise ValueError(f"{name} must contain {numel} elements, got {tensor.numel()}")


def _raise_if_any_rank_failed(
    local_error: str | None,
    *,
    device: torch.device,
    what: str,
) -> None:
    """Turn rank-local startup failures into one collective failure on every rank."""
    ok = torch.tensor(int(local_error is None), dtype=torch.int32, device=device)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if int(ok.item()):
        return
    errors: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    detail = "; ".join(error for error in errors if error is not None)
    raise RuntimeError(f"MultimemARTail {what} failed across TP ranks: {detail}")


class MultimemARTail:
    """One-barrier all-reduce + residual + two exact RMSNorms."""

    def __init__(
        self,
        ao_buffers: Sequence[torch.Tensor],
        *,
        group: str,
        tail: TailKernels,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("MultimemARTail needs an initialised process group")
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        validated = None
        local_error = None
        try:
            if self.world != WORLD:
                raise ValueError(f"supports world=8, got {self.world}")
            if not isinstance(group, str) or not group:
                raise TypeError("group must be the non-empty symmetric-memory group name")
            if not isinstance(tail, TailKernels):
                raise TypeError("tail must be a TailKernels instance")
            if tail.hidden != HIDDEN:
                raise ValueError(f"supports hidden=3072, got {tail.hidden}")
            if tail.q_len != Q:
                raise ValueError(f"supports q_len={Q}, got {tail.q_len}")
            if len(ao_buffers) != 2:
                raise ValueError(
                    f"one-barrier AR needs exactly two alternating AO buffers, got "
                    f"{len(ao_buffers)}"
                )

            buffers = tuple(ao_buffers)
            if not all(isinstance(buf, torch.Tensor) for buf in buffers):
                raise TypeError("both AO buffers must be torch.Tensor instances")
            device = buffers[0].device
            if device.type != "cuda":
                raise ValueError(f"AO buffers must be CUDA tensors, got {device}")
            cap = torch.cuda.get_device_capability(device)
            if cap != (10, 0):
                raise ValueError(f"verified only on compute capability 10.0, got {cap[0]}.{cap[1]}")

            q = tail.q_len
            for index, buf in enumerate(buffers):
                _check_cuda_tensor(
                    f"ao_buffers[{index}]", buf, dtype=torch.bfloat16, device=device
                )
                if buf.ndim != 2 or buf.shape[1] != HIDDEN or buf.shape[0] < q:
                    raise ValueError(
                        f"ao_buffers[{index}] must have shape [rows>=Q, 3072] with "
                        f"Q={q}, got {tuple(buf.shape)}"
                    )
                if (buf.storage_offset() != 0
                        or buf.data_ptr() != buf.untyped_storage().data_ptr()):
                    raise ValueError(
                        f"ao_buffers[{index}] must be the complete symmetric allocation, "
                        "not an offset view"
                    )
            if buffers[0].shape != buffers[1].shape:
                raise ValueError(
                    "the two AO allocations must have identical shapes, got "
                    f"{tuple(buffers[0].shape)} and {tuple(buffers[1].shape)}"
                )
            storage_ptrs = {buf.untyped_storage().data_ptr() for buf in buffers}
            if len(storage_ptrs) != 2:
                raise ValueError(
                    "one-barrier AR needs two distinct symmetric allocations; the "
                    "supplied AO buffers alias the same storage"
                )

            for name, out, dtype in (
                ("tail.h", tail.h, torch.bfloat16),
                ("tail.hn", tail.hn, torch.bfloat16),
            ):
                _check_cuda_tensor(
                    name, out, dtype=dtype, device=device, numel=q * HIDDEN
                )
            _check_cuda_tensor(
                "tail.a_scale", tail.a_scale, dtype=torch.float32,
                device=device, numel=1,
            )
            _check_cuda_tensor(
                "tail.eps", tail.eps, dtype=torch.float32, device=device, numel=1
            )
            validated = buffers, device, q
        except Exception as exc:  # noqa: BLE001 - aggregate before any rendezvous
            local_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"

        flag_device = (validated[1] if validated is not None else
                       torch.device("cuda", torch.cuda.current_device()))
        _raise_if_any_rank_failed(
            local_error, device=flag_device, what="preflight"
        )
        if validated is None:  # all-rank failure above always raises; helps type checkers
            raise AssertionError("unreachable: successful TP preflight has no values")
        buffers, device, q = validated

        import torch.distributed._symmetric_memory as symm_mem

        handles = []
        pointers: dict[int, tuple[int, int]] = {}
        for index, buf in enumerate(buffers):
            handle = None
            handle_error = None
            multicast_ptr = signal_ptrs = 0
            try:
                handle = symm_mem.rendezvous(buf, group)
                multicast_ptr = int(handle.multicast_ptr)
                signal_ptrs = int(handle.signal_pad_ptrs_dev)
                if not multicast_ptr:
                    raise RuntimeError("no multicast pointer; NVLS is unavailable")
                if not signal_ptrs:
                    raise RuntimeError("no symmetric-memory signal pointer table")
            except Exception as exc:  # noqa: BLE001 - agree before next rendezvous
                handle_error = (
                    f"rank {self.rank} buffer {index}: {type(exc).__name__}: {exc}"
                )
            _raise_if_any_rank_failed(
                handle_error, device=device, what=f"buffer {index} rendezvous"
            )
            if handle is None:  # collective validation above always raises on failure
                raise AssertionError("unreachable: rendezvous succeeded without a handle")
            handles.append(handle)
            pointers[buf.data_ptr()] = (multicast_ptr, signal_ptrs)
        alias_error = (
            f"rank {self.rank}: both AO buffers have multicast_ptr "
            f"{int(handles[0].multicast_ptr):#x}"
            if handles[0].multicast_ptr == handles[1].multicast_ptr else None
        )
        _raise_if_any_rank_failed(
            alias_error, device=device, what="multicast alias check"
        )

        # Keep tensors and handles alive for every captured graph that embeds their
        # multicast and signal-table addresses.
        self.group = group
        self.tail = tail
        self.q_len = q
        self.device = device
        self.ao_buffers = buffers
        self._handles = tuple(handles)
        self._pointers = pointers
        self._extension = _ext()
        # The second RMSNorm's inverse per row, published by the MoE-layer tail.
        self.inv = torch.empty(q, dtype=torch.float32, device=device)

    def __call__(
        self,
        ao: torch.Tensor,
        residual: torch.Tensor,
        w_ao: torch.Tensor,
        w_ffn: torch.Tensor,
        a_scale: float,
        f_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce ``ao`` and write/return ``(tail.h, tail.hn)``.

        An fp32 ``residual`` (the recursive draft's fp32 residual stream) keeps the sum
        fp32: ``(tail.out32, tail.hn)``, with the second RMSNorm of the unrounded sum.

        ``f_scale`` is not consumed by this kernel, but is passed to ``TailKernels._prep``
        so a later ``tail_post`` reads the layer's scale."""
        q = self.q_len
        _check_cuda_tensor(
            "ao", ao, dtype=torch.bfloat16, device=self.device, numel=q * HIDDEN
        )
        try:
            multicast_ptr, signal_ptrs = self._pointers[ao.data_ptr()]
        except KeyError as exc:
            raise ValueError(
                "ao must be the [:Q] base view of one of the two registered symmetric "
                "AO buffers"
            ) from exc

        fp32 = residual.dtype == torch.float32
        _check_cuda_tensor(
            "residual",
            residual,
            dtype=torch.float32 if fp32 else torch.bfloat16,
            device=self.device,
            numel=q * HIDDEN,
        )
        _check_cuda_tensor(
            "w_ao", w_ao, dtype=torch.float32, device=self.device, numel=HIDDEN
        )
        _check_cuda_tensor(
            "w_ffn", w_ffn, dtype=torch.float32, device=self.device, numel=HIDDEN
        )

        self.tail._prep(a_scale, f_scale)
        if fp32:
            self._extension.ar_tail_sandwich_h32_launch(
                multicast_ptr, residual.data_ptr(), w_ao.data_ptr(), w_ffn.data_ptr(),
                self.tail.a_scale.data_ptr(), self.tail.eps.data_ptr(),
                self.tail.out32.data_ptr(), self.tail.hn.data_ptr(), signal_ptrs,
                self.rank, self.world, q,
            )
            return self.tail.out32.view(q, HIDDEN), self.tail.hn.view(q, HIDDEN)
        self._extension.ar_tail_direct1bar_launch(
            multicast_ptr,
            residual.data_ptr(),
            w_ao.data_ptr(),
            w_ffn.data_ptr(),
            self.tail.a_scale.data_ptr(),
            self.tail.eps.data_ptr(),
            self.tail.h.data_ptr(),
            self.tail.hn.data_ptr(),
            0,
            signal_ptrs,
            self.rank,
            self.world,
            q,
        )
        return self.tail.h.view(q, HIDDEN), self.tail.hn.view(q, HIDDEN)

    def h_inv_xpad_fp8_pdl_overlap_local(
        self,
        ao: torch.Tensor,
        residual: torch.Tensor,
        w_ao: torch.Tensor,
        w_ffn: torch.Tensor,
        inv_out: torch.Tensor,
        x_pad: torch.Tensor,
        x_pad_fp8: torch.Tensor,
        a_scale: float,
        f_scale: float,
        *,
        fp32_residual: bool = False,
    ) -> torch.Tensor:
        """The MoE layer's attention tail: publish exact BF16 and E4M3 Router/MoE inputs."""
        q = self.q_len
        _check_cuda_tensor(
            "ao", ao, dtype=torch.bfloat16, device=self.device, numel=q * HIDDEN
        )
        try:
            multicast_ptr, signal_ptrs = self._pointers[ao.data_ptr()]
        except KeyError as exc:
            raise ValueError(
                "ao must be the [:Q] base view of a registered symmetric AO buffer"
            ) from exc
        _check_cuda_tensor(
            "residual", residual, dtype=torch.bfloat16,
            device=self.device, numel=q * HIDDEN,
        )
        for name, weight in (("w_ao", w_ao), ("w_ffn", w_ffn)):
            _check_cuda_tensor(
                name, weight, dtype=torch.float32,
                device=self.device, numel=HIDDEN,
            )
        _check_cuda_tensor(
            "inv_out", inv_out, dtype=torch.float32,
            device=self.device, numel=q,
        )
        _check_cuda_tensor("x_pad", x_pad, dtype=torch.bfloat16, device=self.device)
        _check_cuda_tensor(
            "x_pad_fp8", x_pad_fp8, dtype=torch.float8_e4m3fn,
            device=self.device,
        )
        expected = (16, HIDDEN)
        if tuple(x_pad.shape) != expected or tuple(x_pad_fp8.shape) != expected:
            raise ValueError(
                "dual-output AR-tail needs BF16/FP8 [16,3072] siblings; got "
                f"{tuple(x_pad.shape)} and {tuple(x_pad_fp8.shape)}"
            )
        self.tail._prep(a_scale, f_scale)
        if fp32_residual:
            self._extension.ar_tail_h32_inv_xpad_fp8_launch(
                multicast_ptr, residual.data_ptr(), w_ao.data_ptr(), w_ffn.data_ptr(),
                self.tail.a_scale.data_ptr(), self.tail.eps.data_ptr(),
                self.tail.h32.data_ptr(), inv_out.data_ptr(), x_pad.data_ptr(),
                x_pad_fp8.data_ptr(), signal_ptrs, self.rank, self.world, q,
            )
            return self.tail.h32.view(q, HIDDEN)
        self._extension.ar_tail_direct1bar_h_inv_xpad_fp8_pdl_overlap_local_launch(
            multicast_ptr,
            residual.data_ptr(),
            w_ao.data_ptr(),
            w_ffn.data_ptr(),
            self.tail.a_scale.data_ptr(),
            self.tail.eps.data_ptr(),
            self.tail.h.data_ptr(),
            inv_out.data_ptr(),
            x_pad.data_ptr(),
            x_pad_fp8.data_ptr(),
            signal_ptrs,
            self.rank,
            self.world,
            q,
        )
        return self.tail.h.view(q, HIDDEN)


class MultimemMLPTail(MultimemARTail):
    """One-barrier MoE all-reduce + non-sandwich ``tail_post`` + the next attention norm."""

    def __init__(
        self,
        mlp_buffers: Sequence[torch.Tensor],
        *,
        group: str,
        tail: TailKernels,
        norm_out: torch.Tensor,
    ) -> None:
        super().__init__(mlp_buffers, group=group, tail=tail)
        _check_cuda_tensor(
            "norm_out",
            norm_out,
            dtype=torch.bfloat16,
            device=self.device,
            numel=self.q_len * HIDDEN,
        )
        self.norm_out = norm_out

    def __call__(
        self,
        mlp_out: torch.Tensor,
        residual: torch.Tensor,
        f_scale: float,
        w_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_len
        _check_cuda_tensor(
            "mlp_out",
            mlp_out,
            dtype=torch.bfloat16,
            device=self.device,
            numel=q * HIDDEN,
        )
        try:
            multicast_ptr, signal_ptrs = self._pointers[mlp_out.data_ptr()]
        except KeyError as exc:
            raise ValueError(
                "mlp_out must be the [:Q] base view of one of the two registered "
                "symmetric MoE output buffers"
            ) from exc
        _check_cuda_tensor(
            "residual",
            residual,
            dtype=torch.bfloat16,
            device=self.device,
            numel=q * HIDDEN,
        )

        a_scale = self.tail._scales[0]
        self.tail._prep(1.0 if a_scale is None else a_scale, f_scale)
        _check_cuda_tensor(
            "w_norm", w_norm, dtype=torch.float32, device=self.device, numel=HIDDEN
        )
        self._extension.mlp_tail_norm_overlap_local_launch(
            multicast_ptr,
            residual.data_ptr(),
            self.tail.f_scale.data_ptr(),
            w_norm.data_ptr(),
            self.tail.eps.data_ptr(),
            self.tail.out.data_ptr(),
            self.norm_out.data_ptr(),
            signal_ptrs,
            self.rank,
            self.world,
            q,
        )
        return self.tail.out.view(q, HIDDEN), self.norm_out.view(q, HIDDEN)


class MtpEhARNorm:
    """SM100 MTP ``eh_proj -> NVLS AR -> attention RMSNorm`` production seam."""

    def __init__(
        self,
        buffers: Sequence[torch.Tensor],
        *,
        group: str,
        norm_out: torch.Tensor,
        eps: float,
        q_len: int,
        ws,
    ) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("MtpEhARNorm needs an initialised process group")
        self.world = dist.get_world_size()
        self.rank = dist.get_rank()
        if self.world != WORLD:
            raise ValueError(f"MtpEhARNorm supports world=8, got {self.world}")
        if len(buffers) != 2:
            raise ValueError(
                f"MtpEhARNorm needs two head-specific buffers, got {len(buffers)}"
            )
        self.buffers = tuple(buffers)
        self.q_len = int(q_len)
        if self.q_len != Q:
            raise ValueError(f"MtpEhARNorm supports Q={Q}, got {self.q_len}")
        device = self.buffers[0].device
        if tuple(torch.cuda.get_device_capability(device)) != (10, 0):
            raise ValueError("MtpEhARNorm is verified only on compute capability 10.0")
        storage_ptrs = set()
        for index, buffer in enumerate(self.buffers):
            _check_cuda_tensor(
                f"buffers[{index}]", buffer, dtype=torch.bfloat16, device=device
            )
            if (buffer.ndim != 2 or buffer.shape[0] < self.q_len
                    or buffer.shape[1] != HIDDEN):
                raise ValueError(
                    f"buffers[{index}] must have shape [rows>={self.q_len},3072], got "
                    f"{tuple(buffer.shape)}"
                )
            if (buffer.storage_offset() != 0
                    or buffer.data_ptr() != buffer.untyped_storage().data_ptr()):
                raise ValueError(
                    f"buffers[{index}] must be a complete symmetric allocation"
                )
            storage_ptrs.add(buffer.untyped_storage().data_ptr())
        if len(storage_ptrs) != 2:
            raise ValueError("MtpEhARNorm head buffers must not alias")
        _check_cuda_tensor(
            "norm_out", norm_out, dtype=torch.bfloat16, device=device,
            numel=self.q_len * HIDDEN,
        )

        import torch.distributed._symmetric_memory as symm_mem

        handles = []
        pointers: dict[int, tuple[int, int]] = {}
        for index, buffer in enumerate(self.buffers):
            handle = None
            local_error = None
            try:
                handle = symm_mem.rendezvous(buffer, group)
                mc = int(handle.multicast_ptr)
                signals = int(handle.signal_pad_ptrs_dev)
                if not mc or not signals:
                    raise RuntimeError("NVLS multicast/signal pointer unavailable")
            except Exception as exc:  # noqa: BLE001 - aggregate before next rendezvous
                local_error = (
                    f"rank {self.rank} head {index}: {type(exc).__name__}: {exc}"
                )
                mc = signals = 0
            _raise_if_any_rank_failed(
                local_error, device=device, what=f"MTP eh head {index} rendezvous"
            )
            if handle is None:
                raise AssertionError("unreachable: rendezvous succeeded without handle")
            handles.append(handle)
            pointers[buffer.data_ptr()] = (mc, signals)

        self.raw_h0 = ws.slot(("mtp_eh_ar", "raw_h0"), (self.q_len, HIDDEN),
                              torch.bfloat16)
        self.eps = ws.slot(("mtp_eh_ar", "eps"), (1,), torch.float32)
        self.eps.fill_(float(eps))
        self.norm_out = norm_out
        self.group = group
        self.device = device
        self._handles = tuple(handles)
        self._pointers = pointers
        self._extension = _ext()

        from .oproj import EH_PROJ_CFG, build_oproj_gemv_q8_transposed

        # The BF16 eh_proj owns 768 of its 6144 input columns per rank at TP8.
        self.producer, _ = build_oproj_gemv_q8_transposed(
            768, HIDDEN, **EH_PROJ_CFG, pdl_trigger=True)

    def __call__(
        self,
        eh_in: torch.Tensor,
        w_eh: torch.Tensor,
        w_norm: torch.Tensor,
        *,
        head: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if head < 0:
            raise ValueError(f"MtpEhARNorm head must be non-negative, got {head}")
        _check_cuda_tensor(
            "eh_in", eh_in, dtype=torch.bfloat16, device=self.device,
            numel=self.q_len * 768,
        )
        _check_cuda_tensor(
            "w_eh", w_eh, dtype=torch.bfloat16, device=self.device,
            numel=HIDDEN * 768,
        )
        _check_cuda_tensor(
            "w_norm", w_norm, dtype=torch.float32, device=self.device,
            numel=HIDDEN,
        )
        buffer = self.buffers[head % len(self.buffers)]
        multicast_ptr, signal_ptrs = self._pointers[buffer.data_ptr()]
        self.producer(eh_in.reshape(self.q_len, 768), w_eh, buffer)
        self._extension.mtp_eh_ar_norm_head0_pdl_launch(
            multicast_ptr, w_norm.data_ptr(), self.eps.data_ptr(),
            self.raw_h0.data_ptr(), self.norm_out.data_ptr(), signal_ptrs,
            self.rank, self.world, self.q_len,
        )
        return (self.raw_h0.view(self.q_len, HIDDEN),
                self.norm_out.view(self.q_len, HIDDEN))


__all__ = ["MultimemARTail", "MultimemMLPTail", "MtpEhARNorm"]
