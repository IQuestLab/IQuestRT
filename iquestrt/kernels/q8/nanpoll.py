"""NaN-polled NVLink reductions and fused tails.

A producer replaces armed NaNs; a consumer accepts only a NaN-free multicast read
and re-arms the preceding buffer. Three attention and two MLP buffers rotate;
draft depths own separate buffers. READ_LIMIT traps persistent NaNs. Preserve
producer release order and buffer ownership across capture and replay."""

import functools
import re
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load_inline

from .. import ar_tail, router, sm100_moe_fp8
from ..arch import cuda_arch
from ..oproj import (
    DENSE_BLOCK_CFG,
    EH_PROJ_CFG,
    OPROJ_BLOCK_CFG,
    build_gemv_block,
    build_oproj_gemv_q8_transposed,
)
from ..sm100_moe_fp8 import DownBlock, build_gate_silu_block
from ..tail import TailKernels
from . import ar_prefetch, oproj_middle
from .codegen import replace_once

ROWS = 16              # a buffer is one producer output, [BM, H]
ARMED = 0x7FC0         # BF16 qNaN as int16: an element its producer has not stored yet
READ_LIMIT = 1 << 22   # switch reads of one 16-byte vector before the tail traps
SECOND_NS = 1000       # the attention tail's second read, after its first
MLP_SECOND_NS = 1000   # the MLP tail's second read, after entry
ATTENTION_BUFFERS, MLP_BUFFERS = 3, 2  # rotating over main layers 1..; layer 0 has its own
FOLD_ARMED = 0x7FC00000  # fp32 qNaN as int32: a fold slot element not stored yet

H, WORLD, Q = ar_tail.HIDDEN, ar_tail.WORLD, ar_tail.Q
TOPK = router.TOPK
FOLD_ROWS = oproj_middle.FOLD_ROWS     # a fold slot row: A, then B
FOLD_READERS = FOLD_ROWS // 4          # the attention tail's fp32 readers, a vector each


def _buffer(layer: int, count: int) -> int:
    if layer < 0:
        raise ValueError(f"main layers are numbered from 0, got layer {layer}")
    return count if layer == 0 else (layer - 1) % count


def attention_buffer(layer: int) -> int:
    """The buffer main layer `layer` writes its OProj partial into: layer 0's own, then
    three in turn."""
    return _buffer(layer, ATTENTION_BUFFERS)


def fold_buffer(layer: int) -> int:
    """The fold slot MoE layer `layer` writes its router partials into: three in turn."""
    if layer < 1:
        raise ValueError(f"the fold slots serve MoE layers 1.., got layer {layer}")
    return (layer - 1) % ATTENTION_BUFFERS


def mlp_buffer(layer: int) -> int:
    """The buffer main layer `layer` writes its MLP output into: layer 0's own, then two
    in turn."""
    return _buffer(layer, MLP_BUFFERS)


def check_attention_count(n: int) -> None:
    """Refuse `n` rotating attention layers a step unless no three attention calls in a
    row share a buffer around layer 0's."""
    if n % ATTENTION_BUFFERS:
        raise ValueError(f"{n} rotating attention layers a step would repeat a buffer "
                         f"around layer 0's: {ATTENTION_BUFFERS} buffers need a multiple "
                         f"of {ATTENTION_BUFFERS}")


def draft_rows(depths: int) -> tuple[int,...]:
    """The rows each draft depth's MoE writes: row 0, the one row read past its attention.
    Depth 0 runs its seed row first (`position.seed_first_logical`)."""
    return (1,) * depths


class DepthTurns:
    """Which buffer a draft tail of one kind re-arms: the one its previous call used.

    It starts as after a step, whose last call is the last depth's."""

    def __init__(self, depths: int):
        if depths < 2:
            raise ValueError(f"a buffer per depth needs at least two depths, got {depths}")
        self.depths = depths
        self.reset()

    def reset(self, last: int | None = None) -> None:
        """As after a step whose last depth was `last` (default the last of `depths`)."""
        last = self.depths - 1 if last is None else last
        if not isinstance(last, int) or not 0 <= last < self.depths:
            raise ValueError(f"draft depth must be in 0..{self.depths - 1}, got {last}")
        self.previous = last

    def check(self, depth) -> None:
        """Refuse a depth out of range, or the depth of the previous call: its buffer is
        re-armed only by the next call of the kind."""
        if not isinstance(depth, int) or not 0 <= depth < self.depths:
            raise ValueError(f"draft depth must be in 0..{self.depths - 1}, got {depth}")
        if depth == self.previous:
            raise ValueError(f"two calls in a row at draft depth {depth} would read its "
                             "buffer before the next call re-arms it")

    def take(self, depth: int) -> int:
        """This call's turn at `depth` -> the depth whose buffer it re-arms."""
        self.check(depth)
        previous, self.previous = self.previous, depth
        return previous


_DOWN_WAIT = '  asm volatile("griddepcontrol.wait;\\n" ::: "memory");\n'
_DOWN_RELEASE = (
    "  // Release the NaN-polled tail: it polls its only peer input (nanpoll.py).\n"
    '  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n')


@functools.lru_cache(None)
def releasing_down_source() -> str:
    """The shipping Down's source with a release right after its PDL wait."""
    return replace_once(sm100_moe_fp8.DOWN_CUDA, _DOWN_WAIT, _DOWN_WAIT + _DOWN_RELEASE)


@functools.lru_cache(None)
def releasing_down_extension():
    with cuda_arch():
        return load_inline(name="iquestrt_nanpoll_down_v1",
                           cpp_sources=sm100_moe_fp8._DOWN_CPP,
                           cuda_sources=releasing_down_source(), functions=["moe_down_launch"],
                           extra_cuda_cflags=["-O3"], verbose=False)


class ReleasingDown(DownBlock):
    """The Down over the first `rows` rows, bit for bit, releasing after its PDL wait."""

    def __init__(self, rows: int):
        if not 1 <= rows <= Q:
            raise ValueError(f"the block Down covers 1..{Q} rows, got {rows}")
        self.rows = rows
        self._ext = releasing_down_extension()


_BARRIER = "  sync_remote_blocks<false, true>(signal_pads, rank, world);\n"
# The draft tails' PDL waits, which their NaN-polled forms drop.
_PDL_WAIT = '  asm volatile("griddepcontrol.wait;" ::: "memory");\n'
_HELPERS = f"""
// ---------------------------------------------------------------- NaN-polled tails
constexpr unsigned NANPOLL_READ_LIMIT = {READ_LIMIT}u;
constexpr unsigned NANPOLL_ARMED = 0x{ARMED:04X}{ARMED:04X}u;  // two BF16 qNaN
""" + r"""
__device__ __forceinline__ U4 nanpoll_ld_acquire(const __nv_bfloat16* mc) {
  U4 v;
  asm volatile(
      "multimem.ld_reduce.acquire.sys.global.add.acc::f32.v4.bf16x2 {%0,%1,%2,%3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(mc) : "memory");
  return v;
}
__device__ __forceinline__ bool nanpoll_nan2(unsigned w) {
  return ((w & 0x7FFFu) > 0x7F80u) | (((w >> 16) & 0x7FFFu) > 0x7F80u);
}
__device__ __forceinline__ bool nanpoll_any_nan(const U4& v) {
  return nanpoll_nan2(v.x) | nanpoll_nan2(v.y) | nanpoll_nan2(v.z) | nanpoll_nan2(v.w);
}
// A thread's sum of squares of the row `h` in the TileLang RMSNorm order: six 4-wide chunks at
// a 512 stride, chunk-minor, each square rounded as TileLang's `mul2` rounds it.
__device__ __forceinline__ float nanpoll_final_norm_sum(const __nv_bfloat16* h, int lane) {
  float sq[24];
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const uint2 u = *reinterpret_cast<const uint2*>(h + i * 512 + lane * 4);
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&u);
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const float2 f = __bfloat1622float2(b[j]);
      sq[i * 4 + 2 * j] = __fmul_rn(f.x, f.x);
      sq[i * 4 + 2 * j + 1] = __fmul_rn(f.y, f.y);
    }
  }
  float sum = 0.0f;
#pragma unroll
  for (int rv = 0; rv < 24; ++rv) sum += sq[(rv % 6) * 4 + rv / 6];
  return sum;
}
"""
# The attention tail's fold: its fp32 read, and the
# routing of a row from the slot staged in shared memory and the norm's two scales.
_FOLD_HELPERS = router.SELECT + f"""
// ---------------------------------------------------------------- the router fold
constexpr int FOLD_NT = DIRECT_NT + {FOLD_READERS};  // the row's BF16 vectors, then the slot's
constexpr int FOLD_ROWS = {FOLD_ROWS}, FOLD_E = {FOLD_ROWS // 2};  // a slot row: A, then B
constexpr unsigned NANPOLL_FOLD_ARMED = 0x{FOLD_ARMED:08X}u;  // fp32 qNaN
""" + r"""
__device__ __forceinline__ float4 nanpoll_ld_acquire_f32(const float* mc) {
  float4 v;
  asm volatile(
      "multimem.ld_reduce.acquire.sys.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"
      : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "l"(mc) : "memory");
  return v;
}
__device__ __forceinline__ bool nanpoll_any_nan_f32(const float4& v) {
  return isnan(v.x) | isnan(v.y) | isnan(v.z) | isnan(v.w);
}
// Row `row`'s routing, one warp: lane l forms L = s(h) (B + a s(ao) A) of experts
// 4l..4l+3 and 128+4l..128+4l+3 from the row's slot `fold` and `scales` = {s(ao), s(h)},
// and selects as the Router does. The two halves below route through it on a tie.
__device__ __forceinline__ void nanpoll_fold_select(
    const float* fold, const float* scales, float scale, int row,
    int* __restrict__ sel_out, float* __restrict__ wt_out) {
  const int lane = threadIdx.x & 31;
  const float c = scale * scales[0], inv2 = scales[1];
  float v[8];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const float4 a = *reinterpret_cast<const float4*>(fold + 128 * half + 4 * lane);
    const float4 b = *reinterpret_cast<const float4*>(fold + FOLD_E + 128 * half + 4 * lane);
    v[4 * half] = inv2 * fmaf(c, a.x, b.x);
    v[4 * half + 1] = inv2 * fmaf(c, a.y, b.y);
    v[4 * half + 2] = inv2 * fmaf(c, a.z, b.z);
    v[4 * half + 3] = inv2 * fmaf(c, a.w, b.w);
  }
  iquestrt_select::select_row(v, row, sel_out, wt_out, true);
}

// The same routing in two halves. Once s(ao) is
// known: lane l's ordered keys of u = B + a s(ao) A, `c` = a s(ao) as above, ranked
// (`iquestrt_select::rank_keys`, the ids through `ids_s`).
struct NanpollFoldRank {
  unsigned top[iquestrt_select::kTopK], ninth;
  int id, rank;
  bool tied;
};

__device__ __forceinline__ NanpollFoldRank nanpoll_fold_rank(const float* fold, float c,
                                                             int* ids_s) {
  const int lane = threadIdx.x & 31;
  unsigned k[8];
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const float4 a = *reinterpret_cast<const float4*>(fold + 128 * half + 4 * lane);
    const float4 b = *reinterpret_cast<const float4*>(fold + FOLD_E + 128 * half + 4 * lane);
    k[4 * half] = iquestrt_select::ordered_key(fmaf(c, a.x, b.x));
    k[4 * half + 1] = iquestrt_select::ordered_key(fmaf(c, a.y, b.y));
    k[4 * half + 2] = iquestrt_select::ordered_key(fmaf(c, a.z, b.z));
    k[4 * half + 3] = iquestrt_select::ordered_key(fmaf(c, a.w, b.w));
  }
  NanpollFoldRank r;
  iquestrt_select::rank_keys(k, ids_s, r.top, r.ninth, r.id, r.rank, r.tied);
  return r;
}

// Once s(h) is known too: the softmax of the eight s(h) u in u's order, or
// `nanpoll_fold_select` where s(h) rounds the 8th and the 9th u to one logit or a round
// of the ranking tied two lanes.
__device__ __forceinline__ void nanpoll_fold_finish(
    const NanpollFoldRank& r, const float* fold, const float* scales, float scale, int row,
    int* __restrict__ sel_out, float* __restrict__ wt_out) {
  using iquestrt_select::key_value;
  using iquestrt_select::kTopK;
  using iquestrt_select::ordered_key;
  const float inv2 = scales[1];
  float value[kTopK];
#pragma unroll
  for (int t = 0; t < kTopK; ++t) value[t] = inv2 * key_value(r.top[t]);
  const bool boundary =
      ordered_key(value[kTopK - 1]) == ordered_key(inv2 * key_value(r.ninth));
  if (r.tied || boundary) {
    nanpoll_fold_select(fold, scales, scale, row, sel_out, wt_out);
    return;
  }
  iquestrt_select::store_top(value, r.id, r.rank, row, sel_out, wt_out);
}
"""

# Thread counts and delays for the attention/MLP tails' second readers,
# and the MLP tail's h of a claimed vector.
_SECOND_HELPERS = f"""
constexpr int ATTN_NT = FOLD_NT + NT + DIRECT_NT;  // then the norm lanes, the second readers
constexpr unsigned long long NANPOLL_SECOND_NS = {SECOND_NS}ULL;
constexpr int MLP_NT = DIRECT_NT + NT + DIRECT_NT;  // row readers, norm lanes, second readers
constexpr unsigned long long NANPOLL_MLP_SECOND_NS = {MLP_SECOND_NS}ULL;
""" + r"""
// A vector's h = BF16(residual + BF16(m * scale)), as the one-read MLP tail forms it.
__device__ __forceinline__ U4 nanpoll_mlp_h(const U4& reduced, const U4& residual_bits,
                                            float s) {
  const __nv_bfloat162* rb = reinterpret_cast<const __nv_bfloat162*>(&reduced);
  const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&residual_bits);
  U4 result;
  __nv_bfloat162* ob = reinterpret_cast<__nv_bfloat162*>(&result);
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const float2 m = __bfloat1622float2(rb[pair]);
    const __nv_bfloat162 scaled = __float22bfloat162_rn(
        make_float2(m.x * s, m.y * s));
    const float2 sf = __bfloat1622float2(scaled);
    const float2 h = __bfloat1622float2(hb[pair]);
    ob[pair] = __float22bfloat162_rn(make_float2(h.x + sf.x, h.y + sf.y));
  }
  return result;
}
"""


def _read(dst: str, off: str, guard: str) -> str:
    """One thread's switch read of its 16 bytes at `off` into `dst`, repeated until
    NaN-free, then `ARMED` over this rank's same 16 bytes of the previous call's buffer
    when `guard` holds."""
    loop = ("{\n    unsigned reads = 0;\n    do {\n"
            f"      {dst} = nanpoll_ld_acquire(mc + {off});\n"
            "      if (++reads > NANPOLL_READ_LIMIT) __trap();\n"
            f"    }} while (nanpoll_any_nan({dst}));\n  }}")
    return (f"  {loop}\n"
            f"  if ({guard}) {{\n"
            "    const U4 armed = {NANPOLL_ARMED, NANPOLL_ARMED, NANPOLL_ARMED, NANPOLL_ARMED};\n"
            f"    *reinterpret_cast<U4*>(rearm + {off}) = armed;\n"
            "  }\n")


def _read_fold(dst: str, off: str) -> str:
    """A fold reader's switch read of its 16 bytes of the slot at `off` into `dst`, as
    `_read`, then fp32 `ARMED` over this rank's same 16 bytes of the previous slot."""
    loop = ("{\n      unsigned reads = 0;\n      do {\n"
            f"        {dst} = nanpoll_ld_acquire_f32(fold_mc + {off});\n"
            "        if (++reads > NANPOLL_READ_LIMIT) __trap();\n"
            f"      }} while (nanpoll_any_nan_f32({dst}));\n    }}")
    return (f"    {loop}\n"
            "    {\n      const float armed = __uint_as_float(NANPOLL_FOLD_ARMED);\n"
            f"      *reinterpret_cast<float4*>(fold_rearm + {off}) =\n"
            "          make_float4(armed, armed, armed, armed);\n    }\n")


def _derive(src, head, end, name, new_name, sig, new_sig, read, new_read, scalar_reads,
            drop=(), hoist=None, replace=()):
    """The kernel `head`..`end` of `src` without its barrier and the texts in `drop`, its
    read of the switch replaced by `new_read` and each `(old, new)` of `replace` made.

    `scalar_reads` counts the kernel's reads of `scale[0]` and `eps[0]`; their values are
    read once into registers, after staging and before the poll:
    right ahead of the read, or ahead of `hoist` when the read sits in a loop."""
    start = src.index(head)
    kernel = src[start:src.index(end, start)]
    kernel = replace_once(kernel, f"void {name}(", f"void {new_name}(")
    kernel = replace_once(kernel, sig, new_sig)
    kernel = replace_once(kernel, _BARRIER, "")
    for text in drop:
        kernel = replace_once(kernel, text, "")
    for old, new in replace:
        kernel = replace_once(kernel, old, new)
    scalars = []
    for scalar, count in zip(("scale", "eps"), scalar_reads):
        kernel, found = re.subn(rf"\b{scalar}\[0\]", f"nanpoll_{scalar}", kernel)
        if found != count:
            raise RuntimeError(f"{name} reads {scalar}[0] {found} times, not {count}; "
                               "update its NaN-polled form")
        if count:
            scalars.append(f"nanpoll_{scalar} = {scalar}[0]")
    reads = "  const float " + ", ".join(scalars) + ";\n"
    if hoist is None:
        kernel = replace_once(kernel, read, reads + new_read)
    else:
        kernel = replace_once(kernel, hoist, reads + hoist)
        kernel = replace_once(kernel, read, new_read)
    body = re.sub(r"//[^\n]*", "", kernel[kernel.index(") {") + 3:])
    for unused in ("signal_pads", "rank", "world"):
        if re.search(rf"\b{unused}\b", body):
            raise RuntimeError(f"{name} reads `{unused}` outside the barrier; "
                               "update its NaN-polled form")
    return kernel


# `ar_prefetch`'s attention tail: one 16-byte vector of the row per thread, and the
# router fold. ATTN_NT threads: the row's readers
# and the fold's, the norm lanes, and the row's second readers. Only the first row
# readers stage; every reader leaves at its read, and the norm lanes start once every
# vector of the row is claimed, in place of the `__syncthreads` after the read; they hand
# the selecting warp s(ao) at barrier 7 and s(h) at barrier 6. `ar_tail`'s rows use
# barriers 1..3.
#
# The attention tail stages with every load ahead of every store: a thread's residual
# and norm weight vectors go to registers first and to shared memory after, the same
# bytes to the same addresses. The MLP tail's norm lanes stage the same way.
_STAGE_LOOPS = """      for (int pack = tid - 32; pack < H / 8; pack += LOADERS) {
        *reinterpret_cast<U4*>(residual_stage + pack * 8) =
            ld_global_u4(residual + row_base + pack * 8);
      }
      for (int pack = tid - 32; pack < H / 4; pack += LOADERS) {
"""
_STAGE_HEAD = """      constexpr int ROW_N = (H / 8 + LOADERS - 1) / LOADERS;
      constexpr int W_N = (H / 4 + LOADERS - 1) / LOADERS;
      U4 res[ROW_N];
#pragma unroll
      for (int i = 0; i < ROW_N; ++i) {
        const int pack = tid - 32 + i * LOADERS;
        if (pack < H / 8) res[i] = ld_global_u4(residual + row_base + pack * 8);
      }
"""
_STAGE_ROW_STORE = """#pragma unroll
      for (int i = 0; i < ROW_N; ++i) {
        const int pack = tid - 32 + i * LOADERS;
        if (pack < H / 8) *reinterpret_cast<U4*>(residual_stage + pack * 8) = res[i];
      }
"""
_STAGE_ATTENTION = (_STAGE_LOOPS + """        *reinterpret_cast<float4*>(w_ao_stage + pack * 4) =
            *reinterpret_cast<const float4*>(w_ao + pack * 4);
        if constexpr (EmitXPad) {
          *reinterpret_cast<float4*>(w_ffn_stage + pack * 4) =
              *reinterpret_cast<const float4*>(w_ffn + pack * 4);
        }
      }
""", _STAGE_HEAD + """      float4 ao[W_N], ffn[W_N];
#pragma unroll
      for (int i = 0; i < W_N; ++i) {
        const int pack = tid - 32 + i * LOADERS;
        if (pack < H / 4) {
          ao[i] = *reinterpret_cast<const float4*>(w_ao + pack * 4);
          if constexpr (EmitXPad) ffn[i] = *reinterpret_cast<const float4*>(w_ffn + pack * 4);
        }
      }
""" + _STAGE_ROW_STORE + """#pragma unroll
      for (int i = 0; i < W_N; ++i) {
        const int pack = tid - 32 + i * LOADERS;
        if (pack < H / 4) {
          *reinterpret_cast<float4*>(w_ao_stage + pack * 4) = ao[i];
          if constexpr (EmitXPad) *reinterpret_cast<float4*>(w_ffn_stage + pack * 4) = ffn[i];
        }
      }
""")
_ATTENTION_READ = """  const __nv_bfloat16* mc_row = mc + static_cast<size_t>(row) * H;
  constexpr int N8 = H / 8;
  for (int i = tid; i < N8; i += DIRECT_NT) {
    const U4 v = mm_ld_reduce_bf16x8(mc_row + 8 * i);
    *reinterpret_cast<U4*>(ao_s + 8 * i) = v;
  }
"""
# The norm lanes start once every vector of the row is claimed.
# Trapped after 30 s, well past a row reader's own read limit.
_NORM_WAIT = """  {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0) :: "memory");
    while (*reinterpret_cast<volatile unsigned*>(&nanpoll_claimed) < DIRECT_NT) {
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
      if (t - t0 > 30000000000ULL) __trap();
    }
  }
  __threadfence_block();
"""
_ATTENTION = dict(
    head=("template <bool WaitPDL = false, bool OverlapLocal = false, bool EmitXPad = false,\n"
          "          bool EarlyPDLTrigger = false>\n__global__ __launch_bounds__(DIRECT_NT, 1) "
          "void ar_tail_direct1bar_h_inv("),
    end="// The next prologue otherwise recomputes this RMSNorm",
    name="ar_tail_direct1bar_h_inv", new_name="nanpoll_attention_tail",
    sig="    uint8_t* __restrict__ x_pad_fp8) {",
    new_sig=("    uint8_t* __restrict__ x_pad_fp8, __nv_bfloat16* __restrict__ rearm,\n"
             "    const float* fold_mc, float* __restrict__ fold_rearm,\n"
             "    int* __restrict__ sel_out, float* __restrict__ wt_out) {"),
    read=_ATTENTION_READ,
    scalar_reads=(1, 2),
    replace=(("__launch_bounds__(DIRECT_NT, 1)", "__launch_bounds__(ATTN_NT, 1)"),
             ("    if (tid >= 32) {\n      constexpr int LOADERS = DIRECT_NT - 32;\n",
              "    if (tid >= 32 && tid < DIRECT_NT) {\n"
              "      constexpr int LOADERS = DIRECT_NT - 32;\n"),
             _STAGE_ATTENTION,
             ("  __shared__ float red[NT];\n",
              "  __shared__ float red[NT];\n"
              "  __shared__ __align__(16) float fold_s[FOLD_ROWS];\n"
              "  __shared__ float fold_scale[2];\n"
              "  __shared__ __align__(16) int fold_ids[iquestrt_select::kTopK];\n"
              "  __shared__ unsigned nanpoll_claim[DIRECT_NT];\n"
              "  __shared__ unsigned nanpoll_claimed;\n"),
             ("  __syncthreads();\n" + _ATTENTION_READ,
              "  if (tid >= FOLD_NT + NT) nanpoll_claim[tid - (FOLD_NT + NT)] = 0u;\n"
              "  if (tid == FOLD_NT) nanpoll_claimed = 0u;\n"
              "  __syncthreads();\n" + _ATTENTION_READ),
             ("  __syncthreads();\n\n  if (tid < NT) {\n    const int lane = tid;\n",
              _NORM_WAIT + "\n  {\n    const int lane = tid - FOLD_NT;\n"),
             ("    const float inv = rsqrtf(sum / 3072.0f + eps[0]);\n",
              "    const float inv = rsqrtf(sum / 3072.0f + eps[0]);\n"
              "    if (lane == 0) fold_scale[0] = inv;\n"
              '    asm volatile("bar.arrive 7, %0;" :: "n"(NT + 32) : "memory");\n'),
             ("    if (lane == 0) inv_out[row] = inv2;\n",
              "    if (lane == 0) {\n      inv_out[row] = inv2;\n      fold_scale[1] = inv2;\n"
              "    }\n"
              '    asm volatile("bar.arrive 6, %0;" :: "n"(NT + 32) : "memory");\n')))
# `ar_prefetch`'s MLP tail: likewise; it re-arms the rows the previous buffer's producer
# writes. `FinalNorm` sums the norm's squares in the final norm's order instead. Derived
# without its read, then rewritten onto MLP_NT threads (`_mlp_early`).
_MLP_TEMPLATE = "template <bool OverlapLocal, bool EarlyTrigger = false"
_MLP_REDUCE = "  sum = reduce_group_128(sum, red, 0, lane);\n"
_MLP_FINAL = "  if constexpr (FinalNorm) sum = nanpoll_final_norm_sum(h_s, lane);\n"
_MLP = dict(
    head=(_MLP_TEMPLATE + ">\n"
          "__global__ __launch_bounds__(DIRECT_NT, 1) void mlp_tail_norm_direct1bar_impl("),
    end="\nvoid ar_tail_direct1bar_launch(",
    name="mlp_tail_norm_direct1bar_impl", new_name="nanpoll_mlp_tail",
    sig="    int rank, int world, int q) {",
    new_sig=("    int rank, int world, int q, __nv_bfloat16* __restrict__ rearm,\n"
             "    int rearm_rows) {"),
    read="  const U4 reduced = mm_ld_reduce_bf16x8(mc + row_base + base8);\n",
    scalar_reads=(1, 1),
    replace=((_MLP_TEMPLATE + ">", _MLP_TEMPLATE + ", bool FinalNorm = false>"),
             (_MLP_REDUCE, _MLP_FINAL + _MLP_REDUCE)))
# `ar_tail`'s eh_proj tail, reorganised onto DIRECT_NT threads with one 16-byte vector a
# thread, without its PDL wait. Its `eps` read moves out of the read's loop.
#
# WHAT THIS TAIL OWNS. Grid 8 CTAs x DIRECT_NT threads: CTA `row` takes row `row` of the
# depth's buffer, all Q of which the eh_proj writes, and thread `threadIdx.x` the one
# vector at `seg * 1024 + lane * 8` for `seg, lane = threadIdx.x / NT, threadIdx.x % NT`
# -- together the row's H elements, each a 16-byte switch read, each read once.
#   Synchronisation before the norm. A thread's read serves the norm of all three of its
#   `lane`'s segments, so the reduced row is staged in shared memory and the first NT
#   threads rebuild their own 24 squares from it -- the same BF16 bits, so the same
#   products -- and sum them in `MtpEhARNorm`'s order (`sq[(rv % 3) * 8 + rv / 3]`: the
#   three segments interleaved, one serial fp32 accumulation) before meeting their 127
#   peers in `reduce_group_128`'s named-barrier pair over NT threads. The extra
#   readers wait at the barrier around that block rather than retiring, and take `inv`
#   from shared memory. What follows is elementwise in `inv`, computed from the thread's
#   own read, so a vector's output bits depend on its own read and on that one reduction.
#   Outputs. `raw_out` is the holder's `raw_h0` and `norm_out` its `norm_out`, both
#   rank-local workspace slots that every depth overwrites: the draft layer reads the raw
#   reduced hidden as its residual base and the normalised row as its prenorm QKV operand
#   (`engine._head_step_impl`), and nothing else reads either. Each thread writes its own
#   vector of both.
#   Re-arm. A thread stores `ARMED` over its own 16 bytes of the previous eh call's buffer
#   once that read is NaN-free, so the launch's threads cover rows 0..Q-1 of it -- the
#   rows its eh_proj wrote -- once each, and no other row. The previous call is
#   `DepthTurns`', not depth - 1: a step's depths 0..r-1 make depth d re-arm d - 1 and
#   depth 0 re-arm the last depth the step before ran, while a request's eager prefill,
#   depths 1..r-1, makes its depth 1 re-arm the buffer the step before it left. Either
#   sequence ends with the buffer of its last depth r-1 holding its partials, every other
#   buffer armed, and `previous` = r-1, so steps and prefills follow one another in any
#   order and a repeated request starts from the state the last one ended in. A captured
#   step's pairing is fixed at capture: its host graph's depth 0 re-arms the buffer of
#   depth w-2 for its verify width w (`reset_turns`), and every later depth the buffer of
#   the depth before it, re-armed after its own producer wrote it.
_EH = dict(
    head=("template <bool NeedRaw>\n"
          "__global__ __launch_bounds__(NT, 1) void mtp_eh_ar_norm_pdl("),
    end="template <bool NeedRaw>\nvoid launch_mtp_eh_ar_norm(",
    name="mtp_eh_ar_norm_pdl", new_name="nanpoll_eh_tail",
    sig="    int rank, int world, int q) {",
    new_sig="    int rank, int world, int q, __nv_bfloat16* __restrict__ rearm) {",
    read="    const U4 reduced = mm_ld_reduce_bf16x8(mc + row_base + base);\n",
    new_read="    const U4 reduced = *reinterpret_cast<const U4*>(eh_s + base);\n",
    scalar_reads=(0, 1),
    drop=(_PDL_WAIT,
          # The thread's own read already wrote its vector of the raw hidden, before the
          # staging barrier; the loop below only rebuilds the norm's squares.
          "    reduced_bits[segment] = reduced;\n",
          "    if constexpr (NeedRaw)\n"
          "      *reinterpret_cast<U4*>(raw_out + row_base + base) = reduced;\n"),
    hoist="  const size_t row_base = static_cast<size_t>(row) * H;\n")
# `ar_tail`'s fp32-residual attention tail of the draft: one vector per thread, without
# its PDL wait; it still releases the Router at entry.
_DRAFT_ATTENTION = dict(
    head="__global__ __launch_bounds__(DIRECT_NT, 1) void ar_tail_h32_inv_xpad_fp8(",
    end="void ar_tail_h32_inv_xpad_fp8_launch(",
    name="ar_tail_h32_inv_xpad_fp8", new_name="nanpoll_draft_attention_tail",
    sig="    uint8_t* __restrict__ x_pad_fp8) {",
    new_sig="    uint8_t* __restrict__ x_pad_fp8, __nv_bfloat16* __restrict__ rearm) {",
    read=("  for (int i = tid; i < H / 8; i += DIRECT_NT)\n"
          "    *reinterpret_cast<U4*>(ao_s + 8 * i) = mm_ld_reduce_bf16x8(mc + row_base + 8 * i);\n"),
    scalar_reads=(1, 2),
    drop=(_PDL_WAIT,),
    # Its second norm weight, into the registers of the lanes that read it, after the
    # shipping staging and before the poll.
    replace=(("      *reinterpret_cast<float4*>(w_ao_rows + pack * 4) =\n"
              "          *reinterpret_cast<const float4*>(w_ao + pack * 4);\n  }\n",
              "      *reinterpret_cast<float4*>(w_ao_rows + pack * 4) =\n"
              "          *reinterpret_cast<const float4*>(w_ao + pack * 4);\n  }\n"
              "  float2 nanpoll_w_ffn[3][4];\n"
              "  if (tid < NT) {\n"
              "#pragma unroll\n"
              "    for (int segment = 0; segment < 3; ++segment)\n"
              "#pragma unroll\n"
              "      for (int pair = 0; pair < 4; ++pair)\n"
              "        nanpoll_w_ffn[segment][pair] = *reinterpret_cast<const float2*>(\n"
              "            w_ffn + segment * 1024 + tid * 8 + 2 * pair);\n"
              "  }\n"),
             ("      const float2 w = *reinterpret_cast<const float2*>(w_ffn + base + 2 * pair);\n",
              "      const float2 w = nanpoll_w_ffn[segment][pair];\n")))
# `ar_tail`'s sandwich tail: the draft depth's exit in its fp32-residual form, and main
# layer 0's two tails in its BF16 form. One vector per thread; it re-arms the rows the
# previous buffer's producer writes.
_SANDWICH = dict(
    head=("template <bool Fp32Residual = false>\n"
          "__global__ __launch_bounds__(DIRECT_NT, 1) void ar_tail_direct1bar_impl("),
    end="// Producer half for the SM100 inv-router chain.",
    name="ar_tail_direct1bar_impl", new_name="nanpoll_sandwich_tail",
    sig="    float* __restrict__ h32_out) {",
    new_sig=("    float* __restrict__ h32_out, __nv_bfloat16* __restrict__ rearm,\n"
             "    int rearm_rows) {"),
    read=_ATTENTION["read"],
    scalar_reads=(1, 2),
    # Its residual and both norm weights, into the registers of the lanes that read them,
    # before the poll.
    replace=(("  __shared__ float red[NT];\n",
              "  __shared__ float red[NT];\n"
              "  float4 nanpoll_w_ao[6], nanpoll_w_ffn[6], nanpoll_residual32[6];\n"
              "  uint2 nanpoll_residual[6];\n"
              "  if (tid < NT) {\n"
              "#pragma unroll\n"
              "    for (int i = 0; i < 6; ++i) {\n"
              "      const int base = i * 512 + tid * 4;\n"
              "      nanpoll_w_ao[i] = *reinterpret_cast<const float4*>(w_ao + base);\n"
              "      if constexpr (Fp32Residual)\n"
              "        nanpoll_residual32[i] = *reinterpret_cast<const float4*>(\n"
              "            residual32 + static_cast<size_t>(row) * H + base);\n"
              "      else\n"
              "        nanpoll_residual[i] = *reinterpret_cast<const uint2*>(\n"
              "            residual + static_cast<size_t>(row) * H + base);\n"
              "      nanpoll_w_ffn[i] = *reinterpret_cast<const float4*>(w_ffn + base);\n"
              "    }\n"
              "  }\n"),
             ("    const float4 w = *reinterpret_cast<const float4*>(w_ao + base);\n",
              "    const float4 w = nanpoll_w_ao[i];\n"),
             ("      const float4 rf = *reinterpret_cast<const float4*>(\n"
              "          residual32 + static_cast<size_t>(row) * H + base);\n",
              "      const float4 rf = nanpoll_residual32[i];\n"),
             ("      const uint2 ru = *reinterpret_cast<const uint2*>(\n"
              "          residual + static_cast<size_t>(row) * H + base);\n",
              "      const uint2 ru = nanpoll_residual[i];\n"),
             ("    const float4 w = *reinterpret_cast<const float4*>(w_ffn + base);\n",
              "    const float4 w = nanpoll_w_ffn[i];\n")))
_ONE_VECTOR = '  static_assert(H / 8 == DIRECT_NT, "one 16-byte vector of the row per thread");\n'


def _attention_read() -> str:
    """The attention tail's read: a fold reader's vector of the row's slot, staged for
    the selecting warp, which ranks the row once the norm has s(ao) and routes it once it
    has s(h); a row reader's vector of the partial, and a delayed reader's second read,
    claimed into `ao_s` by the first NaN-free one."""
    return ("  if (tid >= DIRECT_NT && tid < FOLD_NT) {\n"
            "    const int j = tid - DIRECT_NT;\n"
            "    const size_t off = static_cast<size_t>(row) * FOLD_ROWS + 4 * j;\n"
            "    float4 f;\n" + _read_fold("f", "off")
            + "    *reinterpret_cast<float4*>(fold_s + 4 * j) = f;\n"
            '    asm volatile("bar.sync 5, %0;" :: "n"(FOLD_NT - DIRECT_NT) : "memory");\n'
            "    if (j < 32) {\n"
            '      asm volatile("bar.sync 7, %0;" :: "n"(NT + 32) : "memory");\n'
            "      const NanpollFoldRank r =\n"
            "          nanpoll_fold_rank(fold_s, nanpoll_scale * fold_scale[0], fold_ids);\n"
            '      asm volatile("bar.sync 6, %0;" :: "n"(NT + 32) : "memory");\n'
            "      nanpoll_fold_finish(r, fold_s, fold_scale, nanpoll_scale, row, sel_out,\n"
            "                          wt_out);\n"
            "    }\n    return;\n  }\n"
            + _ONE_VECTOR + _SECOND_READ + _FIRST_READ)


# A row vector's claim: its first NaN-free reader re-arms it, stores it into `ao_s` and
# counts it for the norm lanes (`_NORM_WAIT`) behind a block fence.
_CLAIM = """      const U4 armed = {NANPOLL_ARMED, NANPOLL_ARMED, NANPOLL_ARMED, NANPOLL_ARMED};
      *reinterpret_cast<U4*>(rearm + off) = armed;
      *reinterpret_cast<U4*>(ao_s + 8 * VEC) = VALUE;
      __threadfence_block();
      atomicAdd(&nanpoll_claimed, 1u);
"""
# A second reader: one read of vector v, NANPOLL_SECOND_NS after the barrier ahead of the
# first readers' first read, unless the vector is claimed by then.
_SECOND_READ = """  if (tid >= FOLD_NT + NT) {
    const int v = tid - (FOLD_NT + NT);
    const size_t off = static_cast<size_t>(row) * H + 8 * v;
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0) :: "memory");
    do {
      __nanosleep(64);
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
    } while (t - t0 < NANPOLL_SECOND_NS);
    if (*reinterpret_cast<volatile unsigned*>(&nanpoll_claim[v]) != 0u) return;
    const U4 b = nanpoll_ld_acquire(mc + off);
    if (!nanpoll_any_nan(b) && atomicExch(&nanpoll_claim[v], 1u) == 0u) {
""" + _CLAIM.replace("VEC", "v").replace("VALUE", "b") + """    }
    return;
  }
"""
# A first reader: the shipped poll of its vector, until it is NaN-free or claimed.
_FIRST_READ = """  if (tid < DIRECT_NT) {
    const size_t off = static_cast<size_t>(row) * H + 8 * tid;
    U4 v;
    unsigned reads = 0;
    do {
      v = nanpoll_ld_acquire(mc + off);
      if (++reads > NANPOLL_READ_LIMIT) __trap();
      if (!nanpoll_any_nan(v)) {
        if (atomicExch(&nanpoll_claim[tid], 1u) == 0u) {
""" + _CLAIM.replace("      ", " " * 10).replace("VEC", "tid").replace("VALUE", "v") + """        }
        break;
      }
    } while (*reinterpret_cast<volatile unsigned*>(&nanpoll_claim[tid]) == 0u);
    return;
  }
"""


# The MLP early read: the derived MLP tail from its staging to its norm lanes'
# entry, rewritten. After the claims' reset at entry, a reader loads its residual vector
# (`ld_global_u4`: an asm load stays ahead of the poll, where a plain one sank below the
# claim) and the scale, and reads; a vector's first NaN-free read claims it, re-arms it,
# forms its h, stores it and counts it. The norm lanes stage the norm weights (loads first)
# and load eps while the row is read, then wait for every claim.
_MLP_STAGE = "  if constexpr (OverlapLocal) {\n    // Warp 0 owns"
_MLP_NORM = "  if (tid >= NT) return;\n  const int lane = tid;\n"
_MLP_ENTRY = """  if (tid < DIRECT_NT) nanpoll_mclaim[tid] = 0u;
  if (tid == DIRECT_NT) nanpoll_mclaimed = 0u;
  __syncthreads();
  unsigned long long nanpoll_t0;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(nanpoll_t0) :: "memory");
"""
_MLP_CLAIM = """      if (row < rearm_rows) {
        const U4 armed = {NANPOLL_ARMED, NANPOLL_ARMED, NANPOLL_ARMED, NANPOLL_ARMED};
        *reinterpret_cast<U4*>(rearm + off) = armed;
      }
      {
        const U4 hv = nanpoll_mlp_h(VALUE, res, s);
        *reinterpret_cast<U4*>(h_s + 8 * VEC) = hv;
        *reinterpret_cast<U4*>(out + off) = hv;
      }
      __threadfence_block();
      atomicAdd(&nanpoll_mclaimed, 1u);
"""
_MLP_OWN = """    const U4 res = ld_global_u4(residual + off);
    const float s = scale[0];
"""
_MLP_SECOND_READ = """  if (tid >= DIRECT_NT + NT) {
    const int v = tid - (DIRECT_NT + NT);
    const size_t off = row_base + 8 * v;
""" + _MLP_OWN + """    while (true) {
      unsigned long long t;
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
      if (t - nanpoll_t0 >= NANPOLL_MLP_SECOND_NS) break;
      __nanosleep(64);
    }
    if (*reinterpret_cast<volatile unsigned*>(&nanpoll_mclaim[v]) != 0u) return;
    const U4 b = nanpoll_ld_acquire(mc + off);
    if (!nanpoll_any_nan(b) && atomicExch(&nanpoll_mclaim[v], 1u) == 0u) {
""" + _MLP_CLAIM.replace("VEC", "v").replace("VALUE", "b") + """    }
    return;
  }
"""
_MLP_FIRST_READ = """  if (tid < DIRECT_NT) {
    const size_t off = row_base + 8 * tid;
""" + _MLP_OWN + """    U4 v;
    unsigned reads = 0;
    do {
      v = nanpoll_ld_acquire(mc + off);
      if (++reads > NANPOLL_READ_LIMIT) __trap();
      if (!nanpoll_any_nan(v)) {
        if (atomicExch(&nanpoll_mclaim[tid], 1u) == 0u) {
""" + _MLP_CLAIM.replace("      ", " " * 10).replace("VEC", "tid").replace("VALUE", "v") + """        }
        break;
      }
    } while (*reinterpret_cast<volatile unsigned*>(&nanpoll_mclaim[tid]) == 0u);
    return;
  }
"""
_MLP_NORM_LANES = """  float nanpoll_eps_lane;
  {
    const int j = tid - DIRECT_NT;
    if constexpr (OverlapLocal) {
      constexpr int W_N = H / 4 / NT;
      float4 nw[W_N];
#pragma unroll
      for (int i = 0; i < W_N; ++i)
        nw[i] = *reinterpret_cast<const float4*>(w_norm + (i * NT + j) * 4);
#pragma unroll
      for (int i = 0; i < W_N; ++i)
        *reinterpret_cast<float4*>(norm_stage + (i * NT + j) * 4) = nw[i];
    }
    nanpoll_eps_lane = eps[0];
  }
  {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0) :: "memory");
    while (*reinterpret_cast<volatile unsigned*>(&nanpoll_mclaimed) < DIRECT_NT) {
      asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
      if (t - t0 > 30000000000ULL) __trap();
    }
  }
  __threadfence_block();
  const float nanpoll_eps = nanpoll_eps_lane;
  const int lane = tid - DIRECT_NT;
"""


def _mlp_early(kernel: str) -> str:
    """The derived MLP tail on MLP_NT threads, including delayed readers."""
    kernel = replace_once(kernel, "__launch_bounds__(DIRECT_NT, 1) void nanpoll_mlp_tail(",
                          "__launch_bounds__(MLP_NT, 1) void nanpoll_mlp_tail(")
    kernel = replace_once(kernel, "  __shared__ float red[NT];\n",
                          "  __shared__ float red[NT];\n"
                          "  __shared__ unsigned nanpoll_mclaim[DIRECT_NT];\n"
                          "  __shared__ unsigned nanpoll_mclaimed;\n")
    start, end = kernel.index(_MLP_STAGE), kernel.index(_MLP_NORM) + len(_MLP_NORM)
    reads = _MLP_SECOND_READ + _MLP_FIRST_READ
    return kernel[:start] + _MLP_ENTRY + _ONE_VECTOR + reads + _MLP_NORM_LANES + kernel[end:]


def _one_vector(off: str, guard: str) -> str:
    """A tail's read of the row's one vector per thread, at `off`, into `ao_s`."""
    return (_ONE_VECTOR + "  {\n"
            f"  const size_t off = {off};\n"
            "  U4 v;\n" + _read("v", "off", guard)
            + "  *reinterpret_cast<U4*>(ao_s + 8 * tid) = v;\n  }\n")


# The eh tail's own read and staging, ahead of the 128-lane norm, and that norm's close
# (the comment above `_EH`). The whole row is in shared memory, so its square loop reads
# `eh_s` (`_EH["new_read"]`) instead of the switch.
_EH_STAGE = (_ONE_VECTOR
             + '  static_assert(DIRECT_NT == 3 * NT, "three segments, one a thread");\n'
             + "  __shared__ __nv_bfloat16 eh_s[H];\n"
             + "  __shared__ float inv_s;\n"
             + "  const int base0 = seg * 1024 + lane * 8;\n"
             + "  U4 poll_bits;\n")
_EH_PUBLISH = ("  if constexpr (NeedRaw)\n"
               "    *reinterpret_cast<U4*>(raw_out + row_base + base0) = poll_bits;\n"
               "  *reinterpret_cast<U4*>(eh_s + base0) = poll_bits;\n"
               "  __syncthreads();\n"
               "  if (threadIdx.x < NT) {\n")


def _eh_replace():
    """`_EH`'s rewrite onto DIRECT_NT threads: the thread split, its one polled read and
    the staging around the norm, then its own vector of the normalised row."""
    return (
        ("__launch_bounds__(NT, 1)", "__launch_bounds__(DIRECT_NT, 1)"),
        ("  const int lane = threadIdx.x;\n",
         "  const int seg = threadIdx.x / NT;\n  const int lane = threadIdx.x % NT;\n"),
        ("  U4 reduced_bits[3];\n",
         _EH_STAGE + _read("poll_bits", "row_base + base0", "true") + _EH_PUBLISH),
        ("  const float inv = rsqrtf(sum / 3072.0f + eps[0]);\n",
         "  if (lane == 0) inv_s = rsqrtf(sum / 3072.0f + eps[0]);\n  }\n"
         "  __syncthreads();\n  const float inv = inv_s;\n"),
        ("#pragma unroll\n  for (int segment = 0; segment < 3; ++segment) {\n"
         "    const int base = segment * 1024 + lane * 8;\n"
         "    const __nv_bfloat162* values =\n"
         "        reinterpret_cast<const __nv_bfloat162*>(&reduced_bits[segment]);\n",
         "  {\n    const int base = base0;\n"
         "    const __nv_bfloat162* values =\n"
         "        reinterpret_cast<const __nv_bfloat162*>(&poll_bits);\n"),
    )

_LAUNCH = r"""
static cudaLaunchConfig_t nanpoll_config(int64_t q, size_t smem, cudaLaunchAttribute* attr) {
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<unsigned>(q), 1, 1);
  config.blockDim = dim3(DIRECT_NT, 1, 1);
  config.dynamicSmemBytes = smem;
  config.stream = at::cuda::getCurrentCUDAStream();
  attr->id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr->val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attr;
  config.numAttrs = 1;
  return config;
}

void nanpoll_attention_tail_launch(int64_t mc, int64_t rearm, int64_t fold_mc,
                                   int64_t fold_rearm, int64_t residual, int64_t w_ao,
                                   int64_t w_ffn, int64_t scale, int64_t eps,
                                   int64_t h_out, int64_t inv_out, int64_t x_pad,
                                   int64_t x_pad_fp8, int64_t sel, int64_t wt, int64_t q) {
  TORCH_CHECK(mc != 0 && rearm != 0 && fold_mc != 0 && fold_rearm != 0 && inv_out != 0 &&
              x_pad != 0 && x_pad_fp8 != 0 && sel != 0 && wt != 0,
              "NaN-polled attention tail pointers must be non-null");
  TORCH_CHECK(q == 8, "the NaN-polled attention tail is Q8-only, got ", q);
  TORCH_CHECK((mc | rearm | fold_mc | fold_rearm | residual | w_ao | w_ffn | h_out | x_pad |
               x_pad_fp8) % 16 == 0,
              "NaN-polled attention tail vector operands must be 16-byte aligned");
  TORCH_CHECK(((inv_out | sel | wt) & 3) == 0,
              "NaN-polled attention tail inv, sel and wt must be 4-byte aligned");
  cudaLaunchAttribute attr{};
  cudaLaunchConfig_t config =
      nanpoll_config(q, H * sizeof(__nv_bfloat16) + 2 * H * sizeof(float), &attr);
  config.blockDim = dim3(ATTN_NT, 1, 1);
  // No PDL wait, an early release, the shipping staging and dual output, and the fold;
  // the signal pads, the rank and the world size are unused.
  const cudaError_t err = cudaLaunchKernelEx(
      &config, nanpoll_attention_tail<false, true, true, true>,
      reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<const float*>(w_ao), reinterpret_cast<const float*>(scale),
      reinterpret_cast<const float*>(eps), reinterpret_cast<__nv_bfloat16*>(h_out),
      reinterpret_cast<float*>(inv_out), static_cast<unsigned int**>(nullptr), 0, 8,
      static_cast<int>(q), reinterpret_cast<const float*>(w_ffn),
      reinterpret_cast<__nv_bfloat16*>(x_pad), reinterpret_cast<uint8_t*>(x_pad_fp8),
      reinterpret_cast<__nv_bfloat16*>(rearm), reinterpret_cast<const float*>(fold_mc),
      reinterpret_cast<float*>(fold_rearm), reinterpret_cast<int*>(sel),
      reinterpret_cast<float*>(wt));
  TORCH_CHECK(err == cudaSuccess, "NaN-polled attention tail launch: ",
              cudaGetErrorString(err));
}

void nanpoll_mlp_tail_launch(int64_t mc, int64_t rearm, int64_t residual, int64_t scale,
                             int64_t w_norm, int64_t eps, int64_t out, int64_t norm_out,
                             int64_t rearm_rows, int64_t q, bool final_norm) {
  TORCH_CHECK(mc != 0 && rearm != 0, "NaN-polled MLP tail pointers must be non-null");
  TORCH_CHECK(q == 8, "the NaN-polled MLP tail is Q8-only, got ", q);
  TORCH_CHECK(rearm_rows >= 1 && rearm_rows <= q, "re-armed rows must be in 1..", q,
              ", got ", rearm_rows);
  TORCH_CHECK((mc | rearm | residual | w_norm | out | norm_out) % 16 == 0,
              "NaN-polled MLP tail vector operands must be 16-byte aligned");
  cudaLaunchAttribute attr{};
  cudaLaunchConfig_t config =
      nanpoll_config(q, H * (sizeof(__nv_bfloat16) + sizeof(float)), &attr);
  config.blockDim = dim3(MLP_NT, 1, 1);  // the MLP early read
  // Rank-local staging and the early release of the next prologue, as shipped, and the
  // final norm's order after the last layer; the signal pads, the rank and the world size
  // are unused.
  const auto kernel =
      final_norm ? nanpoll_mlp_tail<true, true, true> : nanpoll_mlp_tail<true, true, false>;
  const cudaError_t err = cudaLaunchKernelEx(
      &config, kernel, reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<const float*>(scale), reinterpret_cast<const float*>(w_norm),
      reinterpret_cast<const float*>(eps), reinterpret_cast<__nv_bfloat16*>(out),
      reinterpret_cast<__nv_bfloat16*>(norm_out), static_cast<unsigned int**>(nullptr), 0,
      8, static_cast<int>(q), reinterpret_cast<__nv_bfloat16*>(rearm),
      static_cast<int>(rearm_rows));
  TORCH_CHECK(err == cudaSuccess, "NaN-polled MLP tail launch: ", cudaGetErrorString(err));
}

void nanpoll_eh_tail_launch(int64_t mc, int64_t rearm, int64_t w_norm, int64_t eps,
                            int64_t raw_out, int64_t norm_out, int64_t q) {
  TORCH_CHECK(mc != 0 && rearm != 0 && w_norm != 0 && eps != 0 && raw_out != 0 &&
              norm_out != 0, "NaN-polled eh tail pointers must be non-null");
  TORCH_CHECK(q == 8, "the NaN-polled eh tail is Q8-only, got ", q);
  TORCH_CHECK((mc | rearm | w_norm | raw_out | norm_out) % 16 == 0,
              "NaN-polled eh tail vector operands must be 16-byte aligned");
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t config = nanpoll_config(q, 0, &attr);
  // No PDL wait; the raw reduced hidden and the attention norm, as shipped. The signal
  // pads, the rank and the world size are unused.
  const cudaError_t err = cudaLaunchKernelEx(
      &config, nanpoll_eh_tail<true>, reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const float*>(w_norm), reinterpret_cast<const float*>(eps),
      reinterpret_cast<__nv_bfloat16*>(raw_out), reinterpret_cast<__nv_bfloat16*>(norm_out),
      static_cast<unsigned int**>(nullptr), 0, 8, static_cast<int>(q),
      reinterpret_cast<__nv_bfloat16*>(rearm));
  TORCH_CHECK(err == cudaSuccess, "NaN-polled eh tail launch: ", cudaGetErrorString(err));
}

void nanpoll_draft_attention_tail_launch(int64_t mc, int64_t rearm, int64_t residual,
                                         int64_t w_ao, int64_t w_ffn, int64_t scale,
                                         int64_t eps, int64_t h32_out, int64_t inv_out,
                                         int64_t x_pad, int64_t x_pad_fp8, int64_t q) {
  TORCH_CHECK(mc != 0 && rearm != 0 && h32_out != 0 && inv_out != 0 && x_pad != 0 &&
              x_pad_fp8 != 0, "NaN-polled draft attention tail pointers must be non-null");
  TORCH_CHECK(q == 8, "the NaN-polled draft attention tail is Q8-only, got ", q);
  TORCH_CHECK((mc | rearm | residual | w_ao | w_ffn | h32_out | x_pad | x_pad_fp8) % 16 == 0,
              "NaN-polled draft attention tail vector operands must be 16-byte aligned");
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t config =
      nanpoll_config(q, H * (sizeof(__nv_bfloat16) + sizeof(float)), &attr);
  // No PDL wait; the shipping staging, early release of the Router and fp32 h.
  const cudaError_t err = cudaLaunchKernelEx(
      &config, nanpoll_draft_attention_tail, reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const __nv_bfloat16*>(residual),
      reinterpret_cast<const float*>(w_ao), reinterpret_cast<const float*>(scale),
      reinterpret_cast<const float*>(eps), reinterpret_cast<float*>(h32_out),
      reinterpret_cast<float*>(inv_out), static_cast<unsigned int**>(nullptr), 0, 8,
      static_cast<int>(q), reinterpret_cast<const float*>(w_ffn),
      reinterpret_cast<__nv_bfloat16*>(x_pad), reinterpret_cast<uint8_t*>(x_pad_fp8),
      reinterpret_cast<__nv_bfloat16*>(rearm));
  TORCH_CHECK(err == cudaSuccess, "NaN-polled draft attention tail launch: ",
              cudaGetErrorString(err));
}

void nanpoll_sandwich_tail_launch(int64_t mc, int64_t rearm, int64_t rearm_rows,
                                  int64_t residual32, int64_t w_ao, int64_t w_ffn,
                                  int64_t scale, int64_t eps, int64_t h32_out,
                                  int64_t hn_out, int64_t q) {
  TORCH_CHECK(mc != 0 && rearm != 0 && residual32 != 0 && h32_out != 0 && hn_out != 0,
              "NaN-polled sandwich tail pointers must be non-null");
  TORCH_CHECK(q == 8, "the NaN-polled sandwich tail is Q8-only, got ", q);
  TORCH_CHECK(rearm_rows >= 1 && rearm_rows <= q, "re-armed rows must be in 1..", q,
              ", got ", rearm_rows);
  TORCH_CHECK((mc | rearm | residual32 | w_ao | w_ffn | h32_out) % 16 == 0 &&
              (hn_out & 7) == 0, "NaN-polled sandwich tail operands must be aligned");
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t config = nanpoll_config(q, 0, &attr);
  // The fp32-residual form; the BF16 residual and outputs, the signal pads, the rank and
  // the world size are unused.
  const cudaError_t err = cudaLaunchKernelEx(
      &config, nanpoll_sandwich_tail<true>, reinterpret_cast<const __nv_bfloat16*>(mc),
      static_cast<const __nv_bfloat16*>(nullptr), reinterpret_cast<const float*>(w_ao),
      reinterpret_cast<const float*>(w_ffn), reinterpret_cast<const float*>(scale),
      reinterpret_cast<const float*>(eps), static_cast<__nv_bfloat16*>(nullptr),
      reinterpret_cast<__nv_bfloat16*>(hn_out), static_cast<float*>(nullptr),
      static_cast<unsigned int**>(nullptr), 0, 8, static_cast<int>(q),
      reinterpret_cast<const float*>(residual32), reinterpret_cast<float*>(h32_out),
      reinterpret_cast<__nv_bfloat16*>(rearm), static_cast<int>(rearm_rows));
  TORCH_CHECK(err == cudaSuccess, "NaN-polled sandwich tail launch: ",
              cudaGetErrorString(err));
}

void nanpoll_layer0_tail_launch(int64_t mc, int64_t rearm, int64_t rearm_rows,
                                int64_t residual, int64_t w_ao, int64_t w_ffn,
                                int64_t scale, int64_t eps, int64_t h_out, int64_t hn_out,
                                int64_t q) {
  TORCH_CHECK(mc != 0 && rearm != 0 && residual != 0 && h_out != 0 && hn_out != 0,
              "NaN-polled layer 0 tail pointers must be non-null");
  TORCH_CHECK(q == 8, "the NaN-polled layer 0 tail is Q8-only, got ", q);
  TORCH_CHECK(rearm_rows >= 1 && rearm_rows <= q, "re-armed rows must be in 1..", q,
              ", got ", rearm_rows);
  TORCH_CHECK((mc | rearm | w_ao | w_ffn) % 16 == 0 && ((residual | h_out | hn_out) & 7) == 0,
              "NaN-polled layer 0 tail operands must be aligned");
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t config = nanpoll_config(q, 0, &attr);
  // The BF16 form, as `ar_tail_direct1bar_launch` ships it to layer 0; the fp32
  // operands, the signal pads, the rank and the world size are unused.
  const cudaError_t err = cudaLaunchKernelEx(
      &config, nanpoll_sandwich_tail<false>, reinterpret_cast<const __nv_bfloat16*>(mc),
      reinterpret_cast<const __nv_bfloat16*>(residual), reinterpret_cast<const float*>(w_ao),
      reinterpret_cast<const float*>(w_ffn), reinterpret_cast<const float*>(scale),
      reinterpret_cast<const float*>(eps), reinterpret_cast<__nv_bfloat16*>(h_out),
      reinterpret_cast<__nv_bfloat16*>(hn_out), static_cast<float*>(nullptr),
      static_cast<unsigned int**>(nullptr), 0, 8, static_cast<int>(q),
      static_cast<const float*>(nullptr), static_cast<float*>(nullptr),
      reinterpret_cast<__nv_bfloat16*>(rearm), static_cast<int>(rearm_rows));
  TORCH_CHECK(err == cudaSuccess, "NaN-polled layer 0 tail launch: ",
              cudaGetErrorString(err));
}
"""
_CPP = """
void nanpoll_attention_tail_launch(int64_t mc, int64_t rearm, int64_t fold_mc,
                                   int64_t fold_rearm, int64_t residual, int64_t w_ao,
                                   int64_t w_ffn, int64_t scale, int64_t eps,
                                   int64_t h_out, int64_t inv_out, int64_t x_pad,
                                   int64_t x_pad_fp8, int64_t sel, int64_t wt, int64_t q);
void nanpoll_mlp_tail_launch(int64_t mc, int64_t rearm, int64_t residual, int64_t scale,
                             int64_t w_norm, int64_t eps, int64_t out, int64_t norm_out,
                             int64_t rearm_rows, int64_t q, bool final_norm);
void nanpoll_eh_tail_launch(int64_t mc, int64_t rearm, int64_t w_norm, int64_t eps,
                            int64_t raw_out, int64_t norm_out, int64_t q);
void nanpoll_draft_attention_tail_launch(int64_t mc, int64_t rearm, int64_t residual,
                                         int64_t w_ao, int64_t w_ffn, int64_t scale,
                                         int64_t eps, int64_t h32_out, int64_t inv_out,
                                         int64_t x_pad, int64_t x_pad_fp8, int64_t q);
void nanpoll_sandwich_tail_launch(int64_t mc, int64_t rearm, int64_t rearm_rows,
                                  int64_t residual32, int64_t w_ao, int64_t w_ffn,
                                  int64_t scale, int64_t eps, int64_t h32_out,
                                  int64_t hn_out, int64_t q);
void nanpoll_layer0_tail_launch(int64_t mc, int64_t rearm, int64_t rearm_rows,
                                int64_t residual, int64_t w_ao, int64_t w_ffn,
                                int64_t scale, int64_t eps, int64_t h_out, int64_t hn_out,
                                int64_t q);
"""


@functools.lru_cache(None)
def consumer_source() -> str:
    """`ar_prefetch`'s source plus the NaN-polled tails, derived from its own."""
    src = ar_prefetch.source()
    attention = _derive(src, **_ATTENTION, new_read=_attention_read())
    mlp = _mlp_early(_derive(src, **_MLP, new_read=""))
    eh = _derive(src, **_EH, replace=_eh_replace())
    draft_attention = _derive(src, **_DRAFT_ATTENTION,
                              new_read=_one_vector("row_base + 8 * tid", "true"))
    sandwich = _derive(src, **_SANDWICH, new_read=_one_vector(
        "static_cast<size_t>(row) * H + 8 * tid", "row < rearm_rows"))
    return (src + _HELPERS + _FOLD_HELPERS + _SECOND_HELPERS + attention + mlp + eh
            + draft_attention + sandwich + _LAUNCH)


@functools.lru_cache(None)
def consumer_extension():
    with cuda_arch():
        return load_inline(name="iquestrt_nanpoll_tails_v14", cpp_sources=_CPP,
                           cuda_sources=consumer_source(),
                           functions=["nanpoll_attention_tail_launch", "nanpoll_mlp_tail_launch",
                                      "nanpoll_eh_tail_launch",
                                      "nanpoll_draft_attention_tail_launch",
                                      "nanpoll_sandwich_tail_launch",
                                      "nanpoll_layer0_tail_launch"],
                           extra_cuda_cflags=["-O3"], verbose=False)


def _armed_buffers(ws, key, rows: tuple[int,...], group: str):
    """`[len(rows), ROWS, H]` symmetric BF16, buffer i's rows `:rows[i]` armed, the rest
    zero, and the multicast address of each buffer."""
    import torch.distributed._symmetric_memory as symm_mem

    count = len(rows)
    buffers = ws.slot(("nanpoll", key), (count, ROWS, H), torch.bfloat16, symm=True)
    buffers.zero_()
    for buffer, r in zip(buffers, rows):
        buffer[:r].view(torch.int16).fill_(ARMED)
    handle = symm_mem.rendezvous(buffers, group)
    if not handle.multicast_ptr:
        raise RuntimeError("no multicast pointer; NVLS is unavailable")
    stride = ROWS * H * buffers.element_size()
    return buffers, tuple(int(handle.multicast_ptr) + i * stride for i in range(count))


def _armed_fold_slots(ws, group: str):
    """`[ATTENTION_BUFFERS, Q, FOLD_ROWS]` symmetric fp32, every element armed, and the
    multicast address of each slot."""
    import torch.distributed._symmetric_memory as symm_mem

    slots = ws.slot(("nanpoll", "fold"), (ATTENTION_BUFFERS, Q, FOLD_ROWS), torch.float32,
                    symm=True)
    slots.view(torch.int32).fill_(FOLD_ARMED)
    handle = symm_mem.rendezvous(slots, group)
    if not handle.multicast_ptr:
        raise RuntimeError("no multicast pointer; NVLS is unavailable")
    stride = Q * FOLD_ROWS * slots.element_size()
    return slots, tuple(int(handle.multicast_ptr) + i * stride
                        for i in range(ATTENTION_BUFFERS))


def _setup(what: str, build) -> None:
    """Run `build` on every rank; raise on all of them if any failed; then wait until
    every rank's buffers are armed, before any rank's producer writes into them."""
    rank, device = dist.get_rank(), torch.device("cuda", torch.cuda.current_device())
    error = None
    try:
        if dist.get_world_size() != WORLD:
            raise ValueError(f"supports world={WORLD}, got {dist.get_world_size()}")
        build()
    except Exception as exc:  # noqa: BLE001 - every rank raises, none waits below
        error = f"rank {rank}: {type(exc).__name__}: {exc}"
    ar_tail._raise_if_any_rank_failed(error, device=device, what=what)
    torch.cuda.synchronize()
    dist.barrier()


def _check_tail(tail: TailKernels) -> None:
    if tail.hidden != H or tail.q_len != Q:
        raise ValueError(f"supports hidden={H}, Q={Q}; got {tail.hidden}, {tail.q_len}")


class NanPolledAttentionTail:
    """One rank's NaN-armed OProj buffers and fold slots, its OProj and polling attention
    tails."""

    def __init__(self, ws, group: str, q_dim: int, *, tail: TailKernels, layers: int,
                 full_layers: Iterable[int], fold: dict, rank: int):
        self.tail, self.q_dim, self.layers = tail, q_dim, layers
        self.full_layers = frozenset(full_layers)
        self.fold, self.rank = fold, rank
        self.device = torch.device("cuda", torch.cuda.current_device())
        # Per layer, the residual base its OProj folded, which its tail must take; the
        # layer the tail last routed.
        self._folded, self._routed = {}, None

        def build():
            _check_tail(tail)
            check_attention_count(layers - 1)
            if q_dim != oproj_middle.K:
                raise ValueError(f"the OProj is built for q_dim={oproj_middle.K}, got {q_dim}")
            if sorted(fold) != list(range(1, layers)):
                raise ValueError(f"fold weights are for the MoE layers 1..{layers - 1}, got "
                                 f"{sorted(fold)}")
            for layer, weights in fold.items():
                ar_tail._check_cuda_tensor(
                    f"layer {layer}'s fold weights", weights, dtype=torch.bfloat16,
                    device=self.device, numel=FOLD_ROWS * oproj_middle.FOLD_K)
            if not 0 <= rank < WORLD:
                raise ValueError(f"rank must be in 0..{WORLD - 1}, got {rank}")
            self.buffers, self._mc = _armed_buffers(ws, "attention",
                                                    (Q,) * (ATTENTION_BUFFERS + 1), group)
            self.fold_slots, self._fold_mc = _armed_fold_slots(ws, group)
            self.sel = torch.zeros(Q * TOPK, dtype=torch.int32, device=self.device)
            self.wt = torch.zeros(Q * TOPK, dtype=torch.float32, device=self.device)
            # (fold, spread) of each layer's OProj: layer 0 plain, the MoE layers folded.
            forms = {(layer > 0, layer in self.full_layers) for layer in range(layers)}
            self._oproj = {(f, spread): oproj_middle.build_oproj_middle_block(
                spread=spread, release_first=not spread, fold=f) for f, spread in forms}
            self._ext = consumer_extension()

        _setup("NaN-polled attention tail setup", build)

    def _layer(self, layer: int) -> int:
        if not 0 <= layer < self.layers:
            raise ValueError(f"main layers are 0..{self.layers - 1}, got {layer}")
        return attention_buffer(layer)

    def _previous(self, layer: int) -> torch.Tensor:
        """The buffer the tail of `layer` re-arms: the previous layer's, the last one's
        at layer 0."""
        return self.buffers[attention_buffer((layer - 1) % self.layers)]

    def _previous_fold(self, layer: int) -> torch.Tensor:
        """The fold slot the tail of MoE layer `layer` re-arms: the previous MoE layer's,
        the last layer's at layer 1."""
        return self.fold_slots[fold_buffer(layer - 1 if layer > 1 else self.layers - 1)]

    def oproj(self, layer: int, ao, w_o, w_scale, residual=None) -> torch.Tensor:
        """This rank's OProj partial of `ao` [Q, q_dim] into `layer`'s buffer -> its
        [Q, H] view; at a MoE layer, and only there, the fold of `residual` [Q, H] BF16,
        the layer's residual base, and of `ao` into its fold slot."""
        out = self.buffers[self._layer(layer)]
        kernel = self._oproj[(layer > 0, layer in self.full_layers)]
        x = ao.reshape(Q, self.q_dim)
        if layer == 0:
            if residual is not None:
                raise ValueError("layer 0's OProj folds no router")
            kernel(x, w_o, w_scale, out)
            return out[:Q]
        ar_tail._check_cuda_tensor("residual", residual, dtype=torch.bfloat16,
                                   device=self.device, numel=Q * H)
        kernel(x, w_o, w_scale, out, residual.reshape(Q, H), self.fold[layer],
               self.fold_slots[fold_buffer(layer)], self.rank)
        self._folded[layer] = residual.data_ptr()
        return out[:Q]

    def _check(self, residual, w_ao, w_ffn) -> None:
        check = ar_tail._check_cuda_tensor
        check("residual", residual, dtype=torch.bfloat16, device=self.device, numel=Q * H)
        for name, weight in (("w_ao", w_ao), ("w_ffn", w_ffn)):
            check(name, weight, dtype=torch.float32, device=self.device, numel=H)

    def h_inv(self, layer: int, residual, w_ao, w_ffn, inv_out, x_pad, x_pad_fp8,
              a_scale: float, f_scale: float) -> torch.Tensor:
        """The attention tail of MoE layer `layer` on its partials -> `tail.h` [Q, H];
        it routes the layer into `sel` and `wt`."""
        buffer = self._layer(layer)
        if layer == 0:
            raise ValueError("layer 0's attention tail is `layer0`")
        self._check(residual, w_ao, w_ffn)
        if self._folded.get(layer) != residual.data_ptr():
            raise ValueError(f"layer {layer}'s tail must take the residual base its OProj "
                             "folded")
        check = ar_tail._check_cuda_tensor
        check("inv_out", inv_out, dtype=torch.float32, device=self.device, numel=Q)
        check("x_pad", x_pad, dtype=torch.bfloat16, device=self.device, numel=16 * H)
        check("x_pad_fp8", x_pad_fp8, dtype=torch.float8_e4m3fn, device=self.device,
              numel=16 * H)
        tail = self.tail
        tail._prep(a_scale, f_scale)
        self._ext.nanpoll_attention_tail_launch(
            self._mc[buffer], self._previous(layer).data_ptr(),
            self._fold_mc[fold_buffer(layer)], self._previous_fold(layer).data_ptr(),
            residual.data_ptr(), w_ao.data_ptr(), w_ffn.data_ptr(), tail.a_scale.data_ptr(),
            tail.eps.data_ptr(), tail.h.data_ptr(), inv_out.data_ptr(), x_pad.data_ptr(),
            x_pad_fp8.data_ptr(), self.sel.data_ptr(), self.wt.data_ptr(), Q)
        self._routed = layer
        return tail.h.view(Q, H)

    def routing(self, layer: int):
        """MoE layer `layer`'s (`sel` [Q*TOPK] int32, ascending within each row; `wt`
        [Q*TOPK] fp32), as `RouterSelect` returns them: what its tail, the last one to
        run, selected."""
        if layer != self._routed:
            raise ValueError(f"layer {layer} is routed by its own attention tail, but the "
                             f"last tail was layer {self._routed}'s")
        return self.sel, self.wt

    def layer0(self, residual, w_ao, w_ffn, a_scale: float, f_scale: float):
        """Layer 0's attention tail on its partials -> (`tail.h`, `tail.hn`) [Q, H]."""
        self._check(residual, w_ao, w_ffn)
        tail = self.tail
        tail._prep(a_scale, f_scale)
        self._ext.nanpoll_layer0_tail_launch(
            self._mc[attention_buffer(0)], self._previous(0).data_ptr(), Q,
            residual.data_ptr(), w_ao.data_ptr(), w_ffn.data_ptr(), tail.a_scale.data_ptr(),
            tail.eps.data_ptr(), tail.h.data_ptr(), tail.hn.data_ptr(), Q)
        return tail.h.view(Q, H), tail.hn.view(Q, H)


class NanPolledMLPTail:
    """One rank's NaN-armed MLP output buffers, its producers and polling tails."""

    def __init__(self, ws, group: str, *, moe, dense, widths, tail: TailKernels,
                 norm_out: torch.Tensor, layers: int):
        self.moe_kernels, self.dense_mlp = moe, dense
        self.widths = tuple(widths)
        self.live = max(self.widths, default=0)
        self.tail, self.norm_out, self.layers = tail, norm_out, layers
        self.device = torch.device("cuda", torch.cuda.current_device())

        def build():
            _check_tail(tail)
            if not self.widths or not all(1 <= w <= Q for w in self.widths):
                raise ValueError(f"live rows must be in 1..{Q}, got {self.widths}")
            ar_tail._check_cuda_tensor("norm_out", norm_out, dtype=torch.bfloat16,
                                       device=self.device, numel=Q * H)
            self.buffers, self._mc = {}, {}
            for w in self.widths:
                self.buffers[w], self._mc[w] = _armed_buffers(ws, f"mlp_w{w}", self.rows(w),
                                                              group)
            self.final_out = torch.empty(Q, H, dtype=torch.bfloat16, device=self.device)
            self._gates = {w: build_gate_silu_block(w)[0] for w in self.widths}
            self._downs = {w: ReleasingDown(w) for w in self.widths}
            self._dense_down = build_gemv_block(dense.inter, H, **DENSE_BLOCK_CFG,
                                                early=True)[0]
            self._ext = consumer_extension()

        _setup("NaN-polled MLP tail setup", build)

    @staticmethod
    def rows(width: int) -> tuple[int,...]:
        """The rows each of a width's buffers' producers write, which its re-arm restores."""
        return (width,) * MLP_BUFFERS + (Q,)

    def _width(self) -> int:
        if self.live not in self.widths:
            raise ValueError(f"no MLP buffers for width {self.live}; built {self.widths}")
        return self.live

    def _layer(self, layer: int) -> int:
        if not 0 <= layer < self.layers:
            raise ValueError(f"main layers are 0..{self.layers - 1}, got {layer}")
        return mlp_buffer(layer)

    def _previous(self, layer: int) -> int:
        """The buffer the tail of `layer` re-arms: the previous layer's, the last one's
        at layer 0."""
        return mlp_buffer((layer - 1) % self.layers)

    def _check_view(self, mlp_out, layer: int) -> None:
        if (mlp_out.data_ptr() != self.buffers[self._width()][self._layer(layer)].data_ptr()
                or tuple(mlp_out.shape) != (Q, H)):
            raise ValueError(f"mlp_out must be the [:Q] view of main layer {layer}'s buffer")

    def dense(self, hn, w_gate_up, gate_up_scale, w_down, down_scale) -> torch.Tensor:
        """Layer 0's dense MLP of `hn` into its buffer -> its [Q, H] view."""
        out = self.buffers[self._width()][mlp_buffer(0)]
        self._dense_down(self.dense_mlp.act(hn, w_gate_up, gate_up_scale), w_down,
                         down_scale, out)
        return out[:Q]

    def sandwich(self, mlp_out, residual, w_ao, w_ffn, a_scale: float, f_scale: float):
        """Layer 0's sandwich tail on `mlp_out`, its `dense` result -> (`tail.h`,
        `tail.hn`) [Q, H]."""
        self._check_view(mlp_out, 0)
        check = ar_tail._check_cuda_tensor
        check("residual", residual, dtype=torch.bfloat16, device=self.device, numel=Q * H)
        for name, weight in (("w_ao", w_ao), ("w_ffn", w_ffn)):
            check(name, weight, dtype=torch.float32, device=self.device, numel=H)
        previous, w = self._previous(0), self._width()
        tail = self.tail
        tail._prep(a_scale, f_scale)
        self._ext.nanpoll_layer0_tail_launch(
            self._mc[w][mlp_buffer(0)], self.buffers[w][previous].data_ptr(),
            self.rows(w)[previous],
            residual.data_ptr(), w_ao.data_ptr(), w_ffn.data_ptr(), tail.a_scale.data_ptr(),
            tail.eps.data_ptr(), tail.h.data_ptr(), tail.hn.data_ptr(), Q)
        return tail.h.view(Q, H), tail.hn.view(Q, H)

    def moe(self, layer: int, sel, wt, fc, fc_scale, proj, proj_scale) -> torch.Tensor:
        """MoE layer `layer`'s MoE over the live rows into its buffer -> [Q, H] view."""
        if layer == 0:
            raise ValueError("layer 0's MLP is `dense`")
        w = self._width()
        out = self.buffers[w][self._layer(layer)]
        m = self.moe_kernels
        self._gates[w](m.x_pad_fp8, fc, fc_scale, sel, m.act_fp8)
        self._downs[w](m.act_fp8, proj, proj_scale, sel, wt, out)
        return out[:Q]

    def __call__(self, layer: int, mlp_out, residual, f_scale: float, w_norm):
        """The MLP tail on `mlp_out`, MoE layer `layer`'s `moe` result -> (`tail.out`,
        `norm_out`) [Q, H]; after the last layer (`tail.out`, `final_out`)."""
        if layer == 0:
            raise ValueError("layer 0's MLP tail is `sandwich`")
        self._check_view(mlp_out, layer)
        check = ar_tail._check_cuda_tensor
        check("residual", residual, dtype=torch.bfloat16, device=self.device, numel=Q * H)
        check("w_norm", w_norm, dtype=torch.float32, device=self.device, numel=H)
        previous, w = self._previous(layer), self._width()
        final = layer == self.layers - 1
        norm_out = self.final_out if final else self.norm_out
        tail = self.tail
        a_scale = tail._scales[0]
        tail._prep(1.0 if a_scale is None else a_scale, f_scale)
        self._ext.nanpoll_mlp_tail_launch(
            self._mc[w][mlp_buffer(layer)], self.buffers[w][previous].data_ptr(),
            residual.data_ptr(), tail.f_scale.data_ptr(), w_norm.data_ptr(),
            tail.eps.data_ptr(), tail.out.data_ptr(), norm_out.data_ptr(),
            self.rows(w)[previous], Q, final)
        return tail.out.view(Q, H), norm_out.view(Q, H)


EH_K = 768  # the eh_proj's input columns per rank at TP8 (`MtpEhARNorm`)


class NanPolledDraftTails:
    """One rank's NaN-armed buffers for the draft's three all-reduces, one per depth and
    kind, their producers and their polling tails."""

    def __init__(self, ws, group: str, *, depths: int, q_dim: int, tail: TailKernels, moe,
                 eh):
        self.depths, self.q_dim, self.tail = depths, q_dim, tail
        self.moe_kernels, self.eh_holder = moe, eh
        self.rows = draft_rows(depths)
        self.turns = {kind: DepthTurns(depths) for kind in ("eh", "attention", "sandwich")}
        self.device = torch.device("cuda", torch.cuda.current_device())

        def build():
            _check_tail(tail)
            self.eh_buffers, self._eh_mc = _armed_buffers(ws, "draft_eh", (Q,) * depths,
                                                          group)
            self.attention_buffers, self._attention_mc = _armed_buffers(
                ws, "draft_attention", (Q,) * depths, group)
            self.sandwich_buffers, self._sandwich_mc = _armed_buffers(
                ws, "draft_sandwich", self.rows, group)
            self._eh = build_oproj_gemv_q8_transposed(EH_K, H, **EH_PROJ_CFG, early=True)[0]
            self._oproj = build_gemv_block(q_dim, H, **OPROJ_BLOCK_CFG, early=True)[0]
            self._gates = {r: build_gate_silu_block(r)[0] for r in set(self.rows)}
            self._downs = {r: ReleasingDown(r) for r in set(self.rows)}
            self._ext = consumer_extension()

        _setup("NaN-polled draft tails setup", build)

    def reset_turns(self, last: int | None = None) -> None:
        """Every kind's next call re-arms depth `last`'s buffer, as after a step whose last
        depth was `last` (default the last of `depths`): before a host graph's depth 0 is
        captured, and after a decode, as the device left them."""
        for turns in self.turns.values():
            turns.reset(last)

    def _check_view(self, what: str, view, buffers, depth: int) -> None:
        if view.data_ptr() != buffers[depth].data_ptr() or tuple(view.shape) != (Q, H):
            raise ValueError(f"{what} must be the [:Q] view of draft depth {depth}'s buffer")

    def eh(self, depth: int, eh_in, w_eh, w_norm):
        """Depth `depth`'s eh_proj and eh tail -> the holder's (`raw_h0`, `norm_out`)."""
        self.eh_proj(depth, eh_in, w_eh)
        return self.eh_tail(depth, w_norm)

    def eh_proj(self, depth: int, eh_in, w_eh) -> None:
        """Depth `depth`'s eh_proj partial of `eh_in` into its buffer."""
        check = ar_tail._check_cuda_tensor
        check("eh_in", eh_in, dtype=torch.bfloat16, device=self.device, numel=Q * EH_K)
        check("w_eh", w_eh, dtype=torch.bfloat16, device=self.device, numel=H * EH_K)
        self._eh(eh_in.reshape(Q, EH_K), w_eh, self.eh_target(depth))

    def eh_target(self, depth: int) -> torch.Tensor:
        """Depth `depth`'s buffer for its eh_proj partial: this holder's GEMV writes it, or
        the draft geometry's launch that runs the GEMV (`geometry.py:draft_entry_eh_func`),
        before `eh_tail`."""
        self.turns["eh"].check(depth)
        return self.eh_buffers[depth]

    def eh_tail(self, depth: int, w_norm):
        """The eh tail on depth `depth`'s `eh_proj` -> the holder's (`raw_h0`, `norm_out`)."""
        ar_tail._check_cuda_tensor("w_norm", w_norm, dtype=torch.float32, device=self.device,
                                   numel=H)
        previous = self.turns["eh"].take(depth)
        h = self.eh_holder
        self._ext.nanpoll_eh_tail_launch(
            self._eh_mc[depth], self.eh_buffers[previous].data_ptr(), w_norm.data_ptr(),
            h.eps.data_ptr(), h.raw_h0.data_ptr(), h.norm_out.data_ptr(), Q)
        return h.raw_h0.view(Q, H), h.norm_out.view(Q, H)

    def oproj(self, depth: int, ao, w_o, w_scale) -> torch.Tensor:
        """Depth `depth`'s OProj partial of `ao` into its buffer -> its [Q, H] view."""
        self.turns["attention"].check(depth)
        out = self.attention_buffers[depth]
        self._oproj(ao.reshape(Q, self.q_dim), w_o, w_scale, out)
        return out[:Q]

    def h_inv(self, depth: int, ao, residual, w_ao, w_ffn, inv_out, x_pad, x_pad_fp8,
              a_scale: float, f_scale: float) -> torch.Tensor:
        """The attention tail on `ao`, depth `depth`'s `oproj` -> `tail.h32` [Q, H]."""
        self.turns["attention"].check(depth)
        self._check_view("ao", ao, self.attention_buffers, depth)
        check = ar_tail._check_cuda_tensor
        check("residual", residual, dtype=torch.bfloat16, device=self.device, numel=Q * H)
        for name, weight in (("w_ao", w_ao), ("w_ffn", w_ffn)):
            check(name, weight, dtype=torch.float32, device=self.device, numel=H)
        check("inv_out", inv_out, dtype=torch.float32, device=self.device, numel=Q)
        check("x_pad", x_pad, dtype=torch.bfloat16, device=self.device, numel=16 * H)
        check("x_pad_fp8", x_pad_fp8, dtype=torch.float8_e4m3fn, device=self.device,
              numel=16 * H)
        previous = self.turns["attention"].take(depth)
        tail = self.tail
        tail._prep(a_scale, f_scale)
        self._ext.nanpoll_draft_attention_tail_launch(
            self._attention_mc[depth], self.attention_buffers[previous].data_ptr(),
            residual.data_ptr(), w_ao.data_ptr(), w_ffn.data_ptr(), tail.a_scale.data_ptr(),
            tail.eps.data_ptr(), tail.h32.data_ptr(), inv_out.data_ptr(), x_pad.data_ptr(),
            x_pad_fp8.data_ptr(), Q)
        return tail.h32.view(Q, H)

    def moe(self, depth: int, sel, wt, fc, fc_scale, proj, proj_scale, *,
            rows: int) -> torch.Tensor:
        """Depth `depth`'s MoE over its rows into its buffer -> its [Q, H] view."""
        self.turns["sandwich"].check(depth)
        if rows != self.rows[depth]:
            raise ValueError(f"draft depth {depth}'s MoE runs over {self.rows[depth]} rows, "
                             f"got rows={rows}")
        out = self.sandwich_buffers[depth]
        m = self.moe_kernels
        self._gates[rows](m.x_pad_fp8, fc, fc_scale, sel, m.act_fp8)
        self._downs[rows](m.act_fp8, proj, proj_scale, sel, wt, out)
        return out[:Q]

    def sandwich(self, depth: int, mlp_out, residual, w_ao, w_ffn, a_scale: float,
                 f_scale: float):
        """The sandwich tail on `mlp_out`, depth `depth`'s `moe`, with the fp32 `residual`
        -> (`tail.out32`, `tail.hn`) [Q, H]."""
        self.turns["sandwich"].check(depth)
        self._check_view("mlp_out", mlp_out, self.sandwich_buffers, depth)
        check = ar_tail._check_cuda_tensor
        check("residual", residual, dtype=torch.float32, device=self.device, numel=Q * H)
        for name, weight in (("w_ao", w_ao), ("w_ffn", w_ffn)):
            check(name, weight, dtype=torch.float32, device=self.device, numel=H)
        previous = self.turns["sandwich"].take(depth)
        tail = self.tail
        tail._prep(a_scale, f_scale)
        self._ext.nanpoll_sandwich_tail_launch(
            self._sandwich_mc[depth], self.sandwich_buffers[previous].data_ptr(),
            self.rows[previous], residual.data_ptr(), w_ao.data_ptr(), w_ffn.data_ptr(),
            tail.a_scale.data_ptr(), tail.eps.data_ptr(), tail.out32.data_ptr(),
            tail.hn.data_ptr(), Q)
        return tail.out32.view(Q, H), tail.hn.view(Q, H)
