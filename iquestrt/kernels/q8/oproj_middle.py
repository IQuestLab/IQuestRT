"""Stage immutable OProj weights before the PDL wait and emit folded router partials.

Activation reads must follow the wait. Fold buffers rotate with attention buffers;
their FP32 algebra differs from routing on rounded BF16 activations."""

import contextlib

import tilelang
import torch

from iquestrt.blockfp8 import BLOCK, dequant
from iquestrt.kernels.q8.codegen import compile_e4m3, postprocessed, replace_once
from iquestrt.kernels.oproj import BM, KB, NO_WS, OPROJ_BLOCK_CFG, T
from iquestrt.kernels.prologue import HEAD_DIM, HIDDEN, Q, Q_HEADS

K, N = Q_HEADS * HEAD_DIM, HIDDEN    # the rank's attention width in, the hidden width out
WORLD, EXPERTS = 8, 256
KS = N // WORLD                      # a rank's columns of the residual base in B
FOLD_A_CTAS = FOLD_B_CTAS = EXPERTS // 8   # 8 experts a fold CTA
FOLD_ROWS = 2 * EXPERTS              # a fold weight row per expert half; a slot row's width
FOLD_K = K + KS                      # a fold weight row: M_r's K columns, then W'_r's KS
OPROJ_CTAS = N // OPROJ_BLOCK_CFG["bn"]   # the OProj's CTAs; the fold CTAs follow them

# The full layers' reservation. The empty asm takes its address, so
# nvcc keeps it, and emits no instruction.
SPREAD = ("  __shared__ __align__(16) uchar iquestrt_spread[30 * 1024];\n"
          "  asm volatile(\"\" :: \"r\"((unsigned)__cvta_generic_to_shared(iquestrt_spread)));\n")


def _spread(code):
    anchor = "  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];\n"
    return replace_once(code, anchor, anchor + SPREAD)


# The fold CTAs. Thread (warp, lane) of a CTA takes MMA rows and
# columns g = lane / 4 and q = lane % 4. Its 16-byte chunk u of an operand row holds
# columns 32u + 8q.. 32u + 8q + 7, which `chunk` feeds to k-steps 2u and 2u + 1 as their
# k 2q, 2q + 1, 2q + 8 and 2q + 9: the same map for the attention or residual row g it
# reads and for the weight row 16 tile + 8 warp + g, so every column's product is summed
# once. Its accumulator columns 2q and 2q + 1 are then the high and low halves of expert
# 8 tile + 4 warp + q, for row g; rows 8..15 of the MMA are zero. An A CTA stages the
# attention rows YS chunks apart in shared memory: a quarter warp's eight reads, two
# rows' chunks q + 4u, then fall in distinct banks.
FOLD = rf"""
// ---------------------------------------------------------------- the router fold
namespace iquestrt_fold {{

constexpr int K = {K}, KS = {KS}, H = {N}, E = {EXPERTS}, ROW = {FOLD_ROWS}, WK = {FOLD_K};
constexpr int NT = {OPROJ_CTAS}, A_CTAS = {FOLD_A_CTAS};
constexpr int Q = {Q}, THREADS = {OPROJ_BLOCK_CFG["threads"]}, YS = K / 8 + 4;
static_assert(Q * (K / 8) % THREADS == 0, "the staged rows' chunks over the CTA's threads");
""" + r"""
__device__ __forceinline__ void mma(float (&c)[4], unsigned a0, unsigned a2, unsigned b0,
                                    unsigned b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void chunk(float (&c)[4], const uint4& x, const uint4& w) {
  mma(c, x.x, x.y, w.x, w.y);
  mma(c, x.z, x.w, w.z, w.w);
}

// The thread's expert from its four accumulator chains: high half, low half, their sum.
__device__ __forceinline__ float expert(const float (&c)[4][4]) {
  const float hi = (c[0][0] + c[1][0]) + (c[2][0] + c[3][0]);
  const float lo = (c[0][1] + c[1][1]) + (c[2][1] + c[3][1]);
  return hi + lo;
}

// CTA `bx` past the OProj's: A for experts 8 tile.. from the attention rows `y`, or B
// from this rank's columns of the residual base `xn`, into row g of the fold slot `out`.
template <bool ReleaseFirst>
__device__ __forceinline__ void cta(const void* y, const void* xn, const void* w,
                                    const float* out_, int rank, int bx) {
  // TileLang declares the slot const: this kernel only passes it here.
  float* out = const_cast<float*>(out_);
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, q = lane & 3;
  const bool a = bx - NT < A_CTAS;
  const int tile = a ? bx - NT : bx - NT - A_CTAS;
  const int col = 8 * tile + 4 * warp + q;
  const uint4* wrow = reinterpret_cast<const uint4*>(
      static_cast<const __nv_bfloat16*>(w) + static_cast<size_t>(16 * tile + 8 * warp + g) * WK
      + 8 * q);
  float acc[4][4] = {};
  if (!a) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const uint4* xrow = reinterpret_cast<const uint4*>(
        static_cast<const __nv_bfloat16*>(xn) + g * H + KS * rank + 8 * q);
#pragma unroll
    for (int u = 0; u < KS / 32; ++u)
      chunk(acc[u & 3], __ldcg(xrow + 4 * u), __ldg(wrow + K / 8 + 4 * u));
    out[g * ROW + E + col] = expert(acc);
    return;
  }
  uint4 wa[K / 32];
#pragma unroll
  for (int u = 0; u < K / 32; ++u) wa[u] = __ldg(wrow + 4 * u);
  if constexpr (ReleaseFirst) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    asm volatile("griddepcontrol.wait;" ::: "memory");
  } else {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  // The CTA's attention rows, once, into the dynamic shared memory every CTA of this
  // kernel reserves.
  extern __shared__ __align__(16) uint4 ys[];
#pragma unroll
  for (int i = 0; i < Q * (K / 8) / THREADS; ++i) {
    const int c = threadIdx.x + THREADS * i;
    ys[(c / (K / 8)) * YS + c % (K / 8)] = __ldcg(static_cast<const uint4*>(y) + c);
  }
  __syncthreads();
  const uint4* yrow = ys + g * YS + q;
#pragma unroll
  for (int u = 0; u < K / 32; ++u) chunk(acc[u & 3], yrow[4 * u], wa[u]);
  out[g * ROW + col] = expert(acc);
}

}  // namespace iquestrt_fold
"""
# The TileLang body's stand-in for the fold CTAs, at its entry (`build_oproj_middle_block`).
_FOLD_CALL = "  iquestrt_fold_cta(x, xn, fold_w, fold_out, rank, ((int)blockIdx.x));\n"


def _fold(release_first: bool):
    """The fold CTAs' code before the kernel, and their branch at its entry."""
    form = "true" if release_first else "false"

    def transform(code):
        i = code.index('extern "C" __global__')
        code = code[:i] + FOLD + code[i:]
        return replace_once(code, _FOLD_CALL, (
            f"  if (((int)blockIdx.x) >= {OPROJ_CTAS}) {{\n"
            f"    iquestrt_fold::cta<{form}>(x, xn, fold_w, fold_out, rank, ((int)blockIdx.x));\n"
            "    return;\n  }\n"))

    return transform


def fold_weights(lw, rank: int, world: int = WORLD) -> torch.Tensor:
    """MoE layer `lw`'s fold weights on rank `rank` -> [FOLD_ROWS, FOLD_K] BF16.

    Row 2e is expert e's high half, row 2e + 1 its low half, of
    columns 0..K-1 of M_r = W diag(w_ffn w_ao) O_r and K.. of W' over the rank's KS
    columns of the hidden width. Both are formed in fp64 from the layer's fp32 router,
    norm weights and dequantized OProj slice."""
    if world != WORLD:
        raise ValueError(f"the fold is built for {WORLD} ranks, got {world}")
    if not 0 <= rank < world:
        raise ValueError(f"rank must be in 0..{world - 1}, got {rank}")
    o = dequant(lw.o_proj, lw.o_scale, (BLOCK, BLOCK), torch.float64)
    if tuple(o.shape) != (N, K) or tuple(lw.router.shape) != (EXPERTS, N):
        raise ValueError(f"layer {lw.idx}: OProj {tuple(o.shape)} and router "
                         f"{tuple(lw.router.shape)} are not the fold's [{N}, {K}], "
                         f"[{EXPERTS}, {N}]")
    w = lw.router.double()
    w_ffn = lw.feed_forward_norm.double()
    cols = slice(rank * KS, (rank + 1) * KS)
    full = torch.cat([(w * (w_ffn * lw.attn_out_norm.double())) @ o,
                      w[:, cols] * w_ffn[cols]], dim=1)
    hi = full.to(torch.bfloat16)
    lo = (full - hi.double()).to(torch.bfloat16)
    return torch.stack([hi, lo], dim=1).reshape(FOLD_ROWS, FOLD_K).contiguous()


def _body(release_first: bool):
    """The OProj CTAs' TileLang body, shared by the plain and the fold forms."""
    k, cfg = K, OPROJ_BLOCK_CFG
    bn, bk, stages = cfg["bn"], cfg["bk"], cfg["stages"]
    kt, sub = k // bk, bk // KB

    @T.macro
    def body(bx, x, w, w_scale, out):
        scale = T.alloc_local((k // KB,), "float32")
        for b in T.serial(k // KB):
            scale[b] = w_scale[bx * bn // KB, b]
        w_all = T.alloc_shared((bn, k), "bfloat16")
        for i, j in T.Parallel(bn, k):
            w_all[i, j] = T.Cast("bfloat16", w[bx * bn + i, j])
        if release_first:
            T.pdl_trigger()
            T.pdl_sync()
        else:
            T.pdl_sync()
            T.pdl_trigger()
        x_t = T.alloc_shared((BM, bk), "bfloat16")
        part = T.alloc_fragment((BM, bn), "float32")
        acc = T.alloc_fragment((BM, bn), "float32")
        T.clear(acc)
        for ki in T.Pipelined(kt, num_stages=stages):
            for i, j in T.Parallel(BM, bk):
                x_t[i, j] = x[T.min(i, Q - 1), ki * bk + j]
            for s in T.serial(sub):
                T.clear(part)
                T.gemm(x_t[0:BM, s * KB : (s + 1) * KB],
                       w_all[0:bn, ki * bk + s * KB : ki * bk + (s + 1) * KB],
                       part, transpose_B=True)
                for i, j in T.Parallel(BM, bn):
                    acc[i, j] += part[i, j] * scale[ki * sub + s]
        for row, feature in T.Parallel(Q, bn):
            out[row, bx * bn + feature] = T.Cast("bfloat16", acc[row, feature])

    return body


def build_oproj_middle_block(spread: bool = False, release_first: bool = False,
                             fold: bool = False):
    """`oproj.build_gemv_block(768, 3072, **OPROJ_BLOCK_CFG, early=True)` with the CTA's
    whole weight tile converted into shared before the wait."""
    threads = OPROJ_BLOCK_CFG["threads"]
    body = _body(release_first)
    if not fold:
        @T.prim_func
        def oproj_middle_block(
            x: T.Tensor((Q, K), "bfloat16"),
            w: T.Tensor((N, K), T.float8_e4m3fn),
            w_scale: T.Tensor((N // KB, K // KB), "float32"),
            out: T.Tensor((BM, N), "bfloat16"),
        ):
            with T.Kernel(OPROJ_CTAS, threads=threads) as bx:
                body(bx, x, w, w_scale, out)

        configs, transforms = NO_WS, []
    else:
        @T.prim_func
        def oproj_middle_block(
            x: T.Tensor((Q, K), "bfloat16"),
            w: T.Tensor((N, K), T.float8_e4m3fn),
            w_scale: T.Tensor((N // KB, K // KB), "float32"),
            out: T.Tensor((BM, N), "bfloat16"),
            xn: T.Tensor((Q, N), "bfloat16"),
            fold_w: T.Tensor((FOLD_ROWS, FOLD_K), "bfloat16"),
            fold_out: T.Tensor((Q, FOLD_ROWS), "float32"),
            rank: T.int32,
        ):
            with T.Kernel(OPROJ_CTAS + FOLD_A_CTAS + FOLD_B_CTAS, threads=threads) as bx:
                # Replaced by the fold CTAs' branch (`_fold`), which returns.
                T.call_extern("handle", "iquestrt_fold_cta", x.data, xn.data, fold_w.data,
                              fold_out.data, rank, bx)
                body(bx, x, w, w_scale, out)

        # The fold CTAs return before the body, whose reads TileLang would otherwise
        # guard against the grid past the OProj's.
        configs = {**NO_WS,
                   tilelang.transform.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True}
        transforms = [_fold(release_first)]
    if spread:
        transforms.append(_spread)

    def transform(code):
        for t in transforms:
            code = t(code)
        return code

    with postprocessed(transform) if transforms else contextlib.nullcontext():
        return compile_e4m3(oproj_middle_block, out_idx=None, pass_configs=configs)
