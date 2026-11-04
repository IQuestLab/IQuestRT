"""The main stack's prenorm QKV: weights staged before the PDL wait, early release."""

from iquestrt.kernels.q8.codegen import compile_e4m3
from iquestrt.kernels.prologue import BM, HIDDEN, KB, KSPLIT, N_QKV, NO_WS, Q, T


def build_qkv_middle_block():
    """`prologue.build_qkv_block` with the whole split's weights converted into shared
    memory before the wait."""
    bn, bk, stages, threads = 32, 256, 3, 128
    nt = N_QKV // bn
    kt_s = HIDDEN // bk // KSPLIT
    sub = bk // KB
    kb_s = kt_s * sub

    @T.prim_func
    def prologue_qkv_prenorm_block_q(
        xn_in: T.Tensor((Q, HIDDEN), "bfloat16"),
        w_qkv: T.Tensor((N_QKV, HIDDEN), T.float8_e4m3fn),
        w_scale: T.Tensor((N_QKV // KB, HIDDEN // KB), "float32"),
        qkv_out: T.Tensor((KSPLIT, BM, N_QKV), "float32"),
    ):
        with T.Kernel(nt, KSPLIT, threads=threads) as (bx, bs):
            scale = T.alloc_local((kb_s,), "float32")
            for b in T.serial(kb_s):
                scale[b] = w_scale[bx * bn // KB, bs * kb_s + b]
            w_all = T.alloc_shared((bn, bk * kt_s), "bfloat16")
            for kb in T.unroll(kt_s):
                for i, j in T.Parallel(bn, bk):
                    w_all[i, kb * bk + j] = T.Cast(
                        "bfloat16", w_qkv[bx * bn + i, (bs * kt_s + kb) * bk + j])
            T.pdl_trigger()
            T.pdl_sync()
            x_t = T.alloc_shared((BM, bk), "bfloat16")
            part = T.alloc_fragment((BM, bn), "float32")
            acc = T.alloc_fragment((BM, bn), "float32")
            T.clear(acc)
            for ki in T.Pipelined(kt_s, num_stages=stages):
                for i, j in T.Parallel(BM, bk):
                    x_t[i, j] = xn_in[T.min(i, Q - 1), (bs * kt_s + ki) * bk + j]
                for s in T.serial(sub):
                    T.clear(part)
                    T.gemm(x_t[0:BM, s * KB : (s + 1) * KB],
                           w_all[0:bn, ki * bk + s * KB : ki * bk + (s + 1) * KB],
                           part, transpose_B=True)
                    for i, j in T.Parallel(BM, bn):
                        acc[i, j] += part[i, j] * scale[ki * sub + s]
            T.copy(acc, qkv_out[bs, 0, bx * bn])

    return compile_e4m3(prologue_qkv_prenorm_block_q, out_idx=None, pass_configs=NO_WS)
