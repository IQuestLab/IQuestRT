"""The attention output projection, the dense MLP and the verify LM head, as our GEMVs."""

import functools

import tilelang
import tilelang.language as T
import torch
import torch.nn.functional as F

NO_WS = {tilelang.transform.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}

BM = 16   # the output slots keep the MMA's M=16 rows; the AR tail reads the first Q
EH_PROJ_CFG = {"bn": 32, "bk": 256, "stages": 3, "threads": 64}
LMHEAD_CFG = {"bn": 160, "bk": 128, "stages": 3, "threads": 128}


@functools.lru_cache(maxsize=None)
def build_oproj_gemv_q8_transposed(
    k: int,
    n: int,
    bn: int,
    bk: int,
    stages: int,
    threads: int,
    pdl_trigger: bool = False,
    early: bool = False,
):
    """BF16 ``W[BN,K] @ X[8,K].T`` -> out[:8] bf16 (the draft's BF16 `eh_proj`).

    ``pdl_trigger`` releases the dependent AR tail after the final store. ``early``
    instead waits for the producer at entry and releases the successor immediately."""
    if early and pdl_trigger:
        raise ValueError("Q8 GEMV cannot carry both early and late PDL triggers")
    q_len = 8
    if n % bn or k % bk:
        raise ValueError(
            f"q8 transposed oproj: N={n} K={k} not divisible by bn={bn} / bk={bk}"
        )
    nt = n // bn
    kt = k // bk

    @T.prim_func
    def oproj_gemv_q8_transposed(
        x: T.Tensor((q_len, k), "bfloat16"),
        w: T.Tensor((n, k), "bfloat16"),
        out: T.Tensor((BM, n), "bfloat16"),
    ):
        with T.Kernel(nt, threads=threads) as bx:
            if early:
                T.pdl_sync()
                T.pdl_trigger()
            x_s = T.alloc_shared((q_len, bk), "bfloat16")
            w_s = T.alloc_shared((bn, bk), "bfloat16")
            acc = T.alloc_fragment((bn, q_len), "float32")
            T.clear(acc)
            for ki in T.Pipelined(kt, num_stages=stages):
                T.copy(x[0, ki * bk], x_s)
                T.copy(w[bx * bn, ki * bk], w_s)
                T.gemm(w_s, x_s, acc, transpose_B=True)
            for feature, row in T.Parallel(bn, q_len):
                out[row, bx * bn + feature] = T.Cast(
                    "bfloat16", acc[feature, row]
                )
            if pdl_trigger:
                T.pdl_trigger()

    return (
        tilelang.compile(
            oproj_gemv_q8_transposed,
            out_idx=None,
            pass_configs=NO_WS,
        ),
        nt,
    )


KB = 128
OPROJ_BLOCK_CFG = {"bn": 16, "bk": 256, "stages": 3, "threads": 64}
DENSE_BLOCK_CFG = {"bn": 32, "bk": 256, "stages": 3, "threads": 128}


@functools.lru_cache(maxsize=None)
def build_gemv_block(
    k: int,
    n: int,
    bn: int,
    bk: int,
    stages: int,
    threads: int,
    pdl_trigger: bool = False,
    early: bool = False,
):
    """``X[8,K] @ dequant(W)[BN,K].T`` -> out[:8] bf16 for block-FP8 `W` (OProj, dense MLP).

    ``pdl_trigger`` releases the dependent AR tail after the final store. ``early``
    instead waits for the producer at entry and releases the successor immediately: the
    draft OProj's and layer 0's dense Down's form as NaN-polled producers
    (`q8/nanpoll.py`)."""
    if early and pdl_trigger:
        raise ValueError("block GEMV cannot carry both early and late PDL triggers")
    q_len = 8
    if n % bn or k % bk or KB % bn or bk % KB:
        raise ValueError(f"block GEMV: N={n} K={k} do not tile by bn={bn} bk={bk} in "
                         f"{KB}-blocks")
    nt, kt, sub = n // bn, k // bk, bk // KB

    @T.prim_func
    def gemv_block(
        x: T.Tensor((q_len, k), "bfloat16"),
        w: T.Tensor((n, k), T.float8_e4m3fn),
        w_scale: T.Tensor((n // KB, k // KB), "float32"),
        out: T.Tensor((BM, n), "bfloat16"),
    ):
        with T.Kernel(nt, threads=threads) as bx:
            scale = T.alloc_local((k // KB,), "float32")
            for b in T.serial(k // KB):
                scale[b] = w_scale[bx * bn // KB, b]
            if early:
                T.pdl_sync()
                T.pdl_trigger()
            x_t = T.alloc_shared((BM, bk), "bfloat16")
            w8_s = T.alloc_shared((bn, bk), T.float8_e4m3fn)
            w_s = T.alloc_shared((bn, bk), "bfloat16")
            part = T.alloc_fragment((BM, bn), "float32")
            acc = T.alloc_fragment((BM, bn), "float32")
            T.clear(acc)
            for ki in T.Pipelined(kt, num_stages=stages):
                for i, j in T.Parallel(BM, bk):
                    x_t[i, j] = x[T.min(i, q_len - 1), ki * bk + j]
                T.copy(w[bx * bn, ki * bk], w8_s)
                for i, j in T.Parallel(bn, bk):
                    w_s[i, j] = T.Cast("bfloat16", w8_s[i, j])
                for s in T.serial(sub):
                    T.clear(part)
                    T.gemm(x_t[0:BM, s * KB : (s + 1) * KB], w_s[0:bn, s * KB : (s + 1) * KB],
                           part, transpose_B=True)
                    for i, j in T.Parallel(BM, bn):
                        acc[i, j] += part[i, j] * scale[ki * sub + s]
            for row, feature in T.Parallel(q_len, bn):
                out[row, bx * bn + feature] = T.Cast("bfloat16", acc[row, feature])
            if pdl_trigger:
                T.pdl_trigger()

    return tilelang.compile(gemv_block, out_idx=None, pass_configs=NO_WS), nt


class OProjGemv:
    """`ao [8, q_dim] @ dequant(w_o).T` into one of two symmetric output slots, PDL-triggering.

    Takes `ao` as the attention output viewed flat, with no widening copy. The two slots
    alternate because the one-barrier AR tail has no output barrier: consecutive calls
    must never share one. The slots are keyed by name, so every holder shares them."""

    def __init__(self, q_dim: int, hidden: int, *, ws, q_len: int):
        if q_len != 8:
            raise ValueError(f"the OProj is compiled for Q=8, got {q_len}")
        self.q_dim, self.hidden, self.q_len = q_dim, hidden, q_len
        self.fn, _ = build_gemv_block(
            q_dim, hidden, **OPROJ_BLOCK_CFG, pdl_trigger=True)
        self.outputs = tuple(
            ws.slot(("oproj", key), (BM, hidden), torch.bfloat16, symm=True)
            for key in ("out", "out_alt"))
        self._next_output = 0

    def __call__(self, ao, w_o, w_scale):
        """ao [Q, q_dim] bf16 contiguous, w_o [hidden, q_dim] E4M3 with its rank-local
        [hidden/128, q_dim/128] scales -> [Q, hidden] bf16."""
        out = self.outputs[self._next_output]
        self._next_output = (self._next_output + 1) % len(self.outputs)
        self.fn(ao.reshape(self.q_len, self.q_dim), w_o, w_scale, out)
        return out[:self.q_len]


class DenseMlp:
    """Main layer 0's dense MLP: block-FP8 Gate/Up, SiLU*Up, block-FP8 Down into `out`.

    `out` is a `[BM, hidden]` producer buffer of the sandwich AR tail; the rank's Down
    output stays unreduced there. SiLU*Up is torch's, with the reference's rounding."""

    def __init__(self, hidden: int, inter: int, *, ws, q_len: int):
        if q_len != 8:
            raise ValueError(f"the dense MLP is compiled for Q=8, got {q_len}")
        self.q_len, self.inter = q_len, inter
        self.gate_up, _ = build_gemv_block(hidden, 2 * inter, **DENSE_BLOCK_CFG)
        self.down, _ = build_gemv_block(inter, hidden, **DENSE_BLOCK_CFG)
        self.gu = ws.slot(("dense", "gate_up"), (BM, 2 * inter), torch.bfloat16)

    def act(self, hn, w_gate_up, gate_up_scale):
        """hn [Q, hidden] bf16 -> SiLU(gate) * up [Q, inter] bf16, the Down's input."""
        self.gate_up(hn, w_gate_up, gate_up_scale, self.gu)
        gu = self.gu[:self.q_len]
        return F.silu(gu[:, :self.inter]) * gu[:, self.inter:]

    def __call__(self, hn, w_gate_up, gate_up_scale, w_down, down_scale, out):
        """hn [Q, hidden] bf16 -> out[:Q] bf16, the rank's partial Down output."""
        self.down(self.act(hn, w_gate_up, gate_up_scale), w_down, down_scale, out)
        return out[:self.q_len]


def build_lmhead_gemv(
    k: int,
    n: int,
    bn: int,
    bk: int,
    stages: int,
    threads: int,
    q_len: int,
):
    """LM-head GEMV: ``q_len`` rows of fp32 logits, fp32 accumulation over bf16."""
    nt, kt = n // bn, k // bk
    if n % bn or k % bk:
        raise ValueError(f"lmhead: N={n} K={k} not divisible by {bn}/{bk}")

    @T.prim_func
    def lmhead_gemv(
        x: T.Tensor((q_len, k), "bfloat16"),
        w: T.Tensor((n, k), "bfloat16"),
        out: T.Tensor((q_len, n), "float32"),
    ):
        with T.Kernel(nt, threads=threads) as bx:
            x_s = T.alloc_shared((BM, bk), "bfloat16")
            w_s = T.alloc_shared((bn, bk), "bfloat16")
            acc = T.alloc_fragment((BM, bn), "float32")
            T.clear(acc)
            for ki in T.Pipelined(kt, num_stages=stages):
                for i, j in T.Parallel(BM, bk):
                    x_s[i, j] = x[T.min(i, q_len - 1), ki * bk + j]
                T.copy(w[bx * bn, ki * bk], w_s)
                T.gemm(x_s, w_s, acc, transpose_B=True)
            # The MMA's padded rows q_len..BM-1 repeat the last row and stay unstored.
            for i, j in T.Parallel(BM, bn):
                if i < q_len:
                    out[i, bx * bn + j] = acc[i, j]

    return tilelang.compile(lmhead_gemv, out_idx=None, pass_configs=NO_WS)


class LmHeadGemv:
    """The verify LM head: [Q, hidden] bf16 x [V_local, hidden] bf16 -> [Q, V_local] fp32."""

    def __init__(self, hidden: int, v_local: int, *, device="cuda", q_len: int):
        self.v_local, self.q_len = v_local, q_len
        self.fn = build_lmhead_gemv(hidden, v_local, **LMHEAD_CFG, q_len=q_len)
        self.out = torch.zeros(q_len, v_local, dtype=torch.float32, device=device)

    def __call__(self, hv, w):
        self.fn(hv, w, self.out)
        return self.out
