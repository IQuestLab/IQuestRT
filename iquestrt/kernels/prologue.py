"""The attention prologue as two kernels instead of six ops."""

import functools

import tilelang
import tilelang.language as T
import torch

NO_WS = {tilelang.transform.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}

BM = 16   # T.gemm's MMA M-padding; rows Q..15 replicate the last row
HIDDEN, N_QKV, Q = 3072, 1024, 8
Q_HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM = 6, 1, 128, 32
KSPLIT = 4
QKV_BN, QKV_BK = 32, 256


# Block-FP8 QKV weights with BF16 activations: the recipe of `oproj.build_gemv_block`,
# with its E4M3 -> BF16 conversion, CUDA 13.2's `cvt.rn.bf16x2.e4m3x2`. A BN32
# tile lies inside one 128-row scale block, so a CTA needs one scale per K block of its
# split, and each block's MMA partial is scaled as it accumulates.
KB = 128


@functools.lru_cache(maxsize=None)
def build_qkv_block():
    """xn @ dequant(W_qkv).T -> qkv partials, when the MLP AR tail already wrote the norm.

    The input comes straight from global memory, so there is no shared->shared edge and
    three pipeline stages are legal. Tiles and split-K order match `build_qkv_norm_block`."""
    bn, bk, stages, threads = QKV_BN, QKV_BK, 3, 128
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
            x_t = T.alloc_shared((BM, bk), "bfloat16")
            w8_s = T.alloc_shared((bn, bk), T.float8_e4m3fn)
            w_s = T.alloc_shared((bn, bk), "bfloat16")
            part = T.alloc_fragment((BM, bn), "float32")
            acc = T.alloc_fragment((BM, bn), "float32")
            T.clear(acc)
            for ki in T.Pipelined(kt_s, num_stages=stages):
                for i, j in T.Parallel(BM, bk):
                    x_t[i, j] = xn_in[T.min(i, Q - 1), (bs * kt_s + ki) * bk + j]
                T.copy(w_qkv[bx * bn, (bs * kt_s + ki) * bk], w8_s)
                for i, j in T.Parallel(bn, bk):
                    w_s[i, j] = T.Cast("bfloat16", w8_s[i, j])
                for s in T.serial(sub):
                    T.clear(part)
                    T.gemm(x_t[0:BM, s * KB : (s + 1) * KB], w_s[0:bn, s * KB : (s + 1) * KB],
                           part, transpose_B=True)
                    for i, j in T.Parallel(BM, bn):
                        acc[i, j] += part[i, j] * scale[ki * sub + s]
            T.copy(acc, qkv_out[bs, 0, bx * bn])

    return tilelang.compile(prologue_qkv_prenorm_block_q, out_idx=None, pass_configs=NO_WS)


@functools.lru_cache(maxsize=None)
def build_qkv_norm_block():
    """Per-row RMSNorm of `hidden_in`, then xn @ dequant(W_qkv).T -> (xn_out, partials)."""
    bn, bk, stages, threads = QKV_BN, QKV_BK, 2, 128
    nt = N_QKV // bn
    kt_s = HIDDEN // bk // KSPLIT
    sub = bk // KB
    kb_s = kt_s * sub

    @T.prim_func
    def prologue_qkv_block_q(
        hidden_in: T.Tensor((Q, HIDDEN), "bfloat16"),
        w_norm: T.Tensor((HIDDEN,), "float32"),
        w_qkv: T.Tensor((N_QKV, HIDDEN), T.float8_e4m3fn),
        w_scale: T.Tensor((N_QKV // KB, HIDDEN // KB), "float32"),
        eps: T.Tensor((1,), "float32"),
        xn_out: T.Tensor((Q, HIDDEN), "bfloat16"),
        qkv_out: T.Tensor((KSPLIT, BM, N_QKV), "float32"),
    ):
        with T.Kernel(nt, KSPLIT, threads=threads) as (bx, bs):
            scale = T.alloc_local((kb_s,), "float32")
            for b in T.serial(kb_s):
                scale[b] = w_scale[bx * bn // KB, bs * kb_s + b]
            xn_s = T.alloc_shared((Q, HIDDEN), "bfloat16")
            x_t = T.alloc_shared((BM, bk), "bfloat16")
            w8_s = T.alloc_shared((bn, bk), T.float8_e4m3fn)
            w_s = T.alloc_shared((bn, bk), "bfloat16")
            part = T.alloc_fragment((BM, bn), "float32")
            acc = T.alloc_fragment((BM, bn), "float32")
            sq = T.alloc_fragment((1, HIDDEN), "float32")
            ssum = T.alloc_fragment((Q,), "float32")
            srow = T.alloc_fragment((1,), "float32")

            for r in T.serial(Q):
                for i in T.Parallel(HIDDEN):
                    sq[0, i] = (
                        T.Cast("float32", hidden_in[r, i])
                        * T.Cast("float32", hidden_in[r, i])
                    )
                T.reduce_sum(sq, srow, dim=1, clear=True)
                ssum[r] = srow[0]

            for r in T.serial(Q):
                inv = T.rsqrt(
                    ssum[r] / T.Cast("float32", HIDDEN) + eps[0]
                )
                for j in T.Parallel(HIDDEN):
                    xn_s[r, j] = T.Cast(
                        "bfloat16",
                        w_norm[j]
                        * (T.Cast("float32", hidden_in[r, j]) * inv),
                    )
            if bx == 0:
                if bs == 0:
                    for r, j in T.Parallel(Q, HIDDEN):
                        xn_out[r, j] = xn_s[r, j]

            T.clear(acc)
            for ki in T.Pipelined(kt_s, num_stages=stages):
                for i, j in T.Parallel(BM, bk):
                    x_t[i, j] = xn_s[T.min(i, Q - 1), (bs * kt_s + ki) * bk + j]
                T.copy(w_qkv[bx * bn, (bs * kt_s + ki) * bk], w8_s)
                for i, j in T.Parallel(bn, bk):
                    w_s[i, j] = T.Cast("bfloat16", w8_s[i, j])
                for s in T.serial(sub):
                    T.clear(part)
                    T.gemm(x_t[0:BM, s * KB : (s + 1) * KB], w_s[0:bn, s * KB : (s + 1) * KB],
                           part, transpose_B=True)
                    for i, j in T.Parallel(BM, bn):
                        acc[i, j] += part[i, j] * scale[ki * sub + s]
            T.copy(acc, qkv_out[bs, 0, bx * bn])

    return tilelang.compile(prologue_qkv_block_q, out_idx=None, pass_configs=NO_WS)


@functools.lru_cache(maxsize=None)
def build_heads(capacity: int, pdl: bool):
    """Sum the QKV partials, q/k RMSNorm, partial rope -> q and the K/V ring slots.

    Grid (heads, Q): each (head, row) pair has its own rope angle and, for K/V, its own
    ring slot. `pdl` waits at entry, before the first read of the partials, and releases
    the attention kernel at once (the main stack's PDL chain)."""
    threads = 128
    q_dim = Q_HEADS * HEAD_DIM
    kv_dim = KV_HEADS * HEAD_DIM
    total = Q_HEADS + 2 * KV_HEADS
    half = ROTARY_DIM // 2
    keep = HEAD_DIM - ROTARY_DIM
    cap = capacity

    @T.prim_func
    def prologue_heads_q(
        qkv: T.Tensor((KSPLIT, BM, q_dim + 2 * kv_dim), "float32"),
        w_qn: T.Tensor((HEAD_DIM,), "float32"),
        w_kn: T.Tensor((HEAD_DIM,), "float32"),
        cos: T.Tensor((Q, ROTARY_DIM), "bfloat16"),
        sin: T.Tensor((Q, ROTARY_DIM), "bfloat16"),
        eps: T.Tensor((1,), "float32"),
        slot: T.Tensor((Q,), "int64"),
        q_out: T.Tensor((Q, Q_HEADS, HEAD_DIM), "bfloat16"),
        k_out: T.Tensor((cap, KV_HEADS, HEAD_DIM), "bfloat16"),
        v_out: T.Tensor((cap, KV_HEADS, HEAD_DIM), "bfloat16"),
    ):
        with T.Kernel(total, Q, threads=threads) as (bh, br):
            if pdl:
                T.pdl_sync()
                T.pdl_trigger()
            # SHARED, not a fragment: the rope reads `row[i]` and `row[i + half]` in one
            # `T.Parallel(half)`, which layout inference refuses on a fragment with the
            # query row in the grid ("row: (i + 16,) and (i,)").
            row = T.alloc_shared((HEAD_DIM,), "bfloat16")
            col = T.alloc_fragment((HEAD_DIM,), "float32")
            sq = T.alloc_fragment((1, HEAD_DIM), "float32")
            ssum = T.alloc_fragment((1,), "float32")
            base = T.alloc_fragment((1,), "int32")
            sl = T.alloc_fragment((1,), "int32")
            sl[0] = T.Cast("int32", slot[br])

            if bh >= Q_HEADS + KV_HEADS:
                base[0] = q_dim + kv_dim + (bh - Q_HEADS - KV_HEADS) * HEAD_DIM
                for d in T.Parallel(HEAD_DIM):
                    col[d] = qkv[0, br, base[0] + d]
                for sp in T.serial(1, KSPLIT):
                    for d in T.Parallel(HEAD_DIM):
                        col[d] += qkv[sp, br, base[0] + d]
                for d in T.Parallel(HEAD_DIM):
                    v_out[sl[0], bh - Q_HEADS - KV_HEADS, d] = T.Cast("bfloat16", col[d])
            else:
                if bh < Q_HEADS:
                    base[0] = bh * HEAD_DIM
                else:
                    base[0] = q_dim + (bh - Q_HEADS) * HEAD_DIM
                for d in T.Parallel(HEAD_DIM):
                    col[d] = qkv[0, br, base[0] + d]
                for sp in T.serial(1, KSPLIT):
                    for d in T.Parallel(HEAD_DIM):
                        col[d] += qkv[sp, br, base[0] + d]
                for d in T.Parallel(HEAD_DIM):
                    sq[0, d] = col[d] * col[d]
                T.reduce_sum(sq, ssum, dim=1, clear=True)
                inv = T.rsqrt(ssum[0] / T.Cast("float32", HEAD_DIM) + eps[0])
                for d in T.Parallel(HEAD_DIM):
                    if bh < Q_HEADS:
                        row[d] = T.Cast("bfloat16", w_qn[d] * (col[d] * inv))
                    else:
                        row[d] = T.Cast("bfloat16", w_kn[d] * (col[d] * inv))
                # Rounding points reproduce `reference.apply_partial_rope`.
                for i in T.Parallel(half):
                    a = T.Cast("bfloat16", row[i] * cos[br, i])
                    b = T.Cast("bfloat16", (-row[i + half]) * sin[br, i])
                    if bh < Q_HEADS:
                        q_out[br, bh, i] = T.Cast("bfloat16", a + b)
                    else:
                        k_out[sl[0], bh - Q_HEADS, i] = T.Cast("bfloat16", a + b)
                for i in T.Parallel(half):
                    a = T.Cast("bfloat16", row[half + i] * cos[br, half + i])
                    b = T.Cast("bfloat16", row[i] * sin[br, half + i])
                    if bh < Q_HEADS:
                        q_out[br, bh, half + i] = T.Cast("bfloat16", a + b)
                    else:
                        k_out[sl[0], bh - Q_HEADS, half + i] = T.Cast("bfloat16", a + b)
                for i in T.Parallel(keep):
                    if bh < Q_HEADS:
                        q_out[br, bh, ROTARY_DIM + i] = row[ROTARY_DIM + i]
                    else:
                        k_out[sl[0], bh - Q_HEADS, ROTARY_DIM + i] = row[ROTARY_DIM + i]

    return tilelang.compile(prologue_heads_q, out_idx=None, pass_configs=NO_WS)


class PrologueKernels:
    """The prologue kernels of one layer, over arena slots."""

    def __init__(self, cfg, shard, *, ws, q_len: int):
        shape = (cfg.hidden_size, shard.q_dim + 2 * shard.kv_dim, shard.num_q_heads,
                 shard.num_kv_heads, cfg.head_dim, cfg.rotary_dim, q_len)
        if shape != (HIDDEN, N_QKV, Q_HEADS, KV_HEADS, HEAD_DIM, ROTARY_DIM, Q):
            raise ValueError(f"the prologue is built for the iquest_q1 TP8 Q8 shape, got {shape}")
        self.h = HIDDEN
        self.qkv_norm_fn = build_qkv_norm_block()
        self.qkv_fn = build_qkv_block()
        # One heads kernel per (KV capacity, PDL); the capacities come from the caches.
        # `q8_profile.install` substitutes the main stack's PDL form through the
        # builder.
        self._heads = {}
        self._heads_builder = build_heads

        def slot(key, shape, dtype):
            return ws.slot(("prologue", key), shape, dtype)

        self.eps = slot("eps", (1,), torch.float32)
        self.xn = slot("xn", (Q, HIDDEN), torch.bfloat16)
        self.qkv = slot("qkv", (KSPLIT, BM, N_QKV), torch.float32)
        self.q = slot("q", (Q, Q_HEADS, HEAD_DIM), torch.bfloat16)
        self._eps_set = False
        self._eps_val = float(cfg.rms_norm_eps)

    def __call__(self, hidden, w_norm, w_qkv, w_scale, w_qn, w_kn, cos, sin, slot,
                 k_cache, v_cache):
        """hidden [Q,H] -> (xn [Q,H], q [Q,Hq,D]); K/V go straight into the ring.

        `w_qkv` is E4M3 with its rank-local [N_QKV/128, H/128] scale table `w_scale`.
        `cos`/`sin` are [Q, rotary_dim] and `slot` is [Q]. No k/v are returned: the step
        reads the whole ring back with `read_prefix`."""
        if not self._eps_set:
            self.eps.fill_(self._eps_val)
            self._eps_set = True
        self.qkv_norm_fn(hidden.reshape(Q, HIDDEN), w_norm, w_qkv, w_scale, self.eps,
                         self.xn, self.qkv)
        return self.xn, self.heads(w_qn, w_kn, cos, sin, slot, k_cache, v_cache)

    def call_prenorm(self, xn, w_qkv, w_scale, w_qn, w_kn, cos, sin, slot, k_cache,
                     v_cache):
        """QKV + heads when `xn` was emitted by the MLP AR tail."""
        xn = xn.reshape(Q, HIDDEN)
        self.qkv_fn(xn, w_qkv, w_scale, self.qkv)
        return xn, self.heads(w_qn, w_kn, cos, sin, slot, k_cache, v_cache)

    def heads(self, w_qn, w_kn, cos, sin, slot, k_cache, v_cache, *, pdl=False):
        """The heads kernel over `self.qkv` -> q [Q,Hq,D]."""
        key = (k_cache.shape[0], pdl)
        fn = self._heads.get(key)
        if fn is None:
            fn = self._heads[key] = self._heads_builder(*key)
        fn(self.qkv, w_qn, w_kn,
           cos.reshape(Q, -1)[:, :ROTARY_DIM].contiguous(),
           sin.reshape(Q, -1)[:, :ROTARY_DIM].contiguous(),
           self.eps, slot.reshape(Q), self.q, k_cache, v_cache)
        return self.q


@functools.lru_cache(maxsize=None)
def rope_inv(rotary_dim: int, theta: float, device) -> torch.Tensor:
    """`1.0 / theta ** (arange(half)/half)`, torch's own expression, computed ONCE."""
    half = rotary_dim // 2
    return 1.0 / (theta ** (torch.arange(0, half, device=device,
                                         dtype=torch.float32) / half))
