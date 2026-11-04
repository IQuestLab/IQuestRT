"""Online, streaming, and draft decode attention.

Mask by each row's position. Keep NEG_INF finite and gate exp with NEG_GUARD
so an all-masked split produces zeros. Reduction order differs from the reference.
TileLang requires evaluated annotations."""

import functools

import tilelang
import tilelang.language as T
import torch

from tilelang.cuda.language.cluster import cluster_sync
from tilelang.layout.swizzle import make_swizzled_layout

from .attention_stream import build_stream, build_stream_combine

# Scalar FP32 sums: on SM100 TileLang 0.1.15 by default adds a sum's elements in two
# interleaved lanes, and a batched reduction in that form fails to compile (its pack
# buffer is never allocated).
PASS_CONFIGS = {
    tilelang.transform.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.transform.PassConfigKey.TL_ENABLE_FP32X2_REDUCTION: False,
}

HEADS, KV_HEADS, DIM, Q = 6, 1, 128, 8
REP = HEADS // KV_HEADS
NEG_INF = -3.0e38   # not float('-inf'): exp(-inf - -inf) is NaN, this stays finite
NEG_GUARD = -1.5e38  # "this running max is still the sentinel", i.e. nothing seen yet
BM = 16             # MMA M-padding: T.gemm needs M >= 16
# The partial buffers are flat and viewed at each cell's split; this bounds the arena.
NS_MAX = 512
# The sliding layers' window (the checkpoint's `sliding_window`) and their physical ring:
# the window plus one 64-slot tile, so a verify's Q writes evict nothing a row still reads.
# Its cell maps each slot to the most recent congruent position and bounds each row by
# its window (`build_online(ring=True, window=WINDOW)`). The 4096 cell keeps the
# power-of-two form of that map without the bound: a run of up to 4096 positions reads
# the ring's first 4096 slots, and those of a full cache, where slot s holds position s
# and every row's window reaches position 0. Every other cell reads slot s as position s.
WINDOW = 4096
RING_KV = WINDOW + 64

ONLINE_GEOMETRY = {
    2048: (64, 32, 3, True),
    4096: (64, 64, 3, True),
    RING_KV: (64, RING_KV // 64, 3, True),
}
STREAM_KV = 200704
# The main stack's read lengths, smallest first (`Engine.kv_len_for`). The sliding layers
# read their ring's first slots at the first two, and the whole ring, `RING_KV`, past them.
MAIN_READS = (2048, 4096, STREAM_KV)
# Cells reading a ring, each slot as the most recent position congruent to it.
RING_READS = (4096, RING_KV)
# Exact compact32 CTA packing by split count; every other split keeps one warp per CTA.
COMBINE_WARP_PACK = {64: 4, RING_KV // 64: 4}
# (bk, stages, n_split) at the draft ring's read length.
DRAFT_GEOMETRY = {640: (128, 3, 5)}


@functools.lru_cache(maxsize=None)
def build_online(kv: int, *, bk: int, n_split: int, group_split: int, ring: bool,
                 window: int | None = None):
    """Head-folded online-softmax split kernel -> per-split (pm, pl, pacc) partials."""
    threads, stages = 128, 1
    if REP % group_split:
        raise ValueError(f"group_split={group_split} must divide rep={REP}")
    rep_g = REP // group_split
    rows = Q * rep_g
    bm = BM if rows <= BM else 2 * BM
    if rows > bm:
        raise ValueError(f"Q*rep_g={rows} exceeds the fold's M padding {bm}")
    if kv % (bk * n_split):
        raise ValueError(f"kv={kv} must be a multiple of bk*n_split={bk * n_split}")
    pow2 = not kv & (kv - 1)
    if window is not None and not (ring and 0 < window <= kv - (Q - 1)):
        raise ValueError(f"a window bounds a ring at least Q - 1 slots longer, got "
                         f"kv={kv} window={window} ring={ring}")
    if ring and not pow2 and window is None:
        raise ValueError(f"a ring of {kv} slots is longer than a window it does not name")
    chunk = kv // n_split
    kt = chunk // bk
    batch = kt == 1
    reduce_batch = 2
    kv_shape = (kv, KV_HEADS, DIM)

    @T.prim_func
    def attn_online_q(
        q: T.Tensor((Q, HEADS, DIM), "bfloat16"),
        k: T.Tensor(kv_shape, "bfloat16"),
        v: T.Tensor(kv_shape, "bfloat16"),
        sink: T.Tensor((KV_HEADS, DIM), "bfloat16"),
        mask: T.Tensor((Q,), "int64"),
        scale: T.Tensor((1,), "float32"),
        pm: T.Tensor((Q, HEADS, n_split), "float32"),
        pl: T.Tensor((Q, HEADS, n_split), "float32"),
        pacc: T.Tensor((Q, HEADS, n_split, DIM), "float32"),
        out: T.Tensor((Q, HEADS, DIM), "bfloat16"),
    ):
        with T.Kernel(KV_HEADS * group_split, n_split, threads=threads) as (bg, bs):
            # `bg` indexes (KV head, subgroup).
            bh = bg // group_split
            sub = bg % group_split
            T.pdl_sync()
            T.pdl_trigger()
            base = bs * chunk
            q_p = T.alloc_shared((bm, DIM), "bfloat16")
            k_s = T.alloc_shared((bk, DIM), "bfloat16")
            v_s = T.alloc_shared((bk, DIM), "bfloat16")
            # P bounces through shared: the QK gemm's C-fragment layout and the PV
            # gemm's A-fragment layout are different register mappings under mma.sync
            # (no warp specialization), so a frag->frag copy cannot shuffle between
            # them ("Layout infer conflict between sf and p_f").
            p_s = T.alloc_shared((bm, bk), "bfloat16")
            sf = T.alloc_fragment((bm, bk), "float32")
            acc = T.alloc_fragment((bm, DIM), "float32")
            m = T.alloc_fragment((bm,), "float32")
            mp = T.alloc_fragment((bm,), "float32")
            cr = T.alloc_fragment((bm,), "float32")
            l = T.alloc_fragment((bm,), "float32")  # noqa: E741 (kernel symbol)
            rs = T.alloc_fragment((bm,), "float32")
            skf = T.alloc_fragment((bm, DIM), "float32")
            sl = T.alloc_fragment((bm,), "float32")

            sc = scale[0]
            # Verify rows are consecutive decode positions; row qr is qpos0 + qr.
            qpos0 = mask[0]
            for r, d in T.Parallel(bm, DIM):
                q_p[r, d] = q[
                    T.min(r // rep_g, Q - 1),
                    bh * REP + sub * rep_g + r % rep_g,
                    d,
                ]
            T.clear(acc)
            T.fill(m, NEG_INF)
            T.clear(l)

            for ki in T.Pipelined(kt, num_stages=stages):
                for j, d in T.Parallel(bk, DIM):
                    k_s[j, d] = k[base + ki * bk + j, bh, d]
                T.clear(sf)
                # FullRow on BOTH gemms: the row reductions and the [bm] running state
                # read the C fragments per-row, so every warp must own whole rows --
                # without it layout inference dies ("Layout may conflict with ReduceOp").
                T.gemm(q_p, k_s, sf, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                # Scale BEFORE mask, mask writes the sentinel LAST: NEG_INF is finite,
                # so `NEG_INF * sc` would land above NEG_GUARD and break the guard.
                # Indexed by the row's QUERY row -- verify rows see different windows.
                for r, j in T.Parallel(bm, bk):
                    sf[r, j] = sf[r, j] * sc
                    qr = T.min(r // rep_g, Q - 1)
                    if ring and pow2:
                        # The newest written absolute position is the last verify row;
                        # power-of-two masking implements floor-mod even before the
                        # first wrap, where p-slot is negative. Eliminating kpos:
                        #   kpos <= row_qpos  <=> age >= Q-1-qr
                        #   kpos >= 0         <=> age <= last_qpos
                        # and kpos > row_qpos-kv is implied by age <= kv-1. The model
                        # bound (200704) keeps every value in int32.
                        last_qpos_i32 = T.Cast("int32", qpos0) + T.int32(Q - 1)
                        slot_i32 = T.Cast("int32", base + ki * bk + j)
                        age_i32 = T.bitwise_and(
                            last_qpos_i32 - slot_i32,
                            T.int32(kv - 1),
                        )
                        mask_value = T.Cast(
                            "int32",
                            (age_i32 >= T.int32(Q - 1) - qr)
                            and (age_i32 <= last_qpos_i32),
                        )
                    elif ring:
                        # The same map by a floor-mod: last_qpos - slot + kv > 0. The
                        # ring outlasts the window, so the window bound is explicit:
                        #   kpos > row_qpos - window  <=> age <= window + Q-2 - qr
                        last_qpos_i32 = T.Cast("int32", qpos0) + T.int32(Q - 1)
                        slot_i32 = T.Cast("int32", base + ki * bk + j)
                        age_i32 = T.floormod(
                            last_qpos_i32 - slot_i32 + T.int32(kv),
                            T.int32(kv),
                        )
                        mask_value = T.Cast(
                            "int32",
                            (age_i32 >= T.int32(Q - 1) - qr)
                            and (age_i32 <= last_qpos_i32)
                            and (age_i32 <= T.int32(window + Q - 2) - qr),
                        )
                    else:
                        # Slot s holds position s.
                        mask_value = T.Cast(
                            "int32",
                            T.Cast("int32", base + ki * bk + j)
                            <= T.Cast("int32", qpos0) + qr,
                        )
                    if mask_value == 0:
                        sf[r, j] = NEG_INF
                # Online update (fill -> reduce clear=False -> explicit max with prev):
                # m is monotone, cr rescales history, and the guard keeps the sentinel
                # from ever entering an exp.
                T.copy(m, mp)
                T.fill(m, NEG_INF)
                T.reduce_max(sf, m, dim=1, clear=False)
                for r in T.Parallel(bm):
                    m[r] = T.max(m[r], mp[r])
                for r in T.Parallel(bm):
                    cr[r] = T.if_then_else(
                        mp[r] > NEG_GUARD,
                        T.exp(mp[r] - m[r]),
                        0.0,
                    )
                for r, j in T.Parallel(bm, bk):
                    sf[r, j] = T.if_then_else(
                        m[r] > NEG_GUARD,
                        T.exp(sf[r, j] - m[r]),
                        0.0,
                    )
                if batch:
                    T.reduce_sum(
                        sf, rs, dim=1, clear=True, batch=reduce_batch
                    )
                else:
                    T.reduce_sum(sf, rs, dim=1, clear=True)
                for r in T.Parallel(bm):
                    l[r] = l[r] * cr[r] + rs[r]
                for r, d in T.Parallel(bm, DIM):
                    acc[r, d] = acc[r, d] * cr[r]
                for r, j in T.Parallel(bm, bk):
                    p_s[r, j] = T.Cast("bfloat16", sf[r, j])
                for j, d in T.Parallel(bk, DIM):
                    v_s[j, d] = v[base + ki * bk + j, bh, d]
                T.gemm(p_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)

            if bs == 0:
                # The sink is one extra logit with no V: it moves only (m, l) and
                # rescales acc. The T.copy is a LAYOUT TIE, not data movement: skf has
                # no other layout-constrained producer/consumer, and inference dies on
                # such free fragments; copying from acc pins skf to acc's FullRow layout
                # and the next loop overwrites the values.
                T.copy(acc, skf)
                for r, d in T.Parallel(bm, DIM):
                    skf[r, d] = (
                        T.Cast("float32", q_p[r, d])
                        * T.Cast("float32", sink[bh, d])
                    )
                if batch:
                    T.reduce_sum(
                        skf, sl, dim=1, clear=True, batch=reduce_batch
                    )
                else:
                    T.reduce_sum(skf, sl, dim=1, clear=True)
                for r in T.Parallel(bm):
                    sl[r] = sl[r] * sc
                    mp[r] = T.max(m[r], sl[r])
                    cr[r] = T.if_then_else(
                        m[r] > NEG_GUARD,
                        T.exp(m[r] - mp[r]),
                        0.0,
                    )
                    l[r] = l[r] * cr[r] + T.exp(sl[r] - mp[r])
                    m[r] = mp[r]
                for r, d in T.Parallel(bm, DIM):
                    acc[r, d] = acc[r, d] * cr[r]

            for r in T.Parallel(bm):
                if r < rows:
                    pm[r // rep_g, bh * REP + sub * rep_g + r % rep_g, bs] = m[r]
                    pl[r // rep_g, bh * REP + sub * rep_g + r % rep_g, bs] = l[r]
            for r, d in T.Parallel(bm, DIM):
                if r < rows:
                    pacc[
                        r // rep_g,
                        bh * REP + sub * rep_g + r % rep_g,
                        bs,
                        d,
                    ] = acc[r, d]

    return tilelang.compile(attn_online_q, out_idx=None, pass_configs=PASS_CONFIGS)


@functools.lru_cache(maxsize=None)
def build_exact(kv: int, *, bk: int, stages: int, n_split: int, rows: int = Q):
    """Two-pass split kernel with the merge folded into one thread block cluster -> out."""
    threads, pv_acc = 128, 16
    if bk % pv_acc:
        raise ValueError(f"pv_acc={pv_acc} must divide bk={bk}")
    if kv % (bk * n_split):
        raise ValueError(f"kv={kv} must be a multiple of bk*n_split={bk * n_split}")
    if not 1 < n_split <= 16:
        raise ValueError(f"the cluster fold needs 2 <= n_split <= 16, got {n_split}")
    if not 1 <= rows <= Q:
        raise ValueError(f"the exact cell computes 1..{Q} rows, got {rows}")
    chunk = kv // n_split
    kt = chunk // bk
    kv_total = 1 + kv
    qw = 1          # query rows per CTA

    @T.prim_func
    def attn_exact_q(
        q: T.Tensor((Q, HEADS, DIM), "bfloat16"),
        k: T.Tensor((kv, KV_HEADS, DIM), "bfloat16"),
        v: T.Tensor((kv, KV_HEADS, DIM), "bfloat16"),
        sink: T.Tensor((KV_HEADS, DIM), "bfloat16"),
        mask: T.Tensor((Q, kv_total), "uint8"),
        scale: T.Tensor((1,), "float32"),
        pm: T.Tensor((Q, HEADS, n_split), "float32"),
        pl: T.Tensor((Q, HEADS, n_split), "float32"),
        pacc: T.Tensor((Q, HEADS, n_split, DIM), "float32"),
        out: T.Tensor((Q, HEADS, DIM), "bfloat16"),
    ):
        with T.ClusterKernel(HEADS, n_split, rows, cluster_dims=(1, n_split, 1),
                             threads=threads) as (bh, bs, bq):
            T.pdl_sync()
            kvh = bh // REP
            base = bs * chunk
            q_p = T.alloc_shared((BM, DIM), "bfloat16")
            k_s = T.alloc_shared((bk, DIM), "bfloat16")
            v_s = T.alloc_shared((bk, DIM), "bfloat16")
            s_sh = T.alloc_shared((BM, bk), "float32")
            s_all = T.alloc_shared((qw, chunk), "float32")
            sink_s = T.alloc_shared((DIM,), "bfloat16")
            # Swizzling the fp32 score staging moves addresses, not values.
            T.annotate_layout({s_sh: make_swizzled_layout(s_sh)})
            q8 = T.alloc_local((8,), "bfloat16")
            sink8 = T.alloc_local((8,), "bfloat16")
            pv = T.alloc_fragment((qw, pv_acc, DIM), "float32")
            sf = T.alloc_fragment((BM, bk), "float32")
            rf = T.alloc_fragment((qw, chunk), "float32")
            mx = T.alloc_fragment((qw,), "float32")
            sm = T.alloc_fragment((qw,), "float32")
            m0 = T.alloc_fragment((qw,), "float32")
            ltot = T.alloc_fragment((qw,), "float32")
            sl = T.alloc_fragment((qw,), "float32")
            acc = T.alloc_fragment((qw, DIM), "float32")

            sc = scale[0]
            # All BM rows of q_p hold this CTA's row: T.gemm computes every row, and
            # undefined shared memory could overflow to inf/NaN in the dead lanes.
            for i, d in T.Parallel(BM, DIM):
                q_p[i, d] = q[bq, bh, d]
            for d in T.Parallel(DIM):
                sink_s[d] = sink[kvh, d]
            T.clear(acc)
            T.clear(pv)

            # -- pass 1: every logit in this split.
            for ki in T.Pipelined(kt, num_stages=stages):
                for j, d in T.Parallel(bk, DIM):
                    k_s[j, d] = k[base + ki * bk + j, kvh, d]
                T.clear(sf)
                T.gemm(q_p, k_s, sf, transpose_B=True)
                T.copy(sf, s_sh)
                for i, j in T.Parallel(qw, bk):
                    s_all[i, ki * bk + j] = s_sh[i, j] * sc
                    if mask[
                        bq + i, 1 + base + ki * bk + j
                    ] == T.Cast("uint8", 0):
                        s_all[i, ki * bk + j] = NEG_INF

            for i in T.serial(qw):
                sl[i] = NEG_INF
            if bs == 0:
                # The chain reads q_p's row (every row of q_p holds this CTA's) and the
                # staged sink row, eight dims at a time into registers.
                for i in T.serial(qw):
                    sl[i] = 0.0
                    for d8 in T.serial(DIM // 8):
                        for e in T.vectorized(8):
                            q8[e] = q_p[0, d8 * 8 + e]
                        for e in T.vectorized(8):
                            sink8[e] = sink_s[d8 * 8 + e]
                        for e in T.serial(8):
                            sl[i] += T.Cast("float32", q8[e]) * T.Cast("float32", sink8[e])
                    sl[i] = sl[i] * sc

            for i, j in T.Parallel(qw, chunk):
                rf[i, j] = s_all[i, j]
            T.reduce_max(rf, mx, dim=1, clear=True)
            for i in T.serial(qw):
                m0[i] = T.max(mx[i], sl[i])

            for i, j in T.Parallel(qw, chunk):
                if m0[i] > NEG_GUARD:
                    rf[i, j] = T.exp(s_all[i, j] - m0[i])
                else:
                    rf[i, j] = 0.0
                s_all[i, j] = rf[i, j]
            T.reduce_sum(rf, sm, dim=1, clear=True)
            for i in T.serial(qw):
                ltot[i] = sm[i]
                if bs == 0:
                    if m0[i] > NEG_GUARD:
                        ltot[i] += T.exp(sl[i] - m0[i])

            # -- pass 2: PV in fp32, so the probabilities are never rounded to bf16.
            for ki in T.Pipelined(kt, num_stages=stages):
                for j, d in T.Parallel(bk, DIM):
                    v_s[j, d] = v[base + ki * bk + j, kvh, d]
                for i, a, d in T.Parallel(qw, pv_acc, DIM):
                    for jj in T.serial(bk // pv_acc):
                        pv[i, a, d] += (s_all[i, ki * bk + jj * pv_acc + a]
                                        * T.Cast("float32",
                                                 v_s[jj * pv_acc + a, d]))
            for a in T.serial(pv_acc):
                for i, d in T.Parallel(qw, DIM):
                    acc[i, d] += pv[i, a, d]

            for i in T.serial(qw):
                pm[bq + i, bh, bs] = m0[i]
                pl[bq + i, bh, bs] = ltot[i]
            for i, d in T.Parallel(qw, DIM):
                pacc[bq + i, bh, bs, d] = acc[i, d]

            # After the cluster barrier the partial stores are visible to bs == 0
            # (cluster-scope release/acquire covers global accesses). The per-lane
            # serial-over-split sums are a different fp32 order from the combine
            # kernels' tree reduction.
            c_mx = T.alloc_fragment((qw, DIM), "float32")
            c_ls = T.alloc_fragment((qw, DIM), "float32")
            c_ac = T.alloc_fragment((qw, DIM), "float32")
            cluster_sync()
            if bs == 0:
                for i, d in T.Parallel(qw, DIM):
                    c_mx[i, d] = NEG_INF
                    for s in T.serial(n_split):
                        c_mx[i, d] = T.max(c_mx[i, d], pm[bq + i, bh, s])
                    c_ls[i, d] = 0.0
                    c_ac[i, d] = 0.0
                    for s in T.serial(n_split):
                        c_ls[i, d] += (pl[bq + i, bh, s]
                                       * T.exp(pm[bq + i, bh, s] - c_mx[i, d]))
                        c_ac[i, d] += (T.exp(pm[bq + i, bh, s] - c_mx[i, d])
                                       * pacc[bq + i, bh, s, d])
                    out[bq + i, bh, d] = T.Cast("bfloat16",
                                                c_ac[i, d] / c_ls[i, d])

    return tilelang.compile(attn_exact_q, out_idx=None, pass_configs=PASS_CONFIGS)


def build_combine(n_split: int, *, compact32: bool):
    """Merge `n_split` partial (m, l, acc) triples into one output row per (row, head).

    The flash-decoding rescale: with M = max_s m_s,
        out = sum_s exp(m_s - M) * acc_s  /  sum_s exp(m_s - M) * l_s.
    All three forms are bit-exact to one another. PDL: wait before the first partial
    read, then release the OProj at once."""
    if not compact32:
        return _build_combine_q(n_split)
    pack = COMBINE_WARP_PACK.get(n_split, 1)
    if pack > 1:
        return _build_combine_q_compact32_warp_pack(n_split, pack)
    return _build_combine_q_compact32(n_split)


@functools.lru_cache(maxsize=None)
def _build_combine_q(n_split: int):
    """One 128-thread CTA per (head, row); the split loops are padded to the thread
    count with NEG_INF/0.0, which contribute nothing to either reduction."""
    threads, dt = 128, DIM
    nd = DIM // dt
    npad = -(-n_split // threads) * threads

    @T.prim_func
    def attn_combine_q(
        pm: T.Tensor((Q, HEADS, n_split), "float32"),
        pl: T.Tensor((Q, HEADS, n_split), "float32"),
        pacc: T.Tensor((Q, HEADS, n_split, DIM), "float32"),
        out: T.Tensor((Q, HEADS, DIM), "bfloat16"),
    ):
        with T.Kernel(HEADS, Q, nd, threads=threads) as (bh, bi, bd):
            T.pdl_sync()
            T.pdl_trigger()
            mr = T.alloc_fragment((1, npad), "float32")
            mx = T.alloc_fragment((1,), "float32")
            w_s = T.alloc_shared((npad,), "float32")
            lt = T.alloc_fragment((1, npad), "float32")
            ls = T.alloc_fragment((1,), "float32")
            acc = T.alloc_fragment((1, dt), "float32")

            for s in T.Parallel(npad):
                mr[0, s] = NEG_INF
                if s < n_split:
                    mr[0, s] = pm[bi, bh, s]
            T.reduce_max(mr, mx, dim=1, clear=True)
            for s in T.Parallel(npad):
                w_s[s] = 0.0
                lt[0, s] = 0.0
                if s < n_split:
                    w_s[s] = T.exp(pm[bi, bh, s] - mx[0])
                    lt[0, s] = pl[bi, bh, s] * w_s[s]
            T.reduce_sum(lt, ls, dim=1, clear=True)
            for d in T.Parallel(dt):
                acc[0, d] = 0.0
                for s in T.serial(n_split):
                    acc[0, d] += w_s[s] * pacc[bi, bh, s, bd * dt + d]
                out[bi, bh, bd * dt + d] = T.Cast("bfloat16", acc[0, d] / ls[0])

    return tilelang.compile(attn_combine_q, out_idx=None, pass_configs=PASS_CONFIGS)


def _compact32_items(n_split: int) -> int:
    """Splits per lane when one warp emulates the 128-thread combine reduction."""
    local_items = -(-n_split // 128)
    if not 1 < n_split <= NS_MAX or local_items > 2 or n_split % local_items:
        raise ValueError(f"compact32 cannot emulate the 128-thread tree at n_split={n_split}")
    return local_items


@functools.lru_cache(maxsize=None)
def _build_combine_q_compact32(n_split: int):
    """One-warp D=32 combine that preserves the 128-thread reduction tree."""
    local_items = _compact32_items(n_split)
    bd = DIM // 4

    @T.prim_func
    def attn_combine_compact32(
        pm: T.Tensor((Q, HEADS, n_split), "float32"),
        pl: T.Tensor((Q, HEADS, n_split), "float32"),
        pacc: T.Tensor((Q, HEADS, n_split, DIM), "float32"),
        out: T.Tensor((Q, HEADS, DIM), "bfloat16"),
    ):
        with T.Kernel(HEADS, Q, 4, threads=32) as (bh, bi, bd_i):
            T.pdl_sync()
            T.pdl_trigger()
            vmax = T.alloc_local((4,), "float32")
            vsum = T.alloc_local((4,), "float32")
            w_local = T.alloc_local((4 * local_items,), "float32")
            red = T.alloc_fragment((1, 32), "float32")
            mx = T.alloc_fragment((1,), "float32")
            ls = T.alloc_fragment((1,), "float32")
            acc = T.alloc_local((1,), "float32")
            tx = T.get_thread_binding()

            for g in T.serial(4):
                vmax[g] = -T.infinity("float32")
                for u in T.serial(local_items):
                    s = (tx + g * 32) * local_items + u
                    vmax[g] = T.max(
                        vmax[g],
                        T.if_then_else(s < n_split, pm[bi, bh, s], NEG_INF),
                    )
            merged_max = T.max(T.max(vmax[0], vmax[2]),
                               T.max(vmax[1], vmax[3]))
            for lane in T.Parallel(32):
                red[0, lane] = merged_max
            T.reduce_max(red, mx, dim=1, clear=True)

            for g in T.serial(4):
                vsum[g] = 0.0
                for u in T.serial(local_items):
                    s = (tx + g * 32) * local_items + u
                    w_local[g * local_items + u] = T.if_then_else(
                        s < n_split, T.exp(pm[bi, bh, s] - mx[0]), 0.0
                    )
                    vsum[g] += T.if_then_else(
                        s < n_split,
                        T.call_extern("float32", "__fmul_rn", pl[bi, bh, s],
                                      w_local[g * local_items + u]),
                        0.0,
                    )
            merged_sum = ((vsum[0] + vsum[2]) +
                          (vsum[1] + vsum[3]))
            for lane in T.Parallel(32):
                red[0, lane] = merged_sum
            T.reduce_sum(red, ls, dim=1, clear=True)

            for d in T.Parallel(bd):
                acc[0] = 0.0
                for s in T.serial(n_split):
                    acc[0] += T.shfl_sync(
                        w_local[(s // local_items // 32) * local_items
                                + s % local_items],
                        (s // local_items) % 32,
                    ) * pacc[bi, bh, s, bd_i * bd + d]
                out[bi, bh, bd_i * bd + d] = T.Cast(
                    "bfloat16", acc[0] / ls[0])

    return tilelang.compile(attn_combine_compact32, out_idx=None,
                            pass_configs=PASS_CONFIGS)


@functools.lru_cache(maxsize=None)
def _build_combine_q_compact32_warp_pack(n_split: int, pack_warps: int):
    """Pack independent exact compact32 D tiles into a wider CTA.

    Every warp retains the XOR max/sum tree, the rounded products and the serial pacc
    walk.  Packing changes only CTA ownership: four 32-thread CTAs become two 64-thread
    CTAs or one 128-thread CTA."""
    if pack_warps not in (2, 4):
        raise ValueError(f"pack_warps must be 2 or 4, got {pack_warps}")
    local_items = _compact32_items(n_split)
    bd = DIM // 4

    @T.prim_func
    def attn_combine_compact32_warp_pack(
        pm: T.Tensor((Q, HEADS, n_split), "float32"),
        pl: T.Tensor((Q, HEADS, n_split), "float32"),
        pacc: T.Tensor((Q, HEADS, n_split, DIM), "float32"),
        out: T.Tensor((Q, HEADS, DIM), "bfloat16"),
    ):
        with T.Kernel(
            HEADS, Q, 4 // pack_warps, threads=pack_warps * 32
        ) as (bh, bi, bg):
            T.pdl_sync()
            T.pdl_trigger()
            tid = T.get_thread_binding()
            warp = tid // 32
            tx = tid % 32
            bd_i = bg * pack_warps + warp
            vmax = T.alloc_local((4,), "float32")
            vsum = T.alloc_local((4,), "float32")
            w_local = T.alloc_local((4 * local_items,), "float32")
            mx = T.alloc_fragment((1,), "float32")
            ls = T.alloc_fragment((1,), "float32")
            acc = T.alloc_local((1,), "float32")

            for g in T.serial(4):
                vmax[g] = -T.infinity("float32")
                for u in T.serial(local_items):
                    s = (tx + g * 32) * local_items + u
                    vmax[g] = T.max(
                        vmax[g],
                        T.if_then_else(s < n_split, pm[bi, bh, s], NEG_INF),
                    )
            merged_max = T.max(
                T.max(vmax[0], vmax[2]), T.max(vmax[1], vmax[3])
            )
            mx[0] = merged_max
            mx[0] = T.max(mx[0], T.shfl_xor(mx[0], 16))
            mx[0] = T.max(mx[0], T.shfl_xor(mx[0], 8))
            mx[0] = T.max(mx[0], T.shfl_xor(mx[0], 4))
            mx[0] = T.max(mx[0], T.shfl_xor(mx[0], 2))
            mx[0] = T.max(mx[0], T.shfl_xor(mx[0], 1))

            for g in T.serial(4):
                vsum[g] = 0.0
                for u in T.serial(local_items):
                    s = (tx + g * 32) * local_items + u
                    w_local[g * local_items + u] = T.if_then_else(
                        s < n_split, T.exp(pm[bi, bh, s] - mx[0]), 0.0
                    )
                    vsum[g] += T.if_then_else(
                        s < n_split,
                        T.call_extern("float32", "__fmul_rn", pl[bi, bh, s],
                                      w_local[g * local_items + u]),
                        0.0,
                    )
            merged_sum = (vsum[0] + vsum[2]) + (vsum[1] + vsum[3])
            ls[0] = merged_sum
            ls[0] += T.shfl_xor(ls[0], 16)
            ls[0] += T.shfl_xor(ls[0], 8)
            ls[0] += T.shfl_xor(ls[0], 4)
            ls[0] += T.shfl_xor(ls[0], 2)
            ls[0] += T.shfl_xor(ls[0], 1)

            acc[0] = 0.0
            for s in T.serial(n_split):
                acc[0] += T.shfl_sync(
                    w_local[
                        (s // local_items // 32) * local_items
                        + s % local_items
                    ],
                    (s // local_items) % 32,
                ) * pacc[bi, bh, s, bd_i * bd + tx]
            out[bi, bh, bd_i * bd + tx] = T.Cast(
                "bfloat16", acc[0] / ls[0]
            )

    return tilelang.compile(
        attn_combine_compact32_warp_pack,
        out_idx=None,
        pass_configs=PASS_CONFIGS,
    )


class DecodeAttention:
    """The attention launches of one Q8 layer, over arena-resident scratch.

    `scale` is a DEVICE tensor rather than baked in, so one compiled kernel serves every
    layer. The partial buffers are ONE flat allocation per role, viewed at each cell's
    split, so every captured graph records fixed addresses."""

    def __init__(self, *, ws, heads: int, dim: int, q_len: int):
        if (heads, dim, q_len) != (HEADS, DIM, Q):
            raise ValueError(
                f"attention is built for H{HEADS}/D{DIM}/Q{Q}, got "
                f"H{heads}/D{dim}/Q{q_len}")
        f32 = torch.float32
        n = Q * HEADS * NS_MAX
        self._scale = ws.slot(("attn", "scale"), (1,), f32)
        self._out = ws.slot(("attn", "out"), (Q, HEADS, DIM), torch.bfloat16)
        self._flat = (ws.slot(("attn", "pm"), (n,), f32),
                      ws.slot(("attn", "pl"), (n,), f32),
                      ws.slot(("attn", "pacc"), (n * DIM,), f32))
        self._scale_val = None
        # `q8_profile.install` substitutes the 2K cell's kernels through these.
        self._online_builder = build_online
        self._combine_builder = build_combine
        # The full read takes one split an SM; its extensions build now, before any
        # capture, on every rank (`attention_stream`).
        self.stream_splits = torch.cuda.get_device_properties(ws.device).multi_processor_count
        if self.stream_splits > NS_MAX:
            raise ValueError(f"{self.stream_splits} SMs exceed the partial arena's {NS_MAX}")
        self._stream = build_stream(self.stream_splits)
        self._stream_combine = build_stream_combine(self.stream_splits)

    def _parts(self, ns: int):
        """The (pm, pl, pacc) triple at this split, as views of the flat buffers."""
        pm, pl, pacc = self._flat
        n = Q * HEADS * ns
        return (pm[:n].view(Q, HEADS, ns), pl[:n].view(Q, HEADS, ns),
                pacc[:n * DIM].view(Q, HEADS, ns, DIM))

    @staticmethod
    def wants_inline_mask(k: torch.Tensor) -> bool:
        """Whether this read takes the [Q] int64 query positions instead of a mask: every
        main read does, the sliding layers' whole ring too; the draft ring's exact cell
        reads the byte mask."""
        return k.shape[0] in MAIN_READS or k.shape[0] == RING_KV

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *,
                 sink_k: torch.Tensor, scale: float, mask: torch.Tensor,
                 rows: int = Q) -> torch.Tensor:
        """q [Q, Hq, D]; k/v [KV, Hkv, D]; sink_k [Hkv, D] -> [Q, Hq, D] bf16."""
        kv = k.shape[0]
        # Refilled only on change: inside a captured graph `fill_` bakes the value in.
        if self._scale_val != scale:
            self._scale.fill_(scale)
            self._scale_val = scale
        out = self._out
        if rows != Q and (kv in MAIN_READS or kv == RING_KV):
            raise ValueError(f"a main read computes all {Q} rows, got rows={rows}")
        if kv == STREAM_KV:
            ns = self.stream_splits
            pm, pl, pacc = self._parts(ns)
            self._stream(q, k, v, sink_k, mask, self._scale, pm, pl, pacc)
            self._stream_combine(pm, pl, pacc, out)
            return out
        cell = ONLINE_GEOMETRY.get(kv)
        if cell is not None:
            bk, ns, group_split, compact32 = cell
            pm, pl, pacc = self._parts(ns)
            self._online_builder(kv, bk=bk, n_split=ns, group_split=group_split,
                                 ring=kv in RING_READS,
                                 window=WINDOW if kv == RING_KV else None)(
                q, k, v, sink_k, mask, self._scale, pm, pl, pacc, out)
            self._combine_builder(ns, compact32=compact32)(pm, pl, pacc, out)
            return out
        cell = DRAFT_GEOMETRY.get(kv)
        if cell is None:
            raise ValueError(
                f"no measured attention cell reads KV={kv}: the main stack reads "
                f"{list(MAIN_READS)} and its ring {RING_KV}, the draft ring "
                f"{sorted(DRAFT_GEOMETRY)}")
        bk, stages, ns = cell
        pm, pl, pacc = self._parts(ns)
        build_exact(kv, bk=bk, stages=stages, n_split=ns, rows=rows)(
            q, k, v, sink_k, mask, self._scale, pm, pl, pacc, out)
        return out
