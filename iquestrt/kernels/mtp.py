"""The MTP head's prologue as one kernel instead of seven ops."""

import functools

import tilelang
import tilelang.language as T
import torch

NO_WS = {tilelang.transform.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def eh_window(tp_rank: int, tp_size: int, hidden: int, w: int):
    """-> (use_embed, col0), or None when the rank's window straddles the seam.

    `None` is not an error: it is the tp=1 case, where one rank owns the whole 2H
    concatenation. The caller keeps the torch path for it."""
    lo, hi = tp_rank * w, (tp_rank + 1) * w
    if hi <= hidden:
        return True, lo
    if lo >= hidden:
        return False, lo - hidden
    return None


PROLOGUE_THREADS = 128     # a row's CTA; its reduction order is the thread count's
# How TileLang lays the norm's squares over those threads: thread t's partial adds
# sq[i * 512 + 4 * t + j] for j in 0..3, i in 0..5 (`broadcast_operand_body` repeats it).
PROLOGUE_VEC = 4


def prologue_body(hidden: int, w: int, col0: int, use_embed: bool, eps: float,
                  indexed_prev: bool = False, broadcast_inputs: bool = False):
    """Row `br`'s TileLang body at `PROLOGUE_THREADS` threads, shared by
    `build_mtp_prologue` and the draft geometry's launch that carries it
    (`geometry.build_draft_entry`)."""
    if indexed_prev and broadcast_inputs:
        raise ValueError("indexed_prev and broadcast_inputs are separate MTP paths")

    @T.macro
    def body(br, ids, embed_tbl, prev, hidx, wn, out):
        xs = T.alloc_shared((hidden,), "float32")
        sq = T.alloc_shared((hidden,), "float32")
        inv = T.alloc_shared((1,), "float32")
        acc = T.alloc_fragment((1,), "float32")
        row = T.alloc_fragment((1,), "int32")

        if use_embed:
            # `embed`; every row keeps it, position 0's too.
            row[0] = T.Cast(
                "int32", ids[0 if broadcast_inputs else br]
            )
            for c in T.Parallel(hidden):
                xs[c] = T.Cast("float32", embed_tbl[row[0], c])
                sq[c] = xs[c] * xs[c]
        else:
            if indexed_prev:
                row[0] = T.Cast("int32", hidx[br])
            for c in T.Parallel(hidden):
                xs[c] = T.Cast(
                    "float32",
                    prev[
                        0 if broadcast_inputs
                        else row[0] if indexed_prev
                        else br,
                        c,
                    ],
                )
                sq[c] = xs[c] * xs[c]

        # RMSNorm: one reduction, the scalar broadcast through shared, two roundings.
        T.clear(acc)
        T.reduce_sum(sq, acc, dim=0)
        if T.get_thread_binding() == 0:
            inv[0] = T.rsqrt(acc[0] / T.Cast("float32", hidden) + eps)
        # The REDUCTION is over all `hidden`; only `w` of the normalised columns
        # are this rank's, and `col0` is where its window starts. Both `wn` and `xs`
        # are indexed by the full column -- normalising the window alone would be a
        # different (and wrong) statistic.
        for c in T.Parallel(w):
            out[br, c] = T.Cast("bfloat16",
                                wn[col0 + c] * (xs[col0 + c] * inv[0]))

    return body


def broadcast_operand_body(hidden: int, w: int, col0: int, use_embed: bool, eps: float,
                           threads: int):
    """The broadcast prologue's operand row into a shared [w] buffer at `threads` threads,
    bit for bit (`geometry.py:draft_entry_eh_func`): thread t computes the partials of
    the prologue's threads t and t + 64, then `tl::AllReduce`'s stages over 128."""
    if 2 * threads != PROLOGUE_THREADS:
        raise ValueError(f"the broadcast operand runs at {PROLOGUE_THREADS // 2} threads, "
                         f"got {threads}")
    span = PROLOGUE_THREADS * PROLOGUE_VEC
    if hidden % span:
        raise ValueError(f"hidden={hidden} is not a multiple of {span}")
    iters = hidden // span

    @T.macro
    def body(ids, embed_tbl, prev, wn, op_s):
        xs = T.alloc_shared((hidden,), "float32")
        sq = T.alloc_shared((hidden,), "float32")
        red = T.alloc_shared((threads,), "float32")
        inv = T.alloc_shared((1,), "float32")
        row = T.alloc_fragment((1,), "int32")
        part = T.alloc_local((2,), "float32")
        tx = T.get_thread_binding()

        if use_embed:
            row[0] = T.Cast("int32", ids[0])
            for c in T.Parallel(hidden):
                xs[c] = T.Cast("float32", embed_tbl[row[0], c])
                sq[c] = xs[c] * xs[c]
        else:
            for c in T.Parallel(hidden):
                xs[c] = T.Cast("float32", prev[0, c])
                sq[c] = xs[c] * xs[c]
        T.sync_threads()

        # The prologue's thread t and t + 64, each in its own order (`PROLOGUE_VEC`).
        part[0] = T.float32(0)
        part[1] = T.float32(0)
        for j in T.serial(PROLOGUE_VEC):
            for i in T.serial(iters):
                part[0] = part[0] + sq[i * span + tx * PROLOGUE_VEC + j]
                part[1] = part[1] + sq[i * span + (tx + threads) * PROLOGUE_VEC + j]
        # `tl::AllReduce<SumOp, 128>`: offset 64 is the pair's own sum, offset 32 goes
        # through shared memory, 16 down to 1 by shuffles, each `x + partner`.
        part[0] = part[0] + part[1]
        red[tx] = part[0]
        T.sync_threads()
        part[0] = part[0] + red[T.bitwise_xor(tx, 32)]
        part[0] = part[0] + T.shfl_xor(part[0], 16)
        part[0] = part[0] + T.shfl_xor(part[0], 8)
        part[0] = part[0] + T.shfl_xor(part[0], 4)
        part[0] = part[0] + T.shfl_xor(part[0], 2)
        part[0] = part[0] + T.shfl_xor(part[0], 1)
        if tx == 0:
            inv[0] = T.rsqrt(part[0] / T.Cast("float32", hidden) + eps)
        T.sync_threads()
        for c in T.Parallel(w):
            op_s[c] = T.Cast("bfloat16", wn[col0 + c] * (xs[col0 + c] * inv[0]))

    return body


@functools.lru_cache(maxsize=None)
def build_mtp_prologue(hidden: int, w: int, col0: int, q_len: int, vocab: int,
                       use_embed: bool, eps: float, indexed_prev: bool = False,
                       broadcast_inputs: bool = False):
    """-> out[q_len, w] bf16, the `eh_proj` operand, already narrowed to this rank."""
    body = prologue_body(hidden, w, col0, use_embed, eps, indexed_prev, broadcast_inputs)
    vt = vocab if use_embed else 1
    ht = hidden if use_embed else 1
    input_q = 1 if broadcast_inputs else q_len

    @T.prim_func
    def mtp_prologue(
        ids: T.Tensor((input_q,), "int64"),
        embed_tbl: T.Tensor((vt, ht), "bfloat16"),
        prev: T.Tensor((input_q, hidden), "bfloat16"),
        hidx: T.Tensor((q_len,), "int64"),
        wn: T.Tensor((hidden,), "float32"),
        out: T.Tensor((q_len, w), "bfloat16"),
    ):
        with T.Kernel(q_len, threads=PROLOGUE_THREADS) as br:
            body(br, ids, embed_tbl, prev, hidx, wn, out)

    return tilelang.compile(mtp_prologue, out_idx=None, pass_configs=NO_WS)


class MtpPrologue:
    """One kernel per head weight-set, plus the narrowed operand slot.

    Built lazily on first use: the window depends only on (tp_rank, tp_size, hidden, w)
    and `w` comes off the loaded `eh_proj`, so there is nothing to decide at __init__
    that is not already decided by the weights."""

    def __init__(self, hidden: int, w: int, q_len: int, vocab: int, eps: float,
                 tp_rank: int, tp_size: int, *, device="cuda", ws=None):
        self.hidden, self.w, self.q_len = hidden, w, q_len
        win = eh_window(tp_rank, tp_size, hidden, w)
        # `None` => the window straddles enorm|hnorm (tp=1). The caller checks `usable`
        # and keeps the torch path; refusing here rather than silently computing half the
        # concatenation is the point.
        self.usable = win is not None
        self.vocab, self.eps = vocab, float(eps)
        if not self.usable:
            self.fn = None
            return
        self.use_embed, self.col0 = win
        self.fn = build_mtp_prologue(hidden, w, self.col0, q_len, vocab,
                                     self.use_embed, float(eps))
        self.fn_indexed = (
            None
            if self.use_embed
            else build_mtp_prologue(
                hidden, w, self.col0, q_len, vocab,
                self.use_embed, float(eps), indexed_prev=True,
            )
        )
        self.fn_broadcast = build_mtp_prologue(
            hidden, w, self.col0, q_len, vocab,
            self.use_embed, float(eps), broadcast_inputs=True,
        )
        if ws is not None:
            self.out = ws.slot(("mtp", "ehop"), (q_len, w), torch.bfloat16)
            self.dummy = ws.slot(("mtp", "none"), (1, 1), torch.bfloat16)
            self.identity = ws.slot(("mtp", "identity"), (q_len,), torch.int64)
        else:
            self.out = torch.zeros(q_len, w, dtype=torch.bfloat16, device=device)
            self.dummy = torch.zeros(1, 1, dtype=torch.bfloat16, device=device)
            self.identity = torch.arange(q_len, dtype=torch.int64, device=device)
        self.identity.copy_(torch.arange(q_len, dtype=torch.int64, device=device))

    def __call__(self, ids, embed_tbl, prev, enorm, hnorm,
                 prev_hidx=None, broadcast_inputs: bool = False):
        """-> [q_len, w] bf16 -- `cat[enorm(e) | hnorm(prev)].narrow(-1, rank*w, w)`.

        `enorm`/`hnorm` are both passed even though one is used, so the call site does
        not have to know which half this rank owns."""
        q, hd = self.q_len, self.hidden
        # The full [hidden] weight vector either way: `col0` is compiled in, and the
        # reduction needs every column regardless of which `w` of them get written.
        wn = enorm if self.use_embed else hnorm
        if broadcast_inputs and prev_hidx is not None:
            raise ValueError(
                "broadcast MTP inputs cannot also carry a previous-row index"
            )
        indexed = prev_hidx is not None and not self.use_embed
        fn = (
            self.fn_broadcast if broadcast_inputs
            else self.fn_indexed if indexed
            else self.fn
        )
        hidx = prev_hidx if indexed else self.identity
        input_q = 1 if broadcast_inputs else q
        fn(ids.reshape(input_q),
           embed_tbl if self.use_embed else self.dummy,
           prev.reshape(input_q, hd), hidx.reshape(q), wn, self.out)
        return self.out.view(q, self.w)
