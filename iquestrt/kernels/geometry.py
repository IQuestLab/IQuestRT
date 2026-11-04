"""Fuse draft positions, cache slots, masks, RoPE tables, and the MTP prologue.

Depth zero swaps the seed row into physical row zero using seed_first_offset.
Later depths broadcast one live token/hidden pair; draft_entry_eh_func computes
their eh_proj partial in the same launch. The NVLink tail reduces that partial.
Use the loaded eh_proj column width to select each rank's input slice.
"""

import functools

import tilelang
import tilelang.language as T
import torch

from .mask import NO_WS
from .mtp import PROLOGUE_THREADS, broadcast_operand_body, prologue_body
from .oproj import BM, EH_PROJ_CFG
from .position import seed_first_offset


def _geometry_body(q_len: int, ring_cap: int, kv_len: int, n_sink: int,
                   sliding_window: int, full: bool, rotary_dim: int, advance: int | None,
                   mask_rows: int | None):
    """The geometry's TileLang body for one CTA of any thread count, shared by
    `build_draft_geometry` and `build_draft_entry`. `advance=None` is depth 0's seed-first
    form, which reads the accepted count from `accepted`; the other forms take `accepted=None`."""
    if q_len <= 0 or kv_len <= 0 or ring_cap <= 0:
        raise ValueError(f"draft geometry needs positive Q/KV/capacity, got "
                         f"{q_len}/{kv_len}/{ring_cap}")
    if n_sink not in (0, 1):
        raise ValueError(f"draft geometry supports zero or one sink, got {n_sink}")
    seed_first = advance is None
    if not seed_first and not 1 <= advance <= q_len:
        raise ValueError(f"advance must be in 1..{q_len}, got {advance}")
    mask_rows = q_len if mask_rows is None else mask_rows
    if not 1 <= mask_rows <= q_len:
        raise ValueError(f"mask_rows must be in 1..{q_len}, got {mask_rows}")
    cols = kv_len + n_sink
    half = rotary_dim // 2
    n_base = 2 if seed_first else 1

    def place(base, r):
        """Row r's position: base + r, or depth 0's seed-first placement of n accepted."""
        if seed_first:
            return base[0] + seed_first_offset(q_len, r, base[1])
        return base[0] + T.Cast("int64", r)

    def ring_ref(base):
        """What `decode_mask_fused` reads as `length_dev - 1` after the positions kernel
        advanced by Q, or by n + 1 at depth 0."""
        if seed_first:
            return base[0] + base[1]
        return base[0] + T.int64(q_len - 1)

    @T.macro
    def body(length_dev, accepted, inv, positions, slots, mask, cos_out, sin_out):
        base = T.alloc_shared((n_base,), "int64")
        kpos_s = T.alloc_shared((cols,), "int64")
        tx = T.get_thread_binding()
        if tx == 0:
            base[0] = length_dev[0]
            if seed_first:
                base[1] = accepted[0]
        T.sync_threads()

        for r in T.Parallel(q_len):
            positions[r] = place(base, r)
            slots[0, r] = T.floormod(place(base, r), T.int64(ring_cap))

        # The key position of a slot does not depend on the query row, so the int64
        # modulo runs once per column (cols of them) instead of once per mask byte
        # (Q x cols): that division was most of the single CTA's time.
        for c in T.Parallel(cols):
            s = c - n_sink
            if full:
                kpos_s[c] = T.Cast("int64", s)
            else:
                p = ring_ref(base)
                kpos_s[c] = p - T.floormod(p - T.Cast("int64", s), T.int64(ring_cap))
        T.sync_threads()
        for z in T.Parallel(mask_rows * cols):
            r = z // cols
            c = z - r * cols
            qpos = place(base, r)
            if n_sink and c == 0:
                mask[r, c] = T.uint8(1)
            else:
                kpos = kpos_s[c]
                if sliding_window:
                    mask[r, c] = T.Cast(
                        "uint8", (kpos <= qpos) and (kpos >= T.int64(0))
                        and (kpos > qpos - T.int64(sliding_window)))
                else:
                    mask[r, c] = T.Cast("uint8", (kpos <= qpos) and (kpos >= T.int64(0)))

        for r, j in T.Parallel(q_len, half):
            ang = T.Cast("float32", place(base, r)) * inv[j]
            c = T.Cast("bfloat16", T.cos(ang))
            s = T.Cast("bfloat16", T.sin(ang))
            cos_out[r, j] = c
            sin_out[r, j] = s
            cos_out[r, half + j] = c
            sin_out[r, half + j] = s

        T.sync_threads()
        if tx == 0:
            if seed_first:
                length_dev[0] = base[0] + base[1] + T.int64(1)
            else:
                length_dev[0] = base[0] + T.int64(advance)

    return body


@functools.lru_cache(maxsize=None)
def build_draft_geometry(q_len: int, ring_cap: int, kv_len: int, n_sink: int,
                         sliding_window: int, full: bool, rotary_dim: int,
                         advance: int, threads: int = 256, mask_rows: int | None = None):
    body = _geometry_body(q_len, ring_cap, kv_len, n_sink, sliding_window, full,
                          rotary_dim, advance, mask_rows)
    cols = kv_len + n_sink

    @T.prim_func
    def draft_geometry(
        length_dev: T.Tensor((1,), "int64"),
        inv: T.Tensor((rotary_dim // 2,), "float32"),
        positions: T.Tensor((q_len,), "int64"),
        slots: T.Tensor((1, q_len), "int64"),
        mask: T.Tensor((q_len, cols), "uint8"),
        cos_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
        sin_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
    ):
        with T.Kernel(1, threads=threads):
            body(length_dev, None, inv, positions, slots, mask, cos_out, sin_out)

    return tilelang.compile(draft_geometry, out_idx=None, pass_configs=NO_WS)


def draft_entry_func(q_len: int, ring_cap: int, kv_len: int, n_sink: int,
                     sliding_window: int, full: bool, rotary_dim: int, advance: int | None,
                     mask_rows: int | None, hidden: int, w: int, col0: int, vocab: int,
                     use_embed: bool, eps: float):
    """Geometry and MTP prologue in q_len + 1 CTAs, before TileLang compilation. At depths
    1..K-1 the prologue's broadcast form; at depth 0 (`advance=None`, seed first) its Q
    rows, or its indexed hidden rows on a rank whose window is the hidden's."""
    seed_first = advance is None
    geometry = _geometry_body(q_len, ring_cap, kv_len, n_sink, sliding_window, full,
                              rotary_dim, advance, mask_rows)
    prologue = prologue_body(hidden, w, col0, use_embed, eps,
                             indexed_prev=seed_first and not use_embed,
                             broadcast_inputs=not seed_first)
    cols = kv_len + n_sink
    vt, ht = (vocab, hidden) if use_embed else (1, 1)

    if seed_first:
        # Depth 0 adds the accepted count and the prologue's Q-row inputs with their
        # indices.
        @T.prim_func
        def draft_entry_seed_first(
            length_dev: T.Tensor((1,), "int64"),
            inv: T.Tensor((rotary_dim // 2,), "float32"),
            positions: T.Tensor((q_len,), "int64"),
            slots: T.Tensor((1, q_len), "int64"),
            mask: T.Tensor((q_len, cols), "uint8"),
            cos_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
            sin_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
            accepted: T.Tensor((1,), "int64"),
            ids: T.Tensor((q_len,), "int64"),
            embed_tbl: T.Tensor((vt, ht), "bfloat16"),
            prev: T.Tensor((q_len, hidden), "bfloat16"),
            hidx: T.Tensor((q_len,), "int64"),
            wn: T.Tensor((hidden,), "float32"),
            out: T.Tensor((q_len, w), "bfloat16"),
        ):
            with T.Kernel(q_len + 1, threads=PROLOGUE_THREADS) as br:
                if br < q_len:
                    prologue(br, ids, embed_tbl, prev, hidx, wn, out)
                else:
                    geometry(length_dev, accepted, inv, positions, slots, mask, cos_out,
                             sin_out)
        return draft_entry_seed_first

    @T.prim_func
    def draft_entry(
        length_dev: T.Tensor((1,), "int64"),
        inv: T.Tensor((rotary_dim // 2,), "float32"),
        positions: T.Tensor((q_len,), "int64"),
        slots: T.Tensor((1, q_len), "int64"),
        mask: T.Tensor((q_len, cols), "uint8"),
        cos_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
        sin_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
        ids: T.Tensor((1,), "int64"),
        embed_tbl: T.Tensor((vt, ht), "bfloat16"),
        prev: T.Tensor((1, hidden), "bfloat16"),
        wn: T.Tensor((hidden,), "float32"),
        out: T.Tensor((q_len, w), "bfloat16"),
    ):
        with T.Kernel(q_len + 1, threads=PROLOGUE_THREADS) as br:
            if br < q_len:
                prologue(br, ids, embed_tbl, prev, None, wn, out)
            else:
                geometry(length_dev, None, inv, positions, slots, mask, cos_out, sin_out)

    return draft_entry


@functools.lru_cache(maxsize=None)
def build_draft_entry(*args):
    """`draft_entry_func(*args)`, compiled."""
    return tilelang.compile(draft_entry_func(*args), out_idx=None, pass_configs=NO_WS)


def draft_entry_eh_func(q_len: int, ring_cap: int, kv_len: int, n_sink: int,
                        sliding_window: int, full: bool, rotary_dim: int, advance: int,
                        mask_rows: int | None, hidden: int, w: int, col0: int, vocab: int,
                        use_embed: bool, eps: float):
    """A broadcast depth's geometry and eh_proj GEMV as ONE launch, the GEMV's CTAs
    building the broadcast prologue's operand themselves: the uncompiled TileLang function."""
    if advance is None:
        raise ValueError("depth 0 keeps its entry")
    bn, bk, threads = EH_PROJ_CFG["bn"], EH_PROJ_CFG["bk"], EH_PROJ_CFG["threads"]
    geometry = _geometry_body(q_len, ring_cap, kv_len, n_sink, sliding_window, full,
                              rotary_dim, advance, mask_rows)
    operand = broadcast_operand_body(hidden, w, col0, use_embed, eps, threads)
    nt, kt = hidden // bn, w // bk
    cols = kv_len + n_sink
    vt, ht = (vocab, hidden) if use_embed else (1, 1)

    @T.prim_func
    def draft_entry_eh(
        length_dev: T.Tensor((1,), "int64"),
        inv: T.Tensor((rotary_dim // 2,), "float32"),
        positions: T.Tensor((q_len,), "int64"),
        slots: T.Tensor((1, q_len), "int64"),
        mask: T.Tensor((q_len, cols), "uint8"),
        cos_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
        sin_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
        ids: T.Tensor((1,), "int64"),
        embed_tbl: T.Tensor((vt, ht), "bfloat16"),
        prev: T.Tensor((1, hidden), "bfloat16"),
        wn: T.Tensor((hidden,), "float32"),
        w_eh: T.Tensor((hidden, w), "bfloat16"),
        out: T.Tensor((BM, hidden), "bfloat16"),
    ):
        with T.Kernel(nt + 1, threads=threads) as bx:
            T.pdl_sync()
            if bx < nt:
                T.pdl_trigger()
                op_s = T.alloc_shared((w,), "bfloat16")
                # `oproj.build_oproj_gemv_q8_transposed`'s tiles and MMAs, its operand
                # tiles from `op_s`. The weight tiles' loads are issued first, so the
                # operand's norm runs while they land.
                x_s = T.alloc_shared((q_len, bk), "bfloat16")
                w_s = T.alloc_shared((bn, bk), "bfloat16")
                acc = T.alloc_fragment((bn, q_len), "float32")
                T.clear(acc)
                for ki in T.Pipelined(kt, num_stages=EH_PROJ_CFG["stages"]):
                    T.copy(w_eh[bx * bn, ki * bk], w_s)
                    if ki == 0:
                        operand(ids, embed_tbl, prev, wn, op_s)
                    for r, kk in T.Parallel(q_len, bk):
                        x_s[r, kk] = op_s[ki * bk + kk]
                    T.gemm(w_s, x_s, acc, transpose_B=True)
                for feature, row in T.Parallel(bn, q_len):
                    out[row, bx * bn + feature] = T.Cast("bfloat16", acc[feature, row])
            else:
                geometry(length_dev, None, inv, positions, slots, mask, cos_out, sin_out)
                # The eh tail has no PDL wait: release it only after these writes.
                T.sync_threads()
                T.call_extern("handle", "__threadfence")
                T.pdl_trigger()

    return draft_entry_eh


@functools.lru_cache(maxsize=None)
def build_draft_entry_eh(*args):
    """`draft_entry_eh_func(*args)`, compiled."""
    return tilelang.compile(draft_entry_eh_func(*args), out_idx=None, pass_configs=NO_WS)


class DraftGeometry:
    """Lazily compiled per static shape, with persistent outputs.

    `kv_len` and `advance` are fixed for a captured graph but only known at its warm-up,
    so the buffers are allocated on the first call under each key and kept -- the same
    arrangement as `DecodeMaskKernels`. Every key allocates its own set; the draft has
    two or three of them for the whole run. A call given `head` writes the same buffers."""

    def __init__(self, q_len: int, rotary_dim: int, *, device: str = "cuda"):
        self.q_len, self.rotary_dim, self.device = q_len, rotary_dim, device
        self._entries: dict = {}
        self._fused: dict = {}
        self._fused_eh: dict = {}

    def __call__(self, cache, layer: int, inv: torch.Tensor, *, kv_len: int, n_sink: int,
                 sliding_window: int | None, advance: int | None,
                 mask_rows: int | None = None, accepted=None, head=None, eh=None):
        """-> (positions[q], {capacity: slots[q]}, mask_u8[q, kv+sink], (cos, sin))."""
        if (advance is None) != (accepted is not None):
            raise ValueError("depth 0's seed-first geometry reads the accepted count, and "
                             "only it does")
        if accepted is not None and head is None:
            raise ValueError("depth 0's seed-first geometry runs with its MTP prologue")
        if eh is not None and (head is None or accepted is not None):
            raise ValueError("the eh_proj GEMV rides with a broadcast depth's prologue only")
        cap = cache.capacity(layer)
        full = cap >= cache.max_model_len
        window = 0 if sliding_window is None else int(sliding_window)
        key = (int(cap), int(kv_len), int(n_sink), window, bool(full),
               None if advance is None else int(advance), mask_rows)
        entry = self._entries.get(key)
        if entry is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(f"draft geometry: no buffers for {key} at capture time "
                                   "-- warm up before capturing")
            q, dev = self.q_len, self.device
            fn = (None if advance is None else
                  build_draft_geometry(q, key[0], key[1], key[2], key[3], key[4],
                                       self.rotary_dim, key[5], mask_rows=key[6]))
            entry = (fn,
                     torch.empty(q, dtype=torch.int64, device=dev),
                     torch.empty((1, q), dtype=torch.int64, device=dev),
                     torch.empty((q, key[1] + key[2]), dtype=torch.uint8, device=dev),
                     torch.empty((q, self.rotary_dim), dtype=torch.bfloat16, device=dev),
                     torch.empty((q, self.rotary_dim), dtype=torch.bfloat16, device=dev))
            self._entries[key] = entry
        fn, positions, slots, mask, cos, sin = entry
        if head is None:
            fn(cache.length_dev, inv, positions, slots, mask, cos, sin)
        else:
            pro, ids, embed_tbl, prev, enorm, hnorm, hidx = head
            if (hidx is None) != (advance is not None):
                raise ValueError("depth 0's prologue reads hidden-row indices, and only it")
            fns, build = ((self._fused, build_draft_entry) if eh is None else
                          (self._fused_eh, build_draft_entry_eh))
            fused = fns.get(key)
            if fused is None:
                fused = fns[key] = build(
                    self.q_len, *key[:5], self.rotary_dim, key[5], key[6],
                    pro.hidden, pro.w, pro.col0, pro.vocab, pro.use_embed, pro.eps)
            tbl, wn = (embed_tbl, enorm) if pro.use_embed else (pro.dummy, hnorm)
            if eh is not None:
                fused(cache.length_dev, inv, positions, slots, mask, cos, sin,
                      ids.reshape(1), tbl, prev.reshape(1, pro.hidden), wn, *eh)
            elif accepted is None:
                fused(cache.length_dev, inv, positions, slots, mask, cos, sin,
                      ids.reshape(1), tbl, prev.reshape(1, pro.hidden), wn, pro.out)
            else:
                # As `MtpPrologue`: an embedding-half rank reads its rows in order.
                fused(cache.length_dev, inv, positions, slots, mask, cos, sin, accepted,
                      ids.reshape(self.q_len), tbl, prev.reshape(self.q_len, pro.hidden),
                      pro.identity if pro.use_embed else hidx, wn, pro.out)
        return positions, {cap: slots[0]}, mask, (cos, sin)
