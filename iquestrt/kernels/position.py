"""One-launch decode counter, positions and ring write slots."""

import functools

import tilelang
import tilelang.language as T
import torch


NO_WS = {tilelang.transform.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


def seed_first_logical(q_len: int, r):
    """Physical row r's logical row with the final seed row swapped to row zero, as a
    TileLang expression."""
    return T.if_then_else(r == 0, q_len - 1, T.if_then_else(r == q_len - 1, 0, r))


def seed_first_offset(q_len: int, r, n):
    """Physical row r's offset from the base at accepted count n, as a TileLang
    expression: the seed, logical row Q - 1, at n; logical row n, a copy of its inputs,
    at Q - 1. Shared with the draft geometry's depth 0
    (`geometry.py:draft_entry_func`)."""
    lo = T.Cast("int64", seed_first_logical(q_len, r))
    last = T.int64(q_len - 1)
    return T.if_then_else(lo == last, n, T.if_then_else(lo == n, last, lo))


@functools.lru_cache(maxsize=None)
def build_decode_positions(q_len: int, ring_capacities: tuple[int,...],
                           threads: int = 32):
    """Advance ``length_dev`` and write absolute positions plus ring slots."""
    if q_len <= 0 or q_len > threads:
        raise ValueError(f"decode positions needs 0 < Q <= threads, got {q_len}/{threads}")
    if any(c <= 0 for c in ring_capacities):
        raise ValueError(f"ring capacities must be positive: {ring_capacities}")
    if len(ring_capacities) > 1:
        raise ValueError(
            "decode positions currently supports one distinct ring capacity; "
            f"got {ring_capacities}")
    cap_extent = 1
    ring_cap = ring_capacities[0] if ring_capacities else 1

    @T.prim_func
    def decode_positions_fused(
        length_dev: T.Tensor((1,), "int64"),
        positions: T.Tensor((q_len,), "int64"),
        slots: T.Tensor((cap_extent, q_len), "int64"),
    ):
        with T.Kernel(1, threads=threads):
            base = T.alloc_shared((1,), "int64")
            tx = T.get_thread_binding()
            if tx == 0:
                base[0] = length_dev[0]
            T.sync_threads()

            for r in T.Parallel(q_len):
                positions[r] = base[0] + T.Cast("int64", r)
            # The target model has exactly one SWA capacity.  Full-only MTP caches use a
            # dummy modulo-1 row which no caller exposes; keeping one static ABI avoids a
            # zero-extent tensor and produces no extra launch.
            for r in T.Parallel(q_len):
                slots[0, r] = T.floormod(
                    base[0] + T.Cast("int64", r), T.int64(ring_cap))

            # Do not let lane 0 update the global counter while another lane still reads
            # the old base to produce a slot.  Consumers launch only after kernel exit.
            T.sync_threads()
            if tx == 0:
                length_dev[0] = base[0] + T.int64(q_len)

    return tilelang.compile(decode_positions_fused, out_idx=None, pass_configs=NO_WS)


class DecodePositionKernels:
    """Persistent outputs keyed by the cache's distinct ring capacities."""

    def __init__(self, q_len: int, *, device: str = "cuda"):
        self.q_len = q_len
        self.device = device
        self._entries = {}

    def __call__(self, cache):
        """-> (positions[q], {capacity: slots[q]}), advancing `cache.length_dev`."""
        caps = tuple(sorted({s.capacity for s in cache.specs
                             if s.capacity < cache.max_model_len}))
        entry = self._entries.get(caps)
        if entry is None:
            fn = build_decode_positions(self.q_len, caps)
            positions = torch.empty(self.q_len, dtype=torch.int64, device=self.device)
            slots = torch.empty((1, self.q_len), dtype=torch.int64,
                                device=self.device)
            entry = (fn, positions, slots)
            self._entries[caps] = entry
        fn, positions, slots = entry
        fn(cache.length_dev, positions, slots)
        return positions, {cap: slots[i] for i, cap in enumerate(caps)}
