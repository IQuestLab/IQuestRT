"""One-launch decode attention masks."""

import functools

import tilelang
import tilelang.language as T
import torch


NO_WS = {tilelang.transform.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@functools.lru_cache(maxsize=None)
def build_decode_mask(q_len: int, kv_len: int, capacity: int,
                      sliding_window: int, n_sink: int, full: bool,
                      threads: int = 256):
    """Build ``[Q, KV+n_sink]`` uint8 visibility from absolute positions."""
    if q_len <= 0 or kv_len <= 0:
        raise ValueError(f"decode mask needs positive Q/KV, got {q_len}/{kv_len}")
    if capacity <= 0:
        raise ValueError(f"decode mask needs positive capacity, got {capacity}")
    if n_sink not in (0, 1):
        raise ValueError(f"decode mask supports zero or one sink, got {n_sink}")
    cols = kv_len + n_sink
    blocks = (q_len * cols + threads - 1) // threads

    @T.prim_func
    def decode_mask_fused(
        qpos: T.Tensor((q_len,), "int64"),
        length_dev: T.Tensor((1,), "int64"),
        out: T.Tensor((q_len, cols), "uint8"),
    ):
        with T.Kernel(blocks, threads=threads) as bx:
            for tx in T.Parallel(threads):
                z = bx * threads + tx
                if z < q_len * cols:
                    r = z // cols
                    c = z - r * cols
                    if n_sink and c == 0:
                        # The learned sink precedes every real key and is
                        # force-visible.
                        out[r, c] = T.uint8(1)
                    else:
                        s = c - n_sink
                        if full:
                            kpos = T.Cast("int64", s)
                        else:
                            # T.floormod matters before the first wrap because
                            # C/C++ '%' is negative for a negative dividend.
                            p = length_dev[0] - T.int64(1)
                            kpos = p - T.floormod(
                                p - T.Cast("int64", s), T.int64(capacity)
                            )
                        if sliding_window:
                            out[r, c] = T.Cast(
                                "uint8",
                                (kpos <= qpos[r])
                                and (kpos >= T.int64(0))
                                and (
                                    kpos
                                    > qpos[r] - T.int64(sliding_window)
                                ),
                            )
                        else:
                            out[r, c] = T.Cast(
                                "uint8",
                                (kpos <= qpos[r]) and (kpos >= T.int64(0)),
                            )

    return tilelang.compile(decode_mask_fused, out_idx=None, pass_configs=NO_WS)


class DecodeMaskKernels:
    """Lazily compiled, persistent mask buffers keyed by static graph shape.

    ``kv_len`` is fixed for the lifetime of a captured graph but is not known when
    ``Engine`` freezes its general workspace.  Buffers therefore live here and are
    allocated on the warm-up call before capture; retaining them in ``self._out`` keeps
    the addresses valid for every replay."""

    def __init__(self, q_len: int, *, device: str = "cuda"):
        self.q_len = q_len
        self.device = device
        self._fn = {}
        self._out = {}

    def __call__(self, q_positions: torch.Tensor, length_dev: torch.Tensor, *,
                 kv_len: int, capacity: int, max_model_len: int,
                 sliding_window: int | None, n_sink: int) -> torch.Tensor:
        full = capacity >= max_model_len
        window = 0 if sliding_window is None else int(sliding_window)
        key = (int(kv_len), int(capacity), window, int(n_sink), bool(full))
        fn = self._fn.get(key)
        if fn is None:
            fn = build_decode_mask(self.q_len, *key)
            self._fn[key] = fn
            self._out[key] = torch.empty(
                (self.q_len, key[0] + key[3]), dtype=torch.uint8, device=self.device
            )
        out = self._out[key]
        fn(q_positions.reshape(self.q_len), length_dev, out)
        return out
