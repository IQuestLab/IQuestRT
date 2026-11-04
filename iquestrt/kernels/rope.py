"""One step's rope cos/sin tables, one launch per distinct theta."""

import functools

import tilelang
import tilelang.language as T
import torch

NO_WS = {tilelang.transform.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True}


@functools.lru_cache(maxsize=None)
def build_rope_tables(q_len: int, rotary_dim: int, threads: int = 32):
    """(positions[q], inv[half]) -> (cos, sin) [q, rotary_dim] bf16, in ONE launch."""
    half = rotary_dim // 2

    @T.prim_func
    def rope_tables_fused(
        pos: T.Tensor((q_len,), "int64"),
        inv: T.Tensor((half,), "float32"),
        cos_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
        sin_out: T.Tensor((q_len, rotary_dim), "bfloat16"),
    ):
        with T.Kernel(1, threads=threads) as _:
            # `emb = cat([ang, ang])`, so column i and column i+half carry the same
            # angle; the two writes below are that duplication, not two computations.
            for r, j in T.Parallel(q_len, half):
                ang = T.Cast("float32", pos[r]) * inv[j]
                c = T.Cast("bfloat16", T.cos(ang))
                s = T.Cast("bfloat16", T.sin(ang))
                cos_out[r, j] = c
                sin_out[r, j] = s
                cos_out[r, half + j] = c
                sin_out[r, half + j] = s

    return tilelang.compile(rope_tables_fused, out_idx=None, pass_configs=NO_WS)


class RopeTables:
    """The fused table plus its output slots, ONE PAIR PER DISTINCT THETA."""

    def __init__(self, q_len: int, rotary_dim: int, n_tables: int = 2, *,
                 device="cuda", ws=None):
        self.q_len, self.rotary_dim = q_len, rotary_dim
        self.fn = build_rope_tables(q_len, rotary_dim)
        shape = (q_len, rotary_dim)

        def slot(key):
            if ws is not None:
                return ws.slot(("rope", key), shape, torch.bfloat16)
            return torch.zeros(shape, dtype=torch.bfloat16, device=device)

        self._pairs = [(slot(f"cos{i}"), slot(f"sin{i}")) for i in range(n_tables)]
        self._index: dict = {}

    def _pair(self, key):
        i = self._index.get(key)
        if i is None:
            if len(self._index) >= len(self._pairs):
                raise ValueError(
                    f"RopeTables was built for {len(self._pairs)} distinct thetas and a "
                    f"{len(self._index) + 1}th ({key}) appeared. The count comes from "
                    "the config and the slots are pre-allocated; reusing a pair would "
                    "make one theta's table overwrite another's.")
            i = self._index[key] = len(self._index)
        return self._pairs[i]

    def __call__(self, positions, inv, key=0):
        cos, sin = self._pair(key)
        self.fn(positions.reshape(self.q_len), inv, cos, sin)
        return cos, sin
