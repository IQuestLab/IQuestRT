"""Full-attention caches and sliding-window rings.

A Q-row verify writes before reading, so ring capacity must be at least W+Q-1.
Prefill reads retained history before writes; settlement keeps only committed rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import Config, ShardSpec


@dataclass(frozen=True)
class LayerCacheSpec:
    idx: int
    capacity: int              # slots in tokens
    window: int | None         # None => full attention

    @property
    def is_ring(self) -> bool:
        """Whether this layer has a sliding window at all."""
        return self.window is not None


class KVCache:
    """Per-layer K/V storage for a single sequence (batch=1).

    Not a pool and not a paged allocator: iquestrt targets one in-flight request, so
    the cache is a fixed set of per-layer tensors sized once at construction."""

    def __init__(
        self,
        cfg: Config,
        shard: ShardSpec,
        *,
        max_model_len: int,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        ring_extra: int = 0,
    ):
        if max_model_len <= 0:
            raise ValueError(f"max_model_len must be positive, got {max_model_len}")

        self.device = torch.device(device)
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.kv_heads = shard.num_kv_heads
        self.head_dim = cfg.head_dim

        self.specs: list[LayerCacheSpec] = []
        self._k: list[torch.Tensor] = []
        self._v: list[torch.Tensor] = []
        # THE DRAFT GETS ITS OWN KVCache, not extra layers here, because `length` is one
        # counter for the whole object: the draft's branch holds entries written from
        # DRAFT hiddens which are rewound at the end of every step, and the stack's are
        # committed. One counter cannot describe both.
        for spec in cfg.layers:
            window = spec.sliding_window
            if window is None:
                cap = max_model_len
            else:
                extra = ring_extra
                cap = min(max_model_len, window + extra)
            self.specs.append(LayerCacheSpec(spec.idx, cap, window))
            buf = torch.zeros((2, cap, self.kv_heads, self.head_dim),
                              dtype=dtype, device=self.device)
            self._k.append(buf[0])
            self._v.append(buf[1])

        # Position of the next token to be written; the only mutable state.
        self.length = 0
        # The same counter on the DEVICE. CUDA Graph capture bakes every host-side value
        # into the graph, so a replayed step would keep reusing capture-time positions:
        # rope would rotate by the wrong angle and the mask would admit the wrong slots,
        # silently. Keeping the authoritative position in a device tensor is what makes
        # a captured decode step correct on step N.
        self.length_dev = torch.zeros(1, dtype=torch.int64, device=self.device)

    def k(self, layer: int) -> torch.Tensor:
        return self._k[layer]

    def v(self, layer: int) -> torch.Tensor:
        return self._v[layer]

    def capacity(self, layer: int) -> int:
        return self.specs[layer].capacity

    # -- write ---------------------------------------------------------------
    def slots(self, layer: int, positions: torch.Tensor, cache=None) -> torch.Tensor:
        """Ring slot for each absolute position, [S] on device."""
        cap = self.specs[layer].capacity
        if cap >= self.max_model_len:
            return positions
        if cache is None:
            return positions % cap
        t = cache.get(cap)
        if t is None:
            t = positions % cap
            cache[cap] = t
        return t

    def write(self, layer: int, positions: torch.Tensor, key: torch.Tensor,
              value: torch.Tensor) -> None:
        """Scatter [S, kv_heads, head_dim] key/value into this layer's ring."""
        s = positions.shape[0]
        for name, t in (("key", key), ("value", value)):
            if t.shape != (s, self.kv_heads, self.head_dim):
                raise ValueError(
                    f"{name} must be [{s}, {self.kv_heads}, {self.head_dim}], "
                    f"got {tuple(t.shape)}"
                )
        cap = self.specs[layer].capacity
        if s > cap:
            positions, key, value = positions[s - cap:], key[s - cap:], value[s - cap:]
        slot = self.slots(layer, positions)
        self._k[layer].index_copy_(0, slot, key.to(self.dtype))
        self._v[layer].index_copy_(0, slot, value.to(self.dtype))

    def advance(self, n: int, *, update_device: bool = True) -> None:
        if self.length + n > self.max_model_len:
            raise ValueError(
                f"sequence would reach {self.length + n} > max_model_len="
                f"{self.max_model_len}"
            )
        self.length += n
        # In-place so the update is a captured op rather than a host assignment. The
        # bound check above stays on the host: it reads `self.length`, never the device
        # tensor, so it costs no sync and stays valid under replay as long as the caller
        # keeps calling advance() -- which every forward does.
        if update_device:
            self.length_dev += n

    def rewind(self, n: int) -> None:
        """Un-advance by `n` positions after a speculative step over-committed.

        A verify step advances by Q because the layer loop needs `length` to cover its
        own queries; only 1 + n_accepted of those positions survive. Rewinding leaves
        the rejected K/V physically in the buffers -- deliberately: the next step
        overwrites it before reading, and erasing it would be one
        extra scatter per layer per step."""
        if not 0 <= n <= self.length:
            raise ValueError(f"cannot rewind {n} of a sequence of length {self.length}")
        self.length -= n
        self.length_dev -= n

    def reset(self) -> None:
        self.length = 0
        self.length_dev.zero_()

    def read_prefix(self, layer: int, n: int | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """First `min(n, capacity)` slots in SLOT order -- fixed shape, zero ops."""
        cap = self.specs[layer].capacity
        if n is None or n >= cap:
            return self._k[layer], self._v[layer]
        return self._k[layer][:n], self._v[layer][:n]
