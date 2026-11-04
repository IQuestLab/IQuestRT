"""Aligned arena for decode buffers. Addresses remain fixed while graphs use them."""

from __future__ import annotations

import torch

ALIGN = 1024   # Meets the decode kernels' TMA and 128-byte alignment requirements.

_DT_BYTES = {
    torch.bfloat16: 2, torch.float16: 2, torch.float32: 4,
    torch.int32: 4, torch.int64: 8, torch.uint8: 1, torch.bool: 1,
    torch.float8_e4m3fn: 1,
}


class Workspace:
    """Byte arena handing out stable, `ALIGN`-aligned views.

    Keys are opaque; the convention used by callers is `(kind, layer, name)` so that a
    per-layer stride falls out of the ordering rather than having to be declared. Two
    requests with the same key MUST agree on shape and dtype -- disagreeing means two
    different tensors were given one name, which would alias them into the same memory and
    produce wrong numbers with no error."""

    def __init__(self, *, device="cuda", align: int = ALIGN, symm_group: str):
        self.device = torch.device(device)
        self.align = align
        # The process-group name `symm=True` slots rendezvous with.
        self.symm_group = symm_group
        self._symm: dict[object, torch.Tensor] = {}
        self._slots: dict[object, tuple[int, tuple[int,...], torch.dtype]] = {}
        self._cursor = 0
        self._buf: torch.Tensor | None = None
        self._views: dict[object, torch.Tensor] = {}
        self._handed_out: dict[object, torch.Tensor] = {}

    # -- registration --------------------------------------------------------
    def slot(self, key, shape, dtype: torch.dtype,
             symm: bool = False) -> torch.Tensor:
        """Register (or look up) a slot and return a tensor for it."""
        shape = tuple(int(s) for s in shape)
        if symm:
            if key in self._symm:
                t = self._symm[key]
                if tuple(t.shape) != shape or t.dtype != dtype:
                    raise ValueError(
                        f"symmetric slot {key!r} already registered as "
                        f"{tuple(t.shape)}/{t.dtype}, requested {shape}/{dtype}")
                return t
            self._symm[key] = self._alloc_symm(shape, dtype)
            return self._symm[key]
        if key in self._slots:
            off, sh, dt = self._slots[key]
            if sh != shape or dt != dtype:
                raise ValueError(
                    f"slot {key!r} already registered as {sh}/{dt}, requested "
                    f"{shape}/{dtype}. Two tensors sharing one key would ALIAS -- that is "
                    "wrong numbers with no error, so this refuses instead of resizing."
                )
            return self._view(key)
        if self._buf is not None:
            raise RuntimeError(
                f"slot {key!r} requested after freeze(). Growing the arena would "
                "reallocate it and move every address already handed out; after CUDA "
                "Graph capture the graph would keep writing to the old pointers, "
                "silently. Register every slot before freezing."
            )
        if dtype not in _DT_BYTES:
            raise ValueError(f"unsupported dtype {dtype}")
        nbytes = _DT_BYTES[dtype]
        for s in shape:
            nbytes *= s
        self._slots[key] = (self._cursor, shape, dtype)
        self._cursor += -(-nbytes // self.align) * self.align
        return self._view(key)

    def _alloc_symm(self, shape, dtype) -> torch.Tensor:
        """A zeroed symmetric tensor, rendezvoused with `symm_group`."""
        import torch.distributed._symmetric_memory as symm_mem

        t = symm_mem.empty(*shape, dtype=dtype, device=self.device)
        t.zero_()
        symm_mem.rendezvous(t, self.symm_group)
        return t

    def freeze(self) -> "Workspace":
        """Allocate the arena once and move every placeholder, contents included, into it.

        Idempotent; further `slot()` keys are refused."""
        if self._buf is None:
            self._buf = torch.zeros(max(self._cursor, self.align),
                                    dtype=torch.uint8, device=self.device)
            self._views.clear()
            for key, ph in self._handed_out.items():
                final = self._view(key)
                final.copy_(ph)         # what the holder wrote before the freeze
                ph.set_(final)          # Preserve tensor identity while fixing its storage.
                if ph.data_ptr() != final.data_ptr():
                    raise RuntimeError(f"failed to repoint placeholder for {key!r}")
            self._handed_out.clear()
        return self

    # -- access --------------------------------------------------------------
    def _view(self, key) -> torch.Tensor:
        if key in self._views:
            return self._views[key]
        off, shape, dtype = self._slots[key]
        if self._buf is None:
            # Pre-freeze: a placeholder, remembered so `freeze()` can repoint it.
            if key not in self._handed_out:
                self._handed_out[key] = torch.zeros(shape, dtype=dtype,
                                                    device=self.device)
            return self._handed_out[key]
        n = 1
        for s in shape:
            n *= s
        v = self._buf[off:off + n * _DT_BYTES[dtype]].view(dtype).view(shape)
        self._views[key] = v
        return v
