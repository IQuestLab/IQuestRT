"""Owned prefill snapshots for serial serving, independent of decode-written KV."""
from dataclasses import dataclass

import torch


@dataclass(eq=False)
class PrefixEntry:
    ids: tuple[int,...]
    layers: list[list[tuple[torch.Tensor, torch.Tensor]]]
    hidden: torch.Tensor
    nbytes: int
    on_gpu: bool
    draft_hidden: torch.Tensor | None = None
    next_token: int | None = None


class PrefixCache:
    def __init__(self, *, max_entries=2, gpu_bytes=1 << 30, max_bytes=16 << 30):
        self.entries: list[PrefixEntry] = []
        self.max_entries, self.gpu_bytes, self.max_bytes = max_entries, gpu_bytes, max_bytes

    def match(self, ids):
        candidates = [e for e in self.entries
                      if len(e.ids) <= len(ids) and tuple(ids[:len(e.ids)]) == e.ids]
        entry = max(candidates, key=lambda e: len(e.ids), default=None)
        if entry is not None:
            self.entries.remove(entry)
            self.entries.append(entry)
        return entry

    @staticmethod
    def restore(entry, caches):
        for index, (cache, layers) in enumerate(zip(caches, entry.layers, strict=True)):
            for layer, (k, v) in enumerate(layers):
                cache.k(layer)[:len(k)].copy_(k, non_blocking=entry.on_gpu)
                cache.v(layer)[:len(v)].copy_(v, non_blocking=entry.on_gpu)
            cache.length = len(entry.ids) - (index != 0)
            cache.length_dev.fill_(cache.length)
        return entry.hidden.to(caches[0].device)

    def store(self, ids, caches, hidden, *, draft_hidden=None, next_token=None):
        if len(ids) < 16 or self.max_entries < 1:
            return
        ids = tuple(ids)
        tensors = [[(cache.k(i)[:min(len(ids), spec.capacity)],
                     cache.v(i)[:min(len(ids), spec.capacity)])
                    for i, spec in enumerate(cache.specs)] for cache in caches]
        size = sum(t.numel() * t.element_size() for layers in tensors for kv in layers for t in kv)
        size += hidden[-1:].numel() * hidden.element_size()
        if draft_hidden is not None:
            size += draft_hidden[-1:].numel() * draft_hidden.element_size()
        # An uncacheable extension must not evict its usable ancestor.
        if size > self.max_bytes:
            return
        self.entries = [e for e in self.entries if ids[:len(e.ids)] != e.ids]
        while self.entries and (len(self.entries) >= self.max_entries
                                or size + sum(e.nbytes for e in self.entries) > self.max_bytes):
            self.entries.pop(0)
        on_gpu = (caches[0].device.type == "cuda" and
                  size + sum(e.nbytes for e in self.entries if e.on_gpu) <= self.gpu_bytes)

        def save(t):
            if on_gpu or t.device.type == "cpu":
                return t.clone()
            out = torch.empty(t.shape, dtype=t.dtype, device="cpu", pin_memory=True)
            out.copy_(t)  # Finish D2H before publishing an entry on any rank.
            return out

        layers = [[(save(k), save(v)) for k, v in ls] for ls in tensors]
        self.entries.append(PrefixEntry(
            ids, layers, save(hidden[-1:]), size, on_gpu,
            save(draft_hidden[-1:]) if draft_hidden is not None else None, next_token))
