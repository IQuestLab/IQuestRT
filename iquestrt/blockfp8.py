"""Block-FP8 checkpoint weights: dequantization and rank-local scales."""

from __future__ import annotations

import torch

BLOCK = 128


def dequant(q: torch.Tensor, scale: torch.Tensor, block: tuple[int, int] = (BLOCK, BLOCK),
            dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """`q [..., R, C]` E4M3 with `scale [..., R/br, C/bc]` -> `q * scale[block]`.

    `block` is the (row, column) extent one scale covers: the checkpoint's 128x128,
    or the granularity of a `shard_scale` table. fp64 is exact; fp32 rounds once."""
    br, bc = block
    if q.shape[-2] != scale.shape[-2] * br or q.shape[-1] != scale.shape[-1] * bc:
        raise ValueError(f"scale {tuple(scale.shape)} at block {block} does not tile "
                         f"weight {tuple(q.shape)}")
    s = scale.to(dtype).repeat_interleave(br, -2).repeat_interleave(bc, -1)
    return q.to(dtype) * s


def shard_scale(scale: torch.Tensor, dim: int, start: int, length: int,
                grain: int) -> torch.Tensor:
    """The scale table of the weight slice `[start, start + length)` along `dim`.

    `scale` is a checkpoint table (one entry per 128 along both weight dims). The
    result has one entry per `grain` along `dim`, each the scale of the 128-block
    that chunk lies in, and the other dims unchanged. `grain` must divide 128 and
    tile the slice, which is what makes every chunk lie in exactly one block."""
    if BLOCK % grain or start % grain or length % grain:
        raise ValueError(f"grain {grain} does not tile slice [{start}, {start + length}) "
                         f"inside {BLOCK}-blocks")
    blocks = (start + torch.arange(0, length, grain, device=scale.device)) // BLOCK
    return scale.index_select(dim, blocks)
