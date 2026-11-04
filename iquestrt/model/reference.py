"""Shared prefill algebra with FP32-dequantized linear operations.

Layer zero uses raw-hidden residuals; other main layers use normalized residuals.
The recursive draft retains FP32 residuals. model/prefill.py supplies the tiled
attention and grouped expert kernels.

This is prefill algebra, not a bit-exact decode oracle. Decode quantizes MoE
inputs and intermediate activations to E4M3, normalizes Q/K from FP32 projection
sums instead of BF16-rounded sums, folds routers, and uses NVLink reductions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..blockfp8 import dequant
from ..config import LayerSpec
from ..kv import KVCache
from ..weights import LayerWeights, ModelWeights, MtpWeights


def linear(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor,
           block: tuple[int, int] = (128, 128)) -> torch.Tensor:
    """`x @ dequant(q, scale).T` in fp32, rounded to `x.dtype`."""
    w = dequant(q, scale, block, dtype=torch.float32)
    return F.linear(x.float(), w).to(x.dtype)


def rms_norm(x: torch.Tensor, weight_fp32: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm in fp32, weight applied in fp32, cast back — matches the reference."""
    dt = x.dtype
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    xf = xf * torch.rsqrt(var + eps)
    return (weight_fp32 * xf).to(dt)


def rope_tables(
    positions: torch.Tensor, rotary_dim: int, theta: float, device, dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    half = rotary_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    ang = positions.float().unsqueeze(-1) * inv  # [S, half]
    # Neox layout: the table is duplicated so cos/sin line up with [x1; x2].
    emb = torch.cat([ang, ang], dim=-1)          # [S, rotary_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_partial_rope(
    t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rotary_dim: int
) -> torch.Tensor:
    """Rotate only the first `rotary_dim` channels; the rest must pass bit-exact.

    t: [S, heads, head_dim];  cos/sin: [S, rotary_dim]"""
    if rotary_dim == t.shape[-1]:
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        return t * c + _rotate_half(t) * s
    rot, keep = t[..., :rotary_dim], t[..., rotary_dim:]
    c, s = cos.unsqueeze(1), sin.unsqueeze(1)
    rot = rot * c + _rotate_half(rot) * s
    return torch.cat([rot, keep], dim=-1)


def dense_mlp(x: torch.Tensor, lw: LayerWeights) -> torch.Tensor:
    gu = linear(x, lw.dense_gate_up, lw.gate_up_scale)
    half = gu.shape[-1] // 2
    act = F.silu(gu[:, :half]) * gu[:, half:]
    return linear(act, lw.dense_down, lw.down_scale)


def layer_forward(
    hidden: torch.Tensor,     # [S, H]
    positions: torch.Tensor,  # [S]
    spec: LayerSpec,
    lw: LayerWeights,
    mw: ModelWeights,
    *,
    rope: tuple[torch.Tensor, torch.Tensor],
    cache: "KVCache | None" = None,
    all_reduce=None,
    fp32_residual: bool = False,
    attention_fn,
    moe_fn,
    prefix_length=0,
) -> torch.Tensor:
    """One decoder layer over the supplied absolute positions."""
    cfg = mw.cfg
    # Scales live in the weights when folding is on; applying them again here
    # would square them.
    a_scale = 1.0 if mw.folded else spec.attn_out_scale
    f_scale = 1.0 if mw.folded else spec.ffn_out_scale
    attn_scale = 1.0 if mw.folded else cfg.softmax_scale
    cos, sin = rope

    xn = rms_norm(hidden, lw.attention_norm, cfg.rms_norm_eps)
    qkv = linear(xn, lw.qkv, lw.qkv_scale)
    qd, kd = mw.shard.q_dim, mw.shard.kv_dim
    q, k, v = qkv.split([qd, kd, kd], dim=-1)
    hd = cfg.head_dim
    q = q.view(-1, mw.shard.num_q_heads, hd)
    k = k.view(-1, mw.shard.num_kv_heads, hd)
    v = v.view(-1, mw.shard.num_kv_heads, hd)
    q = rms_norm(q, lw.q_norm, cfg.rms_norm_eps)
    k = rms_norm(k, lw.k_norm, cfg.rms_norm_eps)
    q = apply_partial_rope(q, cos, sin, cfg.rotary_dim)
    k = apply_partial_rope(k, cos, sin, cfg.rotary_dim)

    prefix = {}
    if prefix_length:
        if cache is None:
            raise ValueError("suffix prefill requires a cache")
        prefix = dict(prefix_k=cache.k(spec.idx), prefix_v=cache.v(spec.idx),
                      prefix_length=prefix_length)
    ao = attention_fn(q, k, v, sink_k=lw.sink_k, scale=attn_scale,
                      sliding_window=spec.sliding_window, **prefix)
    if cache is not None:
        cache.write(spec.idx, positions, k, v)
    ao = linear(ao.reshape(-1, mw.shard.q_dim), lw.o_proj, lw.o_scale)
    residual_base = hidden if spec.use_sandwich_norm else xn
    if fp32_residual:
        if not spec.use_sandwich_norm:
            raise ValueError("an fp32 residual is defined for the sandwich draft layer only")
        residual_base = residual_base.float()
    # o_proj is ROW-parallel (its contraction dim is sharded), so each rank holds a
    # PARTIAL sum here. The reduce must happen BEFORE attn_out_norm: RMSNorm is
    # non-linear, so normalising a partial sum gives a wrong answer that still looks
    # like a plausible activation.
    if all_reduce is not None:
        all_reduce(ao)
    h = residual_base + rms_norm(ao, lw.attn_out_norm, cfg.rms_norm_eps) * a_scale
    hn = rms_norm(h, lw.feed_forward_norm, cfg.rms_norm_eps).to(ao.dtype)
    mlp_out = dense_mlp(hn, lw) if spec.is_dense_mlp else moe_fn(hn, lw, cfg)
    if all_reduce is not None:
        all_reduce(mlp_out)                  # experts_proj / dense_down are row-parallel
    if spec.use_sandwich_norm:
        mlp_out = rms_norm(mlp_out, lw.ffn_out_norm, cfg.rms_norm_eps)
    return h + mlp_out * f_scale


def embed(input_ids: torch.Tensor, mw: ModelWeights) -> torch.Tensor:
    """Look up token embeddings from the loaded checkpoint."""
    return mw.embed_tokens[input_ids]


@torch.inference_mode()
def forward(
    input_ids: torch.Tensor,
    mw: ModelWeights,
    cache: "KVCache | None" = None,
    *,
    all_reduce=None,
    attention_fn,
    moe_fn,
    suffix=False,
) -> torch.Tensor:
    """Input rows -> [S, H] post-final-norm hidden, at zero unless `suffix=True`."""
    cfg = mw.cfg
    if input_ids.dim() != 1:
        raise ValueError(f"expected 1-D input_ids, got {tuple(input_ids.shape)}")
    n = input_ids.shape[0]
    start = cache.length if cache is not None and suffix else 0
    if cache is not None:
        if cache.length and not suffix:
            raise ValueError(
                f"prefill on a cache already holding {cache.length} tokens; "
                "call reset() first (one-shot prefill, see iquestrt/kv.py)")
        cache.advance(n)
    positions = torch.arange(start, start + n, device=input_ids.device)
    hidden = embed(input_ids, mw)
    # One table per DISTINCT theta, hoisted out of the layer loop.
    dt = mw.embed_tokens.dtype
    rope = {}
    for spec in cfg.layers:
        if spec.rope_theta not in rope:
            rope[spec.rope_theta] = rope_tables(
                positions, cfg.rotary_dim, spec.rope_theta, input_ids.device, dt)
    for spec, lw in zip(cfg.layers, mw.layers):
        hidden = layer_forward(hidden, positions, spec, lw, mw,
                               rope=rope[spec.rope_theta], cache=cache,
                               all_reduce=all_reduce, attention_fn=attention_fn,
                               moe_fn=moe_fn, prefix_length=start)
    return rms_norm(hidden, mw.final_norm, cfg.rms_norm_eps)


def mtp_forward(
    prev_hidden: torch.Tensor,      # [S, H]  previous stage's POST-final-norm hidden
    input_ids: torch.Tensor,        # [S]     the token one step ahead of `prev_hidden`
    positions: torch.Tensor,        # [S]     `prev_hidden`'s position, one before the token's
    mtw: MtpWeights,
    mw: ModelWeights,
    *,
    spec: LayerSpec,
    cache: "KVCache | None" = None,
    all_reduce=None,
    fp32_residual: bool = False,
    attention_fn,
    moe_fn,
    prefix_length=0,
) -> torch.Tensor:
    """One MTP head over a whole sequence. Returns [S, H], post-`final_layernorm`."""
    cfg = mw.cfg
    e = embed(input_ids, mw)
    cat = torch.cat([rms_norm(e, mtw.enorm, cfg.rms_norm_eps),
                     rms_norm(prev_hidden, mtw.hnorm, cfg.rms_norm_eps)], dim=-1)
    # This rank owns one contiguous column block of the full [H, 2H], and the block's
    # width IS the loaded shard's width -- read it off the tensor rather than
    # recomputing it from tp_size, so the slice cannot disagree with the weight it
    # multiplies.
    w = mtw.eh_proj.shape[1]
    eh_in = cat.narrow(-1, mw.tp_rank * w, w).to(mtw.eh_proj.dtype)
    h = F.linear(eh_in, mtw.eh_proj)
    if all_reduce is not None:
        all_reduce(h)
    rope = rope_tables(positions, cfg.rotary_dim, spec.rope_theta, h.device, h.dtype)
    out = layer_forward(h, positions, spec, mtw.layer, mw, rope=rope, cache=cache,
                        all_reduce=all_reduce, fp32_residual=fp32_residual,
                        attention_fn=attention_fn, moe_fn=moe_fn, prefix_length=prefix_length)
    return rms_norm(out, mtw.final_norm, cfg.rms_norm_eps).to(h.dtype)
