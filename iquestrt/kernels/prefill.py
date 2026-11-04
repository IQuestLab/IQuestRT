"""Tiled prefill attention and grouped block-FP8 experts for SM100."""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["N", "PREFIX"])
def _attention(Q, K, V, Sink, Out, PK, PV, N, PREFIX, H: tl.constexpr,
               HK: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
               WINDOW: tl.constexpr, HAS_SINK: tl.constexpr,
               HAS_PREFIX: tl.constexpr, CAPACITY: tl.constexpr,
               PKS: tl.constexpr, PKH: tl.constexpr, PVS: tl.constexpr, PVH: tl.constexpr,
               QS: tl.constexpr, QH: tl.constexpr, KS: tl.constexpr,
               KH: tl.constexpr, VS: tl.constexpr, VH: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    rows = block * BM + tl.arange(0, BM)
    positions = rows + PREFIX
    dims = tl.arange(0, D)
    kv_head = head // (H // HK)
    q = tl.load(Q + rows[:, None] * QS + head * QH + dims[None, :],
                rows[:, None] < N, 0)
    if HAS_SINK:
        sink = tl.load(Sink + kv_head * D + dims).to(q.dtype)
        maximum = tl.sum(q.to(tl.float32) * sink[None, :].to(tl.float32), 1) * SCALE
        denominator = tl.full((BM,), 1, tl.float32)
    else:
        maximum = tl.full((BM,), -float("inf"), tl.float32)
        denominator = tl.zeros((BM,), tl.float32)
    acc = tl.zeros((BM, D), tl.float32)
    begin = 0
    if WINDOW:
        begin = tl.maximum(0, PREFIX + block * BM - WINDOW + 1) // BN * BN
    end = PREFIX + tl.minimum((block + 1) * BM, N)
    for offset in range(begin, end, BN):
        cols = offset + tl.arange(0, BN)
        local = cols - PREFIX
        k = tl.load(K + local[None, :] * KS + kv_head * KH + dims[:, None],
                    (local[None, :] >= 0) & (local[None, :] < N), 0)
        if HAS_PREFIX:
            slots = cols % CAPACITY if WINDOW else cols
            old = tl.load(PK + slots[None, :] * PKS + kv_head * PKH + dims[:, None],
                          cols[None, :] < PREFIX, 0)
            k = tl.where(cols[None, :] < PREFIX, old, k)
        scores = tl.dot(q, k) * SCALE
        valid = (cols[None, :] <= positions[:, None]) & (cols[None, :] < PREFIX + N)
        if WINDOW:
            valid &= cols[None, :] > positions[:, None] - WINDOW
        scores = tl.where(valid, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        # An aligned window tile can precede every visible key of a row.
        # Without a sink its initial maximum is -inf; avoid -inf - -inf.
        next_maximum = tl.where(next_maximum == -float("inf"), 0., next_maximum)
        alpha = tl.exp(maximum - next_maximum)
        probabilities = tl.exp(scores - next_maximum[:, None])
        denominator = denominator * alpha + tl.sum(probabilities, 1)
        v = tl.load(V + local[:, None] * VS + kv_head * VH + dims[None, :],
                    (local[:, None] >= 0) & (local[:, None] < N), 0)
        if HAS_PREFIX:
            old = tl.load(PV + slots[:, None] * PVS + kv_head * PVH + dims[None, :],
                          cols[:, None] < PREFIX, 0)
            v = tl.where(cols[:, None] < PREFIX, old, v)
        acc = acc * alpha[:, None]
        acc = tl.dot(probabilities, v.to(tl.float32), acc, input_precision="tf32x3")
        maximum = next_maximum
    tl.store(Out + (rows[:, None] * H + head) * D + dims[None, :],
             acc / denominator[:, None], rows[:, None] < N)


def attention(q, k, v, *, sink_k, scale, sliding_window,
              prefix_k=None, prefix_v=None, prefix_length=0):
    """Causal BF16 rows, optionally attending to a retained prefix before writes.

    New K/V stay separate from the prefix ring: a suffix larger than the ring
    must not overwrite history still needed by its earliest query rows."""
    n, heads, dim = q.shape
    if n < 1 or dim != 128 or heads % k.shape[1]:
        raise ValueError("prefill attention requires nonempty D=128 grouped-query inputs")
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("prefill attention requires BF16 Q/K/V")
    if k.shape != v.shape or k.shape[0] != n or k.shape[2] != dim:
        raise ValueError("prefill attention requires matching full-prompt K/V")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("prefill attention requires contiguous head dimensions")
    if sink_k is not None and (sink_k.shape != k.shape[1:] or not sink_k.is_contiguous()):
        raise ValueError("prefill attention requires a contiguous [KV heads, D] sink")
    if sliding_window is not None and sliding_window < 1:
        raise ValueError("sliding window must be positive")
    if prefix_length:
        if prefix_k is None or prefix_v is None or prefix_length < 0:
            raise ValueError("a positive prefix length requires both prefix K/V tensors")
        if prefix_k.shape != prefix_v.shape or prefix_k.shape[1:] != k.shape[1:]:
            raise ValueError("prefix K/V must match current KV heads and head dimension")
        required = prefix_length if sliding_window is None else min(prefix_length, sliding_window - 1)
        if prefix_k.shape[0] < required:
            raise ValueError("prefix cache does not cover the required attention history")
        if any(t.dtype != q.dtype or t.stride(-1) != 1 for t in (prefix_k, prefix_v)):
            raise ValueError("prefix K/V must have BF16 contiguous head dimensions")
    else:
        prefix_k, prefix_v = k, v
    # The cold path never reads the prefix capacity. Keep that compile-time
    # argument constant so new prompt lengths reuse the same kernel.
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    _attention[(triton.cdiv(n, 32), heads)](
        q, k, v, q if sink_k is None else sink_k, out, prefix_k, prefix_v,
        n, prefix_length, heads, k.shape[1], dim, scale, sliding_window or 0, sink_k is not None,
        bool(prefix_length), prefix_k.shape[0] if prefix_length else 1,
        prefix_k.stride(0), prefix_k.stride(1), prefix_v.stride(0), prefix_v.stride(1),
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), v.stride(0), v.stride(1),
        32, 64, num_warps=4, num_stages=2, enable_fp_fusion=False,
    )
    return out


@triton.jit
def _combine(Down, Weights, Inverse, Out, H: tl.constexpr,
             TOP_K: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    acc = tl.zeros((B,), tl.float32)
    # Routes are in ascending expert order.
    for slot in range(TOP_K):
        route = tl.load(Inverse + row * TOP_K + slot)
        value = tl.load(Down + route * H + cols, cols < H, 0).to(tl.bfloat16).to(tl.float32)
        weight = tl.load(Weights + row * TOP_K + slot)
        # SM100 ptxas fuses packed mul.rn.f32x2/add.rn.f32x2 into FFMA2 even
        # with --fmad=false. Scalar explicit rounding keeps the reference's
        # separately rounded multiply and index_add (confirmed in SASS).
        product = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[value, weight],
            dtype=tl.float32, is_pure=True, pack=1,
        )
        acc = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[acc, product],
            dtype=tl.float32, is_pure=True, pack=1,
        )
    tl.store(Out + row * H + cols, acc, cols < H)


def moe(x, lw, cfg):
    """Group routes once, retaining the reference's FP32 GEMMs and BF16 rounding.

    Tensor-core reassociation is amplified through the BF16 residual stream, so
    keep each expert's original GEMM geometry; batch dequantization, gathers and
    elementwise work across experts instead.
    One histogram read replaces per-expert unique/nonzero synchronization."""
    logits = F.linear(x.to(lw.router.dtype), lw.router)
    weights, ids = torch.topk(logits, cfg.top_k, dim=-1)
    weights = torch.softmax(weights, dim=-1, dtype=torch.float32)
    ids, order = ids.sort(dim=-1)
    weights = weights.gather(1, order)
    _, order = ids.flatten().sort(stable=True)
    counts = torch.bincount(ids.flatten(), minlength=lw.experts_fc.shape[0]).tolist()
    inverse = torch.empty_like(order)
    inverse.scatter_(0, order, torch.arange(order.numel(), device=x.device))
    gathered = x.float().index_select(0, order // cfg.top_k)
    gu = _expert_linear(gathered, lw.experts_fc, lw.fc_scale, counts, (64, 128)).bfloat16()
    del gathered
    intermediate = lw.experts_proj.shape[2]
    # Keep both BF16 roundings at the reference's activation boundary.
    act = F.silu(gu[:, :intermediate]) * gu[:, intermediate:]
    del gu
    down = _expert_linear(act.float(), lw.experts_proj, lw.proj_scale, counts, (128, 64))
    out = torch.empty_like(x)
    _combine[(x.shape[0], triton.cdiv(x.shape[1], 256))](
        down, weights, inverse, out, x.shape[1], cfg.top_k, 256,
        enable_fp_fusion=False,
    )
    return out


def _expert_linear(x, weights, scales, counts, block):
    """Original per-expert FP32 GEMMs into a contiguous grouped output."""
    from ..blockfp8 import dequant

    w = dequant(weights, scales, block, dtype=torch.float32)
    out = torch.empty((x.shape[0], weights.shape[1]), dtype=torch.float32, device=x.device)
    start = 0
    for expert, count in enumerate(counts):
        if count:
            end = start + count
            torch.mm(x[start:end], w[expert].t(), out=out[start:end])
            start = end
    return out
