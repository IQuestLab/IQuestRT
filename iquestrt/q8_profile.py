"""Install the fixed SM100 Q8 kernel chain and its PDL scheduling.

Fused router logits are a numerical-drift path. Expert prefetch changes cache
residency only; the main stack joins its side stream after the final norm."""

from __future__ import annotations

from .kernels.attention import RING_KV
from .kernels.q8 import expert_prefetch, nanpoll, oproj_middle, qkv_middle
from .paths import prefetch_fold
from .kernels.q8.attention_early import build_attention_early, build_ring_early
from .kernels.q8.prewait import build_combine_prewait, build_heads_prewait
from .kernels.prologue import build_heads


def install(e) -> None:
    """Substitute the main-stack callables of a fully built Engine `e`."""
    pro = e.prologue_fn
    qkv_fn = qkv_middle.build_qkv_middle_block()
    layers = len(e.cfg.layers)
    full = [i for i, spec in enumerate(e.cfg.layers) if spec.sliding_window is None]
    moe = [(i, lw) for i, (spec, lw) in enumerate(zip(e.cfg.layers, e.weights.layers))
           if not spec.is_dense_mlp]
    if [i for i, _ in moe] != list(range(1, layers)):
        raise ValueError("the NaN-polled tails expect a dense main layer 0 and MoE layers 1..n")
    mw = e.weights
    fold = {i: oproj_middle.fold_weights(lw, mw.tp_rank, mw.shard.tp_size) for i, lw in moe}
    attn = nanpoll.NanPolledAttentionTail(e.ws, e.ar_tail_fn.group, e.oproj_fn.q_dim,
                                          tail=e.tail_fn, layers=layers, full_layers=full,
                                          fold=fold, rank=mw.tp_rank)

    def main_only(original, candidate):
        def call(*a, **kw):
            return original(*a, **kw) if e.in_draft else candidate(*a, **kw)

        return call

    # The residual base of the main layer the QKV last ran, which that layer's OProj
    # folds into its router logits and its attention tail adds (`nanpoll.py`).
    residual = [None]

    def qkv(xn, w, w_scale, wqn, wkn, cos, sin, slot, k, v):
        xn = xn.reshape(8, pro.h)
        residual[0] = xn
        qkv_fn(xn, w, w_scale, pro.qkv)
        return xn, pro.heads(wqn, wkn, cos, sin, slot, k, v, pdl=True)

    # Only `qkv` asks for the PDL heads.
    def heads_builder(capacity, pdl):
        return build_heads_prewait(capacity) if pdl else build_heads(capacity, pdl)

    # The online cells are main-only by construction: the draft heads read their 640-slot
    # ring through the exact kernel. They are the 2K and 4K cells and the sliding layers'
    # whole ring.
    early = {2048: build_attention_early(), 4096: build_ring_early(4096),
             RING_KV: build_ring_early(RING_KV)}

    def attention_builder(kv, **kw):
        return early[kv]

    def combine_builder(n_split, **kw):
        return build_combine_prewait(n_split, **kw)

    pro.call_prenorm = main_only(pro.call_prenorm, qkv)
    pro._heads_builder = heads_builder

    # Every main layer writes its OProj partial into the NaN-armed buffer of the layer
    # whose weights the call reads, and its tail polls the switch's sum of it; a MoE
    # layer's OProj also folds the layer's router logits, and its tail routes the layer.
    # The main OProj holder runs only on the main stack, and only the draft depths call
    # the Router.
    by_oproj = {lw.o_proj.data_ptr(): i for i, lw in enumerate(e.weights.layers)}
    by_norm = {lw.attn_out_norm.data_ptr(): i for i, lw in moe}
    by_router = {lw.router_decode_bf16.data_ptr(): i for i, lw in moe}

    def oproj(x, w, scale, out):
        layer = by_oproj[w.data_ptr()]
        return attn.oproj(layer, x, w, scale, residual=residual[0] if layer else None)

    def routing(x_pad, weight):
        return attn.routing(by_router[weight.data_ptr()])

    def layer0_attention(ao, residual, w_ao, w_ffn, a_scale, f_scale):
        return attn.layer0(residual, w_ao, w_ffn, a_scale, f_scale)

    # Each draft depth's three all-reduces -- eh_proj, attention, sandwich -- write into
    # NaN-armed buffers of their own and poll the switch's sum of them.
    draft = nanpoll.NanPolledDraftTails(e.ws, e.ar_tail_fn.group, depths=e.k_max,
                                        q_dim=e.oproj_fn.q_dim, tail=e.tail_fn, moe=e.moe,
                                        eh=e.mtp_eh_ar_fn)
    e.draft_tails = draft         # the Engine sets their turns around capture and decode

    def h_inv(ao, residual, w_ao, w_ffn, inv_out, x_pad, x_pad_fp8, a_scale, f_scale, *,
              fp32_residual=False):
        if e.in_draft:
            if not fp32_residual:
                raise ValueError("the draft's residual stream is fp32")
            return draft.h_inv(e.draft_depth, ao, residual, w_ao, w_ffn, inv_out, x_pad,
                               x_pad_fp8, a_scale, f_scale)
        if fp32_residual:
            raise ValueError("the main stack's residual stream is BF16")
        return attn.h_inv(by_norm[w_ao.data_ptr()], residual, w_ao, w_ffn, inv_out, x_pad,
                          x_pad_fp8, a_scale, f_scale)

    # Every main layer writes its MLP output into a NaN-armed buffer: layer 0's dense
    # MLP, whose sandwich tail polls the switch's sum of it, and the MoE of layers 1-87,
    # whose sum the next layer's MLP tail polls, or the final tail after the last layer.
    # Only the main stack calls them.
    last = layers - 1
    by_fc = {lw.experts_fc.data_ptr(): i for i, lw in moe}
    by_next_norm = {e.weights.layers[i + 1].attention_norm.data_ptr(): i
                    for i, _ in moe if i < last}
    nan = nanpoll.NanPolledMLPTail(e.ws, e.ar_tail_fn.group, moe=e.moe, dense=e.dense_fn,
                                   widths=e.widths, tail=e.tail_fn,
                                   norm_out=e.prologue_fn.xn, layers=layers)
    e.mlp_tails = nan             # `Engine._set_live` picks a width's buffers

    def dense(hn, w_gate_up, gate_up_scale, w_down, down_scale, out):
        return nan.dense(hn, w_gate_up, gate_up_scale, w_down, down_scale)

    def final_tail(h, mlp_out, f_scale, w_norm):
        return nan(last, mlp_out, h, f_scale, w_norm)[1]

    class MainMoe:
        """The MoE holder, running main layers 1-87 and the draft depths into the
        NaN-armed buffers."""

        def __init__(self, holder):
            self.holder = holder

        def __getattr__(self, name):
            return getattr(self.holder, name)

        def __call__(self, sel, wt, fc, fc_scale, proj, proj_scale, *, rows=None):
            if e.in_draft:
                return draft.moe(e.draft_depth, sel, wt, fc, fc_scale, proj, proj_scale,
                                 rows=rows)
            layer = by_fc.get(fc.data_ptr())
            if layer is None:
                return self.holder(sel, wt, fc, fc_scale, proj, proj_scale, rows=rows)
            if rows != nan.live:
                raise ValueError(f"main layer {layer}'s MoE runs over the {nan.live} live "
                                 f"rows, got rows={rows}")
            return nan.moe(layer, sel, wt, fc, fc_scale, proj, proj_scale)

    # After the MLP tail that opens a sliding-window main layer, the side stream predicts
    # that layer's Gate experts and prefetches them into L2 while its attention runs, in a
    # step of at most five live rows; `Engine._main_stack` joins it after the final norm
    # (`expert_prefetch.py`). Its predictor weights, refit for this checkpoint, come from
    # the prefetch file the README downloads.
    prefetched = {i: e.weights.layers[i] for i in expert_prefetch.prefetched_layers(e.cfg.layers)}
    e.expert_prefetch = expert_prefetch.ExpertPrefetch(
        prefetched, fold=expert_prefetch.load_fold(prefetch_fold(), prefetched, e.device),
        widths=e.widths, device=e.device)
    by_attention_norm = {e.weights.layers[i].attention_norm.data_ptr(): i for i in prefetched}

    class PrefetchingMLPTail:
        """The NaN-polled MLP tail, forking the prefetch for the layer it opens."""

        def __init__(self, tail):
            self.tail = tail

        def __getattr__(self, name):
            return getattr(self.tail, name)

        def __call__(self, mlp_out, residual, f_scale, w_norm):
            out = self.tail(by_next_norm[w_norm.data_ptr()], mlp_out, residual, f_scale,
                            w_norm)
            layer = by_attention_norm.get(w_norm.data_ptr())
            if layer is not None:
                e.expert_prefetch.fork(layer, out[1])
            return out

    class Holder:
        """A holder whose call runs `call`: a main layer's or the current draft depth's
        NaN-polled form."""

        def __init__(self, holder, call):
            self.holder, self._call = holder, call

        def __getattr__(self, name):
            return getattr(self.holder, name)

        def __call__(self, *a, **kw):
            return self._call(*a, **kw)

    def draft_eh(eh_in, w_eh, w_norm, *, head):
        return draft.eh(head, eh_in, w_eh, w_norm)

    def draft_oproj(ao, w_o, w_scale):
        return draft.oproj(e.draft_depth, ao, w_o, w_scale)

    def draft_sandwich(mlp_out, residual, w_ao, w_ffn, a_scale, f_scale):
        return draft.sandwich(e.draft_depth, mlp_out, residual, w_ao, w_ffn, a_scale,
                              f_scale)

    e.oproj_fn.fn = oproj
    e.ar_tail_fn.h_inv_xpad_fp8_pdl_overlap_local = h_inv
    e.router_fn = main_only(e.router_fn, routing)
    e.ar_tail_fn = Holder(e.ar_tail_fn, layer0_attention)
    e.dense_fn = Holder(e.dense_fn, dense)
    e.main_sandwich_ar_fn = Holder(e.main_sandwich_ar_fn, nan.sandwich)
    e.moe = MainMoe(e.moe)
    e.mlp_ar_tail_fn = PrefetchingMLPTail(nan)
    e.final_tail_fn = final_tail
    e.mtp_eh_ar_fn = Holder(e.mtp_eh_ar_fn, draft_eh)
    e._oproj_draft = Holder(e._oproj_draft, draft_oproj)
    e.mtp_tail_ar0_fn = Holder(e.mtp_tail_ar0_fn, draft_sandwich)
    e.attn_fn._online_builder = attention_builder
    e.attn_fn._combine_builder = combine_builder
