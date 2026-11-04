"""Single-request IQuest-Q1 decode on SM100, TP=8.

Fixed-address graphs verify up to seven drafts and select later depths on-device.
Padding stays finite; only committed rows survive cache settlement. Host and device
counters must agree after replay. Streaming emits only committed token blocks.

generate_mtp prefills the target and the draft, then verifies the sampled token
and its proposed continuation. Sampler.verify decides the accepted prefix and
replacement token; draft depths prepare the following step. DraftControl settles
both device counters, and the host mirrors the committed lengths after replay.
The target and recursive draft own separate caches. Padded or rejected K/V may
remain in storage but must be masked until overwritten. Draft depth zero uses
target positions; each later depth advances one position and consumes one live row.
StopRule chooses depths 3..7 using only preceding draft distributions."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Iterable
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from . import q8_profile
from .config import Config, DraftConfig
from .kernels.ar_tail import MtpEhARNorm, MultimemARTail, MultimemMLPTail
from .kernels.attention import MAIN_READS, RING_KV, WINDOW, DecodeAttention
from .kernels.draft_count import DraftControl, StopRule
from .kernels.sm100_lmhead import build_draft_lmhead_fp8
from .kernels.geometry import DraftGeometry
from .kernels.mask import DecodeMaskKernels
from .kernels.moe_gemv import MoeDecodeKernels
from .kernels.mtp import MtpPrologue
from .kernels.oproj import BM, DenseMlp, LmHeadGemv, OProjGemv
from .kernels.position import DecodePositionKernels
from .kernels.prologue import PrologueKernels, rope_inv
from .kernels.rope import RopeTables
from .kernels.router import RouterSelect
from .kernels.sampler import TEMPERATURE, TOP_K, TOP_P, Sampler
from .kernels.tail import TailKernels
from .kv import KVCache
from .model.prefill import forward, mtp_forward
from .weights import ModelWeights
from .workspace import Workspace

# The KV ring and every full-attention read are sized for this.
MAX_MODEL_LEN = 200704
# The full-attention read length is the smallest main read in `kernels/attention` that
# covers the run (`Engine.kv_len_for`): the 2048 or 4096 online cell, else the whole
# cache, which the streaming kernel reads only up to the step's live positions.

Q = 8
N_DRAFT = 5
K_MAX = 7
FLOOR = 3
# The verify widths a step can run at: the committed token and FLOOR..K_MAX drafts.
WIDTHS = tuple(range(FLOOR + 1, K_MAX + 2))
# A padding row's accept candidate. Token ids are non-negative, and the rejection rule
# (`Sampler.verify`) treats a negative candidate as the end of the drafts.
NO_DRAFT = -1

# CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED. Asked of the DRIVER, because NVLink topology says
# there is a switch and says nothing about whether this container may create multicast
# objects -- two different failures, one of them fixable from here.
_CU_MULTICAST_SUPPORTED = 132


def _multicast_supported() -> bool:
    import ctypes

    try:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuInit(0)
        dev = ctypes.c_int()
        cu.cuDeviceGet(ctypes.byref(dev), 0)
        v = ctypes.c_int()
        if cu.cuDeviceGetAttribute(ctypes.byref(v), _CU_MULTICAST_SUPPORTED, dev) != 0:
            return False
        return bool(v.value)
    except Exception:                                                   # noqa: BLE001
        return False


def decode_clock() -> float:
    """Start the decode clock: the device idle, then every rank at once."""
    torch.cuda.synchronize()
    # NCCL's barrier blocks the host until the collective completes on this rank.
    dist.barrier(device_ids=[torch.cuda.current_device()])
    return time.perf_counter()


def cut_at_stop(block: list[int], stops) -> list[int]:
    """Keep a speculative block only through its first stop token."""
    k = next((j for j, token in enumerate(block) if token in stops), None)
    return block if k is None else block[:k + 1]


class _Step(NamedTuple):
    """What every layer of one fixed-shape stack shares within a step."""

    cache: KVCache
    kv_len: int                 # host constant baked into the graph: the KV read length
    positions: torch.Tensor     # [Q] absolute positions, from the device counter
    rope: dict                  # theta -> (cos, sin)
    slots: dict | None          # KVCache.slots' per-capacity ring write slots
    masks: dict                 # (kv, window, n_sink, "u8") -> uint8 attention mask
    attn_rows: int = Q          # the attention's rows: Q, or row 0 alone at every draft depth
    ehop: torch.Tensor | None = None    # the eh_proj operand, when the geometry built it
    eh_proj: bool = False       # its launch also wrote the depth's eh_proj partial


class Engine:
    """One in-flight request, one set of per-layer caches, sized once."""

    def __init__(self, cfg: Config, *, weights: ModelWeights, draft: DraftConfig):
        """Build the chain on this rank. Call after `dist.init_process_group` with the
        current CUDA device set; `weights` must be loaded with `draft=draft`."""
        self.cfg, self.weights, self.draft = cfg, weights, draft
        self.device = torch.device("cuda")
        self.max_model_len = MAX_MODEL_LEN
        # Read the rank from the loaded shard so the MTP prologue's eh_proj window cannot
        # disagree with the weight it multiplies.
        self.tp_rank = weights.tp_rank
        self.tp_size = dist.get_world_size() if dist.is_initialized() else 1
        # Q-wide kernels, and a verify's live rows the committed token and the drafts it
        # reads, the rest padding: up to K_MAX, the draft's trained depths, which every
        # per-depth buffer and per-width build is sized for (`Q`, `K_MAX`, `WIDTHS`).
        self._q = Q
        self.k_max, self.widths = K_MAX, WIDTHS
        # "rule" uses StopRule; an integer fixes K; ("forced", K) uses depth graphs
        # with a fixed count. Capture keys include this choice.
        self.draft_count: int | tuple[str, int] | str = "rule"
        self._live = N_DRAFT + 1        # the width being captured (`_set_live`)
        self._require_supported()
        self.all_reduce = dist.all_reduce
        self._prepare_decode_weights()

        sh = cfg.shard(self.tp_size)
        dev = str(self.device)
        group = dist.group.WORLD.group_name
        q = self._q
        # The sliding layers' rings hold their window and one tile more, so a verify's
        # rows see their whole window.
        self.cache = KVCache(cfg, sh, max_model_len=MAX_MODEL_LEN, device=self.device,
                             ring_extra=RING_KV - WINDOW)
        # THE RECURSIVE DRAFT HAS ONE LAYER AND ONE CACHE. Depth d attends to the K/V
        # depths 0..d-1 wrote, so a cache per depth would make every depth blind to the
        # branch it is extending. The spec is copied with idx=0 so the layer addresses
        # layer 0 of its own cache.
        self._mtp_specs = [dataclasses.replace(draft.spec, idx=0)]
        self._mtp_caches = [KVCache(
            dataclasses.replace(cfg, layers=(self._mtp_specs[0],), num_layers=1), sh,
            max_model_len=MAX_MODEL_LEN + 2 * q + self.k_max, device=self.device,
            ring_extra=self._branch_ring_extra(),
        )]

        self.mask_fn = DecodeMaskKernels(q, device=dev)
        self.position_fn = DecodePositionKernels(q, device=dev)
        self.sampler = Sampler(weights.shard.vocab_size, self.k_max)
        self.ws = Workspace(device=dev, symm_group=group)

        self.attn_fn = DecodeAttention(ws=self.ws, heads=sh.num_q_heads,
                                       dim=cfg.head_dim, q_len=q)

        # The one-barrier AR consumers have no output barrier, so each producer
        # alternates two symmetric allocations and consecutive fused calls never share
        # one; the even layer count keeps every replay starting on the same slot.
        main_sandwich_out = tuple(
            self.ws.slot(("io", f"main_sandwich_out_{i}"), (BM, cfg.hidden_size),
                         torch.bfloat16, symm=True) for i in range(2))
        mtp_eh_out = tuple(
            self.ws.slot(("io", f"mtp_eh_out_{i}"), (16, cfg.hidden_size),
                         torch.bfloat16, symm=True) for i in range(2))

        self.prologue_fn = PrologueKernels(cfg, sh, ws=self.ws, q_len=q)
        self.mtp_eh_ar_fn = MtpEhARNorm(mtp_eh_out, group=group,
                                        norm_out=self.prologue_fn.xn,
                                        eps=cfg.rms_norm_eps, q_len=q, ws=self.ws)
        self.tail_fn = TailKernels(cfg.hidden_size, cfg.rms_norm_eps, ws=self.ws,
                                   q_len=q)

        self.moe = MoeDecodeKernels(cfg.num_experts, cfg.top_k, cfg.hidden_size,
                                    sh.moe_intermediate, ws=self.ws, q_len=q,
                                    live_rows=self.widths)
        self.dense_fn = DenseMlp(cfg.hidden_size, sh.dense_intermediate, ws=self.ws,
                                 q_len=q)

        self.mlp_ar_tail_fn = MultimemMLPTail(self.moe.outputs, group=group,
                                              tail=self.tail_fn,
                                              norm_out=self.prologue_fn.xn)
        self.mtp_tail_ar0_fn = MultimemARTail(self.moe.outputs, group=group,
                                              tail=self.tail_fn)
        self.main_sandwich_ar_fn = MultimemARTail(main_sandwich_out, group=group,
                                                  tail=self.tail_fn)

        # Two OProj holders over ONE pair of symmetric outputs (the slots are keyed by
        # name): the draft's triggers the AR tail after its stores; the main stack's
        # is the NaN-armed form `q8_profile.install` substitutes.
        self._oproj_draft = OProjGemv(sh.q_dim, cfg.hidden_size, ws=self.ws, q_len=q)
        self._oproj_main = OProjGemv(sh.q_dim, cfg.hidden_size, ws=self.ws, q_len=q)
        self.oproj_fn = self._oproj_draft

        self.ar_tail_fn = MultimemARTail(self.oproj_fn.outputs, group=group,
                                         tail=self.tail_fn)
        self.mtp_prologue_fn = MtpPrologue(cfg.hidden_size,
                                           weights.mtp[0].eh_proj.shape[1], q,
                                           weights.embed_tokens.shape[0],
                                           cfg.rms_norm_eps, self.tp_rank, self.tp_size,
                                           device=dev, ws=self.ws)
        # `rope_tables` as ONE kernel, once per distinct theta per step and once per
        # draft depth; the draft's theta is among those counted.
        n_theta = len({s.rope_theta for s in cfg.layers}
                      | {s.rope_theta for s in self._mtp_specs})
        self.rope_table_fn = RopeTables(q, cfg.rotary_dim, n_theta, device=dev,
                                        ws=self.ws)
        self.router_fn = RouterSelect(cfg.num_experts, cfg.top_k, cfg.hidden_size,
                                      device=dev, ws=self.ws, q_len=q)

        self.in_draft = False
        self._draft_kernels(False)
        # Freeze BEFORE any forward. A frozen arena cannot move, so the addresses a
        # captured graph bakes in are final.
        self.ws.freeze()
        q8_profile.install(self)

        # The draft heads' LM head: row-scaled E4M3 weights, a BN16 single-warp GEMV
        # writing the fp32 logits each depth samples from.
        self._draft_lm_fn = build_draft_lmhead_fp8(
            weights.lm_head.shape[1], weights.lm_head.shape[0])
        self._draft_logits = torch.empty((1, weights.lm_head.shape[0]),
                                         device=self.device, dtype=torch.float32)
        # Depths 1..K-1: one kernel builds the depth's position, ring slot, mask and
        # rope rows and publishes the net one-row advance.
        self._draft_geometry = DraftGeometry(q, cfg.rotary_dim, device=dev)
        # The step's last launch: whether the next depth runs, else the next verify's rows,
        # both caches' rewinds and the next step (`kernels/draft_count`). A step commits
        # one token at least, so a request's steps fit a log of MAX_MODEL_LEN.
        sampler = self.sampler
        self._control = DraftControl(q, self.k_max, FLOOR, NO_DRAFT, rule=StopRule(
            sampler.q_ids, sampler.q_probs, sampler.q_logits, depths=self.k_max,
            floor=FLOOR), max_steps=MAX_MODEL_LEN)
        # The step's inputs and outputs, allocated once per Engine, so their addresses
        # outlive every graph's private pool and every graph captured on it shares them.
        # Row 0 feeds the main stack, row 1 the rejection rule (`verify_rows`).
        i64 = torch.int64
        self._g_rows = torch.zeros((2, q), dtype=i64, device=dev)
        self._g_cand, self._g_acc = self._g_rows[0], self._g_rows[1]
        self._g_dtok = torch.zeros(q, dtype=i64, device=dev)
        self._g_hidx = torch.arange(q, dtype=i64, device=dev)
        self._g_result = torch.zeros(3 + self.k_max, dtype=i64, device=dev)
        self._g_draft = self._g_result[2:2 + self.k_max]
        # The count the request's prefill drafted, which the stop rule decides.
        self._first = FLOOR
        self.lmhead_fn = None

        self._g_steps: dict[int, torch.cuda.CUDAGraph] = {}   # verify width -> host graph
        self._g_kv_len = 0
        self._mtp_kv_len = None
        self.mtp_decode_ms = float("nan")
        # Tokens whose production is covered by `mtp_decode_ms`. The first returned
        # token is computed by prefill before the decode clock starts, so it is not in
        # TPOT's denominator.
        self.mtp_timed_tokens = 0
        self._prefix_cache = None
        self._serve_reuse_graphs = False
        self._serving_stats_enabled = False
        self._capture_key = None
        self.last_request_stats = {}

    def enable_serving_reuse(self) -> None:
        """Enable bounded prompt snapshots, graph reuse and phase timings."""
        from .prefix import PrefixCache

        self._prefix_cache = PrefixCache()
        self._serve_reuse_graphs = True
        self._serving_stats_enabled = True

    # -- construction --------------------------------------------------------
    @staticmethod
    def check_supported(cfg: Config, draft: DraftConfig, tp_size: int) -> None:
        """Validate device and metadata before loading weights."""
        sh = cfg.shard(max(tp_size, 1))
        layers = cfg.layers
        checks = {
            "torch.distributed initialised with 8 ranks": tp_size == 8,
            "compute capability 10.0": tuple(torch.cuda.get_device_capability()) == (10, 0),
            "NVLink multicast": _multicast_supported(),
            "iquest_q1 shapes (H3072, 6/1 local heads x D128, E256 top-8, I_local 192, "
            "V_local 20000)": (
                (cfg.hidden_size, sh.num_q_heads, sh.num_kv_heads, cfg.head_dim,
                 cfg.num_experts, cfg.top_k, sh.moe_intermediate, sh.vocab_size)
                == (3072, 6, 1, 128, 256, 8, 192, 20000)),
            f"sliding layers at a {WINDOW}-position window": all(
                s.sliding_window in (None, WINDOW) for s in layers),
            "sandwich dense layer 0, MoE elsewhere, even layer count": (
                bool(layers) and layers[0].use_sandwich_norm and layers[0].is_dense_mlp
                and not any(s.use_sandwich_norm or s.is_dense_mlp for s in layers[1:])
                and len(layers) % 2 == 0),
            "recursive draft trained for at least K7, run at K3-K7 on the Q8 kernels, "
            "with a sandwich MoE layer": (
                (Q, K_MAX, FLOOR) == (8, 7, 3)
                and draft.slots >= K_MAX
                and draft.spec.use_sandwich_norm
                and not draft.spec.is_dense_mlp),
            "BF16 activations": cfg.torch_dtype == "bfloat16",
            "iquest_q1's sampling defaults (T=1, top-k 20, top-p 0.95)": (
                cfg.sampling == (TEMPERATURE, TOP_K, TOP_P)),
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError("IQuestRT supports only its fixed TP=8 configuration; failed: "
                             + "; ".join(failed))

    def _require_supported(self) -> None:
        self.check_supported(self.cfg, self.draft, self.tp_size)
        if self.weights.dtype != torch.bfloat16 or len(self.weights.mtp) != 1:
            raise ValueError("IQuestRT requires BF16 activations and exactly one loaded MTP layer")

    def _prepare_decode_weights(self) -> None:
        """Materialise the decode-only weight copies on every rank, failing together.

        BF16 router weights for the fused router GEMV and a row-scaled E4M3 copy of the
        BF16 LM head for the draft heads. A rank that fails here must not leave its
        peers waiting in the first collective."""
        error = None
        try:
            self.weights.ensure_fp8_lm_head_decode_copy()
            self.weights.ensure_bf16_router_decode_copy()
        except Exception as exc:  # noqa: BLE001 - fail every TP rank together below
            error = exc
        ok = torch.tensor(int(error is None), dtype=torch.int32, device=self.device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        if not int(ok.item()):
            raise RuntimeError("decode weight preparation failed on "
                               + ("this" if error else "another") + " TP rank") from error

    def _branch_ring_extra(self) -> int:
        """Extra branch-ring slots needed by K recursive writes, tile-aligned.

        The floor, `2Q + K - 3` past a step's start, covers the schedule's reach of
        `Q + 2K - 2` at most (depth K-1's broadcast rows after a full accept). The draft's
        attention cell is built for the ring this gives (`attention.DRAFT_GEOMETRY`)."""
        window = self._mtp_specs[0].sliding_window
        if not window:
            return 0
        floor = window + 2 * self._q + K_MAX - 3
        return -(-floor // 128) * 128 - window

    def _draft_kernels(self, draft: bool, depth: int | None = None) -> None:
        """Select the main stack's or draft depth `depth`'s callables in the shared holders.

        Synchronisation-free on purpose: it runs while recording the one-step graph, and
        Python callable selection is what decides which kernel nodes are captured. The
        draft's NaN-polled all-reduces read `draft_depth` for their buffers
        (`q8_profile.install`)."""
        self.in_draft = draft
        self.draft_depth = depth if draft else None
        self.oproj_fn = self._oproj_draft if draft else self._oproj_main

    def _set_live(self, width: int) -> None:
        """Select verify width `width`'s live rows in the main stack's row-sized holders:
        the MoE launches, the NaN-polled MLP tails' buffers and the expert prefetch. Like
        `_draft_kernels`, it decides what the step being captured records."""
        if width not in self.widths:
            raise ValueError(f"verify widths are {self.widths}, got {width}")
        self._live = width
        self.mlp_tails.live = width
        self.expert_prefetch.rows = width

    # -- state ---------------------------------------------------------------
    def reset(self) -> None:
        """Drop the sequence but keep weights and cache ALLOCATIONS."""
        self.cache.reset()
        for c in self._mtp_caches:
            c.reset()

    @staticmethod
    def kv_len_for(need: int) -> int:
        """The full-attention read length for `need` live positions (see MAX_MODEL_LEN)."""
        need = min(MAX_MODEL_LEN, need)
        return min(kv for kv in MAIN_READS if kv >= need)

    # -- the fixed-shape step -----------------------------------------------
    def _layer(self, s: _Step, spec, lw, x, *, xn=None, pending=None, sandwich=None,
               next_norm=None, moe_rows=None, fp32_residual=False):
        """One decoder layer at Q rows. Returns what the next consumer takes."""
        mw, cache = self.weights, s.cache
        # Scales live in the weights when folding is on; applying them again here would
        # square them.
        a_scale = 1.0 if mw.folded else spec.attn_out_scale
        f_scale = 1.0 if mw.folded else spec.ffn_out_scale
        attn_scale = 1.0 if mw.folded else self.cfg.softmax_scale
        kv = (*s.rope[spec.rope_theta], cache.slots(spec.idx, s.positions, s.slots),
              cache.k(spec.idx), cache.v(spec.idx))
        pro = self.prologue_fn
        if pending is not None:
            prev_h, prev_out, prev_f = pending
            x, xn = self.mlp_ar_tail_fn(prev_out, prev_h, prev_f, lw.attention_norm)
        if xn is None:
            xn, q = pro(x, lw.attention_norm, lw.qkv, lw.qkv_scale, lw.q_norm, lw.k_norm,
                        *kv)
        else:
            xn, q = pro.call_prenorm(xn, lw.qkv, lw.qkv_scale, lw.q_norm, lw.k_norm, *kv)

        # Fixed-shape attention: read the whole bounded prefix and let the mask hide
        # what is not live, so nothing depends on a host-side position.
        pk, pv = cache.read_prefix(spec.idx, s.kv_len)
        n_sink = 1 if lw.sink_k is not None else 0
        if self.attn_fn.wants_inline_mask(pk):
            # The kernel derives visibility from the query positions itself.
            mask = s.positions.reshape(-1)
        else:
            key = (pk.shape[0], spec.sliding_window, n_sink, "u8")
            mask = s.masks.get(key)
            if mask is None:
                mask = s.masks[key] = self.mask_fn(
                    s.positions, cache.length_dev, kv_len=pk.shape[0],
                    capacity=cache.capacity(spec.idx),
                    max_model_len=cache.max_model_len,
                    sliding_window=spec.sliding_window, n_sink=n_sink)
        ao = self.attn_fn(q, pk, pv, sink_k=lw.sink_k, scale=attn_scale, mask=mask,
                          rows=s.attn_rows)
        ao = self.oproj_fn(ao.reshape(-1, mw.shard.q_dim), lw.o_proj, lw.o_scale)

        residual = x if spec.use_sandwich_norm else xn
        tail, moe = self.ar_tail_fn, self.moe
        if spec.is_dense_mlp:
            if fp32_residual:
                raise ValueError("the fp32 residual is wired for the draft's MoE layer only")
            h, hn = tail(ao, residual, lw.attn_out_norm, lw.feed_forward_norm,
                         a_scale, f_scale)
            # Straight into the sandwich tail's symmetric producer buffer.
            mlp_out = self.dense_fn(hn, lw.dense_gate_up, lw.gate_up_scale,
                                    lw.dense_down, lw.down_scale, sandwich.ao_buffers[0])
        else:
            h = tail.h_inv_xpad_fp8_pdl_overlap_local(
                ao, residual, lw.attn_out_norm, lw.feed_forward_norm, tail.inv,
                moe.x_pad, moe.x_pad_fp8, a_scale, f_scale, fp32_residual=fp32_residual)
            sel, wt = self.router_fn(moe.x_pad, lw.router_decode_bf16)
            mlp_out = moe(sel, wt, lw.experts_fc, lw.fc_scale, lw.experts_proj,
                          lw.proj_scale, rows=moe_rows)

        if sandwich is not None:
            #   h'  = h + norm(AR(mlp), ffn_out_norm) * f_scale     (fp32 if `h` is)
            #   out = norm(h', next_norm)        both norms ride one input barrier
            return sandwich(mlp_out, h, lw.ffn_out_norm, next_norm, f_scale, f_scale)
        return h, mlp_out, float(f_scale)

    def _main_stack(self, ids: torch.Tensor, kv_len: int) -> torch.Tensor:
        """Verify Q candidate rows through the 88 layers -> [Q, H] post-final-norm.

        Positions come from the DEVICE counter (`position_fn` advances it and writes
        them), so every replay of a captured step ropes and masks at its own positions."""
        cfg, mw, cache = self.cfg, self.weights, self.cache
        cache.advance(self._q, update_device=False)
        positions, slots = self.position_fn(cache)
        x = mw.embed_tokens[ids]
        # One table per DISTINCT theta, hoisted out of the layer loop.
        rope = {}
        for spec in cfg.layers:
            if spec.rope_theta not in rope:
                rope[spec.rope_theta] = self.rope_table_fn(
                    positions, rope_inv(cfg.rotary_dim, spec.rope_theta,
                                        str(ids.device)), spec.rope_theta)
        s = _Step(cache, kv_len, positions, rope, slots, {})
        out = None
        for i, (spec, lw) in enumerate(zip(cfg.layers, mw.layers)):
            if i == 0:
                out = self._layer(s, spec, lw, x, sandwich=self.main_sandwich_ar_fn,
                                  next_norm=mw.layers[1].attention_norm)
            elif i == 1:
                out = self._layer(s, spec, lw, out[0], xn=out[1], moe_rows=self._live)
            else:
                out = self._layer(s, spec, lw, None, pending=out, moe_rows=self._live)
        out = self.final_tail_fn(*out, mw.final_norm)
        # The side stream's expert prefetches, forked by the MLP tails, rejoin here
        # (`q8_profile.install`).
        self.expert_prefetch.join()
        return out

    def _head_step(self, j: int, prev_hidden: torch.Tensor, tokens, n_new: int,
                   prev_hidx: torch.Tensor | None = None,
                   broadcast_inputs: bool = False) -> torch.Tensor:
        """Run draft depth `j` at Q rows; returns row 0, final-normed, the one row the
        draft LM head and the next depth read."""
        self._draft_kernels(True, j)
        try:
            return self._head_step_impl(j, prev_hidden, tokens, n_new, prev_hidx,
                                        broadcast_inputs)
        finally:
            self._draft_kernels(False)

    def _draft_step(self, j: int, head=None, eh=None) -> _Step:
        """Draft depth `j`'s geometry from the draft cache's device counter: the rows'
        positions, ring slots, the mask and, at depths 1..K-1, the rope tables.
        Past the attention only row 0 is read, so its
        attention computes that row alone. `head`, the depth's MTP prologue inputs
        (token, hidden, hidden-row indices: None at the broadcast depths 1..K-1), has one
        launch build the geometry, depth 0's rope tables and slots too, and the prologue's
        operand, `ehop` (`kernels/geometry.py:draft_entry_func`). `eh`, the
        eh_proj weight, at depths 1..K-1 has that launch be the eh_proj GEMV, writing
        the depth's partial for its eh tail (`_Step.eh_proj`)."""
        cache, spec, q = self._mtp_caches[0], self._mtp_specs[0], self._q
        cap = cache.capacity(spec.idx)
        kv_read = min(self._mtp_kv_len or cap, cap)
        n_sink = int(self.weights.mtp[0].layer.sink_k is not None)
        key = (kv_read, spec.sliding_window, n_sink, "u8")
        fused = ehop = None
        if head is not None:
            pro, mtw = self.mtp_prologue_fn, self.weights.mtp[0]
            fused = (pro, head[0], self.weights.embed_tokens, head[1], mtw.enorm,
                     mtw.hnorm, head[2])
            ehop = pro.out.view(q, pro.w)
        if j > 0:
            # The geometry kernel consumes the pre-depth counter, builds row 0's mask
            # and publishes the net one-row advance.
            cache.advance(1, update_device=False)
            if eh is not None:
                eh, ehop = (eh, self.draft_tails.eh_target(j)), None
            pos, slots, mask, rope = self._draft_geometry(
                cache, spec.idx,
                rope_inv(self.cfg.rotary_dim, spec.rope_theta, str(self.device)),
                kv_len=kv_read, n_sink=n_sink, sliding_window=spec.sliding_window,
                advance=1, mask_rows=1, head=fused, eh=eh)
            return _Step(cache, self._mtp_kv_len, pos, {spec.rope_theta: rope}, slots,
                         {key: mask}, attn_rows=1, ehop=ehop, eh_proj=eh is not None)
        # The seed at b + n, read off the verify's result; the counter advances by the
        # n + 1 committed rows (`position.py`, SEED FIRST).
        cache.advance(q, update_device=False)
        pos, slots, mask, rope = self._draft_geometry(
            cache, spec.idx,
            rope_inv(self.cfg.rotary_dim, spec.rope_theta, str(self.device)),
            kv_len=kv_read, n_sink=n_sink, sliding_window=spec.sliding_window,
            advance=None, accepted=self._g_result[:1], head=fused)
        return _Step(cache, self._mtp_kv_len, pos, {spec.rope_theta: rope}, slots,
                     {key: mask}, attn_rows=1, ehop=ehop)

    def _head_step_impl(self, j, prev_hidden, tokens, n_new, prev_hidx,
                        broadcast_inputs):
        spec, q = self._mtp_specs[0], self._q
        mw, mtw = self.weights, self.weights.mtp[0]
        if j > 0 and n_new != 1:
            raise ValueError(f"draft depth {j} has one live row, got {n_new}")
        live = max(1, min(n_new, q))
        if torch.is_tensor(tokens):
            t = tokens[-live:]
        else:
            t = torch.tensor(self._pad_q(list(tokens)[-live:], q), dtype=torch.int64,
                             device=self.device)
        prev = prev_hidden[-live:]
        if not broadcast_inputs and prev.shape[0] < q:
            prev = torch.cat([prev, prev[-1:].expand(q - prev.shape[0], -1)], dim=0)
        # Head: embed + enorm/hnorm + the eh_proj window -- with the captured depth's
        # geometry in one launch -- then eh_proj with its row-parallel AR and the
        # layer's attention RMSNorm in one consumer. The eager prefill's depths keep
        # the two launches.
        captured = broadcast_inputs or j == 0
        s = self._draft_step(j, head=(t, prev, prev_hidx) if captured else None,
                             eh=mtw.eh_proj if broadcast_inputs else None)
        if s.eh_proj:
            h, xn = self.draft_tails.eh_tail(j, mtw.layer.attention_norm)
        else:
            op = s.ehop
            if op is None:
                op = self.mtp_prologue_fn(t, mw.embed_tokens, prev, mtw.enorm, mtw.hnorm,
                                          prev_hidx=prev_hidx,
                                          broadcast_inputs=broadcast_inputs)
            h, xn = self.mtp_eh_ar_fn(op.to(mtw.eh_proj.dtype), mtw.eh_proj,
                                      mtw.layer.attention_norm, head=j)
        _, g = self._layer(s, spec, mtw.layer, h, xn=xn, sandwich=self.mtp_tail_ar0_fn,
                           next_norm=mtw.final_norm, moe_rows=1,
                           fp32_residual=self.draft.fp32_residual)
        return g[:1]

    # -- host-side bookkeeping ----------------------------------------------
    def _settle(self, cache, committed: int) -> None:
        """After the step loop, move `cache`'s host mirror by the `committed` positions its
        steps committed, 1 + n_acc each.

        The device counter moved inside the graphs: each verify advanced it by Q and its
        step's end gave back `Q - 1 - n_acc` (`kernels/draft_count`). The host mirror
        moves by the net, and `advance` keeps its bound check."""
        cache.advance(committed, update_device=False)

    def _branch_read_len(self) -> int:
        """How many branch slots a draft depth reads: ALL of them, always.

        The ring holds `window + _branch_ring_extra()` slots and every one can carry a
        live position; bounding the read by the main stack's `kv_len` would let a short
        request drop real entries."""
        c = self._mtp_caches[0]
        return max(c.capacity(i) for i in range(len(c.specs)))

    def _check_branch_length(self) -> None:
        """The draft cache holds the committed prefix, so it is as long as the stack.

        Host counters only, cheap enough for every step: a captured advance that does
        not match its rewind shows up on the first step instead of as a quietly lower
        acceptance rate ten thousand tokens later."""
        got, want = self._mtp_caches[0].length, self.cache.length
        if got != want:
            raise AssertionError(
                f"the recursive draft's cache holds {got} positions and the main stack "
                f"{want}. They must agree: depth 0 covers exactly the committed rows, "
                "so any gap is an advance whose rewind did not match -- rope then "
                "rotates by the wrong angle and the mask admits the wrong slots, "
                "silently.")

    @staticmethod
    def verify_rows(cand: list[int], q: int) -> tuple[list[int], list[int]]:
        """The committed token and drafts `cand` -> (input row, accept row), `q` wide.

        The input row repeats the last candidate into the padding rows, which only have
        to carry a real token; the accept row puts `NO_DRAFT` there, so the rejection
        rule stops at the last draft."""
        if not 0 < len(cand) <= q:
            raise ValueError(f"{len(cand)} candidates do not fit {q} verify rows")
        pad = q - len(cand)
        return list(cand) + [cand[-1]] * pad, list(cand) + [NO_DRAFT] * pad

    def _load_candidates(self, cand: list[int]) -> None:
        """Write a verify's input and accept rows in one host-to-device copy: the first
        step's, from prefill. Every later step's are written by the step before it."""
        self._g_rows.copy_(torch.tensor(self.verify_rows(cand, self._q),
                                        dtype=torch.int64, device=self.device))

    def _pad_q(self, xs: list, q: int | None = None) -> list:
        """The last `q` (default Q) elements, the last one repeated up to `q`."""
        q = self._q if q is None else q
        tail = list(xs[-q:])
        return tail + [tail[-1]] * (q - len(tail))

    # -- capture and generation ---------------------------------------------
    def _counts(self) -> tuple[int | None, int]:
        """(the fixed count or None, the first step's count) of `draft_count`: under the
        stop rule, what the request's prefill drafted."""
        if isinstance(self.draft_count, int):
            return self.draft_count, self.draft_count
        if self.draft_count == "rule":
            return None, self._first
        return None, self.draft_count[1]

    def _prefill_drafts(self, depth: int) -> bool:
        """Whether the eager prefill drafts depth `depth`, as a step would: the first
        step's count, or the stop rule's decision from depths 0..depth-1 (their drafts
        in `_g_draft`)."""
        if self.draft_count != "rule":
            return depth < self._counts()[1]
        if depth < FLOOR or depth >= self.k_max:
            return depth < FLOOR
        self._bind_rule()
        return self._control.decide(depth, self._g_result)

    def _bind_rule(self) -> None:
        """Point the stop rule at the buffers of the sampler that drafts, which a control
        can swap."""
        s = self.sampler
        self._control.rule.bind(s.q_ids, s.q_probs, s.q_logits)

    def _depth(self, j: int, prev: torch.Tensor) -> torch.Tensor:
        """Draft depth `j` inside the captured step, its LM head and its draw into
        `_g_draft[j]` -> its row 0, the next depth's input. Depth 0 consumes the token row
        and hidden-row indices the verify wrote; depth j > 0 one live row, depth j-1's
        draft token."""
        mw = self.weights
        prev = self._head_step(j, prev, self._g_draft[j - 1:j] if j else self._g_dtok,
                               1 if j else self._q, prev_hidx=None if j else self._g_hidx,
                               broadcast_inputs=bool(j))
        self._draft_lm_fn(prev[-1:].to(torch.bfloat16), mw.lm_head_decode_fp8,
                          mw.lm_head_decode_scale, self._draft_logits)
        self.sampler.draft(self._draft_logits, j, self._g_draft[j:j + 1])
        return prev

    def capture_mtp(self, kv_len: int) -> None:
        """Capture the verifier and draft graphs at fixed addresses, preserving live state."""
        q = self._q
        mw = self.weights
        self._g_kv_len = kv_len
        # The draft's own bound is its ring, not the stack's read length.
        self._mtp_kv_len = self._branch_read_len()
        if self.lmhead_fn is None:
            # Read V_local from the loaded tensor.
            self.lmhead_fn = LmHeadGemv(
                mw.lm_head.shape[1], mw.lm_head.shape[0], device=self.device, q_len=q)
        fixed, first = self._counts()
        widths = (fixed + 1,) if fixed else self.widths
        ends = (self._g_result, self._g_rows, self.cache.length_dev,
                self._mtp_caches[0].length_dev)

        lens = [self.cache.length] + [c.length for c in self._mtp_caches]
        drawn = [t.clone() for t in self.sampler.state()]
        # Warm-ups write KV as well as counters. Preserve rings and the full-layer
        # boundary so graph reuse and recapture see the same request history.
        live = []
        for cache, length in zip([self.cache] + self._mtp_caches, lens):
            for i, spec in enumerate(cache.specs):
                start = 0 if spec.is_ring else max(0, length - 2 * q - self.k_max)
                end = spec.capacity if spec.is_ring else min(spec.capacity, length + 2 * q + self.k_max)
                for tensor in (cache.k(i), cache.v(i)):
                    view = tensor[start:end]
                    live.append((view, view.clone()))

        def rewind_all():
            for view, saved in live:
                view.copy_(saved)
            for c, n in zip([self.cache] + self._mtp_caches, lens):
                c.length = n
                c.length_dev.fill_(n)
            for t, saved in zip(self.sampler.state(), drawn):
                t.copy_(saved)

        def head(width, depths):
            """The main stack over `width` live rows, the verify over width - 1 drafts,
            then depths 0..depths-1 -> the last depth's row 0."""
            self._set_live(width)
            hv = self._main_stack(self._g_cand, self._g_kv_len)
            self.sampler.verify(self.lmhead_fn(hv, mw.lm_head), self._g_acc, self._g_dtok,
                                self._g_hidx, self._g_result[:2], drafts=width - 1)
            prev = hv
            for j in range(depths):
                prev = self._depth(j, prev)
            return prev

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            # Every width once, and three warm-ups at least: a collective first seen
            # during capture has no initialised communicator buffers.
            for width in list(widths) + [first + 1] * (3 - len(widths)):
                rewind_all()
                head(width, first)
                self._control.end(first, *ends)
        torch.cuda.current_stream().wait_stream(side)
        rewind_all()
        torch.cuda.synchronize()
        self._g_steps = {}
        self._control.release()
        if fixed:
            self.draft_tails.reset_turns(fixed - 1)
            graph = self._g_steps[fixed + 1] = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                head(fixed + 1, fixed)
                self._control.end(fixed, *ends, loop=True)
            self._control.add_step(fixed + 1, graph)
        else:
            # The controls record the decision they take: the stop rule's or the count.
            self._control.by_rule = self.draft_count == "rule"
            if self._control.by_rule:
                self._bind_rule()               # the graphs keep its buffers' addresses
            self._control.count.fill_(first)
            for width in widths:
                rewind_all()                    # recording moved the host mirrors
                self.draft_tails.reset_turns(width - 2)
                graph = self._g_steps[width] = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph):
                    prev = head(width, FLOOR)
                    self._control.control(FLOOR, *ends)
                self._control.add_step(width, graph)
            # Every depth's row 0 is the same workspace row, so `prev` feeds them all.
            self.draft_tails.reset_turns(FLOOR - 1)
            for j in range(FLOOR, self.k_max):
                rewind_all()
                self._control.capture_depth(j, lambda j=j: (
                    self._depth(j, prev), self._control.control(j + 1, *ends)))
        # As the warm-ups left the device: their last depth's buffers hold its partials.
        self.draft_tails.reset_turns(first - 1)
        rewind_all()
        torch.cuda.synchronize()
        self._capture_key = (kv_len, self.draft_count)

    @torch.inference_mode()
    def generate_mtp(
        self, prompt_ids: list[int], max_new_tokens: int, *,
        eos_token_id: int | Iterable[int] | None = None,
        seed: int = 0,
        on_tokens=None,
        cancelled=None,
    ) -> tuple[list[int], list[int]]:
        """Return generated tokens and accepted counts per step.

        Stops are included in the output. Callbacks emit committed blocks; all ranks must
        agree on cancellation. Callback failures are fatal to the tensor-parallel job."""
        # The transient Q-wide verify peak is a caller-visible budget and must fail
        # before any engine state is touched.
        peak = (len(prompt_ids) + max_new_tokens - 2 + self._q
                if max_new_tokens >= 2 else len(prompt_ids))
        if peak > MAX_MODEL_LEN:
            raise ValueError(
                f"prompt {len(prompt_ids)} + max_new_tokens {max_new_tokens} peaks at "
                f"{peak} cached positions, over max_model_len {MAX_MODEL_LEN}; "
                f"the Q={self._q} verify advances before rejected rows are rewound")
        stops = (set() if eos_token_id is None else
                 {eos_token_id} if isinstance(eos_token_id, int) else set(eos_token_id))
        t_phase = time.perf_counter()
        prefix_cache = self._prefix_cache
        entry = prefix_cache.match(prompt_ids) if prefix_cache is not None else None
        reused = len(entry.ids) if entry is not None else 0
        stats = self.last_request_stats = dict(input_tokens=len(prompt_ids),
                                              cache_read_tokens=reused,
                                              prefill_tokens=len(prompt_ids) - reused)

        def phase(name, start):
            if self._serving_stats_enabled:
                torch.cuda.synchronize()
            end = time.perf_counter()
            stats[name] = (end - start) * 1000
            return end

        self.reset()
        previous_hidden = (prefix_cache.restore(entry, [self.cache] + self._mtp_caches)
                           if entry is not None else None)
        t_phase = phase("restore_ms", t_phase)
        Q = self._q
        L = len(prompt_ids)
        mw = self.weights
        hw = mw.lm_head_fp32()
        # One kv_len for the whole run: a host constant baked into the graph.
        kv_len = self.kv_len_for(min(MAX_MODEL_LEN, L + max_new_tokens + Q))

        # -- prefill: uncached target/draft rows, then depths 1..r-1 eagerly, r
        # the first step's count (`_prefill_drafts`).
        h = (forward(torch.tensor(prompt_ids[reused:], dtype=torch.int64, device=self.device),
                     mw, self.cache, all_reduce=self.all_reduce)
             if reused < L else previous_hidden)
        # The first token sits at position L.
        self.sampler.start(seed, L)
        tok = torch.empty(1, dtype=torch.int64, device=self.device)
        self.sampler.sample(F.linear(h[-1:].to(hw.dtype), hw), tok)
        y = int(tok)
        t_phase = phase("target_prefill_ms", t_phase)
        if on_tokens is not None:
            on_tokens([y])
        self._mtp_kv_len = self._branch_read_len()
        c = self._mtp_caches[0]
        same_pair = (entry is not None and entry.draft_hidden is not None
                     and entry.next_token == (prompt_ids[reused] if reused < L else y))
        draft_start = reused if same_pair else max(0, reused - 1)
        prev = (torch.cat((previous_hidden, h))
                if reused and reused < L and not same_pair else h)
        toks = list(prompt_ids[draft_start + 1:]) + [y]
        if same_pair:
            c.length = reused
            c.length_dev.fill_(reused)
        c.advance(L - draft_start)
        # Depth 0 runs at the MAIN STACK's positions: row p carries h_p and y_{p+1} and
        # predicts y_{p+2} while sitting at position p; its seed is row L - 1, and depth j
        # runs at L - 1 + j.
        if draft_start == L:
            prev = entry.draft_hidden.to(self.device)
        else:
            suffix = dict(prefix_length=draft_start) if draft_start else {}
            prev = mtp_forward(prev, torch.tensor(toks, dtype=torch.int64, device=self.device),
                               torch.arange(draft_start, L, device=self.device), mw.mtp[0], mw,
                               spec=self._mtp_specs[0], cache=c, all_reduce=self.all_reduce,
                               fp32_residual=self.draft.fp32_residual, **suffix)
        t_phase = phase("draft_prefill_ms", t_phase)
        if prefix_cache is not None and reused != L:
            prefix_cache.store(prompt_ids, [self.cache] + self._mtp_caches, h,
                               draft_hidden=prev, next_token=y)
        t_phase = phase("snapshot_ms", t_phase)
        # The drafts come from the step's own draft head, so every distribution the
        # first verify reads is one the step itself would have produced. They land in the
        # result's draft slots, where a step leaves its own and the stop rule reads them.
        drafts = []
        while self._prefill_drafts(len(drafts)):
            j = len(drafts)
            if j:
                prev = self._head_step(j, prev, toks, n_new=1)
            self._draft_lm_fn(prev[-1:].to(torch.bfloat16), mw.lm_head_decode_fp8,
                              mw.lm_head_decode_scale, self._draft_logits)
            self.sampler.draft(self._draft_logits, j, self._g_draft[j:j + 1])
            drafts.append(int(self._g_draft[j]))
            toks = toks[1:] + [drafts[-1]]
        first = self._first = len(drafts)
        # Give the branch back: depths 1..first-1 wrote from DRAFT hiddens.
        c.rewind(first - 1)

        out: list[int] = [y]
        accepted: list[int] = []
        phase("initial_draft_ms", t_phase)
        stats["graph_reused"] = bool(self._serve_reuse_graphs and self._g_steps
                                     and self._capture_key == (kv_len, self.draft_count))
        stats["capture_ms"] = 0.
        if stats["graph_reused"]:
            self.draft_tails.reset_turns(first - 1)
        else:
            t_phase = time.perf_counter()
            self.capture_mtp(kv_len)
            phase("capture_ms", t_phase)
        # The first verify's rows; every later step's are written by the step before.
        cand = [y] + drafts
        self._load_candidates(cand)
        # The clock starts AFTER capture, on every rank at once (`decode_clock`).
        t_wall = decode_clock()
        last, steps = first, []
        was_cancelled = False
        if len(out) < max_new_tokens and y not in stops:
            if on_tokens is None and cancelled is None:
                # Every step, on the device, then the ONE host read.
                steps = self._control.run(len(cand), max_new_tokens - len(out), stops)
            else:
                remaining, previous = max_new_tokens - 1, [0, *cand]
                width = len(cand)
                while remaining > 0:
                    if cancelled is not None and cancelled():
                        was_cancelled = True
                        break
                    chunk = self._control.run(width, min(32, remaining), stops)
                    block = []
                    for row in chunk:
                        block.extend(previous[2:2 + row[0]])
                        block.append(row[1])
                        previous = row
                    steps.extend(chunk)
                    emitted = cut_at_stop(block, stops)[:remaining]
                    if on_tokens is not None:
                        on_tokens(emitted)
                    remaining -= len(emitted)
                    if emitted[-1] in stops:
                        break
                    width = chunk[-1][-1] + 1
        # Each step committed its accepted drafts -- the first n of those the step before
        # made, laid out as a result row [n, y, d_0...] -- and its y.
        accepted += [1 + row[0] for row in steps]
        committed = []
        for before, row in zip([[0, *cand], *steps], steps):
            committed += before[2:2 + row[0]]
            committed.append(row[1])
        self._settle(self.cache, len(committed))
        self._settle(self._mtp_caches[0], len(committed))
        self._check_branch_length()
        emitted = committed
        if steps:
            # The chain ends at the step holding the decode's last token: no stop and
            # not the budget before that step, one of them in it.
            last = steps[-1][-1]
            head = committed[:len(committed) - accepted[-1]]
            emitted = head + cut_at_stop(committed[len(head):], stops)
            ends = (was_cancelled or emitted[-1] in stops
                    or len(out) + len(emitted) >= max_new_tokens)
            if not stops.isdisjoint(head) or len(out) + len(head) >= max_new_tokens or not ends:
                raise RuntimeError(f"the step loop ran {len(steps)} steps, but its tokens "
                                   "end the decode at another")
        out.extend(emitted)
        # The last step's last depth left its draft buffers holding partials: the next
        # request's eager depths re-arm them first.
        self.draft_tails.reset_turns(last - 1)
        torch.cuda.synchronize()
        out = out[:max_new_tokens]
        # Wall clock over the loop only, INCLUDING the launch, the read and the walk over
        # the steps -- what a user waits for.
        self.mtp_decode_ms = (time.perf_counter() - t_wall) * 1000.0
        # `out[0]` came from prefill, before `t_wall`: count only tokens produced inside
        # the timed loop and returned. A prompt whose prefill token is already a stop
        # executes no timed decode work and has no defined TPOT.
        self.mtp_timed_tokens = max(len(out) - 1, 0)
        return out, accepted
