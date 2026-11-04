"""Load block-FP8 checkpoints into rank-local packed layouts and scale tables."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from safetensors import safe_open

from .blockfp8 import BLOCK, shard_scale
from .config import Config, DraftConfig, ShardSpec

Sink = Callable[[torch.Tensor], None]
E4M3 = torch.float8_e4m3fn
# The rank-local scale granularity of an expert's Gate/Up rows and Down columns.
EXPERT_GRAIN = 64


@dataclass
class LayerWeights:
    """One decoder layer. `None` fields are absent by design, not missing."""

    idx: int
    attention_norm: torch.Tensor          # [H] fp32
    attn_out_norm: torch.Tensor           # [H] fp32 (attn_out_scale folded in)
    feed_forward_norm: torch.Tensor       # [H] fp32
    ffn_out_norm: torch.Tensor | None     # [H] fp32, sandwich layers only

    qkv: torch.Tensor                     # [q_local + 2*kv_local, H] E4M3, fused
    qkv_scale: torch.Tensor               # [rows/128, H/128] fp32
    o_proj: torch.Tensor                  # [H, q_local] E4M3
    o_scale: torch.Tensor                 # [H/128, q_local/128] fp32
    q_norm: torch.Tensor                  # [head_dim] fp32 (softmax_scale folded)
    k_norm: torch.Tensor                  # [head_dim] fp32
    sink_k: torch.Tensor | None           # [kv_local, head_dim]

    # MoE layers
    router: torch.Tensor | None           # [E, H] fp32 (replicated)
    experts_fc: torch.Tensor | None       # [E, 2*I_local, H] E4M3, [gate; up]
    fc_scale: torch.Tensor | None         # [E, 2*I_local/64, H/128] fp32
    experts_proj: torch.Tensor | None     # [E, H, I_local] E4M3
    proj_scale: torch.Tensor | None       # [E, H/128, I_local/64] fp32
    # Dense layers
    dense_gate_up: torch.Tensor | None    # [2*dense_I_local, H] E4M3, [gate; up]
    gate_up_scale: torch.Tensor | None    # [2*dense_I_local/128, H/128] fp32
    dense_down: torch.Tensor | None       # [H, dense_I_local] E4M3
    down_scale: torch.Tensor | None       # [H/128, dense_I_local/128] fp32

    # The lossless BF16 router copy the fused router GEMV reads, made by the Engine
    # (`ModelWeights.ensure_bf16_router_decode_copy`); `router` stays fp32 for prefill.
    router_decode_bf16: torch.Tensor | None = field(default=None, repr=False)


@dataclass
class MtpWeights:
    """One multi-token-prediction head."""

    enorm: torch.Tensor                   # [H] fp32
    hnorm: torch.Tensor                   # [H] fp32
    eh_proj: torch.Tensor                 # [H, 2H/tp]  row-parallel
    final_norm: torch.Tensor              # [H] fp32
    layer: LayerWeights


@dataclass
class ModelWeights:
    cfg: Config
    shard: ShardSpec
    tp_rank: int
    device: torch.device
    dtype: torch.dtype
    folded: bool

    embed_tokens: torch.Tensor            # [V, H] replicated
    final_norm: torch.Tensor              # [H] fp32
    lm_head: torch.Tensor                 # [V_local, H]
    layers: list[LayerWeights]

    # Empty unless `load_weights(draft=...)`, and then exactly ONE entry: the
    # recursive draft's single layer, which runs at every depth.
    mtp: list[MtpWeights] = field(default_factory=list)
    _lm_head_f32: torch.Tensor | None = field(default=None, repr=False)
    # Draft-only row-scaled E4M3 copy of the shared vocabulary projection. The
    # canonical BF16 matrix remains resident for verify and prefill. One FP32 scale
    # belongs to each vocabulary row.
    lm_head_decode_fp8: torch.Tensor | None = field(default=None, repr=False)
    lm_head_decode_scale: torch.Tensor | None = field(default=None, repr=False)

    def lm_head_fp32(self) -> torch.Tensor:
        """The lm_head in fp32, converted once and cached.

        The prefill's first token is sampled from an fp32 GEMM against it, with fp32
        logits as the verify head writes them. +246 MB at TP=8."""
        if self._lm_head_f32 is None:
            self._lm_head_f32 = self.lm_head.float()
        return self._lm_head_f32

    def ensure_bf16_router_decode_copy(self) -> "ModelWeights":
        """Materialise one lossless BF16 decode copy of every loaded router."""
        router_layers = [lw for lw in self.layers if lw.router is not None]
        router_layers.extend(
            mtp.layer for mtp in self.mtp if mtp.layer.router is not None
        )
        for lw in router_layers:
            router = lw.router
            if router.dtype != torch.float32:
                raise ValueError(
                    f"layer {lw.idx} canonical router must be fp32, got {router.dtype}"
                )
            compact = lw.router_decode_bf16
            if compact is None:
                compact = router.to(torch.bfloat16)
                # Compare bits, not tolerances: checkpoint-origin BF16 values are exactly
                # representable in FP32, so any mismatch means this is not the promised
                # storage-only transformation.
                if not torch.equal(compact.float().view(torch.int32),
                                   router.view(torch.int32)):
                    raise ValueError(
                        f"layer {lw.idx} router is not losslessly representable as BF16"
                    )
                lw.router_decode_bf16 = compact
            elif (compact.dtype != torch.bfloat16
                  or compact.shape != router.shape
                  or compact.device != router.device):
                raise ValueError(
                    f"layer {lw.idx} has an invalid BF16 decode router copy: "
                    f"shape={tuple(compact.shape)} dtype={compact.dtype} "
                    f"device={compact.device}"
                )
            elif not torch.equal(compact.float().view(torch.int32),
                                 router.view(torch.int32)):
                # ModelWeights can be shared by multiple Engine instances. Revalidate an
                # existing secondary tensor as well as a newly-created one so stale or
                # externally-mutated storage cannot silently violate the exact backend's
                # contract on the second Engine.
                raise ValueError(
                    f"layer {lw.idx} BF16 decode router copy no longer matches the "
                    "canonical fp32 router bit-for-bit"
                )
        return self

    def ensure_fp8_lm_head_decode_copy(
        self, *, row_chunk: int = 128
    ) -> "ModelWeights":
        """Materialise and revalidate a row-scaled E4M3 draft LM-head copy.

        Quantisation is intentionally storage-only and lazy. ``lm_head`` stays
        canonical BF16 because verify keeps its arithmetic.
        Chunking bounds the temporary FP32 allocation while the 20,000 x 3,072
        TP8 shard is converted."""
        if row_chunk < 1:
            raise ValueError(f"row_chunk must be positive, got {row_chunk}")
        if self.dtype != torch.bfloat16:
            raise ValueError(
                "the FP8 draft LM head needs canonical BF16 model weights; "
                f"got ModelWeights.dtype={self.dtype}"
            )
        target_device = self.device
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())

        canonical = self.lm_head
        expected_shape = (self.shard.vocab_size, self.cfg.hidden_size)
        if (
            canonical.dtype != torch.bfloat16
            or tuple(canonical.shape) != expected_shape
            or canonical.device != target_device
            or not canonical.is_contiguous()
        ):
            raise ValueError(
                "canonical LM-head must be contiguous BF16 "
                f"{expected_shape} on {target_device}; got "
                f"shape={tuple(canonical.shape)} dtype={canonical.dtype} "
                f"device={canonical.device} contiguous={canonical.is_contiguous()}"
            )

        compact = self.lm_head_decode_fp8
        scales = self.lm_head_decode_scale
        creating = compact is None and scales is None
        if (compact is None) != (scales is None):
            raise ValueError(
                "ModelWeights has only one half of its FP8 draft LM-head copy"
            )
        if creating:
            compact = torch.empty_like(canonical, dtype=torch.float8_e4m3fn)
            scales = torch.empty(
                canonical.shape[0], dtype=torch.float32, device=canonical.device
            )
        elif (
            compact.dtype != torch.float8_e4m3fn
            or compact.shape != canonical.shape
            or compact.device != canonical.device
            or not compact.is_contiguous()
            or scales.dtype != torch.float32
            or scales.shape != canonical.shape[:1]
            or scales.device != canonical.device
            or not scales.is_contiguous()
        ):
            raise ValueError(
                "ModelWeights has an invalid FP8 draft LM-head copy: "
                f"weight={tuple(compact.shape)}/{compact.dtype}/{compact.device}/"
                f"contiguous={compact.is_contiguous()}, "
                f"scale={tuple(scales.shape)}/{scales.dtype}/{scales.device}/"
                f"contiguous={scales.is_contiguous()}"
            )

        fp8_max = 448.0
        for start in range(0, canonical.shape[0], row_chunk):
            stop = min(start + row_chunk, canonical.shape[0])
            source = canonical[start:stop].float()
            reference_scale = (
                source.abs().amax(dim=-1).clamp_min(1.0e-8) / fp8_max
            )
            reference_fp8 = (
                source / reference_scale.unsqueeze(-1)
            ).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
            if creating:
                compact[start:stop].copy_(reference_fp8)
                scales[start:stop].copy_(reference_scale)
            elif (
                not torch.equal(
                    compact[start:stop].view(torch.uint8),
                    reference_fp8.view(torch.uint8),
                )
                or not torch.equal(scales[start:stop], reference_scale)
            ):
                raise ValueError(
                    "FP8 draft LM-head copy no longer matches canonical BF16 "
                    f"weights in row range [{start}, {stop})"
                )
        if creating:
            self.lm_head_decode_fp8 = compact
            self.lm_head_decode_scale = scales
        return self


# ---------------------------------------------------------------------------
# sinks
# ---------------------------------------------------------------------------
def _shard_dim(t: torch.Tensor, rank: int, world: int, dim: int) -> torch.Tensor:
    n = t.shape[dim]
    if n % world:
        raise ValueError(f"cannot shard dim {dim} of size {n} across {world} ranks")
    step = n // world
    return t.narrow(dim, rank * step, step)


def _store(dst: torch.Tensor, t: torch.Tensor) -> None:
    """`dst.copy_(t)` after checking the shape, and that E4M3 meets only E4M3.

    A BF16 tensor copied into an E4M3 destination would be re-quantized without a
    scale, silently: stored FP8 values are taken as they are or not at all."""
    if tuple(t.shape) != tuple(dst.shape):
        raise ValueError(f"shape mismatch writing into {tuple(dst.shape)}: "
                         f"got {tuple(t.shape)}")
    if (dst.dtype == E4M3) != (t.dtype == E4M3):
        raise ValueError(f"dtype mismatch writing into {dst.dtype}: got {t.dtype}")
    dst.copy_(t)


def _writer(dst: torch.Tensor, *, scale: float = 1.0,
            shard: tuple[int, int, int] | None = None) -> Sink:
    """Write an incoming checkpoint tensor into `dst`, sharding and scaling it.

    `shard` is (rank, world, dim). Scaling runs in fp32 before the cast to
    `dst.dtype` so a folded weight is a correctly-rounded product."""

    def sink(t: torch.Tensor) -> None:
        if shard is not None:
            t = _shard_dim(t, shard[0], shard[1], shard[2])
        if scale != 1.0:
            t = t.float() * scale
        _store(dst, t)

    return sink


def _split_writer(parts: list[tuple[torch.Tensor, tuple[int, int, int]]]) -> Sink:
    """Write one checkpoint tensor into several destinations (fused gate/up).

    Each part names its own (offset, length, dim) window of the SOURCE, so
    `experts.fc`'s two halves can come from different offsets of one tensor."""

    def sink(t: torch.Tensor) -> None:
        for dst, (off, length, dim) in parts:
            _store(dst, t.narrow(dim, off, length))

    return sink


def _scale_writer(parts: list[tuple[torch.Tensor, int, int, int, int]], *,
                  scale: float = 1.0) -> Sink:
    """Write rank-local scale tables cut from one checkpoint `weight_scale_inv`.

    Each part is (dst, dim, start, length, grain): the weight slice `[start, start +
    length)` along `dim`, tabulated per `grain` (`blockfp8.shard_scale`). `scale`
    folds a constant into the table in fp32."""

    def sink(t: torch.Tensor) -> None:
        for dst, dim, start, length, grain in parts:
            table = shard_scale(t, dim, start, length, grain)
            if scale != 1.0:
                table = table.float() * scale
            _store(dst, table)

    return sink


def _build(cfg: Config, shard: ShardSpec, tp_rank: int, tp_size: int,
           device: torch.device, dtype: torch.dtype,
           fold: bool,
           draft: "DraftConfig | None" = None,
           ) -> tuple[ModelWeights, dict[str, Sink], dict[str, Sink]]:
    """Preallocate every destination and return the name -> writer tables.

    TWO tables, because the recursive draft lives in its OWN directory: its tensors
    are named `mtp.*` there, not `model.layers.*`, and matching one file set's names
    against the other's sinks would silently deliver nothing. The second table
    is returned separately so the caller runs the loader over the draft's files with
    the draft's names, and `pending` still catches a miss on either side."""
    f32, hd = torch.float32, cfg.head_dim
    # The norms stay fp32 regardless: they are [H] or [head_dim], so the whole set is under
    # 3 MB per rank and the RMSNorm reduction is fp32 in both engines anyway.
    sinks: dict[str, Sink] = {}

    def empty(shape, dt=dtype):
        return torch.empty(shape, dtype=dt, device=device)

    embed = empty((cfg.vocab_size, cfg.hidden_size))
    final_norm = empty((cfg.hidden_size,), f32)
    lm_head = empty((shard.vocab_size, cfg.hidden_size))

    sinks["model.embed_tokens.weight"] = _writer(embed)
    sinks["model.norm.weight"] = _writer(final_norm)
    sinks["lm_head.weight"] = _writer(lm_head, shard=(tp_rank, tp_size, 0))

    def build_layer(spec, p: str, into: "dict[str, Sink] | None" = None, *,
                    fused_qkv_scale: bool = False) -> LayerWeights:
        """Allocate one decoder layer and register its sinks under prefix `p`."""
        dst = sinks if into is None else into
        qd, kd, hid, r = shard.q_dim, shard.kv_dim, cfg.hidden_size, tp_rank

        qkv = empty((qd + 2 * kd, hid), E4M3)
        qkv_scale = empty(((qd + 2 * kd) // BLOCK, hid // BLOCK), f32)
        o_proj = empty((hid, qd), E4M3)
        o_scale = empty((hid // BLOCK, qd // BLOCK), f32)
        attention_norm = empty((hid,), f32)
        attn_out_norm = empty((hid,), f32)
        feed_forward_norm = empty((hid,), f32)
        q_norm = empty((hd,), f32)
        k_norm = empty((hd,), f32)
        ffn_out_norm = empty((hid,), f32) if spec.use_sandwich_norm else None
        sink_k = empty((shard.num_kv_heads, hd)) if cfg.enable_sink_attention else None

        # q, k and v rows, each this rank's slice, side by side in one fused matrix.
        rows = {"q": (0, qd, 0), "k": (qd, kd, cfg.q_dim),
                "v": (qd + kd, kd, cfg.q_dim + cfg.kv_dim)}
        fused = []
        for name, (off, n, full_off) in rows.items():
            dst[f"{p}.self_attn.{name}_proj.weight"] = _writer(
                qkv.narrow(0, off, n), shard=(r, tp_size, 0))
            part = (qkv_scale.narrow(0, off // BLOCK, n // BLOCK), 0, r * n, n, BLOCK)
            if fused_qkv_scale:
                fused.append((part[0], 0, full_off + r * n, n, BLOCK))
            else:
                dst[f"{p}.self_attn.{name}_proj.weight_scale_inv"] = _scale_writer([part])
        if fused_qkv_scale:
            dst[f"{p}.self_attn.qkv_proj.weight_scale_inv"] = _scale_writer(fused)
        # o_proj is row-parallel: its CONTRACTION dim is sharded, not its output.
        dst[f"{p}.self_attn.o_proj.weight"] = _writer(o_proj, shard=(r, tp_size, 1))
        dst[f"{p}.self_attn.o_proj.weight_scale_inv"] = _scale_writer(
            [(o_scale, 1, r * qd, qd, BLOCK)])
        dst[f"{p}.attention_norm.weight"] = _writer(attention_norm)
        dst[f"{p}.attn_out_norm.weight"] = _writer(
            attn_out_norm, scale=spec.attn_out_scale if fold else 1.0)
        dst[f"{p}.feed_forward_norm.weight"] = _writer(feed_forward_norm)
        dst[f"{p}.self_attn.q_norm.weight"] = _writer(
            q_norm, scale=cfg.softmax_scale if fold else 1.0)
        dst[f"{p}.self_attn.k_norm.weight"] = _writer(k_norm)
        if ffn_out_norm is not None:
            dst[f"{p}.ffn_out_norm.weight"] = _writer(
                ffn_out_norm, scale=spec.ffn_out_scale if fold else 1.0)
        if sink_k is not None:
            dst[f"{p}.self_attn.sink_k"] = _writer(
                sink_k, shard=(r, tp_size, 0))

        # On non-sandwich layers there is no norm between the MLP and the scale,
        # so ffn_out_scale folds into the MLP's output-projection scale table.
        mlp_out_scale = (
            spec.ffn_out_scale if (fold and not spec.use_sandwich_norm) else 1.0)

        router = fc = fc_scale = proj = proj_scale = None
        gate_up = gate_up_scale = down = down_scale = None
        if spec.is_dense_mlp:
            di = shard.dense_intermediate
            gate_up = empty((2 * di, hid), E4M3)
            gate_up_scale = empty((2 * di // BLOCK, hid // BLOCK), f32)
            down = empty((hid, di), E4M3)
            down_scale = empty((hid // BLOCK, di // BLOCK), f32)
            for half, name in enumerate(("gate", "up")):
                dst[f"{p}.mlp.{name}_proj.weight"] = _writer(
                    gate_up.narrow(0, half * di, di), shard=(r, tp_size, 0))
                dst[f"{p}.mlp.{name}_proj.weight_scale_inv"] = _scale_writer(
                    [(gate_up_scale.narrow(0, half * di // BLOCK, di // BLOCK), 0, r * di,
                      di, BLOCK)])
            dst[f"{p}.mlp.down_proj.weight"] = _writer(down, shard=(r, tp_size, 1))
            dst[f"{p}.mlp.down_proj.weight_scale_inv"] = _scale_writer(
                [(down_scale, 1, r * di, di, BLOCK)], scale=mlp_out_scale)
        else:
            # Router runs replicated, in fp32 because the reference computes logits in
            # fp32 and bf16 flips near-tied experts -- routing is a top-8 of 256, a
            # discrete output. The decode router reads a lossless BF16 copy
            # (`ensure_bf16_router_decode_copy`).
            router = empty((cfg.num_experts, hid), f32)
            dst[f"{p}.mlp.router.weight"] = _writer(router)
            i_full, il, g = cfg.moe_intermediate, shard.moe_intermediate, EXPERT_GRAIN
            fc = empty((cfg.num_experts, 2 * il, hid), E4M3)
            fc_scale = empty((cfg.num_experts, 2 * il // g, hid // BLOCK), f32)
            proj = empty((cfg.num_experts, hid, il), E4M3)
            proj_scale = empty((cfg.num_experts, hid // BLOCK, il // g), f32)
            # fc rows are [gate; up]; each half shards independently, then the two
            # local halves sit side by side so a kernel still sees one fused
            # [2*I_local, H] block per expert.
            dst[f"{p}.mlp.experts.fc"] = _split_writer([
                (fc.narrow(1, 0, il), (r * il, il, 1)),
                (fc.narrow(1, il, il), (i_full + r * il, il, 1)),
            ])
            dst[f"{p}.mlp.experts.proj"] = _writer(proj, shard=(r, tp_size, 2))
            # One scale tensor per expert and projection in the checkpoint.
            for e in range(cfg.num_experts):
                pe = f"{p}.mlp.experts.{e}"
                for half, name in enumerate(("gate", "up")):
                    dst[f"{pe}.{name}_proj.weight_scale_inv"] = _scale_writer(
                        [(fc_scale[e].narrow(0, half * il // g, il // g), 0, r * il, il, g)])
                dst[f"{pe}.down_proj.weight_scale_inv"] = _scale_writer(
                    [(proj_scale[e], 1, r * il, il, g)], scale=mlp_out_scale)

        return LayerWeights(
            idx=spec.idx, attention_norm=attention_norm, attn_out_norm=attn_out_norm,
            feed_forward_norm=feed_forward_norm, ffn_out_norm=ffn_out_norm,
            qkv=qkv, qkv_scale=qkv_scale, o_proj=o_proj, o_scale=o_scale,
            q_norm=q_norm, k_norm=k_norm, sink_k=sink_k,
            router=router, experts_fc=fc, fc_scale=fc_scale, experts_proj=proj,
            proj_scale=proj_scale, dense_gate_up=gate_up, gate_up_scale=gate_up_scale,
            dense_down=down, down_scale=down_scale,
        )

    layers = [build_layer(spec, f"model.layers.{spec.idx}") for spec in cfg.layers]

    mtp: list[MtpWeights] = []
    draft_sinks: dict[str, Sink] = {}
    if draft is not None:
        # ONE head, and it is the whole model: the same weights run at every recursive
        # depth.
        #
        # The sub-tree under `mtp.mtp_model_layer` is the layout `build_layer`
        # allocates -- same q/k/v/o, four norms, sink_k, router, `experts.fc`
        # [E, 2I, H] and `experts.proj` [E, H, I] with per-expert scales -- except for
        # its one fused q/k/v scale table.
        #
        # THREE OF THE DRAFT'S TENSORS ARE DELIBERATELY UNCLAIMED. It also ships
        # `embed_tokens`, `lm_head` and `target_final_norm` so it can stand alone in
        # HF, and all three are bit-identical to the target's (`target_final_norm` IS
        # `model.norm.weight`). Registering no sink for them means they arrive, match
        # nothing, and are dropped.
        enorm = empty((cfg.hidden_size,), f32)
        hnorm = empty((cfg.hidden_size,), f32)
        mtp_final = empty((cfg.hidden_size,), f32)
        eh_proj = empty((cfg.hidden_size, 2 * cfg.hidden_size // tp_size))
        draft_sinks["mtp.enorm.weight"] = _writer(enorm)
        draft_sinks["mtp.hnorm.weight"] = _writer(hnorm)
        draft_sinks["mtp.final_layernorm.weight"] = _writer(mtp_final)
        draft_sinks["mtp.eh_proj.weight"] = _writer(
            eh_proj, shard=(tp_rank, tp_size, 1))
        mtp.append(MtpWeights(
            enorm=enorm, hnorm=hnorm, eh_proj=eh_proj,
            final_norm=mtp_final,
            layer=build_layer(draft.spec, "mtp.mtp_model_layer", into=draft_sinks,
                              fused_qkv_scale=True)))

    mw = ModelWeights(
        cfg=cfg, shard=shard, tp_rank=tp_rank, device=device, dtype=dtype,
        folded=fold, embed_tokens=embed, final_norm=final_norm, lm_head=lm_head,
        layers=layers, mtp=mtp,
    )
    return mw, sinks, draft_sinks


def _shard_files(path: Path) -> tuple[list[str], dict[str, str]]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        wmap = json.loads(index.read_text())["weight_map"]
        return sorted({str(path / v) for v in wmap.values()}), wmap
    only = path / "model.safetensors"
    if not only.exists():
        raise FileNotFoundError(f"no index and no model.safetensors in {path}")
    with safe_open(str(only), framework="pt", device="cpu") as f:
        wmap = {k: only.name for k in f.keys()}
    return [str(only)], wmap


def load_weights(
    cfg: Config,
    *,
    tp_rank: int = 0,
    tp_size: int = 1,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    fold_constants: bool = True,
    loader: str = "instanttensor",
    process_group=None,
    draft: "DraftConfig | None" = None,
) -> ModelWeights:
    device = torch.device(device)
    dtype = dtype or {"bfloat16": torch.bfloat16, "float16": torch.float16,
                      "float32": torch.float32}[cfg.torch_dtype]
    shard = cfg.shard(tp_size)
    path = Path(cfg.path)
    files, wmap = _shard_files(path)

    mw, sinks, draft_sinks = _build(
        cfg, shard, tp_rank, tp_size, device, dtype, fold_constants,
        draft=draft)
    # TWO file sets when a draft is loaded, and the draft's names are matched only
    # against the draft's files. `pending` spans both, so a tensor missing from either
    # is fatal.
    loads: list[tuple[list[str], dict[str, str], dict[str, Sink], Path]] = [
        (files, wmap, sinks, path)]
    if draft is not None:
        dpath = Path(draft.path)
        dfiles, dwmap = _shard_files(dpath)
        loads.append((dfiles, dwmap, draft_sinks, dpath))
    pending = {n for _f, _w, sk, _p in loads for n in sk}

    def _run_loader(fset: list[str], wm: dict[str, str],
                    sk: dict[str, Sink], base: Path) -> None:
        """Deliver one file set into one sink table."""
        if loader == "instanttensor":
            import instanttensor
            import torch.distributed as _dist

            # instanttensor needs a concrete device INDEX. `torch.device("cuda")` has
            # `.index is None`, which reaches the C binding as `device_idx=None` and
            # fails with an opaque "incompatible function arguments".
            dev_idx = device.index if device.index is not None else torch.cuda.current_device()

            # Coordinate reads across ranks through NCCL rather than reading the
            # whole checkpoint independently on every rank. Preserve an explicit
            # group; single-process loading keeps None. Use a local variable so this
            # nested function does not shadow the caller's process_group parameter.
            pg = process_group
            if pg is None and tp_size > 1 and _dist.is_initialized():
                pg = _dist.group.WORLD

            with instanttensor.safe_open(
                fset, framework="pt", device=dev_idx, process_group=pg,
                backend=instanttensor.Backend.AIO_BUFFERED,
            ) as f:
                for name, tensor in f.tensors():
                    sink = sk.get(name)
                    if sink is None:
                        continue
                    sink(tensor)
                    pending.discard(name)
        elif loader == "safetensors":
            handles: dict[str, object] = {}
            for name in list(sk):
                sh = wm.get(name)
                if sh is None:
                    continue
                h = handles.get(sh)
                if h is None:
                    h = safe_open(str(base / sh), framework="pt", device="cpu")
                    handles[sh] = h
                sk[name](h.get_tensor(name))  # type: ignore[attr-defined]
                pending.discard(name)
            handles.clear()
        else:
            raise ValueError(
                f"unknown loader {loader!r}; use 'instanttensor' or 'safetensors'")

    for _fset, _wm, _sk, _base in loads:
        _run_loader(_fset, _wm, _sk, _base)

    if pending:
        raise KeyError(
            f"{len(pending)} expected tensors were never delivered by the "
            f"{loader} loader, e.g. {sorted(pending)[:5]}. A silently missing "
            "weight is worse than a crash, so this is fatal."
        )
    return mw
