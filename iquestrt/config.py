"""Resolve model shapes, per-layer attention, quantization, and draft compatibility."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class LayerSpec:
    """Everything the executor needs about one decoder layer, precomputed."""

    idx: int
    sliding_window: int | None  # None => full attention
    rope_theta: float
    is_dense_mlp: bool          # True => dense MLP at dense_intermediate_size
    use_sandwich_norm: bool     # True => extra ffn_out_norm, raw-hidden residual
    attn_out_scale: float
    ffn_out_scale: float


@dataclass(frozen=True)
class Config:
    path: str
    hidden_size: int
    num_layers: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    rotary_dim: int
    rms_norm_eps: float
    vocab_size: int
    num_experts: int
    top_k: int
    moe_intermediate: int
    dense_intermediate: int
    enable_sink_attention: bool
    layers: tuple[LayerSpec,...]
    torch_dtype: str
    eos_token_ids: tuple[int,...]
    # generation_config.json's (temperature, top_k, top_p) when it samples, else None.
    sampling: tuple[float, int, float] | None = None
    raw: dict = field(repr=False, default_factory=dict)

    # -- derived ---------------------------------------------------------
    @property
    def q_dim(self) -> int:
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def softmax_scale(self) -> float:
        return self.head_dim**-0.5

    def shard(self, tp_size: int) -> "ShardSpec":
        return ShardSpec.build(self, tp_size)


@dataclass(frozen=True)
class ShardSpec:
    """Per-rank shapes at tensor-parallel size `tp_size`."""

    tp_size: int
    num_q_heads: int
    num_kv_heads: int
    q_dim: int
    kv_dim: int
    moe_intermediate: int
    dense_intermediate: int
    vocab_size: int

    @classmethod
    def build(cls, cfg: Config, tp_size: int) -> "ShardSpec":
        for name, total in (
            ("num_attention_heads", cfg.num_q_heads),
            ("num_key_value_heads", cfg.num_kv_heads),
            ("moe_intermediate", cfg.moe_intermediate),
            ("dense_intermediate", cfg.dense_intermediate),
            ("vocab_size", cfg.vocab_size),
        ):
            if total % tp_size:
                raise ValueError(
                    f"{name}={total} is not divisible by tp_size={tp_size}; "
                    "iquestrt does not replicate or pad shards."
                )
        return cls(
            tp_size=tp_size,
            num_q_heads=cfg.num_q_heads // tp_size,
            num_kv_heads=cfg.num_kv_heads // tp_size,
            q_dim=cfg.q_dim // tp_size,
            kv_dim=cfg.kv_dim // tp_size,
            moe_intermediate=cfg.moe_intermediate // tp_size,
            dense_intermediate=cfg.dense_intermediate // tp_size,
            vocab_size=cfg.vocab_size // tp_size,
        )


def _resolve_layer_types(raw: dict, num_layers: int) -> list[str]:
    """Concatenate first + repeated hybrid block + last, then validate the length."""
    if not raw.get("use_hybrid_layers", False):
        return ["full_attention"] * num_layers
    first = list(raw.get("first_layers_types") or [])
    block = list(raw.get("hybrid_layers_types_block") or [])
    nblock = int(raw.get("num_hybrid_layers_block") or 0)
    last = list(raw.get("last_layers_types") or [])
    types = first + block * nblock + last
    if len(types) != num_layers:
        raise ValueError(
            f"layer-type table has {len(types)} entries "
            f"({len(first)} first + {len(block)}x{nblock} hybrid + {len(last)} last) "
            f"but num_hidden_layers={num_layers}. The three sources must cover "
            "every layer exactly once."
        )
    unknown = set(types) - {"full_attention", "sliding_attention"}
    if unknown:
        raise ValueError(f"unknown layer types {sorted(unknown)}")
    return types


# The one quantization the loader and the decode kernels implement.
QUANTIZATION = {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}


def _check_quantization(p: Path, raw: dict) -> None:
    """Refuse any quantization other than the checkpoint's block FP8."""
    q = raw.get("quantization_config") or {}
    got = {k: q.get(k) for k in QUANTIZATION}
    if got != QUANTIZATION:
        raise ValueError(
            f"{p} has quantization {got}; iquestrt reads only block-FP8 checkpoints "
            f"({QUANTIZATION}): its loader keeps the E4M3 values and per-block scales as "
            "stored and every decode kernel applies them, so no other format can load.")


def load_config(model_path: str | Path) -> Config:
    p = Path(model_path)
    raw = json.loads((p / "config.json").read_text())

    arch = raw.get("architectures") or []
    if raw.get("model_type") != "iquest_q1":
        raise ValueError(
            f"iquestrt targets iquest_q1; got model_type="
            f"{raw.get('model_type')!r} architectures={arch}"
        )
    if raw.get("moe_router_dtype", "fp32") != "fp32":
        raise ValueError(
            f"moe_router_dtype={raw['moe_router_dtype']!r}: iquestrt computes router "
            "logits in fp32 only")
    _check_quantization(p, raw)
    if raw.get("use_over_encoding", False):
        raise ValueError(
            f"{p} has use_over_encoding=True, which IQuestRT does not support. "
            "Loading it would silently omit the n-gram embedding mix. "
            "Use a checkpoint without over-encoding."
        )

    num_layers = int(raw["num_hidden_layers"])
    types = _resolve_layer_types(raw, num_layers)
    window = int(raw.get("sliding_window") or 0) or None
    rope_theta = float(raw.get("rope_theta", 10000.0))
    swa_rope_theta = float(raw.get("swa_rope_theta", rope_theta))
    mlp_only = set(raw.get("mlp_only_layers") or [])
    no_rope = set(raw.get("no_rope_layers") or [])
    if no_rope:
        raise ValueError(
            f"no_rope_layers={sorted(no_rope)} is set; iquestrt has no no-rope path yet."
        )
    if int(raw.get("shared_kv_num_layers") or 0):
        raise ValueError("shared_kv_num_layers > 0 (cross-layer KV) is not supported yet.")
    layers: list[LayerSpec] = []
    for i, t in enumerate(types):
        sliding = window if t == "sliding_attention" else None
        if t == "sliding_attention" and window is None:
            raise ValueError(f"layer {i} is sliding_attention but sliding_window is unset")
        first = i == 0
        layers.append(
            LayerSpec(
                idx=i,
                sliding_window=sliding,
                rope_theta=swa_rope_theta if sliding is not None else rope_theta,
                is_dense_mlp=i in mlp_only,
                use_sandwich_norm=first,
                attn_out_scale=float(
                    raw.get("first_layer_attn_out_scale", 1.0) if first
                    else raw.get("attn_out_scale", 1.0)
                ),
                ffn_out_scale=float(
                    raw.get("first_layer_ffn_out_scale", 1.0) if first
                    else raw.get("ffn_out_scale", 1.0)
                ),
            )
        )

    head_dim = int(raw.get("head_dim") or raw["hidden_size"] // raw["num_attention_heads"])
    rotary_dim = int(raw.get("rotary_dim") or head_dim)
    if not 0 < rotary_dim <= head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim={rotary_dim} must be even and <= head_dim={head_dim}")

    eos = raw.get("eos_token_id")
    eos_ids = tuple(eos) if isinstance(eos, (list, tuple)) else (int(eos),) if eos is not None else ()
    sampling = None
    if (p / "generation_config.json").exists():
        gen = json.loads((p / "generation_config.json").read_text())
        if gen.get("do_sample"):
            # transformers' defaults for what the file leaves out.
            sampling = (float(gen.get("temperature", 1.0)), int(gen.get("top_k", 50)),
                        float(gen.get("top_p", 1.0)))

    return Config(
        path=str(p),
        hidden_size=int(raw["hidden_size"]),
        num_layers=num_layers,
        num_q_heads=int(raw["num_attention_heads"]),
        num_kv_heads=int(raw["num_key_value_heads"]),
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
        vocab_size=int(raw["vocab_size"]),
        num_experts=int(raw["num_experts"]),
        top_k=int(raw["num_experts_per_tok"]),
        moe_intermediate=int(raw["intermediate_size"]),
        dense_intermediate=int(raw.get("dense_intermediate_size") or raw["intermediate_size"]),
        enable_sink_attention=bool(raw.get("enable_sink_attention", False)),
        layers=tuple(layers),
        torch_dtype=str(raw.get("dtype") or raw.get("torch_dtype") or "bfloat16"),
        eos_token_ids=eos_ids,
        sampling=sampling,
        raw=raw,
    )


# The recursive draft ships inside the target checkpoint, at `<model>/mtp`.
DRAFT_DIR = "mtp"


@dataclass(frozen=True)
class DraftConfig:
    """The recursive draft model: ONE layer, trained for `slots` recursive depths."""

    path: str
    spec: LayerSpec
    slots: int                    # depths it was trained for
    # The layer keeps its residual stream in fp32: `h = x + attn` and `h + ffn` are not
    # rounded to BF16 before `final_layernorm` (`fp32_residual_connection`).
    fp32_residual: bool


def load_draft_config(path: str | Path, target: Config) -> DraftConfig:
    """Parse the `iquest_q1_mtp` draft directory against its target."""
    p = Path(path)
    raw = json.loads((p / "config.json").read_text())
    if raw.get("model_type") != "iquest_q1_mtp":
        raise ValueError(
            f"{p} is model_type={raw.get('model_type')!r}, not 'iquest_q1_mtp'. "
            "This loader is for the recursive draft, which is the only speculation "
            "method iquestrt has.")
    _check_quantization(p, raw)
    for key, want in (("hidden_size", target.hidden_size),
                      ("vocab_size", target.vocab_size),
                      ("num_target_layers", target.num_layers)):
        if int(raw[key]) != want:
            raise ValueError(f"draft {key}={raw[key]} but the target has {want}")
    # SLIDING, with its own window: 512 against the target's 4096. The layer is a
    # sandwich MoE layer and takes the `first_layer_*` residual scales, as the
    # reference implementation's `IQuestQ1RecursiveInnerLayer` does.
    tgt = target.raw
    return DraftConfig(
        path=str(p),
        spec=LayerSpec(
            idx=target.num_layers,
            sliding_window=int(raw["sliding_window"]),
            # Sliding layers take `swa_rope_theta`; both sides say 10000, so the main
            # stack's existing table serves this layer too and no second one is built.
            rope_theta=float(raw.get("swa_rope_theta", tgt.get("swa_rope_theta", 10000.0))),
            is_dense_mlp=False,
            use_sandwich_norm=True,
            attn_out_scale=float(tgt.get("first_layer_attn_out_scale", 1.0)),
            ffn_out_scale=float(tgt.get("first_layer_ffn_out_scale", 1.0)),
        ),
        slots=int(raw["num_draft_slots"]),
        fp32_residual=bool(raw.get("fp32_residual_connection", False)),
    )


def resolve_draft(model: str | Path, cfg: Config) -> DraftConfig:
    """The recursive draft of `model`: the checkpoint's own `<model>/mtp`.

    The draft ships inside the target it was trained against, so there is nothing to
    pair and nothing to configure."""
    return load_draft_config(Path(model) / DRAFT_DIR, cfg)
