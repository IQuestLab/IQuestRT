"""Request budgets and the fixed sampler's supported API parameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SamplingParams:
    max_tokens: int
    seed: int
    stop: tuple[str,...] = ()


def sampling_values(req):
    """Reject unsupported overrides instead of silently changing their meaning."""
    values = dict(temperature=1.0, top_p=0.95, top_k=20)
    for key, value in values.items():
        if key in req and (type(req[key]) not in (int, float) or req[key] != value):
            raise ValueError(f"IQuestRT supports only {key}={value}")
    return values
