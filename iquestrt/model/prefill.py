"""Fast prefill with the reference's layer algebra and dedicated GPU kernels."""

from ..kernels.prefill import attention, moe
from . import reference


def forward(input_ids, weights, cache=None, *, all_reduce=None):
    return reference.forward(input_ids, weights, cache, all_reduce=all_reduce,
                             attention_fn=attention, moe_fn=moe, suffix=True)


def mtp_forward(*args, **kwargs):
    return reference.mtp_forward(*args, **kwargs, attention_fn=attention, moe_fn=moe)
