"""Release successors before waiting; read before the wait only what nothing running writes."""

import functools

from ..attention import (
    COMBINE_WARP_PACK,
    _build_combine_q,
    _build_combine_q_compact32,
    _build_combine_q_compact32_warp_pack,
)
from ..prologue import build_heads
from .codegen import compile_transformed, replace_once

PDL_PAIR = "  cudaGridDependencySynchronize();\n  cudaTriggerProgrammaticLaunchCompletion();"
RELEASE_FIRST = "  cudaTriggerProgrammaticLaunchCompletion();\n  cudaGridDependencySynchronize();"
# The 2K cell's combine packs its 32 splits two warps to a CTA.
PACK = {**COMBINE_WARP_PACK, 32: 2}


def transform(code):
    return replace_once(code, PDL_PAIR, RELEASE_FIRST)


@functools.lru_cache(None)
def build_combine_prewait(n_split: int, *, compact32: bool):
    """`attention.build_combine` releasing the OProj before its wait, bit for bit.

    The OProj then stages its weights while the combine waits for the online kernel."""
    if not compact32:
        return compile_transformed(_build_combine_q, transform, n_split)
    pack = PACK.get(n_split, 1)
    if pack > 1:
        return compile_transformed(_build_combine_q_compact32_warp_pack, transform, n_split,
                                   pack)
    return compile_transformed(_build_combine_q_compact32, transform, n_split)


# The heads kernel's inputs other than the QKV partials: its row's ring slot, eps, the
# q or k norm weight and the row's rope values. The step writes slots and rope tables
# before its first layer; eps and the weights never change.
_ROPE = ("((((int)blockIdx.y) * 32) + (((int)threadIdx.x) & 15))",
         "(((((int)blockIdx.y) * 32) + (((int)threadIdx.x) & 15)) + 16)")
_READS = (
    "  const int pw_slot = ((int)slot[((int)blockIdx.y)]);\n"
    "  const float pw_eps = eps[0];\n"
    "  const float pw_norm = (((int)blockIdx.x) < 6) ? w_qn[((int)threadIdx.x)]"
    " : w_kn[((int)threadIdx.x)];\n"
    f"  const bfloat16_t pw_cos0 = cos_1[{_ROPE[0]}];\n"
    f"  const bfloat16_t pw_sin0 = sin_1[{_ROPE[0]}];\n"
    f"  const bfloat16_t pw_cos1 = cos_1[{_ROPE[1]}];\n"
    f"  const bfloat16_t pw_sin1 = sin_1[{_ROPE[1]}];\n")


def heads_transform(code):
    """Read the heads kernel's immutable inputs into registers before its wait."""
    code = replace_once(code, PDL_PAIR + "\n  sl[0] = ((int)slot[((int)blockIdx.y)]);\n",
                        _READS + PDL_PAIR + "\n  sl[0] = pw_slot;\n")
    code = replace_once(code, " + eps[0]));", " + pw_eps));")
    for norm in ("w_qn", "w_kn"):
        code = replace_once(code, f"({norm}[((int)threadIdx.x)] * (col[0] * inv))",
                            "(pw_norm * (col[0] * inv))")
    for i, index in enumerate(_ROPE):
        code = replace_once(code, f"* cos_1[{index}])", f"* pw_cos{i})")
        code = replace_once(code, f"* sin_1[{index}])", f"* pw_sin{i})")
    after = code[code.index(PDL_PAIR):]
    for name in ("slot[", "eps[", "w_qn[", "w_kn[", "cos_1[", "sin_1["):
        if name in after:
            raise RuntimeError(f"the heads kernel reads {name} after its wait; update this patch")
    return code


@functools.lru_cache(None)
def build_heads_prewait(capacity: int):
    """`prologue.build_heads(capacity, pdl=True)` with its immutable inputs read before
    the wait, bit for bit (`heads_transform`)."""
    return compile_transformed(build_heads, heads_transform, capacity, True)
