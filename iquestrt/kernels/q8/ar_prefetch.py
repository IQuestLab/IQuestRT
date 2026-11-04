"""Main-only immutable FFN prefetch and MLP early release specializations."""

import functools

from .. import ar_tail


@functools.lru_cache(None)
def source():
    """`ar_tail._CUDA` with the main-only specializations below."""
    source = ar_tail._CUDA
    old, new = (
        "  auto* residual_stage = reinterpret_cast<__nv_bfloat16*>(local_stage);\n  auto* w_ao_stage = reinterpret_cast<float*>(residual_stage + H);\n\n#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))\n",
        "  auto* residual_stage = reinterpret_cast<__nv_bfloat16*>(local_stage);\n  auto* w_ao_stage = reinterpret_cast<float*>(residual_stage + H);\n  auto* w_ffn_stage = w_ao_stage + H;\n\n#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))\n",
    )
    if source.count(old) != 1:
        raise RuntimeError("all-reduce tail source contract changed; update this patch")
    source = source.replace(old, new)
    old, new = (
        "        *reinterpret_cast<float4*>(w_ao_stage + pack * 4) =\n            *reinterpret_cast<const float4*>(w_ao + pack * 4);\n      }\n    }\n",
        "        *reinterpret_cast<float4*>(w_ao_stage + pack * 4) =\n            *reinterpret_cast<const float4*>(w_ao + pack * 4);\n        if constexpr (EmitXPad) {\n          *reinterpret_cast<float4*>(w_ffn_stage + pack * 4) =\n              *reinterpret_cast<const float4*>(w_ffn + pack * 4);\n        }\n      }\n    }\n",
    )
    if source.count(old) != 1:
        raise RuntimeError("all-reduce tail source contract changed; update this patch")
    source = source.replace(old, new)
    old, new = (
        "        for (int half = 0; half < 2; ++half) {\n          const float4 nw = *reinterpret_cast<const float4*>(\n              w_ffn + base + half * 4);\n          const float2 h0 = __bfloat1622float2(hb[half * 2]);\n          const float2 h1 = __bfloat1622float2(hb[half * 2 + 1]);\n",
        "        for (int half = 0; half < 2; ++half) {\n          const float4 nw = *reinterpret_cast<const float4*>(\n              (OverlapLocal ? w_ffn_stage : w_ffn) + base + half * 4);\n          const float2 h0 = __bfloat1622float2(hb[half * 2]);\n          const float2 h1 = __bfloat1622float2(hb[half * 2 + 1]);\n",
    )
    if source.count(old) != 1:
        raise RuntimeError("all-reduce tail source contract changed; update this patch")
    source = source.replace(old, new)
    old, new = (
        "// its exact attention-normalized value once, while the reduced MoE values are live.\ntemplate <bool OverlapLocal>\n__global__ __launch_bounds__(DIRECT_NT, 1) void mlp_tail_norm_direct1bar_impl(\n    const __nv_bfloat16* mc,\n",
        "// its exact attention-normalized value once, while the reduced MoE values are live.\ntemplate <bool OverlapLocal, bool EarlyTrigger = false>\n__global__ __launch_bounds__(DIRECT_NT, 1) void mlp_tail_norm_direct1bar_impl(\n    const __nv_bfloat16* mc,\n",
    )
    if source.count(old) != 1:
        raise RuntimeError("all-reduce tail source contract changed; update this patch")
    source = source.replace(old, new)
    old, new = (
        "  U4 residual_bits;\n\n  sync_remote_blocks<false, true>(signal_pads, rank, world);\n",
        '  U4 residual_bits;\n\n#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)\n  if constexpr (EarlyTrigger) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");\n#endif\n  sync_remote_blocks<false, true>(signal_pads, rank, world);\n',
    )
    if source.count(old) != 1:
        raise RuntimeError("all-reduce tail source contract changed; update this patch")
    source = source.replace(old, new)
    old, new = (
        "  config.gridDim = dim3(static_cast<unsigned>(q), 1, 1);\n  config.blockDim = dim3(DIRECT_NT, 1, 1);\n  config.dynamicSmemBytes = H * sizeof(__nv_bfloat16) + H * sizeof(float);\n  config.stream = stream;\n  cudaLaunchAttribute attr{};\n",
        "  config.gridDim = dim3(static_cast<unsigned>(q), 1, 1);\n  config.blockDim = dim3(DIRECT_NT, 1, 1);\n  config.dynamicSmemBytes = H * sizeof(__nv_bfloat16) + 2 * H * sizeof(float);\n  config.stream = stream;\n  cudaLaunchAttribute attr{};\n",
    )
    if source.count(old) != 1:
        raise RuntimeError("all-reduce tail source contract changed; update this patch")
    source = source.replace(old, new)
    old, new = (
        "  auto stream = at::cuda::getCurrentCUDAStream();\n  constexpr int dynamic_smem = H * (sizeof(__nv_bfloat16) + sizeof(float));\n  mlp_tail_norm_direct1bar_impl<true>\n      <<<static_cast<int>(q), DIRECT_NT, dynamic_smem, stream>>>(\n      reinterpret_cast<const __nv_bfloat16*>(mc),\n",
        "  auto stream = at::cuda::getCurrentCUDAStream();\n  constexpr int dynamic_smem = H * (sizeof(__nv_bfloat16) + sizeof(float));\n  mlp_tail_norm_direct1bar_impl<true, true>\n      <<<static_cast<int>(q), DIRECT_NT, dynamic_smem, stream>>>(\n      reinterpret_cast<const __nv_bfloat16*>(mc),\n",
    )
    if source.count(old) != 1:
        raise RuntimeError("all-reduce tail source contract changed; update this patch")
    return source.replace(old, new)
