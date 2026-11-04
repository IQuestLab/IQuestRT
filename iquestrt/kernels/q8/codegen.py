"""Scoped compiler postprocessing for the SM100 generated-code contract."""

import contextlib
import os
import threading

import tilelang
import tvm_ffi
from tilelang.engine import register_cuda_postproc

NVCC = "tilelang_callback_cuda_compile"
_lock = threading.RLock()


@contextlib.contextmanager
def postprocessed(transform):
    """Apply `transform` to the CUDA source of the one kernel compiled inside the block.

    Nests: an inner block's transform runs on the outer one's output."""
    with _lock:
        original = tvm_ffi.get_global_func("tilelang_callback_cuda_postproc", allow_missing=True)
        nvcc = tvm_ffi.get_global_func(NVCC, allow_missing=True)
        old_cache = os.environ.get("TILELANG_DISABLE_CACHE")
        state = []

        def restore_cache():
            if old_cache is None:
                os.environ.pop("TILELANG_DISABLE_CACHE", None)
            else:
                os.environ["TILELANG_DISABLE_CACHE"] = old_cache

        def callback(code, target):
            if original is not None:
                code = original(code, target)
            code = transform(code)
            state.append(code)
            return code

        def compile_cached(*args):
            restore_cache()                 # the binary cache is keyed by this source
            try:
                return nvcc(*args)
            finally:
                os.environ["TILELANG_DISABLE_CACHE"] = "1"

        try:
            os.environ["TILELANG_DISABLE_CACHE"] = "1"
            register_cuda_postproc(callback)
            if nvcc is not None:
                tvm_ffi.register_global_func(NVCC, compile_cached, override=True)
            yield
        finally:
            register_cuda_postproc(original if original is not None else lambda code, target: code)
            if nvcc is not None:
                tvm_ffi.register_global_func(NVCC, nvcc, override=True)
            restore_cache()
        if len(state) != 1:
            raise RuntimeError("Expected exactly one transformed kernel")


def compile_transformed(builder, transform, *args, **kwargs):
    with postprocessed(transform):
        return builder.__wrapped__(*args, **kwargs)


# Convert finite E4M3 pairs exactly using integer rearrangement and BF16 scaling.
# The checkpoint quantizer emits no NaN codes; 0x7F/0xFF are not supported here.
E4M3_BF16 = r"""
__device__ __forceinline__ __nv_bfloat162 iquestrt_e4m3x2_to_bf16x2(const __nv_fp8x2_storage_t x) {
  const unsigned t = __byte_perm((unsigned)x, 0u, 0x1404u);
  const unsigned r = (t & 0x80008000u) | ((t >> 4) & 0x07F007F0u);
  const unsigned k = 0x7B807B80u;  // 2^120 in both lanes
  return __hmul2(*reinterpret_cast<const __nv_bfloat162 *>(&r),
                 *reinterpret_cast<const __nv_bfloat162 *>(&k));
}
"""


def e4m3_to_bf16(code):
    call = "__tl_cvt_e4m3x2_to_bfloat162("
    if call not in code:
        raise RuntimeError("no TileLang E4M3 -> BF16 conversion to replace; update this patch")
    i = code.index('extern "C" __global__')
    return code[:i] + E4M3_BF16 + code[i:].replace(call, "iquestrt_e4m3x2_to_bf16x2(")


def compile_e4m3(prim_func, **kwargs):
    """`tilelang.compile` with every E4M3 -> BF16 conversion on the integer pipe."""
    with postprocessed(e4m3_to_bf16):
        return tilelang.compile(prim_func, **kwargs)


def replace_once(code, old, new):
    if code.count(old) != 1:
        raise RuntimeError("generated-code contract changed; update this patch")
    return code.replace(old, new)
