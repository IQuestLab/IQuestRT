"""The CUDA architecture the kernel extensions are built for."""

import os
from contextlib import contextmanager


@contextmanager
def cuda_arch(arch: str = "10.0"):
    """Build for `arch` alone, whatever TORCH_CUDA_ARCH_LIST holds, and restore it after.

    The kernels use instructions that older architectures lack, so a preset list of
    several architectures, as NVIDIA's PyTorch containers set, would fail to compile."""
    previous = os.environ.get("TORCH_CUDA_ARCH_LIST")
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = previous
