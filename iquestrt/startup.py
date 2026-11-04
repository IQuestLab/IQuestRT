"""Check deployment prerequisites before allocating model weights or compiling kernels."""

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def check_checkpoint_files(path):
    from .weights import _shard_files

    files, weight_map = _shard_files(Path(path))
    if not weight_map:
        raise ValueError(f"checkpoint has an empty weight map: {path}")
    for name in files:
        if not Path(name).is_file():
            raise FileNotFoundError(f"missing checkpoint shard: {name}")
        if Path(name).stat().st_size == 0:
            raise ValueError(f"empty checkpoint shard: {name}")


def check_deployment(cfg, draft):
    """Check each rank's device, model metadata, files, and compilation tools.

    This includes prefetch header validation, not weight authentication: its
    router digest is matched to loaded weights during Engine construction.
    Call after selecting this rank's CUDA device and initializing distributed.
    """
    import torch.distributed as dist

    from .engine import Engine
    from .kernels.q8 import expert_prefetch
    from .paths import PREFETCH_FOLD_ENV, prefetch_fold

    if sys.version_info[:2] != (3, 12):
        raise ValueError("IQuestRT requires Python 3.12")
    Engine.check_supported(cfg, draft, dist.get_world_size())
    check_checkpoint_files(cfg.path)
    check_checkpoint_files(draft.path)
    fold = Path(prefetch_fold())
    if not fold.is_file() or fold.stat().st_size == 0:
        raise FileNotFoundError(
            f"missing or empty prefetch file: {fold}; download the prefetch file for this "
            f"checkpoint revision (see README) and set {PREFETCH_FOLD_ENV} to it")
    expert_prefetch.check_fold(fold, expert_prefetch.prefetched_layers(cfg.layers))
    check_build_tools()


def check_build_tools():
    """Kernels compile at first start: check the compiler, Ninja, the Python headers, and
    the CUDA 13.2 toolkit."""
    for tool in ("c++", "ninja"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"required build tool is not on PATH: {tool}")
    headers = Path(sysconfig.get_path("include"), "Python.h")
    if not headers.is_file():
        raise RuntimeError(f"Python development headers not found: {headers}; install them, "
                           "for example with apt-get install python3-dev")
    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = Path(CUDA_HOME or "") / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError("CUDA toolkit not found; set CUDA_HOME to the CUDA 13.2 installation")
    version = subprocess.check_output([str(nvcc), "--version"], text=True, timeout=10)
    if "release 13.2," not in version:
        raise RuntimeError("IQuestRT requires the CUDA 13.2 toolkit for kernel compilation")


def progress(stage):
    import torch.distributed as dist

    if dist.get_rank() == 0:
        print(f"IQuestRT: {stage}", file=sys.stderr, flush=True)
