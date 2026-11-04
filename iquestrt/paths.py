"""Checkpoint and prefetch-file locations, read at call time."""

import os

DEFAULT_MODEL = "./models/IQuest-Q1-FP8"
DEFAULT_PREFETCH_FOLD = "./models/prefetch.safetensors"
MODEL_ENV = "IQUESTRT_MODEL"
PREFETCH_FOLD_ENV = "IQUESTRT_PREFETCH_FOLD"


def default_model() -> str:
    return os.environ.get(MODEL_ENV) or DEFAULT_MODEL


def prefetch_fold() -> str:
    return os.environ.get(PREFETCH_FOLD_ENV) or DEFAULT_PREFETCH_FOLD
