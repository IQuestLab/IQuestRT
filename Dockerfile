FROM nvidia/cuda:13.2.1-devel-ubuntu24.04@sha256:0e1f7b8e96fa9ec5e36d4709a38c62df7b5665977446081811c12b8234d874bf

COPY --from=ghcr.io/astral-sh/uv:0.13.3@sha256:bee0a2a458fd956c5df7b8b4c5fed2257b0af34866e3da7b51db0be3acb375d1 /uv /usr/local/bin/uv

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:${PATH} \
    CUDA_HOME=/usr/local/cuda

# Triton's kernel cache joins torch's under /root/.cache, which `docker run` can mount.
ENV TRITON_CACHE_DIR=/root/.cache/triton

# JIT kernels need the CUDA toolkit and Python headers; the container runtime supplies
# the host driver.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-dev build-essential ninja-build libaio-dev ca-certificates \
    && apt-get purge -y cuda-compat-13-2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/iquestrt
COPY pyproject.toml uv.lock ./
# Cache the large CUDA/PyTorch dependency layer across source and README edits.
RUN uv sync --locked --extra serve --no-install-project

COPY README.md LICENSE NOTICE ./
COPY iquestrt/ ./iquestrt/
COPY examples/ ./examples/
COPY tools/ ./tools/
RUN uv sync --locked --extra serve --no-editable \
    && uv pip check

CMD ["/bin/bash"]
