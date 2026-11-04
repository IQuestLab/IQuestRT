# IQuestRT

Low-latency [IQuest-Q1-FP8](https://huggingface.co/xhxlb/IQuest-Q1-FP8) inference
with fused kernels, CUDA Graphs, NVLink collectives, and speculative decoding.
See the [technical report](https://iquestlab.github.io/IQuestRT/) for the runtime design.

## Requirements

- Eight NVIDIA GPUs with compute capability 10.0 and NVLink multicast; TP=8 and one active request.
- Linux x86_64, Python 3.12, CUDA toolkit 13.2, and a compatible NVIDIA driver.
- A C++ compiler, Ninja, and Python and libaio development headers. On Ubuntu: `apt-get install build-essential ninja-build python3-dev libaio-dev`.
- About 330 GB for the checkpoint, plus space for the environment and compilation caches.

Other models, GPU architectures, and parallel configurations are unsupported.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
git clone https://github.com/IQuestLab/IQuestRT.git
cd IQuestRT
uv sync --locked --extra serve
```

uv creates the environment automatically. Dependencies are pinned in `uv.lock`;
use `uv run` for the commands below without activating the environment.

## Prepare the model

Download the full checkpoint, including `mtp/`, and its expert prefetch file:

```bash
uv run --locked hf download xhxlb/IQuest-Q1-FP8 --revision d13b9457e78909349253d96e22bc02dca46aaeeb \
  --local-dir ./models/IQuest-Q1-FP8
curl -fL -o ./models/prefetch.safetensors \
  https://github.com/IQuestLab/IQuestRT/releases/download/v0.1.0/prefetch-d13b945.safetensors
echo "2b09cd07a95c5570f64255f8fc5f874b532989f02d0d80d409de2c34030182a0  ./models/prefetch.safetensors" | sha256sum -c
export IQUESTRT_MODEL="$PWD/models/IQuest-Q1-FP8"
export IQUESTRT_PREFETCH_FOLD="$PWD/models/prefetch.safetensors"
```

The prefetch file holds a linear expert predictor for each sliding-window layer: its
router weights folded with the feed-forward normalization, then refit on recorded
routing. It only decides which expert weights are cached early; routing and outputs
never depend on it. It matches checkpoint revision `d13b945`, and the runtime refuses
a prefetch file made for other router weights. Weights, tokenizer assets, and the
prefetch file, which is derived from the checkpoint, retain the terms on the model card.

## Generate

```bash
uv run --locked torchrun --standalone --nproc_per_node=8 examples/offline.py \
  --prompt "Write a Python function that merges overlapping intervals." --max-new 8192
```

First startup includes kernel compilation. Repeating a run with the same arguments,
including `--seed` (default 0), produces the same output.
Decode uses mixed precision and may differ numerically from BF16/FP32 reference calculations.

## Serve

```bash
uv run --locked --extra serve torchrun --standalone --nproc_per_node=8 -m iquestrt.serve \
  --host 127.0.0.1 --port 8000
```

The server checks deployment prerequisites before loading model weights; add `--check`
to exit after the check.
The server has no authentication; use an authenticated gateway for remote access.
While it listens on a loopback address, it accepts only loopback `Host` headers,
so web pages cannot reach it through DNS rebinding.

```bash
curl --fail-with-body http://127.0.0.1:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"IQuest-Q1","max_tokens":8192,"messages":[{"role":"user","content":"What is 17 + 25?"}]}'
```

With the Claude Code CLI installed, run `bash tools/claude_code.sh http://127.0.0.1:8000`.
For an authenticated gateway, set `ANTHROPIC_AUTH_TOKEN` or `ANTHROPIC_API_KEY`.

The API supports Anthropic Messages (`POST /v1/messages` and `/v1/messages/count_tokens`),
streaming, and client tool calls.
Images, constrained JSON output, and continuous batching are unsupported.
Sampling is fixed at temperature 1, top-k 20, and top-p 0.95; requests with other
values return HTTP 400.
The context window holds 200,704 tokens for the prompt and its output together;
a prompt that fills it returns HTTP 400.

| Parameter | Support |
|---|---|
| `max_tokens` | Includes thinking and is limited by remaining context; increase it if output ends before an answer |
| `thinking.type` | `enabled` (default), `adaptive`, or `disabled`; enabled and adaptive use the same thinking prompt |
| `thinking.budget_tokens` | Unsupported; any request containing it returns HTTP 400 |
| `output_config.effort` | `low`, `medium`, `high`, or `max`; accepted as metadata without changing generation |
| Tool `strict` | Only `false`; tool output is not constrained to its schema |
| `tool_choice` | `auto` or `none`; `any`, `tool`, and `disable_parallel_tool_use: true` return HTTP 400 |
| `stop_sequences` | Up to 16 sequences of at most 128 characters each |

Tool calls require all matching results in the next user message, before ordinary text.
Special-token text inside request content is tokenized as plain text; only the chat
template produces control tokens. Rank 0 keeps up to 16 MiB of tokenized pieces of recent
prompts, so resent conversation history is not tokenized again.
Requests must use `Content-Type: application/json`; invalid requests return HTTP 400.
Bodies over 32 MiB return HTTP 413, and bodies not received within 30 seconds return HTTP 408.
The server runs one request with up to eight queued; saturation returns HTTP 529,
and an unavailable engine returns HTTP 503. Retry 529/503 responses with backoff.
Disconnecting cancels generation; interrupted streams end with an error.
Readiness and statistics are available at `/health`, `/v1/models`, `/metrics`, and `/stats`.

The CPU tests run without a GPU or checkpoint:

```bash
uv run --locked --extra serve python -m unittest discover -s tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and GPU validation.

## Benchmark

After model preparation, measure a prompt with repeated cold calls:

```bash
uv run --locked torchrun --standalone --nproc_per_node=8 examples/offline.py \
  --prompt "Write a Python function that merges overlapping intervals. Explain its complexity." \
  --seed 0 --max-new 512 --benchmark-runs 3 \
  --model-revision d13b9457e78909349253d96e22bc02dca46aaeeb \
  --output ../iquestrt-benchmark.json
```

Choose a new output file. The JSON records timing, output hashes, environment
metadata, and cross-rank agreement. Running the example does not require Git;
the report does not collect the repository commit or working-tree status.
Complete-call timing includes prefill and graph
capture; decode timing excludes the first prefill token. Compilation inside a
measured call is included. Model revision is caller-supplied metadata.

## Docker

Run the [image on Docker Hub](https://hub.docker.com/r/iquestlabworkspace/iquestrt),
built from this repository's `Dockerfile`, after preparing `./models` on the host:

```bash
docker run --rm --gpus all --ipc=host --network=host \
  -v "$PWD/models:/models:ro" -v iquestrt-cache:/root/.cache \
  -e IQUESTRT_MODEL=/models/IQuest-Q1-FP8 \
  -e IQUESTRT_PREFETCH_FOLD=/models/prefetch.safetensors \
  iquestlabworkspace/iquestrt:0.1.0 \
  torchrun --standalone --nproc_per_node=8 -m iquestrt.serve
```

Requires the NVIDIA container runtime with access to all eight GPUs and NVLink.
The `iquestrt-cache` volume keeps compiled kernels between runs.

## License

[Apache-2.0](LICENSE) for the runtime. Code adapted from PyTorch keeps its
[BSD-3-Clause license](iquestrt/kernels/LICENSE-PyTorch.txt). The IQuest name and
logo are not licensed for use. Bundled fonts use the
[SIL Open Font License 1.1](docs/assets/brand/OFL.txt); see their
[provenance](docs/assets/brand/FONTS.txt) and [NOTICE](NOTICE).
