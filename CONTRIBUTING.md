# Contributing

Run all commands below from the repository root. See [README.md](README.md) for
requirements, installation, and model preparation.

## Development setup

```bash
uv sync --locked --extra serve
```

`uv.lock` pins every dependency and resolves only for Linux x86_64. To change a
dependency, edit `pyproject.toml`, run `uv lock`, and commit both files.

There is no automatic formatter; follow the style of the surrounding code.

## Testing

### CPU tests

`tests/` holds `unittest` tests for the serving layer and the deployment checks. They
need no GPU, checkpoint, or network:

```bash
uv run --locked --extra serve python -m unittest discover -s tests
```

Add or update tests there when changing request validation, chat rendering,
tokenization, or HTTP behavior.

### GPU validation

Changes to kernels, the engine, or weight loading need eight GPUs with compute
capability 10.0 and a prepared checkpoint. Run the deployment check and the generation
example from the README.

Modules in `iquestrt/kernels/q8/` build kernels by exact text replacement on other
kernels' CUDA sources. Editing a kernel's source can break those replacements, and the
failure appears only when the kernels compile on a GPU.

A change that should not alter numerics must reproduce the generated tokens exactly.
Run the same benchmark on the base commit and on your change, writing each report to
a new file, then compare `output_sha256` and `acceptance_sha256` across the two reports:

```bash
uv run --locked torchrun --standalone --nproc_per_node=8 examples/offline.py \
  --prompt "Write a Python function that merges overlapping intervals. Explain its complexity." \
  --seed 0 --max-new 512 --benchmark-runs 2 \
  --model-revision d13b9457e78909349253d96e22bc02dca46aaeeb \
  --output ../iquestrt-base.json
```

For server changes, start the server and exercise the affected endpoints.

## Pull requests

Keep each pull request focused. Describe the behavior change and how you validated
it: the commands, the hardware, and the results, such as matching output hashes.
Contributions are licensed under the Apache License 2.0, as stated in Section 5 of
the [LICENSE](LICENSE).

## Security

Do not report vulnerabilities in public issues. Use **Report a vulnerability** on the
repository's Security tab to report them privately.
