"""Generate a response, or measure repeated cold calls, on eight NVIDIA GPUs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def provenance(model, revision):
    """Record the loaded package and local assets without hashing all model weights."""
    import iquestrt
    import torch
    from packaging.utils import canonicalize_name
    from iquestrt.paths import prefetch_fold

    root = Path(iquestrt.__file__).resolve().parent.parent
    source = {"commit": None, "dirty": None}
    if (root / ".git").exists():
        source = {
            "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
            "dirty": bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)),
        }
    files = [Path(model) / name for name in
             ("config.json", "mtp/config.json", "tokenizer_config.json", "tokenizer.json")]
    files.append(Path(prefetch_fold()))
    package_names = {canonicalize_name(d.metadata["Name"])
                     for d in importlib.metadata.distributions()}
    return {
        "runtime": source, "declared_model_revision": revision,
        "asset_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        # Report installed versions even when build tools expose vendored metadata.
        "packages": {name: importlib.metadata.version(name) for name in sorted(package_names)},
        "gpu_compute_capability": list(torch.cuda.get_device_capability()),
        "torch_cuda": torch.version.cuda,
        "driver": subprocess.check_output([
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).splitlines(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="checkpoint directory; defaults to IQUESTRT_MODEL")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--benchmark-runs", type=int, default=0,
                        help="measure N cold calls with the same prompt and seed")
    parser.add_argument("--model-revision", help="declared checkpoint revision for benchmark provenance")
    parser.add_argument("--output", type=Path, help="new benchmark JSON file; parent must exist")
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        parser.error("run with torchrun --standalone --nproc_per_node=8")
    if args.max_new < 1:
        parser.error("--max-new must be positive")
    if args.benchmark_runs < 0:
        parser.error("--benchmark-runs must be nonnegative")
    if args.benchmark_runs and (not args.output or not args.model_revision):
        parser.error("benchmark mode requires --output and --model-revision")
    if not args.benchmark_runs and (args.output or args.model_revision):
        parser.error("--output and --model-revision require --benchmark-runs")
    if args.output and (args.output.exists() or not args.output.parent.is_dir()):
        parser.error("--output must name a new file in an existing directory")

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from iquestrt.config import load_config, resolve_draft
    from iquestrt.engine import Engine
    from iquestrt.paths import default_model
    from iquestrt.weights import load_weights
    from iquestrt.startup import check_deployment, progress

    model = args.model or default_model()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    try:
        cfg = load_config(model)
        draft = resolve_draft(model, cfg)
        progress("checking deployment")
        check_deployment(cfg, draft)
        progress("loading target and draft weights")
        weights = load_weights(cfg, tp_rank=dist.get_rank(), tp_size=8,
                               device="cuda", loader="instanttensor", draft=draft)
        progress("building runtime and compiling kernels")
        engine = Engine(cfg, weights=weights, draft=draft)
        tokenizer = AutoTokenizer.from_pretrained(model)
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True,
            add_generation_prompt=True)
        stops = set(cfg.eos_token_ids)
        vocab = tokenizer.get_vocab()
        stops.update(vocab[t] for t in ("<|im_end|>", "<|iquest_end|>") if t in vocab)
        records = []
        for run in range(args.benchmark_runs or 1):
            torch.cuda.synchronize()
            dist.barrier()
            started = time.perf_counter()
            tokens, accepted = engine.generate_mtp(ids, args.max_new, seed=args.seed,
                                                   eos_token_id=stops)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if args.benchmark_runs:
                local = dict(call_seconds=elapsed, decode_seconds=engine.mtp_decode_ms / 1000,
                             output_tokens=len(tokens), timed_tokens=engine.mtp_timed_tokens,
                             output_sha256=digest(tokens), acceptance_sha256=digest(accepted),
                             steps=len(accepted),
                             stop_reason="end_turn" if tokens[-1] in stops else "max_tokens")
                ranks = [None] * dist.get_world_size()
                dist.all_gather_object(ranks, local)
                invariant = lambda r: {k: v for k, v in r.items() if not k.endswith("seconds")}
                if any(invariant(r) != invariant(local) for r in ranks):
                    raise RuntimeError("benchmark outputs differ across ranks")
                record = dict(local, run=run, rank_agreement=True,
                              call_seconds=max(r["call_seconds"] for r in ranks),
                              decode_seconds=max(r["decode_seconds"] for r in ranks))
                record["decode_tokens_per_second"] = (
                    record["timed_tokens"] / record["decode_seconds"] if record["timed_tokens"] else None)
                records.append(record)
                progress(f"benchmark run {run + 1}/{args.benchmark_runs} complete")
        if args.benchmark_runs and dist.get_rank() == 0:
            timed = sum(r["timed_tokens"] for r in records)
            report = dict(
                schema_version=1, provenance=provenance(model, args.model_revision),
                prompt=args.prompt, prompt_sha256=hashlib.sha256(args.prompt.encode()).hexdigest(),
                input_tokens=len(ids), input_ids_sha256=digest(ids), seed=args.seed,
                max_new=args.max_new, sampling=dict(temperature=1, top_k=20, top_p=0.95),
                cache_policy="cold prefill and fresh graph capture on every call",
                timing="max rank wall; call includes prefill/capture; decode excludes first token",
                runs=records, pooled_decode_tokens_per_second=(
                    timed / sum(r["decode_seconds"] for r in records) if timed else None))
            with args.output.open("x") as f:
                json.dump(report, f, indent=2, allow_nan=False)
                f.write("\n")
            progress(f"benchmark written to {args.output}")
        elif not args.benchmark_runs and dist.get_rank() == 0:
            print(tokenizer.decode(tokens, skip_special_tokens=True))
            progress(f"generated {len(tokens)} tokens; "
                     f"stop_reason={'end_turn' if tokens[-1] in stops else 'max_tokens'}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
