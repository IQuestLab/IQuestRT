"""Start the Anthropic Messages API server on the fixed TP=8 configuration."""

import argparse
import os
from datetime import timedelta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", help="target checkpoint; draft is its mtp/ directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--served-model-name", default="IQuest-Q1")
    ap.add_argument("--chat-template", help="verified Jinja template override")
    ap.add_argument("--check", action="store_true", help="check deployment without loading weights")
    args = ap.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 8:
        ap.error("IQuestRT serving requires torchrun --nproc_per_node=8")

    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from iquestrt.config import load_config, resolve_draft
    from iquestrt.engine import Engine
    from iquestrt.paths import default_model
    from iquestrt.weights import load_weights
    from iquestrt.startup import check_deployment, progress

    from .driver import EngineDriver
    from .server import Server

    model = args.model or default_model()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    try:
        group = dist.new_group(backend="gloo", timeout=timedelta(minutes=30))
        rank = dist.get_rank()
        progress("checking deployment")
        cfg = load_config(model)
        draft = resolve_draft(model, cfg)
        check_deployment(cfg, draft)
        tokenizer = AutoTokenizer.from_pretrained(model)
        if args.chat_template:
            from pathlib import Path
            tokenizer.chat_template = Path(args.chat_template).read_text(encoding="utf-8")
            if not tokenizer.chat_template.strip():
                raise ValueError("chat template must not be empty")
        dist.barrier()
        if args.check:
            progress("deployment check passed; weights were not loaded")
            return
        progress("loading target and draft weights")
        weights = load_weights(cfg, tp_rank=rank, tp_size=8, device="cuda",
                               loader="instanttensor", draft=draft)
        progress("building runtime and compiling kernels")
        engine = Engine(cfg, weights=weights, draft=draft)
        engine.enable_serving_reuse()
        driver = EngineDriver(engine, tokenizer, group, rank=rank, local_rank=local_rank,
                              model_name=args.served_model_name)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            driver.start()
            try:
                progress(f"starting HTTP server on {args.host}:{args.port}")
                Server(driver, host=args.host, port=args.port).run()
            finally:
                driver.close()
        else:
            driver.run()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
