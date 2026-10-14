"""Encode tasks.parquet with only the checkpoint's Wan text encoder loaded."""
# ruff: noqa: E402

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import save_file

from openwam.deploy.model_loader import _find_latest_checkpoint
from openwam.deploy.prompt_embeddings import FORMAT, TEXT_PREFIX, checkpoint_identity
from openwam.model.video_backbone.wan.encode import encode_text
from openwam.model.video_backbone.wan.pipeline_builder import _build_tokenizer, _import_class

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-name")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT.parent / "d4rt_real/v1/multitask_3.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--force", action="store_true", help="Replace an existing embedding artifact")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output = args.output or args.ckpt_dir / "prompt_embeddings.safetensors"
    if output.exists() and not args.force:
        parser.error(f"Output already exists: {output}; use --force to replace it")
    cfg = OmegaConf.load(args.ckpt_dir / "config.yaml")
    if not str(cfg.model.video_backbone.name).startswith("wan"):
        parser.error("This precompute tool currently supports Wan checkpoints")
    checkpoint = args.ckpt_dir / args.ckpt_name if args.ckpt_name else Path(_find_latest_checkpoint(str(args.ckpt_dir)))
    tasks = pd.read_parquet(args.dataset / "meta/tasks.parquet").sort_values("task_index")
    prompts = [
        {"task_index": int(row.task_index), "prompt": str(name), "key": f"{index:03d}"}
        for index, (name, row) in enumerate(tasks.iterrows())
    ]
    if not prompts or any(not item["prompt"].strip() for item in prompts):
        parser.error("Dataset tasks must have nonempty prompt text in the parquet index")
    identity = checkpoint_identity(checkpoint, cfg)
    component = identity["text_component"]
    cls = _import_class(component["model_class"])
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[str(cfg.training.mixed_precision)]
    started = time.perf_counter()
    logger.info("Constructing text encoder on meta (no random weight allocation)")
    with torch.device("meta"):
        encoder = cls(**component.get("extra_kwargs", {}))
    # Read the encoder tensors alone; never materialize the video/action models.
    logger.info("Loading only text encoder from %s", checkpoint)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        weights = {
            key.removeprefix(TEXT_PREFIX): source.get_tensor(key)
            for key in source.keys()
            if key.startswith(TEXT_PREFIX)
        }
    encoder.load_state_dict(weights, strict=True, assign=True)
    del weights
    encoder.to(device=args.device, dtype=dtype).eval()
    tokenizer = _build_tokenizer(identity["tokenizer_config"], str(args.ckpt_dir))
    logger.info("Text encoder ready in %.2fs; encoding %d tasks", time.perf_counter() - started, len(prompts))
    tensors = {}
    with torch.inference_mode():
        for item in prompts:
            tick = time.perf_counter()
            context, lengths = encode_text(
                [item["prompt"]], tokenizer=tokenizer, text_encoder=encoder, device=args.device
            )
            tensors[f"context_{item['key']}"] = context.cpu().contiguous()
            tensors[f"seq_lens_{item['key']}"] = lengths.cpu().contiguous()
            logger.info("Task %d: %.2fs — %s", item["task_index"], time.perf_counter() - tick, item["prompt"])
    # Check source stability before committing the cache.
    if checkpoint_identity(checkpoint, cfg) != identity:
        raise RuntimeError("Checkpoint/tokenizer changed during precomputation")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        save_file(
            tensors,
            str(temporary),
            metadata={
                "format": FORMAT,
                "identity": json.dumps(identity, sort_keys=True),
                "prompts": json.dumps(prompts),
            },
        )
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Saved %d prompts to %s (%.2f MiB)", len(prompts), output, output.stat().st_size / 2**20)


if __name__ == "__main__":
    main()
