"""Persistent Wan prompt embeddings bound to a checkpoint and its tokenizer."""

import hashlib
import json
import struct
from pathlib import Path

import torch
from omegaconf import OmegaConf
from safetensors import safe_open

FORMAT = "openwam.wan_prompt_embeddings.v1"
TEXT_PREFIX = "video_backbone.text_encoder."


def checkpoint_identity(checkpoint_path: str | Path, cfg) -> dict:
    path = Path(checkpoint_path)
    with path.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = stream.read(header_size)
    tokenizer_cfg = OmegaConf.to_container(cfg.model.video_backbone.tokenizer, resolve=True)
    tokenizer_dir = path.parent / tokenizer_cfg["subdir"]
    tokenizer_hash = hashlib.sha256()
    for file in sorted(tokenizer_dir.rglob("*")):
        if file.is_file():
            tokenizer_hash.update(str(file.relative_to(tokenizer_dir)).encode())
            tokenizer_hash.update(file.read_bytes())
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint tokenizer not found: {tokenizer_dir}")
    stat = path.stat()
    return {
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "checkpoint_header_sha256": hashlib.sha256(header).hexdigest(),
        "tokenizer_sha256": tokenizer_hash.hexdigest(),
        "tokenizer_config": tokenizer_cfg,
        "text_component": next(
            OmegaConf.to_container(component, resolve=True)
            for component in cfg.model.video_backbone.components
            if component.attr == "text_encoder"
        ),
        "encoder_dtype": str(cfg.training.mixed_precision),
    }


def load_prompt_embeddings(cache_path, *, checkpoint_path, cfg, device="cpu", dtype=None) -> dict:
    """Read only the small embedding artifact; fail on mismatch or malformed data."""
    expected = checkpoint_identity(checkpoint_path, cfg)
    result = {}
    with safe_open(str(cache_path), framework="pt", device=str(device)) as source:
        metadata = source.metadata() or {}
        if metadata.get("format") != FORMAT or json.loads(metadata.get("identity", "null")) != expected:
            raise ValueError("Prompt embedding cache does not match this checkpoint/tokenizer; regenerate it")
        prompts = json.loads(metadata.get("prompts", "[]"))
        if not prompts:
            raise ValueError("Prompt embedding cache contains no prompts")
        for item in prompts:
            prompt = item["prompt"]
            if not isinstance(prompt, str) or not prompt.strip() or prompt in result:
                raise ValueError("Prompt embedding cache has empty or duplicate prompt keys")
            context = source.get_tensor(f"context_{item['key']}")
            lengths = source.get_tensor(f"seq_lens_{item['key']}")
            if context.shape != (1, 512, 4096) or not context.is_floating_point() or not context.isfinite().all():
                raise ValueError(f"Invalid Wan context for {prompt!r}: {tuple(context.shape)}")
            if lengths.shape != (1,) or lengths.dtype != torch.int64 or not 1 <= lengths.item() <= 512:
                raise ValueError(f"Invalid Wan text sequence length for {prompt!r}")
            if torch.count_nonzero(context[:, lengths.item() :]).item() != 0:
                raise ValueError(f"Wan padding must be zero for {prompt!r}")
            result[prompt] = (context.to(dtype=dtype or context.dtype), lengths)
    return result
