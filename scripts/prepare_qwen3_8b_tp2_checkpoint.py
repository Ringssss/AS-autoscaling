#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


VOCAB_PARALLEL_KEYS = {"lm_head.weight", "model.embed_tokens.weight"}
METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def convert_rank(source: Path, destination: Path, rank: int, tp_size: int) -> None:
    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            tensor = checkpoint.get_tensor(key)
            if key in VOCAB_PARALLEL_KEYS:
                if tensor.shape[0] % tp_size:
                    raise ValueError(f"{key} cannot be divided across {tp_size} ranks")
                rows = tensor.shape[0] // tp_size
                tensor = tensor.narrow(0, rank * rows, rows).contiguous()
            tensors[key] = tensor
    layer_ids = sorted(
        {
            int(key.split(".")[2])
            for key in tensors
            if key.startswith("model.layers.")
        }
    )
    for layer_id in layer_ids:
        attention = f"model.layers.{layer_id}.self_attn"
        qkv_keys = [
            f"{attention}.q_proj.weight",
            f"{attention}.k_proj.weight",
            f"{attention}.v_proj.weight",
        ]
        tensors[f"{attention}.qkv_proj.weight"] = torch.cat(
            [tensors.pop(key) for key in qkv_keys], dim=0
        ).contiguous()
        mlp = f"model.layers.{layer_id}.mlp"
        gate_up_keys = [f"{mlp}.gate_proj.weight", f"{mlp}.up_proj.weight"]
        tensors[f"{mlp}.gate_up_proj.weight"] = torch.cat(
            [tensors.pop(key) for key in gate_up_keys], dim=0
        ).contiguous()
    temporary = destination.with_suffix(".tmp.safetensors")
    save_file(tensors, temporary)
    os.replace(temporary, destination)


def validate(output: Path, tp_size: int, vocab_size: int) -> dict:
    expected_rows = vocab_size // tp_size
    files = []
    for rank in range(tp_size):
        path = output / f"model-rank-{rank}-part-0.safetensors"
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            shapes = {
                key: tuple(checkpoint.get_slice(key).get_shape())
                for key in VOCAB_PARALLEL_KEYS
            }
        if len(keys) != 291:
            raise ValueError(f"rank {rank} has {len(keys)} tensors, expected 291")
        if any(shape[0] != expected_rows for shape in shapes.values()):
            raise ValueError(f"rank {rank} has invalid vocabulary shard shapes: {shapes}")
        files.append({"rank": rank, "path": str(path), "shapes": shapes})
    return {"tp_size": tp_size, "vocab_size": vocab_size, "files": files}


def main(args: argparse.Namespace) -> None:
    config = json.loads((args.input / "config.json").read_text())
    vocab_size = int(config["vocab_size"])
    if vocab_size % args.tp_size:
        raise ValueError("vocabulary size must be divisible by TP size")
    args.output.mkdir(parents=True, exist_ok=True)
    for filename in METADATA_FILES:
        shutil.copy2(args.input / filename, args.output / filename)
    for rank in range(args.tp_size):
        source = args.input / f"model{rank}-mp{args.tp_size}.safetensors"
        destination = args.output / f"model-rank-{rank}-part-0.safetensors"
        if destination.exists() and not args.force:
            continue
        convert_rank(source, destination, rank, args.tp_size)
    result = validate(args.output, args.tp_size, vocab_size)
    (args.output / "agentshift_conversion.json").write_text(
        json.dumps(
            {
                "source": str(args.input.resolve()),
                "operation": (
                    "split replicated vocabulary tensors by rank and fuse each "
                    "layer's q/k/v and gate/up tensors for SGLang sharded_state"
                ),
                **result,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/mnt/models/Qwen3-8B-mp2"))
    parser.add_argument(
        "--output", type=Path, default=Path("/mnt/models/Qwen3-8B-sglang-tp2")
    )
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    main(parser.parse_args())
