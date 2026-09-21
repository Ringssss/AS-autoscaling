#!/usr/bin/env python3
"""Build content controls from recorded ToolBench function observations."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from experiments.fluidkv.audit_traces import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    pairs, arrays = [], {}
    for batch in pq.ParquetFile(args.parquet).iter_batches(batch_size=32):
        for row in batch.to_pylist():
            conv = row["conversations"]
            messages = list(zip(conv["from"], conv["value"]))
            prefix = []
            for index, (role, value) in enumerate(messages):
                if role in ("function", "tool") and prefix:
                    parent = tokenizer.encode("\n".join(prefix) + "\n", add_special_tokens=False).ids
                    suffix = tokenizer.encode(f"{role}: {value}\nassistant:", add_special_tokens=False).ids
                    child = parent + suffix
                    if len(parent) >= 64 and len(child) <= args.max_tokens:
                        ordinal = len(pairs)
                        parent_id, child_id = 2 * ordinal, 2 * ordinal + 1
                        pairs.append({
                            "parent_id": parent_id, "child_id": child_id,
                            "parent_tokens": len(parent), "child_tokens": len(child),
                            "source_record_sha256": hashlib.sha256(row["id"].encode()).hexdigest(),
                            "function_message_index": index,
                            "arrival_delta_seconds": None,
                        })
                        arrays[str(parent_id)] = np.array(parent, dtype=np.int32)
                        arrays[str(child_id)] = np.array(child, dtype=np.int32)
                        break
                prefix.append(f"{role}: {value}")
            if len(pairs) >= args.pairs:
                break
        if len(pairs) >= args.pairs:
            break
    if len(pairs) != args.pairs:
        raise ValueError(f"only {len(pairs)} eligible conversations")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "source": str(args.parquet), "source_sha256": sha256(args.parquet),
        "tokenizer": str(args.tokenizer), "tokenizer_sha256": sha256(args.tokenizer),
        "scope": "Recorded function-call text control; serialized role labels; no external APIs executed and no observed tool latency.",
        "codec": "Tokenize parent text and function-observation suffix separately, concatenate IDs to preserve the exact boundary.",
        "pairs": pairs,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    np.savez_compressed(args.output / "tokens.npz", **arrays)
    print(json.dumps({"pairs": len(pairs), "parent_tokens": [p["parent_tokens"] for p in pairs]}))


if __name__ == "__main__":
    main()
