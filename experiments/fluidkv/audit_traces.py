#!/usr/bin/env python3
"""Audit native trace fields and prepare immutable replay pilot inputs."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from agentshift.workloads.production import inspect_qwen_rows, parse_kimi_line, surrogate_tokens


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kimi-root", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--max-pair-tokens", type=int, default=8192)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    kimi_counts = Counter()
    received, lengths, cached, output_lengths = [], [], [], []
    request_ids = set()
    sources = []
    paths = sorted(args.kimi_root.glob("[0-2][0-9].redacted.log"))
    if len(paths) != 24:
        raise ValueError("expected 24 hourly Kimi log files")
    with (args.output / "kimi_requests.jsonl").open("w") as normalized:
        for path in paths:
            digest = hashlib.sha256()
            scanned = 0
            with path.open("rb") as handle:
                for line_index, raw in enumerate(handle, 1):
                    digest.update(raw)
                    row = parse_kimi_line(raw.decode("utf-8"))
                    if row is None:
                        continue
                    scanned += 1
                    row.update(source_file=path.name, source_line=line_index)
                    normalized.write(json.dumps(row) + "\n")
                    kimi_counts["requests"] += 1
                    kimi_counts["input_ids_redacted"] += row["input_ids_redacted"]
                    kimi_counts["nonempty_conversation_ids"] += bool(row["conversation_id"])
                    kimi_counts["duplicate_request_ids"] += row["request_id"] in request_ids
                    request_ids.add(row["request_id"])
                    received.append(row["arrival_seconds"])
                    lengths.append(row["input_length"])
                    output_lengths.append(row["output_length"])
                    if row["cached_tokens"] is not None:
                        cached.append(row["cached_tokens"])
                    else:
                        kimi_counts["missing_cached_tokens"] += 1
            sources.append(dict(path=str(path), sha256=digest.hexdigest(), rows=scanned))
            print(json.dumps({"source": path.name, "rows": scanned}), flush=True)
    kimi = dict(kimi_counts)
    kimi.update({
        "sources": sources,
        "arrival_span_seconds": max(received) - min(received),
        "input_p50_p90_p99": np.percentile(lengths, [50, 90, 99]).tolist(),
        "output_p50_p90_p99": np.percentile(output_lengths, [50, 90, 99]).tolist(),
        "cached_token_sum": sum(cached),
        "branch_replay_supported_by_observed_fields": False,
        "limitation": "No token/hash identity or observed conversation linkage; no native DualMap prefix keys.",
    })
    rows = [json.loads(line) for line in args.qwen.open()]
    qwen, pairs = inspect_qwen_rows(rows, args.max_pair_tokens)
    qwen.update(path=str(args.qwen), sha256=sha256(args.qwen))
    by_id = {row["chat_id"]: row for row in rows}
    selected = pairs[:args.pairs]
    arrays = {}
    for pair in selected:
        for name in ("parent", "child"):
            chat_id = pair[name + "_id"]
            arrays[str(chat_id)] = np.asarray(surrogate_tokens(by_id[chat_id]), dtype=np.int32)
    np.savez_compressed(args.output / "qwen_pilot_tokens.npz", **arrays)
    pilot = {
        "source": qwen["path"], "source_sha256": qwen["sha256"],
        "selection": "earliest child arrivals among exact block-aligned parents within token limit",
        "codec": "fluidkv-block-v1; SHAKE256 to 16 IDs in [1000,31000); exact input lengths",
        "scope": "selected pair microbenchmark, not a continuous production replay or observed tool-gap experiment",
        "eligible_pairs": len(pairs), "pairs": selected,
    }
    (args.output / "qwen_pilot_manifest.json").write_text(json.dumps(pilot, indent=2) + "\n")
    summary = {"kimi": kimi, "qwen_bailian_a": qwen}
    (args.output / "trace_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"kimi_requests": kimi["requests"], "qwen": qwen, "pilot_pairs": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
