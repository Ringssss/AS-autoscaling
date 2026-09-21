#!/usr/bin/env python3
"""Record native split coverage without inventing parent or tool events."""

import argparse
import json
from pathlib import Path

import numpy as np

from agentshift.workloads.production import inspect_qwen_rows
from experiments.fluidkv.audit_traces import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {}
    for name, filename in (("A", "qwen_traceA_blksz_16.jsonl"), ("Coder", "qwen_coder_blksz_16.jsonl"), ("Thinking", "qwen_thinking_blksz_16.jsonl")):
        path = args.root / filename
        rows = [json.loads(line) for line in path.open()]
        audit, _ = inspect_qwen_rows(rows, max_pair_tokens=32768)
        by_id = {row["chat_id"]: row for row in rows}
        linked = [row for row in rows if row["parent_chat_id"] in by_id]
        audit.update({
            "path": str(path), "sha256": sha256(path),
            "input_p50_p90_p99": np.percentile([row["input_length"] for row in rows], [50, 90, 99]).tolist(),
            "output_p50_p90_p99": np.percentile([row["output_length"] for row in rows], [50, 90, 99]).tolist(),
            "arrival_delta_p50_p90_p99": np.percentile([row["timestamp"] - by_id[row["parent_chat_id"]]["timestamp"] for row in linked], [50, 90, 99]).tolist(),
            "combined_length_over_32768": sum(row["input_length"] + row["output_length"] > 32768 for row in rows),
            "timing_warning": "Arrival deltas include unobserved inference and application/user/tool time.",
        })
        result[name] = audit
        print(json.dumps({"split": name, "requests": len(rows), "parent_links": len(linked), "combined_over_context": audit["combined_length_over_32768"]}), flush=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
