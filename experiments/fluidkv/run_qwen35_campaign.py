#!/usr/bin/env python3
"""Execute sequential, same-engine comparisons on independent native splits."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dualmap-source", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", required=True)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--output-cap", type=int, default=0)
    parser.add_argument("--retire-at", type=float, default=5)
    parser.add_argument("--retire-engine", type=int, default=2)
    parser.add_argument("--transfer-base-port", type=int, default=30200)
    parser.add_argument("--time-scale", type=float, default=1)
    parser.add_argument("--selection", choices=["chronological", "exact-parent-cohort"], default="chronological")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--splits", nargs="+", default=["A", "Coder", "Thinking"])
    parser.add_argument("--policies", nargs="+", default=["dualmap", "dualmap-full-copy", "fluidkv-planner-prototype", "sticky"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = {"A": "qwen_traceA_blksz_16.jsonl", "Coder": "qwen_coder_blksz_16.jsonl", "Thinking": "qwen_thinking_blksz_16.jsonl"}
    manifest = {"scope": "Initial same-model logical-retirement campaign, not full FluidKV or physical autoscaling evaluation.", "runs": []}
    for index, split in enumerate(args.splits):
        output = args.output / split
        cmd = [sys.executable, "experiments/fluidkv/benchmark_routing.py",
               "--trace", str(args.trace_root / files[split]), "--dualmap-source", str(args.dualmap_source),
               "--output", str(output), "--engines", *args.engines,
               "--requests", str(args.requests), "--output-cap", str(args.output_cap),
               "--time-scale", str(args.time_scale), "--selection", args.selection,
               "--duration", "7200", "--context-length", "32768", "--max-input", "32768",
               "--cache-tokens", "65536", "--bytes-per-token", "20480", "--slo", "5",
               "--hybrid-snapshot-slots", "48",
               "--retire-at", str(args.retire_at), "--repetitions", str(args.repetitions),
               "--retire-engine", str(args.retire_engine),
               "--transfer-port", str(args.transfer_base_port + index * 100), "--run-timeout", "3600",
               "--request-timeout", "1800", "--causal", "--policies", *args.policies]
        started = time.time()
        with (args.output / f"{split}.log").open("w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, env=os.environ.copy())
        manifest["runs"].append({"split": split, "command": cmd, "exit_code": code,
                                 "started_unix": started, "elapsed_seconds": time.time() - started})
        (args.output / "campaign.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(manifest["runs"][-1]), flush=True)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
