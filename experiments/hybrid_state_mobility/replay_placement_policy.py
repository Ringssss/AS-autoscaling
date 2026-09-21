#!/usr/bin/env python3
"""Replay real prefix links with token-only and heterogeneous move decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


HYBRID_BYTES_PER_TOKEN = 20_480
HYBRID_SNAPSHOT_BYTES = 64_389_120
MHA_BYTES_PER_TOKEN = 147_456


def replay_trace(
    path: Path, *, bandwidth_bytes_per_second: int, gap_seconds: float
) -> dict:
    data = np.load(path)
    mask = (
        data["usable_parent_link"].astype(bool)
        & (data["parent_index"] >= 0)
        & np.isfinite(data["arrival_gap_s"])
        & (data["parent_prefix_tokens"] >= 0)
    )
    prefixes = data["parent_prefix_tokens"][mask].astype(np.float64)

    token_estimate = prefixes * MHA_BYTES_PER_TOKEN / bandwidth_bytes_per_second
    actual_copy = (
        prefixes * HYBRID_BYTES_PER_TOKEN + HYBRID_SNAPSHOT_BYTES
    ) / bandwidth_bytes_per_second
    token_moves = token_estimate <= gap_seconds
    actual_moves = actual_copy <= gap_seconds

    return {
        "trace": path.name,
        "links": int(prefixes.size),
        "actual_move_rate": float(np.mean(actual_moves)) if prefixes.size else None,
        "token_only_move_rate": float(np.mean(token_moves)) if prefixes.size else None,
        "decision_mismatch_rate": (
            float(np.mean(token_moves != actual_moves)) if prefixes.size else None
        ),
        "optimistic_token_only_rate": (
            float(np.mean(token_moves & ~actual_moves)) if prefixes.size else None
        ),
        "conservative_token_only_rate": (
            float(np.mean(~token_moves & actual_moves)) if prefixes.size else None
        ),
    }


def make_report(scenarios: list[dict]) -> str:
    lines = [
        "# Token-Only Placement Replay",
        "",
        "Decision rates compare a Qwen3-8B token-only bytes/token estimate with",
        "Qwen3.5's actual full KV plus one Mamba snapshot at the same bandwidth.",
        "",
        "| Gap (ms) | A mismatch | Coder mismatch | Thinking mismatch | Dominant error |",
        "|---:|---:|---:|---:|---|",
    ]
    for scenario in scenarios:
        by_trace = {item["trace"]: item for item in scenario["traces"]}
        a = by_trace["qwen_traceA_requests.npz"]
        coder = by_trace["qwen_coder_requests.npz"]
        thinking = by_trace["qwen_thinking_requests.npz"]
        optimistic = sum(
            item["optimistic_token_only_rate"] for item in scenario["traces"]
        )
        conservative = sum(
            item["conservative_token_only_rate"] for item in scenario["traces"]
        )
        lines.append(
            f"| {scenario['gap_ms']:g} | {a['decision_mismatch_rate']:.3%} | "
            f"{coder['decision_mismatch_rate']:.3%} | "
            f"{thinking['decision_mismatch_rate']:.3%} | "
            f"{'optimistic' if optimistic > conservative else 'conservative'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bandwidth-gib-s", type=float, default=20.28)
    parser.add_argument(
        "--gap-ms",
        type=float,
        default=[1, 2, 5, 10, 20, 30, 50, 100],
    )
    args = parser.parse_args()

    bandwidth = int(args.bandwidth_gib_s * 1024**3)
    scenarios = []
    for gap_ms in args.gap_ms:
        results = [
            replay_trace(
                path,
                bandwidth_bytes_per_second=bandwidth,
                gap_seconds=gap_ms / 1000,
            )
            for path in args.trace
        ]
        scenarios.append(
            {
                "gap_ms": gap_ms,
                "bandwidth_gib_s": args.bandwidth_gib_s,
                "traces": results,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "policy_replay.json").write_text(
        json.dumps({"scenarios": scenarios}, indent=2) + "\n"
    )
    report = make_report(scenarios)
    (args.output_dir / "policy_replay.md").write_text(report)
    print(report, end="")


if __name__ == "__main__":
    main()
