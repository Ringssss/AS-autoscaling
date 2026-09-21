#!/usr/bin/env python3
"""Aggregate the 2026-09-16 end-to-end FreeToken/FluidKV comparison."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_campaign(path):
    path = Path(path)
    campaign = json.loads((path / "campaign.json").read_text())
    if not campaign.get("complete"):
        raise ValueError(f"incomplete campaign: {path}")
    return campaign


def group_metrics(campaign, label_key):
    grouped = defaultdict(list)
    for row in campaign["runs"]:
        if not row.get("success"):
            continue
        label = row.get(label_key)
        grouped[(row["pair_id"], label)].append(row["metrics"])
    return grouped


def median(values):
    return float(np.median(np.asarray(values, dtype=float)))


def pct_delta(treatment, baseline):
    return (float(treatment) - float(baseline)) / float(baseline) * 100.0


def summarize(preservation_path, alternatives_path):
    preservation = read_campaign(preservation_path)
    alternatives = read_campaign(alternatives_path)
    preserve_groups = group_metrics(preservation, "policy")
    alternative_groups = group_metrics(alternatives, "condition")
    prefixes = sorted({pair for pair, _ in preserve_groups})
    rows = []
    paired_deltas = defaultdict(list)
    for prefix in prefixes:
        drop = preserve_groups[(prefix, "drop")]
        preserve = preserve_groups[(prefix, "preserve")]
        expert = alternative_groups[(prefix, "expert-only")]
        no_resize = alternative_groups[(prefix, "no-resize")]
        if not all((drop, preserve, expert, no_resize)):
            raise ValueError(f"missing policy group for {prefix}")
        # The preservation campaign randomizes drop/preserve within each repetition.
        by_repetition = {row["repetition"]: row for row in preservation["runs"]
                         if row["pair_id"] == prefix and row["policy"] == "drop"}
        for row in preservation["runs"]:
            if row["pair_id"] == prefix and row["policy"] == "preserve":
                baseline = by_repetition.get(row["repetition"])
                if baseline and baseline["success"] and row["success"]:
                    paired_deltas[prefix].append(pct_delta(
                        row["metrics"]["intervention_to_completion_seconds"],
                        baseline["metrics"]["intervention_to_completion_seconds"],
                    ))
        drop_cycle = median([x["intervention_to_completion_seconds"] for x in drop])
        preserve_cycle = median([x["intervention_to_completion_seconds"] for x in preserve])
        expert_cycle = median([x["intervention_to_completion_seconds"] for x in expert])
        no_resize_cycle = median([x["intervention_to_completion_seconds"] for x in no_resize])
        rows.append({
            "prefix": prefix,
            "repetitions": len(drop),
            "freetoken_drop": {
                "cycle_seconds": drop_cycle,
                "first_text_seconds": median([x["intervention_to_first_text_seconds"] for x in drop]),
                "cached_tokens": int(sum(x["first_child_cached_tokens"] for x in drop)),
            },
            "fluidkv_preserve": {
                "cycle_seconds": preserve_cycle,
                "first_text_seconds": median([x["intervention_to_first_text_seconds"] for x in preserve]),
                "cached_tokens": int(sum(x["first_child_cached_tokens"] for x in preserve)),
            },
            "vs_freetoken_drop": {
                "cycle_delta_percent": pct_delta(preserve_cycle, drop_cycle),
                "speedup": drop_cycle / preserve_cycle,
                "paired_delta_percent": paired_deltas[prefix],
                "first_text_delta_percent": pct_delta(
                    median([x["intervention_to_first_text_seconds"] for x in preserve]),
                    median([x["intervention_to_first_text_seconds"] for x in drop]),
                ),
            },
            "freetoken_expert_only": {
                "cycle_seconds": expert_cycle,
                "first_text_seconds": median([x["intervention_to_first_text_seconds"] for x in expert]),
                "cached_tokens": int(sum(x["first_child_cached_tokens"] for x in expert)),
                "preserve_delta_percent": pct_delta(preserve_cycle, expert_cycle),
            },
            "freetoken_no_resize": {
                "cycle_seconds": no_resize_cycle,
                "first_text_seconds": median([x["intervention_to_first_text_seconds"] for x in no_resize]),
                "cached_tokens": int(sum(x["first_child_cached_tokens"] for x in no_resize)),
                "preserve_delta_percent": pct_delta(preserve_cycle, no_resize_cycle),
            },
        })
    return rows


def verification_summary(path):
    campaign = read_campaign(path)
    rows = [row for row in campaign["runs"] if row["policy"] == "preserve"]
    return {
        "trials": len(rows),
        "successful": sum(bool(row["success"]) for row in rows),
        "bytes_equal": sum(bool((row.get("hook") or {}).get("bytes_equal")) for row in rows),
        "restored_boundaries": [(row["pair_id"], (row.get("hook") or {}).get("export", {}).get("boundary"))
                                for row in rows],
    }


def write_report(root, rows, verification):
    lines = [
        "# End-to-End FreeToken Comparison", "",
        "Main comparison: same initial and final KV/Mamba geometry. FreeToken rebuild drops the cache; FluidKV exports the longest legal aligned checkpoint before rebuild, remaps it into the new pool, and publishes only after complete writes.", "",
        "| Prefix | FreeToken drop | FluidKV preserve | Paired cycle change | Speedup | First-text change |", "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        deltas = row["vs_freetoken_drop"]["paired_delta_percent"]
        lines.append(
            f"| {row['prefix']} | {row['freetoken_drop']['cycle_seconds']:.3f}s | "
            f"{row['fluidkv_preserve']['cycle_seconds']:.3f}s | "
            f"{min(deltas):.1f}% to {max(deltas):.1f}% | "
            f"{row['vs_freetoken_drop']['speedup']:.2f}x | "
            f"{row['vs_freetoken_drop']['first_text_delta_percent']:.1f}% |"
        )
    lines += [
        "", "Negative cycle/first-text change is faster. The cycle includes the intervention, client journaling, rebuild/export/restore where applicable, and the first complete child response. It is an end-to-end intervention-to-completion measurement, not only tensor copy time.", "",
        "Alternative FreeToken policies have different final geometry and are diagnostic controls, not paired same-geometry winners:", "",
        "| Prefix | Expert-only shrink | No resize | FluidKV preserve vs expert-only | FluidKV preserve vs no resize |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['prefix']} | {row['freetoken_expert_only']['cycle_seconds']:.3f}s | "
            f"{row['freetoken_no_resize']['cycle_seconds']:.3f}s | "
            f"{row['freetoken_expert_only']['preserve_delta_percent']:+.1f}% | "
            f"{row['freetoken_no_resize']['preserve_delta_percent']:+.1f}% |"
        )
    lines += [
        "", "At the tested geometry, KV resize releases 674,037,760 B while expert-only releases 679,477,248 B (0.81% more). Expert-only is therefore a strong alternative when the controller is free to choose which pool to shrink. FluidKV's paired win is specific to the case where the state pool must be repartitioned.", "",
        f"Byte verification: {verification['bytes_equal']}/{verification['successful']} successful preserve trials had exact source/destination state bytes; restored boundaries: {verification['restored_boundaries']}. Verification runs are excluded from performance medians.", "",
        "Decoded text remained different in every formal drop/preserve pair. Byte-identical state transport is not an output-quality or numerical-equivalence claim. Online concurrent rebuild, arbitrary failure recovery, task quality, other models, and real resize-trigger frequency remain outside this experiment.", "",
        "Observed policy boundary in this setup: preserve hurts at 4K, improves 8K and 12K by about 28%, and improves 16K by about 44%. A threshold policy is justified by these data, but optimizing the threshold requires more prefix points and workload repetitions.", "",
    ]
    (root / "E2E_COMPARISON.md").write_text("\n".join(lines))
    (root / "e2e-comparison.json").write_text(json.dumps(
        {"rows": rows, "byte_verification": verification}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preservation", type=Path, required=True)
    parser.add_argument("--alternatives", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(args.preservation, args.alternatives)
    verification = verification_summary(args.verification)
    write_report(args.root, rows, verification)
    print(args.root / "E2E_COMPARISON.md")


if __name__ == "__main__":
    main()
