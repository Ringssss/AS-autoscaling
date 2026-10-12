"""Summarize raw paired tool-window trials, including foreground externalities."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(directory):
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines()]
    failures = [r for r in rows if r.get("failed")]
    if failures:
        raise ValueError("failed trials present; diagnose before summarizing")
    groups = {}
    for row in rows:
        if not all(row["checks"].values()):
            raise ValueError("correctness gate failed")
        key = row["repeat"], row["prefix_tokens"], row["controlled_tool_seconds"]
        by_mode = groups.setdefault(key, {})
        if row["mode"] in by_mode:
            raise ValueError("duplicate policy in paired trial")
        by_mode[row["mode"]] = row
        times = row["timeline"]
        if not (0 <= times["tool_return"] <= times["continuation_dispatch"] <= times["continuation_first_token"] <= times["continuation_finished"]):
            raise ValueError("invalid causal continuation timeline")
        if not (times["foreground_scheduled_arrival"] <= times["foreground_dispatch"] <= times["foreground_first_token"] <= times["foreground_finished"]):
            raise ValueError("invalid foreground timeline")
        if row["mode"] == "copy-during-tool" and not (times["copy_started"] <= times["copy_committed"] <= times["continuation_dispatch"]):
            raise ValueError("copy not committed before continuation")
    modes = {"sticky-hot-owner", "route-cold-idle", "copy-during-tool"}
    if any(set(v) != modes for v in groups.values()):
        raise ValueError("incomplete paired trial")
    result = []
    for length, gap in sorted({(k[1], k[2]) for k in groups}):
        pairs = [v for k, v in groups.items() if k[1:] == (length, gap)]
        summary = {"prefix_tokens": length, "controlled_tool_seconds": gap, "paired_repetitions": len(pairs), "modes": {}}
        for mode in sorted(modes):
            selected = [v[mode] for v in pairs]
            metrics = {}
            for key in ("post_tool_ttft_seconds", "post_tool_e2e_seconds", "tool_start_to_completion_seconds", "foreground_arrival_to_completion_seconds", "copy_exposed_seconds"):
                values = [r[key] for r in selected]
                metrics[key] = {"median": float(np.median(values)), "min": min(values), "max": max(values)}
            metrics["foreground_active_at_tool_return"] = sum(r["foreground_active_at_tool_return"] for r in selected)
            summary["modes"][mode] = metrics
        for baseline in ("sticky-hot-owner", "route-cold-idle"):
            summary["median_paired_ttft_ratio_vs_" + baseline] = float(np.median([
                v[baseline]["post_tool_ttft_seconds"] / v["copy-during-tool"]["post_tool_ttft_seconds"] for v in pairs]))
            summary["median_paired_foreground_delta_seconds_vs_" + baseline] = float(np.median([
                v["copy-during-tool"]["foreground_arrival_to_completion_seconds"] - v[baseline]["foreground_arrival_to_completion_seconds"] for v in pairs]))
        result.append(summary)
    return {"completed_continuations": len(rows), "paired_cases": len(groups), "checks_passed": True,
            "statistics": "Median and range of paired component repetitions; no p99 or population confidence claim.", "cases": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = summarize(args.directory)
    (args.directory / "paired_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
