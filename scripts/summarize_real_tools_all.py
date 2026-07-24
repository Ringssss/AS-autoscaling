from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


SCENARIO_ORDER = [
    "sticky",
    "reroute",
    "agentix",
    "ttl",
    "tokencake-source",
    "tokencake-remote",
    "symphony",
    "on-return",
    "agentshift",
    "oracle",
]


def p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def main(args) -> None:
    source = Path(args.input)
    payload = json.loads(source.read_text())
    grouped = defaultdict(list)
    for record in payload["records"]:
        grouped[
            (
                int(record["prefix_length"]),
                record["tool"],
                record["scenario"],
            )
        ].append(record)
    means = {
        key: statistics.mean(row["post_tool_seconds"] for row in records)
        for key, records in grouped.items()
    }
    rows = []
    for (prefix_length, tool, scenario), records in grouped.items():
        post = [record["post_tool_seconds"] for record in records]
        tool_times = [record["tool_seconds"] for record in records]
        predictions = [record["tool_prediction_seconds"] for record in records]
        mean_post = statistics.mean(post)
        reroute = means[(prefix_length, tool, "reroute")]
        on_return = means[(prefix_length, tool, "on-return")]
        sticky = means[(prefix_length, tool, "sticky")]
        rows.append(
            {
                "prefix_length": prefix_length,
                "tool": tool,
                "scenario": scenario,
                "trials": len(records),
                "tool_mean_ms": statistics.mean(tool_times) * 1000,
                "prediction_mean_ms": statistics.mean(predictions) * 1000,
                "prediction_abs_error_mean_ms": statistics.mean(
                    abs(actual - predicted)
                    for actual, predicted in zip(tool_times, predictions)
                )
                * 1000,
                "post_tool_mean_ms": mean_post * 1000,
                "post_tool_median_ms": statistics.median(post) * 1000,
                "post_tool_p95_ms": p95(post) * 1000,
                "exposed_mean_ms": statistics.mean(
                    record["exposed_migration_seconds"] for record in records
                )
                * 1000,
                "full_prefix_hit_rate": statistics.mean(
                    record["full_prefix_hit"] for record in records
                ),
                "speedup_vs_reroute": reroute / mean_post,
                "speedup_vs_on_return": on_return / mean_post,
                "overhead_vs_sticky": mean_post / sticky,
            }
        )
    order = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    rows.sort(
        key=lambda row: (
            row["prefix_length"],
            row["tool"],
            order[row["scenario"]],
        )
    )
    prefix = Path(args.output_prefix) if args.output_prefix else source.with_suffix("")
    json_path = Path(f"{prefix}-summary.json")
    csv_path = Path(f"{prefix}-summary.csv")
    markdown_path = Path(f"{prefix}-summary.md")
    json_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "baseline_implementation": payload["baseline_implementation"],
                "config": payload["config"],
                "tool_measurements": payload["tool_measurements"],
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
    )
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Real Coding-Tool Baseline Summary",
        "",
        "Literature-named baselines are mechanism-equivalent implementations in "
        "the same SGLang testbed, not official reproductions.",
        "",
        "| Prefix | Tool | Strategy | Tool mean | Post-tool mean | p95 | Full hit | vs reroute | vs on-return |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {prefix_length} | {tool} | {scenario} | {tool_mean_ms:.2f} ms | "
            "{post_tool_mean_ms:.2f} ms | {post_tool_p95_ms:.2f} ms | "
            "{full_prefix_hit_rate:.0%} | {speedup_vs_reroute:.2f}x | "
            "{speedup_vs_on_return:.2f}x |".format(**row)
        )
    markdown_path.write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
                "rows": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-prefix")
    main(parser.parse_args())
