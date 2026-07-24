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
    "shared-cas",
    "on-return",
    "agentshift",
    "oracle",
]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def summarize(input_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(input_path.read_text())
    grouped = defaultdict(list)
    for record in payload["records"]:
        key = (
            int(record["prefix_length"]),
            float(record["gap_ms"]),
            record["scenario"],
        )
        grouped[key].append(record)

    means = {
        key: statistics.mean(row["post_tool_seconds"] for row in rows)
        for key, rows in grouped.items()
    }
    summary_rows = []
    for key, records in grouped.items():
        prefix_length, gap_ms, scenario = key
        post = [row["post_tool_seconds"] for row in records]
        reroute = means.get((prefix_length, gap_ms, "reroute"))
        on_return = means.get((prefix_length, gap_ms, "on-return"))
        sticky = means.get((prefix_length, gap_ms, "sticky"))
        mean_post = statistics.mean(post)
        row = {
            "prefix_length": prefix_length,
            "gap_ms": gap_ms,
            "scenario": scenario,
            "trials": len(records),
            "post_tool_mean_ms": mean_post * 1000,
            "post_tool_median_ms": statistics.median(post) * 1000,
            "post_tool_p95_ms": percentile(post, 0.95) * 1000,
            "next_turn_mean_ms": statistics.mean(
                item["next_turn_seconds"] for item in records
            )
            * 1000,
            "exposed_mean_ms": statistics.mean(
                item["exposed_migration_seconds"] for item in records
            )
            * 1000,
            "handoff_mean_ms": statistics.mean(
                item["handoff_wall_seconds"] for item in records
            )
            * 1000,
            "full_prefix_hit_rate": statistics.mean(
                item["cached_tokens"] >= prefix_length for item in records
            ),
            "mean_cached_tokens": statistics.mean(
                item["cached_tokens"] for item in records
            ),
            "mean_transfer_bytes": statistics.mean(
                item["tier_total_bytes"]
                or item.get("migration_bytes", 0)
                for item in records
            ),
            "speedup_vs_reroute": reroute / mean_post if reroute else None,
            "speedup_vs_on_return": on_return / mean_post if on_return else None,
            "overhead_vs_sticky": mean_post / sticky if sticky else None,
        }
        summary_rows.append(row)
    order = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    summary_rows.sort(
        key=lambda row: (
            row["prefix_length"],
            row["gap_ms"],
            order.get(row["scenario"], len(order)),
        )
    )
    return payload, summary_rows


def write_markdown(path: Path, payload: dict, rows: list[dict]) -> None:
    lines = [
        "# Blocked-Window Baseline Summary",
        "",
        "All named literature baselines are mechanism-equivalent implementations "
        "in the same SGLang testbed, not official reproductions.",
        "",
        f"Source artifact: `{payload.get('source_artifact', '')}`",
        "",
        "| Prefix | Gap | Strategy | Mean post-tool | p95 | Full hit | vs reroute | vs on-return | vs sticky |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        vs_reroute = row["speedup_vs_reroute"]
        vs_on_return = row["speedup_vs_on_return"]
        vs_sticky = row["overhead_vs_sticky"]
        lines.append(
            "| {prefix_length} | {gap_ms:g} ms | {scenario} | "
            "{post_tool_mean_ms:.2f} ms | {post_tool_p95_ms:.2f} ms | "
            "{full_prefix_hit_rate:.0%} | {vs_reroute} | {vs_on_return} | "
            "{vs_sticky} |".format(
                **row,
                vs_reroute=f"{vs_reroute:.2f}x" if vs_reroute else "n/a",
                vs_on_return=(
                    f"{vs_on_return:.2f}x" if vs_on_return else "n/a"
                ),
                vs_sticky=f"{vs_sticky:.2f}x" if vs_sticky else "n/a",
            )
        )
    path.write_text("\n".join(lines) + "\n")


def main(args) -> None:
    input_path = Path(args.input)
    payload, rows = summarize(input_path)
    output_prefix = Path(args.output_prefix) if args.output_prefix else input_path.with_suffix("")
    payload["source_artifact"] = str(input_path)
    json_path = Path(f"{output_prefix}-summary.json")
    csv_path = Path(f"{output_prefix}-summary.csv")
    markdown_path = Path(f"{output_prefix}-summary.md")
    json_path.write_text(json.dumps({"config": payload["config"], "rows": rows}, indent=2, sort_keys=True))
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(markdown_path, payload, rows)
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
