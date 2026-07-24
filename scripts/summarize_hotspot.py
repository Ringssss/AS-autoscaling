from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


SCENARIO_ORDER = [
    "sticky",
    "agentix",
    "ttl",
    "tokencake-source",
    "reroute",
    "ttl-reroute",
    "on-return",
    "agentix-on-return",
    "agentshift",
]


def main(args) -> None:
    source = Path(args.input)
    payload = json.loads(source.read_text())
    grouped = defaultdict(list)
    for record in payload["records"]:
        grouped[
            (
                int(record["prefix_length"]),
                int(record["agent_count"]),
                record["return_pattern"],
                record["scenario"],
            )
        ].append(record)
    means = {
        key: statistics.mean(row["makespan_seconds"] for row in records)
        for key, records in grouped.items()
    }
    rows = []
    for key, records in grouped.items():
        prefix_length, agent_count, pattern, scenario = key
        mean_makespan = means[key]
        reference_key = (prefix_length, agent_count, pattern)
        sticky = means.get((*reference_key, "sticky"))
        reroute = means.get((*reference_key, "reroute"))
        on_return = means.get((*reference_key, "on-return"))
        rows.append(
            {
                "prefix_length": prefix_length,
                "agent_count": agent_count,
                "return_pattern": pattern,
                "scenario": scenario,
                "trials": len(records),
                "makespan_mean_ms": mean_makespan * 1000,
                "makespan_p95_ms": max(
                    record["makespan_seconds"] for record in records
                )
                * 1000,
                "request_p95_mean_ms": statistics.mean(
                    record["post_tool_p95_seconds"] for record in records
                )
                * 1000,
                "relocated_fraction": statistics.mean(
                    record["relocated_fraction"] for record in records
                ),
                "ownership_relocated_fraction": statistics.mean(
                    record["ownership_relocated_fraction"] for record in records
                ),
                "full_prefix_hit_rate": statistics.mean(
                    record["full_prefix_hit_rate"] for record in records
                ),
                "reprefilled_tokens_mean": statistics.mean(
                    record["historical_reprefilled_tokens"] for record in records
                ),
                "queue_relief_after_return_mean_ms": statistics.mean(
                    record["queue_relief_after_first_return_seconds"]
                    for record in records
                )
                * 1000,
                "hbm_relief_mean_ms": statistics.mean(
                    record["source_hbm_relief_seconds"] for record in records
                )
                * 1000,
                "speedup_vs_sticky": sticky / mean_makespan if sticky else None,
                "speedup_vs_reroute": reroute / mean_makespan if reroute else None,
                "speedup_vs_on_return": (
                    on_return / mean_makespan if on_return else None
                ),
            }
        )
    order = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    rows.sort(
        key=lambda row: (
            row["prefix_length"],
            row["agent_count"],
            row["return_pattern"],
            order.get(row["scenario"], len(order)),
        )
    )
    prefix = Path(args.output_prefix) if args.output_prefix else source.with_suffix("")
    json_path = Path(f"{prefix}-summary.json")
    csv_path = Path(f"{prefix}-summary.csv")
    markdown_path = Path(f"{prefix}-summary.md")
    json_path.write_text(
        json.dumps(
            {"source": str(source), "config": payload["config"], "rows": rows},
            indent=2,
            sort_keys=True,
        )
    )
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Correlated-Return Hotspot Summary",
        "",
        "| Prefix | Agents | Pattern | Strategy | Makespan | Request p95 | Relocated | Owner moved | Full hit | Re-prefill |",
        "| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {prefix_length} | {agent_count} | {return_pattern} | {scenario} | "
            "{makespan_mean_ms:.1f} ms | {request_p95_mean_ms:.1f} ms | "
            "{relocated_fraction:.0%} | {ownership_relocated_fraction:.0%} | "
            "{full_prefix_hit_rate:.0%} | {reprefilled_tokens_mean:.0f} |".format(
                **row
            )
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
