from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


SCENARIO_ORDER = ["sticky", "ttl", "ttl-reroute", "agentshift"]


def main(args) -> None:
    source = Path(args.input)
    payload = json.loads(source.read_text())
    grouped = defaultdict(list)
    for record in payload["records"]:
        grouped[(float(record["gap_seconds"]), record["scenario"])].append(record)

    mean_makespans = {
        key: statistics.mean(row["agent_makespan_seconds"] for row in records)
        for key, records in grouped.items()
    }
    rows = []
    for (gap_seconds, scenario), records in grouped.items():
        agent_makespan = mean_makespans[(gap_seconds, scenario)]
        rows.append(
            {
                "gap_ms": gap_seconds * 1000,
                "scenario": scenario,
                "trials": len(records),
                "ttl_expired": any(row["ttl_expired"] for row in records),
                "agent_makespan_mean_ms": agent_makespan * 1000,
                "agent_makespan_max_ms": max(
                    row["agent_makespan_seconds"] for row in records
                )
                * 1000,
                "post_tool_mean_ms": statistics.mean(
                    row["post_tool_mean_seconds"] for row in records
                )
                * 1000,
                "post_tool_p95_mean_ms": statistics.mean(
                    row["post_tool_p95_seconds"] for row in records
                )
                * 1000,
                "pressure_makespan_mean_ms": statistics.mean(
                    row["pressure_makespan_seconds"] for row in records
                )
                * 1000,
                "full_prefix_hit_rate": statistics.mean(
                    row["full_prefix_hit_rate"] for row in records
                ),
                "destination_full_hit_rate": (
                    statistics.mean(
                        row["destination_full_hit_rate"]
                        for row in records
                        if row["destination_full_hit_rate"] is not None
                    )
                    if any(
                        row["destination_full_hit_rate"] is not None
                        for row in records
                    )
                    else None
                ),
                "relocated_fraction": statistics.mean(
                    row["relocated_fraction"] for row in records
                ),
                "ownership_relocated_fraction": statistics.mean(
                    row["ownership_relocated_fraction"] for row in records
                ),
                "speedup_vs_sticky": (
                    mean_makespans.get((gap_seconds, "sticky"), agent_makespan)
                    / agent_makespan
                ),
                "speedup_vs_ttl": (
                    mean_makespans.get((gap_seconds, "ttl"), agent_makespan)
                    / agent_makespan
                ),
            }
        )

    order = {name: index for index, name in enumerate(SCENARIO_ORDER)}
    rows.sort(key=lambda row: (row["gap_ms"], order.get(row["scenario"], 99)))
    prefix = Path(args.output_prefix) if args.output_prefix else source.with_suffix("")
    json_path = Path(f"{prefix}-summary.json")
    csv_path = Path(f"{prefix}-summary.csv")
    markdown_path = Path(f"{prefix}-summary.md")
    json_path.write_text(
        json.dumps(
            {
                "source": str(source),
                "config": payload["config"],
                "calibration": payload["calibration"],
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

    calibration = payload["calibration"]
    lines = [
        "# TTL Capacity-Pressure Summary",
        "",
        (
            f"Calibrated TTL: {calibration['ttl_seconds'] * 1000:.1f} ms; "
            f"held-out tool-gap coverage: {calibration['heldout_coverage']:.1%}."
        ),
        "",
        "| Gap | Strategy | TTL expired | Agent makespan | Post-tool mean | "
        "Pressure makespan | Relocated | Owner moved | Full hit | Destination hit |",
        "| ---: | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        destination_hit = row["destination_full_hit_rate"]
        lines.append(
            "| {gap_ms:.0f} ms | {scenario} | {expired} | "
            "{agent_makespan_mean_ms:.1f} ms | {post_tool_mean_ms:.1f} ms | "
            "{pressure_makespan_mean_ms:.1f} ms | {relocated_fraction:.0%} | "
            "{ownership_relocated_fraction:.0%} | {full_prefix_hit_rate:.0%} | "
            "{destination_hit} |".format(
                expired="yes" if row["ttl_expired"] else "no",
                destination_hit=(
                    f"{destination_hit:.0%}" if destination_hit is not None else "n/a"
                ),
                **row,
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
