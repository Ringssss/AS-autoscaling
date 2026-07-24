from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


MODE_ORDER = ["no-migration", "sync", "async"]


def mean_present(records, field):
    values = [row[field] for row in records if row.get(field) is not None]
    return statistics.mean(values) if values else None


def main(args) -> None:
    source = Path(args.input)
    payload = json.loads(source.read_text())
    grouped = defaultdict(list)
    for record in payload["records"]:
        grouped[
            (
                int(record["prefix_length"]),
                int(record["concurrency"]),
                record["mode"],
            )
        ].append(record)

    baselines = {
        (prefix_length, concurrency): mean_present(records, "foreground_throughput_tokens_per_second")
        for (prefix_length, concurrency, mode), records in grouped.items()
        if mode == "no-migration"
    }
    rows = []
    for (prefix_length, concurrency, mode), records in grouped.items():
        throughput = mean_present(records, "foreground_throughput_tokens_per_second")
        baseline = baselines.get((prefix_length, concurrency))
        rows.append(
            {
                "prefix_length": prefix_length,
                "concurrency": concurrency,
                "mode": mode,
                "trials": len(records),
                "throughput_tokens_per_second": throughput,
                "throughput_change_fraction": (
                    throughput / baseline - 1 if throughput is not None and baseline else None
                ),
                "ttft_p95_mean_ms": mean_present(records, "ttft_p95_seconds") * 1000,
                "arrival_probe_ttft_mean_ms": mean_present(
                    records, "arrival_probe_ttft_seconds"
                )
                * 1000,
                "tpot_mean_ms": mean_present(records, "tpot_mean_seconds") * 1000,
                "tpot_p95_mean_ms": mean_present(records, "tpot_p95_seconds") * 1000,
                "tpot_p99_mean_ms": mean_present(records, "tpot_p99_seconds") * 1000,
                "max_token_gap_mean_ms": mean_present(records, "max_token_gap_seconds") * 1000,
                "migration_wall_mean_ms": mean_present(records, "migration_wall_seconds") * 1000,
                "migration_worker_mean_ms": mean_present(records, "migration_worker_seconds") * 1000,
                "migration_queue_mean_ms": mean_present(records, "migration_queue_seconds") * 1000,
            }
        )

    order = {name: index for index, name in enumerate(MODE_ORDER)}
    rows.sort(
        key=lambda row: (
            row["prefix_length"],
            row["concurrency"],
            order.get(row["mode"], 99),
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
        "# Foreground Interference Summary",
        "",
        "| Prefix | Concurrency | Mode | Throughput | Change | Arrival TTFT | "
        "TPOT p95 | TPOT p99 | Max token gap | Migration wall |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {prefix_length} | {concurrency} | {mode} | "
            "{throughput_tokens_per_second:.1f} tok/s | {change:+.1%} | "
            "{arrival_probe_ttft_mean_ms:.1f} ms | {tpot_p95_mean_ms:.2f} ms | "
            "{tpot_p99_mean_ms:.2f} ms | {max_token_gap_mean_ms:.2f} ms | "
            "{migration_wall_mean_ms:.1f} ms |".format(
                change=row["throughput_change_fraction"] or 0.0,
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
