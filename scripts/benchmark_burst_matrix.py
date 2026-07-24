from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from benchmark_e2e import Benchmark


async def main(args) -> None:
    benchmark = Benchmark(args)
    records = []
    for context_length in args.burst_context_lengths:
        for repeat in range(args.burst_repeats):
            for scenario in ("sticky", "reroute", "agentshift"):
                record = await benchmark.run_burst(
                    scenario, context_length, args.burst_concurrency, repeat
                )
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    summary = {}
    for context_length in args.burst_context_lengths:
        summary[str(context_length)] = {}
        for scenario in ("sticky", "reroute", "agentshift"):
            rows = [
                row
                for row in records
                if row["context_length"] == context_length
                and row["scenario"] == scenario
            ]
            summary[str(context_length)][scenario] = {
                "makespan_mean_seconds": statistics.mean(
                    row["makespan_seconds"] for row in rows
                ),
                "post_tool_mean_seconds": statistics.mean(
                    row["post_tool_makespan_seconds"] for row in rows
                ),
                "p95_mean_seconds": statistics.mean(
                    row["p95_request_seconds"] for row in rows
                ),
                "reprefilled_tokens_mean": statistics.mean(
                    row["historical_reprefilled_tokens"] for row in rows
                ),
            }
        ours = summary[str(context_length)]["agentshift"][
            "post_tool_mean_seconds"
        ]
        summary[str(context_length)]["speedup"] = {
            scenario: summary[str(context_length)][scenario][
                "post_tool_mean_seconds"
            ]
            / ours
            for scenario in ("sticky", "reroute")
        }

    output = {
        "config": vars(args),
        "state_db": str(benchmark.state_path),
        "records": records,
        "summary": summary,
    }
    output_path = Path(args.output_dir) / f"burst-matrix-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--burst-context-lengths",
        type=int,
        nargs="+",
        default=[2048, 8192, 16384, 32000],
    )
    parser.add_argument("--burst-concurrency", type=int, default=8)
    parser.add_argument("--burst-output-tokens", type=int, default=64)
    parser.add_argument("--burst-repeats", type=int, default=3)
    parser.add_argument("--background-concurrency", type=int, default=4)
    parser.add_argument("--background-prompt-tokens", type=int, default=128)
    parser.add_argument("--background-output-tokens", type=int, default=256)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--tool-gap-seconds", type=float, default=0.25)
    parser.add_argument("--transfer-port", type=int, default=29990)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--sync-transfer", action="store_true")
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
