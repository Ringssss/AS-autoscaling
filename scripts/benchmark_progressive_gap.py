#!/usr/bin/env python3
"""Measure standard and progressive AgentShift across controlled tool waits.

This is an additive benchmark driver.  It reuses the production benchmark's
request preparation and migration paths while replacing external tools with a
controlled sleep command whose stdout size is fixed across delays.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

from benchmark_real_tools_all import RealToolAllBaselineBenchmark


def controlled_tool(python: str, wait_ms: float) -> tuple[str, list[str]]:
    name = f"wait-{wait_ms:g}ms"
    code = f"import time; time.sleep({wait_ms / 1000.0!r}); print('x' * 64)"
    return name, [python, "-c", code]


async def main(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    benchmark = RealToolAllBaselineBenchmark(args)
    benchmark.tools = [
        controlled_tool(args.python, wait_ms) for wait_ms in args.wait_ms
    ]
    tool_measurements = await benchmark.measure_tools()

    records = []
    warmups = []
    for prefix_length in args.prefix_lengths:
        calibration = await benchmark.calibrate_prefix(prefix_length)
        print(json.dumps({"calibration": calibration}, sort_keys=True), flush=True)

        warmup_name, warmup_command = benchmark.tools[0]
        for scenario in ("agentshift", "progressive"):
            warmup = await benchmark.run_one(
                scenario,
                warmup_name,
                warmup_command,
                prefix_length,
                -1,
            )
            warmups.append(warmup)

        for repeat in range(args.repeats):
            for tool_name, command in benchmark.tools:
                scenario_order = ["agentshift", "progressive"]
                random.Random(
                    args.seed
                    + prefix_length * 1009
                    + repeat * 17
                    + round(benchmark.tool_predictions[tool_name] * 1e6)
                ).shuffle(scenario_order)
                for scenario in scenario_order:
                    record = await benchmark.run_one(
                        scenario,
                        tool_name,
                        command,
                        prefix_length,
                        repeat,
                    )
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)

    output = {
        "baseline_implementation": "unchanged AgentShift migration",
        "progressive_implementation": "progressive private continuation",
        "config": vars(args),
        "tool_measurements": tool_measurements,
        "calibrations": benchmark.calibrations,
        "warmups": warmups,
        "state_db": str(benchmark.state_path),
        "records": records,
    }
    output_path = Path(args.output_dir) / f"progressive-gap-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="http://127.0.0.1:32200")
    parser.add_argument("--destination", default="http://127.0.0.1:32201")
    parser.add_argument(
        "--wait-ms",
        type=float,
        nargs="+",
        default=[0, 10, 25, 50, 75, 100, 150, 250, 400, 600],
    )
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[32768])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--tool-calibration-repeats", type=int, default=3)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--max-tool-result-tokens", type=int, default=1024)
    parser.add_argument("--ttl-hbm-cost-per-gib-second", type=float, default=0.25)
    parser.add_argument("--kv-bytes-per-token", type=int, default=147456)
    parser.add_argument("--agentix-source-queue-ms", type=float, default=0.0)
    parser.add_argument("--agentix-destination-queue-ms", type=float, default=0.0)
    parser.add_argument("--poll-interval", type=float, default=0.005)
    parser.add_argument("--progressive-layer-group-size", type=int, default=4)
    parser.add_argument("--transfer-port", type=int, default=32300)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--tool-cwd", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default="results/progressive-gap")
    parser.set_defaults(tools=())
    asyncio.run(main(parser.parse_args()))
