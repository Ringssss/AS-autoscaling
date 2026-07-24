from __future__ import annotations

import argparse
import json
import time
from bisect import bisect_left
from pathlib import Path

from agentshift.workloads.traces import iter_flowprefill_turns, iter_kimi_requests


PERCENTILES = (0.5, 0.9, 0.95, 0.99)


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, max(0, int(quantile * len(values)) - 1))
    return values[index]


def distribution(values: list[float]) -> dict[str, float]:
    result = {"count": len(values), "mean": sum(values) / len(values) if values else 0}
    result.update({f"p{int(q * 100)}": percentile(values, q) for q in PERCENTILES})
    result["max"] = max(values) if values else 0
    return result


def measured_migration_curve(path: str | Path) -> tuple[list[int], list[float]]:
    benchmark = json.loads(Path(path).read_text())
    grouped: dict[int, list[float]] = {}
    for record in benchmark["records"]:
        if record["kind"] == "single" and record["scenario"] == "agentshift":
            grouped.setdefault(record["context_length"], []).append(
                record["migration_seconds"]
            )
    contexts = sorted(grouped)
    seconds = [sum(grouped[item]) / len(grouped[item]) for item in contexts]
    return contexts, seconds


def interpolate_migration(tokens: int, contexts: list[int], seconds: list[float]) -> float:
    if tokens <= contexts[0]:
        return seconds[0]
    if tokens >= contexts[-1]:
        return seconds[-1] * tokens / contexts[-1]
    upper = bisect_left(contexts, tokens)
    lower = upper - 1
    ratio = (tokens - contexts[lower]) / (contexts[upper] - contexts[lower])
    return seconds[lower] + ratio * (seconds[upper] - seconds[lower])


def analyze_kimi(path: str | Path) -> dict:
    contexts: list[float] = []
    outputs: list[float] = []
    for _, context, output in iter_kimi_requests(path):
        contexts.append(context)
        outputs.append(output)
    return {
        "schema_scope": "request timestamp and input/output lengths; no session or tool labels",
        "context_tokens": distribution(contexts),
        "output_tokens": distribution(outputs),
        "fraction_context_le_32768": sum(item <= 32768 for item in contexts)
        / len(contexts),
        "fraction_context_le_40960": sum(item <= 40960 for item in contexts)
        / len(contexts),
    }


def analyze_flowprefill(
    path: str | Path, contexts: list[int], seconds: list[float]
) -> dict:
    prefixes: list[float] = []
    incremental_inputs: list[float] = []
    outputs: list[float] = []
    gaps: list[float] = []
    estimated_migrations: list[float] = []
    roots: set[str] = set()
    turns = 0
    multi_turns = 0
    hidden = 0
    for turn in iter_flowprefill_turns(path):
        turns += 1
        roots.add(turn.agent_id)
        prefixes.append(turn.cumulative_prefix_tokens)
        incremental_inputs.append(turn.incremental_input_tokens)
        outputs.append(turn.output_tokens)
        if turn.turn > 1 or turn.tool_gap_seconds > 0:
            multi_turns += 1
            gaps.append(turn.tool_gap_seconds)
            estimate = interpolate_migration(
                turn.cumulative_prefix_tokens, contexts, seconds
            )
            estimated_migrations.append(estimate)
            hidden += turn.tool_gap_seconds >= estimate
    return {
        "schema_scope": (
            "parent-linked multi-turn requests; inter-turn delta is a blocked-window "
            "proxy and is not a labeled tool duration"
        ),
        "requests": turns,
        "root_sessions": len(roots),
        "multi_turn_requests": multi_turns,
        "cumulative_prefix_tokens": distribution(prefixes),
        "incremental_input_tokens": distribution(incremental_inputs),
        "output_tokens": distribution(outputs),
        "inter_turn_proxy_seconds": distribution(gaps),
        "estimated_migration_seconds": distribution(estimated_migrations),
        "estimated_fully_hidden_fraction": hidden / multi_turns if multi_turns else 0,
        "fraction_prefix_le_40960": sum(item <= 40960 for item in prefixes)
        / len(prefixes),
    }


def main(args) -> None:
    contexts, seconds = measured_migration_curve(args.benchmark)
    result = {
        "benchmark": str(Path(args.benchmark).resolve()),
        "measured_migration_curve": dict(zip(map(str, contexts), seconds)),
        "kimi": analyze_kimi(args.kimi),
        "flowprefill": analyze_flowprefill(args.flowprefill, contexts, seconds),
    }
    output = Path(args.output_dir) / f"trace-analysis-{time.time_ns()}.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output), **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kimi",
        default="/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv",
    )
    parser.add_argument(
        "--flowprefill",
        default=(
            "/home/zhujianian/graphpool/third_party/"
            "FlowPrefill_trace_build/qwen_traceA_blksz_16.jsonl"
        ),
    )
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output-dir", default="results")
    main(parser.parse_args())
