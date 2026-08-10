#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agentshift.workloads.traces import iter_flowprefill_turns


@dataclass(frozen=True, slots=True)
class SourceRequest:
    offset_seconds: float
    source: str
    context_tokens: int
    output_tokens: int


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0.0),
    }


def read_csv_window(
    path: Path, source: str, window_seconds: float
) -> list[SourceRequest]:
    requests: list[SourceRequest] = []
    first: datetime | None = None
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = datetime.fromisoformat(row["TIMESTAMP"])
            if first is None:
                first = timestamp
            offset = (timestamp - first).total_seconds()
            if offset > window_seconds:
                break
            requests.append(
                SourceRequest(
                    offset,
                    source,
                    int(row["ContextTokens"]),
                    int(row["GeneratedTokens"]),
                )
            )
    return requests


def read_flowprefill_window(
    path: Path, window_seconds: float
) -> tuple[list[SourceRequest], list[float]]:
    requests: list[SourceRequest] = []
    gaps: list[float] = []
    for turn in iter_flowprefill_turns(path):
        if turn.tool_gap_seconds > 0:
            gaps.append(turn.tool_gap_seconds)
        if turn.arrival_seconds <= window_seconds:
            requests.append(
                SourceRequest(
                    turn.arrival_seconds,
                    "flowprefill",
                    turn.cumulative_prefix_tokens,
                    turn.output_tokens,
                )
            )
    return requests, gaps


def thin(
    requests: list[SourceRequest], target: int, rng: random.Random
) -> list[SourceRequest]:
    if target <= 0:
        return []
    if target >= len(requests):
        return requests
    indexes = sorted(rng.sample(range(len(requests)), target))
    return [requests[index] for index in indexes]


def main(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    azure_code = read_csv_window(
        args.azure_code, "azure-code", args.source_window_seconds
    )
    azure_conversation = read_csv_window(
        args.azure_conversation,
        "azure-conversation",
        args.source_window_seconds,
    )
    kimi = read_csv_window(args.kimi, "kimi-k2.5", args.source_window_seconds)
    flow, flow_gaps = read_flowprefill_window(
        args.flowprefill, args.source_window_seconds
    )
    if not flow_gaps:
        raise RuntimeError("FlowPrefill trace has no inter-turn gaps")

    azure_code_target = args.phase_counts[0] // 2
    azure_conversation_target = args.phase_counts[0] - azure_code_target
    phases = [
        thin(azure_code, azure_code_target, rng)
        + thin(azure_conversation, azure_conversation_target, rng),
        thin(kimi, args.phase_counts[1], rng),
        thin(flow, args.phase_counts[2], rng),
    ]
    events = []
    clipped_contexts = 0
    clipped_outputs = 0
    original_contexts: list[float] = []
    effective_contexts: list[float] = []
    effective_gaps: list[float] = []
    for phase_index, phase in enumerate(phases):
        for request in sorted(phase, key=lambda item: item.offset_seconds):
            context = min(
                args.max_context_tokens,
                max(args.min_context_tokens, request.context_tokens),
            )
            output = min(
                args.max_first_output_tokens,
                max(args.min_first_output_tokens, request.output_tokens),
            )
            raw_gap = flow_gaps[rng.randrange(len(flow_gaps))]
            gap = min(
                args.max_tool_gap_seconds,
                max(args.min_tool_gap_seconds, raw_gap * args.tool_gap_scale),
            )
            clipped_contexts += context != request.context_tokens
            clipped_outputs += output != request.output_tokens
            original_contexts.append(request.context_tokens)
            effective_contexts.append(context)
            effective_gaps.append(gap)
            events.append(
                {
                    "event_id": len(events),
                    "arrival_seconds": (
                        phase_index * args.phase_seconds
                        + request.offset_seconds
                        * args.phase_seconds
                        / args.source_window_seconds
                    ),
                    "phase": phase_index,
                    "source": request.source,
                    "original_context_tokens": request.context_tokens,
                    "context_tokens": context,
                    "original_output_tokens": request.output_tokens,
                    "first_output_tokens": output,
                    "second_output_tokens": args.second_output_tokens,
                    "tool_result_tokens": args.tool_result_tokens,
                    "tool_gap_seconds": gap,
                }
            )

    output = {
        "schema_version": 1,
        "generated_at_unix": time.time(),
        "methodology": (
            "Deterministic thinning of each trace's first source window. Azure and "
            "Kimi provide request arrivals and lengths but no sessions or tool labels; "
            "their two-turn gaps are sampled from FlowPrefill inter-turn deltas and "
            "time-compressed as configured."
        ),
        "config": {
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "duration_seconds": args.phase_seconds * 3,
        },
        "source_window_counts": {
            "azure_code": len(azure_code),
            "azure_conversation": len(azure_conversation),
            "kimi_k2.5": len(kimi),
            "flowprefill": len(flow),
        },
        "phase_event_counts": [len(phase) for phase in phases],
        "distortion": {
            "context_clipped_fraction": clipped_contexts / max(1, len(events)),
            "output_clipped_fraction": clipped_outputs / max(1, len(events)),
            "original_context_tokens": distribution(original_contexts),
            "effective_context_tokens": distribution(effective_contexts),
            "effective_tool_gap_seconds": distribution(effective_gaps),
        },
        "events": events,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = args.name or str(time.time_ns())
    output_path = args.output_dir / f"autoscaling-manifest-{suffix}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "events": len(events),
                "phase_event_counts": output["phase_event_counts"],
                "distortion": output["distortion"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--azure-code",
        type=Path,
        default=Path(
            "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_code_1week.csv"
        ),
    )
    parser.add_argument(
        "--azure-conversation",
        type=Path,
        default=Path(
            "/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv"
        ),
    )
    parser.add_argument(
        "--kimi",
        type=Path,
        default=Path("/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv"),
    )
    parser.add_argument(
        "--flowprefill",
        type=Path,
        default=Path(
            "/home/zhujianian/graphpool/third_party/"
            "FlowPrefill_trace_build/qwen_traceA_blksz_16.jsonl"
        ),
    )
    parser.add_argument("--phase-seconds", type=float, default=600.0)
    parser.add_argument("--source-window-seconds", type=float, default=600.0)
    parser.add_argument("--phase-counts", type=int, nargs=3, default=[6000, 600, 3000])
    parser.add_argument("--min-context-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=16384)
    parser.add_argument("--min-first-output-tokens", type=int, default=16)
    parser.add_argument("--max-first-output-tokens", type=int, default=128)
    parser.add_argument("--second-output-tokens", type=int, default=32)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--tool-gap-scale", type=float, default=0.02)
    parser.add_argument("--min-tool-gap-seconds", type=float, default=0.5)
    parser.add_argument("--max-tool-gap-seconds", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--name")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/autoscaling/manifests")
    )
    main(parser.parse_args())
