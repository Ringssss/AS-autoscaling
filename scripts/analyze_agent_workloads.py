from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from agentshift.workloads.traces import iter_flowprefill_turns


QUANTILES = (0.5, 0.9, 0.95, 0.99)
WORKER_PATTERNS = {
    "input_tokens": re.compile(r"input_ids_len=(\d+)"),
    "completion_tokens": re.compile(r"'completion_tokens': (\d+)"),
    "cached_tokens": re.compile(r"'cached_tokens': (\d+)"),
    "e2e_seconds": re.compile(r"'e2e_latency': ([0-9.eE+-]+)"),
    "ttft_seconds": re.compile(r"'ttft': ([0-9.eE+-]+)"),
    "received_ts": re.compile(r"'request_received_ts': ([0-9.eE+-]+)"),
    "finished_ts": re.compile(r"'request_finished_ts': ([0-9.eE+-]+)"),
}


class ReservoirDistribution:
    def __init__(self, limit: int = 200_000, seed: int = 20260723):
        self.limit = limit
        self.rng = random.Random(seed)
        self.samples: list[float] = []
        self.count = 0
        self.total = 0.0
        self.maximum = 0.0

    def add(self, value: float) -> None:
        self.count += 1
        self.total += value
        self.maximum = max(self.maximum, value)
        if len(self.samples) < self.limit:
            self.samples.append(value)
            return
        replacement = self.rng.randrange(self.count)
        if replacement < self.limit:
            self.samples[replacement] = value

    def summary(self) -> dict[str, float]:
        result = {
            "count": self.count,
            "sample_count": len(self.samples),
            "mean": self.total / self.count if self.count else 0.0,
            "max": self.maximum if self.count else 0.0,
        }
        result.update(
            {
                f"p{int(quantile * 100)}": percentile(self.samples, quantile)
                for quantile in QUANTILES
            }
        )
        return result


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def distribution(values: Iterable[float]) -> dict[str, float]:
    materialized = list(values)
    result = {
        "count": len(materialized),
        "mean": statistics.fmean(materialized) if materialized else 0.0,
        "max": max(materialized) if materialized else 0.0,
    }
    result.update(
        {f"p{int(quantile * 100)}": percentile(materialized, quantile) for quantile in QUANTILES}
    )
    return result


def analyze_length_csv(path: Path, row_limit: int | None = None) -> dict:
    contexts = ReservoirDistribution(seed=20260723)
    outputs = ReservoirDistribution(seed=20260724)
    contexts_ge_16k = 0
    contexts_ge_32k = 0
    with path.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            context = float(row["ContextTokens"])
            contexts.add(context)
            outputs.add(float(row["GeneratedTokens"]))
            contexts_ge_16k += context >= 16384
            contexts_ge_32k += context >= 32768
            if row_limit is not None and index + 1 >= row_limit:
                break
    return {
        "path": str(path.resolve()),
        "schema_scope": "request timestamps and input/output lengths; no session or tool labels",
        "context_tokens": contexts.summary(),
        "output_tokens": outputs.summary(),
        "fraction_context_ge_16k": contexts_ge_16k / contexts.count,
        "fraction_context_ge_32k": contexts_ge_32k / contexts.count,
        "context_cdf_samples": [
            {"quantile": quantile, "tokens": percentile(contexts.samples, quantile)}
            for quantile in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
        ],
        "quantile_method": "deterministic reservoir sample; counts, means, maxima, and fractions are exact",
        "row_limit": row_limit,
    }


def analyze_worker_logs(directory: Path, limit: int | None) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
    parsed = 0
    files = sorted(path for path in directory.glob("*.redacted.log") if not path.name.startswith("._"))
    for path in files:
        with path.open(errors="replace") as handle:
            for line in handle:
                if not line.startswith("Finish:"):
                    continue
                row: dict[str, float] = {}
                for key, pattern in WORKER_PATTERNS.items():
                    match = pattern.search(line)
                    if match:
                        row[key] = float(match.group(1))
                if len(row) != len(WORKER_PATTERNS):
                    continue
                for key, value in row.items():
                    values[key].append(value)
                values["cache_hit_fraction"].append(
                    min(1.0, row["cached_tokens"] / max(1.0, row["input_tokens"]))
                )
                parsed += 1
                if limit is not None and parsed >= limit:
                    break
        if limit is not None and parsed >= limit:
            break
    return {
        "path": str(directory.resolve()),
        "schema_scope": (
            "completed inference requests with service timestamps and cache hits; "
            "conversation_id is redacted, so cross-turn sessions cannot be reconstructed"
        ),
        "files": len(files),
        "parsed_requests": parsed,
        "input_tokens": distribution(values["input_tokens"]),
        "completion_tokens": distribution(values["completion_tokens"]),
        "cached_tokens": distribution(values["cached_tokens"]),
        "cache_hit_fraction": distribution(values["cache_hit_fraction"]),
        "ttft_seconds": distribution(values["ttft_seconds"]),
        "e2e_seconds": distribution(values["e2e_seconds"]),
        "observed_span_seconds": (
            max(values["finished_ts"]) - min(values["received_ts"])
            if values["received_ts"]
            else 0.0
        ),
    }


def migration_curve(path: Path) -> tuple[list[int], list[float], float]:
    payload = json.loads(path.read_text())
    samples: dict[int, list[float]] = defaultdict(list)
    bytes_per_token: list[float] = []
    for row in payload["records"]:
        if row.get("scenario") != "agentshift":
            continue
        tokens = int(row.get("prefix_length", row.get("context_length", 0)))
        seconds = float(row.get("migration_wall_seconds", row.get("migration_seconds", 0.0)))
        moved_bytes = int(row.get("migration_bytes", row.get("bytes_transferred", 0)))
        if tokens > 0 and seconds > 0:
            samples[tokens].append(seconds)
        if tokens > 0 and moved_bytes > 0:
            bytes_per_token.append(moved_bytes / tokens)
    contexts = sorted(samples)
    return (
        contexts,
        [statistics.median(samples[context]) for context in contexts],
        statistics.median(bytes_per_token),
    )


def interpolate(tokens: int, contexts: list[int], seconds: list[float]) -> float:
    if tokens <= contexts[0]:
        return seconds[0] * tokens / contexts[0]
    if tokens >= contexts[-1]:
        return seconds[-1] * tokens / contexts[-1]
    for index in range(1, len(contexts)):
        if tokens <= contexts[index]:
            lower_tokens, upper_tokens = contexts[index - 1], contexts[index]
            ratio = (tokens - lower_tokens) / (upper_tokens - lower_tokens)
            return seconds[index - 1] + ratio * (seconds[index] - seconds[index - 1])
    raise AssertionError("unreachable")


def stable_engine(agent_id: str, engine_count: int) -> int:
    digest = hashlib.blake2b(agent_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % engine_count


def burst_distribution(
    arrivals: list[tuple[float, str]], window_seconds: float, engine_count: int
) -> dict:
    if not arrivals:
        return {"cluster": distribution([]), "max_per_engine": distribution([])}
    origin = min(timestamp for timestamp, _ in arrivals)
    cluster_buckets: dict[int, int] = defaultdict(int)
    engine_buckets: dict[tuple[int, int], int] = defaultdict(int)
    for timestamp, agent_id in arrivals:
        bucket = int((timestamp - origin) / window_seconds)
        cluster_buckets[bucket] += 1
        engine_buckets[(bucket, stable_engine(agent_id, engine_count))] += 1
    per_bucket_max: dict[int, int] = defaultdict(int)
    for (bucket, _), count in engine_buckets.items():
        per_bucket_max[bucket] = max(per_bucket_max[bucket], count)
    return {
        "cluster": distribution(cluster_buckets.values()),
        "max_per_engine": distribution(per_bucket_max.values()),
    }


def analyze_flowprefill(
    path: Path,
    contexts: list[int],
    migration_seconds: list[float],
    bytes_per_token: float,
    engine_count: int,
) -> dict:
    sessions: dict[str, list] = defaultdict(list)
    all_prefixes: list[float] = []
    gaps: list[float] = []
    child_arrivals: list[tuple[float, str]] = []
    hidden = 0
    for turn in iter_flowprefill_turns(path):
        sessions[turn.agent_id].append(turn)
        all_prefixes.append(turn.cumulative_prefix_tokens)
        if turn.turn > 1 or turn.tool_gap_seconds > 0:
            gaps.append(turn.tool_gap_seconds)
            child_arrivals.append((turn.arrival_seconds, turn.agent_id))
            hidden += turn.tool_gap_seconds >= interpolate(
                turn.cumulative_prefix_tokens, contexts, migration_seconds
            )

    prefixes_by_turn: dict[int, list[float]] = defaultdict(list)
    growth_by_turn: dict[int, list[float]] = defaultdict(list)
    session_turns = []
    for turns in sessions.values():
        ordered = sorted(turns, key=lambda item: (item.turn, item.arrival_seconds))
        session_turns.append(len(ordered))
        previous = 0
        for turn in ordered:
            prefixes_by_turn[turn.turn].append(turn.cumulative_prefix_tokens)
            growth_by_turn[turn.turn].append(turn.cumulative_prefix_tokens - previous)
            previous = turn.cumulative_prefix_tokens

    prefix_bins = ((4096, "le_4k"), (16384, "4k_16k"), (32768, "16k_32k"))
    coverage_rows = []
    for upper, label in prefix_bins:
        lower = 0 if not coverage_rows else prefix_bins[len(coverage_rows) - 1][0]
        rows = [
            turn
            for turns in sessions.values()
            for turn in turns
            if turn.tool_gap_seconds > 0 and lower < turn.cumulative_prefix_tokens <= upper
        ]
        coverage_rows.append(
            {
                "bin": label,
                "count": len(rows),
                "fully_hidden_fraction": (
                    sum(
                        turn.tool_gap_seconds
                        >= interpolate(turn.cumulative_prefix_tokens, contexts, migration_seconds)
                        for turn in rows
                    )
                    / len(rows)
                    if rows
                    else 0.0
                ),
            }
        )
    rows = [
        turn
        for turns in sessions.values()
        for turn in turns
        if turn.tool_gap_seconds > 0 and turn.cumulative_prefix_tokens > 32768
    ]
    coverage_rows.append(
        {
            "bin": "gt_32k",
            "count": len(rows),
            "fully_hidden_fraction": (
                sum(
                    turn.tool_gap_seconds
                    >= interpolate(turn.cumulative_prefix_tokens, contexts, migration_seconds)
                    for turn in rows
                )
                / len(rows)
                if rows
                else 0.0
            ),
        }
    )

    return {
        "path": str(path.resolve()),
        "schema_scope": (
            "parent-linked requests; inter-turn deltas and child arrivals are proxies "
            "for blocked windows and tool returns, not labeled tool events"
        ),
        "requests": sum(len(turns) for turns in sessions.values()),
        "sessions": len(sessions),
        "multi_turn_requests": len(gaps),
        "turns_per_session": distribution(session_turns),
        "cumulative_prefix_tokens": distribution(all_prefixes),
        "resident_kv_gib": distribution(
            tokens * bytes_per_token / (1024**3) for tokens in all_prefixes
        ),
        "inter_turn_proxy_seconds": distribution(gaps),
        "estimated_fully_hidden_fraction": hidden / len(gaps) if gaps else 0.0,
        "coverage_by_prefix": coverage_rows,
        "prefix_by_turn": [
            {
                "turn": turn,
                "count": len(prefixes_by_turn[turn]),
                "p50": percentile(prefixes_by_turn[turn], 0.5),
                "p90": percentile(prefixes_by_turn[turn], 0.9),
                "growth_p50": percentile(growth_by_turn[turn], 0.5),
            }
            for turn in sorted(prefixes_by_turn)
            if turn <= 20
        ],
        "return_bursts": {
            str(int(window * 1000)): burst_distribution(child_arrivals, window, engine_count)
            for window in (0.01, 0.05, 0.1, 0.5)
        },
    }


def main(args: argparse.Namespace) -> None:
    contexts, seconds, bytes_per_token = migration_curve(Path(args.migration_benchmark))
    length_traces = {"kimi_k25": analyze_length_csv(Path(args.kimi_csv))}
    if args.include_azure:
        length_traces.update(
            {
                "azure_conversation": analyze_length_csv(
                    Path(args.azure_conversation_csv), args.azure_row_limit
                ),
                "azure_code": analyze_length_csv(
                    Path(args.azure_code_csv), args.azure_row_limit
                ),
            }
        )
    output = {
        "generated_at_unix": time.time(),
        "migration_calibration": {
            "source": str(Path(args.migration_benchmark).resolve()),
            "tokens": contexts,
            "seconds": seconds,
            "bytes_per_token": bytes_per_token,
        },
        "length_traces": length_traces,
        "kimi_worker": analyze_worker_logs(Path(args.kimi_worker_dir), args.worker_limit),
        "flowprefill": analyze_flowprefill(
            Path(args.flowprefill), contexts, seconds, bytes_per_token, args.engine_count
        ),
    }
    output_path = Path(args.output_dir) / f"agent-workloads-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path), **output}, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kimi-csv",
        default="/mnt/models/kimik25/kimi-k25-trace/kimi_k25_conv_1day.csv",
    )
    parser.add_argument(
        "--azure-conversation-csv",
        default="/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_conv_1week.csv",
    )
    parser.add_argument(
        "--azure-code-csv",
        default="/mnt/models/AzureLLMInferenceTrace/AzureLLMInferenceTrace_code_1week.csv",
    )
    parser.add_argument(
        "--kimi-worker-dir",
        default=(
            "/mnt/models/kimik25/"
            "kimi-k25-workers-5p3d-all-decodes_finish_fullids_redacted_2026-04-07"
        ),
    )
    parser.add_argument("--worker-limit", type=int)
    parser.add_argument("--include-azure", action="store_true")
    parser.add_argument("--azure-row-limit", type=int, default=2_000_000)
    parser.add_argument(
        "--flowprefill",
        default=(
            "/home/zhujianian/graphpool/third_party/"
            "FlowPrefill_trace_build/qwen_traceA_blksz_16.jsonl"
        ),
    )
    parser.add_argument(
        "--migration-benchmark",
        default="results/blocked-window-1784565792789187724.json",
    )
    parser.add_argument("--engine-count", type=int, default=8)
    parser.add_argument("--output-dir", default="results")
    main(parser.parse_args())
