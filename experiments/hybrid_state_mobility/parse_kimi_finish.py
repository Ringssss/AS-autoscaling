#!/usr/bin/env python3
"""Extract lightweight fields from redacted Kimi Finish logs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


PATTERNS = {
    "rid": re.compile(r"rid='([^']+)'"),
    "input_len": re.compile(r"Finish: input_ids_len=(\d+)"),
    "completion_tokens": re.compile(r"'completion_tokens': (\d+)"),
    "cached_tokens": re.compile(r"'cached_tokens': (\d+)"),
    "e2e_latency": re.compile(r"'e2e_latency': ([0-9.eE+-]+)"),
    "ttft": re.compile(r"'ttft': ([0-9.eE+-]+)"),
    "request_received_ts": re.compile(r"'request_received_ts': ([0-9.eE+-]+)"),
    "request_finished_ts": re.compile(r"'request_finished_ts': ([0-9.eE+-]+)"),
    "dp_rank": re.compile(r"'dp_rank': (-?\d+)"),
    "bootstrap_host": re.compile(r"bootstrap_host='([^']+)'"),
    "conversation_id": re.compile(r"conversation_id='([^']+)'"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    values: dict[str, list] = {name: [] for name in PATTERNS}
    source_hour: list[np.int8] = []
    malformed = 0
    files = sorted(args.trace_root.glob("[0-2][0-9].redacted.log"))
    if not files:
        raise FileNotFoundError(f"no hourly redacted logs under {args.trace_root}")

    for path in files:
        hour = int(path.name.split(".", maxsplit=1)[0])
        with path.open("r", errors="replace") as handle:
            for line in handle:
                if not line.startswith("Finish:"):
                    continue
                matched = 0
                for name, pattern in PATTERNS.items():
                    match = pattern.search(line)
                    if match is None:
                        if name == "conversation_id":
                            values[name].append("")
                        elif name in {"rid", "bootstrap_host"}:
                            values[name].append("")
                        else:
                            values[name].append(-1)
                        continue
                    matched += 1
                    raw = match.group(1)
                    if name in {
                        "input_len",
                        "completion_tokens",
                        "cached_tokens",
                        "dp_rank",
                    }:
                        values[name].append(int(raw))
                    elif name in {
                        "e2e_latency",
                        "ttft",
                        "request_received_ts",
                        "request_finished_ts",
                    }:
                        values[name].append(float(raw))
                    else:
                        values[name].append(raw)
                source_hour.append(hour)
                if matched < len(PATTERNS) - 1:
                    malformed += 1

    arrays = {}
    for name, raw in values.items():
        if name in {"rid", "bootstrap_host", "conversation_id"}:
            arrays[name] = np.asarray(raw, dtype=np.str_)
        else:
            arrays[name] = np.asarray(raw, dtype=np.float64)
    arrays["source_hour"] = np.asarray(source_hour, dtype=np.int8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)

    input_len = arrays["input_len"]
    cached = arrays["cached_tokens"]
    received = arrays["request_received_ts"]
    finished = arrays["request_finished_ts"]
    summary = {
        "rows": int(input_len.size),
        "malformed": malformed,
        "hours": int(arrays["source_hour"].max() + 1),
        "input_len_p50_p90_p99_max": np.percentile(input_len, [50, 90, 99]).tolist()
        + [int(input_len.max())],
        "cached_p50_p90_p99_max": np.percentile(cached, [50, 90, 99]).tolist()
        + [int(cached.max())],
        "completion_p50_p90_p99_max": np.percentile(
            arrays["completion_tokens"], [50, 90, 99]
        ).tolist()
        + [int(arrays["completion_tokens"].max())],
        "span_seconds": [float(received.min()), float(received.max())],
        "finished_span_seconds": [float(finished.min()), float(finished.max())],
        "unique_rids": int(np.unique(arrays["rid"]).size),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
