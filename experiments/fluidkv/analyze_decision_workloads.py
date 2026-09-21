#!/usr/bin/env python3
"""Audit unfiltered trace populations and predeclared continuous replay windows.

All Qwen timing is arrival timing, never measured tool latency. Block hashes
identify anonymous input content; any generated token replay remains surrogate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np


QWEN_FILES = {"A": "qwen_traceA_blksz_16.jsonl", "Coder": "qwen_coder_blksz_16.jsonl",
              "Thinking": "qwen_thinking_blksz_16.jsonl"}
DEFAULT_STARTS = (600, 3000, 5400)
DEFAULT_DURATIONS = (60, 120, 300)


def distribution(values):
    values = list(values)
    if not values:
        return {"count": 0, "min": None, "p50": None, "p90": None, "p99": None,
                "max": None, "mean": None, "sum": 0}
    numbers = np.asarray(values, dtype=np.float64)
    p50, p90, p99 = np.percentile(numbers, [50, 90, 99])
    return {"count": len(values), "min": float(numbers.min()), "p50": float(p50),
            "p90": float(p90), "p99": float(p99), "max": float(numbers.max()),
            "mean": float(numbers.mean()), "sum": float(numbers.sum())}


def input_identity_sha256(row):
    """Hash anonymous input identity, not model-generated surrogate token IDs."""
    payload = json.dumps([row["input_length"], row["hash_ids"]], separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def parent_overlap(parent, child):
    matched = 0
    for left, right in zip(parent["hash_ids"], child["hash_ids"]):
        if left != right:
            break
        matched += 1
    overlap = min(16 * matched, parent["input_length"], child["input_length"])
    aligned = parent["input_length"] % 16 == 0
    exact = aligned and overlap == parent["input_length"]
    return {"overlap_tokens": overlap, "parent_partial_block": not aligned,
            "exact_block_aligned_parent_match": exact,
            "snapshot_eligible": exact and child["input_length"] > parent["input_length"]}


def validated_index(rows):
    by_id = {}
    for row in rows:
        chat = row["chat_id"]
        if chat in by_id:
            raise ValueError(f"duplicate chat ID {chat}")
        if row["input_length"] <= 0 or len(row["hash_ids"]) != math.ceil(row["input_length"] / 16):
            raise ValueError(f"invalid hash block count for {chat}")
        if row["output_length"] < 0 or not math.isfinite(row["timestamp"]):
            raise ValueError(f"invalid length or timestamp for {chat}")
        by_id[chat] = row
    for child in rows:
        parent = by_id.get(child["parent_chat_id"])
        if parent is not None and parent["timestamp"] >= child["timestamp"]:
            raise ValueError(f"non-increasing parent timestamp for {child['chat_id']}")
    return by_id


def analyze_qwen_rows(rows):
    by_id = validated_index(rows)
    ordered = sorted(rows, key=lambda r: (r["timestamp"], r["chat_id"]))
    rank = {row["chat_id"]: index for index, row in enumerate(ordered)}
    counts = Counter()
    gaps, distances, overlaps, eligible_lengths = [], [], [], []
    for child in ordered:
        if child["parent_chat_id"] < 0:
            counts["root_requests"] += 1
            continue
        parent = by_id.get(child["parent_chat_id"])
        if parent is None:
            counts["missing_parents"] += 1
            continue
        counts["linked_requests"] += 1
        overlap = parent_overlap(parent, child)
        overlaps.append(overlap["overlap_tokens"])
        gaps.append(child["timestamp"] - parent["timestamp"])
        distances.append(rank[child["chat_id"]] - rank[parent["chat_id"]] - 1)
        counts["parent_input_overlap_tokens"] += overlap["overlap_tokens"]
        counts["linked_child_input_tokens"] += child["input_length"]
        counts["partial_parent_boundaries"] += overlap["parent_partial_block"]
        counts["exact_block_aligned_parent_matches"] += overlap["exact_block_aligned_parent_match"]
        counts["snapshot_eligible_children"] += overlap["snapshot_eligible"]
        if overlap["snapshot_eligible"]:
            eligible_lengths.append(parent["input_length"])
    result = {key: counts[key] for key in (
        "root_requests", "missing_parents", "linked_requests", "parent_input_overlap_tokens",
        "linked_child_input_tokens", "partial_parent_boundaries", "exact_block_aligned_parent_matches",
        "snapshot_eligible_children")}
    span = ordered[-1]["timestamp"] - ordered[0]["timestamp"] if ordered else 0
    result.update({"requests": len(rows), "span_seconds": span,
                   "observed_requests_per_second": len(rows) / span if span else None,
                   "types": dict(Counter(r.get("type", "unknown") for r in rows)),
                   "field_presence": dict(Counter(key for row in rows for key in row)),
                   "input_lengths": distribution(r["input_length"] for r in rows),
                   "output_lengths": distribution(r["output_length"] for r in rows),
                   "parent_overlap_lengths": distribution(overlaps),
                   "eligible_parent_lengths": distribution(eligible_lengths),
                   "arrival_deltas_seconds": distribution(gaps),
                   "intervening_requests": distribution(distances),
                   "snapshot_eligibility_definition": "Observed parent input fully matches, parent boundary is block-aligned, child extends parent; requires an actual retained engine snapshot to hit.",
                   "timing_semantics": "request arrivals; no completion or tool durations"})
    if ordered:
        origin = ordered[0]["timestamp"]
        bins = Counter(int((r["timestamp"] - origin) // 60) for r in ordered)
        result["arrival_bins_60_seconds"] = [
            {"start_seconds": index * 60, "requests": bins[index],
             "observed_seconds": min(60, max(0, span - index * 60))}
            for index in range(int(span // 60) + 1)]
    return result


def select_continuous_window(rows, *, start_seconds, duration_seconds,
                             warmup_seconds=None, context_length=32768, time_origin=None):
    """Return original rows plus audit for a half-open continuous interval.

    No request cap, ancestor injection, output truncation, or timestamp rewriting
    is performed. Offsets default to the earliest source arrival. Warmup defaults
    to the measurement duration. Excluded context-incompatible rows are listed.
    """
    by_id = validated_index(rows)
    if duration_seconds <= 0 or start_seconds < 0 or context_length <= 0:
        raise ValueError("window duration/context must be positive and start nonnegative")
    warmup_seconds = duration_seconds if warmup_seconds is None else warmup_seconds
    if warmup_seconds < 0 or start_seconds < warmup_seconds:
        raise ValueError("warmup must be nonnegative and fit before measurement")
    origin = min((r["timestamp"] for r in rows), default=0) if time_origin is None else time_origin
    lo, boundary, hi = (origin + start_seconds - warmup_seconds,
                        origin + start_seconds, origin + start_seconds + duration_seconds)
    candidates = sorted((r for r in rows if lo <= r["timestamp"] < hi),
                        key=lambda r: (r["timestamp"], r["chat_id"]))
    selected, excluded, selection = [], [], []
    for row in candidates:
        phase = "warmup" if row["timestamp"] < boundary else "measurement"
        if row["input_length"] + row["output_length"] > context_length:
            excluded.append({"chat_id": row["chat_id"], "phase": phase,
                             "reason": "input_plus_output_exceeds_context",
                             "input_length": row["input_length"], "output_length": row["output_length"]})
            continue
        selected.append(row)
        selection.append({"chat_id": row["chat_id"], "parent_chat_id": row["parent_chat_id"],
                          "phase": phase, "timestamp": row["timestamp"],
                          "arrival_offset_seconds": row["timestamp"] - lo,
                          "input_length": row["input_length"], "output_length": row["output_length"],
                          "hash_ids": row["hash_ids"], "input_identity_sha256": input_identity_sha256(row)})
    selected_ids = {row["chat_id"] for row in selected}
    excluded_ids = {row["chat_id"] for row in excluded}
    manifest = {"start_seconds": start_seconds, "duration_seconds": duration_seconds,
                "warmup_seconds": warmup_seconds, "time_origin": origin,
                "replay_start_timestamp": lo, "measurement_start_timestamp": boundary,
                "measurement_end_timestamp": hi, "context_length": context_length,
                "arrival_rate_scale": 1, "source_interval_requests": len(candidates),
                "selection": selection, "excluded": excluded,
                "input_identity_semantics": "SHA256 of JSON [input_length, hash_ids]; not original tokens",
                "selection_rule": "All original requests in fixed half-open warmup and measurement intervals, except input+output exceeding context. No policy-outcome selection."}
    for phase, duration in (("warmup", warmup_seconds), ("measurement", duration_seconds)):
        phase_rows = [row for row in selected if (row["timestamp"] < boundary) == (phase == "warmup")]
        locations, counts = Counter(), Counter()
        for child in phase_rows:
            parent = by_id.get(child["parent_chat_id"])
            if child["parent_chat_id"] < 0:
                location = "root"
            elif parent is None:
                location = "missing_source"
            elif parent["chat_id"] in excluded_ids:
                location = "excluded_context"
            elif parent["timestamp"] < lo:
                location = "before_warmup"
            elif parent["chat_id"] in selected_ids:
                location = "warmup" if parent["timestamp"] < boundary else "measurement"
            else:
                location = "outside_replay"
            locations[location] += 1
            if parent is not None:
                overlap = parent_overlap(parent, child)
                counts["linked_in_source"] += 1
                counts["source_parent_overlap_tokens"] += overlap["overlap_tokens"]
                counts["snapshot_eligible_in_source"] += overlap["snapshot_eligible"]
                if parent["chat_id"] in selected_ids:
                    counts["parent_in_replay"] += 1
                    counts["parent_overlap_tokens_in_replay"] += overlap["overlap_tokens"]
                    counts["snapshot_eligible_parent_in_replay"] += overlap["snapshot_eligible"]
                    if overlap["snapshot_eligible"]:
                        counts["snapshot_eligible_parent_tokens_in_replay"] += parent["input_length"]
        phase_audit = {key: counts[key] for key in (
            "linked_in_source", "source_parent_overlap_tokens", "snapshot_eligible_in_source",
            "parent_in_replay", "parent_overlap_tokens_in_replay", "snapshot_eligible_parent_in_replay",
            "snapshot_eligible_parent_tokens_in_replay")}
        phase_audit.update({"requests": len(phase_rows), "parent_location": dict(locations),
                           "offered_requests_per_second": len(phase_rows) / duration if duration else None,
                           "input_lengths": distribution(r["input_length"] for r in phase_rows),
                           "output_lengths": distribution(r["output_length"] for r in phase_rows),
                           "context_excluded_requests": sum(r["phase"] == phase for r in excluded),
                           "background_not_snapshot_eligible": len(phase_rows) - counts["snapshot_eligible_parent_in_replay"]})
        manifest[phase] = phase_audit
    return selected, manifest


def analyze_kimi_rows(rows):
    counts = Counter()
    cached, ttft, e2e, fractions, arrivals = [], [], [], [], []
    request_ids = set()
    for row in rows:
        counts["duplicate_request_ids"] += row["request_id"] in request_ids
        request_ids.add(row["request_id"])
        counts["redacted_input_records"] += bool(row.get("input_ids_redacted"))
        counts["nonempty_conversation_ids"] += bool(row.get("conversation_id"))
        arrivals.append(row["arrival_seconds"])
        hit = row.get("cached_tokens")
        if hit is None:
            counts["missing_cached_tokens"] += 1
        else:
            cached.append(hit)
            counts["cached_tokens_above_input_length"] += hit > row["input_length"]
        finished = row.get("finished_seconds")
        elapsed = None if finished is None else finished - row["arrival_seconds"]
        first = row.get("ttft_seconds")
        if elapsed is not None and math.isfinite(elapsed) and elapsed > 0:
            e2e.append(elapsed)
        else:
            counts["invalid_e2e_records"] += 1
        if first is not None and math.isfinite(first) and first >= 0:
            ttft.append(first)
            if elapsed is not None and math.isfinite(elapsed) and elapsed > 0:
                fractions.append(first / elapsed)
                counts["ttft_above_e2e"] += first > elapsed
        else:
            counts["invalid_ttft_records"] += 1
    result = {key: counts[key] for key in (
        "duplicate_request_ids", "redacted_input_records", "nonempty_conversation_ids",
        "missing_cached_tokens", "cached_tokens_above_input_length", "invalid_e2e_records",
        "invalid_ttft_records", "ttft_above_e2e")}
    result.update({"requests": len(rows), "cached_token_sum": sum(cached),
                   "span_seconds": max(arrivals) - min(arrivals) if arrivals else 0,
                   "input_lengths": distribution(r["input_length"] for r in rows),
                   "output_lengths": distribution(r["output_length"] for r in rows),
                   "cached_tokens": distribution(cached), "ttft_seconds": distribution(ttft),
                   "e2e_seconds": distribution(e2e), "ttft_over_e2e": distribution(fractions),
                   "field_presence": dict(Counter(key for row in rows for key in row)),
                   "native_lineage_replay_supported": False,
                   "limitations": "Observed Finish records only; redacted token identities and absent conversation/parent linkage cannot support native prefix or agent replay. Arrival/finish fields are observed timing, not tool durations."})
    return result


def load_jsonl(path):
    digest, rows = hashlib.sha256(), []
    with path.open("rb") as source:
        for raw in source:
            digest.update(raw)
            if raw.strip():
                rows.append(json.loads(raw))
    return rows, {"path": str(path.resolve()), "sha256": digest.hexdigest(), "rows": len(rows)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--kimi-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=float, nargs="+", default=DEFAULT_STARTS)
    parser.add_argument("--durations", type=float, nargs="+", default=DEFAULT_DURATIONS)
    parser.add_argument("--context-length", type=int, default=32768)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    result = {"schema_version": 1, "selection_predeclared_before_policy_runs": True,
              "starts_seconds": args.starts, "durations_seconds": args.durations,
              "context_length": args.context_length, "splits": {}}
    lines = ["# Continuous workload opportunity audit", "",
             f"Windows are fixed by arrival time, retain background traffic, and include an equally long preceding warmup. No policy outcomes or parent desirability determine selection. Input+output exceeding the {args.context_length:,}-token engine context is excluded and listed; original output lengths are retained. Arrival gaps are not tool gaps.", "",
             "Snapshot eligibility below is an input-identity prerequisite, not proof that a real engine retained compatible hybrid state. Replay uses anonymous surrogate tokens, not original text or multimedia.", "",
             "| Split | Requests | Linked | Eligible complete-parent extensions | Req/s | Input p50/p90 | Output p50/p90 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    window_lines = ["", "| Split | Start / duration (s) | Warmup / measured requests | Excluded | Parents in replay / linked | Eligible children | Eligible parent tokens | Output tokens, warmup + measured |",
                    "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for split, filename in QWEN_FILES.items():
        rows, source = load_jsonl(args.trace_root / filename)
        population = analyze_qwen_rows(rows)
        population["source"] = source
        windows = []
        for start in args.starts:
            for duration in args.durations:
                _, manifest = select_continuous_window(rows, start_seconds=start,
                                                       duration_seconds=duration,
                                                       context_length=args.context_length)
                name = f"{split}-start-{start:g}-duration-{duration:g}.json"
                manifest["source"] = source
                write_json(args.output / name, manifest)
                compact = {key: value for key, value in manifest.items() if key not in ("selection", "excluded")}
                compact["manifest"] = name
                compact["excluded_requests"] = len(manifest["excluded"])
                windows.append(compact)
                measured, warmup = manifest["measurement"], manifest["warmup"]
                window_lines.append(f"| {split} | {start:g} / {duration:g} | {warmup['requests']} / {measured['requests']} | {len(manifest['excluded'])} | {measured['parent_in_replay']} / {measured['linked_in_source']} | {measured['snapshot_eligible_parent_in_replay']} | {measured['snapshot_eligible_parent_tokens_in_replay']} | {int(warmup['output_lengths']['sum'] + measured['output_lengths']['sum'])} |")
        result["splits"][split] = {"population": population, "windows": windows}
        ip, op = population["input_lengths"], population["output_lengths"]
        lines.append(f"| {split} | {len(rows)} | {population['linked_requests']} | {population['snapshot_eligible_children']} | {population['observed_requests_per_second']:.2f} | {ip['p50']:.0f} / {ip['p90']:.0f} | {op['p50']:.0f} / {op['p90']:.0f} |")
        write_json(args.output / f"{split}-population.json", population)
        print(json.dumps({"split": split, "requests": len(rows), "windows": len(windows),
                          "elapsed_seconds": time.perf_counter() - began}), flush=True)
    if args.kimi_jsonl:
        rows, source = load_jsonl(args.kimi_jsonl)
        kimi = analyze_kimi_rows(rows)
        kimi["source"] = source
        result["kimi"] = kimi
        write_json(args.output / "kimi-observation.json", kimi)
        lines += ["", f"Kimi: {kimi['requests']:,} observed Finish records; {kimi['nonempty_conversation_ids']} nonempty conversation IDs. TTFT/E2E p50/p90: {kimi['ttft_over_e2e']['p50']:.3f} / {kimi['ttft_over_e2e']['p90']:.3f}. These records characterize observed lengths/timing/cache counts but cannot supply native lineage or prefix identities."]
    lines += window_lines
    lines += ["", f"Recommended first continuous intervals: the {min(args.durations):g}-second windows at start={min(args.starts):g} for calibration, and starts={','.join(f'{value:g}' for value in sorted(args.starts)[1:])} for held-out comparisons, each with {min(args.durations):g} seconds of warmup. This rule uses the shortest predeclared duration and temporal separation, never policy outcomes. Measured request and decode-token counts above determine whether full original-rate replay fits the budget; if arrival scaling is needed, label it and retain every request. A finite warmup underestimates reuse whose parents arrived earlier.",
              "", "The current full-parent-only materialization path applies to the eligible column. Other background and partial-prefix requests must remain in replay. Eligibility does not establish idle time for preparation, cold recomputation cost, contention, or physical GPU savings; these require runtime observations.", ""]
    result["recommended_windows"] = {"selection_rule": "Shortest predeclared duration; first start calibrates, later starts are held out. No policy outcomes used.",
                                       "calibration_start_seconds": min(args.starts),
                                       "heldout_start_seconds": sorted(args.starts)[1:],
                                       "duration_seconds": min(args.durations),
                                       "warmup_seconds": min(args.durations)}
    result["elapsed_seconds"] = time.perf_counter() - began
    write_json(args.output / "workload-audit.json", result)
    (args.output / "WORKLOAD_AUDIT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
