#!/usr/bin/env python3
"""Report observed continuous runs without turning incomplete evidence into wins."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time

import numpy as np

from agentshift.controller.preparation_decisions import CostCurve
from agentshift.workloads.production import surrogate_tokens
from experiments.fluidkv.analyze_decision_workloads import parent_overlap
from experiments.fluidkv.audit_identity_ambiguity import classify_pair


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def stats(values):
    values = [value for value in values if finite(value)]
    if not values:
        return {"n": 0, "min": None, "mean": None, "median": None, "p50": None, "p90": None, "p99": None, "max": None}
    p50, p90, p99 = np.percentile(values, [50, 90, 99]).tolist()
    return {"n": len(values), "min": min(values), "mean": float(np.mean(values)), "median": p50, "p50": p50,
            "p90": p90, "p99": p99, "max": max(values)}


def metrics(rows, offered, slo, observation):
    good = [row for row in rows if row.get("success")]
    result = {"offered": offered, "successful": len(good), "failed_or_unfinished": offered-len(good),
              "slo_seconds": slo, "actual_cached_tokens": sum(row.get("actual_cached_tokens", 0) for row in good),
              "actual_output_tokens": sum(row.get("completion_tokens", 0) for row in good),
              "latency_statistics_scope": "Mean and quantiles use finite latency observations from successful requests only; failed/unfinished requests remain in offered-load and SLO denominators. Mean/p90 are secondary diagnostics, not replacement primary outcomes."}
    for name in ("ttft_seconds", "arrival_ttft_seconds", "e2e_seconds", "arrival_e2e_seconds",
                 "router_wait_seconds", "preparation_wait_seconds"):
        result[name] = stats(row.get(name) for row in good)
    for prefix, field in (("", "ttft_seconds"), ("arrival_", "arrival_ttft_seconds")):
        passed = sum(finite(row.get(field)) and row[field] <= slo for row in good)
        result[prefix+"slo_attainment"] = passed/offered if offered else None
        result[prefix+"slo_goodput_requests_per_second"] = passed/observation if observation else None
    result["observation_seconds"] = observation
    result["slo_sensitivity"] = {str(limit): sum(finite(row.get("ttft_seconds")) and row["ttft_seconds"] <= limit for row in good)/offered if offered else None
                                 for limit in (1.0, 5.0)}
    return result


def inspected_source_snapshot(parent, child, selected, events, known_prefix, preparations=(), physical_release=None):
    """Replay only authenticated source observations already made by dispatch.

    Later empty observations remove boundaries they actually queried. A shorter
    query cannot disprove a longer snapshot. None of these records reserve state.
    """
    dispatch = child.get("dispatched_seconds")
    result = {"observed_reusable_tokens": 0, "valid_observations": 0,
              "source_rejections_applied": 0,
              "source_released_before_dispatch": False,
              "observed_seconds": None, "observation_age_at_dispatch_seconds": None,
              "source_availability_at_dispatch_proven": False,
              "scope": "Candidate at child dispatch supported by an earlier engine observation, potentially stale. Explicit source rejection, contradictory inspection or confirmed source release invalidates it. This is not historical proactive-copy opportunity or a reservation."}
    if not finite(dispatch) or known_prefix <= 0:
        return result
    candidates, digests = [], {}
    for event in events:
        if event.get("event") not in ("prefix_inspection", "prefix_inspection_failed"):
            continue
        when = event.get("seconds")
        if event.get("engine") != parent["target"] or not finite(when) or when < 0 or when > dispatch:
            continue
        if event["event"] == "prefix_inspection_failed":
            candidates.append((when, None, None, event))
            continue
        query_chat = event.get("chat_id")
        record = parent if query_chat == parent["chat_id"] else child if query_chat == child["chat_id"] else None
        if record is None or (record is child and event.get("phase") != "routing"):
            continue
        query_count = event.get("query_tokens")
        if type(query_count) is not int or not 0 < query_count <= record.get("input_tokens", 0):
            continue
        recorded_digest = record.get("prompt_token_sha256")
        if not recorded_digest:
            continue
        expected_digest = recorded_digest
        if query_count != record["input_tokens"]:
            metadata = selected.get(record["chat_id"])
            if metadata is None:
                continue
            key = (record["chat_id"], query_count)
            if key not in digests:
                tokens = surrogate_tokens(metadata)
                # A reconstructed anonymous query must also agree with the
                # actual request fingerprint before its prefix hash is used.
                if hashlib.sha256(json.dumps(tokens).encode()).hexdigest() != recorded_digest:
                    continue
                digests[key] = hashlib.sha256(json.dumps(tokens[:query_count]).encode()).hexdigest()
            expected_digest = digests[key]
        if event.get("query_token_sha256") != expected_digest:
            continue
        response = event.get("result")
        if not isinstance(response, dict) or response.get("query_tokens") != query_count:
            continue
        boundaries, matched = response.get("snapshot_boundaries"), response.get("matched_tokens")
        if (type(matched) is not int or not 0 <= matched <= query_count or not isinstance(boundaries, list)
                or any(type(value) is not int or not 0 < value <= matched for value in boundaries)
                or any(left >= right for left, right in zip(boundaries, boundaries[1:]))):
            continue
        candidates.append((when, query_count, boundaries, event))
    for move in preparations:
        when = move.get("finish_seconds")
        if (move.get("source") != parent["target"] or move.get("chat_id") != parent["chat_id"]
                or not move.get("attempted") or move.get("success")
                or not finite(when) or not 0 <= when <= dispatch):
            continue
        missing = re.search(r"source has only (\d+) of (\d+) (?:prefix )?tokens", str(move.get("error", "")))
        if not missing:
            continue
        matched, query_count = map(int, missing.groups())
        if query_count != move.get("input_tokens") or not 0 <= matched < query_count <= parent["input_tokens"]:
            continue
        candidates.append((when, query_count, None, {"event": "source_snapshot_rejected", "matched_tokens": matched,
                                                     "error": move["error"], "seconds": when}))
    surviving = {}
    for when, query_count, boundaries, event in sorted(candidates, key=lambda item: item[0]):
        if query_count is None:
            surviving.clear()
            continue
        if event.get("event") == "source_snapshot_rejected":
            result["source_rejections_applied"] += 1
            result["last_source_rejection_seconds"] = when
            result["last_source_rejection_error"] = event["error"]
            surviving = {boundary: value for boundary, value in surviving.items()
                         if not event["matched_tokens"] < boundary <= query_count}
            continue
        result["valid_observations"] += 1
        coverage = min(query_count, known_prefix)
        surviving = {boundary: value for boundary, value in surviving.items() if boundary > coverage}
        for boundary in boundaries:
            if boundary <= known_prefix and boundary < child["input_tokens"]:
                surviving[boundary] = event
    release = physical_release or {}
    confirmed = release.get("confirmed_seconds")
    if release.get("released") and finite(confirmed) and 0 <= confirmed <= dispatch:
        surviving.clear()
        result.update(source_released_before_dispatch=True, confirmed_source_release_seconds=confirmed)
    if surviving:
        boundary = max(surviving)
        event = surviving[boundary]
        result.update(observed_reusable_tokens=boundary, observed_seconds=event["seconds"],
                      observation_age_at_dispatch_seconds=dispatch-event["seconds"],
                      query_chat_id=event["chat_id"], query_phase=event.get("phase"),
                      query_tokens=event["query_tokens"], query_token_sha256=event["query_token_sha256"])
    return result


def analyze_run(summary, records, selection=None, cold_curve=None, *, events=None, inspection_enabled=None):
    events = events or []
    if inspection_enabled is None:
        inspection_enabled = any(event.get("event") in ("prefix_inspection", "prefix_inspection_failed") for event in events)
    policy, repetition = summary["policy"], summary["repetition"]
    scoped = [row for row in records if row.get("policy", policy) == policy
              and row.get("repetition", repetition) == repetition]
    terminal = {row["chat_id"]: row for row in scoped}
    selected = {row["chat_id"]: row for row in (selection or {}).get("selection", [])}
    measured = [row for row in terminal.values() if row.get("phase") == "measurement"]
    expected = [row for row in selected.values() if row["phase"] == "measurement"]
    offered = max(len(measured), len(expected), summary["measurement"].get("offered", 0))
    slo = summary["measurement"].get("slo_seconds", 1.0)
    observation = summary.get("measurement_observation_seconds", summary["measurement"].get("observation_seconds", 0))
    linked_ids = {row["chat_id"] for row in expected if row["parent_chat_id"] in selected}
    if not selected:
        linked_ids = {row["chat_id"] for row in measured if row.get("parent_chat_id") in terminal}
    children = [row for row in measured if row["chat_id"] in linked_ids]
    child_metrics = metrics(children, len(linked_ids), slo, observation)
    exact = set()
    for row in expected:
        parent = selected.get(row["parent_chat_id"])
        if parent and parent_overlap(parent, row)["snapshot_eligible"]:
            exact.add(row["chat_id"])
    child_metrics["exact_parent_extensions"] = len(exact) if selected else None
    child_metrics["lineage_scope"] = "Parent also selected in this replay; all selected children, including missing terminal rows, remain in denominator."
    cold_cases = []
    for child in children:
        parent = terminal.get(child.get("parent_chat_id"))
        if (not child.get("success") or parent is None or not parent.get("success")
                or parent.get("target") is None or child.get("target") is None
                or parent["target"] == child["target"]):
            continue
        parent_tokens = parent.get("input_tokens", 0)
        overlap = identity = None
        if child["chat_id"] in selected and parent["chat_id"] in selected:
            overlap = parent_overlap(selected[parent["chat_id"]], selected[child["chat_id"]])
            identity = classify_pair(selected[parent["chat_id"]], selected[child["chat_id"]])
        verified = identity["confirmed_shared_complete_tokens"] if identity else parent_tokens
        missing = max(0, verified-child.get("actual_cached_tokens", 0))
        if not missing:
            continue
        source_release = summary.get("physical_release") if summary.get("retired_engine") == parent["target"] else None
        snapshot = inspected_source_snapshot(parent, child, selected, events, verified,
                                              summary.get("preparations", []), source_release) if identity else None
        cold_cases.append({"parent_chat_id": parent["chat_id"], "child_chat_id": child["chat_id"],
            "source": parent["target"], "target": child["target"], "parent_input_tokens": parent_tokens,
            "actual_cached_tokens": child.get("actual_cached_tokens", 0),
            "verified_overlap_tokens": verified if overlap else None,
            "verified_missing_parent_tokens": missing if overlap else None,
            "verified_common_prefix_tokens": verified if identity else None,
            "lost_known_prefix_tokens": missing if identity else None,
            "parent_identity_category": identity["category"] if identity else None,
            "source_snapshot": snapshot,
            "exact_snapshot_eligible": overlap["snapshot_eligible"] if overlap else None,
            "identity_status": "verified_anonymous_hash_prefix" if overlap else "unverified_no_selection_metadata",
            "ttft_seconds": child.get("ttft_seconds"), "arrival_ttft_seconds": child.get("arrival_ttft_seconds"),
            "e2e_seconds": child.get("e2e_seconds")})
    preparations = []
    for move in summary.get("preparations", []):
        candidates = [row for row in terminal.values()
                      if row.get("parent_chat_id") == move.get("chat_id") and row.get("target") == move.get("target")]
        consistent, late = [], []
        for child in candidates:
            dispatched, ready = child.get("dispatched_seconds"), move.get("finish_seconds")
            if not finite(dispatched) or not finite(ready):
                continue
            if ready > dispatched:
                late.append(child["chat_id"])
                continue
            identity_ok = True
            if child["chat_id"] in selected and move.get("chat_id") in selected:
                identity_ok = classify_pair(selected[move["chat_id"]], selected[child["chat_id"]])["confirmed_shared_complete_tokens"] >= move["input_tokens"]
            if (move.get("success") and child.get("success") and identity_ok
                    and child.get("actual_cached_tokens", 0) >= move["input_tokens"]):
                consistent.append(child["chat_id"])
        preparations.append({**move, "consistent_with_consumption_children": consistent,
                             "late_for_matching_children": late,
                             "no_observed_consumption": bool(move.get("success") and not consistent)})
    prep = {"decisions": len(preparations), "attempted": sum(bool(r.get("attempted")) for r in preparations),
            "successful": sum(bool(r.get("success")) for r in preparations),
            "consistent_with_consumption": sum(bool(r["consistent_with_consumption_children"]) for r in preparations),
            "late_for_matching_child": sum(bool(r.get("success") and r["late_for_matching_children"]) for r in preparations),
            "no_observed_consumption": sum(r["no_observed_consumption"] for r in preparations),
            "transfer_bytes": sum(r.get("transfer_bytes", 0) for r in preparations),
            "reconstruction_tokens": sum(r.get("reconstruction_tokens", 0) for r in preparations),
            "preparation_output_tokens": sum(r.get("preparation_output_tokens", 0) for r in preparations),
            "reasons": dict(Counter(r.get("reason", "attempted") for r in preparations)),
            "records": preparations,
            "attribution": "Timing, target and actual hits are consistent with consumption; preexisting or independently created cache can produce the same association. No-observed-consumption is not definitive unused-state proof."}
    headroom = {"estimated_cases": 0, "candidate_cases": 0, "estimated_service_seconds_sum": 0.0,
                "unestimated_outside_calibration": 0, "lower_length_clamped_cases": 0,
                "mode": "engine_observed_checkpoints" if inspection_enabled else "legacy_strict_whole_parent_subset",
                "unestimated_missing_snapshot_evidence": 0, "observed_snapshot_no_incremental_hit": 0,
                "unestimated_released_source": 0,
                "source_availability_proven": False, "is_cluster_upper_bound": False, "cases": [],
                "scope": "Conditional isolated-service diagnostic for source candidates at child dispatch, not historical proactive-copy opportunity. With inspection enabled, select only an authenticated earlier source snapshot within the verified common complete-block prefix; later contradictory observations, explicit source rejection and confirmed source release before dispatch invalidate it. Such state may still be stale and is hypothesized usable at the destination. Without inspection, preserve the explicitly labelled legacy whole-parent subset assumption. Subtract actual hits, retain a small-input service floor, reject above-calibration lengths. Preparation, queueing, batching, interference and release coupling are excluded. Unknown fragments are never equated. This is not a cluster upper bound; missing or released source evidence stays unestimated and does not establish zero original-trace opportunity."}
    for case in cold_cases:
        if not case["verified_common_prefix_tokens"]:
            continue
        if inspection_enabled:
            headroom["candidate_cases"] += 1
            restorable = (case["source_snapshot"] or {}).get("observed_reusable_tokens", 0)
            if not restorable:
                headroom["unestimated_missing_snapshot_evidence"] += 1
                headroom["unestimated_released_source"] += bool((case["source_snapshot"] or {}).get("source_released_before_dispatch"))
                continue
            if restorable <= case["actual_cached_tokens"]:
                headroom["observed_snapshot_no_incremental_hit"] += 1
                continue
        else:
            if not case["exact_snapshot_eligible"]:
                continue
            headroom["candidate_cases"] += 1
            if (case["source_snapshot"] or {}).get("source_released_before_dispatch"):
                headroom["unestimated_missing_snapshot_evidence"] += 1
                headroom["unestimated_released_source"] += 1
                continue
            restorable = case["parent_input_tokens"]
        if cold_curve is None:
            continue
        child = terminal[case["child_chat_id"]]
        total, cached = child["input_tokens"], case["actual_cached_tokens"]
        before, after = max(0, total-cached), max(0, total-max(cached, restorable))
        low, high = cold_curve.points[0][0], cold_curve.points[-1][0]
        if before > high or after > high:
            headroom["unestimated_outside_calibration"] += 1
            continue
        clamped = before < low or after < low
        saved = max(0, cold_curve.seconds(max(low, before))-cold_curve.seconds(max(low, after)))
        headroom["cases"].append({"child_chat_id": case["child_chat_id"],
                                  "restorable_prefix_tokens": restorable,
                                  "source_snapshot": case["source_snapshot"] if inspection_enabled else None,
                                  "remaining_input_before": before, "remaining_input_after": after,
                                  "lower_length_clamped": clamped, "estimated_service_seconds": saved})
        headroom["estimated_cases"] += 1
        headroom["lower_length_clamped_cases"] += clamped
        headroom["estimated_service_seconds_sum"] += saved
    headroom["calibration_available"] = cold_curve is not None
    headroom["result_status"] = ("estimated_conditionally" if headroom["estimated_cases"] else
                                  "no_eligible_observed_cold_reroutes" if not headroom["candidate_cases"] else "unavailable")
    expected_terminal = max(len(selected), summary["measurement"].get("offered", 0)
                            + (summary.get("warmup") or {}).get("offered", 0))
    return {"population": metrics(measured, offered, slo, observation), "children": child_metrics,
            "preparation": prep, "cold_reroute_cases": cold_cases, "conditional_service_headroom": headroom,
            "integrity": {"foreign_policy_records": len(records)-len(scoped),
                          "duplicate_terminal_records": len(scoped)-len(terminal),
                          "missing_terminal_chat_ids": sorted(set(selected)-set(terminal)),
                          "terminal_count_shortfall": max(0, expected_terminal-len(terminal)),
                          "selection_available": bool(selected)}}


def get_path(value, keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


PAIR_METRICS = {
    "population_ttft_p50_seconds": ("population", "ttft_seconds", "p50"),
    "population_arrival_ttft_p50_seconds": ("population", "arrival_ttft_seconds", "p50"),
    "population_e2e_p50_seconds": ("population", "e2e_seconds", "p50"),
    "population_arrival_e2e_p50_seconds": ("population", "arrival_e2e_seconds", "p50"),
    "population_ttft_mean_seconds": ("population", "ttft_seconds", "mean"),
    "population_ttft_p90_seconds": ("population", "ttft_seconds", "p90"),
    "population_e2e_mean_seconds": ("population", "e2e_seconds", "mean"),
    "population_e2e_p90_seconds": ("population", "e2e_seconds", "p90"),
    "population_arrival_ttft_mean_seconds": ("population", "arrival_ttft_seconds", "mean"),
    "population_arrival_ttft_p90_seconds": ("population", "arrival_ttft_seconds", "p90"),
    "population_arrival_e2e_mean_seconds": ("population", "arrival_e2e_seconds", "mean"),
    "population_arrival_e2e_p90_seconds": ("population", "arrival_e2e_seconds", "p90"),
    "population_slo_attainment": ("population", "slo_attainment"),
    "population_arrival_slo_attainment": ("population", "arrival_slo_attainment"),
    "population_slo_goodput": ("population", "slo_goodput_requests_per_second"),
    "population_slo_1s_attainment": ("population", "slo_sensitivity", "1.0"),
    "population_slo_5s_attainment": ("population", "slo_sensitivity", "5.0"),
    "child_ttft_p50_seconds": ("children", "ttft_seconds", "p50"),
    "child_e2e_p50_seconds": ("children", "e2e_seconds", "p50"),
    "model_occupied_gpu_seconds": ("model_occupied_gpu_seconds",),
    "model_occupied_gpu_seconds_including_group_setup": ("model_occupied_gpu_seconds_including_group_setup",),
    "retirement_to_last_foreground_seconds": ("drain_components", "retirement_to_last_foreground_seconds"),
    "foreground_finish_to_signal_seconds": ("drain_components", "foreground_finish_to_signal_seconds"),
    "shutdown_confirmation_seconds": ("drain_components", "shutdown_confirmation_seconds"),
    "retirement_to_release_seconds": ("drain_components", "retirement_to_release_seconds"),
}


def paired_comparisons(runs):
    groups = defaultdict(list)
    for run in runs:
        if run.get("eligible_for_pairing"):
            groups[run["condition_id"]].append(run)
    result = []
    for condition, group in sorted(groups.items()):
        index = defaultdict(list)
        for run in group:
            index[(run["policy"], run["repetition"])].append(run)
        for policy in sorted({run["policy"] for run in group} - {"dualmap"}):
            pairs = []
            ambiguous = []
            for repetition in sorted({run["repetition"] for run in group}):
                left, right = index[("dualmap", repetition)], index[(policy, repetition)]
                if len(left) == len(right) == 1:
                    pairs.append((left[0], right[0]))
                elif left and right:
                    ambiguous.append(repetition)
            if not pairs and not ambiguous:
                continue
            entry = {"condition_id": condition, "baseline": "dualmap", "policy": policy,
                     "paired_repetitions": [left["repetition"] for left, _ in pairs],
                     "timed_out_repetitions": [left["repetition"] for left, right in pairs
                                                if left.get("timed_out") or right.get("timed_out")],
                     "ambiguous_repetitions_excluded": ambiguous,
                     "unit": "One paired run within one fixed window and resource condition; requests are not independent repetitions.",
                     "delta_direction": "variant minus DualMap; negative latency/cost and positive attainment/goodput favor variant",
                     "metrics": {}}
            for name, keys in PAIR_METRICS.items():
                deltas = []
                for left, right in pairs:
                    baseline, variant = get_path(left, keys), get_path(right, keys)
                    if finite(baseline) and finite(variant):
                        deltas.append({"repetition": left["repetition"], "baseline": baseline,
                                       "variant": variant, "delta": variant-baseline,
                                       "baseline_run": left["path"], "variant_run": right["path"]})
                entry["metrics"][name] = {**stats(row["delta"] for row in deltas), "paired_deltas": deltas}
            result.append(entry)
    return result


def read_json(path, issues):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        issues.append({"path": str(path), "error": str(exc), "status": "unreadable_or_being_written"})
        return None


def read_jsonl(path, issues):
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            issues.append({"path": str(path), "line": number, "error": str(exc), "status": "partial_record"})
    return rows


def ancestor_file(folder, name, root):
    while folder == root or root in folder.parents:
        path = folder/name
        if path.exists():
            return path
        folder = folder.parent
    return None


def latest_profile(root):
    preferred = root/"production-calibration-local"/"decision_profile.json"
    if preferred.exists():
        return preferred
    candidates = list(root.rglob("decision_profile.json"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def curve_from_profile(profile):
    if not profile:
        return None
    if "cold" in profile:
        points = profile["cold"]
    elif "lengths" in profile:
        # Current probe's cold request includes the 64-token suffix.
        suffix = profile.get("args", {}).get("suffix_tokens", 64)
        points = [(int(n)+suffix, row["cold_ttft_seconds"]) for n, row in profile["lengths"].items()
                  if "cold_ttft_seconds" in row]
    else:
        return None
    return CostCurve(tuple(sorted((int(n), float(t)) for n, t in points))) if points else None


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def condition_for(args, selection, folder, root, issues):
    keys = ("trace", "start", "duration", "warmup", "time_scale", "retire_offset", "retire_engine",
            "physical_release", "slo", "prefill_seconds_per_token", "warm_ttft_seconds",
            "copy_interference_seconds", "minimum_gain_seconds", "busy_copy_multiplier", "cache_tokens",
            "snapshot_slots", "state_observation", "source_observation", "inspect_prefixes",
            "request_timeout", "run_timeout", "engines")
    condition = {key: args.get(key) for key in keys}
    condition["selection_sha256"] = stable_digest((selection or {}).get("selection", []))
    curve_path = ancestor_file(folder, "cost_curves.json", root)
    condition["cost_curve_sha256"] = stable_digest(read_json(curve_path, issues)) if curve_path else None
    server_path = ancestor_file(folder, "server_info.json", root)
    if server_path:
        info = read_json(server_path, issues)
        # Runtime loads and counters are not experimental backend identity.
        server_keys = ("model_path", "attention_backend", "mamba_backend", "linear_attn_backend",
                       "disable_cuda_graph", "enable_deterministic_inference", "max_total_tokens",
                       "max_mamba_cache_size", "context_length", "mamba_ssm_dtype", "max_running_requests",
                       "dtype", "kv_cache_dtype", "chunked_prefill_size", "max_prefill_tokens",
                       "disable_overlap_schedule", "tp_size", "dp_size", "ep_size",
                       "cuda_graph_bs", "cuda_graph_max_bs", "disable_cuda_graph_padding",
                       "mamba_scheduler_strategy", "mamba_track_interval", "page_size")
        servers = info if isinstance(info, list) else [info] if isinstance(info, dict) else []
        condition["backend"] = [{key: (item.get("server_args") or {}).get(key, item.get(key))
                                  for key in server_keys} for item in servers]
        for backend, item in zip(condition["backend"], servers):
            states = item.get("internal_states", [])
            states = states if isinstance(states, list) else [states]
            backend["effective_max_running_requests_per_dp"] = sorted({
                state["effective_max_running_requests_per_dp"] for state in states
                if isinstance(state, dict) and type(state.get("effective_max_running_requests_per_dp")) is int})
    else:
        condition["backend"] = None
    # Without manifests/args, different directories cannot establish pairing.
    if not selection or not args:
        condition["unverified_run_directory"] = str(folder)
    return stable_digest(condition), condition


def lifecycle_observations(value, origin):
    """Extract measured lifecycle facts without copying process environments."""
    records = []
    if isinstance(value, dict):
        if "released" in value or {"started_monotonic", "ready_monotonic"} <= value.keys():
            row = {"artifact": origin, **{key: value[key] for key in (
                "released", "graceful", "forced_kill", "shutdown_seconds", "gpu_uuids", "error") if key in value}}
            if finite(value.get("started_monotonic")) and finite(value.get("ready_monotonic")):
                row["ready_seconds"] = value["ready_monotonic"]-value["started_monotonic"]
            records.append(row)
        for key, child in value.items():
            # Raw telemetry and captured process identity are already in source.
            if key not in ("observations", "identity", "environment", "server_info"):
                records.extend(lifecycle_observations(child, origin+"/"+key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            records.extend(lifecycle_observations(child, origin+f"/{index}"))
    return records


def effective_concurrency(events, backend):
    configured = {str(index): item.get("max_running_requests") for index, item in enumerate(backend or [])}
    limits, peaks, samples = defaultdict(set), {}, Counter()
    for event in events:
        if event.get("event") != "engine_load":
            continue
        for measurement in event.get("measurements", []):
            engine = str(measurement.get("engine"))
            response = measurement.get("response", {})
            for load in response.get("loads", []):
                limit, running = load.get("max_running_requests"), load.get("num_running_reqs")
                if type(limit) is int and limit >= 0:
                    limits[engine].add(limit)
                    samples[engine] += 1
                if type(running) is int and running >= 0:
                    peaks[engine] = max(peaks.get(engine, 0), running)
    return {"configured_max_running_requests_by_engine": configured,
            "observed_effective_limits_by_engine": {engine: sorted(values) for engine, values in limits.items()},
            "observed_running_peak_by_engine": peaks, "sample_count_by_engine": dict(samples),
            "scope": "Configured admission setting and measured /v1/loads runtime limit are distinct. Peaks are sampled observations, not a continuous maximum. Retired engines naturally have fewer samples."}


def drain_components(summary, records, events):
    source = summary.get("retired_engine")
    retirement = next((event.get("seconds") for event in events if event.get("event") == "logical_retirement"
                       and event.get("engine") == source), None)
    source_rows = [row for row in records if row.get("target") == source and source is not None]
    inflight = [row for row in source_rows if finite(retirement) and finite(row.get("dispatched_seconds"))
                and finite(row.get("finished_seconds")) and row["dispatched_seconds"] <= retirement < row["finished_seconds"]]
    last_foreground = max((row["finished_seconds"] for row in source_rows if finite(row.get("finished_seconds"))), default=None)
    last_preparation = max((move["finish_seconds"] for move in summary.get("preparations", [])
                            if move.get("source") == source and finite(move.get("finish_seconds"))), default=None)
    last_inspection = max((event["seconds"] for event in events if event.get("engine") == source
                            and event.get("event") in ("prefix_inspection", "prefix_inspection_failed")
                            and finite(event.get("seconds"))), default=None)
    release = summary.get("physical_release") or {}
    confirmed = release.get("confirmed_seconds") if release.get("released") else None
    signal = None
    if finite(confirmed) and finite(release.get("confirmed_monotonic")) and finite(release.get("signal_monotonic")):
        signal = confirmed-(release["confirmed_monotonic"]-release["signal_monotonic"])
    difference = lambda end, start: max(0, end-start) if finite(end) and finite(start) else None
    return {"source_engine": source, "retirement_seconds": retirement,
            "source_requests": len(source_rows), "inflight_at_retirement": len(inflight) if finite(retirement) else None,
            "inflight_chat_ids": [row["chat_id"] for row in inflight],
            "inflight_requested_output_tokens": sum(row.get("requested_output_tokens", 0) for row in inflight),
            "last_source_foreground_finish_seconds": last_foreground,
            "last_source_preparation_finish_seconds": last_preparation,
            "last_source_inspection_seconds": last_inspection,
            "physical_signal_seconds": signal, "physical_confirmed_seconds": confirmed,
            "retirement_to_last_foreground_seconds": difference(last_foreground, retirement),
            "foreground_finish_to_signal_seconds": difference(signal, last_foreground),
            "shutdown_confirmation_seconds": difference(confirmed, signal),
            "retirement_to_release_seconds": difference(confirmed, retirement),
            "forced_kill": release.get("forced_kill"), "graceful": release.get("graceful"),
            "scope": "Observed source timeline. Requested output tokens are total lengths, not tokens remaining at retirement. Warmup placement and in-flight work can vary across runs. Preparation, inspection and foreground intervals overlap; release differences cannot be attributed solely to copying. Missing terminal rows make this breakdown incomplete."}


def build_report(root, calibration_path=None):
    root = Path(root).resolve()
    issues, runs, incomplete, standalone_lifecycle = [], [], [], []
    profile_path = Path(calibration_path).resolve() if calibration_path else latest_profile(root)
    if profile_path and profile_path.is_dir():
        profile_path = latest_profile(profile_path)
    profile = read_json(profile_path, issues) if profile_path else None
    default_curve = curve_from_profile(profile)
    run_folders = set()
    for path in sorted(root.rglob("summary.json")):
        summary = read_json(path, issues)
        if not isinstance(summary, dict) or not {"policy", "repetition", "measurement"} <= summary.keys():
            if isinstance(summary, dict) and "runs" not in summary:
                standalone_lifecycle.extend(lifecycle_observations(summary, str(path.relative_to(root))))
            continue
        folder = path.parent
        run_folders.add(folder)
        arguments_path = ancestor_file(folder, "arguments.json", root)
        selection_path = ancestor_file(folder, "selection.json", root)
        args = read_json(arguments_path, issues) if arguments_path else {}
        selection = read_json(selection_path, issues) if selection_path else None
        records = read_jsonl(folder/"requests.jsonl", issues)
        events = read_jsonl(folder/"events.jsonl", issues)
        curve_path = ancestor_file(folder, "cost_curves.json", root)
        curve = curve_from_profile(read_json(curve_path, issues)) if curve_path else default_curve
        analysis = analyze_run(summary, records, selection, curve, events=events,
                               inspection_enabled=(args or {}).get("inspect_prefixes"))
        condition_id, condition = condition_for(args or {}, selection, folder, root, issues)
        integrity = analysis["integrity"]
        complete_records = not (integrity["missing_terminal_chat_ids"] or integrity["terminal_count_shortfall"] or integrity["duplicate_terminal_records"]
                                or integrity["foreign_policy_records"])
        fatal_error = summary.get("fatal_error")
        if fatal_error is not None:
            incomplete.append({"path": str(folder.relative_to(root)), "status": "invalid_harness_fatal",
                               "fatal_error": fatal_error, **integrity})
        elif not complete_records:
            incomplete.append({"path": str(folder.relative_to(root)), "status": "terminal_accounting_incomplete", **integrity})
        elif summary.get("timed_out"):
            incomplete.append({"path": str(folder.relative_to(root)), "status": "timed_out_accounted"})
        run = {"path": str(folder.relative_to(root)), "policy": summary["policy"], "repetition": summary["repetition"],
               "condition_id": condition_id, "condition": condition, "timed_out": summary.get("timed_out", False),
               "fatal_error": fatal_error, "valid_policy_experiment": complete_records and fatal_error is None,
               "eligible_for_pairing": complete_records and fatal_error is None,
               "measurement_observation_seconds": summary.get("measurement_observation_seconds"),
               "model_occupied_gpu_seconds": summary.get("model_occupied_gpu_seconds"),
               "model_occupied_gpu_seconds_including_group_setup": summary.get("model_occupied_gpu_seconds_including_group_setup"),
               "gpu_accounting_scope": summary.get("gpu_accounting_scope"),
               "physical_release": summary.get("physical_release"), "upstream_rebalanced_requests": summary.get("upstream_rebalanced_requests"),
               "prefix_inspection_events": sum(event.get("event") == "prefix_inspection" for event in events),
               "prefix_inspection_failed_events": sum(event.get("event") == "prefix_inspection_failed" for event in events),
               "effective_concurrency": effective_concurrency(events, condition.get("backend")),
               "drain_components": drain_components(summary, records, events),
               "headroom_calibration_path": str(curve_path or profile_path) if (curve_path or profile_path) else None,
               "raw_summary_metrics": {"measurement": summary["measurement"], "warmup": summary.get("warmup")}, **analysis}
        runs.append(run)
    expected_folders = set()
    for path in sorted(root.rglob("arguments.json")):
        args = read_json(path, issues)
        if not isinstance(args, dict) or not {"policies", "repetitions", "start", "duration"} <= args.keys():
            continue
        repetition_start = args.get("repetition_start", 0)
        for repetition in range(repetition_start, repetition_start+args["repetitions"]):
            for policy in args["policies"]:
                folder = path.parent/f"{repetition:02d}-{policy}"
                expected_folders.add(folder)
                if folder not in run_folders:
                    item = {"path": str(folder.relative_to(root)),
                            "status": "in_progress_or_interrupted" if folder.exists() else "not_started"}
                    if (folder/"requests.jsonl").exists():
                        partial_rows = read_jsonl(folder/"requests.jsonl", issues)
                        manifest_path = ancestor_file(folder, "selection.json", root)
                        manifest = read_json(manifest_path, issues) if manifest_path else None
                        selected = (manifest or {}).get("selection", [])
                        estimate = {"policy": policy, "repetition": repetition,
                                    "measurement": {"offered": sum(row["phase"] == "measurement" for row in selected),
                                                    "slo_seconds": args.get("slo", 1)},
                                    "warmup": {"offered": sum(row["phase"] == "warmup" for row in selected)}}
                        partial = analyze_run(estimate, partial_rows, manifest, inspection_enabled=args.get("inspect_prefixes"))
                        item["partial_accounting"] = {key: partial[key] for key in ("population", "children", "integrity")}
                        item["partial_accounting"]["scope"] = "Observed terminals plus complete offered manifest; missing terminals remain unfinished, not successful. No terminal run summary or trusted observation/GPU horizon exists, so this entry is excluded from paired performance comparison."
                    incomplete.append(item)
    for path in sorted(root.rglob("requests.jsonl")):
        if path.parent not in run_folders and path.parent not in expected_folders:
            # Restrict to continuous benchmark records; ignore other campaigns.
            records = read_jsonl(path, issues)
            if any("phase" in row and "scheduled_seconds" in row and "policy" in row for row in records):
                incomplete.append({"path": str(path.parent.relative_to(root)), "status": "in_progress_or_interrupted"})
    declared_campaigns = []
    for path in sorted(root.rglob("campaign.json")):
        campaign = read_json(path, issues)
        if not isinstance(campaign, dict) or "specifications" not in campaign.get("contract", {}):
            continue
        specifications = campaign["contract"]["specifications"]
        recorded = {row["index"]: row for row in campaign.get("runs", [])}
        declared_campaigns.append({"path": str(path.relative_to(root)), "complete": campaign.get("complete", False),
                                   "planned_runs": len(specifications), "recorded_runs": len(recorded),
                                   "failed_runs": sum(row.get("exit_code") != 0 for row in recorded.values())})
        for index, spec in enumerate(specifications):
            state = recorded.get(index)
            default_folder = path.parent/(f"{index:03d}-{spec['split']}-s{spec['start']:g}"
                                          f"-x{spec['time_scale']:g}-r{spec['repetition']}-{spec['policy']}")
            folder = Path(state["output"]).resolve() if state and state.get("output") else default_folder
            if root not in folder.parents and folder != root:
                # Runner may have persisted a project-relative output path.
                folder = default_folder
            label = str(folder.relative_to(root))
            has_terminal = any(run_folder == folder or folder in run_folder.parents for run_folder in run_folders)
            already_incomplete = any(item["path"] == label or item["path"].startswith(label+"/") for item in incomplete)
            if state and state.get("exit_code") != 0:
                incomplete.append({"path": label, "status": "campaign_process_or_restart_failed",
                                   "exit_code": state.get("exit_code"), "error": state.get("error")})
            elif not state and not already_incomplete:
                incomplete.append({"path": label, "status": "awaiting_campaign_finalization" if has_terminal else
                                   "in_progress_or_interrupted" if folder.exists() else "not_started"})
            elif state and not has_terminal and not already_incomplete:
                incomplete.append({"path": label, "status": "campaign_success_without_terminal_summary"})
            if state and finite(state.get("restart_seconds")):
                standalone_lifecycle.append({"artifact": str(path.relative_to(root))+f"/runs/{index}",
                                             "run": label, "ready_seconds": state["restart_seconds"]})
    audit_path = root/"workload-audit"/"workload-audit.json"
    workload = read_json(audit_path, issues) if audit_path.exists() else None
    if workload:
        workload = {"path": str(audit_path), "recommended_windows": workload.get("recommended_windows"),
                    "splits": {name: {"population": item["population"], "windows": item["windows"]}
                               for name, item in workload.get("splits", {}).items()}, "kimi": workload.get("kimi")}
    ambiguity_path = root/"workload-audit"/"identity-ambiguity.json"
    ambiguity = read_json(ambiguity_path, issues) if ambiguity_path.exists() else None
    if ambiguity:
        ambiguity = {"path": str(ambiguity_path), **ambiguity}
    lifecycle = standalone_lifecycle
    for run in runs:
        if run["physical_release"]:
            release = run["physical_release"]
            lifecycle.append({"run": run["path"], **{key: release.get(key) for key in (
                "released", "graceful", "forced_kill", "shutdown_seconds", "confirmed_seconds", "gpu_uuids")}})
    lifecycle_paths = set()
    for pattern in ("*lifecycle*.json", "*release*.json", "*restart*.json"):
        lifecycle_paths.update(root.rglob(pattern))
    for path in sorted(lifecycle_paths):
        value = read_json(path, issues)
        lifecycle.extend(lifecycle_observations(value, str(path.relative_to(root))))
    declared_unfinished = any(not campaign["complete"] for campaign in declared_campaigns)
    status = "incomplete" if incomplete or declared_unfinished else "discovered_batches_finished" if runs else "no_continuous_results"
    return {"schema_version": 1, "generated_unix": time.time(), "root": str(root), "status": status,
            "campaign_completion_declared": bool(declared_campaigns) and not declared_unfinished and not incomplete,
            "declared_campaigns": declared_campaigns,
            "completion_scope": "Only discovered frozen campaign manifests/batches, not all approved experimental work.",
            "runs": runs, "incomplete_runs": incomplete,
            "artifact_issues": issues, "paired_comparisons": paired_comparisons(runs),
            "calibration": {"path": str(profile_path), "profile": profile} if profile_path else None,
            "workload": workload, "identity_ambiguity": ambiguity, "lifecycle": lifecycle,
            "interpretation": ["Run-level paired deltas are descriptive. Three repetitions do not establish significance, and requests are not independent repetitions.",
                "Windows and rates are separate conditions. Fixed retirement tests state mobility under a prescribed event, not an autoscaling policy.",
                "GPU seconds cover reported model residency only; restart and full demand-cycle costs require separate lifecycle evidence. All resident standby engines remain occupied.",
                "Native runs offer the same scheduled window but may finish draining at different times. GPU seconds and goodput therefore use run-specific observation horizons; compare SLO/failures alongside cost, and do not present the difference as savings over an equal fixed wall-clock demand cycle.",
                "Runs with fatal harness errors retain all offered requests and diagnostic metrics but are invalid policy experiments and excluded from paired comparisons; a harness failure is not evidence of an algorithm losing.",
                "The native selected-pair cold/cached numerical gate previously failed. Synthetic source/migration agreement does not repair that gate.",
                "No observed consumption is not proof of unused state; preparation/target/hit association is not causal attribution.",
                "Conditional isolated-service headroom is not a cluster performance or GPU-cost upper bound.",
                "Whole-parent certification is constrained by anonymized partial final blocks; low certification is not evidence of rare original prefix reuse. Matching complete blocks and earlier legal checkpoints can provide additional opportunity.",
                "Completion status covers discovered batches only. The full approved experimental campaign is not declared complete by this report."]}


def display(value, digits=3):
    return f"{value:.{digits}f}" if finite(value) else "—"


def focus_campaign(report, campaign_path=None):
    root = Path(report["root"])
    if campaign_path:
        path = Path(campaign_path).resolve()
        path = path/"campaign.json" if path.is_dir() else path
    else:
        paths = [root/item["path"] for item in report.get("declared_campaigns", [])]
        if not paths:
            return None
        path = max(paths, key=lambda item: json.loads(item.read_text()).get("started_unix", 0))
    state = json.loads(path.read_text())
    specifications = state.get("contract", {}).get("specifications", [])
    if not specifications:
        return None
    prefix = str(path.parent.relative_to(root))
    runs = [run for run in report["runs"] if prefix == "." or run["path"].startswith(prefix+"/")]
    valid = [run for run in runs if run.get("valid_policy_experiment")]
    groups = defaultdict(list)
    for run in valid:
        groups[(run["condition_id"], run["policy"])].append(run)
    metrics_to_summarize = {**PAIR_METRICS,
        "physical_confirmed_seconds": ("drain_components", "physical_confirmed_seconds"),
        "measurement_observation_seconds": ("measurement_observation_seconds",)}
    summary_groups = []
    for (condition, policy), values in sorted(groups.items(), key=lambda item: (item[1][0]["condition"].get("start", 0), item[0][1])):
        summary_groups.append({"condition_id": condition, "condition": values[0]["condition"], "policy": policy,
            "repetitions": sorted(run["repetition"] for run in values), "n": len(values),
            "measurement_offered_total": sum(run["population"]["offered"] for run in values),
            "measurement_failed_total": sum(run["population"]["failed_or_unfinished"] for run in values),
            "metrics": {name: stats(get_path(run, keys) for run in values) for name, keys in metrics_to_summarize.items()},
            "preparation_totals": {key: sum(run["preparation"][key] for run in values) for key in (
                "attempted", "successful", "consistent_with_consumption", "late_for_matching_child",
                "no_observed_consumption", "transfer_bytes", "reconstruction_tokens", "preparation_output_tokens")},
            "physical_releases": sum(bool((run.get("physical_release") or {}).get("released")) for run in values),
            "forced_kill_count": sum(bool(run["drain_components"]["forced_kill"]) for run in values),
            "run_paths": [run["path"] for run in values]})
    recorded = state.get("runs", [])
    finished = bool(state.get("complete")) and len(recorded) == len(specifications)
    all_valid = finished and len(valid) == len(specifications) and all(item.get("exit_code") == 0 for item in recorded)
    return {"path": str(path), "status": "complete" if all_valid else "finished_with_invalid_runs" if finished else "in_progress",
            "planned_runs": len(specifications), "recorded_process_runs": len(recorded), "valid_terminal_runs": len(valid),
            "invalid_terminal_runs": len(runs)-len(valid), "contract": state["contract"],
            "groups": summary_groups, "paired_comparisons": paired_comparisons(valid),
            "restart_seconds": stats(item.get("restart_seconds") for item in recorded),
            "metric_roles": {"primary": "1-second eligible-TTFT SLO; unchanged from frozen campaign.",
                             "secondary": "5-second SLO sensitivity; successful-request mean, p50 and p90 TTFT/E2E; source-drain/GPU occupancy diagnostics. Do not select a favorable secondary metric to declare a win."},
            "claims_scope": "Only the selected frozen campaign. Separate windows/conditions; descriptive paired repetitions. No automatic conference-worthiness or universal performance conclusion."}


def render_current_campaign(current):
    lines = ["# 本轮实验结果", ""]
    complete = current["status"] == "complete"
    lines += [f"状态：**{'已完成' if complete else '进行中，以下为阶段性记录' if current['status'] == 'in_progress' else '已结束，存在无效实验'}**。有效结果 {current['valid_terminal_runs']}/{current['planned_runs']} 轮；已记录进程结束 {current['recorded_process_runs']} 轮。", ""]
    specifications = current["contract"]["specifications"]
    first = specifications[0]
    starts = sorted({spec["start"] for spec in specifications})
    scales = sorted({spec["time_scale"] for spec in specifications})
    lines += [f"固定窗口起点 {starts} 秒；每窗 warmup {first.get('warmup')} 秒、测量 {first.get('duration')} 秒；到达速率缩放 {scales}。在测量窗口内按固定规则退出 E{first.get('retire_engine')}。比较范围为同一 DualMap 路由上的四种启发式策略，不能据此评价完整全局优化器或一般自动扩缩容。", "",
              "下表以每轮指标为单位，展示中位数 [最小值, 最大值]；请求失败数累计，延迟不把请求当独立重复样本。各窗口单独统计。", "",
              "| 窗口 / 策略 | 有效轮数 | 失败 / 提供请求 | TTFT p50 (s) | E2E p50 (s) | 1s SLO (%) | 5s SLO (%) |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def interval(value, scale=1, digits=3):
        return (f"{display(value['median']*scale,digits)} [{display(value['min']*scale,digits)}, {display(value['max']*scale,digits)}]"
                if value["n"] else "—")
    for group in current["groups"]:
        metrics = group["metrics"]
        lines.append(f"| {group['condition']['start']:g}s / {group['policy']} | {group['n']} | {group['measurement_failed_total']} / {group['measurement_offered_total']} | {interval(metrics['population_ttft_p50_seconds'])} | {interval(metrics['population_e2e_p50_seconds'])} | {interval(metrics['population_slo_1s_attainment'],100,1)} | {interval(metrics['population_slo_5s_attainment'],100,1)} |")
    lines += ["", "TTFT/SLO 从因果可执行时刻计时；原始到达时钟的延迟和 SLO 保留于完整 JSON。", "",
              "| 窗口 / 策略 | 1s SLO goodput (req/s) | 模型占用 GPU-s，含通信组准备 | 实际释放时刻 (s) | 强制终止 / 释放 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for group in current["groups"]:
        metrics = group["metrics"]
        lines.append(f"| {group['condition']['start']:g}s / {group['policy']} | {interval(metrics['population_slo_goodput'])} | {interval(metrics['model_occupied_gpu_seconds_including_group_setup'],1,1)} | {interval(metrics['physical_confirmed_seconds'],1,1)} | {group['forced_kill_count']} / {group['physical_releases']} |")
    lines += ["", "| 窗口 / 策略 | 准备尝试 / 成功 | 与实际复用一致 / 迟到 / 未观察到复用 | 复制 GiB | 重建输入 / 准备输出 tokens |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for group in current["groups"]:
        prep = group["preparation_totals"]
        lines.append(f"| {group['condition']['start']:g}s / {group['policy']} | {prep['attempted']} / {prep['successful']} | {prep['consistent_with_consumption']} / {prep['late_for_matching_child']} / {prep['no_observed_consumption']} | {prep['transfer_bytes']/1024**3:.3f} | {prep['reconstruction_tokens']} / {prep['preparation_output_tokens']} |")
    lines += ["", "复用关联要求目标、前缀、准备完成时间和实际 cache hit 一致；这不排除目标原有缓存或其他请求产生缓存。迟到与未观察到复用可能重叠。当前策略不执行重建时，重建量为零不表示复制省下了这些计算。", "",
              "| 窗口 / 变体相对 DualMap | 配对轮数 | 1s SLO 差 (百分点) | 5s SLO 差 (百分点) | GPU-s 差，含准备 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    conditions = {group["condition_id"]: group["condition"] for group in current["groups"]}
    for pair in current["paired_comparisons"]:
        metrics = pair["metrics"]
        lines.append(f"| {conditions[pair['condition_id']]['start']:g}s / {pair['policy']} | {len(pair['paired_repetitions'])} | {interval(metrics['population_slo_1s_attainment'],100,1)} | {interval(metrics['population_slo_5s_attainment'],100,1)} | {interval(metrics['model_occupied_gpu_seconds_including_group_setup'],1,1)} |")
    lines += ["", "差值为变体减 DualMap：SLO/goodput 越大越好，延迟/GPU-s 越小越好。计划三轮配对；当前实际轮数见表。分布是描述性证据，不作显著性主张。", "",
              "退出时间受 warmup 放置、退出时仍在生成的请求、元数据与准备工作、进程退出等待共同影响。完整报告分开记录这些阶段，不能把释放时刻变化全部归因于复制。各轮排空结束时刻可能不同；GPU-s 不是相同固定计费区间的成本比较，重启成本亦单独记录。", "",
              f"已观察到的重启至 ready 时间：{interval(current['restart_seconds'],1,1)} 秒（n={current['restart_seconds']['n']}）。", "",
              "范围限制：匿名 trace 的最后不足 16-token 块身份不完整；checkpoint 观察不等于锁定且可能被淘汰；native cold/cached 一致性 gate 仍未通过。未在本轮追加不退出控制、其他节点拓扑、全局 MILP 或新的 GPU 实验。这些属于后续验证，不能由当前四个启发式自动推断。", ""]
    lines += ["<details>", "<summary>次要诊断：成功请求的平均延迟与 p90 配对变化</summary>", "",
              "主要指标仍为预定的 1s TTFT SLO。以下平均值/p90 只含成功且延迟有效的请求；失败请求仍计入 SLO 分母。每项展示各轮配对差值的中位数 [最小值, 最大值]，保留恶化方向；不能挑选其中有利的指标替代主结论。原始到达时钟的相同指标保留在 JSON。", "",
              "| 窗口 / 变体相对 DualMap | 配对轮数 | 平均 TTFT 差 (s) | p90 TTFT 差 (s) | 平均 E2E 差 (s) | p90 E2E 差 (s) |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for pair in current["paired_comparisons"]:
        metric = pair["metrics"]
        lines.append(f"| {conditions[pair['condition_id']]['start']:g}s / {pair['policy']} | {len(pair['paired_repetitions'])} | {interval(metric['population_ttft_mean_seconds'])} | {interval(metric['population_ttft_p90_seconds'])} | {interval(metric['population_e2e_mean_seconds'])} | {interval(metric['population_e2e_p90_seconds'])} |")
    lines += ["", "</details>", ""]
    return "\n".join(lines)


def render_markdown(report):
    lines = ["# FluidKV decision validation", "", f"Observed status: **{report['status']}**. Completed run summaries: {len(report['runs'])}; unfinished or incomplete runs: {len(report['incomplete_runs'])}.", ""]
    lines += report["interpretation"][:1] + [""]
    fatal_runs = [run for run in report["runs"] if run.get("fatal_error") is not None]
    if fatal_runs:
        lines += [f"**Invalid harness runs: {len(fatal_runs)}.** Their offered-load denominators and diagnostic metrics are retained below, but they are excluded from policy comparisons and cannot establish algorithm performance.", ""]
    if report.get("identity_ambiguity"):
        lines += ["**Identity limitation:** "+report["identity_ambiguity"]["critical_caveat"], "",
                  "| Split | Whole parent certified | All complete blocks match, final fragment unknown | Confirmed complete-block divergence |", "| --- | ---: | ---: | ---: |"]
        for split, item in report["identity_ambiguity"]["splits"].items():
            counts = item["population"]["counts"]
            lines.append(f"| {split} | {counts['confirmed_complete_parent']} | {counts['append_compatible_unknown_tail']} | {counts['confirmed_complete_block_divergence']} |")
        lines += ["", "[Identity audit and runtime checkpoint analysis](workload-audit/identity-ambiguity.md)", ""]
    if report.get("figures"):
        for name, files in report["figures"]["figures"].items():
            lines += [f"![{name}](figures/{Path(files['png']).name})", "",
                      f"[SVG](figures/{Path(files['svg']).name}) · [Raw figure data and provenance](figures/figure-data.json)", ""]
    if report["calibration"]:
        profile = report["calibration"]["profile"] or {}
        lines += [f"Calibration: `{report['calibration']['path']}`. This calibration is an isolated service measurement; each run records the curve actually used when available.", ""]
        if "lengths" in profile:
            lines += ["| Prefix tokens | Cold TTFT (ms) | Warm TTFT (ms) | Copy idle (ms) |", "| ---: | ---: | ---: | ---: |"]
            for length, value in sorted(profile["lengths"].items(), key=lambda x: int(x[0])):
                lines.append(f"| {length} | {display(1000*value['cold_ttft_seconds'])} | {display(1000*value['warm_ttft_seconds'])} | {display(1000*value['copy_idle_seconds'])} |")
            lines += [""]
        gate = profile.get("numerical_gate")
        if gate:
            lines += [f"Numerical gate: `{json.dumps(gate, sort_keys=True)}`", ""]
    if report["workload"]:
        lines += ["| Split | Full population | Linked requests | Strict whole-parent identity certification |", "| --- | ---: | ---: | ---: |"]
        for name, data in report["workload"]["splits"].items():
            pop = data["population"]
            lines.append(f"| {name} | {pop['requests']} | {pop['linked_requests']} | {pop['snapshot_eligible_children']} |")
        lines += ["", "Original Qwen arrival gaps are not tool durations. Continuous windows retain background and report context exclusions. Kimi lacks native lineage/token identity and is used only for observed timing/length/cache characterization.", ""]
    if report["runs"]:
        lines += ["| Run | Offered / success | TTFT p50 (ms), eligible / arrival | E2E p50 (s), eligible / arrival | SLO, eligible / arrival | Child n / eligible TTFT p50 (ms) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for run in report["runs"]:
            pop, child = run["population"], run["children"]
            ms = lambda item: display(1000*item["p50"]) if finite(item["p50"]) else "—"
            status = " **[invalid harness]**" if run.get("fatal_error") is not None else ""
            lines.append(f"| {run['path']}{status} | {pop['offered']} / {pop['successful']} | {ms(pop['ttft_seconds'])} / {ms(pop['arrival_ttft_seconds'])} | {display(pop['e2e_seconds']['p50'])} / {display(pop['arrival_e2e_seconds']['p50'])} | {display(pop['slo_attainment'])} / {display(pop['arrival_slo_attainment'])} | {child['offered']} / {ms(child['ttft_seconds'])} |")
        lines += ["", "| Run | Preparations attempted / successful | Consistent with consumption / late / no observed consumption | Bytes | GPU-seconds reported / including group setup |", "| --- | ---: | ---: | ---: | ---: |"]
        for run in report["runs"]:
            prep = run["preparation"]
            lines.append(f"| {run['path']} | {prep['attempted']} / {prep['successful']} | {prep['consistent_with_consumption']} / {prep['late_for_matching_child']} / {prep['no_observed_consumption']} | {prep['transfer_bytes']} | {display(run['model_occupied_gpu_seconds'])} / {display(run['model_occupied_gpu_seconds_including_group_setup'])} |")
        lines += ["", "Late and no-observed-consumption counts can overlap. GPU figures are raw residency accounting, not a claim of cost savings or a complete restart cycle.", ""]
        with_limits = [run for run in report["runs"] if run["effective_concurrency"]["observed_effective_limits_by_engine"]]
        if with_limits:
            lines += ["| Run | Configured request limit by engine | Effective runtime limit from /v1/loads | Sampled running peak |", "| --- | --- | --- | --- |"]
            for run in with_limits:
                limits = run["effective_concurrency"]
                lines.append(f"| {run['path']} | {json.dumps(limits['configured_max_running_requests_by_engine'], sort_keys=True)} | {json.dumps(limits['observed_effective_limits_by_engine'], sort_keys=True)} | {json.dumps(limits['observed_running_peak_by_engine'], sort_keys=True)} |")
            lines += ["", "The configured limit can exceed the engine's effective limit because other resources constrain admission. Sampled peaks do not prove sustained utilization. Retired engines have fewer observations.", ""]
        with_drain = [run for run in report["runs"] if finite(run["drain_components"]["retirement_seconds"])]
        if with_drain:
            lines += ["| Run | In flight at retirement | Retirement → last foreground (s) | Foreground finish → shutdown signal (s) | Shutdown confirmation (s) | Graceful / forced kill |", "| --- | ---: | ---: | ---: | ---: | --- |"]
            for run in with_drain:
                drain = run["drain_components"]
                lines.append(f"| {run['path']} | {drain['inflight_at_retirement']} | {display(drain['retirement_to_last_foreground_seconds'])} | {display(drain['foreground_finish_to_signal_seconds'])} | {display(drain['shutdown_confirmation_seconds'])} | {drain['graceful']} / {drain['forced_kill']} |")
            lines += ["", "Source drain depends on warmup placement and in-flight output lengths. Preparation and inspection can overlap foreground work; overall release-time differences are not attributed solely to migration. JSON retains each source event boundary and paired component deltas.", ""]
    if report["paired_comparisons"]:
        lines += ["| Condition / variant | Run pairs | Eligible TTFT p50 delta (ms), median [min,max] | E2E p50 delta (s), median [min,max] |", "| --- | ---: | ---: | ---: |"]
        def interval(metric, factor=1):
            return (f"{display(factor*metric['median'])} [{display(factor*metric['min'])}, {display(factor*metric['max'])}]"
                    if metric["n"] else "—")
        for pair in report["paired_comparisons"]:
            lines.append(f"| {pair['condition_id'][:10]} / {pair['policy']} | {len(pair['paired_repetitions'])} | {interval(pair['metrics']['population_ttft_p50_seconds'], 1000)} | {interval(pair['metrics']['population_e2e_p50_seconds'])} |")
        lines += ["", "Deltas are variant minus DualMap within the same window and repetition. JSON retains every paired value, condition, run path, metric sample count and range. Accounted timeouts remain in SLO comparisons and are flagged; missing latency measurements are not invented. No request-level pseudo-replication or significance claim is made.", ""]
    if report["runs"]:
        lines += ["| Run | Checkpoint evidence mode | Conditional service cases estimated / candidates | Missing snapshot evidence | Estimated isolated service saved (s) |", "| --- | --- | ---: | ---: | ---: |"]
        for run in report["runs"]:
            headroom = run["conditional_service_headroom"]
            amount = display(headroom["estimated_service_seconds_sum"]) if headroom["estimated_cases"] else headroom["result_status"]
            lines.append(f"| {run['path']} | {headroom['mode']} | {headroom['estimated_cases']} / {headroom['candidate_cases']} | {headroom['unestimated_missing_snapshot_evidence']} | {amount} |")
        lines += ["", "For inspection runs, this diagnostic considers source candidates at child dispatch, not earlier proactive-copy opportunities. An authenticated source checkpoint must lie within the confirmed common complete-block prefix and precede dispatch. Later contradictory inspections, explicit source rejection and confirmed source release invalidate earlier evidence. Its age is retained, and actual availability at dispatch is still unproven. Missing or released-state evidence stays unestimated, not zero opportunity. Earlier runs retain a labelled strict whole-parent assumption; new runs never silently fall back to it.", "",
                  "The hypothetical checkpoint is placed at the selected destination and actual existing hits are deducted. Preparation, queueing, batching and contention are excluded; these are not cluster upper bounds. Cases beyond calibration are unestimated, and small residuals retain the smallest measured service cost. Unknown final fragments are never presumed equal.", ""]
    if report["incomplete_runs"]:
        lines += ["Unfinished or incomplete runs:", ""]
        for row in report["incomplete_runs"]:
            annotation = ""
            if row.get("partial_accounting"):
                pop = row["partial_accounting"]["population"]
                annotation = f"; measurement offered {pop['offered']}, successful terminals {pop['successful']}, failed/unfinished {pop['failed_or_unfinished']} (no final timing/GPU summary)"
            lines.append(f"- `{row['path']}`: {row['status']}{annotation}")
        lines += [""]
    if report["lifecycle"]:
        lines += ["| Lifecycle evidence | Confirmed released | Graceful | Shutdown (s) | Restart ready (s) |", "| --- | --- | --- | ---: | ---: |"]
        for event in report["lifecycle"]:
            lines.append(f"| {event.get('run', event.get('artifact'))} | {event.get('released', '—')} | {event.get('graceful', '—')} | {display(event.get('shutdown_seconds'))} | {display(event.get('ready_seconds'))} |")
        lines += ["", "Restart durations are observed launch-to-readiness intervals. A release observation alone does not establish savings over a complete demand cycle.", ""]
    if report["artifact_issues"]:
        lines += [f"Artifact read/format issues: {len(report['artifact_issues'])}; paths and errors are retained in JSON.", ""]
    lines += report["interpretation"][1:] + [""]
    if report.get("current_campaign"):
        return render_current_campaign(report["current_campaign"])+"\n\n## 历史控制、机制验证与完整明细\n\n"+"\n".join(lines)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="Profile JSON or directory; default production-calibration-local when available, otherwise newest decision_profile.json below root")
    parser.add_argument("--output", type=Path, help="Report directory; default root. Only generated report files are replaced.")
    parser.add_argument("--plots", action="store_true", help="Generate PNG/SVG figures and figure-data.json beside the report")
    parser.add_argument("--campaign", type=Path, help="Highlight only this frozen campaign; default latest declared campaign. Historical runs remain separate.")
    args = parser.parse_args()
    report = build_report(args.root, args.calibration)
    report["current_campaign"] = focus_campaign(report, args.campaign)
    output = args.output or args.root
    output.mkdir(parents=True, exist_ok=True)
    if args.plots:
        from experiments.fluidkv.plot_decision_validation import generate_figures
        report["figures"] = generate_figures(report, output/"figures")
    (output/"decision-validation-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    (output/"RESULTS.md").write_text(render_markdown(report))
    if report["current_campaign"]:
        (output/"current-campaign-results.json").write_text(json.dumps(report["current_campaign"], indent=2, allow_nan=False)+"\n")
        (output/"CURRENT_CAMPAIGN_RESULTS.md").write_text(render_current_campaign(report["current_campaign"]))
    print(json.dumps({"status": report["status"], "runs": len(report["runs"]),
                      "incomplete": len(report["incomplete_runs"]), "report": str(output/"RESULTS.md")}))


if __name__ == "__main__":
    main()
