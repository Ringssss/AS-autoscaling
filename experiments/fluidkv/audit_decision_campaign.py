#!/usr/bin/env python3
"""Independent CPU audit of a frozen campaign's saved artifacts, never engines."""

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import math
from pathlib import Path

from agentshift.workloads.production import surrogate_tokens
from experiments.fluidkv.report_decision_validation import condition_for, stable_digest


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_campaign(campaign_dir, runtime_source_record=None):
    campaign_dir = Path(campaign_dir).resolve()
    project = Path(__file__).resolve().parents[2]
    state = read(campaign_dir/"campaign.json")
    specs = state["contract"]["specifications"]
    recorded = {row["index"]: row for row in state.get("runs", [])}
    checks, warnings, failures = [], [], []
    source_record = campaign_dir/"serving-source-sha256.json"
    source_checks = []
    if source_record.exists():
        for name, expected in read(source_record).items():
            path = project/name
            actual = digest(path) if path.exists() else None
            source_checks.append({"path": name, "expected_sha256": expected, "current_sha256": actual,
                                  "matches_frozen_record": actual == expected})
    else:
        warnings.append("Campaign has no serving-source-sha256.json.")
    if runtime_source_record:
        for name, expected in read(Path(runtime_source_record)).get("files", {}).items():
            if name.startswith("sglang-hybrid-state/"):
                path = project.parent/name
                actual = digest(path) if path.exists() else None
                source_checks.append({"path": name, "expected_sha256": expected, "current_sha256": actual,
                                      "matches_frozen_record": actual == expected})
    if any(not row["matches_frozen_record"] for row in source_checks):
        failures.append("Current serving/runtime source differs from supplied frozen source evidence.")
    warnings.append("Source checks compare final filesystem bytes with frozen manifests. They do not retroactively attest the bytes loaded by each historical process; no per-process loaded-source hashes are available.")
    selections, conditions, token_fingerprints = defaultdict(set), defaultdict(set), {}
    for index, spec in enumerate(specs):
        run_record = recorded.get(index)
        folder = campaign_dir/(f"{index:03d}-{spec['split']}-s{spec['start']:g}-x{spec['time_scale']:g}-r{spec['repetition']}-{spec['policy']}")
        run = folder/f"{spec['repetition']:02d}-{spec['policy']}"
        summary_path = run/"summary.json"
        check = {"index": index, "specification": spec, "path": str(run.relative_to(campaign_dir)), "checks": {}}
        if not summary_path.exists():
            check["status"] = "not_started" if not folder.exists() else "incomplete"
            checks.append(check)
            continue
        if run_record is None:
            check["status"] = "awaiting_campaign_finalization"
            checks.append(check)
            continue
        args, manifest, summary = read(folder/"arguments.json"), read(folder/"selection.json"), read(summary_path)
        rows = [json.loads(line) for line in (run/"requests.jsonl").read_text().splitlines() if line.strip()]
        selected = {row["chat_id"]: row for row in manifest["selection"]}
        ids = [row["chat_id"] for row in rows]
        counts = Counter(ids)
        errors = check["checks"]
        errors["all_selected_terminal_exactly_once"] = set(ids) == set(selected) and all(count == 1 for count in counts.values())
        errors["policy_repetition_match"] = all(row.get("policy") == spec["policy"] and row.get("repetition") == spec["repetition"] for row in rows)
        errors["summary_valid"] = summary.get("fatal_error") is None and not summary.get("timed_out", False)
        errors["process_succeeded"] = run_record is not None and run_record.get("exit_code") == 0
        errors["frozen_run_arguments_match"] = all(args.get(key) == spec.get(key) for key in ("start", "duration", "warmup", "time_scale", "retire_engine", "retire_offset"))
        errors["context_and_input_lengths_preserved"] = True
        errors["successful_output_lengths_preserved"] = True
        errors["prompt_sha_matches_selected_surrogate"] = True
        errors["phase_and_clock_match"] = True
        for row in rows:
            original = selected.get(row["chat_id"])
            if original is None:
                continue
            errors["context_and_input_lengths_preserved"] &= (row["input_tokens"] == original["input_length"]
                and row["requested_output_tokens"] == original["output_length"]
                and original["input_length"]+original["output_length"] <= manifest["context_length"])
            if row.get("success"):
                errors["successful_output_lengths_preserved"] &= row["completion_tokens"] == original["output_length"]
            if row.get("prompt_token_sha256"):
                identity = stable_digest([original["input_length"], original["hash_ids"]])
                if identity not in token_fingerprints:
                    token_fingerprints[identity] = hashlib.sha256(json.dumps(surrogate_tokens(original)).encode()).hexdigest()
                errors["prompt_sha_matches_selected_surrogate"] &= token_fingerprints[identity] == row["prompt_token_sha256"]
            expected_schedule = (original["timestamp"]-manifest["replay_start_timestamp"])/args["time_scale"]
            errors["phase_and_clock_match"] &= row.get("phase") == original["phase"] and math.isclose(row["scheduled_seconds"], expected_schedule, abs_tol=2e-6)
            if row.get("success"):
                errors["phase_and_clock_match"] &= math.isclose(row["arrival_ttft_seconds"], row["ttft_seconds"]+row["eligibility_delay_seconds"], abs_tol=2e-6)
        for phase in ("warmup", "measurement"):
            offered = sum(row["phase"] == phase for row in selected.values())
            successful = [row for row in rows if row["phase"] == phase and row.get("success")]
            errors[phase+"_denominators_match"] = summary[phase]["offered"] == offered and summary[phase]["successful"] == len(successful)
            for slo in (1.0, 5.0):
                observed = summary[phase].get("slo_sensitivity", {}).get(str(slo))
                if observed is not None:
                    expected = sum(row["ttft_seconds"] <= slo for row in successful)/offered if offered else None
                    errors[phase+f"_slo_{slo}_matches"] = math.isclose(observed, expected, abs_tol=1e-12)
        metadata_issues = []
        condition_id, condition = condition_for(args, manifest, folder, campaign_dir, metadata_issues)
        check["backend"] = condition["backend"]
        group_key = (spec["split"], spec["start"], spec["duration"], spec["warmup"], spec["time_scale"])
        selections[group_key].add(stable_digest(manifest["selection"]))
        conditions[group_key].add(condition_id)
        check["selection_sha256"] = stable_digest(manifest["selection"])
        curve_path = Path(args["costs"])
        curve_path = curve_path if curve_path.is_absolute() else project/curve_path
        errors["calibration_file_matches_frozen_hash"] = digest(curve_path) == state["contract"]["costs_sha256"]
        errors["configured_pool_matches_contract"] = args["cache_tokens"] == state["contract"]["cache_tokens"]
        errors["common_backend_within_run"] = len({stable_digest(item) for item in condition["backend"]}) == 1
        release = summary.get("physical_release") or {}
        errors["physical_release_confirmed"] = bool(release.get("released")) if state["contract"].get("physical_release") else True
        elapsed, setup = summary["duration_seconds"], summary.get("group_setup_seconds", 0)
        release_at = release.get("confirmed_seconds") if release.get("released") else elapsed
        charged = len(args["engines"])*(elapsed+setup)-max(0,elapsed-release_at)
        errors["gpu_occupancy_equation_matches"] = math.isclose(charged, summary["model_occupied_gpu_seconds_including_group_setup"], abs_tol=1e-6)
        full_window = (args["warmup"]+args["duration"])/args["time_scale"]
        errors["full_offered_window_observed"] = elapsed >= full_window or summary.get("fatal_error") is not None or summary.get("timed_out")
        errors["observation_horizon_matches"] = math.isclose(summary["measurement_observation_seconds"], max(full_window,elapsed)-args["warmup"]/args["time_scale"], abs_tol=1e-6)
        errors["restart_ready_recorded"] = not state["contract"].get("physical_release") or (run_record is not None and isinstance(run_record.get("restart_seconds"),(int,float)) and run_record["restart_seconds"] > 0)
        check.update(status="pass" if all(errors.values()) else "failed_checks",
                     measurement_offered=summary["measurement"]["offered"],
                     measurement_successful=summary["measurement"]["successful"],
                     forced_kill=release.get("forced_kill"), graceful=release.get("graceful"))
        if check["status"] != "pass":
            failures.append({"index": index, "failed_checks": [key for key,value in errors.items() if not value]})
        checks.append(check)
    consistency = [{"window": list(key), "selection_hashes": sorted(selections[key]),
                    "condition_hashes": sorted(conditions[key]), "same_selection": len(selections[key]) == 1,
                    "same_condition": len(conditions[key]) == 1} for key in sorted(selections)]
    if any(not row["same_selection"] or not row["same_condition"] for row in consistency):
        failures.append("Selection or configured condition differs within a fixed window.")
    finished = bool(state.get("complete")) and len(recorded) == len(specs)
    return {"generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "campaign": str(campaign_dir),
            "status": "failed_checks" if failures else "pass" if finished and all(row["status"] == "pass" for row in checks) else "incomplete",
            "planned_runs": len(specs), "recorded_process_runs": len(recorded), "terminal_audits": checks,
            "within_window_consistency": consistency, "source_checks": source_checks,
            "per_process_loaded_source_identity_proven": False, "warnings": warnings, "failures": failures,
            "scope": "Saved-artifact consistency audit only. No GPU or engine endpoint access. Request failures remain failures; a physical forced shutdown is reported rather than recast as graceful. Equal offered windows can have unequal drain/occupancy horizons."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--runtime-source-record", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_campaign(args.campaign, args.runtime_source_record)
    output = args.output or args.campaign/"independent-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": result["status"], "planned": result["planned_runs"],
                      "recorded": result["recorded_process_runs"], "failures": result["failures"], "output": str(output)}))


if __name__ == "__main__":
    main()
