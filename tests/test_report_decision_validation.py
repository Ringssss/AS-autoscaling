import json

import pytest

from agentshift.controller.preparation_decisions import CostCurve
from experiments.fluidkv.report_decision_validation import analyze_run, build_report, condition_for, drain_components, effective_concurrency, paired_comparisons


def selection(parent_length=32, parent_hashes=None, child_hashes=None):
    return {"start_seconds": 10, "duration_seconds": 10, "warmup_seconds": 10,
            "context_length": 32768, "selection": [
        dict(chat_id=1, parent_chat_id=-1, phase="warmup", input_length=parent_length,
             output_length=1, hash_ids=parent_hashes or [1, 2], timestamp=1),
        dict(chat_id=2, parent_chat_id=1, phase="measurement", input_length=48,
             output_length=1, hash_ids=child_hashes or [1, 2, 3], timestamp=11),
        dict(chat_id=3, parent_chat_id=-1, phase="measurement", input_length=16,
             output_length=1, hash_ids=[4], timestamp=12)]}


def request(chat, parent, *, phase="measurement", target=1, hit=0, dispatch=2, success=True, policy="dualmap"):
    return dict(chat_id=chat, parent_chat_id=parent, policy=policy, repetition=0,
                phase=phase, target=target, input_tokens=48 if chat == 2 else 32,
                actual_cached_tokens=hit, dispatched_seconds=dispatch,
                success=success, ttft_seconds=0.2 if success else None,
                arrival_ttft_seconds=0.5 if success else None,
                e2e_seconds=1 if success else None, arrival_e2e_seconds=1.3 if success else None,
                completion_tokens=int(success), preparation_wait_seconds=0,
                router_wait_seconds=0.01, finished_seconds=dispatch+1)


def summary(preparations=None):
    return dict(policy="dualmap", repetition=0, timed_out=False,
                measurement={"offered": 2, "slo_seconds": 0.3}, warmup={"offered": 1},
                measurement_observation_seconds=10, model_occupied_gpu_seconds=60,
                preparations=preparations or [])


def test_report_uses_manifest_to_keep_missing_terminal_requests():
    records = [request(1, -1, phase="warmup", target=0), request(2, 1)]
    result = analyze_run(summary(), records, selection())
    assert result["population"]["offered"] == 2
    assert result["population"]["successful"] == 1
    assert result["population"]["slo_attainment"] == 0.5
    assert result["population"]["arrival_slo_attainment"] == 0
    assert result["integrity"]["missing_terminal_chat_ids"] == [3]
    assert result["children"]["offered"] == 1


def test_preparation_consumption_requires_target_ready_time_and_real_hit():
    moves = [dict(chat_id=1, source=0, target=1, input_tokens=32, success=True,
                  attempted=True, finish_seconds=1, transfer_bytes=100),
             dict(chat_id=1, source=0, target=1, input_tokens=32, success=True,
                  attempted=True, finish_seconds=3, transfer_bytes=100),
             dict(chat_id=1, source=0, target=0, input_tokens=32, success=True,
                  attempted=True, finish_seconds=1, transfer_bytes=100)]
    records = [request(1, -1, phase="warmup", target=0), request(2, 1, hit=32)]
    result = analyze_run(summary(moves), records, selection())
    assert result["preparation"]["consistent_with_consumption"] == 1
    assert result["preparation"]["late_for_matching_child"] == 1
    assert result["preparation"]["no_observed_consumption"] == 2
    assert result["preparation"]["transfer_bytes"] == 300


def test_partial_prefix_never_becomes_full_parent_headroom():
    records = [request(1, -1, phase="warmup", target=0), request(2, 1, hit=16)]
    curve = CostCurve(((16, 0.1), (32, 0.2), (64, 0.4)))
    result = analyze_run(summary(), records, selection(18, [1, 2], [1, 9, 3]), curve)
    assert result["children"]["exact_parent_extensions"] == 0
    assert not result["cold_reroute_cases"]  # All 16 verified shared tokens hit.
    assert result["conditional_service_headroom"]["estimated_cases"] == 0
    assert result["conditional_service_headroom"]["source_availability_proven"] is False


def test_headroom_is_incremental_to_actual_hits_and_not_cluster_bound():
    records = [request(1, -1, phase="warmup", target=0), request(2, 1, hit=16)]
    curve = CostCurve(((16, 0.1), (32, 0.2), (64, 0.4)))
    result = analyze_run(summary(), records, selection(), curve)
    headroom = result["conditional_service_headroom"]
    assert headroom["estimated_cases"] == 1
    assert headroom["estimated_service_seconds_sum"] == pytest.approx(0.1)
    assert headroom["is_cluster_upper_bound"] is False
    assert result["cold_reroute_cases"][0]["verified_missing_parent_tokens"] == 16


def test_parent_from_other_policy_cannot_form_reroute_case():
    records = [request(1, -1, phase="warmup", target=0, policy="full-copy"), request(2, 1)]
    result = analyze_run(summary(), records, selection())
    assert not result["cold_reroute_cases"]
    assert result["integrity"]["foreign_policy_records"] == 1


def test_pairing_uses_run_units_preserves_repetitions_and_separates_conditions():
    runs = []
    for repetition in range(3):
        for policy, latency in (("dualmap", 0.3), ("full-copy", 0.2 + repetition * 0.01)):
            runs.append(dict(condition_id="same-window", policy=policy, repetition=repetition,
                             path=f"{repetition}-{policy}", eligible_for_pairing=True,
                             population={"ttft_seconds": {"p50": latency}},
                             children={}, model_occupied_gpu_seconds=None))
    runs.append(dict(condition_id="other-window", policy="full-copy", repetition=0,
                     path="unpaired", eligible_for_pairing=True, population={}, children={}))
    pairs = paired_comparisons(runs)
    assert len(pairs) == 1
    assert pairs[0]["paired_repetitions"] == [0, 1, 2]
    assert pairs[0]["metrics"]["population_ttft_p50_seconds"]["n"] == 3
    assert pairs[0]["metrics"]["population_ttft_p50_seconds"]["median"] == pytest.approx(-0.09)


def test_empty_or_partial_campaign_never_reports_complete(tmp_path):
    empty = build_report(tmp_path)
    assert empty["status"] == "no_continuous_results"
    folder = tmp_path / "window"
    folder.mkdir()
    (folder / "arguments.json").write_text(json.dumps({"policies": ["dualmap", "full-copy"],
        "repetitions": 1, "start": 10, "duration": 10}))
    (folder / "selection.json").write_text(json.dumps(selection()))
    run = folder / "00-dualmap"
    run.mkdir()
    (run / "requests.jsonl").write_text(json.dumps(request(2, 1)) + "\n" + '{"incomplete":')
    report = build_report(tmp_path)
    assert report["status"] == "incomplete"
    assert len(report["incomplete_runs"]) == 2
    assert any(item["status"] == "in_progress_or_interrupted" for item in report["incomplete_runs"])
    partial = next(item for item in report["incomplete_runs"] if item["status"] == "in_progress_or_interrupted")
    assert partial["partial_accounting"]["population"]["offered"] == 2
    assert partial["partial_accounting"]["population"]["successful"] == 1
    assert partial["partial_accounting"]["population"]["failed_or_unfinished"] == 1


def test_discovery_avoids_aggregate_summary_double_counting(tmp_path):
    folder = tmp_path / "window"
    run = folder / "00-dualmap"
    run.mkdir(parents=True)
    (folder / "arguments.json").write_text(json.dumps({"policies": ["dualmap"], "repetitions": 1,
                                                       "start": 10, "duration": 10}))
    manifest = selection()
    (folder / "selection.json").write_text(json.dumps(manifest))
    data = summary()
    (run / "summary.json").write_text(json.dumps(data))
    (folder / "summary.json").write_text(json.dumps({"runs": [data]}))
    records = [request(1, -1, phase="warmup", target=0), request(2, 1), request(3, -1)]
    (run / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    report = build_report(tmp_path)
    assert len(report["runs"]) == 1
    assert report["status"] == "discovered_batches_finished"
    assert report["campaign_completion_declared"] is False


def test_summary_without_manifest_or_request_rows_is_incomplete(tmp_path):
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "summary.json").write_text(json.dumps(summary()))
    report = build_report(tmp_path)
    assert report["status"] == "incomplete"
    assert report["runs"][0]["integrity"]["terminal_count_shortfall"] == 3
    assert not report["runs"][0]["eligible_for_pairing"]


def test_standalone_lifecycle_summarizes_observed_restart_duration(tmp_path):
    lifecycle = {"release": {"released": True, "graceful": True, "shutdown_seconds": 1.5},
                 "restart": {"started_monotonic": 100, "ready_monotonic": 145, "gpu_uuids": ["gpu"]}}
    (tmp_path / "lifecycle-smoke.json").write_text(json.dumps(lifecycle))
    report = build_report(tmp_path)
    assert len(report["lifecycle"]) == 2
    assert next(item for item in report["lifecycle"] if "ready_seconds" in item)["ready_seconds"] == 45
    assert report["status"] == "no_continuous_results"


def test_accounted_timeout_retains_slo_pair_instead_of_success_only_selection():
    runs = [dict(condition_id="window", policy="dualmap", repetition=0, path="baseline",
                 eligible_for_pairing=True, timed_out=False,
                 population={"slo_attainment": 1.0}),
            dict(condition_id="window", policy="full-copy", repetition=0, path="timeout",
                 eligible_for_pairing=True, timed_out=True,
                 population={"slo_attainment": 0.25})]
    pair = paired_comparisons(runs)[0]
    assert pair["timed_out_repetitions"] == [0]
    assert pair["metrics"]["population_slo_attainment"]["median"] == -0.75


@pytest.mark.parametrize("as_list", [False, True])
def test_backend_condition_ignores_process_identity_but_preserves_configuration(tmp_path, as_list):
    args = {"start": 10, "duration": 10}
    ids = []
    for index, (pid, memory, graph) in enumerate([(101, 500, False), (202, 700, False), (202, 700, True)]):
        folder = tmp_path / str(index)
        folder.mkdir()
        server = {"pid": pid, "memory_used": memory, "model_path": "qwen",
                  "server_args": {"attention_backend": "fa3", "disable_cuda_graph": graph}}
        (folder / "server_info.json").write_text(json.dumps([server] if as_list else server))
        identity, detail = condition_for(args, selection(), folder, tmp_path, [])
        ids.append(identity)
        assert "pid" not in json.dumps(detail["backend"])
        assert "memory_used" not in json.dumps(detail["backend"])
    assert ids[0] == ids[1]
    assert ids[1] != ids[2]


def test_runner_manifest_tracks_future_specs_and_nonzero_repetition_start(tmp_path):
    spec = dict(split="Coder", start=3000, time_scale=1, repetition=1, policy="dualmap")
    folder = tmp_path / "000-Coder-s3000-x1-r1-dualmap"
    run = folder / "01-dualmap"
    run.mkdir(parents=True)
    args = dict(start=3000, duration=10, policies=["dualmap"], repetitions=1, repetition_start=1)
    (folder / "arguments.json").write_text(json.dumps(args))
    (folder / "selection.json").write_text(json.dumps(selection()))
    data = summary()
    data["repetition"] = 1
    (run / "summary.json").write_text(json.dumps(data))
    records = [request(1, -1, phase="warmup", target=0), request(2, 1), request(3, -1)]
    for row in records:
        row["repetition"] = 1
    (run / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
    campaign = {"complete": False, "contract": {"specifications": [spec, {**spec, "policy": "full-copy"}]},
                "runs": [{"index": 0, "exit_code": 0, "output": str(folder), "specification": spec}]}
    (tmp_path / "campaign.json").write_text(json.dumps(campaign))
    report = build_report(tmp_path)
    assert report["status"] == "incomplete"
    assert len(report["runs"]) == 1
    assert len(report["incomplete_runs"]) == 1
    assert report["incomplete_runs"][0]["path"].endswith("r1-full-copy")
    assert report["declared_campaigns"][0]["planned_runs"] == 2


def test_figure_uses_per_pair_full_copy_cost_and_retains_noisy_point():
    from experiments.fluidkv.plot_decision_validation import calibration_points
    records = []
    for repetition, (copy, child) in enumerate([(0.01, 0.3), (0.4, 0.02), (0.1, 0.1)]):
        records.append(dict(phase="measured", success=True, prefix_tokens=256, suffix_tokens=64,
                            repetition=repetition, results={
            "cold-reroute": {"child": {"ttft_seconds": 0.2}},
            "sticky": {"child": {"ttft_seconds": 0.08}},
            "prepared-full-copy": {"preparation_wall_seconds": copy, "child": {"ttft_seconds": child}}}))
    points, excluded = calibration_points(records)
    assert len(points) == 3 and not excluded
    assert [point["copy_plus_child"] for point in points] == pytest.approx([0.31, 0.42, 0.2])
    assert all(not point["included_in_median_guide"] for point in points)


def test_figure_does_not_substitute_synthetic_smoke_for_native_pairs():
    from experiments.fluidkv.plot_decision_validation import native_pairs
    report = {"runs": [
        {"condition_id": "smoke", "condition": {"trace": "/tmp/smoke.jsonl"}},
        {"condition_id": "native", "condition": {"trace": "/data/qwen_coder_blksz_16.jsonl"}}],
        "paired_comparisons": [{"condition_id": "smoke"}, {"condition_id": "native"}]}
    assert [pair["condition_id"] for pair in native_pairs(report)] == ["native"]


def inspection_records_and_manifest():
    import hashlib
    from agentshift.workloads.production import surrogate_tokens
    manifest = selection(70, [1, 2, 3, 4, 5], [1, 2, 3, 4, 9, 10])
    manifest["selection"][1]["input_length"] = 90
    records = [request(1, -1, phase="warmup", target=0), request(2, 1, hit=16)]
    records[0]["input_tokens"], records[1]["input_tokens"] = 70, 90
    for record, selected in zip(records, manifest["selection"]):
        record["prompt_token_sha256"] = hashlib.sha256(json.dumps(surrogate_tokens(selected)).encode()).hexdigest()
    return records, manifest


def prefix_event(record, boundaries, *, seconds=1, phase="prefill", query_tokens=None):
    query_tokens = record["input_tokens"] if query_tokens is None else query_tokens
    return dict(event="prefix_inspection", engine=record["target"], chat_id=record["chat_id"],
                query_tokens=query_tokens, phase=phase, query_token_sha256=record["prompt_token_sha256"],
                seconds=seconds, wall_seconds=0.001, result=dict(query_tokens=query_tokens,
                    matched_tokens=max(boundaries, default=0), snapshot_boundaries=boundaries,
                    available_token_slots=1, evictable_token_slots=2,
                    available_snapshot_slots=1, evictable_snapshot_slots=2))


def test_partial_identity_uses_observed_earlier_boundary_not_parent_tail():
    records, manifest = inspection_records_and_manifest()
    event = prefix_event(records[0], [32, 64])
    curve = CostCurve(((16, 0.1), (128, 0.8)))
    result = analyze_run(summary(), records, manifest, curve, events=[event], inspection_enabled=True)
    case = result["cold_reroute_cases"][0]
    assert case["verified_common_prefix_tokens"] == 64
    assert case["lost_known_prefix_tokens"] == 48
    assert case["source_snapshot"]["observed_reusable_tokens"] == 64
    assert case["source_snapshot"]["observation_age_at_dispatch_seconds"] == 1
    headroom = result["conditional_service_headroom"]
    assert headroom["mode"] == "engine_observed_checkpoints"
    assert headroom["estimated_cases"] == 1
    assert headroom["cases"][0]["restorable_prefix_tokens"] == 64
    assert headroom["cases"][0]["remaining_input_after"] == 26
    assert headroom["estimated_service_seconds_sum"] == pytest.approx(0.3)
    assert not headroom["source_availability_proven"]


@pytest.mark.parametrize("change", ["wrong_hash", "future", "wrong_engine", "wrong_chat", "malformed"])
def test_invalid_or_future_inspection_cannot_establish_headroom(change):
    records, manifest = inspection_records_and_manifest()
    event = prefix_event(records[0], [64])
    if change == "wrong_hash": event["query_token_sha256"] = "wrong"
    if change == "future": event["seconds"] = 3
    if change == "wrong_engine": event["engine"] = 1
    if change == "wrong_chat": event["chat_id"] = 999
    if change == "malformed": event["result"]["snapshot_boundaries"] = [64, 32]
    result = analyze_run(summary(), records, manifest, CostCurve(((16, 0.1), (128, 0.8))),
                         events=[event], inspection_enabled=True)
    assert result["conditional_service_headroom"]["estimated_cases"] == 0
    assert result["conditional_service_headroom"]["unestimated_missing_snapshot_evidence"] == 1


def test_newer_empty_observation_invalidates_previous_source_snapshot():
    records, manifest = inspection_records_and_manifest()
    events = [prefix_event(records[0], [64], seconds=1), prefix_event(records[0], [], seconds=1.5)]
    result = analyze_run(summary(), records, manifest, CostCurve(((16, 0.1), (128, 0.8))),
                         events=events, inspection_enabled=True)
    assert result["cold_reroute_cases"][0]["source_snapshot"]["observed_reusable_tokens"] == 0
    assert result["conditional_service_headroom"]["estimated_cases"] == 0


def test_prepared_short_query_hash_can_confirm_earlier_parent_boundary():
    import hashlib
    from agentshift.workloads.production import surrogate_tokens
    records, manifest = inspection_records_and_manifest()
    event = prefix_event(records[0], [32], phase="prepared", query_tokens=32)
    event["query_token_sha256"] = hashlib.sha256(json.dumps(surrogate_tokens(manifest["selection"][0])[:32]).encode()).hexdigest()
    result = analyze_run(summary(), records, manifest, CostCurve(((16, 0.1), (128, 0.8))),
                         events=[event], inspection_enabled=True)
    assert result["conditional_service_headroom"]["cases"][0]["restorable_prefix_tokens"] == 32


def test_inspection_enabled_no_evidence_cannot_fall_back_to_whole_parent_assumption():
    records = [request(1, -1, phase="warmup", target=0), request(2, 1)]
    result = analyze_run(summary(), records, selection(), CostCurve(((16, 0.1), (64, 0.4))),
                         events=[], inspection_enabled=True)
    assert result["conditional_service_headroom"]["estimated_cases"] == 0
    assert result["conditional_service_headroom"]["mode"] == "engine_observed_checkpoints"


def test_child_routing_observation_rechecks_earlier_source_snapshot():
    records, manifest = inspection_records_and_manifest()
    event = prefix_event(records[1], [32], phase="routing", seconds=1.5)
    event["engine"] = 0
    events = [prefix_event(records[0], [64], seconds=1), event]
    result = analyze_run(summary(), records, manifest, CostCurve(((16, 0.1), (128, 0.8))),
                         events=events, inspection_enabled=True)
    case = result["cold_reroute_cases"][0]
    assert case["source_snapshot"]["observed_reusable_tokens"] == 32
    assert case["source_snapshot"]["query_phase"] == "routing"


def test_failed_source_inspection_clears_prior_availability_evidence():
    records, manifest = inspection_records_and_manifest()
    events = [prefix_event(records[0], [64], seconds=1),
              dict(event="prefix_inspection_failed", engine=0, seconds=1.5)]
    result = analyze_run(summary(), records, manifest, CostCurve(((16, 0.1), (128, 0.8))),
                         events=events, inspection_enabled=True)
    assert result["conditional_service_headroom"]["estimated_cases"] == 0


def test_prepared_earlier_boundary_consumption_does_not_require_whole_parent():
    records, manifest = inspection_records_and_manifest()
    records[1]["actual_cached_tokens"] = 64
    moves = [dict(chat_id=1, source=0, target=1, input_tokens=64, success=True,
                  attempted=True, finish_seconds=1, transfer_bytes=100)]
    result = analyze_run(summary(moves), records, manifest, events=[], inspection_enabled=True)
    assert result["preparation"]["consistent_with_consumption"] == 1
    assert result["children"]["exact_parent_extensions"] == 0


def test_inspection_and_checkpoint_backend_configuration_separate_conditions(tmp_path):
    identities = []
    for index, (inspect, observation, strategy) in enumerate([
            (False, "prefill", "no_buffer"), (True, "prefill", "no_buffer"),
            (True, "completion", "no_buffer"), (True, "prefill", "extra_buffer")]):
        folder = tmp_path / str(index)
        folder.mkdir()
        (folder / "server_info.json").write_text(json.dumps({"model_path": "qwen",
            "mamba_scheduler_strategy": strategy, "disable_overlap_schedule": True}))
        args = dict(start=10, inspect_prefixes=inspect, source_observation=observation)
        identities.append(condition_for(args, selection(), folder, tmp_path, [])[0])
    assert len(set(identities)) == 4


def test_report_discovers_raw_inspection_events_for_partial_checkpoint(tmp_path):
    folder = tmp_path / "window"
    run = folder / "00-dualmap"
    run.mkdir(parents=True)
    records, manifest = inspection_records_and_manifest()
    (folder / "arguments.json").write_text(json.dumps(dict(start=10, duration=10, inspect_prefixes=True,
        policies=["dualmap"], repetitions=1)))
    (folder / "selection.json").write_text(json.dumps(manifest))
    (folder / "cost_curves.json").write_text(json.dumps({"cold": [[16, 0.1], [128, 0.8]]}))
    (run / "summary.json").write_text(json.dumps(summary()))
    (run / "requests.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records))
    (run / "events.jsonl").write_text(json.dumps(prefix_event(records[0], [64]))+"\n")
    report = build_report(tmp_path)
    measured = report["runs"][0]
    assert measured["prefix_inspection_events"] == 1
    assert measured["conditional_service_headroom"]["estimated_cases"] == 1
    assert measured["conditional_service_headroom"]["cases"][0]["restorable_prefix_tokens"] == 64
    assert report["status"] == "incomplete"  # Missing background terminal remains visible.


def test_effective_concurrency_reports_runtime_limit_without_relabeling_configured_limit():
    events = [dict(event="engine_load", seconds=10, measurements=[
        {"engine": 0, "response": {"loads": [{"max_running_requests": 12, "num_running_reqs": 7}]}},
        {"engine": 1, "response": {"loads": [{"max_running_requests": 12, "num_running_reqs": 12}]}}]),
        dict(event="engine_load", seconds=20, measurements=[
        {"engine": 1, "response": {"loads": [{"max_running_requests": 12, "num_running_reqs": 4}]}}])]
    result = effective_concurrency(events, [{"max_running_requests": 16}, {"max_running_requests": 16}])
    assert result["configured_max_running_requests_by_engine"] == {"0": 16, "1": 16}
    assert result["observed_effective_limits_by_engine"] == {"0": [12], "1": [12]}
    assert result["observed_running_peak_by_engine"] == {"0": 7, "1": 12}
    assert result["sample_count_by_engine"] == {"0": 1, "1": 2}


def test_pairing_separates_effective_runtime_limits_even_when_config_matches(tmp_path):
    identities = []
    for index, effective in enumerate((12, 16)):
        folder = tmp_path/str(index)
        folder.mkdir()
        (folder/"server_info.json").write_text(json.dumps([{"model_path": "qwen", "max_running_requests": 16,
            "internal_states": [{"effective_max_running_requests_per_dp": effective}]}]))
        identities.append(condition_for({"start": 10}, selection(), folder, tmp_path, [])[0])
    assert identities[0] != identities[1]


def test_fatal_harness_run_keeps_full_denominator_but_is_not_policy_comparison(tmp_path):
    folder = tmp_path/"window"
    run = folder/"00-dualmap"
    run.mkdir(parents=True)
    (folder/"arguments.json").write_text(json.dumps(dict(start=10,duration=10,policies=["dualmap"],repetitions=1)))
    (folder/"selection.json").write_text(json.dumps(selection()))
    data = summary()
    data["fatal_error"] = "control cleanup accounting failed"
    (run/"summary.json").write_text(json.dumps(data))
    records = [request(1,-1,phase="warmup",target=0), request(2,1), request(3,-1,success=False)]
    (run/"requests.jsonl").write_text("".join(json.dumps(row)+"\n" for row in records))
    report = build_report(tmp_path)
    diagnostic = report["runs"][0]
    assert diagnostic["population"]["offered"] == 2
    assert diagnostic["population"]["failed_or_unfinished"] == 1
    assert diagnostic["population"]["slo_attainment"] == 0.5
    assert not diagnostic["eligible_for_pairing"]
    assert diagnostic["valid_policy_experiment"] is False
    assert diagnostic["fatal_error"] == data["fatal_error"]
    assert report["incomplete_runs"][0]["status"] == "invalid_harness_fatal"


def test_source_prepare_rejection_invalidates_stale_inspected_checkpoint():
    records, manifest = inspection_records_and_manifest()
    moves = [dict(chat_id=1, source=0, target=1, input_tokens=64, success=False,
                  attempted=True, finish_seconds=1.5, transfer_bytes=0,
                  error='0 /start: 400 source has only 0 of 64 tokens')]
    result = analyze_run(summary(moves), records, manifest, CostCurve(((16, 0.1), (128, 0.8))),
                         events=[prefix_event(records[0], [64], seconds=1)], inspection_enabled=True)
    snapshot = result["cold_reroute_cases"][0]["source_snapshot"]
    assert snapshot["observed_reusable_tokens"] == 0
    assert snapshot["source_rejections_applied"] == 1
    assert result["conditional_service_headroom"]["estimated_cases"] == 0


def test_destination_capacity_rejection_does_not_disprove_source_snapshot():
    records, manifest = inspection_records_and_manifest()
    moves = [dict(chat_id=1, source=0, target=1, input_tokens=64, success=False,
                  attempted=True, finish_seconds=1.5, transfer_bytes=0,
                  error='destination lacks full-KV capacity')]
    result = analyze_run(summary(moves), records, manifest, CostCurve(((16, 0.1), (128, 0.8))),
                         events=[prefix_event(records[0], [64], seconds=1)], inspection_enabled=True)
    assert result["cold_reroute_cases"][0]["source_snapshot"]["observed_reusable_tokens"] == 64


def test_drain_breakdown_retains_foreground_tail_and_shutdown_separately():
    data = summary([dict(source=0, finish_seconds=8.5)])
    data.update(retired_engine=0, physical_release={"released": True, "confirmed_seconds": 11,
        "confirmed_monotonic": 111, "signal_monotonic": 109, "shutdown_seconds": 2,
        "graceful": False, "forced_kill": True})
    records = [request(1,-1,target=0,dispatch=3), request(2,1,target=0,dispatch=4), request(3,-1,target=1,dispatch=1)]
    records[0]["finished_seconds"] = 8
    records[1]["finished_seconds"] = 6
    records[0]["requested_output_tokens"] = 100
    records[1]["requested_output_tokens"] = 200
    events = [dict(event="logical_retirement", engine=0, seconds=5),
              dict(event="prefix_inspection", engine=0, seconds=8.6)]
    result = drain_components(data, records, events)
    assert result["inflight_at_retirement"] == 2
    assert result["inflight_requested_output_tokens"] == 300
    assert result["last_source_foreground_finish_seconds"] == 8
    assert result["last_source_preparation_finish_seconds"] == 8.5
    assert result["last_source_inspection_seconds"] == 8.6
    assert result["physical_signal_seconds"] == 9
    assert result["retirement_to_last_foreground_seconds"] == 3
    assert result["foreground_finish_to_signal_seconds"] == 1
    assert result["shutdown_confirmation_seconds"] == 2
    assert result["forced_kill"] is True and result["graceful"] is False


def test_confirmed_source_release_invalidates_checkpoint_only_before_child_dispatch():
    records, manifest = inspection_records_and_manifest()
    curve = CostCurve(((16, 0.1), (128, 0.8)))
    event = prefix_event(records[0], [64], seconds=1)
    data = summary()
    data.update(retired_engine=0, physical_release={"released": True, "confirmed_seconds": 1.5})
    result = analyze_run(data, records, manifest, curve, events=[event], inspection_enabled=True)
    checkpoint = result["cold_reroute_cases"][0]["source_snapshot"]
    assert checkpoint["observed_reusable_tokens"] == 0
    assert checkpoint["source_released_before_dispatch"] is True
    assert result["conditional_service_headroom"]["estimated_cases"] == 0
    assert result["conditional_service_headroom"]["unestimated_released_source"] == 1
    # A later release cannot retroactively invalidate the dispatch-time state.
    data["physical_release"]["confirmed_seconds"] = 2.5
    later = analyze_run(data, records, manifest, curve, events=[event], inspection_enabled=True)
    assert later["conditional_service_headroom"]["estimated_cases"] == 1
    # Unconfirmed shutdown is not proof of disappearance.
    data["physical_release"].update(released=False, confirmed_seconds=1.5)
    unconfirmed = analyze_run(data, records, manifest, curve, events=[event], inspection_enabled=True)
    assert unconfirmed["conditional_service_headroom"]["estimated_cases"] == 1
