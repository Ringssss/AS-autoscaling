#!/usr/bin/env python3
"""Compile same-model evidence and retain experimental validity boundaries."""

import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    output = {"campaigns": {}, "correctness": {}}
    lines = ["# Qwen3.5-35B-A3B Native Trace Campaign", "", "This is a prototype validation campaign. The complete FluidKV system, physical GPU release, and full failure-recovery protocol are not implemented in this harness. No NSDI-scale performance claim follows from these pilots.", "", "## Model and input contract", "", "Three H100 PCIe engines use the same Qwen3.5-35B-A3B weights, BF16, TP1, FP32 recurrent state, 65,536 KV token slots, 64 recurrent slots, FA3 full attention, Triton linear attention and deterministic inference. CUDA graphs are disabled. Each run persists complete server settings. Kimi's original logs remain characterization-only because prefix identities and lineage are unavailable.", "", "Qwen-Bailian A, Coder and Thinking are separate traces. Anonymous block hashes produce deterministic token surrogates, not original semantic content. Input plus requested output must fit the 32,768-token context. Recorded output lengths are retained when output-cap is zero. Original arrival timestamps and any explicit time scale are recorded; causal eligibility delays are separate from router wait.", ""]
    pair_path = args.root / "deterministic-pairs32/summary.json"
    if pair_path.exists():
        pair = json.loads(pair_path.read_text())
        output["correctness"]["deterministic_pairs"] = pair["policies"]
        policies = pair["policies"]
        lines += ["## Correctness gate", "", f"Deterministic mode: sticky matches cold in {policies['sticky']['output_matches_cold']}/{policies['sticky']['samples']} selected pairs. Both migration modes match sticky in {policies['on-return']['output_matches_sticky']}/{policies['on-return']['samples']} and {policies['prepared-full-copy']['output_matches_sticky']}/{policies['prepared-full-copy']['samples']} cases. The cold-equivalence gate remains failed; the individual ToolBench case that became equal under deterministic mode was not a general fix.", ""]
    boundary_path = args.root / "completed-boundary/summary.json"
    if boundary_path.exists():
        boundary = json.loads(boundary_path.read_text())
        output["correctness"]["completed_boundary"] = boundary
        lines += ["A separate boundary control generates 1, 8 and 64 parent tokens. In the tested conversation, the prompt remains reusable and the committed prompt-plus-processed-output boundary migrates with matching source/destination continuation. The final sampled token has not yet been processed, so the committed boundary excludes it. This is a constructed continuation control, not the recorded trace child.", ""]
    reservation_path = args.root / "reservation-failure-fixed/summary.json"
    if reservation_path.exists():
        reservation = json.loads(reservation_path.read_text())
        output["reservation_failure_regression"] = reservation
        count = sum(all(record["checks"].values()) for record in reservation["records"])
        lines += ["## Capacity failure and protocol repair", "", "The initial Coder full-copy run exposed an unmatched NCCL send: the source started before the destination rejected insufficient KV capacity. Its group later timed out, so the initial campaign was stopped. These interrupted results must not be pooled with repaired runs.", "", "The repaired async protocol prepares and pins source state, allocates destination resources, and checks both PREPARED acknowledgements before starting NCCL. Prepared reservations can be cleaned without issuing communication. Source packing now occurs during preparation; its foreground cost needs remeasurement.", "", f"Live capacity-exhaustion regression: {count}/{len(reservation['records'])} runs rejected the transfer, recorded ABORTED, cleaned the private reservation, and immediately reused the same group for a successful migration with matching continuation. This covers admission failure, not arbitrary post-launch network or engine failure.", ""]
    for manifest_path in sorted(args.root.glob("*/campaign.json")):
        campaign = json.loads(manifest_path.read_text())
        name = manifest_path.parent.name
        result = {"manifest": campaign, "splits": {}}
        lines += [f"## {name}", "", "| Split | Policy | Success/offered | Internal parent links | TTFT p50 (s) | SLO attainment | Prepared states | Transfer MiB | Rebuilt tokens |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for path in sorted(manifest_path.parent.glob("*/summary.json")):
            run = {"split": path.parent.name}
            summary = json.loads(path.read_text())
            request_path = path.parent / "requests.jsonl"
            records = [json.loads(line) for line in request_path.open()]
            selected_ids = set(summary["selected_chat_ids"])
            selected_rows = {}
            for record in records:
                selected_rows[record["chat_id"]] = record
            linked = sum(row["parent_chat_id"] in selected_ids for row in selected_rows.values())
            summary["verified_internal_parent_links"] = linked
            summary["successful_materializations"] = {}
            summary["child_metrics"] = {}
            summary["cold_reroute_cases"] = []
            for row in summary["runs"]:
                preparation = row["materialization"]
                moves = preparation["records"] if preparation else []
                transferred = sum(move["transfer_bytes"] for move in moves)
                recomputed = sum(move["reconstruction_tokens"] for move in moves)
                successes = sum(move["success"] for move in moves)
                ttft = row["ttft_p50_p90"][0] if row["ttft_p50_p90"] else float("nan")
                lines.append(f"| {run['split']} | {row['policy']} | {row['successful']}/{row['offered']} | {linked} | {ttft:.3f} | {row['slo_attainment']:.3f} | {successes}/{len(moves)} | {transferred / 2**20:.2f} | {recomputed} |")
                summary["successful_materializations"][f"{row['policy']}-{row['repetition']}"] = dict(Counter(move["method"] for move in moves if move["success"]))
                children = [record for record in records if record["policy"] == row["policy"]
                            and record["repetition"] == row["repetition"]
                            and record["parent_chat_id"] in selected_ids and record["success"]]
                child_metrics = {
                    "successful_children": len(children),
                    "offered_children": linked,
                    "ttft_p50_ms": float(np.median([record["ttft_seconds"] for record in children]) * 1000) if children else None,
                    "e2e_p50_seconds": float(np.median([record["e2e_seconds"] for record in children])) if children else None,
                    "ttft_fraction_of_e2e_p50": float(np.median([record["ttft_seconds"] / record["e2e_seconds"] for record in children])) if children else None,
                    "slo_attainment": sum(record["ttft_seconds"] <= row["slo_seconds"] for record in children) / linked if linked else None,
                    "original_arrival_ttft_p50_ms": float(np.median([record["ttft_seconds"] + record["eligibility_delay_seconds"] for record in children]) * 1000) if children else None,
                    "actual_cached_tokens": sum(record["actual_cached_tokens"] for record in children),
                    "preparation_ready_before_matching_child": 0,
                }
                retirement_elapsed = row.get("retirement_elapsed_seconds")
                if moves and retirement_elapsed is None:
                    child_metrics["preparation_ready_before_matching_child"] = None
                if retirement_elapsed is not None:
                    base = min(record["trace_arrival_seconds"] for record in records)
                    for move in moves:
                        finished = move.get("finished_since_retirement_seconds")
                        if not move["success"] or finished is None:
                            continue
                        prepared_at = retirement_elapsed + finished
                        for child in children:
                            dispatched_at = ((child["trace_arrival_seconds"] - base) / summary["time_scale"]
                                             + child["eligibility_delay_seconds"] + child["router_wait_seconds"])
                            if (child["parent_chat_id"] == move["chat_id"] and child["target"] == move["target"]
                                    and prepared_at <= dispatched_at and child["actual_cached_tokens"] >= move["input_tokens"]):
                                child_metrics["preparation_ready_before_matching_child"] += 1
                summary["child_metrics"][f"{row['policy']}-{row['repetition']}"] = child_metrics
            baseline = {record["chat_id"]: record for record in records if record["policy"] == "dualmap" and record["repetition"] == 0}
            for child_id, child in baseline.items():
                parent = baseline.get(child["parent_chat_id"])
                if (parent is None or not child["success"] or parent["target"] == child["target"]
                        or child["actual_cached_tokens"] >= parent["input_tokens"]):
                    continue
                variants = {record["policy"]: {"ttft_ms": record["ttft_seconds"] * 1000 if record["ttft_seconds"] is not None else None,
                            "cached_tokens": record["actual_cached_tokens"], "target": record["target"]}
                            for record in records if record["chat_id"] == child_id and record["repetition"] == 0}
                summary["cold_reroute_cases"].append({"parent_chat_id": parent["chat_id"], "child_chat_id": child_id,
                    "parent_input_tokens": parent["input_tokens"], "variants": variants})
            result["splits"][run["split"]] = summary
        lines += ["", "Child-only metrics (TTFT starts at causal eligibility; original-arrival latency is retained separately in JSON):", "", "| Split | Policy | Successful children | Child TTFT p50 (ms) | Child cached tokens | Prepared before matching child |", "| --- | --- | ---: | ---: | ---: | ---: |"]
        for split, summary in result["splits"].items():
            for policy, metrics in summary["child_metrics"].items():
                if metrics["successful_children"]:
                    lines.append(f"| {split} | {policy} | {metrics['successful_children']} | {metrics['ttft_p50_ms']:.2f} | {metrics['actual_cached_tokens']} | {metrics['preparation_ready_before_matching_child']} |")
        lines += ["", "The JSON also records child E2E latency and TTFT/E2E fractions. Reducing a small prefill component does not establish a comparable gain in whole-turn latency or GPU cost."]
        cases = [(split, case) for split, summary in result["splits"].items() for case in summary["cold_reroute_cases"]]
        if cases:
            lines += ["", "Individual cold-reroute cases (one observation per variant, not speedup estimates):", "", "| Split / child | Parent tokens | Policy | Cached tokens | TTFT (ms) |", "| --- | ---: | --- | ---: | ---: |"]
            for split, case in cases:
                for policy, value in case["variants"].items():
                    latency = f"{value['ttft_ms']:.2f}" if value["ttft_ms"] is not None else "failed"
                    lines.append(f"| {split} / {case['child_chat_id']} | {case['parent_input_tokens']} | {policy} | {value['cached_tokens']} | {latency} |")
        lines += ["", "These single-run values are diagnostic. Chronological windows with no selected parent links cannot show next-turn reuse benefit. Exact-parent cohorts select complete block-aligned parents and are mechanism studies, not representative full-trace samples. First-use compilation and queue-timing changes can affect policy order; repeat paired warmed runs before attributing latency differences to materialization.", ""]
        output["campaigns"][name] = result
    lines += ["## Baseline parity and remaining gaps", "", "All Qwen3.5 policies receive the same snapshot-aware cache estimator. It tracks observed prompt/committed-output boundaries with a bounded metadata cache, rather than treating arbitrary matching KV blocks as executable hybrid state. Actual engine cache hits and estimator error are retained; this is an estimate, not an exact engine cache directory.", "", "DualMap retains upstream hash selection and queue rebalancing. The membership adapter retires the final logical instance, requeues waiting work, and lets in-flight requests finish. Sticky can retain a parent's old placement. The full-copy extension prepares completed state on a remaining instance. The planner prototype selects full copy or reconstruction using the existing cost planner and the same ownership store. Destination choice is a simple least-loaded heuristic, not the proposed full joint elasticity scheduler. Source cleanup unpins state but does not prove physical memory or GPU release.", "", "Required next gates: calibrated warmed multi-window repetitions with meaningful parent coverage; model-valid snapshot/replay options for partial-prefix relationships; continuous placement and resource-reservation integration; actual source eviction and shutdown accounting; early-return and failure recovery; explicit bounds on cached/full-prefill numerical differences. Do not infer full FluidKV superiority over DualMap from these pilot tables.", ""]
    (args.root / "RESULTS.md").write_text("\n".join(lines))
    (args.root / "results_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(args.root / "RESULTS.md")


if __name__ == "__main__":
    main()
