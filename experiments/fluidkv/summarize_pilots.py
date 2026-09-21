#!/usr/bin/env python3
"""Summarize pilot evidence without hiding cold/cached output differences."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def pair_summary(path):
    groups = defaultdict(dict)
    for line in (path / "records.jsonl").open():
        row = json.loads(line)
        groups[row["repetition"], row["ordinal"]][row["policy"]] = row
    if any(len(group) != 4 for group in groups.values()):
        raise ValueError(f"incomplete paired run: {path}")
    result = {
        "pair_repetitions": len(groups),
        "unique_pairs": len({ordinal for _, ordinal in groups}),
        "sticky_matches_cold": 0, "migrated_matches_sticky": {},
        "policies": {},
    }
    for mode in ("sticky", "cold-reroute", "on-return", "prepared-full-copy"):
        rows = [g[mode] for g in groups.values()]
        result["policies"][mode] = {
            "ttft_p50_ms": float(np.median([r["post_ready_ttft_seconds"] for r in rows]) * 1000),
            "e2e_p50_ms": float(np.median([r["post_ready_e2e_seconds"] for r in rows]) * 1000),
            "migration_p50_ms": float(np.median([r["migration_seconds"] for r in rows]) * 1000),
            "cache_length_and_owner_checks_passed": sum(all(v for k, v in r["checks"].items() if not k.startswith("output_matches_")) for r in rows),
        }
    result["sticky_matches_cold"] = sum(g["sticky"]["child"]["output_ids"] == g["cold-reroute"]["child"]["output_ids"] for g in groups.values())
    for mode in ("on-return", "prepared-full-copy"):
        result["migrated_matches_sticky"][mode] = sum(g[mode]["child"]["output_ids"] == g["sticky"]["child"]["output_ids"] for g in groups.values())
    result["cold_cached_equivalence_unresolved"] = result["sticky_matches_cold"] != len(groups)
    result["migration_changes_output_relative_to_sticky"] = any(v != len(groups) for v in result["migrated_matches_sticky"].values())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    audit = json.loads((args.root / "trace_audit.json").read_text())
    result = {"audit": {"kimi_requests": audit["kimi"]["requests"], "qwen": audit["qwen_bailian_a"]}, "pair_pilots": {}, "routing_pilots": {}}
    lines = ["# FluidKV Native Trace Pilot Results", "", "These are pilot measurements, not a completed NSDI evaluation or a FluidKV-versus-DualMap speedup claim.", "", "## Inputs", "", f"Kimi: {audit['kimi']['requests']:,} original Finish records. All input IDs are redacted; conversation IDs are absent. Native branch reconstruction and cache-aware routing cannot be recovered from these fields.", "", f"Qwen-Bailian A: {audit['qwen_bailian_a']['requests']:,} requests, {audit['qwen_bailian_a']['linked_requests']:,} parent links. Its original timestamps and anonymous block hashes are retained. The selected migration pilot uses 32 of 1,071 exact block-aligned parent pairs under the 8,192-token limit, not a representative random sample of all links.", ""]
    for name in ("pairs32-r3", "toolbench16-r1"):
        if not (args.root / name / "records.jsonl").exists():
            continue
        summary = pair_summary(args.root / name)
        result["pair_pilots"][name] = summary
        n = summary["pair_repetitions"]
        lines += [f"## {name}", "", f"{summary['unique_pairs']} input pairs; {n} pair repetitions; one generated parent token and eight child tokens. Qwen inputs are hash-token surrogates; ToolBench inputs use recorded function-call text. Neither supplies observed tool execution times.", "", "| Policy | TTFT median (ms) | E2E median (ms) | Migration median (ms) |", "| --- | ---: | ---: | ---: |"]
        for mode, metrics in summary["policies"].items():
            lines.append(f"| {mode} | {metrics['ttft_p50_ms']:.2f} | {metrics['e2e_p50_ms']:.2f} | {metrics['migration_p50_ms']:.2f} |")
        lines += ["", "Prepared-full-copy starts child eligibility only after state preparation. On-return includes preparation on the critical path. This isolates destination readiness and does not show that a trace-derived tool gap hid the transfer.", "", f"Output agreement: prepared-full-copy versus sticky {summary['migrated_matches_sticky']['prepared-full-copy']}/{n}; on-return versus sticky {summary['migrated_matches_sticky']['on-return']}/{n}; sticky versus cold {summary['sticky_matches_cold']}/{n}.", "", "The raw benchmark retains failed cold-equivalence checks. Migration can preserve cached-path behavior while cached-versus-cold inference still differs. Do not mark the entire pipeline correct or use cache hits alone as correctness evidence.", ""]
    diagnostic = args.root / "toolbench-same-engine-diagnostic.json"
    if diagnostic.exists():
        d = json.loads(diagnostic.read_text())
        result["same_engine_diagnostic"] = {"equal_runs": d["equal_runs"], "repetitions": len(d["records"])}
        lines += ["## Same-engine diagnostic", "", f"Without any transfer or ownership change, the selected ToolBench pair matches between cold and parent-prefilled execution in {d['equal_runs']}/{len(d['records'])} repetitions. The captured top-token log probabilities differ at the first divergent output position. This isolates a baseline cached/full-prefill discrepancy; it does not establish its numerical or implementation root cause.", ""]
    for path in sorted(args.root.glob("routing-*/summary.json")):
        d = json.loads(path.read_text())
        result["routing_pilots"][path.parent.name] = d
        lines += [f"## {path.parent.name}", "", f"Pinned upstream DualMap on a common SGLang backend; {d['selected_requests']} requests per run, time scale {d['time_scale']}, output cap {d['output_cap']} (zero means recorded output length). No physical DRAM tier is claimed. Cache estimates and actual backend hits are reported separately.", "", "| Policy | Repetition | Success/offered | TTFT p50 (ms) | SLO attainment | SLO goodput (req/s) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for row in d["runs"]:
            ttft = row["ttft_p50_p90"][0] * 1000 if row["ttft_p50_p90"] else float("nan")
            lines.append(f"| {row['policy']} | {row['repetition']} | {row['successful']}/{row['offered']} | {ttft:.2f} | {row['slo_attainment']:.3f} | {row['slo_goodput_requests_per_second']:.3f} |")
        logs = [json.loads(line) for line in (path.parent / "requests.jsonl").open()]
        aggregates = {}
        for policy in sorted({row["policy"] for row in logs}):
            successful = [row for row in logs if row["policy"] == policy and row["success"]]
            offered = sum(row["offered"] for row in d["runs"] if row["policy"] == policy)
            elapsed = sum(row["duration_seconds"] for row in d["runs"] if row["policy"] == policy)
            ready_delays = [row["ttft_seconds"] for row in successful]
            original_arrival_delays = [row["ttft_seconds"] + row["eligibility_delay_seconds"] for row in successful]
            slo = next(row["slo_seconds"] for row in d["runs"] if row["policy"] == policy)
            aggregates[policy] = {
                "offered": offered, "successful": len(successful),
                "eligible_ttft_p50_p90_ms": (np.percentile(ready_delays, [50, 90]) * 1000).tolist() if successful else None,
                "original_arrival_ttft_p50_p90_ms": (np.percentile(original_arrival_delays, [50, 90]) * 1000).tolist() if successful else None,
                "eligible_slo_attainment": sum(value <= slo for value in ready_delays) / offered,
                "original_arrival_slo_attainment": sum(value <= slo for value in original_arrival_delays) / offered,
                "eligible_slo_goodput": sum(value <= slo for value in ready_delays) / elapsed,
                "actual_cached_tokens": sum(row["actual_cached_tokens"] for row in successful),
                "estimated_cache_absolute_error_tokens": sum(abs(row["estimated_cached_tokens"] - row["actual_cached_tokens"]) for row in successful),
            }
        result["routing_pilots"][path.parent.name]["aggregate"] = aggregates
        lines += ["", "This compares router policies only. FluidKV's joint planner, proactive state preparation and physical scale-in are not part of this routing pilot.", ""]
    lines += ["## Remaining experiments", "", "Resolve or explicitly bound hybrid cached/full-prefill numerical behavior. Then integrate model-valid copy/replay plans and lifecycle scheduling into the continuous harness. Run equal-budget SLO curves, fixed and controller-selected membership changes, physical GPU release accounting, failure recovery, and architecture/topology coverage. Kimi lifecycle claims require additional native identity data or a separately labelled constructed workload.", ""]
    (args.root / "pilot_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.root / "PILOT_RESULTS.md").write_text("\n".join(lines))
    print(args.root / "PILOT_RESULTS.md")


if __name__ == "__main__":
    main()
