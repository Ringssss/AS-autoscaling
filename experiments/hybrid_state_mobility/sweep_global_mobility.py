#!/usr/bin/env python3
"""Sweep global mobility policy sensitivity over replay-derived events."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from experiments.hybrid_state_mobility.simulate_global_mobility import (
    ReplaySimulation,
)


def scale_tool_gaps(events: list[dict], multiplier: float) -> list[dict]:
    by_agent = collections.defaultdict(list)
    for event in events:
        by_agent[event["agent_id"]].append(dict(event))
    result = []
    for rows in by_agent.values():
        rows.sort(key=lambda row: row["turn_index"])
        previous_tool_end = None
        for row in rows:
            duration = row["tool_duration_s"] * multiplier
            if previous_tool_end is None:
                arrival = row["arrival_rel_s"]
            else:
                arrival = previous_tool_end
            tool_start = arrival + row["llm_e2e_template_s"]
            tool_end = tool_start + duration
            row["tool_duration_s"] = duration
            row["arrival_rel_s"] = arrival
            row["tool_start_rel_s"] = tool_start
            row["tool_end_rel_s"] = tool_end
            previous_tool_end = tool_end
            result.append(row)
    return sorted(result, key=lambda row: row["arrival_rel_s"])


def summarize(result: dict) -> dict:
    return {
        "policy": result["policy"],
        "migrations": result["migrations"],
        "drain_migrations": result["drain_migrations"],
        "retiring_states_at_deadline": (
            result["retirement_at_deadline"]["retiring_states"]
            if result["retirement_at_deadline"]
            else None
        ),
        "retiring_tokens_at_deadline": (
            result["retirement_at_deadline"]["retiring_tokens"]
            if result["retirement_at_deadline"]
            else None
        ),
        "final_retiring_snapshots": result["final_retiring_snapshots"],
        "transfer_bytes": result["transfer_bytes"],
        "load_p50_p90_max": result["load_p50_p90_max"],
    }


def run_setting(
    events: list[dict],
    *,
    values: dict,
) -> list[dict]:
    results = []
    for policy in ("sticky", "token-only", "heterogeneous"):
        simulation = ReplaySimulation(
            events,
            engine_count=values["engine_count"],
            token_slots=values["token_slots"],
            snapshot_slots=values["snapshot_slots"],
            scale_in_time=values["scale_in_time"],
            retirement_deadline=values["retirement_deadline"],
            policy_name=policy,
            bandwidth_bytes_per_second=int(values["bandwidth_gib_s"] * 1024**3),
            fixed_transfer_seconds=values["fixed_transfer_ms"] / 1000,
        )
        results.append(summarize(simulation.run()))
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_events = [json.loads(line) for line in args.manifest.open()]
    defaults = {
        "engine_count": 8,
        "token_slots": 262144,
        "snapshot_slots": 32,
        "scale_in_time": 60.0,
        "retirement_deadline": 180.0,
        "bandwidth_gib_s": 4.37,
        "fixed_transfer_ms": 20.0,
    }
    sweeps = {
        "bandwidth": [],
        "tool_gap": [],
        "snapshot_capacity": [],
        "token_capacity": [],
        "scale_in_time": [],
        "deadline": [],
    }
    output = {"base": defaults, "sweeps": sweeps}

    for value in (1.0, 2.0, 4.37, 8.0, 16.0, 32.0):
        values = defaults | {"bandwidth_gib_s": value}
        sweeps["bandwidth"].append(
            {"value": value, "policies": run_setting(base_events, values=values)}
        )

    for value in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0):
        events = scale_tool_gaps(base_events, value)
        sweeps["tool_gap"].append(
            {"value": value, "policies": run_setting(events, values=defaults)}
        )

    for value in (4, 8, 16, 24, 32, 48):
        values = defaults | {"snapshot_slots": value}
        sweeps["snapshot_capacity"].append(
            {"value": value, "policies": run_setting(base_events, values=values)}
        )

    for value in (65536, 98304, 131072, 196608, 262144, 327680):
        values = defaults | {"token_slots": value}
        sweeps["token_capacity"].append(
            {"value": value, "policies": run_setting(base_events, values=values)}
        )

    for value in (30.0, 60.0, 90.0, 120.0):
        values = defaults | {"scale_in_time": value}
        sweeps["scale_in_time"].append(
            {"value": value, "policies": run_setting(base_events, values=values)}
        )

    for value in (120.0, 180.0, 240.0, 300.0):
        values = defaults | {"retirement_deadline": value}
        sweeps["deadline"].append(
            {"value": value, "policies": run_setting(base_events, values=values)}
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
