#!/usr/bin/env python3
"""Event-driven global mobility replay over the Kimi/LooGLE/ToolBench manifest."""

from __future__ import annotations

import argparse
import collections
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from agentshift.controller.global_mobility import (
    EngineMobilityState,
    GlobalMobilityCandidate,
    GlobalMobilityPolicy,
)
from agentshift.controller.materialization import (
    BranchStatePlanner,
    MaterializationMethod,
)
from agentshift.state.descriptor import StateDescriptor


@dataclass
class AgentRuntime:
    agent_id: str
    events: list[dict]
    next_event: int = 0
    phase: str = "new"
    owner_engine: str = ""
    prompt_tokens: int = 0
    blocked: bool = False
    current_event: dict | None = None


class ReplaySimulation:
    def __init__(
        self,
        events: list[dict],
        *,
        engine_count: int,
        token_slots: int,
        snapshot_slots: int,
        scale_in_time: float,
        retirement_deadline: float,
        policy_name: str,
        bandwidth_bytes_per_second: int,
        fixed_transfer_seconds: float,
        retention_seconds: float,
    ):
        self.events = sorted(events, key=lambda event: event["arrival_rel_s"])
        by_agent = collections.defaultdict(list)
        for event in self.events:
            by_agent[event["agent_id"]].append(event)
        for rows in by_agent.values():
            rows.sort(key=lambda row: row["turn_index"])
        self.agents = {
            agent_id: AgentRuntime(agent_id, rows)
            for agent_id, rows in by_agent.items()
        }
        self.engine_ids = [f"engine-{index:02d}" for index in range(engine_count)]
        self.token_slots = token_slots
        self.snapshot_slots = snapshot_slots
        self.scale_in_time = scale_in_time
        self.retirement_deadline = retirement_deadline
        self.policy_name = policy_name
        self.bandwidth = bandwidth_bytes_per_second
        self.fixed_transfer_seconds = fixed_transfer_seconds
        self.retention_seconds = retention_seconds
        self.retiring: set[str] = set()
        self.used_tokens = {engine: 0 for engine in self.engine_ids}
        self.used_snapshots = {engine: 0 for engine in self.engine_ids}
        self.active = {engine: set() for engine in self.engine_ids}
        self.warm = {engine: set() for engine in self.engine_ids}
        self.planner = BranchStatePlanner(
            bandwidth_bytes_per_second=bandwidth_bytes_per_second,
            fixed_copy_seconds=fixed_transfer_seconds,
        )
        self.policy = GlobalMobilityPolicy(self.planner)
        self.stats = collections.Counter()
        self.method_counts = collections.Counter()
        self.transfer_bytes = 0
        self.transfer_seconds = 0.0
        self.load_samples: list[tuple[float, list[int]]] = []
        self.retirement_samples: list[dict] = []
        self.migration_events: list[dict] = []
        self.cache_hit_tokens = 0
        self.recompute_tokens = 0

    def engine_states(self) -> list[EngineMobilityState]:
        result = []
        for engine in self.engine_ids:
            result.append(
                EngineMobilityState(
                    engine_id=engine,
                    free_token_slots=self.token_slots - self.used_tokens[engine],
                    free_state_snapshots=self.snapshot_slots
                    - self.used_snapshots[engine],
                    active_requests=len(self.active[engine]),
                    warm_states=len(self.warm[engine]),
                    retiring=engine in self.retiring,
                )
            )
        return result

    def allocate(self, agent: AgentRuntime, event: dict) -> bool:
        owner = agent.owner_engine
        if agent.phase == "new":
            needed_tokens = event["prompt_tokens"]
            needed_snapshots = 1
            if (
                self.used_tokens[owner] + needed_tokens > self.token_slots
                or self.used_snapshots[owner] + needed_snapshots > self.snapshot_slots
            ):
                return False
            self.used_tokens[owner] += needed_tokens
            self.used_snapshots[owner] += needed_snapshots
            agent.prompt_tokens = needed_tokens
            return True

        growth = event["prompt_tokens"] - agent.prompt_tokens
        if growth and self.used_tokens[owner] + growth > self.token_slots:
            return False
        self.used_tokens[owner] += growth
        agent.prompt_tokens = event["prompt_tokens"]
        return True

    def move_state(self, agent: AgentRuntime, destination: str) -> None:
        source = agent.owner_engine
        self.used_tokens[source] -= agent.prompt_tokens
        self.used_snapshots[source] -= 1
        self.used_tokens[destination] += agent.prompt_tokens
        self.used_snapshots[destination] += 1
        if agent.agent_id in self.warm[source]:
            self.warm[source].remove(agent.agent_id)
            self.warm[destination].add(agent.agent_id)
        agent.owner_engine = destination

    def decide(self, agent: AgentRuntime, event: dict) -> None:
        if self.policy_name == "sticky":
            return
        descriptor = StateDescriptor.hybrid_qwen35(
            descriptor_id=event["event_id"],
            model_identity="qwen35-h100-bf16-tp1",
            branch_id=event["branch_id"],
            prefix_tokens=event["prompt_tokens"],
        )
        actual_bytes = descriptor.total_bytes
        actual_duration = self.fixed_transfer_seconds + actual_bytes / self.bandwidth
        gap = event["tool_end_rel_s"] - event["tool_start_rel_s"]

        if self.policy_name == "stateless-reroute":
            states = self.engine_states()
            source = next(
                state for state in states if state.engine_id == agent.owner_engine
            )
            candidates = [state for state in states if not state.retiring]
            if not candidates:
                return
            destination = min(
                candidates, key=lambda state: (state.load, state.engine_id)
            )
            if destination.engine_id == source.engine_id:
                return
            if not source.retiring and source.load < destination.load + 1:
                return
            self.move_state(agent, destination.engine_id)
            self.stats["migrations"] += 1
            self.stats["drain_migrations"] += int(source.retiring)
            self.method_counts["stateless_reconstruct"] += 1
            self.recompute_tokens += event["prompt_tokens"]
            self.migration_events.append(
                {
                    "time": event["tool_start_rel_s"],
                    "agent_id": agent.agent_id,
                    "event_id": event["event_id"],
                    "source_engine": source.engine_id,
                    "destination_engine": destination.engine_id,
                    "method": "stateless_reconstruct",
                    "tokens": event["prompt_tokens"],
                    "bytes": 0,
                    "drain": source.retiring,
                }
            )
            return

        if self.policy_name == "token-only":
            states = self.engine_states()
            source = next(
                state for state in states if state.engine_id == agent.owner_engine
            )
            candidates = [state for state in states if not state.retiring]
            if not candidates:
                return
            destination = min(
                candidates,
                key=lambda state: (state.load, state.engine_id),
            )
            if destination.engine_id == source.engine_id:
                return
            if not source.retiring and source.load < destination.load + 1:
                return
            token_only_duration = (
                self.fixed_transfer_seconds
                + event["prompt_tokens"] * 147_456 / self.bandwidth
            )
            token_only_fits = token_only_duration <= gap
            actual_fits = actual_duration <= gap
            if token_only_fits and not actual_fits:
                self.stats["optimistic_token_only"] += 1
            if not token_only_fits and actual_fits:
                self.stats["conservative_token_only"] += 1
            if not token_only_fits:
                return
            self.stats["token_only_decisions"] += 1
            if not actual_fits:
                return
            if destination.free_token_slots < descriptor.token_slots:
                self.stats["destination_token_capacity_miss"] += 1
                return
            if destination.free_state_snapshots < 1:
                self.stats["destination_snapshot_capacity_miss"] += 1
                return
            self.move_state(agent, destination.engine_id)
            self.stats["migrations"] += 1
            self.stats["drain_migrations"] += int(source.retiring)
            self.method_counts[MaterializationMethod.COPY_FULL_STATE.value] += 1
            self.migration_events.append(
                {
                    "time": event["tool_start_rel_s"],
                    "agent_id": agent.agent_id,
                    "event_id": event["event_id"],
                    "source_engine": source.engine_id,
                    "destination_engine": destination.engine_id,
                    "method": "token_only_copy",
                    "tokens": event["prompt_tokens"],
                    "bytes": actual_bytes,
                    "drain": source.retiring,
                }
            )
            self.transfer_bytes += actual_bytes
            self.transfer_seconds += actual_duration
            return

        if self.policy_name != "heterogeneous":
            raise ValueError(f"unknown policy: {self.policy_name}")
        candidate = GlobalMobilityCandidate(
            agent_id=agent.agent_id,
            source_engine=agent.owner_engine,
            descriptor=descriptor,
            remaining_gap_seconds=gap,
            future_service_seconds=event["target_output_tokens"] / 1000.0,
        )
        decision = self.policy.select(candidate, self.engine_states())
        if decision is None:
            self.stats["no_feasible_destination"] += 1
            return
        source_engine = agent.owner_engine
        self.move_state(agent, decision.destination_engine)
        self.migration_events.append(
            {
                "time": event["tool_start_rel_s"],
                "agent_id": agent.agent_id,
                "event_id": event["event_id"],
                "source_engine": source_engine,
                "destination_engine": decision.destination_engine,
                "method": decision.plan.selected.method.value,
                "tokens": event["prompt_tokens"],
                "bytes": decision.plan.selected.transfer_bytes,
                "drain": decision.drain_urgent,
            }
        )
        self.stats["migrations"] += 1
        self.stats["drain_migrations"] += int(decision.drain_urgent)
        self.method_counts[decision.plan.selected.method.value] += 1
        self.transfer_bytes += decision.plan.selected.transfer_bytes
        self.transfer_seconds += decision.plan.selected.completion_seconds

    def sample(self, now: float) -> None:
        loads = [
            len(self.active[engine]) + len(self.warm[engine])
            for engine in self.engine_ids
        ]
        self.load_samples.append((now, loads))
        if self.retiring:
            self.retirement_samples.append(
                {
                    "time": now,
                    "retiring_states": sum(
                        self.used_snapshots[engine] for engine in self.retiring
                    ),
                    "retiring_tokens": sum(
                        self.used_tokens[engine] for engine in self.retiring
                    ),
                }
            )

    def run(self) -> dict:
        timeline = []
        for agent in self.agents.values():
            if agent.events:
                first = agent.events[0]
                timeline.append((first["arrival_rel_s"], 3, "arrival", agent))
        timeline.append((self.scale_in_time, 1, "scale", None))
        timeline.append((self.retirement_deadline, 3, "sample", None))

        while timeline:
            now, priority, kind, agent = min(
                timeline, key=lambda item: (item[0], item[1])
            )
            timeline.remove((now, priority, kind, agent))
            if kind == "scale":
                active_count = len(self.engine_ids) // 2
                self.retiring = set(self.engine_ids[active_count:])
                self.stats["scale_in_event"] += 1
                self.sample(now)
                continue
            if kind == "sample":
                self.sample(now)
                continue

            if kind == "arrival":
                event = agent.events[agent.next_event]
                if agent.phase != "new" and self.policy_name != "stateless-reroute":
                    self.cache_hit_tokens += agent.prompt_tokens
                if agent.phase == "new":
                    agent.owner_engine = event["source_engine"]
                if not self.allocate(agent, event):
                    agent.blocked = True
                    self.stats["source_capacity_blocks"] += 1
                    continue
                agent.blocked = False
                agent.phase = "active"
                agent.current_event = event
                self.active[agent.owner_engine].add(agent.agent_id)
                self.warm[agent.owner_engine].discard(agent.agent_id)
                agent.next_event += 1
                timeline.append((event["tool_start_rel_s"], 1, "tool-start", agent))
            elif kind == "tool-start":
                event = agent.current_event
                if event is None:
                    raise RuntimeError(
                        f"tool-start without current event: {agent.agent_id}"
                    )
                self.active[agent.owner_engine].discard(agent.agent_id)
                agent.phase = "tool"
                self.warm[agent.owner_engine].add(agent.agent_id)
                self.decide(agent, event)
                timeline.append((event["tool_end_rel_s"], 2, "tool-end", agent))
                self.sample(now)
            elif kind == "tool-end":
                agent.phase = "warm"
                timeline.append(
                    (
                        now + self.retention_seconds,
                        2,
                        "expire",
                        agent,
                    )
                )
                if agent.next_event < len(agent.events):
                    next_event = agent.events[agent.next_event]
                    timeline.append((next_event["arrival_rel_s"], 3, "arrival", agent))
            elif kind == "expire" and agent.phase == "warm":
                owner = agent.owner_engine
                self.used_tokens[owner] -= agent.prompt_tokens
                self.used_snapshots[owner] -= 1
                self.warm[owner].discard(agent.agent_id)
                agent.phase = "expired"
                self.stats["expired_warm_states"] += 1
            elif kind != "expire":
                raise AssertionError(kind)

        # Advance any remaining tool-end events omitted when an agent was blocked.
        loads = [load for _, loads in self.load_samples for load in loads]
        retirement_at_deadline = next(
            (
                sample
                for sample in reversed(self.retirement_samples)
                if sample["time"] <= self.retirement_deadline
            ),
            None,
        )
        return {
            "policy": self.policy_name,
            "events": len(self.events),
            "agents": len(self.agents),
            "migrations": self.stats["migrations"],
            "drain_migrations": self.stats["drain_migrations"],
            "method_counts": dict(self.method_counts),
            "transfer_bytes": self.transfer_bytes,
            "transfer_seconds": self.transfer_seconds,
            "counts": dict(self.stats),
            "load_p50_p90_max": np.percentile(loads, [50, 90]).tolist()
            + [max(loads, default=0)],
            "retirement_at_deadline": retirement_at_deadline,
            "final_retiring_snapshots": sum(
                self.used_snapshots[engine] for engine in self.retiring
            ),
            "final_retiring_tokens": sum(
                self.used_tokens[engine] for engine in self.retiring
            ),
            "cache_hit_tokens": self.cache_hit_tokens,
            "recompute_tokens": self.recompute_tokens,
            "migration_events": self.migration_events,
            "load_series": self.load_samples,
            "final_owner_distribution": {
                engine: self.used_snapshots[engine] for engine in self.engine_ids
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-count", type=int, default=8)
    parser.add_argument("--token-slots", type=int, default=262144)
    parser.add_argument("--snapshot-slots", type=int, default=32)
    parser.add_argument("--scale-in-time", type=float, default=60.0)
    parser.add_argument("--retirement-deadline", type=float, default=180.0)
    parser.add_argument("--bandwidth-gib-s", type=float, default=4.37)
    parser.add_argument("--fixed-transfer-ms", type=float, default=20.0)
    parser.add_argument("--retention-seconds", type=float, default=300.0)
    args = parser.parse_args()

    events = [json.loads(line) for line in args.manifest.open()]
    results = []
    for policy in (
        "sticky",
        "stateless-reroute",
        "token-only",
        "heterogeneous",
    ):
        simulation = ReplaySimulation(
            events,
            engine_count=args.engine_count,
            token_slots=args.token_slots,
            snapshot_slots=args.snapshot_slots,
            scale_in_time=args.scale_in_time,
            retirement_deadline=args.retirement_deadline,
            policy_name=policy,
            bandwidth_bytes_per_second=int(args.bandwidth_gib_s * 1024**3),
            fixed_transfer_seconds=args.fixed_transfer_ms / 1000,
            retention_seconds=args.retention_seconds,
        )
        results.append(simulation.run())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"policies": results}, indent=2) + "\n")
    print(json.dumps({"policies": results}, indent=2))


if __name__ == "__main__":
    main()
