from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.benchmark_e2e import flush, timed_generate

from agentshift.controller.migration import MigrationCoordinator
from agentshift.controller.placement import order_admissible_first
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


@dataclass(frozen=True, slots=True)
class PolicyAgent:
    agent_id: str
    prefix_tokens: int
    completed: tuple[int, ...]
    gap_seconds: float
    estimated_copy_seconds: float
    ordinal: int


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def projected_cost(order: tuple[PolicyAgent, ...] | list[PolicyAgent]) -> tuple[int, float, float]:
    completion = 0.0
    exposed = []
    for agent in order:
        completion += agent.estimated_copy_seconds
        exposed.append(max(0.0, completion - agent.gap_seconds))
    return (sum(value > 0 for value in exposed), sum(exposed), max(exposed, default=0.0))


def order_agents(policy: str, agents: list[PolicyAgent]) -> list[PolicyAgent]:
    if policy == "fifo":
        return list(agents)
    if policy == "shortest-kv":
        return sorted(agents, key=lambda item: (item.prefix_tokens, item.ordinal))
    if policy == "largest-kv":
        return sorted(agents, key=lambda item: (-item.prefix_tokens, item.ordinal))
    if policy == "earliest-return":
        return sorted(agents, key=lambda item: (item.gap_seconds, item.ordinal))
    if policy == "longest-gap":
        return sorted(agents, key=lambda item: (-item.gap_seconds, item.ordinal))
    if policy == "least-slack":
        return sorted(
            agents,
            key=lambda item: (
                item.gap_seconds - item.estimated_copy_seconds,
                item.ordinal,
            ),
        )
    if policy == "agentshift-score":
        return order_admissible_first(
            agents,
            deadline_seconds=lambda item: item.gap_seconds,
            duration_seconds=lambda item: item.estimated_copy_seconds,
            size=lambda item: item.prefix_tokens,
        )
    if policy == "oracle":
        if len(agents) > 8:
            raise ValueError("oracle enumeration supports at most eight agents")
        return list(min(itertools.permutations(agents), key=projected_cost))
    raise ValueError(f"unknown policy: {policy}")


class MigrationPolicyBenchmark:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.state_path = (
            Path(args.output_dir) / f"migration-policy-state-{time.time_ns()}.db"
        )
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            base_port=args.transfer_port,
            async_transfer=True,
            transfer_poll_interval=args.poll_interval,
        )
        self.serial = 0
        self.copy_estimates: dict[int, float] = {}

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    async def calibrate_copy_times(self) -> dict[int, float]:
        await self.coordinator.initialize_transfer_pair("engine-a", "engine-b")
        for prefix in sorted(set(self.args.prefix_lengths)):
            samples = []
            for _ in range(self.args.handoff_calibration_repeats):
                await self.clean()
                self.serial += 1
                agent_id = f"policy-calibration-{prefix}-{self.serial}"
                prompt = [31000 + self.serial] + [100] * (prefix - 1)
                _, first = await timed_generate(
                    self.source,
                    prompt,
                    max_new_tokens=self.args.first_output_tokens,
                    rid=f"{agent_id}-first",
                )
                completed = tuple(prompt + first["output_ids"])
                self.store.register_agent(
                    AgentContinuation(
                        agent_id=agent_id,
                        committed_step=1,
                        owner_engine="engine-a",
                        owner_epoch=1,
                        token_ids=completed,
                        pending_tool_future=f"tool-{agent_id}",
                    )
                )
                handoff_started = time.perf_counter()
                migration = await self.coordinator.migrate(agent_id, "engine-b")
                samples.append(time.perf_counter() - handoff_started)
                await self.coordinator.acknowledge_destination(
                    migration.migration_id
                )
                await self.destination.release_prefix(
                    agent_id, migration.new_epoch, allow_missing=True
                )
            self.copy_estimates[prefix] = (
                statistics.median(samples) * self.args.handoff_safety_factor
            )
        await self.clean()
        return dict(self.copy_estimates)

    async def prepare(self, policy: str) -> list[PolicyAgent]:
        await self.clean()
        agents = []
        bandwidth = self.args.estimated_bandwidth_gib_s * 1024**3
        for ordinal, (prefix, gap_ms) in enumerate(
            zip(self.args.prefix_lengths, self.args.gap_ms, strict=True)
        ):
            self.serial += 1
            agent_id = f"policy-{policy}-{self.serial}-{ordinal}"
            prompt = [30000 + self.serial] + [100] * (prefix - 1)
            _, first = await timed_generate(
                self.source,
                prompt,
                max_new_tokens=self.args.first_output_tokens,
                rid=f"{agent_id}-first",
            )
            completed = tuple(prompt + first["output_ids"])
            self.store.register_agent(
                AgentContinuation(
                    agent_id=agent_id,
                    committed_step=1,
                    owner_engine="engine-a",
                    owner_epoch=1,
                    token_ids=completed,
                    pending_tool_future=f"tool-{agent_id}",
                )
            )
            # Protect each completed turn as soon as the agent blocks on its
            # tool. Otherwise later prefills may evict a candidate before the
            # migration policy gets to make a decision.
            pin = await self.source.pin_prefix(agent_id, 1, completed)
            if int(pin["token_count"]) != len(completed):
                raise RuntimeError(
                    f"source retained only {pin['token_count']} of "
                    f"{len(completed)} tokens for {agent_id}"
                )
            kv_bytes = len(completed) * self.args.kv_bytes_per_token
            agents.append(
                PolicyAgent(
                    agent_id=agent_id,
                    prefix_tokens=len(completed),
                    completed=completed,
                    gap_seconds=gap_ms / 1000.0,
                    estimated_copy_seconds=self.copy_estimates.get(
                        prefix,
                        self.args.fixed_copy_ms / 1000.0 + kv_bytes / bandwidth,
                    ),
                    ordinal=ordinal,
                )
            )
        await self.coordinator.initialize_transfer_pair("engine-a", "engine-b")
        return agents

    async def next_turn(
        self,
        agent: PolicyAgent,
        migration,
        tool_started: float,
        migration_completed: float,
        handoff_seconds: float,
    ) -> dict[str, Any]:
        deadline = tool_started + agent.gap_seconds
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
        ready = max(deadline, migration_completed)
        request_started = time.perf_counter()
        next_seconds, result = await timed_generate(
            self.destination,
            list(agent.completed) + [200] * self.args.tool_result_tokens,
            max_new_tokens=self.args.next_output_tokens,
            rid=f"{agent.agent_id}-next",
        )
        finished = time.perf_counter()
        await self.coordinator.acknowledge_destination(migration.migration_id)
        return {
            "agent_id": agent.agent_id,
            "prefix_tokens": agent.prefix_tokens,
            "gap_seconds": agent.gap_seconds,
            "estimated_copy_seconds": agent.estimated_copy_seconds,
            "actual_copy_seconds": migration.transfer_seconds,
            "actual_handoff_seconds": handoff_seconds,
            "migration_tokens": migration.token_count,
            "migration_completed_seconds": migration_completed - tool_started,
            "completed_in_gap": migration_completed <= deadline,
            "exposed_migration_seconds": max(0.0, migration_completed - deadline),
            "post_tool_seconds": max(0.0, migration_completed - deadline) + next_seconds,
            "cached_tokens": int(result["meta_info"]["cached_tokens"]),
            "request_started_seconds": request_started - tool_started,
            "finished_seconds": finished - tool_started,
            "migration_bytes": migration.bytes_transferred,
        }

    async def run_one(self, policy: str, repeat: int) -> dict[str, Any]:
        agents = await self.prepare(policy)
        order = order_agents(policy, agents)
        tool_started = time.perf_counter()
        next_tasks = []
        for agent in order:
            handoff_started = time.perf_counter()
            migration = await self.coordinator.migrate(agent.agent_id, "engine-b")
            handoff_seconds = time.perf_counter() - handoff_started
            if migration.token_count != agent.prefix_tokens:
                raise RuntimeError(
                    f"migration copied only {migration.token_count} of "
                    f"{agent.prefix_tokens} tokens for {agent.agent_id}"
                )
            migration_completed = time.perf_counter()
            next_tasks.append(
                asyncio.create_task(
                    self.next_turn(
                        agent,
                        migration,
                        tool_started,
                        migration_completed,
                        handoff_seconds,
                    )
                )
            )
        records = await asyncio.gather(*next_tasks)
        exposed = [record["exposed_migration_seconds"] for record in records]
        post_tool = [record["post_tool_seconds"] for record in records]
        output = {
            "policy": policy,
            "repeat": repeat,
            "order_prefix_tokens": [agent.prefix_tokens for agent in order],
            "order_gap_ms": [agent.gap_seconds * 1000 for agent in order],
            "completed_in_gap_fraction": statistics.mean(
                record["completed_in_gap"] for record in records
            ),
            "exposed_migration_sum_seconds": sum(exposed),
            "exposed_migration_p95_seconds": percentile(exposed, 0.95),
            "post_tool_mean_seconds": statistics.mean(post_tool),
            "post_tool_p95_seconds": percentile(post_tool, 0.95),
            "makespan_seconds": max(record["finished_seconds"] for record in records),
            "full_hit_fraction": statistics.mean(
                record["cached_tokens"] >= record["prefix_tokens"]
                for record in records
            ),
            "records": records,
        }
        for agent in agents:
            continuation = self.store.get_agent(agent.agent_id)
            await self.destination.release_prefix(
                agent.agent_id, continuation.owner_epoch, allow_missing=True
            )
        await self.clean()
        return output


async def main(args) -> None:
    if len(args.prefix_lengths) != len(args.gap_ms):
        raise ValueError("--prefix-lengths and --gap-ms must have equal length")
    benchmark = MigrationPolicyBenchmark(args)
    calibrations = await benchmark.calibrate_copy_times()
    outputs = []
    for repeat in range(args.repeats):
        policies = list(args.policies)
        random.Random(args.seed + repeat).shuffle(policies)
        for policy in policies:
            result = await benchmark.run_one(policy, repeat)
            outputs.append(result)
            print(json.dumps({key: value for key, value in result.items() if key != "records"}, sort_keys=True), flush=True)
    output = {
        "workload": "heterogeneous-prefix blocked-agent scheduling",
        "config": vars(args),
        "handoff_calibrations_seconds": calibrations,
        "state_db": str(benchmark.state_path),
        "records": outputs,
    }
    output_path = Path(args.output_dir) / f"migration-policy-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=[
            "fifo",
            "shortest-kv",
            "largest-kv",
            "earliest-return",
            "longest-gap",
            "least-slack",
            "agentshift-score",
            "oracle",
        ],
    )
    parser.add_argument(
        "--prefix-lengths",
        type=int,
        nargs="+",
        default=[4096, 8192, 12288, 16384, 24576, 32768],
    )
    parser.add_argument(
        "--gap-ms",
        type=float,
        nargs="+",
        default=[240, 60, 500, 100, 360, 160],
    )
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--next-output-tokens", type=int, default=4)
    parser.add_argument("--estimated-bandwidth-gib-s", type=float, default=70.0)
    parser.add_argument("--fixed-copy-ms", type=float, default=8.0)
    parser.add_argument("--handoff-safety-factor", type=float, default=1.25)
    parser.add_argument("--handoff-calibration-repeats", type=int, default=3)
    parser.add_argument("--kv-bytes-per-token", type=int, default=147456)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.002)
    parser.add_argument("--transfer-port", type=int, default=31300)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
