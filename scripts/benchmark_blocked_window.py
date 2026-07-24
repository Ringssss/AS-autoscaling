from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_e2e import flush, timed_generate

from agentshift.controller.baselines import agentix_style_route, calibrate_ttl
from agentshift.controller.migration import MigrationCoordinator
from agentshift.controller.tiered import (
    SharedSemanticHandoffCoordinator,
    TieredPrefixCoordinator,
)
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


class BlockedWindowBenchmark:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.state_path = (
            Path(args.output_dir) / f"blocked-window-state-{time.time_ns()}.db"
        )
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            base_port=args.transfer_port,
            tp_size=args.tp_size,
            async_transfer=True,
            transfer_poll_interval=args.poll_interval,
        )
        self.tiered = TieredPrefixCoordinator(
            poll_interval=args.poll_interval,
            operation_timeout=300,
        )
        self.shared_handoff = SharedSemanticHandoffCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            poll_interval=args.poll_interval,
            operation_timeout=300,
        )
        self.serial = 0
        self.copy_estimates: dict[int, float] = {}
        self.restore_estimates: dict[tuple[str, int], float] = {}
        self.calibrations: dict[int, dict[str, Any]] = {}

    def next_id(self, scenario: str, prefix_length: int) -> str:
        self.serial += 1
        return f"gap-{scenario}-{prefix_length}-{self.serial}"

    @staticmethod
    def prompt(length: int, salt: int) -> list[int]:
        return [16000 + salt] + [100] * (length - 1)

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    async def calibrate(self, prefix_length: int) -> dict[str, Any]:
        sticky_samples = []
        reroute_samples = []
        completed_tokens = 0
        for calibration_repeat in range(self.args.calibration_repeats):
            await self.clean()
            self.serial += 1
            prompt = self.prompt(prefix_length, self.serial)
            _, first = await timed_generate(
                self.source,
                prompt,
                max_new_tokens=self.args.first_output_tokens,
                rid=(
                    f"calibration-{prefix_length}-{self.serial}-"
                    f"{calibration_repeat}-turn-1"
                ),
            )
            completed = tuple(prompt + first["output_ids"])
            completed_tokens = len(completed)
            next_prompt = list(completed) + [200] * self.args.tool_result_tokens
            sticky_seconds, _ = await timed_generate(
                self.source,
                next_prompt,
                max_new_tokens=self.args.next_output_tokens,
                rid=(
                    f"calibration-{prefix_length}-{self.serial}-"
                    f"{calibration_repeat}-sticky"
                ),
            )
            reroute_seconds, _ = await timed_generate(
                self.destination,
                next_prompt,
                max_new_tokens=self.args.next_output_tokens,
                rid=(
                    f"calibration-{prefix_length}-{self.serial}-"
                    f"{calibration_repeat}-reroute"
                ),
            )
            sticky_samples.append(sticky_seconds)
            reroute_samples.append(reroute_seconds)
        sticky_seconds = statistics.median(sticky_samples)
        reroute_seconds = statistics.median(reroute_samples)
        recompute_seconds = max(0.0, reroute_seconds - sticky_seconds)
        kv_bytes = completed_tokens * self.args.kv_bytes_per_token
        gap_samples = (
            self.args.ttl_calibration_gap_ms
            if self.args.ttl_calibration_gap_ms
            else self.args.gap_ms
        )
        ttl = calibrate_ttl(
            (value / 1000.0 for value in gap_samples),
            recompute_seconds=recompute_seconds,
            kv_gib=kv_bytes / (1024**3),
            hbm_cost_per_gib_second=self.args.ttl_hbm_cost_per_gib_second,
        )
        record = {
            "prefix_length": prefix_length,
            "completed_tokens": completed_tokens,
            "sticky_seconds": sticky_seconds,
            "reroute_seconds": reroute_seconds,
            "sticky_samples_seconds": sticky_samples,
            "reroute_samples_seconds": reroute_samples,
            "recompute_seconds": recompute_seconds,
            "kv_bytes": kv_bytes,
            "ttl_seconds": ttl.ttl_seconds,
            "ttl_expected_cost_seconds": ttl.expected_cost_seconds,
            "ttl_hbm_cost_per_gib_second": ttl.hbm_cost_per_gib_second,
        }
        self.calibrations[prefix_length] = record
        await self.clean()
        return record

    async def _prepare(
        self, scenario: str, prefix_length: int
    ) -> tuple[str, tuple[int, ...], list[int]]:
        await self.clean()
        agent_id = self.next_id(scenario, prefix_length)
        prompt = self.prompt(prefix_length, self.serial)
        _, first = await timed_generate(
            self.source,
            prompt,
            max_new_tokens=self.args.first_output_tokens,
            rid=f"{agent_id}-turn-1",
        )
        completed = tuple(prompt + first["output_ids"])
        next_prompt = list(completed) + [200] * self.args.tool_result_tokens
        if scenario in (
            "on-return",
            "proactive",
            "agentshift",
            "oracle",
            "shared-cas",
        ):
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
            await self.coordinator._ensure_group("engine-a", "engine-b")
        return agent_id, completed, next_prompt

    @staticmethod
    async def _sleep_until(deadline: float) -> None:
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

    @staticmethod
    async def _tool_timer(seconds: float) -> float:
        await asyncio.sleep(seconds)
        return time.perf_counter()

    async def run_one(
        self, scenario: str, prefix_length: int, gap_ms: float, repeat: int
    ) -> dict[str, Any]:
        agent_id, completed, next_prompt = await self._prepare(
            scenario, prefix_length
        )
        requested_gap = gap_ms / 1000.0
        tool_started = time.perf_counter()
        tool_task = asyncio.create_task(self._tool_timer(requested_gap))
        migration = None
        migration_coordinator = self.coordinator
        handoff_started = None
        next_engine = self.destination
        policy_engine = "engine-b"
        tier_out = None
        tier_in = None
        checkpoint_id = None
        shared_checkpoint_bytes = 0
        source_hbm_release_seconds = 0.0

        if scenario == "sticky":
            tool_completed = await tool_task
            destination_ready = tool_completed
            next_engine = self.source
            policy_engine = "engine-a"
        elif scenario == "ttl":
            ttl_seconds = self.calibrations[prefix_length]["ttl_seconds"]
            await asyncio.sleep(min(requested_gap, ttl_seconds))
            if requested_gap > ttl_seconds:
                await flush(self.source)
            tool_completed = await tool_task
            destination_ready = tool_completed
            next_engine = self.source
            policy_engine = "engine-a"
        elif scenario == "reroute":
            tool_completed = await tool_task
            destination_ready = tool_completed
        elif scenario == "agentix":
            calibration = self.calibrations[prefix_length]
            decision = agentix_style_route(
                source_engine="engine-a",
                destination_engine="engine-b",
                source_queue_seconds=self.args.agentix_source_queue_ms / 1000.0,
                destination_queue_seconds=(
                    self.args.agentix_destination_queue_ms / 1000.0
                ),
                next_call_service_seconds=calibration["sticky_seconds"],
                destination_reprefill_seconds=calibration["recompute_seconds"],
            )
            policy_engine = decision.engine
            next_engine = self.source if decision.engine == "engine-a" else self.destination
            tool_completed = await tool_task
            destination_ready = tool_completed
        elif scenario == "on-return":
            tool_completed = await tool_task
            handoff_started = time.perf_counter()
            migration = await self.coordinator.migrate(agent_id, "engine-b")
            destination_ready = time.perf_counter()
        elif scenario in ("proactive", "agentshift"):
            handoff_started = time.perf_counter()
            migration_task = asyncio.create_task(
                self.coordinator.migrate(agent_id, "engine-b")
            )
            tool_completed = await tool_task
            migration = await migration_task
            destination_ready = time.perf_counter()
        elif scenario == "oracle":
            handoff_started = time.perf_counter()
            migration_task = asyncio.create_task(
                self.coordinator.migrate(agent_id, "engine-b")
            )
            tool_completed = await tool_task
            migration = await migration_task
            destination_ready = time.perf_counter()
        elif scenario == "shared-cas":
            handoff_started = time.perf_counter()
            migration_coordinator = self.shared_handoff
            migration_task = asyncio.create_task(
                self.shared_handoff.handoff(agent_id, "engine-b")
            )
            tool_completed = await tool_task
            migration = await migration_task
            destination_ready = time.perf_counter()
        elif scenario == "tokencake-source":
            handoff_started = time.perf_counter()
            await self.source.pin_prefix(agent_id, 1, completed)
            checkpoint_id = f"tc-source-{prefix_length}-{time.time_ns()}"
            tier_out = await self.tiered.run(
                self.source,
                operation="private_offload",
                checkpoint_id=checkpoint_id,
                agent_id=agent_id,
                owner_epoch=1,
                token_ids=completed,
                release_gpu=True,
            )
            source_hbm_release_seconds = time.perf_counter() - tool_started
            restore_estimate = self.restore_estimates.get(
                (scenario, prefix_length), 0.0
            )
            await self._sleep_until(
                tool_started + max(0.0, requested_gap - restore_estimate)
            )
            tier_in = await self.tiered.run(
                self.source,
                operation="private_restore",
                checkpoint_id=checkpoint_id,
                agent_id=agent_id,
                owner_epoch=1,
                token_ids=completed,
                release_gpu=False,
            )
            self.restore_estimates[(scenario, prefix_length)] = tier_in.wall_seconds
            destination_ready = time.perf_counter()
            tool_completed = await tool_task
            next_engine = self.source
            policy_engine = "engine-a"
        elif scenario in ("tokencake-remote", "symphony"):
            handoff_started = time.perf_counter()
            await self.source.pin_prefix(agent_id, 1, completed)
            checkpoint_id = f"{scenario}-{prefix_length}-{time.time_ns()}"
            tier_out = await self.tiered.run(
                self.source,
                operation="shared_export",
                checkpoint_id=checkpoint_id,
                agent_id=agent_id,
                owner_epoch=1,
                token_ids=completed,
                release_gpu=True,
            )
            source_hbm_release_seconds = time.perf_counter() - tool_started
            shared_checkpoint_bytes = sum(
                Path(
                    f"/dev/shm/agentshift-prefixes/{checkpoint_id}.tp{rank}.bin"
                ).stat().st_size
                for rank in range(self.args.tp_size)
            )
            if scenario == "tokencake-remote":
                restore_estimate = self.restore_estimates.get(
                    (scenario, prefix_length), 0.0
                )
                await self._sleep_until(
                    tool_started + max(0.0, requested_gap - restore_estimate)
                )
            tier_in = await self.tiered.run(
                self.destination,
                operation="shared_import",
                checkpoint_id=checkpoint_id,
                agent_id=agent_id,
                owner_epoch=1,
                token_ids=completed,
                release_gpu=False,
            )
            self.restore_estimates[(scenario, prefix_length)] = tier_in.wall_seconds
            destination_ready = time.perf_counter()
            tool_completed = await tool_task
        else:
            raise ValueError(f"unknown scenario: {scenario}")

        exposed_migration = max(0.0, destination_ready - tool_completed)
        if scenario == "oracle":
            exposed_migration = max(
                0.0,
                migration.transfer_seconds - (tool_completed - tool_started),
            )
        next_turn_seconds, next_result = await timed_generate(
            next_engine,
            next_prompt,
            max_new_tokens=self.args.next_output_tokens,
            rid=f"{agent_id}-turn-2",
        )
        post_tool_seconds = exposed_migration + next_turn_seconds
        actual_gap = tool_completed - tool_started
        handoff_wall = (
            destination_ready - handoff_started if handoff_started is not None else 0.0
        )
        overlap_seconds = (
            max(0.0, min(tool_completed, destination_ready) - handoff_started)
            if handoff_started is not None
            else 0.0
        )

        if migration is not None:
            previous = self.copy_estimates.get(prefix_length)
            observed = migration.transfer_seconds
            self.copy_estimates[prefix_length] = (
                observed if previous is None else 0.5 * previous + 0.5 * observed
            )

        record = {
            "scenario": scenario,
            "prefix_length": prefix_length,
            "gap_ms": gap_ms,
            "repeat": repeat,
            "actual_gap_seconds": actual_gap,
            "exposed_migration_seconds": exposed_migration,
            "next_turn_seconds": next_turn_seconds,
            "post_tool_seconds": post_tool_seconds,
            "cached_tokens": int(next_result["meta_info"]["cached_tokens"]),
            "selected_engine": policy_engine,
            "calibrated_ttl_seconds": self.calibrations.get(
                prefix_length, {}
            ).get("ttl_seconds", 0.0),
            "migration_wall_seconds": migration.transfer_seconds
            if migration
            else 0.0,
            "migration_worker_seconds": migration.worker_transfer_seconds
            if migration
            else 0.0,
            "migration_queue_seconds": migration.queue_seconds if migration else 0.0,
            "migration_bytes": migration.bytes_transferred if migration else 0,
            "handoff_wall_seconds": handoff_wall,
            "hidden_ratio": overlap_seconds / handoff_wall if handoff_wall else 0.0,
            "tier_out_wall_seconds": tier_out.wall_seconds if tier_out else 0.0,
            "tier_out_worker_seconds": (
                tier_out.worker_seconds if tier_out else 0.0
            ),
            "tier_in_wall_seconds": tier_in.wall_seconds if tier_in else 0.0,
            "tier_in_worker_seconds": tier_in.worker_seconds if tier_in else 0.0,
            "tier_total_bytes": (
                (tier_out.bytes_transferred if tier_out else 0)
                + (tier_in.bytes_transferred if tier_in else 0)
            ),
            "shared_checkpoint_bytes": shared_checkpoint_bytes,
            "source_hbm_release_seconds": source_hbm_release_seconds,
            "ownership_relief_seconds": (
                destination_ready - tool_started if migration else 0.0
            ),
            "oracle_analytical_bound": scenario == "oracle",
        }

        if migration is not None:
            await migration_coordinator.acknowledge_destination(
                migration.migration_id
            )
            record["source_hbm_release_seconds"] = (
                time.perf_counter() - tool_started
            )
            await self.destination.release_prefix(agent_id, migration.new_epoch)
        if tier_in is not None:
            await next_engine.release_prefix(agent_id, 1)
            await self.tiered.cleanup(
                next_engine,
                tier_in,
                drop_checkpoint=scenario in ("tokencake-remote", "symphony"),
            )
        if tier_out is not None:
            await self.tiered.cleanup(
                self.source,
                tier_out,
                drop_checkpoint=scenario == "tokencake-source",
            )
        await self.clean()
        return record


async def main(args) -> None:
    benchmark = BlockedWindowBenchmark(args)
    records: list[dict[str, Any]] = []
    warmups: list[dict[str, Any]] = []
    for prefix_length in args.prefix_lengths:
        calibration = await benchmark.calibrate(prefix_length)
        print(json.dumps({"calibration": calibration}, sort_keys=True), flush=True)
        scenario_set = set(args.scenarios)
        warmup_scenarios = []
        if scenario_set.intersection({"on-return", "proactive", "agentshift", "oracle"}):
            warmup_scenarios.append("agentshift")
        if "shared-cas" in scenario_set:
            warmup_scenarios.append("shared-cas")
        if "tokencake-source" in scenario_set:
            warmup_scenarios.append("tokencake-source")
        if scenario_set.intersection({"tokencake-remote", "symphony"}):
            warmup_scenarios.append("symphony")
        for warmup_scenario in warmup_scenarios:
            warmup = await benchmark.run_one(
                warmup_scenario, prefix_length, 0.0, -1
            )
            warmups.append(warmup)
            print(
                json.dumps(
                    {
                        "warmup": warmup_scenario,
                        "prefix_length": prefix_length,
                        "handoff_wall_seconds": warmup["handoff_wall_seconds"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        for gap_ms in args.gap_ms:
            for repeat in range(args.repeats):
                scenario_order = list(args.scenarios)
                random.Random(
                    args.seed + prefix_length * 1009 + int(gap_ms * 10) + repeat
                ).shuffle(scenario_order)
                for scenario in scenario_order:
                    record = await benchmark.run_one(
                        scenario, prefix_length, gap_ms, repeat
                    )
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
    output = {
        "baseline_implementation": "mechanism-equivalent",
        "config": vars(args),
        "calibrations": benchmark.calibrations,
        "warmups": warmups,
        "state_db": str(benchmark.state_path),
        "records": records,
    }
    output_path = Path(args.output_dir) / f"blocked-window-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=[
            "sticky",
            "reroute",
            "agentix",
            "ttl",
            "tokencake-source",
            "tokencake-remote",
            "symphony",
            "shared-cas",
            "on-return",
            "agentshift",
            "oracle",
        ],
    )
    parser.add_argument(
        "--prefix-lengths", type=int, nargs="+", default=[1024, 4096, 16384, 32768]
    )
    parser.add_argument(
        "--gap-ms",
        type=float,
        nargs="+",
        default=[0, 10, 25, 50, 100, 250, 500, 1000],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--next-output-tokens", type=int, default=1)
    parser.add_argument("--ttl-calibration-gap-ms", type=float, nargs="+")
    parser.add_argument("--ttl-hbm-cost-per-gib-second", type=float, default=0.25)
    parser.add_argument("--kv-bytes-per-token", type=int, default=147456)
    parser.add_argument("--agentix-source-queue-ms", type=float, default=0.0)
    parser.add_argument("--agentix-destination-queue-ms", type=float, default=0.0)
    parser.add_argument("--poll-interval", type=float, default=0.005)
    parser.add_argument("--transfer-port", type=int, default=29950)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
