from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from benchmark_e2e import flush, timed_generate
from benchmark_real_tools import tool_specs

from agentshift.controller.baselines import agentix_style_route, calibrate_ttl
from agentshift.controller.migration import MigrationCoordinator
from agentshift.controller.tiered import TieredPrefixCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


async def run_tool(command: list[str], cwd: str):
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    finished = time.perf_counter()
    if process.returncode != 0:
        raise RuntimeError(
            f"tool failed ({process.returncode}): {' '.join(command)}\n"
            f"{stderr.decode(errors='replace')}"
        )
    return finished - started, finished, stdout, stderr


class RealToolAllBaselineBenchmark:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.state_path = (
            Path(args.output_dir) / f"real-tools-all-state-{time.time_ns()}.db"
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
        self.serial = 0
        self.calibrations: dict[int, dict[str, Any]] = {}
        self.tool_predictions: dict[str, float] = {}
        self.copy_estimates: dict[int, float] = {}
        self.restore_estimates: dict[tuple[str, int], float] = {}

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    @staticmethod
    def prompt(length: int, salt: int) -> list[int]:
        return [23000 + salt] + [100] * (length - 1)

    @staticmethod
    async def sleep_until(deadline: float) -> None:
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

    async def measure_tools(self) -> list[dict[str, Any]]:
        measurements = []
        for tool_name, command in tool_specs(self.args.python):
            samples = []
            output_bytes = 0
            for _ in range(self.args.tool_calibration_repeats):
                elapsed, _, stdout, stderr = await run_tool(
                    command, self.args.tool_cwd
                )
                samples.append(elapsed)
                output_bytes = len(stdout) + len(stderr)
            prediction = statistics.median(samples)
            self.tool_predictions[tool_name] = prediction
            measurements.append(
                {
                    "tool": tool_name,
                    "command": command,
                    "samples_seconds": samples,
                    "prediction_seconds": prediction,
                    "output_bytes": output_bytes,
                }
            )
        return measurements

    async def calibrate_prefix(self, prefix_length: int) -> dict[str, Any]:
        sticky_samples = []
        reroute_samples = []
        completed_tokens = 0
        for repeat in range(self.args.calibration_repeats):
            await self.clean()
            self.serial += 1
            prompt = self.prompt(prefix_length, self.serial)
            _, first = await timed_generate(
                self.source,
                prompt,
                max_new_tokens=self.args.first_output_tokens,
                rid=f"real-cal-{prefix_length}-{self.serial}-{repeat}-turn-1",
            )
            completed = tuple(prompt + first["output_ids"])
            completed_tokens = len(completed)
            next_prompt = list(completed) + [200] * 32
            sticky, _ = await timed_generate(
                self.source,
                next_prompt,
                max_new_tokens=1,
                rid=f"real-cal-{prefix_length}-{self.serial}-{repeat}-sticky",
            )
            reroute, _ = await timed_generate(
                self.destination,
                next_prompt,
                max_new_tokens=1,
                rid=f"real-cal-{prefix_length}-{self.serial}-{repeat}-reroute",
            )
            sticky_samples.append(sticky)
            reroute_samples.append(reroute)
        sticky = statistics.median(sticky_samples)
        reroute = statistics.median(reroute_samples)
        recompute = max(0.0, reroute - sticky)
        kv_bytes = completed_tokens * self.args.kv_bytes_per_token
        ttl = calibrate_ttl(
            self.tool_predictions.values(),
            recompute_seconds=recompute,
            kv_gib=kv_bytes / (1024**3),
            hbm_cost_per_gib_second=self.args.ttl_hbm_cost_per_gib_second,
        )
        record = {
            "prefix_length": prefix_length,
            "completed_tokens": completed_tokens,
            "sticky_seconds": sticky,
            "reroute_seconds": reroute,
            "recompute_seconds": recompute,
            "kv_bytes": kv_bytes,
            "ttl_seconds": ttl.ttl_seconds,
            "ttl_expected_cost_seconds": ttl.expected_cost_seconds,
        }
        self.calibrations[prefix_length] = record
        await self.clean()
        return record

    async def prepare(self, scenario: str, prefix_length: int):
        await self.clean()
        self.serial += 1
        agent_id = f"real-all-{scenario}-{prefix_length}-{self.serial}"
        prompt = self.prompt(prefix_length, self.serial)
        _, first = await timed_generate(
            self.source,
            prompt,
            max_new_tokens=self.args.first_output_tokens,
            rid=f"{agent_id}-turn-1",
        )
        completed = tuple(prompt + first["output_ids"])
        if scenario in ("on-return", "agentshift", "oracle"):
            self.store.register_agent(
                AgentContinuation(
                    agent_id,
                    1,
                    "engine-a",
                    1,
                    completed,
                    f"tool-{agent_id}",
                )
            )
        return agent_id, completed

    async def run_one(
        self,
        scenario: str,
        tool_name: str,
        command: list[str],
        prefix_length: int,
        repeat: int,
    ) -> dict[str, Any]:
        agent_id, completed = await self.prepare(scenario, prefix_length)
        tool_started = time.perf_counter()
        tool_task = asyncio.create_task(run_tool(command, self.args.tool_cwd))
        migration = None
        tier_out = None
        tier_in = None
        checkpoint_id = None
        handoff_started = None
        source_hbm_release_seconds = 0.0
        next_engine = self.destination
        selected_engine = "engine-b"

        if scenario == "sticky":
            tool_seconds, tool_completed, stdout, stderr = await tool_task
            destination_ready = tool_completed
            next_engine = self.source
            selected_engine = "engine-a"
        elif scenario == "reroute":
            tool_seconds, tool_completed, stdout, stderr = await tool_task
            destination_ready = tool_completed
        elif scenario == "agentix":
            calibration = self.calibrations[prefix_length]
            decision = agentix_style_route(
                source_engine="engine-a",
                destination_engine="engine-b",
                source_queue_seconds=self.args.agentix_source_queue_ms / 1000,
                destination_queue_seconds=(
                    self.args.agentix_destination_queue_ms / 1000
                ),
                next_call_service_seconds=calibration["sticky_seconds"],
                destination_reprefill_seconds=calibration["recompute_seconds"],
            )
            selected_engine = decision.engine
            next_engine = self.source if decision.engine == "engine-a" else self.destination
            tool_seconds, tool_completed, stdout, stderr = await tool_task
            destination_ready = tool_completed
        elif scenario == "ttl":
            ttl_seconds = self.calibrations[prefix_length]["ttl_seconds"]
            done, _ = await asyncio.wait({tool_task}, timeout=ttl_seconds)
            if not done:
                await flush(self.source)
            tool_seconds, tool_completed, stdout, stderr = await tool_task
            destination_ready = tool_completed
            next_engine = self.source
            selected_engine = "engine-a"
        elif scenario == "on-return":
            tool_seconds, tool_completed, stdout, stderr = await tool_task
            handoff_started = time.perf_counter()
            migration = await self.coordinator.migrate(agent_id, "engine-b")
            destination_ready = time.perf_counter()
        elif scenario == "agentshift":
            handoff_started = time.perf_counter()
            migration_task = asyncio.create_task(
                self.coordinator.migrate(agent_id, "engine-b")
            )
            tool_seconds, tool_completed, stdout, stderr = await tool_task
            migration = await migration_task
            destination_ready = time.perf_counter()
        elif scenario == "oracle":
            estimate = self.copy_estimates.get(prefix_length, 0.0)
            prediction = self.tool_predictions[tool_name]
            await self.sleep_until(tool_started + max(0.0, prediction - estimate))
            handoff_started = time.perf_counter()
            migration_task = asyncio.create_task(
                self.coordinator.migrate(agent_id, "engine-b")
            )
            tool_seconds, tool_completed, stdout, stderr = await tool_task
            migration = await migration_task
            destination_ready = time.perf_counter()
        elif scenario == "tokencake-source":
            handoff_started = time.perf_counter()
            await self.source.pin_prefix(agent_id, 1, completed)
            checkpoint_id = f"real-tcs-{prefix_length}-{time.time_ns()}"
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
            prediction = self.tool_predictions[tool_name]
            restore = self.restore_estimates.get((scenario, prefix_length), 0.0)
            await self.sleep_until(tool_started + max(0.0, prediction - restore))
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
            tool_seconds, tool_completed, stdout, stderr = await tool_task
            next_engine = self.source
            selected_engine = "engine-a"
        elif scenario in ("tokencake-remote", "symphony"):
            handoff_started = time.perf_counter()
            await self.source.pin_prefix(agent_id, 1, completed)
            checkpoint_id = f"real-{scenario}-{prefix_length}-{time.time_ns()}"
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
            if scenario == "tokencake-remote":
                prediction = self.tool_predictions[tool_name]
                restore = self.restore_estimates.get((scenario, prefix_length), 0.0)
                await self.sleep_until(tool_started + max(0.0, prediction - restore))
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
            tool_seconds, tool_completed, stdout, stderr = await tool_task
        else:
            raise ValueError(f"unknown scenario: {scenario}")

        if migration is not None:
            previous = self.copy_estimates.get(prefix_length)
            observed = migration.transfer_seconds
            self.copy_estimates[prefix_length] = (
                observed if previous is None else (previous + observed) / 2
            )
        exposed = max(0.0, destination_ready - tool_completed)
        result_tokens = max(
            16,
            min(
                self.args.max_tool_result_tokens,
                (len(stdout) + len(stderr) + 3) // 4,
            ),
        )
        next_seconds, second = await timed_generate(
            next_engine,
            list(completed) + [200] * result_tokens,
            max_new_tokens=1,
            rid=f"{agent_id}-turn-2",
        )
        handoff_seconds = (
            destination_ready - handoff_started if handoff_started else 0.0
        )
        record = {
            "scenario": scenario,
            "tool": tool_name,
            "command": command,
            "prefix_length": prefix_length,
            "repeat": repeat,
            "tool_seconds": tool_seconds,
            "tool_prediction_seconds": self.tool_predictions[tool_name],
            "tool_output_bytes": len(stdout) + len(stderr),
            "tool_result_tokens": result_tokens,
            "selected_engine": selected_engine,
            "migration_seconds": migration.transfer_seconds if migration else 0.0,
            "tier_out_seconds": tier_out.wall_seconds if tier_out else 0.0,
            "tier_in_seconds": tier_in.wall_seconds if tier_in else 0.0,
            "tier_total_bytes": (
                (tier_out.bytes_transferred if tier_out else 0)
                + (tier_in.bytes_transferred if tier_in else 0)
            ),
            "handoff_seconds": handoff_seconds,
            "exposed_migration_seconds": exposed,
            "next_turn_seconds": next_seconds,
            "post_tool_seconds": exposed + next_seconds,
            "cached_tokens": int(second["meta_info"]["cached_tokens"]),
            "full_prefix_hit": int(second["meta_info"]["cached_tokens"])
            >= len(completed),
            "source_hbm_release_seconds": source_hbm_release_seconds,
        }
        if migration is not None:
            await self.coordinator.acknowledge_destination(migration.migration_id)
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
    benchmark = RealToolAllBaselineBenchmark(args)
    tool_measurements = await benchmark.measure_tools()
    records = []
    warmups = []
    for prefix_length in args.prefix_lengths:
        calibration = await benchmark.calibrate_prefix(prefix_length)
        print(json.dumps({"calibration": calibration}, sort_keys=True), flush=True)
        warmup_tool, warmup_command = tool_specs(args.python)[0]
        for scenario in ("agentshift", "tokencake-source", "symphony"):
            warmup = await benchmark.run_one(
                scenario,
                warmup_tool,
                warmup_command,
                prefix_length,
                -1,
            )
            warmups.append(warmup)
        for repeat in range(args.repeats):
            for tool_name, command in tool_specs(args.python):
                scenario_order = list(args.scenarios)
                random.Random(
                    args.seed + prefix_length * 1009 + repeat * 17 + len(tool_name)
                ).shuffle(scenario_order)
                for scenario in scenario_order:
                    record = await benchmark.run_one(
                        scenario,
                        tool_name,
                        command,
                        prefix_length,
                        repeat,
                    )
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
    output = {
        "baseline_implementation": "mechanism-equivalent",
        "config": vars(args),
        "tool_measurements": tool_measurements,
        "calibrations": benchmark.calibrations,
        "warmups": warmups,
        "state_db": str(benchmark.state_path),
        "records": records,
    }
    output_path = Path(args.output_dir) / f"real-tools-all-{time.time_ns()}.json"
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
            "on-return",
            "agentshift",
            "oracle",
        ],
    )
    parser.add_argument(
        "--prefix-lengths", type=int, nargs="+", default=[4096, 16384, 32768]
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--tool-calibration-repeats", type=int, default=3)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--max-tool-result-tokens", type=int, default=1024)
    parser.add_argument("--ttl-hbm-cost-per-gib-second", type=float, default=0.25)
    parser.add_argument("--kv-bytes-per-token", type=int, default=147456)
    parser.add_argument("--agentix-source-queue-ms", type=float, default=0.0)
    parser.add_argument("--agentix-destination-queue-ms", type=float, default=0.0)
    parser.add_argument("--poll-interval", type=float, default=0.005)
    parser.add_argument("--transfer-port", type=int, default=30100)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--tool-cwd", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
