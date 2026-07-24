from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_e2e import flush, timed_generate

from agentshift.controller.baselines import calibrate_ttl
from agentshift.controller.migration import MigrationCoordinator, MigrationResult
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


class TTLPressureBenchmark:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.state_path = Path(args.output_dir) / f"ttl-pressure-state-{time.time_ns()}.db"
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            base_port=args.transfer_port,
            tp_size=args.tp_size,
            async_transfer=True,
            transfer_poll_interval=args.poll_interval,
        )
        self.serial = 0

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    @staticmethod
    def prompt(length: int, salt: int, token: int = 100) -> list[int]:
        return [30000 + salt] + [token] * (length - 1)

    @staticmethod
    async def sleep_until(deadline: float) -> None:
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

    def load_gap_samples(self) -> tuple[list[float], list[float]]:
        payload = json.loads(Path(self.args.tool_trace).read_text())
        values = [
            float(record["tool_seconds"])
            for record in payload["records"]
            if record["scenario"] == "sticky"
        ]
        if len(values) < 3:
            raise ValueError("tool trace needs at least three sticky samples")
        split = max(1, int(len(values) * self.args.training_fraction))
        return values[:split], values[split:]

    async def calibrate(self) -> dict[str, Any]:
        await self.clean()
        warm = []
        cold = []
        completed_tokens = 0
        for repeat in range(self.args.calibration_repeats):
            self.serial += 1
            prompt = self.prompt(self.args.prefix_length, self.serial)
            _, first = await timed_generate(
                self.source,
                prompt,
                max_new_tokens=4,
                rid=f"ttl-pressure-cal-{self.serial}-first",
            )
            completed = tuple(prompt + first["output_ids"])
            completed_tokens = len(completed)
            next_prompt = list(completed) + [200] * self.args.tool_result_tokens
            warm_seconds, _ = await timed_generate(
                self.source,
                next_prompt,
                max_new_tokens=1,
                rid=f"ttl-pressure-cal-{self.serial}-warm",
            )
            cold_seconds, _ = await timed_generate(
                self.destination,
                next_prompt,
                max_new_tokens=1,
                rid=f"ttl-pressure-cal-{self.serial}-cold",
            )
            warm.append(warm_seconds)
            cold.append(cold_seconds)
            await self.clean()
        recompute = max(0.0, statistics.median(cold) - statistics.median(warm))
        train, test = self.load_gap_samples()
        ttl = calibrate_ttl(
            train,
            recompute_seconds=recompute,
            kv_gib=(completed_tokens * self.args.kv_bytes_per_token) / (1024**3),
            hbm_cost_per_gib_second=self.args.hbm_cost_per_gib_second,
        )
        return {
            "ttl_seconds": ttl.ttl_seconds,
            "expected_training_cost_seconds": ttl.expected_cost_seconds,
            "recompute_seconds": recompute,
            "training_samples": len(train),
            "test_samples": len(test),
            "training_coverage": statistics.mean(value <= ttl.ttl_seconds for value in train),
            "heldout_coverage": statistics.mean(value <= ttl.ttl_seconds for value in test),
            "training_gap_p50_seconds": statistics.median(train),
            "heldout_gap_p50_seconds": statistics.median(test),
        }

    async def prepare_agents(self, scenario: str, repeat: int):
        await self.clean()
        agent_ids = []
        prompts = []
        for _ in range(self.args.agent_count):
            self.serial += 1
            agent_ids.append(f"ttl-pressure-{scenario}-{repeat}-{self.serial}")
            prompts.append(self.prompt(self.args.prefix_length, self.serial))
        first = await asyncio.gather(
            *[
                timed_generate(
                    self.source,
                    prompt,
                    max_new_tokens=4,
                    rid=f"{agent_id}-first",
                )
                for agent_id, prompt in zip(agent_ids, prompts)
            ]
        )
        completed = [
            tuple(prompt + result["output_ids"])
            for prompt, (_, result) in zip(prompts, first)
        ]
        return agent_ids, completed

    async def run_one(
        self,
        scenario: str,
        gap_seconds: float,
        ttl_seconds: float,
        repeat: int,
    ) -> dict[str, Any]:
        agent_ids, completed = await self.prepare_agents(scenario, repeat)
        half = self.args.agent_count // 2
        selected = [index < half for index in range(self.args.agent_count)]
        clients = [
            self.destination
            if scenario in ("agentshift", "ttl-reroute") and selected[index]
            else self.source
            for index in range(self.args.agent_count)
        ]
        pinned = scenario in ("ttl", "ttl-reroute")
        source_retained = [
            pinned or (scenario == "agentshift" and not selected[index])
            for index in range(self.args.agent_count)
        ]
        for retain, agent_id, tokens in zip(source_retained, agent_ids, completed):
            if retain:
                await self.source.pin_prefix(agent_id, 1, tokens)

        ready = [asyncio.Event() for _ in agent_ids]
        migrations: list[MigrationResult] = []
        tool_started = time.perf_counter()
        if scenario == "agentshift":
            async def migrate_selected():
                for index, move in enumerate(selected):
                    if not move:
                        ready[index].set()
                for index, (agent_id, tokens) in enumerate(zip(agent_ids, completed)):
                    if not selected[index]:
                        continue
                    self.store.register_agent(
                        AgentContinuation(
                            agent_id,
                            1,
                            "engine-a",
                            1,
                            tokens,
                            f"tool-{agent_id}",
                        )
                    )
                    migrations.append(
                        await self.coordinator.migrate(agent_id, "engine-b")
                    )
                    ready[index].set()

            migration_task = asyncio.create_task(migrate_selected())
        else:
            migration_task = None
            for event in ready:
                event.set()

        pressure_started = time.perf_counter()
        async def run_pressure(index: int, salt: int):
            elapsed, result = await timed_generate(
                self.source,
                self.prompt(
                    self.args.pressure_prefix_length,
                    salt,
                    token=101,
                ),
                max_new_tokens=1,
                rid=f"pressure-{scenario}-{repeat}-{index}-{salt}",
            )
            return {
                "elapsed": elapsed,
                "completed": time.perf_counter(),
                "cached_tokens": int(result["meta_info"]["cached_tokens"]),
            }

        pressure_tasks = []
        for index in range(self.args.pressure_requests):
            self.serial += 1
            pressure_tasks.append(
                asyncio.create_task(
                    run_pressure(index, self.serial)
                )
            )

        expired = pinned and ttl_seconds < gap_seconds
        if expired:
            async def expire_ttl():
                await self.sleep_until(tool_started + ttl_seconds)
                for agent_id in agent_ids:
                    await self.source.release_prefix(
                        agent_id,
                        1,
                        evict_after_release=False,
                    )

            expiry_task = asyncio.create_task(expire_ttl())
        else:
            expiry_task = None

        async def next_turn(index: int):
            tool_return = tool_started + gap_seconds
            await self.sleep_until(tool_return)
            await ready[index].wait()
            started = time.perf_counter()
            elapsed, result = await timed_generate(
                clients[index],
                list(completed[index]) + [200] * self.args.tool_result_tokens,
                max_new_tokens=1,
                rid=f"{agent_ids[index]}-next",
            )
            return {
                "tool_return": tool_return,
                "started": started,
                "completed": time.perf_counter(),
                "elapsed": elapsed,
                "cached_tokens": int(result["meta_info"]["cached_tokens"]),
            }

        next_tasks = [
            asyncio.create_task(next_turn(index))
            for index in range(self.args.agent_count)
        ]
        if migration_task is not None:
            await migration_task
        if expiry_task is not None:
            await expiry_task
        next_results = await asyncio.gather(*next_tasks)
        pressure_results = await asyncio.gather(*pressure_tasks)

        for retain, agent_id in zip(source_retained, agent_ids):
            if retain and not expired:
                await self.source.release_prefix(
                    agent_id,
                    1,
                    evict_after_release=False,
                )
        for migration in migrations:
            await self.coordinator.acknowledge_destination(migration.migration_id)
            await self.destination.release_prefix(
                migration.agent_id, migration.new_epoch
            )
        post_tool = [
            result["completed"] - result["tool_return"] for result in next_results
        ]
        hits = [result["cached_tokens"] for result in next_results]
        destination_hits = [
            hit >= len(tokens)
            for client, hit, tokens in zip(clients, hits, completed)
            if client is self.destination
        ]
        record = {
            "scenario": scenario,
            "repeat": repeat,
            "prefix_length": self.args.prefix_length,
            "agent_count": self.args.agent_count,
            "gap_seconds": gap_seconds,
            "ttl_seconds": ttl_seconds,
            "ttl_expired": expired,
            "pressure_requests": self.args.pressure_requests,
            "pressure_prefix_length": self.args.pressure_prefix_length,
            "pressure_makespan_seconds": max(
                result["completed"] for result in pressure_results
            )
            - pressure_started,
            "pressure_mean_request_seconds": statistics.mean(
                result["elapsed"] for result in pressure_results
            ),
            "agent_makespan_seconds": max(
                result["completed"] for result in next_results
            )
            - (tool_started + gap_seconds),
            "post_tool_mean_seconds": statistics.mean(post_tool),
            "post_tool_p95_seconds": percentile(post_tool, 0.95),
            "full_prefix_hit_rate": statistics.mean(
                hit >= len(tokens) for hit, tokens in zip(hits, completed)
            ),
            "destination_full_hit_rate": (
                statistics.mean(destination_hits) if destination_hits else None
            ),
            "relocated_fraction": statistics.mean(
                client is self.destination for client in clients
            ),
            "ownership_relocated_fraction": len(migrations) / self.args.agent_count,
            "transfer_bytes": sum(item.bytes_transferred for item in migrations),
        }
        await self.clean()
        return record


async def main(args) -> None:
    benchmark = TTLPressureBenchmark(args)
    calibration = await benchmark.calibrate()
    print(json.dumps({"calibration": calibration}, sort_keys=True), flush=True)
    records = []
    for gap_seconds in args.gap_seconds:
        for repeat in range(args.repeats):
            for scenario in args.scenarios:
                record = await benchmark.run_one(
                    scenario,
                    gap_seconds,
                    calibration["ttl_seconds"],
                    repeat,
                )
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
    output = {
        "baseline_implementation": "mechanism-equivalent",
        "config": vars(args),
        "calibration": calibration,
        "state_db": str(benchmark.state_path),
        "records": records,
    }
    output_path = Path(args.output_dir) / f"ttl-pressure-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--scenarios", nargs="+", default=["sticky", "ttl", "ttl-reroute", "agentshift"]
    )
    parser.add_argument("--prefix-length", type=int, default=16384)
    parser.add_argument("--agent-count", type=int, default=8)
    parser.add_argument("--gap-seconds", type=float, nargs="+", default=[0.4, 0.5])
    parser.add_argument("--pressure-requests", type=int, default=12)
    parser.add_argument("--pressure-prefix-length", type=int, default=16384)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--tool-trace", default="results/real-tools-all-1784548799599095961.json")
    parser.add_argument("--training-fraction", type=float, default=0.67)
    parser.add_argument("--hbm-cost-per-gib-second", type=float, default=0.25)
    parser.add_argument("--kv-bytes-per-token", type=int, default=147456)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=0.005)
    parser.add_argument("--transfer-port", type=int, default=30300)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
