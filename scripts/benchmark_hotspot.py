from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_e2e import flush, timed_generate

from agentshift.controller.migration import MigrationCoordinator, MigrationResult
from agentshift.controller.tiered import TierOperationResult, TieredPrefixCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


class HotspotBenchmark:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.state_path = Path(args.output_dir) / f"hotspot-state-{time.time_ns()}.db"
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
        self.calibrations: dict[int, dict[str, float]] = {}
        self.tokencake_warmups: list[dict[str, Any]] = []

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    @staticmethod
    def prompt(length: int, salt: int) -> list[int]:
        return [26000 + salt] + [100] * (length - 1)

    @staticmethod
    async def sleep_until(deadline: float) -> None:
        await asyncio.sleep(max(0.0, deadline - time.perf_counter()))

    async def calibrate(self, prefix_length: int) -> dict[str, float]:
        warm_samples = []
        cold_samples = []
        for repeat in range(self.args.calibration_repeats):
            await self.clean()
            self.serial += 1
            prompt = self.prompt(prefix_length, self.serial)
            _, first = await timed_generate(
                self.source,
                prompt,
                max_new_tokens=self.args.first_output_tokens,
                rid=f"hotspot-cal-{prefix_length}-{self.serial}-first",
            )
            completed = tuple(prompt + first["output_ids"])
            next_prompt = list(completed) + [200] * self.args.tool_result_tokens
            warm, _ = await timed_generate(
                self.source,
                next_prompt,
                max_new_tokens=self.args.burst_output_tokens,
                rid=f"hotspot-cal-{prefix_length}-{repeat}-warm",
            )
            cold, _ = await timed_generate(
                self.destination,
                next_prompt,
                max_new_tokens=self.args.burst_output_tokens,
                rid=f"hotspot-cal-{prefix_length}-{repeat}-cold",
            )
            warm_samples.append(warm)
            cold_samples.append(cold)
        result = {
            "warm_service_seconds": statistics.median(warm_samples),
            "cold_service_seconds": statistics.median(cold_samples),
            "recompute_seconds": max(
                0.0, statistics.median(cold_samples) - statistics.median(warm_samples)
            ),
        }
        self.calibrations[prefix_length] = result
        await self.clean()
        return result

    def return_offsets(self, count: int, pattern: str, spread_ms: float) -> list[float]:
        spread = spread_ms / 1000.0
        if pattern == "simultaneous":
            return [0.0] * count
        if pattern == "uniform":
            if count == 1:
                return [0.0]
            return [spread * index / (count - 1) for index in range(count)]
        if pattern == "heavy-tail":
            rng = random.Random(self.args.seed + self.serial)
            samples = [min(10.0, rng.paretovariate(2.0) - 1.0) for _ in range(count)]
            maximum = max(samples) or 1.0
            return [spread * value / maximum for value in samples]
        if pattern == "multi-wave":
            return [0.0 if index < count // 2 else spread for index in range(count)]
        raise ValueError(f"unknown return pattern: {pattern}")

    def agentix_assignment(self, count: int, prefix_length: int) -> list[bool]:
        calibration = self.calibrations[prefix_length]
        warm = calibration["warm_service_seconds"]
        cold = calibration["cold_service_seconds"]
        source_work = 0.0
        destination_work = 0.0
        use_destination = []
        for _ in range(count):
            source_finish = source_work + warm
            destination_finish = destination_work + cold
            move = destination_finish < source_finish
            use_destination.append(move)
            if move:
                destination_work = destination_finish
            else:
                source_work = source_finish
        return use_destination

    async def prepare_agents(
        self,
        scenario: str,
        prefix_length: int,
        count: int,
        repeat: int,
    ) -> tuple[list[str], list[tuple[int, ...]]]:
        await self.clean()
        agent_ids = []
        prompts = []
        for index in range(count):
            self.serial += 1
            agent_ids.append(
                f"hotspot-{scenario}-{prefix_length}-{count}-{repeat}-{self.serial}"
            )
            prompts.append(self.prompt(prefix_length, self.serial))
        first_results = await asyncio.gather(
            *[
                timed_generate(
                    self.source,
                    prompt,
                    max_new_tokens=self.args.first_output_tokens,
                    rid=f"{agent_id}-turn-1",
                )
                for agent_id, prompt in zip(agent_ids, prompts)
            ]
        )
        completed = [
            tuple(prompt + result["output_ids"])
            for prompt, (_, result) in zip(prompts, first_results)
        ]
        return agent_ids, completed

    async def _direct_prepare(
        self,
        agent_ids: list[str],
        completed: list[tuple[int, ...]],
        selected: list[bool],
        ready_events: list[asyncio.Event],
        *,
        start_at: float,
    ) -> tuple[list[MigrationResult], float]:
        await self.sleep_until(start_at)
        migrations = []
        for index, move in enumerate(selected):
            if not move:
                ready_events[index].set()
        for index, (agent_id, tokens, move) in enumerate(
            zip(agent_ids, completed, selected)
        ):
            if not move:
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
            migration = await self.coordinator.migrate(agent_id, "engine-b")
            migrations.append(migration)
            ready_events[index].set()
        return migrations, time.perf_counter()

    async def _tokencake_source_prepare(
        self,
        agent_ids: list[str],
        completed: list[tuple[int, ...]],
        ready_events: list[asyncio.Event],
    ) -> tuple[list[TierOperationResult], list[TierOperationResult], float]:
        for agent_id, tokens in zip(agent_ids, completed):
            await self.source.pin_prefix(agent_id, 1, tokens)
        offloads = []
        for agent_id, tokens in zip(agent_ids, completed):
            offloads.append(
                await self.tiered.run(
                    self.source,
                    operation="private_offload",
                    checkpoint_id=f"tc-{agent_id}",
                    agent_id=agent_id,
                    owner_epoch=1,
                    token_ids=tokens,
                    release_gpu=True,
                )
            )
        restores = []
        for index, (agent_id, tokens) in enumerate(zip(agent_ids, completed)):
            restores.append(
                await self.tiered.run(
                    self.source,
                    operation="private_restore",
                    checkpoint_id=f"tc-{agent_id}",
                    agent_id=agent_id,
                    owner_epoch=1,
                    token_ids=tokens,
                    release_gpu=False,
                )
            )
            ready_events[index].set()
        return offloads, restores, time.perf_counter()

    async def warm_tokencake(self, prefix_length: int, count: int) -> None:
        agent_ids, completed = await self.prepare_agents(
            "tokencake-warmup", prefix_length, count, -1
        )
        ready_events = [asyncio.Event() for _ in agent_ids]
        started = time.perf_counter()
        offloads, restores, _ = await self._tokencake_source_prepare(
            agent_ids, completed, ready_events
        )
        elapsed = time.perf_counter() - started
        for agent_id in agent_ids:
            await self.source.release_prefix(agent_id, 1)
        for result in restores:
            await self.tiered.cleanup(self.source, result)
        for result in offloads:
            await self.tiered.cleanup(self.source, result, drop_checkpoint=True)
        self.tokencake_warmups.append(
            {
                "prefix_length": prefix_length,
                "agent_count": count,
                "wall_seconds": elapsed,
                "bytes": sum(
                    result.bytes_transferred for result in offloads + restores
                ),
            }
        )
        await self.clean()

    async def run_one(
        self,
        scenario: str,
        prefix_length: int,
        count: int,
        gap_ms: float,
        pattern: str,
        spread_ms: float,
        repeat: int,
    ) -> dict[str, Any]:
        agent_ids, completed = await self.prepare_agents(
            scenario, prefix_length, count, repeat
        )
        half = count // 2
        if scenario in ("reroute", "ttl-reroute", "on-return", "agentshift"):
            selected = [index < half for index in range(count)]
        elif scenario in ("agentix", "agentix-on-return"):
            selected = self.agentix_assignment(count, prefix_length)
        else:
            selected = [False] * count
        clients = [
            self.destination if move else self.source for move in selected
        ]
        ready_events = [asyncio.Event() for _ in agent_ids]
        migrations: list[MigrationResult] = []
        offloads: list[TierOperationResult] = []
        restores: list[TierOperationResult] = []
        tool_started = time.perf_counter()
        base_return = tool_started + gap_ms / 1000.0
        offsets = self.return_offsets(count, pattern, spread_ms)

        if scenario == "agentshift":
            preparation_task = asyncio.create_task(
                self._direct_prepare(
                    agent_ids,
                    completed,
                    selected,
                    ready_events,
                    start_at=tool_started,
                )
            )
        elif scenario in ("on-return", "agentix-on-return"):
            preparation_task = asyncio.create_task(
                self._direct_prepare(
                    agent_ids,
                    completed,
                    selected,
                    ready_events,
                    start_at=base_return,
                )
            )
        elif scenario == "tokencake-source":
            preparation_task = asyncio.create_task(
                self._tokencake_source_prepare(agent_ids, completed, ready_events)
            )
        else:
            preparation_task = None
            for event in ready_events:
                event.set()

        if scenario in ("ttl", "ttl-reroute") and gap_ms > self.args.ttl_ms:
            await self.sleep_until(tool_started + self.args.ttl_ms / 1000.0)
            await flush(self.source)

        async def run_next(index: int):
            tool_return = base_return + offsets[index]
            await self.sleep_until(tool_return)
            await ready_events[index].wait()
            request_started = time.perf_counter()
            elapsed, result = await timed_generate(
                clients[index],
                list(completed[index]) + [200] * self.args.tool_result_tokens,
                max_new_tokens=self.args.burst_output_tokens,
                rid=f"{agent_ids[index]}-turn-2",
            )
            return {
                "tool_return": tool_return,
                "request_started": request_started,
                "completed_at": time.perf_counter(),
                "request_seconds": elapsed,
                "cached_tokens": int(result["meta_info"]["cached_tokens"]),
            }

        request_tasks = [
            asyncio.create_task(run_next(index)) for index in range(count)
        ]
        preparation_done = tool_started
        if preparation_task is not None:
            try:
                prepared = await preparation_task
            except Exception:
                for task in request_tasks:
                    task.cancel()
                await asyncio.gather(*request_tasks, return_exceptions=True)
                raise
            if scenario == "tokencake-source":
                offloads, restores, preparation_done = prepared
            else:
                migrations, preparation_done = prepared
        results = await asyncio.gather(*request_tasks)

        first_return = min(item["tool_return"] for item in results)
        last_completion = max(item["completed_at"] for item in results)
        post_tool = [
            item["completed_at"] - item["tool_return"] for item in results
        ]
        cache_hits = [item["cached_tokens"] for item in results]
        historical_tokens = sum(len(tokens) for tokens in completed)
        historical_hits = sum(
            min(len(tokens), hit) for tokens, hit in zip(completed, cache_hits)
        )
        record = {
            "scenario": scenario,
            "prefix_length": prefix_length,
            "agent_count": count,
            "gap_ms": gap_ms,
            "return_pattern": pattern,
            "spread_ms": spread_ms,
            "repeat": repeat,
            "makespan_seconds": last_completion - first_return,
            "post_tool_mean_seconds": statistics.mean(post_tool),
            "post_tool_p50_seconds": statistics.median(post_tool),
            "post_tool_p95_seconds": percentile(post_tool, 0.95),
            "post_tool_p99_seconds": percentile(post_tool, 0.99),
            "relocated_fraction": statistics.mean(selected),
            "ownership_relocated_fraction": len(migrations) / count,
            "source_next_turns": sum(client is self.source for client in clients),
            "destination_next_turns": sum(
                client is self.destination for client in clients
            ),
            "historical_tokens": historical_tokens,
            "historical_cache_hits": historical_hits,
            "historical_reprefilled_tokens": historical_tokens - historical_hits,
            "full_prefix_hit_rate": statistics.mean(
                hit >= len(tokens) for hit, tokens in zip(cache_hits, completed)
            ),
            "preparation_seconds": preparation_done - tool_started,
            "queue_relief_after_first_return_seconds": max(
                0.0, preparation_done - first_return
            )
            if migrations
            else 0.0,
            "transfer_bytes": sum(item.bytes_transferred for item in migrations),
            "tier_bytes": sum(item.bytes_transferred for item in offloads + restores),
        }

        for migration in migrations:
            await self.coordinator.acknowledge_destination(migration.migration_id)
            await self.destination.release_prefix(
                migration.agent_id, migration.new_epoch
            )
        if migrations:
            record["source_hbm_relief_seconds"] = time.perf_counter() - tool_started
        else:
            record["source_hbm_relief_seconds"] = 0.0
        if restores:
            for agent_id in agent_ids:
                await self.source.release_prefix(agent_id, 1)
            for result in restores:
                await self.tiered.cleanup(self.source, result)
            for result in offloads:
                await self.tiered.cleanup(
                    self.source, result, drop_checkpoint=True
                )
        await self.clean()
        return record


async def main(args) -> None:
    benchmark = HotspotBenchmark(args)
    records = []
    for prefix_length in args.prefix_lengths:
        calibration = await benchmark.calibrate(prefix_length)
        print(json.dumps({"calibration": calibration}, sort_keys=True), flush=True)
        for count in args.agent_counts:
            if count * (prefix_length + args.first_output_tokens) > args.source_token_budget:
                print(
                    json.dumps(
                        {
                            "skipped": True,
                            "prefix_length": prefix_length,
                            "agent_count": count,
                            "reason": "source token budget",
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            if "tokencake-source" in args.scenarios and args.warm_tokencake:
                await benchmark.warm_tokencake(prefix_length, count)
                print(
                    json.dumps(
                        {"tokencake_warmup": benchmark.tokencake_warmups[-1]},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            for pattern in args.return_patterns:
                for repeat in range(args.repeats):
                    scenario_order = list(args.scenarios)
                    random.Random(
                        args.seed + prefix_length * 1009 + count * 17 + repeat
                    ).shuffle(scenario_order)
                    for scenario in scenario_order:
                        record = await benchmark.run_one(
                            scenario,
                            prefix_length,
                            count,
                            args.gap_ms,
                            pattern,
                            args.spread_ms,
                            repeat,
                        )
                        records.append(record)
                        print(json.dumps(record, sort_keys=True), flush=True)
    output = {
        "baseline_implementation": "mechanism-equivalent",
        "config": vars(args),
        "calibrations": benchmark.calibrations,
        "tokencake_warmups": benchmark.tokencake_warmups,
        "state_db": str(benchmark.state_path),
        "records": records,
    }
    output_path = Path(args.output_dir) / f"hotspot-{time.time_ns()}.json"
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
            "ttl-reroute",
            "tokencake-source",
            "on-return",
            "agentix-on-return",
            "agentshift",
        ],
    )
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[8192, 16384, 32768])
    parser.add_argument("--agent-counts", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument(
        "--return-patterns",
        nargs="+",
        default=["simultaneous", "uniform", "heavy-tail", "multi-wave"],
    )
    parser.add_argument("--gap-ms", type=float, default=500.0)
    parser.add_argument("--spread-ms", type=float, default=250.0)
    parser.add_argument("--ttl-ms", type=float, default=500.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--calibration-repeats", type=int, default=3)
    parser.add_argument(
        "--warm-tokencake",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--burst-output-tokens", type=int, default=32)
    parser.add_argument("--source-token-budget", type=int, default=260000)
    parser.add_argument("--poll-interval", type=float, default=0.005)
    parser.add_argument("--transfer-port", type=int, default=30200)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
