from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentshift.controller.migration import MigrationCoordinator, MigrationResult
from agentshift.engine.sglang import SGLangAgentShiftClient, generate
from agentshift.runtime.executor import ManagedAgentExecutor
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore, StaleLease
from benchmark_e2e import flush


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * quantile) - 1))
    return ordered[index]


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def output_digest(result: dict[str, Any]) -> str:
    payload = json.dumps(result.get("output_ids", []), separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


@dataclass
class CountingClient:
    delegate: SGLangAgentShiftClient
    generate_calls: int = 0
    generated_tokens: int = 0
    call_started_at: list[float] = field(default_factory=list)
    call_finished_at: list[float] = field(default_factory=list)

    @property
    def engine_id(self) -> str:
        return self.delegate.engine_id

    async def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        require_success: bool = True,
    ) -> dict[str, Any]:
        if path != "/generate":
            return await self.delegate.post(
                path, payload, require_success=require_success
            )
        self.generate_calls += 1
        self.call_started_at.append(time.perf_counter())
        result = await self.delegate.post(path, payload, require_success=require_success)
        self.call_finished_at.append(time.perf_counter())
        self.generated_tokens += len(result.get("output_ids", []))
        return result


class FencingBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        timestamp = time.time_ns()
        self.state_path = Path(args.output_dir) / f"fencing-state-{timestamp}.db"
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            base_port=args.transfer_port,
            tp_size=args.tp_size,
            async_transfer=True,
        )
        self.serial = 0

    def prompt(self) -> list[int]:
        self.serial += 1
        return [24000 + self.serial] + [100] * (self.args.prefix_length - 1)

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    async def warm_up(self) -> None:
        prompt = [100] * 128
        await asyncio.gather(
            generate(
                self.source,
                prompt,
                max_new_tokens=4,
                rid="fencing-warmup-source",
            ),
            generate(
                self.destination,
                prompt,
                max_new_tokens=4,
                rid="fencing-warmup-destination",
            ),
        )
        await self.clean()

    async def prepare(self, scenario: str, repeat: int) -> tuple[
        str, tuple[int, ...], MigrationResult
    ]:
        await self.clean()
        agent_id = f"fencing-{scenario}-{repeat}-{self.serial + 1}"
        prompt = self.prompt()
        first = await generate(
            self.source,
            prompt,
            max_new_tokens=self.args.first_output_tokens,
            rid=f"{agent_id}-step-1",
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
        migration = await self.coordinator.migrate(agent_id, "engine-b")
        return agent_id, completed, migration

    async def cleanup(self, migration: MigrationResult) -> None:
        try:
            await self.coordinator.acknowledge_destination(migration.migration_id)
        finally:
            await self.destination.release_prefix(
                migration.agent_id,
                migration.new_epoch,
                allow_missing=True,
            )
            await self.clean()

    def second_prompt(self, completed: tuple[int, ...]) -> list[int]:
        return list(completed) + [200] * self.args.tool_result_tokens

    async def direct_turn(
        self,
        client: CountingClient,
        prompt: list[int],
        rid: str,
    ) -> tuple[float, dict[str, Any]]:
        started = time.perf_counter()
        result = await generate(
            client,
            prompt,
            max_new_tokens=self.args.output_tokens,
            rid=rid,
        )
        return time.perf_counter() - started, result

    async def run_router_only(
        self, *, duplicate: bool, repeat: int
    ) -> dict[str, Any]:
        scenario = "router-duplicate" if duplicate else "router-normal"
        agent_id, completed, migration = await self.prepare(scenario, repeat)
        source = CountingClient(self.source)
        destination = CountingClient(self.destination)
        prompt = self.second_prompt(completed)
        rid = f"{agent_id}-step-2"
        started = time.perf_counter()
        if duplicate:
            source_result, destination_result = await asyncio.gather(
                self.direct_turn(source, prompt, rid),
                self.direct_turn(destination, prompt, rid),
            )
            executions = [
                ("engine-a", source_result[0], source_result[1]),
                ("engine-b", destination_result[0], destination_result[1]),
            ]
        else:
            destination_result = await self.direct_turn(destination, prompt, rid)
            executions = [("engine-b", destination_result[0], destination_result[1])]
        wall_seconds = time.perf_counter() - started
        generated_tokens = sum(len(result["output_ids"]) for _, _, result in executions)
        committed_tokens = len(executions[-1][2]["output_ids"])
        record = {
            "scenario": scenario,
            "repeat": repeat,
            "duplicate_delivery": duplicate,
            "fencing": False,
            "logical_steps": 1,
            "accepted_executors": len(executions),
            "generate_rpc_count": source.generate_calls + destination.generate_calls,
            "source_generate_rpc_count": source.generate_calls,
            "destination_generate_rpc_count": destination.generate_calls,
            "generated_tokens": generated_tokens,
            "committed_tokens": committed_tokens,
            "wasted_decode_tokens": generated_tokens - committed_tokens,
            "wasted_decode_fraction": (
                (generated_tokens - committed_tokens) / generated_tokens
                if generated_tokens
                else 0.0
            ),
            "synthetic_effect_submissions": len(executions),
            "duplicate_effect_submissions": max(0, len(executions) - 1),
            "wasted_engine_seconds": (
                source_result[0] if duplicate else 0.0
            ),
            "wasted_gpu_seconds": (
                source_result[0] * self.args.tp_size if duplicate else 0.0
            ),
            "wall_seconds": wall_seconds,
            "migration_seconds": migration.transfer_seconds,
            "migration_bytes": migration.bytes_transferred,
            "cached_tokens": {
                engine: int(result["meta_info"]["cached_tokens"])
                for engine, _, result in executions
            },
            "full_prefix_hit": {
                engine: int(result["meta_info"]["cached_tokens"])
                >= migration.token_count
                for engine, _, result in executions
            },
            "output_tokens": {
                engine: len(result["output_ids"])
                for engine, _, result in executions
            },
            "output_digests": {
                engine: output_digest(result) for engine, _, result in executions
            },
            "execution_seconds": {
                engine: elapsed for engine, elapsed, _ in executions
            },
        }
        await self.cleanup(migration)
        return record

    async def run_fenced(self, *, duplicate: bool, repeat: int) -> dict[str, Any]:
        scenario = "fenced-duplicate" if duplicate else "fenced-normal"
        agent_id, completed, migration = await self.prepare(scenario, repeat)
        source = CountingClient(self.source)
        destination = CountingClient(self.destination)
        executor = ManagedAgentExecutor(
            self.store,
            {"engine-a": source, "engine-b": destination},
        )
        prompt = self.second_prompt(completed)
        rid = f"{agent_id}-step-2"

        async def execute(
            engine: str, epoch: int
        ) -> tuple[str, float, Any, float | None, float | None]:
            started = time.perf_counter()
            client = source if engine == "engine-a" else destination
            try:
                result = await executor.run_turn(
                    agent_id=agent_id,
                    owner_engine=engine,
                    owner_epoch=epoch,
                    step_id=2,
                    input_ids=prompt,
                    max_new_tokens=self.args.output_tokens,
                    rid=rid,
                )
            except Exception as exc:
                return engine, time.perf_counter() - started, exc, None, None
            finished = time.perf_counter()
            claim_seconds = client.call_started_at[-1] - started
            commit_seconds = finished - client.call_finished_at[-1]
            return (
                engine,
                finished - started,
                result,
                claim_seconds,
                commit_seconds,
            )

        started = time.perf_counter()
        if duplicate:
            outcomes = await asyncio.gather(
                execute("engine-a", migration.old_epoch),
                execute("engine-b", migration.new_epoch),
            )
        else:
            outcomes = [await execute("engine-b", migration.new_epoch)]
        wall_seconds = time.perf_counter() - started
        accepted = [item for item in outcomes if isinstance(item[2], dict)]
        rejected = [item for item in outcomes if isinstance(item[2], Exception)]
        generated_tokens = sum(len(item[2]["output_ids"]) for item in accepted)
        stale_rejections = [item for item in rejected if isinstance(item[2], StaleLease)]
        record = {
            "scenario": scenario,
            "repeat": repeat,
            "duplicate_delivery": duplicate,
            "fencing": True,
            "logical_steps": 1,
            "accepted_executors": len(accepted),
            "generate_rpc_count": source.generate_calls + destination.generate_calls,
            "source_generate_rpc_count": source.generate_calls,
            "destination_generate_rpc_count": destination.generate_calls,
            "generated_tokens": generated_tokens,
            "committed_tokens": generated_tokens,
            "wasted_decode_tokens": 0,
            "wasted_decode_fraction": 0.0,
            "synthetic_effect_submissions": len(accepted),
            "duplicate_effect_submissions": max(0, len(accepted) - 1),
            "wasted_engine_seconds": 0.0,
            "wasted_gpu_seconds": 0.0,
            "stale_rejections": len(stale_rejections),
            "other_rejections": len(rejected) - len(stale_rejections),
            "stale_rejection_seconds": [item[1] for item in stale_rejections],
            "managed_claim_seconds": [
                item[3] for item in accepted if item[3] is not None
            ],
            "managed_commit_seconds": [
                item[4] for item in accepted if item[4] is not None
            ],
            "managed_control_seconds": [
                item[3] + item[4]
                for item in accepted
                if item[3] is not None and item[4] is not None
            ],
            "wall_seconds": wall_seconds,
            "migration_seconds": migration.transfer_seconds,
            "migration_bytes": migration.bytes_transferred,
            "cached_tokens": {
                engine: int(result["meta_info"]["cached_tokens"])
                for engine, _, result, _, _ in accepted
            },
            "full_prefix_hit": {
                engine: int(result["meta_info"]["cached_tokens"])
                >= migration.token_count
                for engine, _, result, _, _ in accepted
            },
            "output_tokens": {
                engine: len(result["output_ids"])
                for engine, _, result, _, _ in accepted
            },
            "output_digests": {
                engine: output_digest(result)
                for engine, _, result, _, _ in accepted
            },
            "execution_seconds": {
                engine: elapsed for engine, elapsed, result, _, _ in outcomes
                if isinstance(result, dict)
            },
            "rejections": {
                engine: type(result).__name__
                for engine, _, result, _, _ in rejected
            },
        }
        await self.cleanup(migration)
        return record


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["scenario"], []).append(record)
    summary = {}
    for scenario, rows in grouped.items():
        rejection_seconds = [
            value for row in rows for value in row.get("stale_rejection_seconds", [])
        ]
        claim_seconds = [
            value for row in rows for value in row.get("managed_claim_seconds", [])
        ]
        commit_seconds = [
            value for row in rows for value in row.get("managed_commit_seconds", [])
        ]
        control_seconds = [
            value for row in rows for value in row.get("managed_control_seconds", [])
        ]
        summary[scenario] = {
            "samples": len(rows),
            "accepted_executors_mean": statistics.fmean(
                row["accepted_executors"] for row in rows
            ),
            "generate_rpc_count_mean": statistics.fmean(
                row["generate_rpc_count"] for row in rows
            ),
            "generated_tokens_mean": statistics.fmean(
                row["generated_tokens"] for row in rows
            ),
            "wasted_decode_tokens_mean": statistics.fmean(
                row["wasted_decode_tokens"] for row in rows
            ),
            "wasted_decode_fraction_mean": statistics.fmean(
                row["wasted_decode_fraction"] for row in rows
            ),
            "duplicate_effect_submissions": sum(
                row["duplicate_effect_submissions"] for row in rows
            ),
            "wasted_engine_seconds_mean": statistics.fmean(
                row["wasted_engine_seconds"] for row in rows
            ),
            "wasted_gpu_seconds_mean": statistics.fmean(
                row["wasted_gpu_seconds"] for row in rows
            ),
            "full_prefix_hit_rate": statistics.fmean(
                hit
                for row in rows
                for hit in row["full_prefix_hit"].values()
            ),
            "wall_seconds": summarize([row["wall_seconds"] for row in rows]),
            "stale_rejection_seconds": (
                summarize(rejection_seconds) if rejection_seconds else None
            ),
            "managed_claim_seconds": (
                summarize(claim_seconds) if claim_seconds else None
            ),
            "managed_commit_seconds": (
                summarize(commit_seconds) if commit_seconds else None
            ),
            "managed_control_seconds": (
                summarize(control_seconds) if control_seconds else None
            ),
            "migration_bytes_mean": statistics.fmean(
                row["migration_bytes"] for row in rows
            ),
        }
    if "router-normal" in summary and "fenced-normal" in summary:
        router = summary["router-normal"]["wall_seconds"]["mean"]
        fenced = summary["fenced-normal"]["wall_seconds"]["mean"]
        summary["failure_free_e2e_delta_seconds"] = fenced - router
    return summary


def validate(records: list[dict[str, Any]]) -> dict[str, Any]:
    router_normal = [row for row in records if row["scenario"] == "router-normal"]
    fenced_normal = [row for row in records if row["scenario"] == "fenced-normal"]
    router_duplicate = [
        row for row in records if row["scenario"] == "router-duplicate"
    ]
    fenced_duplicate = [
        row for row in records if row["scenario"] == "fenced-duplicate"
    ]
    checks = {
        "normal_paths_execute_once": all(
            row["accepted_executors"] == 1 and row["generate_rpc_count"] == 1
            for row in router_normal + fenced_normal
        ),
        "router_only_duplicate_reaches_both_engines": all(
            row["accepted_executors"] == 2
            and row["source_generate_rpc_count"] == 1
            and row["destination_generate_rpc_count"] == 1
            for row in router_duplicate
        ),
        "fencing_rejects_source_before_generate": all(
            row["accepted_executors"] == 1
            and row["stale_rejections"] == 1
            and row["source_generate_rpc_count"] == 0
            and row["destination_generate_rpc_count"] == 1
            for row in fenced_duplicate
        ),
        "all_executed_turns_have_full_prefix_hits": all(
            all(row["full_prefix_hit"].values()) for row in records
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


async def main(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    benchmark = FencingBenchmark(args)
    await benchmark.coordinator.initialize_transfer_pair("engine-a", "engine-b")
    await benchmark.warm_up()
    records = []
    scenarios = [
        (benchmark.run_router_only, False),
        (benchmark.run_fenced, False),
        (benchmark.run_router_only, True),
        (benchmark.run_fenced, True),
    ]
    rng = random.Random(args.seed)
    for repeat in range(args.repeats):
        repeat_scenarios = list(scenarios)
        rng.shuffle(repeat_scenarios)
        for runner, duplicate in repeat_scenarios:
            record = await runner(duplicate=duplicate, repeat=repeat)
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    output = {
        "config": vars(args),
        "state_db": str(benchmark.state_path),
        "records": records,
        "summary": aggregate(records),
        "validation": validate(records),
    }
    output_path = Path(args.output_dir) / f"fencing-microbench-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(
        json.dumps(
            {
                "output": str(output_path),
                "summary": output["summary"],
                "validation": output["validation"],
            },
            indent=2,
        )
    )
    if not output["validation"]["passed"]:
        raise RuntimeError("fencing microbenchmark validation failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31200")
    parser.add_argument("--destination", default="http://127.0.0.1:31201")
    parser.add_argument("--prefix-length", type=int, default=16384)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--transfer-port", type=int, default=30500)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--output-dir", default="results/fencing-microbench")
    asyncio.run(main(parser.parse_args()))
