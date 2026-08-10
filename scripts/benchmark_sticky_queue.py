#!/usr/bin/env python3
"""Measure the queueing cost of keeping warm agents on a busy source engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

from benchmark_e2e import flush

from agentshift.engine.sglang import (
    SGLangAgentShiftClient,
    generate,
    stream_generate,
)


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    position = quantile * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(
            (record["source_load"], record["returning_agents"]), []
        ).append(record)

    summary: dict[str, Any] = {}
    for (source_load, returning_agents), rows in sorted(groups.items()):
        samples = [
            sample
            for row in rows
            for sample in row["request_ttft_seconds"]
        ]
        cached = [
            value
            for row in rows
            for value in row["cached_tokens"]
        ]
        key = f"{source_load}:{returning_agents}"
        summary[key] = {
            "samples": len(samples),
            "ttft_p50_ms": 1000.0 * percentile(samples, 0.50),
            "ttft_p95_ms": 1000.0 * percentile(samples, 0.95),
            "ttft_p99_ms": 1000.0 * percentile(samples, 0.99),
            "ttft_mean_ms": 1000.0 * statistics.fmean(samples),
            "burst_makespan_mean_ms": 1000.0
            * statistics.fmean(row["burst_makespan_seconds"] for row in rows),
            "cached_tokens_min": min(cached),
            "cached_tokens_mean": statistics.fmean(cached),
        }
    return summary


class StickyQueueBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.source = SGLangAgentShiftClient(
            "engine-a", args.source, timeout=args.timeout
        )
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=args.timeout
        )
        self.serial = 0

    def next_serial(self) -> int:
        self.serial += 1
        return self.serial

    @staticmethod
    def prompt(length: int, salt: int, fill_token: int = 100) -> list[int]:
        return [16000 + salt] + [fill_token] * (length - 1)

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    async def prepare_agents(
        self, returning_agents: int, repeat: int, source_load: str
    ) -> tuple[list[str], list[tuple[int, ...]]]:
        agent_ids: list[str] = []
        prompts: list[list[int]] = []
        for index in range(returning_agents):
            salt = self.next_serial()
            agent_ids.append(
                f"sticky-queue-{source_load}-{returning_agents}-{repeat}-{index}-{salt}"
            )
            prompts.append(self.prompt(self.args.prefix_tokens, salt))

        first_results = await asyncio.gather(
            *[
                generate(
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
            for prompt, result in zip(prompts, first_results)
        ]
        return agent_ids, completed

    async def start_background(
        self, returning_agents: int, repeat: int
    ) -> list[asyncio.Task]:
        first_token_events = [
            asyncio.Event() for _ in range(self.args.background_concurrency)
        ]
        tasks = []
        for index, event in enumerate(first_token_events):
            salt = self.next_serial()
            tasks.append(
                asyncio.create_task(
                    stream_generate(
                        self.source,
                        self.prompt(
                            self.args.background_prompt_tokens,
                            salt,
                            fill_token=101,
                        ),
                        max_new_tokens=self.args.background_output_tokens,
                        rid=(
                            f"sticky-background-{returning_agents}-{repeat}-"
                            f"{index}-{salt}"
                        ),
                        first_token_event=event,
                        ignore_eos=True,
                    )
                )
            )
        if first_token_events:
            await asyncio.wait_for(
                asyncio.gather(*(event.wait() for event in first_token_events)),
                timeout=self.args.timeout,
            )
        return tasks

    async def run_one(
        self, source_load: str, returning_agents: int, repeat: int
    ) -> dict[str, Any]:
        await self.clean()
        agent_ids, completed = await self.prepare_agents(
            returning_agents, repeat, source_load
        )

        background_tasks: list[asyncio.Task] = []
        if source_load == "busy":
            background_tasks = await self.start_background(returning_agents, repeat)

        burst_started = time.perf_counter()
        results = await asyncio.gather(
            *[
                stream_generate(
                    self.source,
                    list(tokens) + [200] * self.args.tool_result_tokens,
                    max_new_tokens=self.args.next_output_tokens,
                    rid=f"{agent_id}-turn-2",
                )
                for agent_id, tokens in zip(agent_ids, completed)
            ]
        )
        burst_makespan = time.perf_counter() - burst_started

        if background_tasks:
            background_results = await asyncio.gather(*background_tasks)
        else:
            background_results = []

        background_completion_tokens = [
            int(result["response"]["meta_info"]["completion_tokens"])
            for result in background_results
        ]
        if background_completion_tokens and any(
            value != self.args.background_output_tokens
            for value in background_completion_tokens
        ):
            raise RuntimeError(
                "background request ended before the configured decode length: "
                f"{background_completion_tokens}"
            )

        ttfts = [result["ttft_seconds"] for result in results]
        cache_hits = [
            int(result["response"]["meta_info"]["cached_tokens"])
            for result in results
        ]
        record = {
            "source_load": source_load,
            "returning_agents": returning_agents,
            "repeat": repeat,
            "prefix_tokens": self.args.prefix_tokens,
            "completed_prefix_tokens": [len(tokens) for tokens in completed],
            "request_ttft_seconds": ttfts,
            "request_e2e_seconds": [result["e2e_seconds"] for result in results],
            "cached_tokens": cache_hits,
            "burst_makespan_seconds": burst_makespan,
            "background_concurrency": len(background_tasks),
            "background_ttft_seconds": [
                result["ttft_seconds"] for result in background_results
            ],
            "background_e2e_seconds": [
                result["e2e_seconds"] for result in background_results
            ],
            "background_completion_tokens": background_completion_tokens,
        }
        await self.clean()
        return record


async def main(args: argparse.Namespace) -> None:
    benchmark = StickyQueueBenchmark(args)
    records: list[dict[str, Any]] = []
    for repeat in range(args.repeats):
        for returning_agents in args.returning_agents:
            for source_load in args.source_loads:
                record = await benchmark.run_one(
                    source_load, returning_agents, repeat
                )
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)

    output = {
        "config": vars(args),
        "measurement": (
            "TTFT is measured at the streaming client from request submission "
            "until receipt of the first output event"
        ),
        "records": records,
        "summary": summarize(records),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sticky-queue-qwen32b-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "summary": output["summary"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument(
        "--source-loads", nargs="+", choices=("idle", "busy"), default=["idle", "busy"]
    )
    parser.add_argument("--returning-agents", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--prefix-tokens", type=int, default=4096)
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--next-output-tokens", type=int, default=1)
    parser.add_argument("--background-concurrency", type=int, default=16)
    parser.add_argument("--background-prompt-tokens", type=int, default=128)
    parser.add_argument("--background-output-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output-dir", default="results/motivation-qwen32b")
    asyncio.run(main(parser.parse_args()))
