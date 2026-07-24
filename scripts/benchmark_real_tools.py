from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

from benchmark_e2e import flush, timed_generate

from agentshift.controller.migration import MigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


def tool_specs(python: str) -> list[tuple[str, list[str]]]:
    return [
        ("git-status", ["git", "status", "--short"]),
        (
            "state-store-tests",
            [python, "-m", "pytest", "-q", "tests/test_state_store.py"],
        ),
        ("control-plane-tests", [python, "-m", "pytest", "-q"]),
    ]


async def run_tool(command: list[str], cwd: str) -> tuple[float, bytes, bytes]:
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    elapsed = time.perf_counter() - started
    if process.returncode != 0:
        raise RuntimeError(
            f"tool failed ({process.returncode}): {' '.join(command)}\n"
            f"{stderr.decode(errors='replace')}"
        )
    return elapsed, stdout, stderr


class RealToolBenchmark:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.state_path = Path(args.output_dir) / f"real-tools-state-{time.time_ns()}.db"
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            base_port=args.transfer_port,
            async_transfer=True,
            transfer_poll_interval=0.005,
        )
        self.serial = 0

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    async def run_one(
        self,
        scenario: str,
        tool_name: str,
        command: list[str],
        prefix_length: int,
        repeat: int,
    ) -> dict[str, Any]:
        await self.clean()
        self.serial += 1
        agent_id = f"real-tool-{scenario}-{tool_name}-{prefix_length}-{self.serial}"
        prompt = [20000 + self.serial] + [100] * (prefix_length - 1)
        _, first = await timed_generate(
            self.source,
            prompt,
            max_new_tokens=4,
            rid=f"{agent_id}-turn-1",
        )
        completed = tuple(prompt + first["output_ids"])
        if scenario != "reroute":
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
            await self.coordinator._ensure_group("engine-a", "engine-b")

        tool_task = asyncio.create_task(run_tool(command, self.args.tool_cwd))
        migration_task = None
        if scenario == "proactive":
            migration_task = asyncio.create_task(
                self.coordinator.migrate(agent_id, "engine-b")
            )
        tool_seconds, stdout, stderr = await tool_task
        tool_completed = time.perf_counter()
        if scenario == "on-return":
            migration_task = asyncio.create_task(
                self.coordinator.migrate(agent_id, "engine-b")
            )
        migration = await migration_task if migration_task is not None else None
        destination_ready = time.perf_counter()
        exposed = (
            max(0.0, destination_ready - tool_completed) if migration else 0.0
        )
        result_tokens = max(
            16,
            min(
                self.args.max_tool_result_tokens,
                (len(stdout) + len(stderr) + 3) // 4,
            ),
        )
        next_seconds, second = await timed_generate(
            self.destination,
            list(completed) + [200] * result_tokens,
            max_new_tokens=1,
            rid=f"{agent_id}-turn-2",
        )
        record = {
            "scenario": scenario,
            "tool": tool_name,
            "command": command,
            "prefix_length": prefix_length,
            "repeat": repeat,
            "tool_seconds": tool_seconds,
            "tool_output_bytes": len(stdout) + len(stderr),
            "tool_result_tokens": result_tokens,
            "migration_seconds": migration.transfer_seconds if migration else 0.0,
            "exposed_migration_seconds": exposed,
            "next_turn_seconds": next_seconds,
            "post_tool_seconds": exposed + next_seconds,
            "cached_tokens": int(second["meta_info"]["cached_tokens"]),
            "fully_hidden": bool(migration and exposed < 0.001),
        }
        if migration is not None:
            await self.coordinator.acknowledge_destination(migration.migration_id)
            await self.destination.release_prefix(agent_id, migration.new_epoch)
        await self.clean()
        return record


async def main(args) -> None:
    benchmark = RealToolBenchmark(args)
    records = []
    for prefix_length in args.prefix_lengths:
        for repeat in range(args.repeats):
            for tool_name, command in tool_specs(args.python):
                for scenario in ("reroute", "on-return", "proactive"):
                    record = await benchmark.run_one(
                        scenario, tool_name, command, prefix_length, repeat
                    )
                    records.append(record)
                    print(json.dumps(record, sort_keys=True), flush=True)
    output = {
        "config": vars(args),
        "state_db": str(benchmark.state_path),
        "records": records,
    }
    output_path = Path(args.output_dir) / f"real-tools-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[4096, 16384, 32000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tool-result-tokens", type=int, default=1024)
    parser.add_argument("--transfer-port", type=int, default=30000)
    parser.add_argument("--tool-cwd", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
