from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from benchmark_e2e import flush, timed_generate

from agentshift.controller.tiered import TieredPrefixCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient


class TierBaselineValidator:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.coordinator = TieredPrefixCoordinator(
            poll_interval=args.poll_interval,
            operation_timeout=300,
        )
        self.serial = 0

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    async def prepare(self, label: str, prefix_length: int):
        await self.clean()
        self.serial += 1
        agent_id = f"tier-{label}-{prefix_length}-{self.serial}"
        prompt = [17000 + self.serial] + [100] * (prefix_length - 1)
        _, first = await timed_generate(
            self.source,
            prompt,
            max_new_tokens=self.args.first_output_tokens,
            rid=f"{agent_id}-turn-1",
        )
        completed = tuple(prompt + first["output_ids"])
        next_prompt = list(completed) + [200] * self.args.tool_result_tokens
        _, reference = await timed_generate(
            self.destination,
            next_prompt,
            max_new_tokens=self.args.output_tokens,
            rid=f"{agent_id}-reference",
        )
        await flush(self.destination)
        pin = await self.source.pin_prefix(agent_id, 1, completed)
        if int(pin["token_count"]) != len(completed):
            raise RuntimeError(
                f"source pin hit {pin['token_count']} of {len(completed)} tokens"
            )
        return agent_id, completed, next_prompt, reference

    @staticmethod
    def validate_result(label, completed, reference, result):
        cached_tokens = int(result["meta_info"]["cached_tokens"])
        if cached_tokens < len(completed):
            raise RuntimeError(
                f"{label} hit {cached_tokens} of {len(completed)} completed tokens"
            )
        if result["output_ids"] != reference["output_ids"]:
            raise RuntimeError(f"{label} output differs from deterministic reference")
        return cached_tokens

    async def private_round_trip(self, prefix_length: int) -> dict:
        agent_id, completed, next_prompt, reference = await self.prepare(
            "private", prefix_length
        )
        checkpoint_id = f"private-{prefix_length}-{time.time_ns()}"
        offload = await self.coordinator.run(
            self.source,
            operation="private_offload",
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            owner_epoch=1,
            token_ids=completed,
            release_gpu=True,
        )
        restore = await self.coordinator.run(
            self.source,
            operation="private_restore",
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            owner_epoch=1,
            token_ids=completed,
            release_gpu=False,
        )
        next_seconds, result = await timed_generate(
            self.source,
            next_prompt,
            max_new_tokens=self.args.output_tokens,
            rid=f"{agent_id}-restored",
        )
        cached_tokens = self.validate_result(
            "private restore", completed, reference, result
        )
        await self.source.release_prefix(agent_id, 1)
        await self.coordinator.cleanup(self.source, restore)
        await self.coordinator.cleanup(
            self.source, offload, drop_checkpoint=True
        )
        return {
            "path": "private_cpu_round_trip",
            "prefix_length": prefix_length,
            "completed_tokens": len(completed),
            "cached_tokens": cached_tokens,
            "outputs_equal": True,
            "offload_wall_seconds": offload.wall_seconds,
            "offload_worker_seconds": offload.worker_seconds,
            "restore_wall_seconds": restore.wall_seconds,
            "restore_worker_seconds": restore.worker_seconds,
            "next_turn_seconds": next_seconds,
            "bytes_each_direction": offload.bytes_transferred,
        }

    async def shared_round_trip(self, prefix_length: int) -> dict:
        agent_id, completed, next_prompt, reference = await self.prepare(
            "shared", prefix_length
        )
        checkpoint_id = f"shared-{prefix_length}-{time.time_ns()}"
        export = await self.coordinator.run(
            self.source,
            operation="shared_export",
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            owner_epoch=1,
            token_ids=completed,
            release_gpu=True,
        )
        checkpoint_path = Path(
            f"/dev/shm/agentshift-prefixes/{checkpoint_id}.tp0.bin"
        )
        if not checkpoint_path.exists():
            raise RuntimeError(f"shared checkpoint missing: {checkpoint_path}")
        checkpoint_bytes = checkpoint_path.stat().st_size
        imported = await self.coordinator.run(
            self.destination,
            operation="shared_import",
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            owner_epoch=1,
            token_ids=completed,
            release_gpu=False,
        )
        next_seconds, result = await timed_generate(
            self.destination,
            next_prompt,
            max_new_tokens=self.args.output_tokens,
            rid=f"{agent_id}-imported",
        )
        cached_tokens = self.validate_result(
            "shared import", completed, reference, result
        )
        await self.destination.release_prefix(agent_id, 1)
        await self.coordinator.cleanup(self.source, export)
        await self.coordinator.cleanup(
            self.destination, imported, drop_checkpoint=True
        )
        if checkpoint_path.exists():
            raise RuntimeError(f"shared checkpoint leaked: {checkpoint_path}")
        return {
            "path": "shared_memory_remote_prefetch",
            "prefix_length": prefix_length,
            "completed_tokens": len(completed),
            "cached_tokens": cached_tokens,
            "outputs_equal": True,
            "export_wall_seconds": export.wall_seconds,
            "export_worker_seconds": export.worker_seconds,
            "import_wall_seconds": imported.wall_seconds,
            "import_worker_seconds": imported.worker_seconds,
            "next_turn_seconds": next_seconds,
            "bytes_each_direction": export.bytes_transferred,
            "checkpoint_bytes": checkpoint_bytes,
        }


async def main(args) -> None:
    validator = TierBaselineValidator(args)
    records = []
    for prefix_length in args.prefix_lengths:
        private = await validator.private_round_trip(prefix_length)
        print(json.dumps(private, sort_keys=True), flush=True)
        records.append(private)
        shared = await validator.shared_round_trip(prefix_length)
        print(json.dumps(shared, sort_keys=True), flush=True)
        records.append(shared)
    await validator.clean()
    output = {
        "config": vars(args),
        "records": records,
        "shared_prefix_directory": "/dev/shm/agentshift-prefixes",
    }
    output_path = Path(args.output_dir) / f"tier-validation-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument("--prefix-lengths", type=int, nargs="+", default=[4096])
    parser.add_argument("--first-output-tokens", type=int, default=4)
    parser.add_argument("--tool-result-tokens", type=int, default=32)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--poll-interval", type=float, default=0.002)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
