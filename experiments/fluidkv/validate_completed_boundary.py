#!/usr/bin/env python3
"""Validate snapshot availability after multi-token parent generation."""

import argparse
import asyncio
import json
import uuid
from pathlib import Path

import aiohttp
import numpy as np

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import flush, stream


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    pair = manifest["pairs"][args.pair]
    arrays = np.load(args.tokens)
    parent = arrays[str(pair["parent_id"])].tolist()
    child = arrays[str(pair["child_id"])].tolist()
    clients = {name: SGLangAgentShiftClient(name, url, timeout=180) for name, url in zip(("source", "destination"), args.engines)}
    store = SQLiteStateStore(args.output / "state.db")
    coordinator = HybridMigrationCoordinator(store, clients, base_port=args.transfer_port)
    records = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        info = []
        for url in args.engines:
            async with session.get(url + "/get_server_info") as response:
                response.raise_for_status()
                info.append(await response.json())
        (args.output / "server_info.json").write_text(json.dumps(info, indent=2) + "\n")
        for outputs in args.parent_outputs:
            await flush(session, args.engines)
            first = await stream(session, args.engines[0], parent, outputs)
            recorded_child = await stream(session, args.engines[0], child, 8)
            await flush(session, args.engines)
            first = await stream(session, args.engines[0], parent, outputs)
            # The final sampled token has not passed through the model yet.
            committed_ids = parent + first["output_ids"][:-1]
            resume_ids = parent + first["output_ids"] + child[len(parent):]
            agent = uuid.uuid4().hex
            store.register_agent(AgentContinuation(agent_id=agent, committed_step=0,
                owner_engine="source", owner_epoch=1, token_ids=tuple(committed_ids)))
            migration = await coordinator.migrate_async(agent, "destination")
            source = await stream(session, args.engines[0], resume_ids, 8)
            destination = await stream(session, args.engines[1], resume_ids, 8)
            record = {
                "parent_output_tokens": outputs, "prompt_tokens": len(parent),
                "recorded_child_cached_tokens_after_parent": recorded_child["meta_info"]["cached_tokens"],
                "committed_boundary_tokens": len(committed_ids),
                "source": source, "destination": destination,
                "transfer_bytes": migration.bytes_transferred,
                "checks": {
                    "source_hit_committed_boundary": source["meta_info"]["cached_tokens"] >= len(committed_ids),
                    "destination_hit_committed_boundary": destination["meta_info"]["cached_tokens"] >= len(committed_ids),
                    "migration_preserves_output": source["output_ids"] == destination["output_ids"],
                    "owner_committed": store.get_agent(agent).owner_engine == "destination",
                },
            }
            records.append(record)
            (args.output / "summary.json").write_text(json.dumps({
                "scope": "Recorded-child cache eligibility and constructed generated-output continuation are separate controls. Constructed continuation is not the native trace input.",
                "pair": pair, "records": records,
            }, indent=2) + "\n")
            print(json.dumps({key: record[key] for key in ("parent_output_tokens", "recorded_child_cached_tokens_after_parent", "committed_boundary_tokens", "checks")}), flush=True)
            if not all(record["checks"].values()):
                raise RuntimeError("completed-boundary migration check failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--parent-outputs", type=int, nargs="+", default=[1, 8, 64])
    parser.add_argument("--engines", nargs=2, default=["http://127.0.0.1:31480", "http://127.0.0.1:31481"])
    parser.add_argument("--transfer-port", type=int, default=29910)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
