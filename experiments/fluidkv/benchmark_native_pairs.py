#!/usr/bin/env python3
"""Paired hybrid mobility microbenchmark on observed Qwen parent prefixes."""

import argparse
import asyncio
import json
import random
import time
import uuid
from pathlib import Path

import aiohttp
import numpy as np

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


async def stream(session, url, tokens, output_tokens, on_first_token=None):
    started = time.perf_counter()
    first = None
    final = None
    async with session.post(url + "/generate", json={
        "input_ids": tokens, "stream": True,
        "sampling_params": {"max_new_tokens": output_tokens, "temperature": 0, "ignore_eos": True},
    }) as response:
        response.raise_for_status()
        async for raw in response.content:
            if not raw.startswith(b"data:"):
                continue
            payload = raw[5:].strip()
            if payload == b"[DONE]":
                continue
            final = json.loads(payload)
            if first is None and (final.get("output_ids") or final.get("meta_info", {}).get("completion_tokens", 0) > 0):
                first = time.perf_counter()
                if on_first_token is not None:
                    await on_first_token(final, first - started)
    if first is None or final is None:
        raise RuntimeError("no output token in generation stream")
    return {
        "ttft_seconds": first - started,
        "e2e_seconds": time.perf_counter() - started,
        "meta_info": final["meta_info"],
        "output_ids": final["output_ids"],
    }


async def flush(session, urls):
    for url in urls:
        async with session.post(url + "/flush_cache") as response:
            response.raise_for_status()
            await response.read()


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    pairs = manifest["pairs"][:args.pairs]
    tokens = np.load(args.tokens)
    urls = [args.source.rstrip("/"), args.destination.rstrip("/")]
    store = SQLiteStateStore(args.output / "state.db")
    clients = {name: SGLangAgentShiftClient(name, url, timeout=180) for name, url in zip(("source", "destination"), urls)}
    coordinator = HybridMigrationCoordinator(store, clients, base_port=args.transfer_port)
    modes = ["sticky", "cold-reroute", "on-return", "prepared-full-copy"]
    rows = []
    rng = random.Random(args.seed)
    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        info = []
        for url in urls:
            async with session.get(url + "/get_server_info") as response:
                response.raise_for_status()
                info.append(await response.json())
        (args.output / "server_info.json").write_text(json.dumps(info, indent=2) + "\n")
        for url in urls:
            await stream(session, url, [1300 + i for i in range(128)], args.output_tokens)
        start_group = time.perf_counter()
        await coordinator._ensure_group("source", "destination")
        group_seconds = time.perf_counter() - start_group
        with (args.output / "records.jsonl").open("w", buffering=1) as log:
            for repetition in range(args.repetitions):
                for ordinal, pair in enumerate(pairs):
                    parent = tokens[str(pair["parent_id"])].tolist()
                    child = tokens[str(pair["child_id"])].tolist()
                    if child[:len(parent)] != parent:
                        raise ValueError("selected pair does not retain exact parent prefix")
                    mode_order = rng.sample(modes, len(modes))
                    results = []
                    for mode in mode_order:
                        await flush(session, urls)
                        parent_result = await stream(session, urls[0], parent, 1)
                        migration_seconds = 0.0
                        moved_bytes = 0
                        owner_committed = None
                        agent_id = uuid.uuid4().hex
                        store.register_agent(AgentContinuation(
                            agent_id=agent_id, committed_step=0, owner_engine="source",
                            owner_epoch=1, token_ids=tuple(parent),
                        ))
                        if mode in ("on-return", "prepared-full-copy"):
                            started = time.perf_counter()
                            migration = await coordinator.migrate_async(agent_id, "destination")
                            migration_seconds = time.perf_counter() - started
                            moved_bytes = migration.bytes_transferred
                            owner_committed = store.get_agent(agent_id).owner_engine == "destination"
                        child_result = await stream(session, urls[0] if mode == "sticky" else urls[1], child, args.output_tokens)
                        exposed = migration_seconds if mode == "on-return" else 0.0
                        result = {
                            "repetition": repetition, "ordinal": ordinal, "policy": mode,
                            "pair": pair, "parent": parent_result, "child": child_result,
                            "migration_seconds": migration_seconds, "transfer_bytes": moved_bytes,
                            "post_ready_ttft_seconds": exposed + child_result["ttft_seconds"],
                            "post_ready_e2e_seconds": exposed + child_result["e2e_seconds"],
                            "owner_committed": owner_committed,
                            "expected_min_cached_tokens": len(parent) if mode != "cold-reroute" else 0,
                        }
                        results.append(result)
                    reference = next(r["child"]["output_ids"] for r in results if r["policy"] == "cold-reroute")
                    sticky_reference = next(r["child"]["output_ids"] for r in results if r["policy"] == "sticky")
                    for result in results:
                        meta = result["child"]["meta_info"]
                        result["checks"] = {
                            "output_matches_cold": result["child"]["output_ids"] == reference,
                            "output_matches_sticky": result["child"]["output_ids"] == sticky_reference,
                            "input_length_preserved": meta["prompt_tokens"] == len(child),
                            "output_length_preserved": meta["completion_tokens"] == args.output_tokens,
                            "source_initially_cold": result["parent"]["meta_info"]["cached_tokens"] == 0,
                            "parent_prefix_hit": meta["cached_tokens"] >= result["expected_min_cached_tokens"] if result["policy"] != "cold-reroute" else meta["cached_tokens"] == 0,
                            "owner_commit": result["owner_committed"] is not False,
                        }
                        log.write(json.dumps(result) + "\n")
                        rows.append(result)
                    print(json.dumps({"repetition": repetition, "pair": ordinal, "checks_passed": all(all(r["checks"].values()) for r in results)}), flush=True)
    summaries = {}
    for mode in modes:
        selected = [r for r in rows if r["policy"] == mode]
        summaries[mode] = {
            "samples": len(selected),
            "all_checks_passed": sum(all(r["checks"].values()) for r in selected),
            "output_matches_cold": sum(r["checks"]["output_matches_cold"] for r in selected),
            "output_matches_sticky": sum(r["checks"]["output_matches_sticky"] for r in selected),
            "post_ready_ttft_p50_p90": np.percentile([r["post_ready_ttft_seconds"] for r in selected], [50, 90]).tolist(),
            "post_ready_e2e_p50_p90": np.percentile([r["post_ready_e2e_seconds"] for r in selected], [50, 90]).tolist(),
            "migration_p50_p90": np.percentile([r["migration_seconds"] for r in selected], [50, 90]).tolist(),
            "total_transfer_bytes": sum(r["transfer_bytes"] for r in selected),
        }
    report = {
        "scope": manifest["scope"],
        "generation_scope": "1 parent output and fixed child output; no real tool-time or continuous-load claim.",
        "prepared_semantics": "State is fully prepared before child eligibility; this is a ready-destination microbenchmark, not a measured trace tool gap.",
        "source_reclamation_semantics": "Async cleanup unpins source; it does not evict the reusable source radix prefix or release a physical GPU.",
        "manifest": str(args.manifest), "source_sha256": manifest["source_sha256"],
        "group_initialization_seconds": group_seconds, "repetitions": args.repetitions,
        "child_output_tokens": args.output_tokens, "policies": summaries,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not all(all(row["checks"].values()) for row in rows):
        raise RuntimeError("pilot contains failed correctness or cache checks; inspect records")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="http://127.0.0.1:31280")
    parser.add_argument("--destination", default="http://127.0.0.1:31281")
    parser.add_argument("--pairs", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--transfer-port", type=int, default=29870)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
