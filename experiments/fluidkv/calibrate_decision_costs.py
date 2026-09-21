#!/usr/bin/env python3
"""Warmed isolated cost probes, explicitly separate from trace performance.

Inputs are deterministic synthetic token sequences. A parent generates one token
so its *prompt* is a legal hybrid snapshot; that generated token is not inserted
into the child. Migration preserves the cached execution path, whose numerical
agreement with cold recomputation is reported as an independent gate.
"""

import argparse
import asyncio
import hashlib
import json
import random
import socket
import sqlite3
import statistics
import subprocess
import time
import uuid
from pathlib import Path

import aiohttp

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import flush, stream


MODES = ("sticky", "cold-reroute", "prepared-full-copy", "reconstruction")


def calibration_store(args):
    database = args.output / "ownership.db"
    if args.metadata_directory is not None:
        args.metadata_directory.mkdir(parents=True, exist_ok=True)
        database = args.metadata_directory / f"{args.output.name}-{uuid.uuid4().hex}.db"
    store = SQLiteStateStore(database)
    filesystem = subprocess.run(["df", "-T", str(database)], text=True, capture_output=True)
    (args.output / "metadata_storage.json").write_text(json.dumps({
        "live_database": str(database.resolve()),
        "backup_database": str((args.output / "ownership.db").resolve()),
        "sqlite_synchronous": store.conn.execute("PRAGMA synchronous").fetchone()[0],
        "sqlite_journal_mode": store.conn.execute("PRAGMA journal_mode").fetchone()[0],
        "filesystem": filesystem.stdout,
        "scope": "All durable migration transitions are included in coordinator wall time. Backup to result directory occurs after measured inference/control intervals.",
    }, indent=2) + "\n")
    return store


def finish_store(store, args):
    started = time.perf_counter()
    if Path(store.path).resolve() != (args.output / "ownership.db").resolve():
        with sqlite3.connect(args.output / "ownership.db") as destination:
            store.conn.backup(destination)
    store.conn.close()
    path = args.output / "metadata_storage.json"
    metadata = json.loads(path.read_text())
    metadata["backup_and_close_seconds"] = time.perf_counter() - started
    metadata["backup_complete"] = True
    path.write_text(json.dumps(metadata, indent=2) + "\n")


def digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def pair_checks(results, prefix_tokens, suffix_tokens, outputs):
    if set(results) != set(MODES):
        return {"mechanism_valid": False, "missing_modes": sorted(set(MODES) - set(results))}
    sticky = results["sticky"]["child"]["output_ids"]
    cold = results["cold-reroute"]["child"]["output_ids"]
    migration_matches = results["prepared-full-copy"]["child"]["output_ids"] == sticky
    per_mode = {}
    for mode, row in results.items():
        meta = row["child"]["meta_info"]
        per_mode[mode] = {
            "source_initially_cold": row["source_prepare"]["meta_info"]["cached_tokens"] == 0,
            "prompt_length": meta["prompt_tokens"] == prefix_tokens + suffix_tokens,
            "output_length": meta["completion_tokens"] == outputs and len(row["child"]["output_ids"]) == outputs,
            "cache_hit": meta["cached_tokens"] == (0 if mode == "cold-reroute" else prefix_tokens),
            "owner_committed": row.get("owner_committed") is not False,
            "matches_sticky": row["child"]["output_ids"] == sticky,
            "matches_cold": row["child"]["output_ids"] == cold,
        }
    engineering = all(all(value for key, value in checks.items()
                          if key not in ("matches_sticky", "matches_cold"))
                      for checks in per_mode.values())
    return {"mechanism_valid": engineering and migration_matches,
            "migration_preserves_sticky": migration_matches,
            "cold_cached_equivalent": cold == sticky,
            "per_mode": per_mode}


class TimedClient(SGLangAgentShiftClient):
    """Record actual coordinator RPC boundaries without changing its protocol."""

    def __init__(self, engine_id, base_url, recorder):
        super().__init__(engine_id, base_url, timeout=180)
        object.__setattr__(self, "recorder", recorder)

    async def control_post(self, path, payload, *, require_success=True):
        started = time.perf_counter()
        event = {"engine": self.engine_id, "path": path,
                 "start_monotonic": started,
                 "prepare_only": bool(payload and payload.get("prepare_only")),
                 "role": payload.get("role") if payload else None}
        try:
            result = await super().control_post(path, payload, require_success=require_success)
            event["response"] = result
            return result
        except Exception as exc:
            event["error"] = repr(exc)
            raise
        finally:
            event["end_monotonic"] = time.perf_counter()
            event["wall_seconds"] = event["end_monotonic"] - started
            self.recorder.append(event)


def migration_stages(events, start, finish):
    prepared = [event for event in events if event["prepare_only"]]
    launched = [event for event in events if event["path"].endswith("/start") and not event["prepare_only"]]
    complete = [event for event in events if "/status/" in event["path"]
                and event.get("response", {}).get("state") == "COMPLETE"]
    cleanup = [event for event in events if "/cleanup/" in event["path"]]
    return {
        "full_coordinator_wall_seconds": finish - start,
        "prepare_rpc_seconds_by_engine": {event["engine"]: event["wall_seconds"] for event in prepared},
        "both_prepared_since_start_seconds": max((event["end_monotonic"] for event in prepared), default=start) - start,
        "launch_to_observed_complete_seconds": max((event["end_monotonic"] for event in complete), default=finish)
            - min((event["start_monotonic"] for event in launched), default=start),
        "cleanup_span_seconds": max((event["end_monotonic"] for event in cleanup), default=finish)
            - min((event["start_monotonic"] for event in cleanup), default=finish),
        "timing_scope": "Client-observed protocol phases include RPC/scheduler wait; launch-to-complete is not isolated NCCL device time.",
    }


def summarize(rows, group_seconds, args):
    measured = [row for row in rows if row["phase"] == "measured" and row.get("success")]
    summary = {
        "schema_version": 1,
        "scope": "Isolated synthetic length calibration, no native lineage/load/tool-gap or GPU-release claim.",
        "protocol": "Each length has an unmeasured warmup pair then three randomized paired repetitions by default. All variants flush caches and materialize the same source prompt. Prepared latency excludes preparation only where explicitly named.",
        "numerical_scope": "Cold/cached disagreement is an independent failed equivalence gate, never waived by source/migration agreement.",
        "group_initialization_seconds": group_seconds,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "completed_pairs": len(measured),
        "failed_pairs": sum(not row.get("success", False) for row in rows),
        "migration_matches_sticky_pairs": sum(row["checks"]["migration_preserves_sticky"] for row in measured),
        "cold_cached_equivalent_pairs": sum(row["checks"]["cold_cached_equivalent"] for row in measured),
        "lengths": {},
    }
    for length in args.lengths:
        selected = [row for row in measured if row["prefix_tokens"] == length]
        if not selected:
            continue
        entry = {"paired_repetitions": len(selected), "policies": {}}
        for mode in MODES:
            values = [row["results"][mode] for row in selected]
            entry["policies"][mode] = {
                "child_ttft_median_seconds": statistics.median(value["child"]["ttft_seconds"] for value in values),
                "child_e2e_median_seconds": statistics.median(value["child"]["e2e_seconds"] for value in values),
                "preparation_wall_median_seconds": statistics.median(value["preparation_wall_seconds"] for value in values),
                "source_prepare_wall_median_seconds": statistics.median(value["source_prepare"]["e2e_seconds"] for value in values),
                "prepare_plus_child_ttft_median_seconds": statistics.median(value["preparation_wall_seconds"] + value["child"]["ttft_seconds"] for value in values),
                "cached_tokens": [value["child"]["meta_info"]["cached_tokens"] for value in values],
                "transfer_bytes": [value.get("transfer_bytes", 0) for value in values],
                "raw_child_ttft_seconds": [value["child"]["ttft_seconds"] for value in values],
                "raw_preparation_wall_seconds": [value["preparation_wall_seconds"] for value in values],
            }
        entry["paired_cold_minus_warm_ttft_seconds"] = [row["results"]["cold-reroute"]["child"]["ttft_seconds"] - row["results"]["sticky"]["child"]["ttft_seconds"] for row in selected]
        entry["paired_copy_on_return_minus_cold_seconds"] = [row["results"]["prepared-full-copy"]["preparation_wall_seconds"] + row["results"]["prepared-full-copy"]["child"]["ttft_seconds"] - row["results"]["cold-reroute"]["child"]["ttft_seconds"] for row in selected]
        summary["lengths"][str(length)] = entry
    return summary


async def contention_control(args, session, urls, coordinator, store, events):
    """Copy during an unrelated request's decode; arrival/TTFT precede copy."""
    rng = random.Random(args.seed + 117)
    records = []
    with (args.output / "contention_records.jsonl").open("w", buffering=1) as log:
        for repetition in range(-1, args.repetitions):
            parent_rng = random.Random(args.seed + repetition + 340)
            parent = [parent_rng.randrange(1100, 19000) for _ in range(8192)]
            unrelated = [parent_rng.randrange(27000, 32000) for _ in range(1024)]
            child = parent + [parent_rng.randrange(20000, 25000) for _ in range(args.suffix_tokens)]
            for mode in rng.sample(["no-copy", "copy-during-decode"], 2):
                row = {"phase": "warmup" if repetition < 0 else "measured",
                       "repetition": repetition, "mode": mode, "success": False,
                       "prefix_tokens": 8192, "foreground_input_tokens": 1024,
                       "foreground_output_tokens": 128, "parent_sha256": digest(parent),
                       "foreground_sha256": digest(unrelated)}
                foreground = None
                try:
                    await flush(session, urls)
                    row["source_prepare"] = await stream(session, urls[0], parent, 1)
                    first_seen = asyncio.Event()

                    async def notify_first(_payload, _latency):
                        first_seen.set()

                    foreground = asyncio.create_task(stream(session, urls[1], unrelated, 128, notify_first))
                    await asyncio.wait_for(first_seen.wait(), 180)
                    if mode == "copy-during-decode":
                        agent = uuid.uuid4().hex
                        store.register_agent(AgentContinuation(agent, 0, "source", 1, tuple(parent)))
                        events.clear()
                        started = time.perf_counter()
                        try:
                            migration = await coordinator.migrate_async(agent, "destination")
                        finally:
                            finish = time.perf_counter()
                            row["migration_events"] = list(events)
                            row["migration_stages"] = migration_stages(events, started, finish)
                        row["transfer_bytes"] = migration.bytes_transferred
                        row["copy_finished_before_foreground"] = not foreground.done()
                        coordinator.acknowledge_async_migration(migration.migration_id)
                    row["foreground"] = await foreground
                    row["decode_seconds"] = row["foreground"]["e2e_seconds"] - row["foreground"]["ttft_seconds"]
                    row["checks"] = {
                        "foreground_output_length": row["foreground"]["meta_info"]["completion_tokens"] == 128,
                        "foreground_initially_cold": row["foreground"]["meta_info"]["cached_tokens"] == 0,
                        "source_initially_cold": row["source_prepare"]["meta_info"]["cached_tokens"] == 0,
                    }
                    if mode == "copy-during-decode":
                        row["sticky_child"] = await stream(session, urls[0], child, args.output_tokens)
                        row["migrated_child"] = await stream(session, urls[1], child, args.output_tokens)
                        row["checks"].update({
                            "migration_preserves_sticky": row["sticky_child"]["output_ids"] == row["migrated_child"]["output_ids"],
                            "migrated_child_hit": row["migrated_child"]["meta_info"]["cached_tokens"] == len(parent),
                            "overlapped_foreground": row["copy_finished_before_foreground"],
                        })
                    row["success"] = all(row["checks"].values())
                    if not row["success"]:
                        raise RuntimeError("Contention control cache/length/continuation/overlap check failed")
                except Exception as exc:
                    row["error"] = repr(exc)
                    if foreground is not None:
                        await asyncio.gather(foreground, return_exceptions=True)
                    raise
                finally:
                    records.append(row)
                    log.write(json.dumps(row) + "\n")
                    paired_deltas = []
                    for index in range(args.repetitions):
                        pair = {record["mode"]: record for record in records
                                if record["repetition"] == index and record["success"]}
                        if len(pair) == 2:
                            paired_deltas.append(pair["copy-during-decode"]["decode_seconds"] - pair["no-copy"]["decode_seconds"])
                    summary = {
                        "scope": "Synthetic one-copy/one-foreground control; 8k source snapshot copied after the first token of an unrelated 1k-input/128-output request. Measures decode/E2E interference, not TTFT interference or saturated-cluster behavior.",
                        "paired_decode_penalty_seconds": paired_deltas,
                        "median_decode_penalty_seconds": statistics.median(paired_deltas) if paired_deltas else None,
                        "records": [{key: record[key] for key in ("phase", "repetition", "mode", "success", "decode_seconds", "checks") if key in record} for record in records],
                    }
                    (args.output / "contention_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                    print(json.dumps({"contention": mode, "repetition": repetition, "success": row["success"],
                                      "decode_seconds": row.get("decode_seconds"), "error": row.get("error")}), flush=True)


async def source_active_control(args, session, urls, coordinator, store, events):
    """Test whether the exact prompt snapshot is already visible during decode."""
    records = []
    with (args.output / "source_active_records.jsonl").open("w", buffering=1) as log:
        for repetition in range(-1, args.repetitions):
            rng = random.Random(args.seed + 912 + repetition)
            parent = [rng.randrange(1100, 19000) for _ in range(8192)]
            child = parent + [rng.randrange(20000, 25000) for _ in range(args.suffix_tokens)]
            row = {"phase": "smoke" if repetition < 0 else "measured", "repetition": repetition,
                   "prefix_tokens": len(parent), "parent_output_tokens": 128,
                   "parent_sha256": digest(parent), "success": False}
            foreground = None
            try:
                await flush(session, urls)
                agent = uuid.uuid4().hex
                store.register_agent(AgentContinuation(agent, 0, "source", 1, tuple(parent)))
                first_seen = asyncio.Event()

                async def notify_first(_payload, _latency):
                    row["first_token_monotonic"] = time.perf_counter()
                    first_seen.set()

                foreground = asyncio.create_task(stream(session, urls[0], parent, 128, notify_first))
                await asyncio.wait_for(first_seen.wait(), 180)
                row["parent_active_at_migration_start"] = not foreground.done()
                events.clear()
                started = time.perf_counter()
                migration = None
                try:
                    migration = await coordinator.migrate_async(agent, "destination")
                    row["active_migration_succeeded"] = True
                    row["transfer_bytes"] = migration.bytes_transferred
                    coordinator.acknowledge_async_migration(migration.migration_id)
                except Exception as exc:
                    row["active_migration_succeeded"] = False
                    row["active_migration_error"] = repr(exc)
                finally:
                    finish = time.perf_counter()
                    row["active_migration_events"] = list(events)
                    row["active_migration_stages"] = migration_stages(events, started, finish)
                    row["parent_active_at_migration_end"] = not foreground.done()
                row["parent"] = await foreground
                row["checks"] = {
                    "started_during_parent_decode": row["parent_active_at_migration_start"],
                    "parent_initially_cold": row["parent"]["meta_info"]["cached_tokens"] == 0,
                    "parent_completed": row["parent"]["meta_info"]["completion_tokens"] == 128,
                }
                if migration is None:
                    cleanup = [event for event in events if "/cleanup/" in event["path"]]
                    row["checks"]["failed_attempt_cleanup"] = len(cleanup) == 2 and all(
                        event.get("response", {}).get("state") == "CLEANED" for event in cleanup)
                    aborted = store.conn.execute("SELECT state FROM migrations WHERE agent_id=?", (agent,)).fetchall()
                    row["checks"]["failed_attempt_aborted"] = len(aborted) == 1 and aborted[0][0] == "ABORTED"
                    events.clear()
                    retry_started = time.perf_counter()
                    migration = await coordinator.migrate_async(agent, "destination")
                    row["after_completion_retry_wall_seconds"] = time.perf_counter() - retry_started
                    row["after_completion_retry_events"] = list(events)
                    coordinator.acknowledge_async_migration(migration.migration_id)
                else:
                    row["checks"]["finished_during_parent_decode"] = row["parent_active_at_migration_end"]
                row["sticky_child"] = await stream(session, urls[0], child, args.output_tokens)
                row["migrated_child"] = await stream(session, urls[1], child, args.output_tokens)
                row["checks"].update({
                    "migration_preserves_sticky": row["sticky_child"]["output_ids"] == row["migrated_child"]["output_ids"],
                    "source_child_hit": row["sticky_child"]["meta_info"]["cached_tokens"] == len(parent),
                    "destination_child_hit": row["migrated_child"]["meta_info"]["cached_tokens"] == len(parent),
                    "owner_committed": store.get_agent(agent).owner_engine == "destination",
                })
                row["success"] = all(row["checks"].values())
                if not row["success"]:
                    raise RuntimeError("Source-active probe integration check failed")
            except Exception as exc:
                row["error"] = repr(exc)
                if foreground is not None:
                    await asyncio.gather(foreground, return_exceptions=True)
                raise
            finally:
                records.append(row)
                log.write(json.dumps(row) + "\n")
                (args.output / "source_active_summary.json").write_text(json.dumps({
                    "scope": "8k synthetic parent, 128 decode outputs. Attempt exact prompt snapshot copy immediately after first token; source generation continues. If unavailable, retain failed attempt and validate cleanup plus same-group retry after completion, then stop.",
                    "records": records,
                }, indent=2) + "\n")
                print(json.dumps({key: row.get(key) for key in ("phase", "repetition", "active_migration_succeeded", "active_migration_error", "checks", "success", "error")}), flush=True)
            if not row["active_migration_succeeded"]:
                break


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    if any(length <= 0 or length % 16 for length in args.lengths):
        raise ValueError("Prefix lengths must be positive multiples of 16")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.transfer_port))
    events = []
    urls = [url.rstrip("/") for url in args.engines]
    clients = {name: TimedClient(name, url, events) for name, url in zip(("source", "destination"), urls[:2])}
    store = calibration_store(args)
    coordinator = HybridMigrationCoordinator(store, clients, base_port=args.transfer_port)
    rng = random.Random(args.seed)
    rows = []
    timeout = aiohttp.ClientTimeout(total=180)
    group_seconds = None
    async with aiohttp.ClientSession(timeout=timeout) as session:
        info = []
        for url in urls:
            async with session.get(url + "/get_server_info") as response:
                response.raise_for_status()
                info.append(await response.json())
        (args.output / "server_info.json").write_text(json.dumps(info, indent=2) + "\n")
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.used,utilization.gpu", "--format=csv"], text=True, capture_output=True)
        (args.output / "gpu_before.txt").write_text(gpu.stdout + gpu.stderr)
        backend_keys = ("model_path", "attention_backend", "mamba_backend", "linear_attn_backend", "mamba_ssm_dtype", "max_total_tokens", "max_mamba_cache_size", "enable_deterministic_inference")
        backend_configs = [{key: server.get("server_args", server).get(key) for key in backend_keys} for server in info]
        (args.output / "backend_validation.json").write_text(json.dumps({"configs": backend_configs, "all_equal": all(config == backend_configs[0] for config in backend_configs)}, indent=2) + "\n")
        if any(value is None for config in backend_configs for value in config.values()):
            raise RuntimeError("Engine backend validation lacks expected fields")
        if not all(config == backend_configs[0] for config in backend_configs):
            raise RuntimeError("Engine backend/capacity configurations differ")
        for url in urls:
            await stream(session, url, [1500 + index for index in range(128)], args.output_tokens)
        group_started = time.perf_counter()
        await coordinator._ensure_group("source", "destination")
        group_seconds = time.perf_counter() - group_started
        (args.output / "group_events.json").write_text(json.dumps(events, indent=2) + "\n")
        if args.source_active_only:
            await source_active_control(args, session, urls, coordinator, store, events)
            await flush(session, urls)
            finish_store(store, args)
            return
        if args.contention_only:
            await contention_control(args, session, urls, coordinator, store, events)
            await flush(session, urls)
            finish_store(store, args)
            return
        with (args.output / "records.jsonl").open("w", buffering=1) as log:
            for length in args.lengths:
                for repetition in range(-1, args.repetitions):
                    phase = "warmup" if repetition < 0 else "measured"
                    token_rng = random.Random(args.seed + length * 100 + repetition)
                    parent = [token_rng.randrange(1100, 19000) for _ in range(length)]
                    child = parent + [token_rng.randrange(20000, 25000) for _ in range(args.suffix_tokens)]
                    modes = rng.sample(list(MODES), len(MODES))
                    row = {"phase": phase, "repetition": repetition, "prefix_tokens": length,
                           "suffix_tokens": args.suffix_tokens, "order": modes,
                           "parent_sha256": digest(parent), "child_sha256": digest(child),
                           "results": {}, "success": False}
                    try:
                        for mode in modes:
                            await flush(session, urls)
                            result = {"source_prepare": await stream(session, urls[0], parent, 1),
                                      "preparation_wall_seconds": 0.0, "owner_committed": None}
                            row["results"][mode] = result
                            if mode == "prepared-full-copy":
                                agent = uuid.uuid4().hex
                                store.register_agent(AgentContinuation(agent, 0, "source", 1, tuple(parent)))
                                events.clear()
                                started = time.perf_counter()
                                try:
                                    migration = await coordinator.migrate_async(agent, "destination")
                                finally:
                                    finish = time.perf_counter()
                                    result["migration_events"] = list(events)
                                    result["migration_stages"] = migration_stages(events, started, finish)
                                    result["preparation_wall_seconds"] = finish - started
                                result["transfer_bytes"] = migration.bytes_transferred
                                result["coordinator_transfer_seconds"] = migration.transfer_seconds
                                result["owner_committed"] = store.get_agent(agent).owner_engine == "destination"
                                coordinator.acknowledge_async_migration(migration.migration_id)
                            elif mode == "reconstruction":
                                result["reconstruction"] = await stream(session, urls[1], parent, 1)
                                result["preparation_wall_seconds"] = result["reconstruction"]["e2e_seconds"]
                                result["reconstructed_input_tokens"] = length - result["reconstruction"]["meta_info"]["cached_tokens"]
                                result["preparation_output_tokens"] = result["reconstruction"]["meta_info"]["completion_tokens"]
                            result["child"] = await stream(session, urls[0] if mode == "sticky" else urls[1], child, args.output_tokens)
                            result["output_sha256"] = digest(result["child"]["output_ids"])
                        row["checks"] = pair_checks(row["results"], length, args.suffix_tokens, args.output_tokens)
                        row["success"] = row["checks"]["mechanism_valid"]
                        if not row["success"]:
                            raise RuntimeError("Cache, length, ownership or migration/sticky check failed")
                    except Exception as exc:
                        row["error"] = repr(exc)
                        raise
                    finally:
                        log.write(json.dumps(row) + "\n")
                        rows.append(row)
                        summary = summarize(rows, group_seconds, args)
                        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                        print(json.dumps({"phase": phase, "prefix_tokens": length, "repetition": repetition,
                                          "success": row["success"], "checks": row.get("checks"),
                                          "error": row.get("error")}), flush=True)
        await flush(session, urls)
    finish_store(store, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engines", nargs="+", default=["http://127.0.0.1:31480", "http://127.0.0.1:31481", "http://127.0.0.1:31482"])
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 8192, 16384])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--suffix-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--transfer-port", type=int, default=31000)
    parser.add_argument("--contention-only", action="store_true")
    parser.add_argument("--source-active-only", action="store_true")
    parser.add_argument("--metadata-directory", type=Path,
                        help="Explicit live SQLite location; FULL/WAL retained, backed up after measurement")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.engines) < 2 or args.repetitions < 1:
        parser.error("At least two engines and one measured repetition are required")
    if args.contention_only and args.source_active_only:
        parser.error("Choose only one optional control per invocation")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
