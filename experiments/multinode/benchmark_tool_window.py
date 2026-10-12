"""Controlled tool-window/hot-owner experiment on live hybrid engines.

This is a mechanism experiment, not execution of DualMap/CacheRoute and not a
production tool-gap replay. A real local file-search subprocess returns a fixed
result; explicitly controlled minimum delays sweep opportunity for overlap.
The same foreground generation arrives at the source in every policy.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import uuid

import aiohttp
import numpy as np
from transformers import AutoTokenizer

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import flush, stream


async def tool(root):
    code = """import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1])/'README.md'
b=p.read_bytes()
print(json.dumps({'tool':'repository-search','sha256':hashlib.sha256(b).hexdigest(),
 'matches':[s for s in b.decode().splitlines() if 'AgentShift' in s][:3]},sort_keys=True))
"""
    began = time.perf_counter()
    process = await asyncio.create_subprocess_exec(sys.executable, "-c", code, str(root),
                                                  stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = await process.communicate()
    if process.returncode:
        raise RuntimeError(err.decode())
    return out.decode().strip(), time.perf_counter() - began


async def run(args):
    root = Path(__file__).resolve().parents[2]
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    payload, _ = await tool(root)
    suffix = tokenizer.encode("\nTool result:\n" + payload + "\nContinue:", add_special_tokens=False)
    corpus = np.load(args.tokens)["tokens"].tolist()
    urls = [args.source, args.destination]
    store = SQLiteStateStore(args.output / "ownership.db")
    clients = {name: SGLangAgentShiftClient(name, url, timeout=300)
               for name, url in zip(("source", "destination"), urls)}
    coordinator = HybridMigrationCoordinator(store, clients, master_address=args.master_address,
                                             base_port=args.base_port)
    foreground_ids = tokenizer.encode("Explain how a compiler optimizes loops. " * 32, add_special_tokens=False)
    rng = random.Random(args.seed)
    records = []
    modes = ["sticky-hot-owner", "route-cold-idle", "copy-during-tool"]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        info = []
        for url in urls:
            async with session.get(url + "/get_server_info") as response:
                response.raise_for_status()
                info.append(await response.json())
            for length in args.prefix_tokens:
                await flush(session, [url])
                await stream(session, url, corpus[:length], 1)
                await stream(session, url, corpus[:length] + suffix, args.output_tokens)
                # The hot-owner baseline executes concurrent decode/prefill.
                # Warm those batch shapes too; serial warmup alone can charge
                # its first measured request for compilation.
                await flush(session, [url])
                await stream(session, url, corpus[:length], 1)
                await asyncio.gather(
                    stream(session, url, foreground_ids, args.foreground_tokens),
                    stream(session, url, corpus[:length] + suffix, args.output_tokens),
                )
            await stream(session, url, foreground_ids, args.foreground_tokens)
        (args.output / "server_info.json").write_text(json.dumps(info, indent=2) + "\n")
        started = time.perf_counter()
        await coordinator._ensure_group("source", "destination")
        group_seconds = time.perf_counter() - started
        await flush(session, urls)
        parent = corpus[:min(args.prefix_tokens)]
        await stream(session, args.source, parent, 1)
        store.register_agent(AgentContinuation("transport-warmup", 1, "source", 1, tuple(parent)))
        started = time.perf_counter()
        await coordinator.migrate("transport-warmup", "destination")
        warm_seconds = time.perf_counter() - started
        with (args.output / "requests.jsonl").open("w", buffering=1) as log:
            for repeat in range(args.repetitions):
                cases = [(length, gap) for length in args.prefix_tokens for gap in args.tool_seconds]
                rng.shuffle(cases)
                for length, gap in cases:
                    paired = []
                    for mode in rng.sample(modes, len(modes)):
                        await flush(session, urls)
                        parent = corpus[:length]
                        # One generated token establishes the exact cached input
                        # boundary; this is a controlled boundary, not a model-
                        # emitted production tool-call trajectory.
                        parent_result = await stream(session, args.source, parent, 1)
                        agent = uuid.uuid4().hex
                        store.register_agent(AgentContinuation(agent, 1, "source", 1, tuple(parent)))
                        began = time.perf_counter()
                        timeline = {"tool_start": 0.0}

                        async def run_tool():
                            result, work_seconds = await tool(root)
                            timeline["tool_compute_finished"] = time.perf_counter() - began
                            await asyncio.sleep(max(0, began + gap - time.perf_counter()))
                            timeline["tool_return"] = time.perf_counter() - began
                            return result, work_seconds

                        async def background():
                            timeline["foreground_scheduled_arrival"] = args.foreground_delay
                            await asyncio.sleep(max(0, began + args.foreground_delay - time.perf_counter()))
                            timeline["foreground_dispatch"] = time.perf_counter() - began
                            async def first(payload, delay):
                                timeline["foreground_first_token"] = time.perf_counter() - began
                            result = await stream(session, args.source, foreground_ids, args.foreground_tokens, on_first_token=first)
                            timeline["foreground_finished"] = time.perf_counter() - began
                            return result

                        async def copy():
                            timeline["copy_started"] = time.perf_counter() - began
                            migration = await coordinator.migrate(agent, "destination")
                            timeline["copy_committed"] = time.perf_counter() - began
                            return migration

                        tool_task = asyncio.create_task(run_tool())
                        bg_task = asyncio.create_task(background())
                        copy_task = asyncio.create_task(copy()) if mode == "copy-during-tool" else None
                        try:
                            result_payload, work_seconds = await tool_task
                            if result_payload != payload:
                                raise RuntimeError("tool result changed between trials")
                            migration = await copy_task if copy_task else None
                            timeline["continuation_dispatch"] = time.perf_counter() - began
                            target = args.source if mode == "sticky-hot-owner" else args.destination
                            async def first(payload, delay):
                                timeline["continuation_first_token"] = time.perf_counter() - began
                            result = await stream(session, target, parent + suffix, args.output_tokens, on_first_token=first)
                            timeline["continuation_finished"] = time.perf_counter() - began
                            bg = await bg_task
                        except BaseException as exc:
                            log.write(json.dumps({"repeat": repeat, "prefix_tokens": length,
                                                  "controlled_tool_seconds": gap, "mode": mode,
                                                  "failed": True, "error": repr(exc), "timeline": timeline}) + "\n")
                            await asyncio.gather(tool_task, bg_task, *([copy_task] if copy_task else []), return_exceptions=True)
                            raise
                        row = {"repeat": repeat, "prefix_tokens": length, "controlled_tool_seconds": gap,
                               "mode": mode, "timeline": timeline, "tool_work_seconds": work_seconds,
                               "tool_result": payload, "generation": result, "foreground": bg,
                               "parent": parent_result, "transfer_bytes": migration.bytes_transferred if migration else 0,
                               "owner": store.get_agent(agent).owner_engine,
                               "post_tool_ttft_seconds": timeline["continuation_first_token"] - timeline["tool_return"],
                               "post_tool_e2e_seconds": timeline["continuation_finished"] - timeline["tool_return"],
                               "tool_start_to_completion_seconds": timeline["continuation_finished"],
                               "foreground_arrival_to_completion_seconds": timeline["foreground_finished"] - args.foreground_delay,
                               "copy_exposed_seconds": max(0, timeline.get("copy_committed", 0) - timeline["tool_return"]),
                               "foreground_active_at_tool_return": timeline["foreground_dispatch"] <= timeline["tool_return"] < timeline["foreground_finished"]}
                        paired.append(row)
                    reference = next(r for r in paired if r["mode"] == "sticky-hot-owner")["generation"]["output_ids"]
                    cold = next(r for r in paired if r["mode"] == "route-cold-idle")["generation"]["output_ids"]
                    for row in paired:
                        result = row["generation"]
                        row["checks"] = {
                            "exact_sticky": result["output_ids"] == reference,
                            "exact_cold": result["output_ids"] == cold,
                            "cache_hit": result["meta_info"]["cached_tokens"] == (0 if row["mode"] == "route-cold-idle" else length),
                            "input_length": result["meta_info"]["prompt_tokens"] == length + len(suffix),
                            "output_length": result["meta_info"]["completion_tokens"] == args.output_tokens,
                            "foreground_length": row["foreground"]["meta_info"]["completion_tokens"] == args.foreground_tokens,
                            "ownership": row["owner"] == ("destination" if row["mode"] == "copy-during-tool" else "source"),
                            "bytes": row["transfer_bytes"] == (length * 20480 + 64389120 if row["mode"] == "copy-during-tool" else 0),
                        }
                        log.write(json.dumps(row) + "\n")
                        records.append(row)
                    print(json.dumps({"repeat": repeat, "length": length, "gap": gap,
                                      "checks": all(all(r["checks"].values()) for r in paired)}), flush=True)
    summaries = []
    for length in args.prefix_tokens:
        for gap in args.tool_seconds:
            for mode in modes:
                selected = [r for r in records if (r["prefix_tokens"], r["controlled_tool_seconds"], r["mode"]) == (length, gap, mode)]
                summaries.append({"prefix_tokens": length, "tool_seconds": gap, "mode": mode,
                                  "samples": len(selected),
                                  **{key + "_median": float(np.median([r[key] for r in selected])) for key in
                                     ("post_tool_ttft_seconds", "post_tool_e2e_seconds", "foreground_arrival_to_completion_seconds", "copy_exposed_seconds")},
                                  "foreground_active_at_return": sum(r["foreground_active_at_tool_return"] for r in selected),
                                  "all_checks_passed": all(all(r["checks"].values()) for r in selected)})
    report = {"scope": "Controlled tool-window experiment with real foreground inference and synchronous hybrid copy; not production trace or full routing-policy evaluation.",
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "group_initialization_seconds": group_seconds, "first_copy_warmup_seconds": warm_seconds,
              "warmup": "serial and concurrent foreground/continuation shapes plus first RCCL transfer",
              "token_file_sha256": hashlib.sha256(args.tokens.read_bytes()).hexdigest(),
              "limitations": ["idle-source and empty-destination admission", "controlled tool delay, not a production measurement", "one foreground request, no admission queue imposed", "fixed destination, not a load-based placement policy", "completed input snapshot, not a generated tool-call trajectory"],
              "summaries": summaries}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(all(r["checks"].values()) for r in records):
        raise SystemExit("correctness gate failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://12.11.3.27:31600")
    parser.add_argument("--destination", default="http://12.11.3.29:31600")
    parser.add_argument("--master-address", default="12.11.3.27")
    parser.add_argument("--model", default="/home/zhujianian/Qwen3.6-35B-A3B")
    parser.add_argument("--tokens", type=Path, default=Path("experiments/multinode/data/longbench_tokens.npz"))
    parser.add_argument("--prefix-tokens", type=int, nargs="+", default=[1024, 32768])
    parser.add_argument("--tool-seconds", type=float, nargs="+", default=[0.05, 0.5])
    parser.add_argument("--foreground-delay", type=float, default=0.2)
    parser.add_argument("--foreground-tokens", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--base-port", type=int, default=30400)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))
