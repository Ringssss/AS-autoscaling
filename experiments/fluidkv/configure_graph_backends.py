#!/usr/bin/env python3
"""Reproducibly reload three exclusively owned, drained graph backends."""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp

from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.fluidkv.engine_lifecycle import capture_owned_engine, release_engine, start_engine


def graph_arguments(argv, max_total_tokens=None, *, mamba_scheduler_strategy=None,
                    disable_overlap_schedule=None):
    if max_total_tokens is not None and max_total_tokens <= 0:
        raise ValueError("max-total-tokens must be positive")
    if mamba_scheduler_strategy not in (None, "no_buffer", "extra_buffer"):
        raise ValueError("invalid Mamba scheduler strategy")
    reset_overlap = mamba_scheduler_strategy is not None or disable_overlap_schedule is not None
    result, index = [], 0
    capacity_written = False
    while index < len(argv):
        argument = argv[index]
        if mamba_scheduler_strategy is not None and (
                argument == "--mamba-scheduler-strategy" or argument.startswith("--mamba-scheduler-strategy=")):
            if argument == "--mamba-scheduler-strategy":
                if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                    raise ValueError("captured Mamba scheduler strategy has no value")
                index += 1
            index += 1
            continue
        if reset_overlap and argument == "--disable-overlap-schedule":
            index += 1
            continue
        if max_total_tokens is not None and (
                argument == "--max-total-tokens" or argument.startswith("--max-total-tokens=")):
            if argument == "--max-total-tokens":
                if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                    raise ValueError("captured max-total-tokens argument has no value")
                index += 1
            if not capacity_written:
                result.extend(["--max-total-tokens", str(max_total_tokens)])
                capacity_written = True
            index += 1
            continue
        if argument == "--disable-cuda-graph":
            index += 1
            continue
        if argument == "--cuda-graph-bs":
            index += 1
            while index < len(argv) and not argv[index].startswith("--"):
                index += 1
            continue
        result.append(argument)
        index += 1
    if max_total_tokens is not None and not capacity_written:
        result.extend(["--max-total-tokens", str(max_total_tokens)])
    if mamba_scheduler_strategy is not None:
        result.extend(["--mamba-scheduler-strategy", mamba_scheduler_strategy])
    if disable_overlap_schedule is True:
        result.append("--disable-overlap-schedule")
    return result + ["--cuda-graph-bs", "1", "2", "4", "8", "16"]


async def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    summary = {"events": [], "status": "running", "max_total_tokens_override": args.max_total_tokens,
               "mamba_scheduler_strategy_override": args.mamba_scheduler_strategy,
               "disable_overlap_schedule_override": args.disable_overlap_schedule,
               "transfer_prerequisite": "Exclusive caller confirms every prior hybrid transfer terminal and cleaned"}

    def save(name, value):
        (root/name).write_text(json.dumps(value, indent=2)+"\n")

    def event(kind, **values):
        record = {"event":kind,"monotonic":time.perf_counter(),**values}
        summary["events"].append(record)
        save("summary.json",summary)
        print(json.dumps(record),flush=True)

    urls = [f"http://127.0.0.1:{port}" for port in (31480,31481,31482)]
    original = [capture_owned_engine(url) for url in urls]
    save("original-identities.json",original)
    candidates = [{**identity,"cmdline":graph_arguments(identity["cmdline"], args.max_total_tokens,
                  mamba_scheduler_strategy=args.mamba_scheduler_strategy,
                  disable_overlap_schedule=args.disable_overlap_schedule)} for identity in original]
    save("candidate-launch-identities.json",candidates)
    runtime = Path("/data-aisoft/zhujianian/cloud/sglang-hybrid-state")
    source_paths = ["python/sglang/srt/mem_cache/hybrid_state_transfer.py",
                    "python/sglang/srt/managers/io_struct.py",
                    "python/sglang/srt/managers/scheduler.py",
                    "python/sglang/srt/entrypoints/http_server.py",
                    "python/sglang/srt/managers/tokenizer_communicator_mixin.py",
                    "python/sglang/srt/managers/scheduler_runtime_checker_mixin.py"]
    hashes = {str(runtime/path):hashlib.sha256((runtime/path).read_bytes()).hexdigest() for path in source_paths}
    save("runtime-source-sha256.json",hashes)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async def get(url,path):
            async with session.get(url+path,timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                if path == "/health":
                    await response.read()
                    return {"status":response.status}
                return await response.json()

        async def idle(url):
            health = await get(url,"/health")
            loads = await get(url,"/v1/loads?include=all")
            if any(row["num_running_reqs"] or row["num_waiting_reqs"] or any(row.get("queues",{}).values()) for row in loads["loads"]):
                raise RuntimeError("foreground work still exists on "+url)
            return {"url":url,"health":health,"loads":loads}

        save("initial-idle.json",[await idle(url) for url in urls])
        event("initial_idle_confirmed",pids=[i["pid"] for i in original])
        for index,identity in enumerate(original):
            result = await release_engine(identity,no_foreground=True,no_transfers=True)
            save(f"release-{index}.json",result)
            event("engine_released",engine=index,**{key:result.get(key) for key in ("released","graceful","forced_kill","shutdown_seconds")})
            if not result["released"]:
                raise RuntimeError("release not confirmed for "+identity["url"])

        async def launch(index,candidate):
            identity = await start_engine(candidate,root/f"engine-{index}.log",readiness_timeout=180)
            save(f"live-identity-{index}.json",identity)
            event("engine_ready",engine=index,pid=identity["pid"],startup_seconds=identity["ready_monotonic"]-identity["started_monotonic"])
            return identity

        event("parallel_launch_begin",engines=3,graph_batch_sizes=[1,2,4,8,16])
        outcomes = await asyncio.gather(*(launch(index,candidate) for index,candidate in enumerate(candidates)),return_exceptions=True)
        failures = [{"engine":index,"error":str(result),"launch_pid":getattr(result,"launch_pid",None),"launch_create_time":getattr(result,"launch_create_time",None)} for index,result in enumerate(outcomes) if isinstance(result,BaseException)]
        if failures:
            save("startup-failures.json",failures)
            event("startup_failed",failures=failures)
            raise RuntimeError("backend startup failed; preserve owned process identities for recovery")
        save("live-identities.json",outcomes)
        info = [await get(url,"/get_server_info") for url in urls]
        save("server-info.json",info)
        keys = ("model_path","context_length","dtype","tp_size","mem_fraction_static","max_total_tokens","max_mamba_cache_size","max_running_requests","mamba_ssm_dtype","mamba_full_memory_ratio","attention_backend","mamba_backend","linear_attn_backend","enable_deterministic_inference","disable_cuda_graph","cuda_graph_bs")
        keys += ("mamba_scheduler_strategy", "disable_overlap_schedule")
        flags = [{key:server.get(key) for key in keys} for server in info]
        if not all(item == flags[0] for item in flags):
            raise RuntimeError("backend configuration mismatch")
        if flags[0]["disable_cuda_graph"] or flags[0]["cuda_graph_bs"] != [1,2,4,8,16]:
            raise RuntimeError("validated graph configuration missing")
        if args.max_total_tokens is not None and flags[0]["max_total_tokens"] != args.max_total_tokens:
            raise RuntimeError("configured token capacity differs from requested override")
        if args.mamba_scheduler_strategy is not None and flags[0]["mamba_scheduler_strategy"] != args.mamba_scheduler_strategy:
            raise RuntimeError("configured Mamba strategy differs from requested override")
        if args.disable_overlap_schedule is not None and flags[0]["disable_overlap_schedule"] != args.disable_overlap_schedule:
            raise RuntimeError("effective overlap setting differs from requested override")
        if args.mamba_scheduler_strategy == "extra_buffer" and args.disable_overlap_schedule is None and flags[0]["disable_overlap_schedule"]:
            raise RuntimeError("extra-buffer default should use normal overlap")
        effective = [await get(url,"/v1/loads?include=all") for url in urls]
        save("effective-capacity.json",effective)
        if any(row["max_total_num_tokens"] != flags[0]["max_total_tokens"]
               for engine in effective for row in engine["loads"]):
            raise RuntimeError("effective runtime token capacity differs from common configuration")
        save("common-backend-flags.json",flags[0])
        event("uniform_backend_confirmed",flags=flags[0])
        probes = []
        for index,url in enumerate(urls):
            await stream(session,url,[1500+i for i in range(128)],8)
            await flush(session,[url])
            parent = [1100+i for i in range(1024)]
            before = await stream(session,url,parent,1)
            child = await stream(session,url,parent+[23000+i for i in range(64)],8)
            result = {"engine":index,"url":url,"parent":before,"child":child,
                      "checks":{"parent_cold":before["meta_info"]["cached_tokens"]==0,
                                "parent_snapshot_hit":child["meta_info"]["cached_tokens"]>=1024,
                                "exact_lengths":child["meta_info"]["prompt_tokens"]==1088 and child["meta_info"]["completion_tokens"]==8}}
            probes.append(result)
            save("cache-probes.json",probes)
            event("cache_probe_complete",engine=index,checks=result["checks"],child_ttft_seconds=child["ttft_seconds"])
            if not all(result["checks"].values()):
                raise RuntimeError("backend cache probe failed")
            await flush(session,[url])
        inspection_probes = []
        if flags[0]["mamba_scheduler_strategy"] == "extra_buffer":
            async def inspect(url, tokens):
                async with session.post(url+"/agentshift/hybrid_state/inspect",json={"token_ids":tokens}) as response:
                    response.raise_for_status()
                    return await response.json()

            for index,url in enumerate(urls):
                await flush(session,[url])
                parent = [1000+(i*17)%16000 for i in range(1007)]
                child = parent[:992]+[25000+i for i in range(64)]
                cold = await inspect(url,child)
                parent_result = await stream(session,url,parent,1)
                observation = await inspect(url,child)
                child_result = await stream(session,url,child,8)
                probe = {"engine":index,"before":cold,"after_parent":observation,"parent":parent_result,"child":child_result,
                         "checks":{"cold_directory_empty":cold["matched_tokens"]==0,
                                   "checkpoint_directory":observation["snapshot_boundaries"]==[960],
                                   "checkpoint_actual_hit":child_result["meta_info"]["cached_tokens"]==960,
                                   "directory_matches_actual":observation["matched_tokens"]==child_result["meta_info"]["cached_tokens"],
                                   "lengths_preserved":child_result["meta_info"]["prompt_tokens"]==1056 and child_result["meta_info"]["completion_tokens"]==8}}
                inspection_probes.append(probe)
                save("unaligned-inspection-probes.json",inspection_probes)
                event("unaligned_inspection_complete",engine=index,checks=probe["checks"],actual_cached_tokens=child_result["meta_info"]["cached_tokens"])
                if not all(probe["checks"].values()):
                    raise RuntimeError("unaligned snapshot inspection probe failed")
                await flush(session,[url])
        summary["sampled_cross_engine_output_equal"] = all(probe["child"]["output_ids"] == probes[0]["child"]["output_ids"] for probe in probes)
        save("final-idle.json",[await idle(url) for url in urls])
        if hashes != {str(runtime/path):hashlib.sha256((runtime/path).read_bytes()).hexdigest() for path in source_paths}:
            raise RuntimeError("runtime source changed during backend provisioning")
        summary["status"] = "uniform_graph_backends_ready"
        event("complete",status=summary["status"],sampled_cross_engine_output_equal=summary["sampled_cross_engine_output_equal"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--max-total-tokens",type=int)
    parser.add_argument("--mamba-scheduler-strategy",choices=["no_buffer","extra_buffer"])
    overlap = parser.add_mutually_exclusive_group()
    overlap.add_argument("--disable-overlap-schedule",dest="disable_overlap_schedule",action="store_true")
    overlap.add_argument("--enable-overlap-schedule",dest="disable_overlap_schedule",action="store_false")
    parser.set_defaults(disable_overlap_schedule=None)
    asyncio.run(run(parser.parse_args()))
