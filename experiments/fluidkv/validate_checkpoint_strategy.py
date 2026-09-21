#!/usr/bin/env python3
"""Bounded real-engine checkpoint coverage probe; reconfigures only 31482."""

import argparse
import asyncio
import hashlib
import json
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
from experiments.fluidkv.engine_lifecycle import capture_owned_engine, release_engine, start_engine


LENGTHS = (95,1007,4093,8191,10031,16383)


def strategy_arguments(argv, *, overlap):
    result,index=[],0
    while index<len(argv):
        arg=argv[index]
        if arg=="--mamba-scheduler-strategy":
            index+=2
            continue
        if arg.startswith("--mamba-scheduler-strategy=") or arg=="--disable-overlap-schedule":
            index+=1
            continue
        result.append(arg)
        index+=1
    result.extend(["--mamba-scheduler-strategy","extra_buffer"])
    if not overlap:
        result.append("--disable-overlap-schedule")
    return result


async def run(args):
    root=args.output
    root.mkdir(parents=True,exist_ok=False)
    summary={"scope":"Immutable prompt checkpoint coverage; not live decode handoff", "events":[],"phases":{}}

    def save(name,value):
        (root/name).write_text(json.dumps(value,indent=2)+"\n")

    def event(kind,**values):
        row={"event":kind,"monotonic":time.perf_counter(),**values}
        summary["events"].append(row)
        save("summary.json",summary)
        print(json.dumps(row),flush=True)

    urls=[f"http://127.0.0.1:{port}" for port in (31480,31481,31482)]
    originals=[capture_owned_engine(url) for url in urls]
    save("original-identities.json",originals)
    current=originals[2]
    runtime=Path("/data-aisoft/zhujianian/cloud/sglang-hybrid-state/python/sglang/srt")
    sources=[runtime/name for name in ("server_args.py","mem_cache/mamba_radix_cache.py","mem_cache/hybrid_state_transfer.py","managers/schedule_batch.py")]
    save("source-sha256-before.json",{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    records=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async def get(url,path):
            async with session.get(url+path,timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                if path=="/health":
                    await response.read()
                    return {"status":response.status}
                return await response.json()

        async def idle(url):
            health=await get(url,"/health")
            loads=await get(url,"/v1/loads?include=all")
            if any(row["num_running_reqs"] or row["num_waiting_reqs"] or any(row.get("queues",{}).values()) for row in loads["loads"]):
                raise RuntimeError("foreground not drained at "+url)
            return {"url":url,"health":health,"loads":loads}

        async def generate(url,tokens,outputs,label,callback=None):
            began=time.perf_counter()
            result=await stream(session,url,tokens,outputs,on_first_token=callback)
            if result["meta_info"]["prompt_tokens"]!=len(tokens) or result["meta_info"]["completion_tokens"]!=outputs:
                raise RuntimeError("generation length mismatch")
            record={"label":label,"url":url,"started_monotonic":began,"finished_monotonic":time.perf_counter(),
                    "input_tokens":len(tokens),"input_sha256":hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),**result}
            records.append(record)
            with (root/"requests.jsonl").open("a") as handle:
                handle.write(json.dumps(record)+"\n")
            return record

        async def clean(url):
            await idle(url)
            await flush(session,[url])

        def tokens(length):
            return [1000+(index*17)%16000 for index in range(length)]

        async def phase(label,lengths=LENGTHS,outputs=(1,64),tails=(16,64)):
            await clean(urls[2])
            await generate(urls[2],[2200+i for i in range(128)],8,label+"-warmup")
            info=await get(urls[2],"/get_server_info")
            save(label+"-server-info.json",info)
            fields=("mamba_scheduler_strategy","disable_overlap_schedule","disable_cuda_graph","cuda_graph_bs","max_total_tokens","max_mamba_cache_size","max_running_requests","context_length","mamba_ssm_dtype","attention_backend","mamba_backend","linear_attn_backend","enable_deterministic_inference")
            event("phase_begin",phase=label,flags={key:info.get(key) for key in fields})
            cases=[]
            for length in lengths:
                parent=tokens(length)
                shared=length//16*16
                expected=shared//64*64
                for output in outputs:
                    for tail in tails:
                        child=parent[:shared]+[25000+i for i in range(tail)]
                        await clean(urls[2])
                        cold=await generate(urls[2],child,16,f"{label}-{length}-{output}-{tail}-cold")
                        await clean(urls[2])
                        parent_result=await generate(urls[2],parent,output,f"{label}-{length}-{output}-{tail}-parent")
                        sticky=await generate(urls[2],child,16,f"{label}-{length}-{output}-{tail}-sticky")
                        hit=sticky["meta_info"]["cached_tokens"]
                        case={"parent_tokens":length,"parent_output_tokens":output,"shared_complete_block_tokens":shared,
                              "divergent_tail_tokens":tail,"expected_aligned_checkpoint":expected,"actual_cached_tokens":hit,
                              "checkpoint_matches_expected":hit==expected,"cached_cold_output_equal":cold["output_ids"]==sticky["output_ids"],
                              "parent_initially_cold":parent_result["meta_info"]["cached_tokens"]==0,
                              "cold_initially_cold":cold["meta_info"]["cached_tokens"]==0,
                              "cold_ttft_seconds":cold["ttft_seconds"],"sticky_ttft_seconds":sticky["ttft_seconds"]}
                        cases.append(case)
                        save(label+"-cases.json",cases)
                    event("parent_length_output_complete",phase=label,parent_tokens=length,parent_output_tokens=output,
                          actual_hits=[case["actual_cached_tokens"] for case in cases[-len(tails):]],
                          cached_cold_equal=[case["cached_cold_output_equal"] for case in cases[-len(tails):]])
            await clean(urls[2])
            summary["phases"][label]={"cases":len(cases),"nonzero_checkpoint_hits":sum(case["actual_cached_tokens"]>0 for case in cases),
                                       "expected_checkpoint_hits":sum(case["checkpoint_matches_expected"] for case in cases),
                                       "cached_cold_output_equal":sum(case["cached_cold_output_equal"] for case in cases)}
            event("phase_complete",phase=label,**summary["phases"][label])
            return cases

        async def reload(label,identity):
            nonlocal current
            await idle(urls[2])
            released=await release_engine(current,no_foreground=True,no_transfers=True)
            save(label+"-release.json",released)
            event("engine_released",phase=label,released=released["released"],graceful=released["graceful"],shutdown_seconds=released.get("shutdown_seconds"))
            if not released["released"]:
                raise RuntimeError("physical release unconfirmed")
            save(label+"-launch-identity.json",identity)
            current=await start_engine(identity,root/(label+"-engine.log"),readiness_timeout=180)
            save(label+"-live-identity.json",current)
            event("engine_ready",phase=label,pid=current["pid"],startup_seconds=current["ready_monotonic"]-current["started_monotonic"])

        async def checkpoint_smoke():
            # Group has one direction only: extra_buffer source E2 → E0.
            await clean(urls[0])
            await clean(urls[2])
            store=SQLiteStateStore(root/"checkpoint-copy.db")
            clients={"source":SGLangAgentShiftClient("source",urls[2],timeout=120),
                     "destination":SGLangAgentShiftClient("destination",urls[0],timeout=120)}
            coordinator=HybridMigrationCoordinator(store,clients,base_port=args.transfer_port)
            group,port=await coordinator._ensure_group("source","destination")
            parent=tokens(1007)
            shared=1007//16*16
            checkpoint=shared//64*64
            first=asyncio.Event()
            async def first_token(_payload,_delay):first.set()
            running=asyncio.create_task(generate(urls[2],parent,64,"active-parent",first_token))
            await first.wait()
            parent_still_running=not running.done()
            transfer_id="checkpoint-probe-"+uuid.uuid4().hex
            try:
                prepared=await clients["source"].start_async_hybrid_state_transfer(transfer_id=transfer_id,role="source",token_ids=tuple(parent[:checkpoint]),group_name=group,port=port,prepare_only=True)
            finally:
                cleanup=await clients["source"].cleanup_async_hybrid_state_transfer(transfer_id)
                await running
            save("active-checkpoint-prepare.json",{"checkpoint_tokens":checkpoint,"parent_active_at_probe":parent_still_running,"prepared":prepared,"cleanup":cleanup})
            child=parent[:shared]+[25000+i for i in range(64)]
            source=await generate(urls[2],child,16,"active-parent-following-child")
            branch="checkpoint-copy-"+uuid.uuid4().hex
            store.register_agent(AgentContinuation(branch,0,"source",1,tuple(parent[:checkpoint])))
            migration=await coordinator.migrate_async(branch,"destination")
            target=await generate(urls[0],child,16,"migrated-checkpoint-child")
            copied={"checkpoint_tokens":checkpoint,"transfer_bytes":migration.bytes_transferred,"transfer_seconds":migration.transfer_seconds,
                    "target_cached_tokens":target["meta_info"]["cached_tokens"],"source_cached_tokens":source["meta_info"]["cached_tokens"],
                    "target_source_output_equal":target["output_ids"]==source["output_ids"],"parent_active_at_probe":parent_still_running}
            save("checkpoint-copy.json",copied)
            event("checkpoint_copy_complete",**copied)
            await clean(urls[0]);await clean(urls[2])

        save("initial-idle.json",[await idle(url) for url in urls])
        await phase("no-buffer")
        candidate={**originals[2],"cmdline":strategy_arguments(originals[2]["cmdline"],overlap=False)}
        try:
            await reload("extra-buffer-no-overlap",candidate)
            cases=await phase("extra-buffer-no-overlap")
            if not all(case["checkpoint_matches_expected"] and case["parent_initially_cold"] and case["cold_initially_cold"] for case in cases):
                raise RuntimeError("extra-buffer did not provide expected aligned checkpoints")
            await checkpoint_smoke()
            if args.overlap_probe:
                overlap={**current,"cmdline":strategy_arguments(current["cmdline"],overlap=True)}
                await reload("extra-buffer-overlap",overlap)
                await phase("extra-buffer-overlap",lengths=(1007,8191,16383),outputs=(1,64),tails=(64,))
            summary["cold_cached_equivalence_passed"]=all(case["cached_cold_output_equal"] for case in cases)
            summary["status"]=("checkpoint_coverage_passed" if summary["cold_cached_equivalence_passed"]
                               else "checkpoint_coverage_passed_numerical_equivalence_failed")
        except Exception as exc:
            event("candidate_failed",error=str(exc),launch_pid=getattr(exc,"launch_pid",None),launch_create_time=getattr(exc,"launch_create_time",None))
            save("failure.json",{"error":str(exc),"launch_pid":getattr(exc,"launch_pid",None),"launch_create_time":getattr(exc,"launch_create_time",None)})
            # Do not hide a failed candidate: preserve diagnostics and restore
            # only this owned endpoint after confirming foreground drain.
            try:
                current=capture_owned_engine(urls[2])
                await idle(urls[2])
                await reload("restore-original",originals[2])
                await generate(urls[2],[1500+i for i in range(128)],8,"restored-control")
                await clean(urls[2])
                summary["status"]="candidate_failed_original_restored"
            except Exception as restore_error:
                event("restore_failed",error=str(restore_error))
                raise
        save("final-idle.json",[await idle(url) for url in urls])
        save("final-identities.json",[capture_owned_engine(url) for url in urls])
        memory=subprocess.check_output(["nvidia-smi","--query-gpu=uuid,memory.total,memory.used,memory.free","--format=csv,noheader,nounits"],text=True)
        (root/"final-gpu-memory.csv").write_text(memory)
        event("complete",status=summary["status"])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--transfer-port",type=int,default=32600)
    parser.add_argument("--overlap-probe",action="store_true")
    asyncio.run(run(parser.parse_args()))
