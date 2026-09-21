#!/usr/bin/env python3
"""Exclusive, bounded graph-readiness and physical-lifecycle experiment.

Only port 31482 is stopped/restarted. Ports 31480 and 31481 are survivor probes.
The caller must have exclusive dedicated engines and a clean transfer ledger.
"""

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

import aiohttp
import psutil

from experiments.fluidkv.engine_lifecycle import capture_owned_engine, release_engine, start_engine


async def run(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    summary = {"scope": "Single-engine CUDA-graph L0 candidate and physical retirement; no serving policy comparison", "events": []}

    def save(name, value):
        (root / name).write_text(json.dumps(value, indent=2) + "\n")

    def event(kind, **values):
        record = {"event": kind, "monotonic": time.perf_counter(), **values}
        summary["events"].append(record)
        save("summary.json", summary)
        print(json.dumps(record), flush=True)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async def get(url, path):
            async with session.get(url + path, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                if path == "/health":
                    await response.read()
                    return {"status": response.status}
                return await response.json()

        async def idle(url):
            health = await get(url, "/health")
            loads = await get(url, "/v1/loads?include=all")
            if any(row["num_running_reqs"] or row["num_waiting_reqs"] or any(row.get("queues", {}).values()) for row in loads["loads"]):
                raise RuntimeError("engine still has active foreground work")
            return {"url": url, "health": health, "loads": loads}

        async def flush(url):
            await idle(url)
            async with session.post(url + "/flush_cache") as response:
                response.raise_for_status()
                return await response.text()

        async def generate(url, tokens, output, tag):
            rid = "graph-readiness-" + uuid.uuid4().hex
            started, first, final = time.perf_counter(), None, None
            arrivals = []
            async with session.post(url + "/generate", json={
                "rid": rid, "input_ids": tokens, "stream": True,
                "sampling_params": {"max_new_tokens": output, "temperature": 0, "ignore_eos": True},
            }) as response:
                response.raise_for_status()
                async for raw in response.content:
                    if not raw.startswith(b"data:"):
                        continue
                    payload = raw[5:].strip()
                    if not payload or payload == b"[DONE]":
                        continue
                    final = json.loads(payload)
                    if final.get("output_ids") or final.get("meta_info", {}).get("completion_tokens", 0):
                        now = time.perf_counter()
                        first = now if first is None else first
                        arrivals.append({"monotonic": now, "completion_tokens": final["meta_info"]["completion_tokens"]})
            finished = time.perf_counter()
            if first is None or final is None:
                raise RuntimeError("generation contained no actual output")
            meta = final["meta_info"]
            if meta["prompt_tokens"] != len(tokens) or meta["completion_tokens"] != output:
                raise RuntimeError("backend changed requested prompt/output length")
            result = {"tag": tag, "rid": rid, "url": url, "input_tokens": len(tokens),
                      "started_monotonic": started, "first_token_monotonic": first,
                      "finished_monotonic": finished, "ttft_seconds": first-started,
                      "e2e_seconds": finished-started,
                      "decode_tokens_per_second": (output-1)/(finished-first) if output > 1 else None,
                      "output_tokens_per_second": output/(finished-started),
                      "meta_info": meta, "output_ids": final["output_ids"],
                      "input_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
                      "stream_arrivals": arrivals}
            with (root / "requests.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            return result

        async def probes(url, prefix):
            await flush(url)
            await generate(url, [1500+i for i in range(128)], 8, prefix+"-warmup")
            single = await generate(url, [4000+i for i in range(256)], 64, prefix+"-single")
            event(prefix+"_single", decode_tokens_per_second=single["decode_tokens_per_second"], ttft_seconds=single["ttft_seconds"])
            began = time.perf_counter()
            batch = await asyncio.gather(*(generate(url, [7000+j*1000+i for i in range(256)], 64, prefix+f"-batch-{j}") for j in range(4)), return_exceptions=True)
            elapsed = time.perf_counter()-began
            failures = [str(row) for row in batch if isinstance(row, BaseException)]
            if failures:
                raise RuntimeError("finite batch4 failed: " + "; ".join(failures))
            event(prefix+"_batch4", elapsed_seconds=elapsed, aggregate_output_tokens_per_second=256/elapsed,
                  decode_tokens_per_second=[row["decode_tokens_per_second"] for row in batch])
            await flush(url)
            parent = [1100+i for i in range(1024)]
            await generate(url, parent, 1, prefix+"-parent")
            child = await generate(url, parent+[23000+i for i in range(64)], 32, prefix+"-cached-child")
            if child["meta_info"]["cached_tokens"] < len(parent):
                raise RuntimeError("expected prompt snapshot unavailable to child")
            await flush(url)
            cold = await generate(url, parent+[23000+i for i in range(64)], 32, prefix+"-cold-child")
            result = {"single_decode_tokens_per_second": single["decode_tokens_per_second"],
                      "batch4_output_tokens_per_second": 256/elapsed,
                      "child_cached_tokens": child["meta_info"]["cached_tokens"],
                      "cached_child_ttft_seconds": child["ttft_seconds"],
                      "cold_child_ttft_seconds": cold["ttft_seconds"],
                      "cold_cached_output_equal": child["output_ids"] == cold["output_ids"]}
            await flush(url)
            return result

        urls = [f"http://127.0.0.1:{port}" for port in (31480,31481,31482)]
        originals = [capture_owned_engine(url) for url in urls]
        save("original-identities.json", originals)
        save("initial-idle.json", [await idle(url) for url in urls])
        save("initial-server-info.json", [await get(url,"/get_server_info") for url in urls])
        event("initial_idle_confirmed", pids=[x["pid"] for x in originals], transfer_prerequisite="Caller reports all hybrid jobs terminal and cleaned")
        summary["eager"] = await probes(urls[2], "eager")
        event("eager_complete", **summary["eager"])
        await idle(urls[2])
        release = await release_engine(originals[2], no_foreground=True, no_transfers=True)
        save("original-release.json", release)
        event("physical_release", **{k:release.get(k) for k in ("released","graceful","forced_kill","shutdown_seconds","confirmed_monotonic")})
        if not release["released"]:
            raise RuntimeError("original engine release not confirmed; stop experiment")
        survivors = []
        for url in urls[:2]:
            survivors.append(await generate(url, [1800+i for i in range(128)],8,"survivor-after-stop"))
            await flush(url)
        save("survivors-after-stop.json", survivors)
        event("survivors_healthy", urls=urls[:2])
        candidate = {**originals[2], "cmdline": [arg for arg in originals[2]["cmdline"] if arg != "--disable-cuda-graph"] + ["--cuda-graph-bs", "1", "2", "4", "8", "16"]}
        save("candidate-launch-identity.json", candidate)
        candidate_identity = None
        try:
            event("candidate_launch_begin", graph_batch_sizes=[1,2,4,8,16])
            candidate_identity = await start_engine(candidate, root/"candidate-engine.log", readiness_timeout=180)
            save("candidate-live-identity.json", candidate_identity)
            event("candidate_ready", pid=candidate_identity["pid"], startup_seconds=candidate_identity["ready_monotonic"]-candidate_identity["started_monotonic"])
            info = await get(urls[2],"/get_server_info")
            save("candidate-server-info.json", info)
            event("candidate_backend", flags={key:info.get(key) for key in ("disable_cuda_graph","cuda_graph_bs","attention_backend","mamba_backend","linear_attn_backend","enable_deterministic_inference","max_total_tokens","max_mamba_cache_size","max_running_requests")})
            summary["graph"] = await probes(urls[2],"graph")
            summary["status"] = "graph_l0_passed"
            event("graph_complete", **summary["graph"])
        except Exception as exc:
            failure = {"error":str(exc), "launch_pid":getattr(exc,"launch_pid",None), "launch_create_time":getattr(exc,"launch_create_time",None)}
            save("candidate-failure.json",failure)
            event("candidate_failed", **failure)
            try:
                current = capture_owned_engine(urls[2])
            except RuntimeError:
                current = None
            if current is not None:
                if current["cmdline"] != candidate["cmdline"]:
                    raise RuntimeError("candidate failure left an unexpected engine; refusing cleanup") from exc
                try:
                    async with session.post(urls[2]+"/abort_request",json={"rid":"graph-readiness-"},timeout=aiohttp.ClientTimeout(total=10)) as response:
                        await response.read()
                except Exception as abort_error:
                    event("candidate_abort_unavailable",error=str(abort_error))
                drained = False
                for _ in range(30):
                    try:
                        await idle(urls[2])
                        drained = True
                        break
                    except Exception:
                        await asyncio.sleep(1)
                if drained and current["gpu_uuids"]:
                    cleanup = await release_engine(current,no_foreground=True,no_transfers=True)
                    save("candidate-failure-release.json",cleanup)
                    if not cleanup["released"]:
                        raise RuntimeError("failed candidate release was not confirmed") from exc
                else:
                    # Failure recovery only: an unhealthy/loading candidate may
                    # not expose HTTP drain. These are exclusively our failed
                    # probes; do not claim this cleanup as lossless retirement.
                    processes = []
                    for identity in current["owned_processes"]:
                        try:
                            process = psutil.Process(identity["pid"])
                            if process.create_time() == identity["create_time"]:
                                processes.append((identity,process))
                        except psutil.NoSuchProcess:
                            pass
                    launched = psutil.Process(current["pid"])
                    if launched.create_time() != current["create_time"]:
                        raise RuntimeError("candidate PID changed before failure cleanup")
                    launched.terminate()
                    await asyncio.to_thread(psutil.wait_procs,[p for _,p in processes],timeout=15)
                    killed = []
                    for identity,process in reversed(processes):
                        try:
                            if process.create_time() == identity["create_time"] and process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                                process.kill()
                                killed.append(process.pid)
                        except psutil.NoSuchProcess:
                            pass
                    await asyncio.to_thread(psutil.wait_procs,[p for _,p in processes],timeout=10)
                    save("candidate-failure-cleanup.json",{"graceful":False,"lossless_claim":False,"killed_pids":killed,"identity":current})
            restored = await start_engine(originals[2],root/"restored-engine.log",readiness_timeout=180)
            save("restored-live-identity.json",restored)
            await generate(urls[2],[1400+i for i in range(128)],8,"restored-eager")
            await flush(urls[2])
            summary["status"] = "graph_failed_eager_restored"
            event("eager_restored",pid=restored["pid"])
        save("final-idle.json",[await idle(url) for url in urls])
        event("complete",status=summary["status"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    asyncio.run(run(parser.parse_args()))
