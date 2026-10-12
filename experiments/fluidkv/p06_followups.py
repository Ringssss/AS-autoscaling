#!/usr/bin/env python3
"""Owned-worker correctness gate and admission-limit controls for p06."""
import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.request

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/skew-readiness-20261009"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


async def correctness():
    skew = module("skew", ROOT / "experiments/fluidkv/p06_skew_readiness.py")
    urls = ["http://12.11.3.27:31881", "http://12.11.3.27:31882"]
    prefix = np.load(ROOT / "experiments/weka_baselines/data/tokens.npy", mmap_mode="r")[:65536].astype(int).tolist()
    suffix = np.random.default_rng(20261009).integers(1000, 240000, 64, dtype=np.int32).tolist()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        await skew.reset(session, urls)
        await skew.generate(session, urls[0], prefix, 16)
        gc = skew.p03.GroupCache()
        if await gc.ensure(session, urls, 0, 1) is None:
            raise RuntimeError(gc.last_error)
        payload = {"token_ids": prefix, "return_digest": True}
        source = await skew.json_post(session, urls[0], "/agentshift/hybrid_state/inspect", payload)
        copy = await skew.p03.copy_state(session, gc, urls, 0, 1, prefix)
        if not copy.get("ok"):
            raise RuntimeError(copy)
        destination = await skew.json_post(session, urls[1], "/agentshift/hybrid_state/inspect", payload)
        # Sequential, same batch shape, same exact inputs: isolate transport.
        a = await skew.generate(session, urls[0], prefix + suffix, 64)
        b = await skew.generate(session, urls[1], prefix + suffix, 64)
        result = {"source_digest": source, "destination_digest": destination, "copy": copy,
                  "source_output": a, "destination_output": b,
                  "state_digest_equal": source["state_sha256"] == destination["state_sha256"],
                  "output_equal": a["output_ids"] == b["output_ids"],
                  "scope": "KV and final executable Mamba snapshot at 64K boundary; same-shape sequential cached continuation, 64 tokens."}
        (BASE / "correctness.json").write_text(json.dumps(result, indent=2) + "\n")
        print("correctness", result["state_digest_equal"], result["output_equal"], flush=True)
        if not result["state_digest_equal"] or not result["output_equal"]:
            raise RuntimeError("transport correctness gate failed; controls stopped")


def restart_owned():
    fleet = json.loads((BASE / "fleet.json").read_text())
    for rec in fleet:
        expected = f"zhujianian-fk-skew-20261009-{rec['url'].rsplit(':', 1)[1]}"
        if rec["name"] != expected:
            raise RuntimeError("refusing unexpected container name")
        identity = subprocess.check_output(["docker", "inspect", "-f", "{{.Id}}", rec["name"]], text=True).strip()
        if identity != rec["container"]:
            raise RuntimeError("owned container identity changed")
        with (BASE / f"{rec['name']}.log").open("w") as f:
            subprocess.run(["docker", "logs", rec["name"]], stdout=f, stderr=subprocess.STDOUT, check=True)
    for rec in fleet:
        subprocess.run(["docker", "rm", "-f", rec["name"]], check=True)
    launch = module("launch", ROOT / "experiments/fluidkv/launch_fleet.py")
    records = []
    for rec in fleet:
        port = int(rec["url"].rsplit(":", 1)[1])
        name = rec["name"] + "-cap64"
        cmd = launch.command_for("node27", "12.11.3.27", rec["cards"], port, name, 64, 512)
        cid = subprocess.check_output(cmd, text=True).strip()
        records.append({"name": name, "container": cid, "cards": rec["cards"], "url": rec["url"], "command": cmd})
        print("restarted", name, flush=True)
    (BASE / "fleet-cap64.json").write_text(json.dumps(records, indent=2) + "\n")
    end, last = time.monotonic() + 900, 0
    while time.monotonic() < end:
        ready = []
        for rec in records:
            try:
                with urllib.request.urlopen(rec["url"] + "/health", timeout=2) as response:
                    ready.append(response.status == 200)
            except Exception:
                ready.append(False)
        if all(ready):
            return
        if time.monotonic() - last > 30:
            print("cap64 loading", ready, flush=True)
            last = time.monotonic()
        time.sleep(3)
    raise RuntimeError("cap64 workers failed to become ready")


def controls():
    for streams, output, reps in [(16, 128, 2), (64, 64, 2)]:
        tag = f"cap64-n{streams}"
        cmd = ["/home/zhujianian/miniconda3/envs/sglang/bin/python3", "-u",
               "experiments/fluidkv/p06_skew_readiness.py", "--prefix", "65536",
               "--streams", str(streams), "--turns", "2", "--output-tokens", str(output),
               "--gap", "1", "--reps", str(reps), "--max-running", "64",
               "--tpot-slo", "0.2", "--warmup", "--policies", "sticky", "balanced", "prepared_copy",
               "--out", str(BASE / tag)]
        (BASE / f"{tag}-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        print("control start", tag, flush=True)
        with (BASE / f"{tag}.log").open("w") as f:
            subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
        print("control done", tag, flush=True)


def main():
    # Start only after every formal trial has ended. Inspect never runs in
    # parallel with foreground generation or migration from that experiment.
    end, last = time.monotonic() + 1800, 0
    while time.monotonic() < end:
        p = BASE / "formal/summary.json"
        data = json.loads(p.read_text()) if p.exists() else {}
        if len(data.get("trials", [])) == 15:
            break
        if time.monotonic() - last > 30:
            print("waiting formal", len(data.get("trials", [])), flush=True)
            last = time.monotonic()
        time.sleep(3)
    else:
        raise RuntimeError("formal trials did not finish")
    asyncio.run(correctness())
    restart_owned()
    controls()


if __name__ == "__main__":
    main()
