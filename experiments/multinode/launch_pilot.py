"""Launch identical TP2 pilot engines on verified free cards of four nodes.

One engine per node uses eight devices, not the requested 32-device campaign.
No existing process is stopped. Availability is checked before each launch.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IMAGE = "harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10-sf_b075"
ENGINES = [
    {"node": "node27", "address": "12.11.3.27", "devices": "6,7"},
    {"node": "node28", "address": "12.11.3.28", "devices": "4,5"},
    {"node": "node29", "address": "12.11.3.29", "devices": "0,1"},
    {"node": "node30", "address": "12.11.3.30", "devices": "0,1"},
]


def command_for(engine, *, max_running=16, mamba_slots=128):
    name = "zhujianian-as-multinode-pilot"
    vendor = "/usr/local/lib/python3.10/dist-packages/sglang/srt"
    overlay = Path("/home/zhujianian/hygon-runtime-overlay")
    cache = Path("/home/zhujianian/.cache/hygon-multinode") / engine["node"]
    cache.mkdir(parents=True, exist_ok=True)
    command = ["docker", "run", "-d", "--name", name, "--network", "host", "--shm-size", "16g",
               "--ulimit", "memlock=-1:-1"]
    for device in ("/dev/kfd", "/dev/mkfd", "/dev/dri", "/dev/infiniband"):
        command += ["--device", device]
    mounts = ["/usr/local/hyhal:/opt/hyhal:ro", "/home/zhujianian/Qwen3.6-35B-A3B:/model:ro",
              f"{cache}:/root/.triton",
              f"{ROOT}/experiments/hygon_runtime/moe_align_block_size.py:{vendor}/layers/moe/moe_runner/triton_utils/moe_align_block_size.py:ro"]
    for file in ("managers/io_struct.py", "managers/tokenizer_control_mixin.py", "managers/scheduler.py",
                 "entrypoints/http_server.py", "managers/hygon_hybrid_control.py"):
        mounts.append(f"{overlay / file}:{vendor}/{file}:ro")
    for mount in mounts:
        command += ["-v", mount]
    env = {"HIP_VISIBLE_DEVICES": engine["devices"], "SGLANG_KV_LAYOUT_HCU_FA": "false",
           "HSA_FORCE_FINE_GRAIN_PCIE": "1",
           "NCCL_DEBUG": "INFO", "NCCL_NET_PLUGIN": "shca", "NCCL_NET": "IBext_v8", "NCCL_IB_DISABLE": "0",
           "NCCL_IB_HCA": "shca_0",
           "NCCL_SOCKET_IFNAME": "ib0", "GLOO_SOCKET_IFNAME": "ib0", "ENGINE_HOST": engine["address"]}
    for key, value in env.items():
        command += ["-e", f"{key}={value}"]
    command += ["--entrypoint", "bash", IMAGE, "-lc",
                'mkdir -p /var/log/hylog; source /opt/dtk/env.sh; unset NCCL_TOPO_FILE; exec python -m sglang.launch_server '
                '--model-path /model --host "$ENGINE_HOST" --port 31600 --tp-size 2 '
                '--mem-fraction-static 0.90 --context-length 131072 --max-total-tokens 262144 '
                f'--chunked-prefill-size 4096 --max-running-requests {max_running} --max-mamba-cache-size {mamba_slots} '
                '--disable-cuda-graph --attention-backend triton --page-size 1 '
                '--moe-runner-backend triton --disable-custom-all-reduce '
                '--mamba-scheduler-strategy extra_buffer']
    return command


def main():
    import shlex
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=[e["node"] for e in ENGINES], required=True)
    parser.add_argument("--max-running", type=int, default=16)
    parser.add_argument("--mamba-slots", type=int, default=128)
    args = parser.parse_args()
    engine = next(e for e in ENGINES if e["node"] == args.node)
    memory = json.loads(subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", args.node,
         "/usr/local/hyhal/bin/hy-smi --showmeminfo vram --json"], text=True))
    occupied = {device: memory[f"card{device}"]["vram Total Used Memory (MiB)"]
                for device in engine["devices"].split(",")
                if int(memory[f"card{device}"]["vram Total Used Memory (MiB)"]) > 1024}
    if occupied:
        raise SystemExit(f"Refusing to share occupied devices on {args.node}: {occupied}")
    command = command_for(engine, max_running=args.max_running, mamba_slots=args.mamba_slots)
    # Names are task-specific. Docker refuses to overwrite an existing container.
    remote = ["ssh", "-o", "BatchMode=yes", args.node, shlex.join(command)]
    subprocess.run(remote, check=True)
    print(json.dumps(engine | {"endpoint": f"http://{engine['address']}:31600"}), flush=True)


if __name__ == "__main__":
    main()
