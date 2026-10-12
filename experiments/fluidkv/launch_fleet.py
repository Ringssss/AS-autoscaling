#!/usr/bin/env python3
"""Launch a multi-node FluidKV fleet with the PATCHED runtime overlay.

Why this exists next to launch_pilot.py: that script mounts
`/home/zhujianian/hygon-runtime-overlay`, which is the OLD 283-line copy with
no chained-import fix. Every measurement in this round was taken on the patched
358-line version (`hygon_hybrid_control.py` sha256 dc960414...), so a fleet
launched from the old overlay would silently reproduce the pre-fix behaviour.
This script mounts the patched tree and names containers distinctly.

Engines are TP2 on adjacent free cards. Card occupancy is asserted against
live hy-smi before each launch, so nothing occupied is taken by accident.
"""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

ROOT = Path('/home/zhujianian/AS-autoscaling')
IMAGE = 'harbor.sourcefind.cn:5443/dcu/admin/base/sglang:0.5.12-ubuntu22.04-dtk2604-py3.10-sf_b075'
RUNTIME = ROOT / 'experiments/weka_baselines/environment/runtime'
MODEL = '/data/node28/shared/zhujianian/Qwen3.6-35B-A3B'

# (node, address, cards, port). node28 cards 0,1 are held by a co-tenant, so it
# contributes three engines from cards 2..7; node27 and node29 are free.
FLEET = {
    'node27': ('12.11.3.27', ['0,1', '2,3', '4,5'], [31781, 31782, 31783]),
    'node28': ('12.11.3.28', ['2,3', '4,5', '6,7'], [31751, 31752, 31753]),
    'node29': ('12.11.3.29', ['0,1', '2,3', '4,5'], [31761, 31762, 31763]),
}


def vram_used(node):
    out = subprocess.check_output(
        ['ssh', '-o', 'BatchMode=yes', node,
         '/usr/local/hyhal/bin/hy-smi --showmeminfo vram --json'], text=True)
    mem = json.loads(out)
    return {d: int(mem[f'card{d}']['vram Total Used Memory (MiB)'])
            for d in '01234567'}


def command_for(node, address, cards, port, name, max_running, mamba_slots):
    vendor = '/usr/local/lib/python3.10/dist-packages/sglang/srt'
    cache = Path('/home/zhujianian/.cache/hygon-fluidkv') / node
    cache.mkdir(parents=True, exist_ok=True)
    cmd = ['docker', 'run', '-d', '--name', name, '--network', 'host',
           '--shm-size', '16g', '--ulimit', 'memlock=-1:-1']
    for dev in ('/dev/kfd', '/dev/mkfd', '/dev/dri', '/dev/infiniband'):
        cmd += ['--device', dev]
    mounts = [f'/usr/local/hyhal:/opt/hyhal:ro',
              f'{MODEL}:/model:ro',
              f'{cache}:/root/.triton',
              f'{ROOT}/experiments/hygon_runtime/moe_align_block_size.py:'
              f'{vendor}/layers/moe/moe_runner/triton_utils/moe_align_block_size.py:ro']
    for f in ('managers/io_struct.py', 'managers/tokenizer_control_mixin.py',
              'managers/scheduler.py', 'entrypoints/http_server.py',
              'managers/hygon_hybrid_control.py'):
        mounts.append(f'{RUNTIME}/{f}:{vendor}/{f}:ro')
    for m in mounts:
        cmd += ['-v', m]
    env = {'HIP_VISIBLE_DEVICES': cards, 'SGLANG_KV_LAYOUT_HCU_FA': 'false',
           'HSA_FORCE_FINE_GRAIN_PCIE': '1', 'NCCL_DEBUG': 'WARN',
           'NCCL_NET_PLUGIN': 'shca', 'NCCL_NET': 'IBext_v8',
           'NCCL_IB_DISABLE': '0', 'NCCL_IB_HCA': 'shca_0',
           'NCCL_SOCKET_IFNAME': 'ib0', 'GLOO_SOCKET_IFNAME': 'ib0',
           'ENGINE_HOST': address}
    for k, v in env.items():
        cmd += ['-e', f'{k}={v}']
    cmd += ['--entrypoint', 'bash', IMAGE, '-lc',
            'mkdir -p /var/log/hylog; source /opt/dtk/env.sh; '
            'unset NCCL_TOPO_FILE; exec python -m sglang.launch_server '
            '--model-path /model --host "$ENGINE_HOST" '
            f'--port {port} --tp-size 2 '
            '--mem-fraction-static 0.90 --context-length 131072 '
            '--max-total-tokens 262144 --chunked-prefill-size 4096 '
            f'--max-running-requests {max_running} '
            f'--max-mamba-cache-size {mamba_slots} '
            '--disable-cuda-graph --attention-backend triton --page-size 1 '
            '--moe-runner-backend triton --disable-custom-all-reduce '
            '--mamba-scheduler-strategy extra_buffer']
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nodes', nargs='+', default=list(FLEET))
    ap.add_argument('--tag', default='fk')
    ap.add_argument('--max-running', type=int, default=64)
    ap.add_argument('--mamba-slots', type=int, default=256)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    out = ROOT / 'experiments/fluidkv/results'
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for node in a.nodes:
        address, card_list, ports = FLEET[node]
        used = vram_used(node)
        for cards, port in zip(card_list, ports):
            busy = {d: used[d] for d in cards.split(',') if used[d] > 1024}
            if busy:
                print(f'SKIP {node} cards {cards}: occupied {busy}', flush=True)
                continue
            name = f'zhujianian-{a.tag}-{node}-{port}'
            cmd = command_for(node, address, cards, port, name,
                              a.max_running, a.mamba_slots)
            if a.dry_run:
                print(f'DRY {node} cards {cards} port {port} name {name}')
                records.append(dict(node=node, cards=cards, port=port, name=name))
                continue
            cid = subprocess.check_output(
                ['ssh', '-o', 'BatchMode=yes', node, shlex.join(cmd)],
                text=True).strip()
            rec = dict(node=node, address=address, cards=cards, port=port,
                       name=name, container=cid[:12],
                       endpoint=f'http://{address}:{port}')
            records.append(rec)
            print(f'started {node} cards {cards} -> {rec["endpoint"]}', flush=True)

    path = out / f'fleet-{a.tag}.json'
    path.write_text(json.dumps(records, indent=2) + '\n')
    print(f'\nwrote {path} ({len(records)} engines)')


if __name__ == '__main__':
    main()
