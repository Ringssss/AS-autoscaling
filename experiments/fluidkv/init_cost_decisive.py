#!/usr/bin/env python3
"""Decisive test: is `init_once` ~0.1 s or ~20 s?

Prior evidence conflicts by two orders of magnitude:

  STORYLINE v3 §E4   "5 bare link setups, same-node pair"
                     16.3 / 21.1 / 21.2 / 21.5 / 31.3 s, 5/5 ok
  characterize_ready_cost.py (this session), same-node pair
                     0.05 / 0.06 / 0.10 s on 3 successful inits

Both cannot be true. Three hypotheses, each with a distinct signature:

  H1 real cost        init depends on something that differed between the
                      runs (engine warmup, concurrent load, topology cache).
                      => init times cluster, and the clusters track the
                         suspected variable.
  H2 port collision   the slow runs bound a port already held by a dead
                      run's TCPStore, so the handshake waits for a timeout.
                      => slow inits are always ~the timeout value, never
                         intermediate, and they vanish when the port is
                         chosen from the generic ephemeral range.
  H3 stale group      re-initializing the SAME name/port within a live
                      engine waits on the previous store.
                      => slow inits correlate with reuse of a (name,port)
                         pair, not with load.

This script separates them by varying exactly one thing per block:
  A. distinct fresh ports each time, engine idle
  B. the SAME port repeated back-to-back
  C. distinct fresh ports while a co-tenant load is running

Each block reports init seconds, the port used, and whether the TCPStore
handshake or the process-group creation was the slow part (the engine's
response is tagged when it times out).
"""
import asyncio
import json
import random
import socket
import time
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path('/home/zhujianian/AS-autoscaling')
TOKS = np.load(ROOT / 'experiments/weka_baselines/data/tokens.npy')
N = 32768

SRC = '12.11.3.27:31781'
DST = '12.11.3.27:31782'
PORT_LO, PORT_HI = 40000, 60000
INIT_TIMEOUT = 200.0


def free_port_guess():
    """Ask the OS for an unused port, then drop it. The engine binds it."""
    s = socket.socket()
    s.bind(('', 0))
    p = s.getsockname()[1]
    s.close()
    return p


async def post(s, ep, path, payload=None, to=INIT_TIMEOUT):
    try:
        async with s.post(f'http://{ep}{path}', json=payload,
                          timeout=aiohttp.ClientTimeout(total=to)) as r:
            return await r.json()
    except Exception as exc:
        return {'ERR': repr(exc)[:160]}


async def init_pair(s, name, port):
    p = lambda rank: {'group_rank': rank, 'group_name': name,
                      'master_address': SRC.split(':')[0], 'port': port,
                      'world_size': 2, 'backend': 'nccl'}
    t0 = time.perf_counter()
    a, b = await asyncio.gather(
        post(s, SRC, '/agentshift/hybrid_state/direct_group/init', p(0)),
        post(s, DST, '/agentshift/hybrid_state/direct_group/init', p(1)))
    dt = time.perf_counter() - t0
    ok = bool(a.get('success') and b.get('success'))
    return dt, ok, a, b


async def flush(s, ep):
    try:
        async with s.post(f'http://{ep}/flush_cache?timeout=60') as r:
            await r.text()
    except Exception:
        pass


async def install(s, ep, ids):
    r = await post(s, ep, '/generate',
                   {'input_ids': ids,
                    'sampling_params': {'temperature': 0, 'max_new_tokens': 16,
                                        'ignore_eos': True}}, to=300)
    return (r.get('meta_info') or {}).get('cached_tokens')


async def load_co_tenant(s, ids):
    """Fire background requests so the engine is busy during init."""
    tasks = []
    for k in range(3):
        off = 60_000_000 + k * 200_000
        sub = [int(x) for x in TOKS[off:off + 16000]]
        tasks.append(asyncio.create_task(
            post(s, SRC, '/generate',
                 {'input_ids': sub,
                  'sampling_params': {'temperature': 0, 'max_new_tokens': 64,
                                      'ignore_eos': True}}, to=300)))
    return tasks


async def main():
    rows = []
    ids = [int(x) for x in TOKS[40_000_000:40_000_000 + N]]
    async with aiohttp.ClientSession() as s:
        for ep in (SRC, DST):
            await flush(s, ep)
        await asyncio.sleep(2)
        print('installing a source prefix...', flush=True)
        c = await install(s, SRC, ids)
        st = await post(s, SRC, '/agentshift/hybrid_state/inspect', {'token_ids': ids})
        print(f'  src matched={st.get("matched_tokens")} '
              f'full={st.get("full_prefix_available")}', flush=True)

        # ---- Block A: distinct fresh ports, idle engine ----
        print('\n[A] distinct fresh ports, engine idle', flush=True)
        for k in range(6):
            port = free_port_guess()
            name = f'A-{k}-{int(time.time() * 1000) % 10**7}'
            dt, ok, a, b = await init_pair(s, name, port)
            rows.append(dict(block='A', k=k, port=port, init_s=dt, ok=ok))
            print(f'  A{k}: port={port} init={dt:7.2f}s ok={ok} '
                  f'{(a.get("ERR") or a.get("detail") or "")[:70]}', flush=True)

        # ---- Block B: the SAME port repeated ----
        print('\n[B] same port repeated with distinct names', flush=True)
        port_b = free_port_guess()
        for k in range(3):
            name = f'B-{k}-{int(time.time() * 1000) % 10**7}'
            dt, ok, a, b = await init_pair(s, name, port_b)
            rows.append(dict(block='B', k=k, port=port_b, init_s=dt, ok=ok))
            print(f'  B{k}: port={port_b} init={dt:7.2f}s ok={ok} '
                  f'{(a.get("ERR") or a.get("detail") or "")[:70]}', flush=True)

        # ---- Block C: distinct fresh ports under co-tenant load ----
        print('\n[C] distinct fresh ports, engine busy', flush=True)
        for k in range(4):
            bg = await load_co_tenant(s, ids)
            await asyncio.sleep(0.4)
            port = free_port_guess()
            name = f'C-{k}-{int(time.time() * 1000) % 10**7}'
            dt, ok, a, b = await init_pair(s, name, port)
            rows.append(dict(block='C', k=k, port=port, init_s=dt, ok=ok))
            print(f'  C{k}: port={port} init={dt:7.2f}s ok={ok} '
                  f'{(a.get("ERR") or a.get("detail") or "")[:70]}', flush=True)
            await asyncio.gather(*bg, return_exceptions=True)

    path = ROOT / 'results/init-cost-decisive.json'
    path.write_text(json.dumps(rows, indent=2) + '\n')
    print('\n===== SUMMARY =====')
    for blk in ('A', 'B', 'C'):
        xs = [r['init_s'] for r in rows if r['block'] == blk]
        oks = sum(1 for r in rows if r['block'] == blk and r['ok'])
        if xs:
            print(f'  block {blk}: n={len(xs)} ok={oks} '
                  f'min={min(xs):.2f} med={np.median(xs):.2f} max={max(xs):.2f}')
    print(f'wrote {path}')


if __name__ == '__main__':
    asyncio.run(main())
