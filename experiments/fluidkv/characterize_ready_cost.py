#!/usr/bin/env python3
"""Characterize the runnable-state handoff cost: init_once vs per_transfer.

The paper's cost model is

    C_ready = init_once(pair) + k * per_transfer

The prior storyline recorded init_once = 16-31 s from "5 bare link setups on a
same-node pair". If that number is real, a pair must migrate ~1.8+ times before
mobility pays for itself, and the mechanism is only viable for long-lived
deployments. If it is not, the amortization argument changes completely.

This script measures BOTH terms directly, on a same-node pair and a cross-node
pair, and separates three quantities that are easy to conflate:

  init_fresh   a brand-new group (two-sided handshake)
  transfer     one broadcast round over an existing group
  cycle        flush + transfer, which is what a real migration costs

Design: install one 32K prefix on the source, then run

  A. fresh group per transfer, 3 times   -> pays init every time
  B. one group, 3 transfers               -> pays init once

The destination must be flushed between transfers because the engine's transfer
path requires an empty destination radix tree; that flush is timed separately
so it is never folded into the copy cost.
"""
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path('/home/zhujianian/AS-autoscaling')
TOKS = np.load(ROOT / 'experiments/weka_baselines/data/tokens.npy')
N = 32768

PAIRS = [
    # tag,        src,              dst,               port
    ('same-node', '12.11.3.27:31781', '12.11.3.27:31782', 52150),
    ('cross-node', '12.11.3.27:31781', '12.11.3.28:31751', 52160),
]


async def post(s, ep, path, payload=None, to=600):
    url = f'http://{ep}{path}'
    try:
        async with s.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=to)) as r:
            return await r.json()
    except Exception as exc:
        return {'ERR': repr(exc)[:200]}


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
                                        'ignore_eos': True}})
    return (r.get('meta_info') or {}).get('cached_tokens')


async def init_pair(s, src, dst, name, port):
    payload = lambda rank: {'group_rank': rank, 'group_name': name,
                            'master_address': src.split(':')[0], 'port': port,
                            'world_size': 2, 'backend': 'nccl'}
    a, b = await asyncio.gather(
        post(s, src, '/agentshift/hybrid_state/direct_group/init', payload(0), to=180),
        post(s, dst, '/agentshift/hybrid_state/direct_group/init', payload(1), to=180))
    return bool(a.get('success') and b.get('success'))


async def do_transfer(s, src, dst, name, port, ids):
    p = {'group_name': name, 'port': port, 'token_ids': ids}
    a, b = await asyncio.gather(
        post(s, src, '/agentshift/hybrid_state/direct_transfer', dict(p, role='source')),
        post(s, dst, '/agentshift/hybrid_state/direct_transfer', dict(p, role='destination')))
    return a, b


async def run_pair(s, tag, src, dst, port):
    ids = [int(x) for x in TOKS[40_000_000:40_000_000 + N]]
    print(f'\n===== {tag}: {src} -> {dst} (n={N}) =====', flush=True)
    for ep in (src, dst):
        await flush(s, ep)
    await asyncio.sleep(2)
    cached = await install(s, src, ids)
    st = await post(s, src, '/agentshift/hybrid_state/inspect', {'token_ids': ids})
    print(f'  src installed: cached={cached} matched={st.get("matched_tokens")} '
          f'full={st.get("full_prefix_available")}', flush=True)

    rows = []
    seq = 0

    # A. fresh group per transfer
    print('  [A] fresh group each time', flush=True)
    for k in range(3):
        seq += 1
        t0 = time.perf_counter()
        ok_init = await init_pair(s, src, dst, f'{tag}-fresh{seq}-{k}', port + 4 * k)
        t_init = time.perf_counter() - t0
        t1 = time.perf_counter()
        a, b = await do_transfer(s, src, dst, f'{tag}-fresh{seq}-{k}', port + 4 * k, ids)
        t_x = time.perf_counter() - t1
        t2 = time.perf_counter()
        await flush(s, dst)
        t_fl = time.perf_counter() - t2
        ok = a.get('state') == 'COMPLETE' and b.get('state') == 'COMPLETE'
        rows.append(dict(pair=tag, mode='fresh', k=k, init_s=t_init, xfer_s=t_x,
                         flush_s=t_fl, ok=ok, init_ok=ok_init))
        print(f'    fresh#{k}: init={t_init:.2f}s xfer={t_x:.2f}s flush={t_fl:.2f}s ok={ok}',
              flush=True)

    # B. one group, reused
    print('  [B] one group, reused', flush=True)
    name = f'{tag}-reused-{int(time.time()) % 100000}'
    t0 = time.perf_counter()
    ok_init = await init_pair(s, src, dst, name, port + 40)
    t_init = time.perf_counter() - t0
    print(f'    init once: {t_init:.2f}s ok={ok_init}', flush=True)
    for k in range(3):
        t1 = time.perf_counter()
        a, b = await do_transfer(s, src, dst, name, port + 40, ids)
        t_x = time.perf_counter() - t1
        t2 = time.perf_counter()
        await flush(s, dst)
        t_fl = time.perf_counter() - t2
        ok = a.get('state') == 'COMPLETE' and b.get('state') == 'COMPLETE'
        rows.append(dict(pair=tag, mode='reused', k=k, init_s=(t_init if k == 0 else 0.0),
                         xfer_s=t_x, flush_s=t_fl, ok=ok, init_ok=ok_init))
        print(f'    reused#{k}: xfer={t_x:.2f}s flush={t_fl:.2f}s ok={ok} '
              f'detail={(a.get("detail") or "")[:60]}', flush=True)
    return rows


async def main():
    out = []
    async with aiohttp.ClientSession() as s:
        for tag, src, dst, port in PAIRS:
            try:
                out += await run_pair(s, tag, src, dst, port)
            except Exception as exc:
                print(f'  PAIR FAILED {tag}: {exc!r}', flush=True)
    path = ROOT / 'results/cready-characterization.json'
    path.write_text(json.dumps(out, indent=2) + '\n')

    print('\n===== SUMMARY =====')
    for mode in ('fresh', 'reused'):
        xs = [r['xfer_s'] for r in out if r['mode'] == mode and r['ok']]
        inits = [r['init_s'] for r in out if r['mode'] == mode and r['init_s']]
        if xs:
            print(f'{mode:8s} n={len(xs)} init={sum(inits):.2f}s total '
                  f'xfer med={np.median(xs):.2f}s range={min(xs):.2f}-{max(xs):.2f}')
    print(f'\nwrote {path}')


if __name__ == '__main__':
    asyncio.run(main())
