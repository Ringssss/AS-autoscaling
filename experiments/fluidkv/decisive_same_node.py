#!/usr/bin/env python3
"""DECISIVE: same-node, long prefix, prewarmed group, cold destination.

Every earlier attempt missed one of the four conditions the arithmetic needs:

  p03    same-node (init cheap) but destination NOT empty -> 0 migrations
  p04b   destination empty but CROSS-node (init 21-51 s) -> copy saved 19-28 s,
         init cost more, and my prewarm missed the cold->cold pairs entirely

This run puts all four in one place:

  same-node    init ~0.06-0.16 s  (measured)
  prewarmed    every directed pair established while the fleet is idle, so no
               migration ever pays a handshake and none can starve
  cold dst     the destination has never served the prefix, so the engine's
               empty-tree requirement is satisfied
  long prefix  130,048 tokens, where recompute is 19-28 s

Paired design: per repetition, the SAME prefix is installed on the same source
and served on the same destination; the only difference between arms is whether
the runnable state is copied first.

Decision rule, fixed before running:
  mobility wins  -> a systems paper exists (scale-out / new-worker regime)
  no win         -> mechanism-only paper, or stop
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import importlib.util
_spec = importlib.util.spec_from_file_location(
    'p03', Path(__file__).parent / 'p03_real_copy_e2e.py')
p03 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p03)

ROOT = Path('/home/zhujianian/AS-autoscaling')
TOKS = np.load(ROOT / 'experiments/weka_baselines/data/tokens.npy')
N = 130048
SLO = 10.0

# Same host: 31781 is the source that owns the prefix, 31782 a cold spare.
SRC = '12.11.3.27:31781'
DST = '12.11.3.27:31782'
SPARE = '12.11.3.27:31783'


def region(idx):
    base = idx * 262144
    return [int(x) for x in TOKS[base:base + N]]


async def flush(s, ep):
    try:
        async with s.post(f'http://{ep}/flush_cache?timeout=60') as r:
            await r.text()
    except Exception:
        pass


async def install(s, ep, ids):
    """A prefix is only resident after a COMPLETED request (max_new_tokens>=16)."""
    return await p03.serve(s, 'http://' + ep, ids)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--out', default=str(ROOT / 'results/decisive-same-node.json'))
    a = ap.parse_args()

    urls = ['http://' + e for e in (SRC, DST, SPARE)]
    rows = []
    async with aiohttp.ClientSession() as s:
        # ---- prewarm every directed pair while idle ----
        gc = p03.GroupCache()
        print('prewarming directed pairs (idle fleet)...', flush=True)
        for src_i in range(3):
            for dst_i in range(3):
                if src_i == dst_i:
                    continue
                t0 = time.perf_counter()
                got = await gc.ensure(s, urls, src_i, dst_i)
                print(f'  {src_i}->{dst_i}: {time.perf_counter() - t0:6.2f}s '
                      f'{"ok" if got else "FAILED"}', flush=True)
        ready = len(gc.groups)
        print(f'{ready}/6 pairs ready\n', flush=True)

        for rep in range(a.reps):
            ids = region(rep)
            print(f'===== rep {rep}  n={N} =====', flush=True)

            # ---- arm R: recompute on the cold destination ----
            for ep in (SRC, DST):
                await flush(s, ep)
            await asyncio.sleep(2)
            await install(s, SRC, ids)
            st = await p03.inspect(s, urls[0], ids)
            print(f'  installed on src: matched={st.get("matched_tokens")}', flush=True)
            t0 = time.perf_counter()
            tt_r, _, meta = await p03.serve(s, urls[1], ids)
            rows.append(dict(rep=rep, arm='recompute', ttft=tt_r,
                             cached=meta.get('cached_tokens') or 0,
                             wall=time.perf_counter() - t0))
            print(f'  [recompute] ttft={tt_r:7.2f}s cached={meta.get("cached_tokens")}',
                  flush=True)

            # ---- arm M: copy the runnable prefix to the same destination ----
            for ep in (SRC, DST):
                await flush(s, ep)
            await asyncio.sleep(2)
            await install(s, SRC, ids)
            st = await p03.inspect(s, urls[0], ids)
            n_reuse = st.get('matched_tokens') or 0
            dst_st = await p03.inspect(s, urls[1], ids)
            dst_empty = (dst_st.get('evictable_token_slots') or 0) == 0
            t0 = time.perf_counter()
            res = await p03.copy_state(s, gc, urls, 0, 1, ids[:n_reuse])
            copy_s = time.perf_counter() - t0
            tt_m, _, meta_m = (await p03.serve(s, urls[1], ids)
                               if res.get('ok') else (float('nan'), 0, {}))
            rows.append(dict(rep=rep, arm='mobility', ttft=tt_m,
                             cached=meta_m.get('cached_tokens') or 0,
                             copy_s=copy_s, copy_ok=bool(res.get('ok')),
                             dst_empty=dst_empty, n_reuse=n_reuse,
                             copy_detail=str(res.get('detail'))[:120]))
            print(f'  [mobility ] ttft={tt_m:7.2f}s cached={meta_m.get("cached_tokens")} '
                  f'copy={copy_s:5.2f}s ok={res.get("ok")} dst_empty={dst_empty}',
                  flush=True)

    Path(a.out).write_text(json.dumps(rows, indent=2) + '\n')
    print(f'\nwrote {a.out}\n')

    R = [r for r in rows if r['arm'] == 'recompute']
    M = [r for r in rows if r['arm'] == 'mobility']
    ok = [m for m in M if m.get('copy_ok')]
    print('===== DECISION =====')
    if R:
        print(f'recompute  n={len(R)} median ttft={np.median([r["ttft"] for r in R]):7.2f}s')
    if M:
        print(f'mobility   n={len(M)} median ttft={np.median([r["ttft"] for r in M]):7.2f}s '
              f'(copies ok {len(ok)}/{len(M)})')
    if R and ok:
        r_med = float(np.median([r['ttft'] for r in R]))
        m_med = float(np.median([r['ttft'] for r in ok]))
        print(f'speedup    {r_med / m_med:.2f}x   '
              f'(init already paid: {ready}/6 pairs prewarmed)')
        print('VERDICT: ' + ('mobility WINS -> systems paper exists'
                             if r_med / m_med > 1.5 else
                             'NO WIN -> mechanism-only or stop'))


if __name__ == '__main__':
    asyncio.run(main())
