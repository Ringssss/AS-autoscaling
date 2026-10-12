#!/usr/bin/env python3
"""E2E: cold-engine scale-out on a SAME-NODE fleet, all pairs prewarmed.

p04b proved cold destinations make migration legal, but ran cross-node where a
handshake costs 21-51 s and my prewarm missed the cold->cold pairs, so 8 of 12
attempts burned 90 s each. This run keeps the cold-destination regime and fixes
both defects:

  same-node fleet     handshake ~0.1-11 s instead of 21-51 s
  ALL pairs prewarmed hot->cold AND cold->cold, since a stream that migrated
                      once now lives on a cold engine and its next hop starts
                      from there

Reporting follows the p03 rule: the eligible subset and all traffic are
reported separately, and the copy is the only difference between arms.
"""
import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
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
SLO = 10.0
FLEET = ['12.11.3.27:31781', '12.11.3.27:31782', '12.11.3.27:31783']


def summarize(arm, n_engines):
    tt = np.array([r['ttft'] for r in arm['records']], float)
    good = tt[np.isfinite(tt)]
    mig = arm['migrations']
    ok_mig = [m for m in mig if m.get('ok')]
    used = [m for m in ok_mig
            if any(r['key'] == m['key'] and r['turn'] == m['turn']
                   and r['engine'] == m['dst']
                   and r['cached_tokens'] >= m['tokens']
                   for r in arm['records'])]
    saved = sum(m['tokens'] * 2.17e-4 - (m.get('seconds') or 0) for m in used)
    moved = [r for r in arm['records'] if r.get('moved')]
    return dict(
        wall=arm['wall'], requests=len(arm['records']),
        errors=sum(1 for r in arm['records'] if r['error']),
        ttft_p50=float(np.percentile(good, 50)) if len(good) else None,
        ttft_p90=float(np.percentile(good, 90)) if len(good) else None,
        slo=float(np.mean(good <= SLO)) if len(good) else None,
        goodput=float(np.sum(good <= SLO) / arm['wall']) if len(good) else None,
        mig_attempted=len(mig), mig_ok=len(ok_mig), mig_used=len(used),
        copy_seconds=sum(m.get('seconds') or 0 for m in ok_mig),
        copy_bytes=sum(m.get('bytes') or 0 for m in ok_mig),
        realized_saving_s=saved,
        # Eligible subset = requests the mechanism actually acted on.
        n_moved=len(moved),
        moved_p50=float(np.median([r['ttft'] for r in moved])) if moved else None,
        blocked=arm.get('blocked') or {})


async def run_arm(s, gc, urls, streams, homes, toks_all, n_hot, mobility):
    recs, migrations = [], []
    blocked = defaultdict(int)
    busy = [0.0] * len(urls)
    resident = {}
    cold = list(range(n_hot, len(urls)))
    lock = asyncio.Lock()

    async def run_stream(rs):
        key = rs[0]['key']
        h = homes[key]
        for r in rs:
            ids = toks_all[r['token_offset']: r['token_offset'] + r['input_tokens']]
            async with lock:
                target = min(cold, key=lambda i: (busy[i], i))
                src = resident.get(key, h)
                moved = False
                if mobility and target != src:
                    st = await p03.inspect(s, urls[src], ids)
                    n_reuse = st.get('matched_tokens') or 0
                    dst = await p03.inspect(s, urls[target], ids)
                    dst_empty = (dst.get('evictable_token_slots') or 0) == 0
                    if n_reuse < p03.MOVE_MIN_TOKENS:
                        blocked['src_has_no_runnable_prefix'] += 1
                    elif not dst_empty:
                        blocked['dst_not_empty'] += 1
                    else:
                        res = await p03.copy_state(s, gc, urls, src, target,
                                                   ids[:n_reuse])
                        migrations.append(dict(key=key, turn=r['turn'], src=src,
                                               dst=target, tokens=n_reuse, **res))
                        moved = bool(res.get('ok'))
            try:
                ttft, tot, meta = await asyncio.wait_for(
                    p03.serve(s, urls[target], ids), timeout=p03.SERVE_BUDGET)
                err = None
            except asyncio.TimeoutError:
                ttft, tot, meta = float('nan'), float('nan'), {}
                err = 'serve budget'
            except Exception as exc:
                ttft, tot, meta = float('nan'), float('nan'), {}
                err = repr(exc)[:160]
            async with lock:
                busy[target] += tot if tot == tot else 0
                recs.append(dict(key=key, turn=r['turn'], engine=target,
                                 ttft=ttft, moved=moved,
                                 cached_tokens=meta.get('cached_tokens') or 0,
                                 input_tokens=r['input_tokens'], error=err))
                if err is None and not (ttft != ttft):
                    resident[key] = target

    t0 = time.perf_counter()
    await asyncio.gather(*(run_stream(rs) for rs in streams))
    return dict(wall=time.perf_counter() - t0, records=recs,
                migrations=migrations, blocked=dict(blocked),
                init_seconds=dict(gc.init_seconds), init_error=gc.last_error)


def schedule(rows, streams, n_hot, hot_frac):
    import hashlib
    by = defaultdict(list)
    for r in rows:
        by[r['key']].append(r)
    for k in by:
        by[k].sort(key=lambda r: r['turn'])
    keys = sorted(by, key=lambda k: hashlib.sha256(k.encode()).hexdigest())[:streams]
    order = [by[k] for k in keys]
    homes = {}
    ranked = sorted(order, key=lambda rs: -max(r['input_tokens'] for r in rs))
    n_pin = max(1, int(len(ranked) * hot_frac)) if hot_frac > 0 else 0
    for i, rs in enumerate(ranked):
        if i < n_pin:
            homes[rs[0]['key']] = 0
        else:
            hsh = hashlib.sha256(rs[0]['key'].encode()).hexdigest()
            homes[rs[0]['key']] = 1 + (int(hsh, 16) % max(1, n_hot - 1))
    return order, homes


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--streams', type=int, default=10)
    ap.add_argument('--hot-frac', type=float, default=0.6)
    ap.add_argument('--n-hot', type=int, default=1)
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--out', default=str(ROOT / 'results/p04c-same-node-cold.json'))
    a = ap.parse_args()

    rows, toks = p03.load_workload()
    streams, homes = schedule(rows, a.streams, a.n_hot, a.hot_frac)
    urls = ['http://' + e for e in FLEET]
    n_req = sum(len(s) for s in streams)
    print(f'fleet {FLEET}', flush=True)
    print(f'hot 0..{a.n_hot - 1}  cold {a.n_hot}..{len(urls) - 1}  '
          f'streams {len(streams)}  requests {n_req}', flush=True)

    out = {'meta': dict(fleet=FLEET, n_hot=a.n_hot, streams=len(streams),
                        requests=n_req, hot_frac=a.hot_frac), 'arms': {}}
    async with aiohttp.ClientSession() as s:
        for rep in range(a.reps):
            for mobility in (True, False):
                tag = ('mobility' if mobility else 'reroute') + f'#r{rep}'
                gc = p03.GroupCache()
                print(f'--- arm {tag} ---', flush=True)
                if mobility:
                    pairs = [(i, j) for i in range(len(urls))
                             for j in range(len(urls)) if i != j]
                    good = 0
                    for i, j in pairs:
                        t0 = time.perf_counter()
                        if await gc.ensure(s, urls, i, j):
                            good += 1
                        print(f'  prewarm {i}->{j}: {time.perf_counter() - t0:6.2f}s',
                              flush=True)
                    print(f'  {good}/{len(pairs)} pairs ready', flush=True)
                await p03.flush(s, urls)
                arm = await run_arm(s, gc, urls, streams, homes, toks,
                                    a.n_hot, mobility)
                sm = summarize(arm, len(urls))
                out['arms'][tag] = dict(summary=sm, migrations=arm['migrations'])
                print(json.dumps(sm, indent=2), flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(f'\nwrote {a.out}\n')
    print(f'{"arm":14s}{"p50":>8}{"p90":>9}{"SLO":>8}{"goodput":>9}'
          f'{"att":>5}{"ok":>4}{"used":>5}{"saved_s":>9}{"wall":>8}')
    for tag, v in out['arms'].items():
        x = v['summary']
        print(f'{tag:14s}{x["ttft_p50"]:8.2f}{x["ttft_p90"]:9.2f}{x["slo"]:8.3f}'
              f'{x["goodput"]:9.3f}{x["mig_attempted"]:5d}{x["mig_ok"]:4d}'
              f'{x["mig_used"]:5d}{x["realized_saving_s"]:9.1f}{x["wall"]:8.0f}')


if __name__ == '__main__':
    asyncio.run(main())
