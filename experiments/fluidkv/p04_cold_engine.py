#!/usr/bin/env python3
"""P0-4: does mobility pay off when a COLD engine exists?

The steady-state end-to-end run (p03) could not fire a single migration: the
engine's transfer path requires an empty destination radix tree, and in steady
state every engine already holds prefixes. That is a real constraint, but it
also means p03 measured the wrong regime. The regime the mechanism is FOR is
the one where capacity is idle somewhere: a freshly joined worker, a scaled-out
replica, or simply a replica that the load-gravity pattern has left cold.

This script constructs that regime deliberately:

  hot set   engines 0..H-1   carry every stream; their prefixes accumulate
  cold set  engines H..N-1   receive no traffic at all, so their trees stay empty

The router's balanced choice sends a request to a cold engine. Mobility then
has a legal destination, and the comparison is:

  baseline   balanced routing, request lands cold, pays a full recompute
  treatment  same routing decision, but the runnable prefix is copied over first

Both arms see identical arrivals, engines and routing decisions; only the copy
is toggled. Metrics are TTFT percentiles, SLO attainment, goodput, and the
realized copy-vs-recompute saving.
"""
import argparse
import asyncio
import hashlib
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


def schedule_cold(rows, streams, n_hot, hot_frac):
    """All streams land on the hot set; an extra share is pinned to one engine."""
    by = defaultdict(list)
    for r in rows:
        by[r['key']].append(r)
    for k in by:
        by[k].sort(key=lambda r: r['turn'])
    keys = sorted(by, key=lambda k: hashlib.sha256(k.encode()).hexdigest())[:streams]
    order = [by[k] for k in keys]
    homes = {}
    if hot_frac <= 0:
        for rs in order:
            homes[rs[0]['key']] = int(
                hashlib.sha256(rs[0]['key'].encode()).hexdigest(), 16) % n_hot
    else:
        ranked = sorted(order, key=lambda rs: -max(r['input_tokens'] for r in rs))
        n_pin = max(1, int(len(ranked) * hot_frac))
        for i, rs in enumerate(ranked):
            if i < n_pin:
                homes[rs[0]['key']] = 0
            else:
                h = hashlib.sha256(rs[0]['key'].encode()).hexdigest()
                homes[rs[0]['key']] = 1 + (int(h, 16) % max(1, n_hot - 1))
    return order, homes


async def run_arm(session, gc, urls, streams, homes, toks_all, fleet,
                  policy, mobility, n_hot):
    """Same shape as p03.run_arm, but the destination set is restricted to the
    cold engines so a legal (empty) destination actually exists."""
    await p03.flush(session, urls)
    busy = [0.0] * len(urls)
    recs = []
    migrations = []
    blocked = defaultdict(int)
    lock = asyncio.Lock()
    resident = {}
    cold = list(range(n_hot, len(urls)))

    async def run_stream(rs):
        key = rs[0]['key']
        h = homes[key]
        for r in rs:
            ids = toks_all[r['token_offset']: r['token_offset'] + r['input_tokens']]
            async with lock:
                if policy == 'cold_balance':
                    target = min(cold, key=lambda i: (busy[i], i))
                else:
                    target = h
                moved = False
                src = resident.get(key, h)
                if mobility and target != src:
                    st = await p03.inspect(session, urls[src], ids)
                    n_reuse = st.get('matched_tokens') or 0
                    dst_st = await p03.inspect(session, urls[target], ids)
                    dst_empty = (dst_st.get('evictable_token_slots') or 0) == 0
                    if n_reuse < p03.MOVE_MIN_TOKENS:
                        blocked['src_has_no_runnable_prefix'] += 1
                    elif not dst_empty:
                        blocked['dst_not_empty'] += 1
                    else:
                        res = await p03.copy_state(session, gc, urls, src,
                                                   target, ids[:n_reuse])
                        migrations.append(dict(key=key, turn=r['turn'],
                                               src=src, dst=target,
                                               tokens=n_reuse, **res))
                        moved = bool(res.get('ok'))
            try:
                ttft, tot, meta = await asyncio.wait_for(
                    p03.serve(session, urls[target], ids),
                    timeout=p03.SERVE_BUDGET)
                err = None
            except asyncio.TimeoutError:
                ttft, tot, meta = float('nan'), float('nan'), {}
                err = 'serve budget exceeded'
            except Exception as exc:
                ttft, tot, meta = float('nan'), float('nan'), {}
                err = repr(exc)[:200]
            async with lock:
                busy[target] += tot if tot == tot else 0
                recs.append(dict(key=key, turn=r['turn'], engine=target,
                                 ttft=ttft, total=tot, moved=moved,
                                 cached_tokens=meta.get('cached_tokens') or 0,
                                 input_tokens=r['input_tokens'], error=err))
                if err is None and not (ttft != ttft):
                    resident[key] = target

    t0 = time.perf_counter()
    await asyncio.gather(*(run_stream(rs) for rs in streams))
    wall = time.perf_counter() - t0
    return dict(wall=wall, records=recs, migrations=migrations,
                init_seconds=dict(gc.init_seconds), blocked=dict(blocked),
                init_error=gc.last_error)


def summarize(arm):
    tt = np.array([r['ttft'] for r in arm['records']], float)
    good = tt[np.isfinite(tt)]
    mig = arm['migrations']
    ok_mig = [m for m in mig if m.get('ok')]
    used = [m for m in ok_mig
            if any(r['key'] == m['key'] and r['turn'] == m['turn']
                   and r['engine'] == m['dst']
                   and r['cached_tokens'] >= m['tokens']
                   for r in arm['records'])]
    # A migrated request that reused the copy skipped R(n) and paid the copy.
    saved = sum(m['tokens'] * 2.17e-4 - (m.get('seconds') or 0) for m in used)
    return dict(
        wall=arm['wall'], requests=len(arm['records']),
        errors=sum(1 for r in arm['records'] if r['error']),
        ttft_p50=float(np.percentile(good, 50)) if len(good) else None,
        ttft_p90=float(np.percentile(good, 90)) if len(good) else None,
        ttft_p99=float(np.percentile(good, 99)) if len(good) else None,
        slo=float(np.mean(good <= SLO)) if len(good) else None,
        goodput=float(np.sum(good <= SLO) / arm['wall']) if len(good) else None,
        mig_attempted=len(mig), mig_ok=len(ok_mig), mig_used=len(used),
        copy_seconds=sum(m.get('seconds') or 0 for m in ok_mig),
        copy_bytes=sum(m.get('bytes') or 0 for m in ok_mig),
        realized_saving_s=saved,
        blocked=arm.get('blocked') or {},
        init_error=(arm.get('init_error') or '')[:200],
    )


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--streams', type=int, default=8)
    ap.add_argument('--hot-frac', type=float, default=0.5)
    ap.add_argument('--n-hot', type=int, default=4)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--out', default=str(ROOT / 'results/p04-cold-engine.json'))
    a = ap.parse_args()

    rows, toks = p03.load_workload()
    streams, homes = schedule_cold(rows, a.streams, a.n_hot, a.hot_frac)
    urls = ['http://' + e for e in p03.ENGINES]
    n_req = sum(len(s) for s in streams)
    print(f'engines {len(urls)} (hot 0..{a.n_hot - 1}, cold {a.n_hot}..{len(urls) - 1})',
          flush=True)
    print(f'streams {len(streams)}  requests {n_req}  hot_frac {a.hot_frac}',
          flush=True)

    out = {'meta': dict(engines=p03.ENGINES, n_hot=a.n_hot, streams=len(streams),
                        requests=n_req, hot_frac=a.hot_frac, reps=a.reps,
                        slo_seconds=SLO, move_min_tokens=p03.MOVE_MIN_TOKENS),
           'arms': {}}
    async with aiohttp.ClientSession() as session:
        for rep in range(a.reps):
            for mobility in (True, False):
                tag = ('mobility' if mobility else 'reroute') + f'#r{rep}'
                gc = p03.GroupCache()
                print(f'--- arm {tag} ---', flush=True)
                arm = await run_arm(session, gc, urls, streams, homes, toks,
                                    p03.ENGINES, 'cold_balance', mobility,
                                    a.n_hot)
                s = summarize(arm)
                out['arms'][tag] = dict(summary=s, migrations=arm['migrations'])
                print(json.dumps(s, indent=2), flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(f'\nwrote {a.out}\n', flush=True)
    print(f'{"arm":16s}{"p50":>8}{"p90":>9}{"SLO":>8}{"goodput":>9}'
          f'{"att":>5}{"ok":>4}{"used":>5}{"saved_s":>9}')
    for tag, v in out['arms'].items():
        s = v['summary']
        print(f'{tag:16s}{s["ttft_p50"]:8.2f}{s["ttft_p90"]:9.2f}'
              f'{s["slo"]:8.3f}{s["goodput"]:9.3f}{s["mig_attempted"]:5d}'
              f'{s["mig_ok"]:4d}{s["mig_used"]:5d}{s["realized_saving_s"]:9.1f}')


if __name__ == '__main__':
    asyncio.run(main())
