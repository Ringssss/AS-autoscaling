#!/usr/bin/env python3
"""P0-4b: cold-engine mobility with groups PRE-ESTABLISHED.

Why a separate script: every attempt to let the end-to-end harness create
groups on demand has wedged engines. The engine's init_group needs BOTH sides
inside the same 90 s window, and its hybrid control channel is a single queue,
so concurrent inits starve each other and the losers leave a peer blocked on a
store nobody will write.

The fix is to establish the pairs SEQUENTIALLY while the fleet is idle, and only
then run traffic. Once a group exists the per-transfer path is robust -- twelve
consecutive transfers over one group completed with the engine healthy.

Scenario, which is the point of the experiment:

  hot engines      carry every stream; their prefixes accumulate
  cold engines     receive nothing at all, so their trees stay empty and a
                   migration has a legal destination

Without a cold engine there is no legal destination in steady state, which is
exactly why the earlier balanced-traffic run could not migrate once.
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


async def prewarm(session, gc, urls, pairs):
    """Establish every needed group while nothing else is running."""
    ok = []
    for src, dst in pairs:
        t0 = time.perf_counter()
        got = await gc.ensure(session, urls, src, dst)
        dt = time.perf_counter() - t0
        print(f'  prewarm {src}->{dst}: {dt:6.2f}s '
              f'{"ok" if got else "FAILED"}', flush=True)
        if got:
            ok.append((src, dst))
    return ok


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
    saved = sum(m['tokens'] * 2.17e-4 - (m.get('seconds') or 0) for m in used)
    return dict(
        wall=arm['wall'], requests=len(arm['records']),
        ttft_p50=float(np.percentile(good, 50)) if len(good) else None,
        ttft_p90=float(np.percentile(good, 90)) if len(good) else None,
        slo=float(np.mean(good <= SLO)) if len(good) else None,
        goodput=float(np.sum(good <= SLO) / arm['wall']) if len(good) else None,
        mig_attempted=len(mig), mig_ok=len(ok_mig), mig_used=len(used),
        copy_seconds=sum(m.get('seconds') or 0 for m in ok_mig),
        realized_saving_s=saved, blocked=arm.get('blocked') or {})


async def run_arm(session, gc, urls, streams, homes, toks_all, n_hot, mobility):
    """Cold engines are the destination; the copy is the only toggle between arms."""
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
                    st = await p03.inspect(session, urls[src], ids)
                    n_reuse = st.get('matched_tokens') or 0
                    dst = await p03.inspect(session, urls[target], ids)
                    dst_empty = (dst.get('evictable_token_slots') or 0) == 0
                    if n_reuse < p03.MOVE_MIN_TOKENS:
                        blocked['src_has_no_runnable_prefix'] += 1
                    elif not dst_empty:
                        blocked['dst_not_empty'] += 1
                    else:
                        res = await p03.copy_state(session, gc, urls, src,
                                                   target, ids[:n_reuse])
                        migrations.append(dict(key=key, turn=r['turn'], src=src,
                                               dst=target, tokens=n_reuse, **res))
                        moved = bool(res.get('ok'))
            try:
                ttft, tot, meta = await asyncio.wait_for(
                    p03.serve(session, urls[target], ids),
                    timeout=p03.SERVE_BUDGET)
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
                hsh = hashlib.sha256(rs[0]['key'].encode()).hexdigest()
                homes[rs[0]['key']] = 1 + (int(hsh, 16) % max(1, n_hot - 1))
    return order, homes


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--streams', type=int, default=8)
    ap.add_argument('--hot-frac', type=float, default=0.5)
    ap.add_argument('--n-hot', type=int, default=2)
    ap.add_argument('--reps', type=int, default=2)
    ap.add_argument('--out', default=str(ROOT / 'results/p04b-cold-engine.json'))
    a = ap.parse_args()

    rows, toks = p03.load_workload()
    streams, homes = schedule(rows, a.streams, a.n_hot, a.hot_frac)
    urls = ['http://' + e for e in p03.ENGINES]
    n_req = sum(len(s) for s in streams)
    print(f'engines {len(urls)}  hot 0..{a.n_hot - 1}  cold {a.n_hot}..{len(urls) - 1}',
          flush=True)
    print(f'streams {len(streams)}  requests {n_req}  hot_frac {a.hot_frac}',
          flush=True)

    out = {'meta': dict(engines=p03.ENGINES, n_hot=a.n_hot,
                        streams=len(streams), requests=n_req,
                        hot_frac=a.hot_frac, reps=a.reps), 'arms': {}}
    async with aiohttp.ClientSession() as session:
        for rep in range(a.reps):
            for mobility in (True, False):
                tag = ('mobility' if mobility else 'reroute') + f'#r{rep}'
                gc = p03.GroupCache()
                print(f'--- arm {tag} ---', flush=True)
                if mobility:
                    # Each hot engine -> each cold engine, established while idle.
                    pairs = [(h, c) for h in range(a.n_hot)
                             for c in range(a.n_hot, len(urls))]
                    good = await prewarm(session, gc, urls, pairs)
                    print(f'  {len(good)}/{len(pairs)} pairs ready', flush=True)
                await p03.flush(session, urls)
                arm = await run_arm(session, gc, urls, streams, homes, toks,
                                    a.n_hot, mobility)
                s = summarize(arm)
                out['arms'][tag] = dict(summary=s, migrations=arm['migrations'])
                print(json.dumps(s, indent=2), flush=True)

    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(f'\nwrote {a.out}\n')
    print(f'{"arm":14s}{"p50":>8}{"p90":>9}{"SLO":>8}{"goodput":>9}'
          f'{"att":>5}{"ok":>4}{"used":>5}{"saved_s":>10}')
    for tag, v in out['arms'].items():
        s = v['summary']
        print(f'{tag:14s}{s["ttft_p50"]:8.2f}{s["ttft_p90"]:9.2f}'
              f'{s["slo"]:8.3f}{s["goodput"]:9.3f}{s["mig_attempted"]:5d}'
              f'{s["mig_ok"]:4d}{s["mig_used"]:5d}{s["realized_saving_s"]:10.1f}')


if __name__ == '__main__':
    asyncio.run(main())
