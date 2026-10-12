"""Affinity-vs-balance Pareto frontier under the REAL session-length skew.

The 128x3 materialized workload has uniform session lengths, which erases the
rate skew that forces routing to exist in the first place. This builder
re-materializes streams from the raw trace preserving the measured length
distribution, then measures the frontier every routing-only system is trapped
on: spending prefix affinity to buy load balance.

A mover (FluidKV-class) escapes the frontier because it can relocate state to
follow the load instead of choosing between the two.
"""
import json, sys, math, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/zhujianian/AS-autoscaling')
from experiments.weka_baselines.oracle.simulate import Engine, BLOCK, ROOT

TRACE = Path('/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl')


from experiments.weka_gravity.audit import flatten


def load_real_streams(limit_traces=393, min_len=2):
    """Real per-stream hash_id sequences, models separated, main agent only.

    Subagent groups are skipped: their nested entries carry no independent
    session identity to chain from. Reuse is measured only where the trace
    itself establishes a sequential chain.
    """
    streams = []
    with TRACE.open() as f:
        for li, line in enumerate(f, 1):
            if li > limit_traces: break
            trace = json.loads(line)
            rows, _ = flatten(trace)
            per = defaultdict(list)
            for r in rows:
                if r.get('subagent'):
                    continue
                if int(r['in']) + int(r['out']) > 131072 or int(r['out']) <= 0:
                    continue
                per[(r['stream'], r['model'])].append(r)
            for (stream, model), reqs in per.items():
                reqs.sort(key=lambda r: float(r['t']))
                if len(reqs) >= min_len:
                    for r in reqs:
                        r['in'] = int(r['in'])
                    streams.append(dict(trace=trace['id'], stream=stream,
                                        model=model, reqs=reqs))
    return streams


def simulate(streams, engines_n, cap_tokens, policy, seed=42, spread_deg=0.0):
    """policy: 'affinity' (sticky by stream) or 'balance' (least in-flight).

    spread_deg in [0,1] blends: 0 = pure affinity, 1 = pure balance.
    A stream keeps its home unless that engine's queued load exceeds
    home_load * (1+spread_deg*k), modelling the usual load-aware escape hatch.
    """
    engines = [Engine(cap_tokens) for _ in range(engines_n)]
    rng = random.Random(seed)
    home = {}
    inflight = [0]*engines_n
    queued = [0.0]*engines_n
    hit = 0; inp = 0; reqs = 0
    full = []
    for s in streams:
        for r in s['reqs']:
            key = s['trace'] + '/' + s['stream'] + '/' + s['model']
            if key not in home:
                home[key] = int(hashlib.sha256(key.encode()).hexdigest(),16) % engines_n
            h = home[key]
            if policy == 'affinity':
                t = h
            elif policy == 'balance':
                t = min(range(engines_n), key=lambda i: (inflight[i], i))
            else:
                # load-aware: stay home unless home is overloaded
                ref = max(1e-9, queued[h])
                thr = ref * (1 + spread_deg * 8)
                cands = [i for i in range(engines_n) if queued[i] < thr]
                t = h if (h in cands or not cands) else min(cands, key=lambda i: queued[i])
            e = engines[t]
            pf = e.prefix_len(r['hash_ids'])
            e.admit(r['hash_ids'])
            hit += pf; inp += r['in']; reqs += 1
            inflight[t] += 1
            queued[t] += r['in']
            full.append((key, t))
            inflight[t] -= 1
            # decay queue so it reflects recent pressure
            for i in range(engines_n): queued[i] *= 0.999
    served = Counter(t for _, t in full)
    load = np.array([served.get(i,0) for i in range(engines_n)], dtype=float)
    # token-weighted imbalance is the one that drives tail latency
    tokload = np.zeros(engines_n)
    for s in streams:
        for r in s['reqs']:
            pass
    return dict(policy=policy, spread=spread_deg, engines=engines_n,
                hit_ratio=hit/inp, requests=reqs, input_tokens=inp,
                imbalance=float(load.max()/max(1e-9, load.mean())),
                per_engine=dict(served))


import hashlib

def main():
    streams = load_real_streams()
    lens = [len(s['reqs']) for s in streams]
    total_reqs = sum(lens)
    print(f"loaded {len(streams)} real main-agent streams, {total_reqs:,} requests")
    print(f"session length p50={np.percentile(lens,50):.0f} p90={np.percentile(lens,90):.0f} "
          f"max={max(lens)}")
    # token-weighted load skew: what a router must actually balance
    tok = np.array([sum(r['in'] for r in s['reqs']) for s in streams], dtype=float)
    print(f"stream token volume: top1%={np.sort(tok)[-max(1,len(tok)//100):].sum()/tok.sum():.1%} "
          f"top10%={np.sort(tok)[-max(1,len(tok)//10):].sum()/tok.sum():.1%}")

    out = []
    print(f"\n{'policy':<24}{'hit%':>8}{'imbalance':>11}   engines=3 cap=1M")
    for deg in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0):
        r = simulate(streams, 3, 1_048_576, 'loadaware', spread_deg=deg)
        out.append(r)
        print(f"loadaware(deg={deg:<5})        {r['hit_ratio']*100:>6.1f}%{r['imbalance']:>11.2f}")
    for pol in ('affinity','balance'):
        r = simulate(streams, 3, 1_048_576, pol)
        out.append(r)
        print(f"{pol:<24}{r['hit_ratio']*100:>6.1f}%{r['imbalance']:>11.2f}")

    (ROOT/'oracle/results/pareto.json').write_text(json.dumps(out, indent=2)+'\n')
    print(f"\nwrote {ROOT/'oracle/results/pareto.json'}")


if __name__ == '__main__':
    main()
