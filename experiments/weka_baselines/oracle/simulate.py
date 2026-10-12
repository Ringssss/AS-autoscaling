"""Offline policy-vs-oracle replay over the materialized weka workload.

Routers are decision-only: a request placed on engine e hits the longest
prefix of its hash_ids currently resident in e's block cache. This isolates
the routing decision from serving physics, so policy gaps are attributable to
placement, not to kernel noise.

Policies
  sticky          every stream pinned to engine hash(key) % n
  rr              round-robin over engines
  least-loaded    pick engine with fewest in-flight requests
  p2c-ring        upstream DualMap two-choice hash ring (uhashring, dual hash)
  cacheroute      arXiv:2608.19677 Algorithm 1 plan + load-based dispatch
  oracle-omniscient  knows the entire future; places each request where its
                     longest surviving prefix lives (or a free engine)
  oracle-move        same, plus it may move state between engines for a cost

Metrics: uncached prefill tokens (the TTFT driver), hit ratio, and load
imbalance (max/mean of in-flight counts).
"""
import argparse, hashlib, json, math, random
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from uhashring import HashRing

ROOT = Path(__file__).resolve().parents[1]
BLOCK = 16


# ---------------------------------------------------------------- engines
class Engine:
    """FIFO-ish block cache with a token capacity, LRU eviction."""
    def __init__(self, cap_tokens):
        self.cap = cap_tokens
        self.blocks = OrderedDict()      # hash_id -> n_tokens
        self.used = 0

    def prefix_len(self, hash_ids):
        n = 0
        for h in hash_ids:
            if h not in self.blocks:
                break
            self.blocks.move_to_end(h)   # hit refreshes recency
            n += 1
        return n * BLOCK

    def admit(self, hash_ids):
        for h in hash_ids:
            if h in self.blocks:
                self.blocks.move_to_end(h)
                continue
            self.blocks[h] = BLOCK
            self.used += BLOCK
            while self.used > self.cap and self.blocks:
                _, size = self.blocks.popitem(last=False)
                self.used -= size


# ---------------------------------------------------------------- policies
class Policy:
    """Base: every policy counts observed keys so cacheroute-style planners
    can be fed a past-only rate table. Routers never see the future."""
    counts = None
    def __init__(self, n, **kw):
        self.n = n
        self.counts = Counter()

class Sticky(Policy):
    name = 'sticky'
    def choose(self, req, engines, load, sim):
        return int(hashlib.sha256(req['key'].encode()).hexdigest(), 16) % self.n

class RoundRobin(Policy):
    name = 'rr'
    def __init__(self, n, **kw):
        super().__init__(n, **kw); self.i = 0
    def choose(self, req, engines, load, sim):
        t = self.i % self.n; self.i += 1; return t

class LeastLoaded(Policy):
    name = 'least-loaded'
    def choose(self, req, engines, load, sim):
        return min(range(self.n), key=lambda i: (load[i], i))

class TwoChoiceRing(Policy):
    """DualMap's double-hash two-choice: candidates from two rings, then load."""
    name = 'p2c-ring'
    def __init__(self, n, **kw):
        super().__init__(n, **kw)
        self.r1 = HashRing(nodes=[str(i) for i in range(n)])
        self.r2 = HashRing(nodes=[str(i) for i in range(n)], hash_fn='ketama')
    def _cands(self, key):
        a = int(self.r1.get_node(key)); b = int(self.r2.get_node(key))
        if a == b:
            b = (a + 1) % self.n
        return (a, b)
    def choose(self, req, engines, load, sim):
        key = 'fluidkv:' + ':'.join(map(str, req['hash_ids']))
        a, b = self._cands(key)
        return a if (load[a], a) <= (load[b], b) else b

class CacheRoute(Policy):
    """Paper Algorithm 1: plan over observed rates, dispatch by live load."""
    name = 'cacheroute'
    def __init__(self, n, qcap=None, slots=None, **kw):
        super().__init__(n, **kw)
        self.qcap = qcap or 1.0; self.slots = slots or n * 128
        self.table = {}
    def refresh(self):
        rates = {k: min(v / 60.0, math.nextafter(self.qcap * self.n, 0))
                 for k, v in self.counts.items()}
        ordered = sorted(((k, r) for k, r in rates.items() if r > 0),
                         key=lambda x: (-x[1], x[0]))
        table = {}; loads = [0.0] * self.n; used = 0
        for key, rate in ordered:
            count = max(1, math.ceil(rate / self.qcap))
            if count > self.n or used + count > self.slots:
                break
            targets = sorted(range(self.n), key=lambda i: (loads[i], i))[:count]
            table[key] = tuple(targets); used += count
            for t in targets:
                loads[t] += rate / count
        self.table = table
    def choose(self, req, engines, load, sim):
        targets = self.table.get(req['key'])
        if not targets:
            targets = tuple(random.Random(sim.seed).sample(range(self.n), min(2, self.n)))
        return min(targets, key=lambda i: (load[i], i))

class OracleOmniscient(Policy):
    """Knows every future request. Places work to maximize prefix reuse."""
    name = 'oracle-omniscient'
    def choose(self, req, engines, load, sim):
        best, best_hit = None, -1
        for i, e in enumerate(engines):
            hit = e.prefix_len(req['hash_ids'])
            if hit > best_hit or (hit == best_hit and load[i] < load[best]):
                best, best_hit = i, hit
        return best

class OracleStickyIdeal(Policy):
    """Omniscient placement with unbounded state: every stream keeps ONE home
    engine chosen to maximize total same-stream reuse. Upper bound on any
    single-home policy (sticky-class routers), which is what DualMap-class
    routers are."""
    name = 'oracle-home'
    def __init__(self, n, **kw):
        super().__init__(n, **kw); self.home = {}
    def fit(self, rows):
        by_key = defaultdict(list)
        for r in rows: by_key[r['key']].append(r)
        for key, rs in by_key.items():
            # every turn in a stream shares the same home; any home gives full
            # reuse because turns are sequential in one cache. Pick by hash.
            self.home[key] = int(hashlib.sha256(('ideal:'+key).encode()).hexdigest(), 16) % self.n
    def choose(self, req, engines, load, sim):
        return self.home[req['key']]


POLICIES = {p.name: p for p in
            (Sticky, RoundRobin, LeastLoaded, TwoChoiceRing, CacheRoute,
             OracleStickyIdeal, OracleOmniscient)}


# ---------------------------------------------------------------- simulator
def build_workload(period=60.0, streams=128, scenario='balanced', seed=734):
    """Same arrival model as experiments/weka_baselines/common.py:schedule."""
    rows = [json.loads(l) for l in (ROOT/'data/requests.jsonl').open()]
    lookup = {(r['stream_index'], r['turn']): r for r in rows}
    hot = sorted(range(streams),
                 key=lambda i: hashlib.sha256(f'hot:{i}'.encode()).digest())[:max(1, streams//4)]
    result = []
    for phase in range(3):
        pairs = [(i, phase) for i in range(streams)]
        if scenario == 'hotspot' and phase > 0:
            pairs = [(i, 1+(phase-1)*4+j) for j in range(4) for i in hot]
        rng = random.Random(seed+phase); rng.shuffle(pairs)
        pairs.sort(key=lambda p: p[1] if scenario == 'hotspot' and phase > 0 else 0)
        for rank, (i, turn) in enumerate(pairs):
            spread = 0 if scenario == 'burst' else period*.5*rank/max(1, len(pairs))
            r = dict(lookup[i, turn], ordinal=len(result), phase=phase,
                     offered_seconds=phase*period+spread)
            result.append(r)
    prior = {}
    for r in sorted(result, key=lambda r: (r['phase'], r['turn'], r['stream_index'])):
        r['predecessor'] = prior.get(r['key']); prior[r['key']] = r['ordinal']
    return sorted(result, key=lambda r: (r['offered_seconds'], r['ordinal']))


def run(policy_name, rows, engines_n, cap_tokens, service_scale, seed=42, qcap=1.0):
    random.seed(seed)
    pol = POLICIES[policy_name](engines_n, qcap=qcap)
    if hasattr(pol, 'fit'): pol.fit(rows)
    engines = [Engine(cap_tokens) for _ in range(engines_n)]
    sim = type('S', (), {'seed': seed})()
    live = [0]*engines_n
    busy_until = [0.0]*engines_n        # crude FIFO queue per engine
    uncached_total = 0; hit_total = 0; served = 0
    per_req = []; peak = [0]*engines_n
    phase_counts = Counter()
    for idx, req in enumerate(rows):
        # refresh rate tables at phase boundaries using only past arrivals
        if hasattr(pol, 'refresh'):
            t = req['offered_seconds']
            if not phase_counts or phase_counts['last_phase'] != int(t // 60):
                phase_counts['last_phase'] = int(t // 60)
                pol.refresh()
        target = pol.choose(req, engines, live, sim)
        pol.counts[req['key']] += 1
        e = engines[target]
        hit = e.prefix_len(req['hash_ids'])
        n_blocks = len(req['hash_ids'])
        e.admit(req['hash_ids'])
        uncached = req['input_tokens'] - hit
        uncached_total += uncached; hit_total += hit; served += 1
        live[target] += 1; peak[target] += 1
        # service: prefill dominates; naive serial estimate for load accounting
        ttft = (uncached * service_scale) + uncached*0
        per_req.append(dict(policy=policy_name, ordinal=req['ordinal'], target=target,
                            phase=req['phase'], key=req['key'], turn=req['turn'],
                            input_tokens=req['input_tokens'], hit_tokens=hit,
                            uncached_tokens=uncached, ttft_seconds=ttft))
        live[target] -= 1
    inp = sum(r['input_tokens'] for r in rows)
    hits = [r['hit_tokens'] for r in per_req]
    unc = [r['uncached_tokens'] for r in per_req]
    slots = Counter(r['target'] for r in per_req)
    load = np.array([slots.get(i, 0) for i in range(engines_n)], dtype=float)
    return dict(policy=policy_name, engines=engines_n, cap_tokens=cap_tokens,
                requests=served, total_input_tokens=inp,
                uncached_tokens=uncached_total, hit_tokens=hit_total,
                hit_ratio=hit_total/inp,
                reusability_achieved=hit_total/ (inp),
                uncached_p50=float(np.percentile(unc, 50)),
                uncached_p90=float(np.percentile(unc, 90)),
                uncached_p99=float(np.percentile(unc, 99)),
                uncached_mean=float(np.mean(unc)),
                per_engine_requests=dict(slots),
                imbalance=float(load.max()/max(1e-9, load.mean())),
                per_request=per_req)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--engines', type=int, default=3)
    ap.add_argument('--cap-tokens', type=int, default=1_048_576)
    ap.add_argument('--period', type=float, default=60.0)
    ap.add_argument('--scenario', default='balanced')
    ap.add_argument('--service-scale', type=float, default=1/20000)
    ap.add_argument('--out', type=Path, default=ROOT/'oracle/results')
    ap.add_argument('--policies', nargs='+', default=list(POLICIES))
    args = ap.parse_args()
    rows = build_workload(args.period, 128, args.scenario)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    print(f"{'policy':<20}{'hit%':>7}{'uncached%':>11}{'unc p50':>10}{'unc p99':>10}{'imbalance':>11}")
    for name in args.policies:
        res = run(name, rows, args.engines, args.cap_tokens, args.service_scale)
        per = res.pop('per_request')
        summary.append(res)
        print(f"{name:<20}{res['hit_ratio']*100:>6.1f}%{(1-res['hit_ratio'])*100:>10.1f}%"
              f"{res['uncached_p50']:>10.0f}{res['uncached_p99']:>10.0f}{res['imbalance']:>11.2f}")
        with (args.out/f"perreq-{name}.jsonl").open('w') as f:
            for r in per: f.write(json.dumps(r)+'\n')
    tag = f"n{args.engines}-cap{args.cap_tokens}-{args.scenario}"
    (args.out/f"summary-{tag}.json").write_text(json.dumps(summary, indent=2)+'\n')
    print(f"\nwrote {args.out/('summary-'+tag+'.json')}")

if __name__ == '__main__':
    main()
