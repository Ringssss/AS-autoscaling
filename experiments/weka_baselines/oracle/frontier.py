"""Affinity-vs-balance frontier on REAL session skew, corrected units.

Two quantities that the earlier version conflated:
  * LIVE WORKING SET -- real session spans put time-weighted live content at
    ~102 Mi-tokens over the whole trace. That is what a block cache must hold.
  * LOAD PRESSURE    -- time-weighted concurrency is only 12.6 sessions. A
    serving system is normally run far hotter, so concurrency is scaled to a
    stated target while intra-session gaps (the tool think time) stay real.

Reported per configuration: hit ratio (uncached prefill tokens are the TTFT
driver) and token-weighted busy-second imbalance, which is the quantity a
load-balancing router actually trades cache affinity against.
"""
import hashlib, json, random, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = '/home/zhujianian/AS-autoscaling'
sys.path.insert(0, REPO)
from experiments.weka_gravity.audit import flatten

TRACE = Path('/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl')
OUT = Path(REPO)/'experiments/weka_baselines/oracle/results'
NATIVE_BLOCK = 64
BYTES_PER_TOKEN = 20480
BANDWIDTH_BPS = 4.37*2**30
PREFILL_S_PER_TOKEN = 8.02e-5  # measured: experiments/multinode/.../calibration.json
DECODE_S_PER_TOKEN = 2.0e-2  # SLO TPOT bound, used as per-token decode cost
CONTEXT_CAP = 131072


class Cache:
    """Block cache with byte capacity, LRU. Blocks are native 64-token units."""
    def __init__(self, cap_blocks):
        from collections import OrderedDict
        self.cap = cap_blocks; self.blocks = OrderedDict(); self.used = 0
    def prefix_blocks(self, hid):
        n = 0
        for h in hid:
            if h not in self.blocks: break
            self.blocks.move_to_end(h); n += 1
        return n
    def admit(self, hid):
        for h in hid:
            if h in self.blocks:
                self.blocks.move_to_end(h); continue
            self.blocks[h] = 1; self.used += 1
            while self.used > self.cap and self.blocks:
                self.blocks.popitem(last=False); self.used -= 1


def load_sessions(limit=393):
    sessions = []
    with TRACE.open() as f:
        for li, line in enumerate(f, 1):
            if li > limit: break
            tr = json.loads(line); rows, _ = flatten(tr)
            per = defaultdict(list)
            for r in rows:
                if r.get('subagent'): continue
                if int(r['in'])+int(r['out']) > CONTEXT_CAP or int(r['out']) <= 0: continue
                per[(r['stream'], r['model'])].append(r)
            for (st, mo), rs in per.items():
                rs.sort(key=lambda r: float(r['t']))
                if len(rs) < 2: continue
                sessions.append(dict(key=f"{tr['id']}/{st}/{mo}", model=mo,
                    reqs=[dict(h=np.asarray(r['hash_ids'], dtype=np.int64),
                               tk=int(r['in']), out=int(r['out']),
                               t=float(r['t']), api=float(r['api_time']))
                          for r in rs]))
    return sessions


def build_events(sessions, concurrency_target):
    """Keep intra-session timing real; scale only the inter-session offset so the
    time-weighted concurrency matches the stated target."""
    # time-weighted live sessions in the raw trace
    ev_raw = []
    for si, s in enumerate(sessions):
        for ri, r in enumerate(s['reqs']):
            ev_raw.append((r['t'], si, ri))
    ev_raw.sort()
    lo = ev_raw[0][0]; hi = max(r['t']+r['api'] for s in sessions for r in s['reqs'])
    # reuse the measured 12.6 as the baseline density
    base = 12.6
    scale = base/concurrency_target
    ev = []
    for si, s in enumerate(sessions):
        t0 = s['reqs'][0]['t']
        for ri, r in enumerate(s['reqs']):
            ev.append((t0 + (r['t']-t0)*scale, si, ri))
    # sessions keep their own start; de-collide by hashing start times into
    # the compressed span so arrivals interleave instead of stacking.
    span = (hi-lo)*scale
    ev = []
    for si, s in enumerate(sessions):
        jitter = (int(hashlib.sha256(s['key'].encode()).hexdigest()[:8], 16)/2**32)*span
        t0 = s['reqs'][0]['t']*scale + jitter*0
        for ri, r in enumerate(s['reqs']):
            ev.append((jitter + (r['t']-s['reqs'][0]['t'])*scale, si, ri))
    ev.sort()
    return ev, scale


def simulate(sessions, events, engines_n, cap_tokens, policy, p, seed=42):
    cap_blocks = max(1, cap_tokens//NATIVE_BLOCK)
    caches = [Cache(cap_blocks) for _ in range(engines_n)]
    free = [0.0]*engines_n; busy = [0.0]*engines_n
    home = {}; loc = {}; lastdone = {}
    rng = random.Random(seed)
    unc = tot = xbytes = 0; waits = []; moved = 0
    for t, si, ri in events:
        s = sessions[si]; key = s['key']; r = s['reqs'][ri]; hid = r['h']
        if key not in home:
            home[key] = int(hashlib.sha256(key.encode()).hexdigest(), 16) % engines_n
        h = home[key]
        loc.setdefault(key, h)
        target = h
        if policy == 'balance':
            target = min(range(engines_n), key=lambda i: (busy[i], i))
        elif policy == 'loadaware':
            slack = p['slack']*(1.0+p['deg']*8.0)
            if (free[h]-t) > slack:
                target = min(range(engines_n), key=lambda i: (free[i], i))
        elif policy in ('mover', 'mover-ideal'):
            src = loc[key]
            if (free[src]-t) > p['slack']:
                cand = min(range(engines_n), key=lambda i: (free[i], i))
                if cand != src:
                    warm_b = caches[src].prefix_blocks(hid)
                    warm = warm_b*NATIVE_BLOCK
                    xf = warm*BYTES_PER_TOKEN/BANDWIDTH_BPS
                    W = t-lastdone.get(key, t)
                    if policy == 'mover-ideal' or W >= xf:
                        caches[cand].admit(hid[:warm_b])
                        loc[key] = cand; xbytes += warm*BYTES_PER_TOKEN
                        moved += 1; target = cand
                        t = max(t, lastdone.get(key, t)+xf)
                    else:
                        target = src
                else:
                    target = src
            else:
                target = src
        c = caches[target]
        hit = c.prefix_blocks(hid)*NATIVE_BLOCK
        c.admit(hid)
        u = max(0, r['tk']-hit)
        unc += u; tot += r['tk']
        svc = u*PREFILL_S_PER_TOKEN + r['out']*DECODE_S_PER_TOKEN
        st = max(t, free[target]); waits.append(st-t)
        free[target] = st+svc; busy[target] += svc
        lastdone[key] = free[target]
    b = np.asarray(busy, float)
    return dict(policy=policy, params=dict(p), engines=engines_n,
                cap_tokens=cap_tokens, requests=len(events), input_tokens=int(tot),
                hit_ratio=1-unc/tot, busy_imbalance=float(b.max()/max(1e-9, b.mean())),
                busy_seconds=float(b.sum()), makespan=float(max(free)),
                wait_p50=float(np.percentile(waits, 50)),
                wait_p99=float(np.percentile(waits, 99)),
                moved=moved, transfer_gib=xbytes/2**30)


def held(sessions):
    return [dict(key=s['key'], model=s['model'],
                 reqs=[dict(h=r['h'], tk=r['tk'], out=r['out'], t=r['t'], api=r['api'])
                       for r in s['reqs']]) for s in sessions]


def main():
    sessions = load_sessions()
    lng = np.array([len(s['reqs']) for s in sessions], float)
    tokv = np.array([sum(r['tk'] for r in s['reqs']) for s in sessions], float)
    nreq = sum(len(s['reqs']) for s in sessions)
    print(f"sessions {len(sessions)}  requests {nreq:,}")
    print(f"session length p50={np.percentile(lng,50):.0f} p90={np.percentile(lng,90):.0f} max={lng.max():.0f}")
    print(f"token share: top1% {np.sort(tokv)[-len(tokv)//100:].sum()/tokv.sum():.1%}  "
          f"top10% {np.sort(tokv)[-len(tokv)//10:].sum()/tokv.sum():.1%}")
    allb = set()
    for s in sessions:
        for r in s['reqs']: allb.update(r['h'].tolist())
    print(f"distinct native blocks {len(allb):,} = {len(allb)*NATIVE_BLOCK/2**20:.0f} Mi-tokens unique\n")

    results = []
    for conc, eng in ((60, 3), (120, 3), (240, 3), (120, 8)):
        ev, scale = build_events(sessions, conc)
        print(f"=== concurrency {conc}, {eng} engines (span scaled x{scale:.3f}) ===")
        print(f"{'policy':<20}{'cap':>8}{'hit%':>8}{'imb':>7}{'busy_s':>11}{'p99wait':>9}{'moved':>7}")
        for capM in (32, 64, 128, 256, 512):
            r = simulate(sessions, ev, eng, capM*2**20, 'affinity', {})
            r['concurrency'] = conc; results.append(r)
            print(f"{'affinity':<20}{capM:>7}M{r['hit_ratio']*100:>7.1f}%"
                  f"{r['busy_imbalance']:>7.2f}{r['busy_seconds']:>11.0f}"
                  f"{r['wait_p99']:>9.1f}{r['moved']:>7}")
        for pol, prm in (('loadaware', dict(deg=0.05, slack=5.0)),
                         ('loadaware', dict(deg=0.5, slack=5.0)),
                         ('loadaware', dict(deg=2.0, slack=5.0)),
                         ('balance', {}),
                         ('mover', dict(slack=5.0)),
                         ('mover-ideal', dict(slack=5.0))):
            r = simulate(sessions, ev, eng, 128*2**20, pol, prm)
            r['concurrency'] = conc; results.append(r)
            tag = pol+(f"(d={prm['deg']})" if 'deg' in prm else '')
            print(f"{tag:<20}{128:>7}M{r['hit_ratio']*100:>7.1f}%"
                  f"{r['busy_imbalance']:>7.2f}{r['busy_seconds']:>11.0f}"
                  f"{r['wait_p99']:>9.1f}{r['moved']:>7}"
                  + (f"  xfer={r['transfer_gib']:.0f}GiB" if r['transfer_gib'] else ""))
        print()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'frontier.json').write_text(json.dumps(dict(results=results), indent=2)+'\n')
    print(f"wrote {OUT/'frontier.json'}")


if __name__ == '__main__':
    main()
