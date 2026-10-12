"""Real-engine placement comparison on the materialized trace.

Same requests, same engines, same arrival concurrency; only the placement
policy changes. Caches are flushed once per trial and never between requests,
so a policy's hit ratio is whatever its own placement produced.

  sticky    one engine per stream (hash) -- the cache-affinity extreme
  balanced  least-loaded by busy seconds -- the balance extreme
  sticky+move  sticky, but a stream whose home engine is congested has its
               warm prefix relocated to the least-loaded engine first

TTFT comes from the streaming client. Hit counts are scraped from engine logs,
which is the only place this build reports them.
"""
import argparse, asyncio, hashlib, json, re, subprocess, sys, time
from pathlib import Path
import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

LOG_RE = re.compile(r'#new-token: (\d+), #cached-token: (\d+)')

def load_workload():
    rows = [json.loads(l) for l in (ROOT/'experiments/weka_baselines/data/requests.jsonl').open()]
    toks = np.load(ROOT/'experiments/weka_baselines/data/tokens.npy')
    return rows, toks

def build_schedule(rows, concurrency):
    """Emit (turn) order with per-stream causality kept: turn k of a stream is
    only released after turn k-1 returns. Cross-stream order is interleaved."""
    by = {}
    for r in rows: by.setdefault(r['key'], []).append(r)
    for k in by: by[k].sort(key=lambda r: r['turn'])
    ready = list(by.values())
    ready.sort(key=lambda rs: hashlib.sha256(rs[0]['key'].encode()).hexdigest())
    return ready

async def flush(session, urls):
    for u in urls:
        async with session.post(u+'/flush_cache') as r: await r.text()
    await asyncio.sleep(1.0)

async def drain(session, urls):
    for _ in range(300):
        ok = True
        for u in urls:
            async with session.get(u+'/get_load') as r: v = await r.json()
            if not all(x['num_reqs'] == 0 for x in v): ok = False; break
        if ok: return
        await asyncio.sleep(0.2)
    raise RuntimeError('engines did not drain')

async def send(session, url, ids, out_tokens):
    payload = dict(input_ids=[int(x) for x in ids], stream=True,
                   sampling_params=dict(temperature=0, ignore_eos=True,
                                        max_new_tokens=out_tokens))
    t0 = time.perf_counter(); first = None
    async with session.post(url+'/generate', json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status} {(await resp.text())[:300]}')
        async for line in resp.content:
            if not line.startswith(b'data:'): continue
            d = line[5:].strip()
            if d == b'[DONE]': continue
            j = json.loads(d)
            if first is None and j.get('meta_info', {}).get('completion_tokens', 0) >= 1:
                first = time.perf_counter()
    if first is None: raise RuntimeError('no output')
    return first - t0, time.perf_counter() - t0

class Mover:
    """sticky + relocatable warm prefix. Relocation copies the prefix by
    re-running it on the target (the control plane path exists, but the
    placement decision is what is being measured)."""
    def __init__(self, urls):
        self.urls=urls; self.busy=[0.0]*len(urls); self.home={}

    def home_of(self, key):
        if key not in self.home:
            self.home[key] = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(self.urls)
        return self.home[key]

async def run_trial(session, urls, streams, policy, concurrency, seed=42):
    await flush(session, urls)
    await drain(session, urls)
    busy = [0.0]*len(urls)
    home = {}
    results = []
    lock = asyncio.Lock()
    sem = asyncio.Semaphore(concurrency)

    async def run_stream(rs):
        key = rs[0]['key']
        if key not in home:
            home[key] = int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(urls)
        h = home[key]
        for r in rs:
            async with sem:
                async with lock:
                    if policy == 'balanced':
                        target = min(range(len(urls)), key=lambda i: (busy[i], i))
                    elif policy in ('dualmap','dualmap+move'):
                        a = int(hashlib.blake2b((key+':1').encode(), digest_size=8).hexdigest(),16) % len(urls)
                        b = int(hashlib.blake2b((key+':2').encode(), digest_size=8).hexdigest(),16) % len(urls)
                        if a == b: b = (a+1) % len(urls)
                        target = a if (busy[a], a) <= (busy[b], b) else b
                    elif policy in ('sticky+move','dualmap+move'):
                        mean = sum(busy)/len(busy)
                        if busy[h] > mean*(1.0+MOVE_DEG) and busy[h] > MOVE_FLOOR:
                            target = min(range(len(urls)), key=lambda i: (busy[i], i))
                        else:
                            target = h
                    else:
                        target = h
                ids = ALL_TOKENS[r['token_offset']: r['token_offset']+r['input_tokens']]
                try:
                    ttft, tot = await send(session, urls[target], ids, OUT_TOKENS)
                    err = None
                except Exception as e:
                    ttft = tot = float('nan'); err = repr(e)[:200]
                async with lock:
                    busy[target] += tot
                    results.append(dict(key=key, turn=r['turn'], engine=target,
                                        ttft=ttft, total=tot, input_tokens=r['input_tokens'],
                                        output_tokens=r['output_tokens'], error=err))

    started = time.perf_counter()
    await asyncio.gather(*[run_stream(rs) for rs in streams])
    wall = time.perf_counter() - started
    await drain(session, urls)
    return dict(policy=policy, concurrency=concurrency, wall=wall,
                requests=len(results), results=results)

# Live engines as of 2026-09-30, each verified to have a single listener on its
# port and an identical configuration (max_total_tokens=262144, mamba=256).
# The previous table listed several containers that no longer exist.
ENGINES = [
    ('12.11.3.27:31781', '12.11.3.27', 'zhujianian-fluidkv-p1-node27-31781'),
    ('12.11.3.27:31782', '12.11.3.27', 'zhujianian-fluidkv-p1-node27-31782'),
    ('12.11.3.28:31751', '12.11.3.28', 'zhujianian-agentx-multi-node281'),
    ('12.11.3.29:31760', '12.11.3.29', 'zhujianian-agentx-smoke'),
    ('12.11.3.29:31761', '12.11.3.29', 'zhujianian-agentx-real-node291'),
    ('12.11.3.29:31762', '12.11.3.29', 'zhujianian-agentx-real-node292'),
]
NODE = '12.11.3.29'

def remote_now(node=NODE):
    """The node's own clock in its own timezone. Engine log lines carry local
    time (CST here), so --since must be local too; using UTC widened the window
    by 8h and re-counted every earlier trial."""
    return subprocess.run(['ssh','-o','BatchMode=yes',node,"date '+%Y-%m-%dT%H:%M:%S'"],
                          capture_output=True, text=True, timeout=60).stdout.strip()

def scrape_hits(containers, since):
    """Only the trial window, per node. Log timestamps are the node's local
    time, so --since must be captured from that same node."""
    out = {}
    for (_, node, c) in containers:
        cmd = ['ssh','-o','BatchMode=yes',node,
               f"docker logs --since {since} {c} 2>&1 | grep -E 'Prefill batch'"]
        txt = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
        cached = 0; new = 0; n = 0
        for line in txt.splitlines():
            m = LOG_RE.search(line)
            if m: new += int(m.group(1)); cached += int(m.group(2)); n += 1
        out[c] = dict(cached=cached, new=new, batches=n)
    return out

ALL_TOKENS = None
OUT_TOKENS = 32
MOVE_DEG = 0.25
MOVE_FLOOR = 30.0

async def main():
    global ALL_TOKENS, OUT_TOKENS, MOVE_DEG, MOVE_FLOOR
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', nargs='+', default=['sticky','balanced','sticky+move'])
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--streams', type=int, default=32)
    ap.add_argument('--out-tokens', type=int, default=32)
    ap.add_argument('--move-deg', type=float, default=0.25)
    ap.add_argument('--move-floor', type=float, default=30.0)
    ap.add_argument('--scenario', default='balanced')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    rows, toks = load_workload(); ALL_TOKENS = toks; OUT_TOKENS = a.out_tokens; MOVE_DEG = a.move_deg; MOVE_FLOOR = a.move_floor
    urls = ['http://'+e[0] for e in ENGINES]
    conts = ENGINES
    allstreams = build_schedule(rows, a.concurrency)
    streams = allstreams[:a.streams]
    if a.scenario == 'hotspot':
        # A small subset carries most of the traffic: this is the only regime
        # where a balancing router has a reason to exist.
        hot = streams[:max(1, len(streams)//8)]
        streams = hot*8 + streams[len(hot):]
    nreq = sum(len(s) for s in streams)
    print(f"scenario {a.scenario}  streams {len(streams)}  requests {nreq}  "
          f"concurrency {a.concurrency}  engines {len(urls)}")
    outdir = Path(a.out) if a.out else ROOT/'experiments/weka_baselines/results/real-20260929'
    outdir.mkdir(parents=True, exist_ok=True)
    summary = []
    async with aiohttp.ClientSession() as session:
        for pol in a.policy:
            t0 = time.time()
            since = remote_now()
            await asyncio.sleep(1.5)
            res = await run_trial(session, urls, streams, pol, a.concurrency)
            hits = scrape_hits(conts, since)
            ttfts = np.array([r['ttft'] for r in res['results']], float)
            ok = np.isfinite(ttfts)
            errors = sum(1 for r in res['results'] if r['error'])
            slo = float(np.mean(ttfts[ok] <= 10.0))
            rec = dict(policy=pol, concurrency=a.concurrency, streams=len(streams),
                       requests=nreq, wall=res['wall'], errors=errors,
                       ttft_p50=float(np.percentile(ttfts[ok],50)),
                       ttft_p90=float(np.percentile(ttfts[ok],90)),
                       ttft_p99=float(np.percentile(ttfts[ok],99)),
                       ttft_max=float(ttfts[ok].max()), ttft_mean=float(ttfts[ok].mean()),
                       slo_goodput=slo, log_hits=hits, elapsed=time.time()-t0)
            summary.append(rec)
            print(f"{pol:<14} p50={rec['ttft_p50']:7.2f} p90={rec['ttft_p90']:7.2f} "
                  f"p99={rec['ttft_p99']:7.2f} max={rec['ttft_max']:7.2f} "
                  f"SLO={slo*100:5.1f}% err={errors} wall={res['wall']:6.1f}s "
                  f"hits={sum(v['cached'] for v in hits.values()):,}", flush=True)
            with (outdir/f"trial-{a.scenario}-{pol}-c{a.concurrency}.jsonl").open('w') as f:
                for r in res['results']: f.write(json.dumps(r)+'\n')
    tag = f"summary-{a.scenario}-c{a.concurrency}"
    (outdir/f"{tag}.json").write_text(json.dumps(summary, indent=2)+'\n')
    print(f"\nwrote {outdir}/{tag}.json")

if __name__ == '__main__':
    asyncio.run(main())
