"""Census of UNTRUNCATED weka traces: session lengths, rate skew, reuse, gaps.

The materialized 128-stream workload forces exactly 3 requests per stream,
which erases rate skew - precisely the condition under which pure affinity
fails. This measures the real distribution.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/zhujianian/AS-autoscaling')
from experiments.weka_gravity.audit import flatten

TRACE = Path('/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl')
sess_len = []; reuse_tok = []; cur_in = []; gaps = []
streams = 0; reqs = 0; zero = 0
per_trace_streams = []
with TRACE.open() as f:
    for li, line in enumerate(f, 1):
        trace = json.loads(line)
        flat, _ = flatten(trace)
        bys = defaultdict(list)
        for r in flat:
            bys[(r['stream'], r['model'])].append(r); reqs += 1
        per_trace_streams.append(len(bys))
        for s, rs in bys.items():
            rs.sort(key=lambda r: r['t']); streams += 1
            sess_len.append(len(rs))
            for p, c in zip(rs, rs[1:]):
                n = 0
                for a, b in zip(p['hash_ids'], c['hash_ids']):
                    if a != b: break
                    n += 1
                reuse_tok.append(64*n); cur_in.append(c['in'])
                gaps.append(c['t']-p['t'])
                if n == 0: zero += 1
        if li % 50 == 0: print(f"  scanned {li} traces", flush=True)

def pct(v, p):
    return float(np.percentile(np.asarray(v, dtype=float), p))

out = {}
print(f"\ntraces {len(per_trace_streams)}   streams {streams}   requests {reqs}")
print(f"\n--- session length (requests per stream) ---")
for p in (10, 50, 90, 99, 100):
    print(f"  p{p:<3} {pct(sess_len,p):>8.1f}")
sl = np.asarray(sess_len, dtype=float)
print(f"  mean {sl.mean():.1f}   share of requests in the longest 1% of streams: "
      f"{np.sort(sl)[-max(1,len(sl)//100):].sum()/sl.sum():.1%}")
print(f"  share of requests in the longest 10% of streams: "
      f"{np.sort(sl)[-max(1,len(sl)//10):].sum()/sl.sum():.1%}")

print(f"\n--- consecutive-request prefix reuse within a stream ---")
ci = np.asarray(cur_in, dtype=float); rt = np.asarray(reuse_tok, dtype=float)
print(f"  pairs {len(rt)}   zero-LCP pairs {zero} ({zero/len(rt):.1%})")
print(f"  reusable / input = {rt.sum()/ci.sum():.1%}")
print(f"  reuse tokens p10/p50/p90/p99: {pct(rt,10):,.0f}/{pct(rt,50):,.0f}/{pct(rt,90):,.0f}/{pct(rt,99):,.0f}")
f_ = rt/ci
print(f"  per-pair reuse fraction: p10 {pct(f_,10):.3f} p50 {pct(f_,50):.3f} p90 {pct(f_,90):.3f}")

print(f"\n--- inter-turn gap (s) ---")
g = np.asarray(gaps, dtype=float)
print(f"  p10/p50/p90/p99/max: {pct(g,10):.1f}/{pct(g,50):.1f}/{pct(g,90):.1f}/{pct(g,99):.1f}/{g.max():.1f}")
print(f"  gap < 1s: {(g<1).mean():.1%}   <10s: {(g<10).mean():.1%}   >300s: {(g>300).mean():.1%}")

out = dict(traces=len(per_trace_streams), streams=streams, requests=reqs,
           session_len={f'p{p}': pct(sess_len,p) for p in (10,50,90,99)},
           session_len_max=int(sl.max()), session_len_mean=float(sl.mean()),
           top1pct_share=float(np.sort(sl)[-max(1,len(sl)//100):].sum()/sl.sum()),
           top10pct_share=float(np.sort(sl)[-max(1,len(sl)//10):].sum()/sl.sum()),
           pairs=len(rt), zero_lcp_pairs=zero, reuse_fraction=float(rt.sum()/ci.sum()),
           gap_p50=pct(g,50), gap_p90=pct(g,90), gap_p99=pct(g,99),
           gap_lt1s=float((g<1).mean()), gap_lt10s=float((g<10).mean()))
(Path('/home/zhujianian/AS-autoscaling/experiments/weka_baselines/oracle/results/rawcensus.json')).write_text(
    json.dumps(out, indent=2)+'\n')
