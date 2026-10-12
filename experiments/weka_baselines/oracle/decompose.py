"""Reconcile the structural census against the simulator ceiling, per request.

For every request compute two prefix-hit quantities over the SAME rows:
  owned  - leading blocks already seen in this request's own stream
           (ceiling for any perfect-affinity / single-home router)
  global - leading blocks already seen anywhere, any stream
           (ceiling for one infinite shared cache)
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, '/home/zhujianian/AS-autoscaling')
from experiments.weka_baselines.oracle.simulate import build_workload, BLOCK, ROOT

rows = build_workload(60.0, 128, 'balanced')
inp = sum(r['input_tokens'] for r in rows)
print(f"requests {len(rows)}   input tokens {inp:,}")

seen = set(); glob = []
for r in rows:
    n = 0
    for h in r['hash_ids']:
        if h in seen: n += 1
        else: break
    glob.append(n*BLOCK)
    seen.update(r['hash_ids'])

last = {}; own = []
for r in rows:
    p = last.get(r['key'])
    n = 0
    if p is not None:
        for a, b in zip(p, r['hash_ids']):
            if a != b: break
            n += 1
    own.append(n*BLOCK)
    last[r['key']] = r['hash_ids']

own = np.array(own); glob = np.array(glob)
inp_arr = np.array([r['input_tokens'] for r in rows])

print(f"\nown-stream perfect-affinity ceiling : {own.sum()/inp:6.2%} of input")
print(f"global infinite-cache ceiling        : {glob.sum()/inp:6.2%} of input")
print(f"cross-stream sharing headroom        : {(glob.sum()-own.sum())/inp:6.2%} of input")

# where the tokens are
irred = glob == 0
print(f"\nirreducibly cold requests (no prior block anywhere): {irred.sum()} / {len(rows)} = {irred.mean():.1%}")
print(f"  their share of total input tokens: {inp_arr[irred].sum()/inp:.1%}")
print(f"  their mean input: {inp_arr[irred].mean():,.0f} vs warm mean {inp_arr[~irred].mean():,.0f}")

# tail concentration of the residual (what any router still pays)
resid = inp_arr - own        # uncached under perfect affinity
order = np.argsort(-resid)
tot_resid = resid.sum()
for frac in (0.01, 0.05, 0.10, 0.25):
    k = max(1, int(frac*len(rows)))
    print(f"  top {frac:>5.0%} requests carry {resid[order[:k]].sum()/tot_resid:6.1%} of residual uncached tokens")
print(f"  residual uncached: mean {resid.mean():,.0f}  p50 {np.percentile(resid,50):,.0f} "
      f"p90 {np.percentile(resid,90):,.0f} p99 {np.percentile(resid,99):,.0f} max {resid.max():,.0f}")

# LCP fraction per request
frac = own/inp_arr
print(f"\nper-request prefix fraction: p10 {np.percentile(frac,10):.3f} p50 {np.percentile(frac,50):.3f} "
      f"p90 {np.percentile(frac,90):.3f}")
print(f"  requests with fraction 0: {(frac==0).sum()}  fraction>0.9: {(frac>0.9).sum()}")

(ROOT/'oracle/results/decompose.json').write_text(json.dumps(dict(
    requests=len(rows), input_tokens=int(inp),
    own_ceiling=float(own.sum()/inp), global_ceiling=float(glob.sum()/inp),
    irreducible_requests=int(irred.sum()),
    irreducible_token_share=float(inp_arr[irred].sum()/inp),
    residual_uncached_mean=float(resid.mean()),
    residual_p99=float(np.percentile(resid,99)),
), indent=2)+'\n')
