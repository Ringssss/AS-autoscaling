"""Direct affinity ceiling on REAL main-agent sessions, no simulator.

Independent of simulate.py and of any cache/U sizing. For each session,
request n's achievable hit under a single home engine with infinite capacity is
the longest common prefix with the best earlier request of the same session.
Also reports the same-session *nearest predecessor* prefix, which is what a
sequential sticky router actually gets.
"""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO='/home/zhujianian/AS-autoscaling'
sys.path.insert(0,REPO)
from experiments.weka_gravity.audit import flatten

TRACE=Path('/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl')
CONTEXT_CAP=131072
NATIVE_BLOCK=64

def lcp(a,b):
    n=0
    for x,y in zip(a,b):
        if x!=y: break
        n+=1
    return n

streams=[]
with TRACE.open() as f:
    for li,line in enumerate(f,1):
        if li>393: break
        tr=json.loads(line)
        rows,_=flatten(tr)
        per=defaultdict(list)
        for r in rows:
            if r.get('subagent'): continue
            if int(r['in'])+int(r['out'])>CONTEXT_CAP or int(r['out'])<=0: continue
            per[(r['stream'],r['model'])].append(r)
        for (st,mo),rs in per.items():
            rs.sort(key=lambda r: float(r['t']))
            if len(rs)<2: continue
            streams.append(dict(key=f"{tr['id']}/{st}/{mo}",model=mo,
                reqs=[dict(h=np.asarray(r['hash_ids'],dtype=np.int64),tk=int(r['in']),
                           out=int(r['out']),t=float(r['t'])) for r in rs]))
print(f"sessions {len(streams)}  requests {sum(len(s['reqs']) for s in streams):,}")

tot=0; best_tot=0; pred_tot=0; zero=0; nreq=0
per_req=[]
block_sizes=[NATIVE_BLOCK,16]
for bs in block_sizes:
    tot=best_tot=pred_tot=0; zero=0; nreq=0
    for s in streams:
        for i,r in enumerate(s['reqs']):
            tot+=r['tk']; nreq+=1
            if i==0:
                per_req.append((0.0,bs)); continue
            # nearest predecessor
            p=s['reqs'][i-1]
            plcp=lcp(p['h'],r['h'])
            # best earlier request in the session
            blcp=max(lcp(q['h'],r['h']) for q in s['reqs'][:i])
            pred_tot+=min(plcp*bs,r['tk'])
            best_tot+=min(blcp*bs,r['tk'])
            if blcp==0: zero+=1
            per_req.append((min(blcp*bs,r['tk'])/max(1,r['tk']),bs))
    print(f"\n--- native hash_ids are {bs}-token blocks ---")
    print(f"input tokens {tot:,}")
    print(f"ceiling: nearest-predecessor sticky  {pred_tot/tot*100:5.2f}%")
    print(f"ceiling: best-earlier-in-session     {best_tot/tot*100:5.2f}%")
    print(f"requests with zero same-stream reuse {zero}/{nreq} = {zero/nreq*100:.1f}%")
    fr=np.array([x[0] for x in per_req if x[1]==bs])
    print(f"per-request reachable fraction p10/50/90 = "
          f"{np.percentile(fr,10):.3f}/{np.percentile(fr,50):.3f}/{np.percentile(fr,90):.3f}")

# how many DISTINCT hash_ids exist across everything (real cache footprint)
allh=set()
for s in streams:
    for r in s['reqs']: allh.update(r['h'].tolist())
print(f"\ndistinct native blocks across all sessions {len(allh):,} "
      f"= {len(allh)*NATIVE_BLOCK/2**20:,.0f} Mi-tokens of unique content")
print(f"total input tokens {sum(r['tk'] for s in streams for r in s['reqs']):,} "
      f"= {sum(r['tk'] for s in streams for r in s['reqs'])/2**20:,.0f} Mi-tokens")
