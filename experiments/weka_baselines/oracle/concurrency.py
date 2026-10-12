"""How many sessions are actually alive at once on the real timeline?
This is what decides whether a per-engine cache is ever capacity-bound."""
import json,sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO='/home/zhujianian/AS-autoscaling';sys.path.insert(0,REPO)
from experiments.weka_gravity.audit import flatten
TRACE=Path('/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl')
CAP=131072

# Real sessions with (start, end) of actual activity. api_time = wall duration of
# the call, so a session is "alive" from first call start to last call start+api_time.
sess=[]
with TRACE.open() as f:
    for li,line in enumerate(f,1):
        if li>393: break
        tr=json.loads(line); rows,_=flatten(tr)
        per=defaultdict(list)
        for r in rows:
            if r.get('subagent'): continue
            if int(r['in'])+int(r['out'])>CAP or int(r['out'])<=0: continue
            per[(r['stream'],r['model'])].append(r)
        for (st,mo),rs in per.items():
            rs.sort(key=lambda r: float(r['t']))
            if len(rs)<2: continue
            sess.append((float(rs[0]['t']),
                         float(rs[-1]['t'])+float(rs[-1]['api_time']),
                         len(rs),sum(int(r['in']) for r in rs)))
print(f"sessions {len(sess)}")
span=np.percentile([e-s for s,e,_,_ in sess],[10,50,90,99])
print(f"session wall span s: p10={span[0]:.0f} p50={span[1]:.0f} p90={span[2]:.0f} p99={span[3]:.0f}")
lo=min(s for s,_,_,_ in sess); hi=max(e for _,e,_,_ in sess)
print(f"trace span {(hi-lo)/3600:.1f} h")

# concurrency over real time, sampled
t0=min(s for s,_,_,_ in sess)
events=[]
for s,e,n,tok in sess:
    events.append((s,1,tok)); events.append((e,-1,-tok))
events.sort()
cur=0; curtok=0; prof=[]
for t,d,dt in events:
    cur+=d; curtok+=dt; prof.append((t,cur,curtok))
c=np.array([p[1] for p in prof]); ct=np.array([p[2] for p in prof])
print(f"\nconcurrent sessions: p10={np.percentile(c,10):.0f} p50={np.percentile(c,50):.0f} "
      f"p90={np.percentile(c,90):.0f} p99={np.percentile(c,99):.0f} max={c.max():.0f}")
print(f"concurrent live tokens: p50={np.percentile(ct,50)/2**20:.1f}Mi "
      f"p90={np.percentile(ct,90)/2**20:.1f}Mi max={ct.max()/2**20:.1f}Mi")
print(f"\ntime-weighted concurrency (sessions): {np.trapezoid(c,[p[0] for p in prof])/(hi-lo):.1f}")
print(f"time-weighted live tokens: {np.trapezoid(ct,[p[0] for p in prof])/(hi-lo)/2**20:.1f}Mi")

# What 3 engines would each have to hold if sessions were pinned by hash
print(f"\nif 3 engines, evenly pinned, average live per engine: "
      f"{np.trapezoid(ct,[p[0] for p in prof])/(hi-lo)/3/2**20:.1f}Mi tokens")
print(f"peak per engine: {ct.max()/3/2**20:.1f}Mi tokens")

# arrival gap distribution -> the time compression factor we should use
gaps=[]
for s,e,n,tok in sess:
    pass
allt=[]
with TRACE.open() as f:
    for li,line in enumerate(f,1):
        if li>393: break
        tr=json.loads(line); rows,_=flatten(tr)
        per=defaultdict(list)
        for r in rows:
            if r.get('subagent'): continue
            if int(r['in'])+int(r['out'])>CAP or int(r['out'])<=0: continue
            per[(r['stream'],r['model'])].append(r)
        for (st,mo),rs in per.items():
            rs.sort(key=lambda r: float(r['t']))
            for a,b in zip(rs,rs[1:]):
                g=float(b['t'])-(float(a['t'])+float(a['api_time']))
                if g>0: gaps.append(g)
g=np.array(gaps)
print(f"\nidle gaps between calls (s): p10={np.percentile(g,10):.2f} p50={np.percentile(g,50):.2f} "
      f"p90={np.percentile(g,90):.1f} p99={np.percentile(g,99):.1f} max={g.max():.0f}")
print(f"gap<1s {np.mean(g<1)*100:.1f}%  <5s {np.mean(g<5)*100:.1f}%  <30s {np.mean(g<30)*100:.1f}%  >600s {np.mean(g>600)*100:.1f}%")
