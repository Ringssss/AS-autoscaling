"""Preselect count-stratified independent windows without policy-result access."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
p=np.load(ROOT/'results/kimi-opportunity/native_numeric_fields.npz')
a=p['arrival'];origin=float(a.min());bins=((a-origin)//10).astype(int);counts=np.bincount(bins)
pool=np.arange(60,len(counts)-1);pool=pool[counts[pool]>0]
# Exclude already selected calibration/evaluation windows from held-out sampling.
pool=pool[~np.isin(pool,[7166,1840,*range(314,320)])]
q=np.quantile(counts[pool],[1/3,2/3]);rng=np.random.default_rng(20260926)
strata=[pool[counts[pool]<=q[0]],pool[(counts[pool]>q[0])&(counts[pool]<=q[1])],pool[counts[pool]>q[1]]]
windows=[];design=[]
for label,members in zip(['low','middle','high'],strata):
    selected=rng.choice(members,4,replace=False)
    design.append(dict(stratum=label,population_windows=len(members),selected_bins=selected.tolist(),weight=len(members)/len(pool)))
    for index in selected:
        index=int(index);start=origin+index*10;ids=np.flatnonzero(bins==index);ids=ids[np.argsort(a[ids],kind='stable')]
        rows=[dict(ordinal=j,native_index=int(i),native_arrival=float(a[i]),offset=float((a[i]-origin)-10*index),input_tokens=int(p['input_tokens'][i]),output_tokens=int(p['output_tokens'][i]),observed_cached_tokens=int(p['cached_tokens'][i])) for j,i in enumerate(ids)]
        windows.append(dict(label=f'{label}-{index}',stratum=label,bin=index,start=start,duration=10,requests=rows,
            training_inputs=p['input_tokens'][(a>=start-60)&(a<start)].tolist(),
            selection='Four random nonempty windows per count tercile; held-out from original windows; seed20260926; no output/latency/performance selection.'))
base=ROOT/'results'
(base/'kimi-trace-coverage-windows.json').write_text(json.dumps(windows,indent=2)+'\n')
(base/'kimi-trace-coverage-design.json').write_text(json.dumps(dict(seed=20260926,thresholds=q.tolist(),population_windows=len(pool),strata=design),indent=2)+'\n')
spec=[dict(stage='coverage',window=w['label'],length=5760,withdraw=True,tool_seconds=.5,rate_scale=1.,output_cap=64,policy=policy,repeat=0) for w in windows for policy in ['p2c','p2c+fluidkv','dualmap','cacheroute']]
(base/'kimi-trace-coverage-jobs.json').write_text(json.dumps(spec,indent=2)+'\n')
print(json.dumps(design,indent=2));print('requests',sum(len(w['requests']) for w in windows))
