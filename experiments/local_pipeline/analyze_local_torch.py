import json,gzip,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent/'results/20260915/profile'
anchor=json.loads((P/'clock.json').read_text())
def stat(v):return {'n':len(v),'p50_ms':float(np.median(v)),'p99_ms':float(np.percentile(v,99)),'max_ms':float(max(v))} if v else {'n':0}
def overlap_union(intervals,a,b):
    total=0;end=a
    for x,y in sorted(intervals):
        x=max(x,a);y=min(y,b)
        if y>x:total+=max(0,y-max(x,end));end=max(end,y)
    return total
out=[];fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True)
for row,ax in zip(map(json.loads,(P/'records.jsonl').read_text().splitlines()),axes):
    res={k:row[k] for k in ['index','target','mode','trigger']};res['engines']={}
    for eng in [0,1]:
        d=json.load(gzip.open(next(P.glob(f"profile-{row['index']}-engine-{eng}/*.gz"))))
        offset=d['baseTimeNanoseconds']/1e9-anchor['epoch']+anchor['monotonic']
        es=[]
        for e in d['traceEvents']:
            if e.get('ph')=='X':
                a=offset+e['ts']/1e6;b=a+e.get('dur',0)/1e6
                if row['start']<=a<=row['finish']:es.append((a,b,e))
        kernels=[(a,b) for a,b,e in es if e.get('cat') in ['kernel','gpu_memcpy','gpu_memset']]
        graphs=sorted((a,b,e) for a,b,e in es if e['name']=='cudaGraphLaunch')
        main_tid=collections.Counter(e['tid'] for a,b,e in graphs).most_common(1)
        if not main_tid:main_tid=collections.Counter(e['tid'] for a,b,e in es if e.get('cat')=='cpu_op').most_common(1)
        main_tid=main_tid[0][0] if main_tid else None
        near=lambda a,b:a<row['trigger']+.25 and b>row['trigger']-.03
        sync=[]
        for a,b,e in es:
            if 'Synchronize' in e['name'] and near(a,b):sync.append({'name':e['name'],'start_rel_ms':(a-row['trigger'])*1000,'duration_ms':(b-a)*1000,'tid':e['tid'],'is_main_thread':e['tid']==main_tid,'gpu_event_union_ms':overlap_union(kernels,a,b)*1000})
        gaps=[]
        for (a,b,e),(c,f,z) in zip(graphs,graphs[1:]):
            if near(a,c):gaps.append({'start_rel_ms':(a-row['trigger'])*1000,'duration_ms':(c-a)*1000,'gpu_event_union_ms':overlap_union(kernels,a,c)*1000})
        controls=[]
        for c in row['control_events']:
            if c['engine']!=['source','destination'][eng]:continue
            a,b=c['start'],c['finish'];controls.append({'phase':c['path'].split('/')[4]+'/'+c['response']['state'],'start_rel_ms':(a-row['trigger'])*1000,'duration_ms':(b-a)*1000,'main_thread_sync_ms':sum(max(0,min(b,y)-max(a,x))*1000 for x,y,e in es if 'Synchronize' in e['name'] and e['tid']==main_tid),'gpu_event_union_ms':overlap_union(kernels,a,b)*1000})
        res['engines'][str(eng)]={'main_tid':main_tid,'graph_count':len(graphs),'all_graph_gap':stat([(c-a)*1000 for (a,b,e),(c,f,z) in zip(graphs,graphs[1:])]),'trigger_graph_gaps':gaps,'trigger_sync_calls':sync,'control_envelopes':controls}
        if eng==(0 if row['target']=='source' else 1):
            for a,b,e in graphs:
                if near(a,b):ax.vlines((a-row['trigger'])*1000,0,1,color='black',lw=.8)
            for s in sync:
                if s['is_main_thread']:ax.broken_barh([(s['start_rel_ms'],s['duration_ms'])],(1.1,.5),facecolors='orange')
            for c in controls:ax.broken_barh([(c['start_rel_ms'],c['duration_ms'])],(1.7,.5),facecolors='steelblue',alpha=.7)
            # Union many GPU kernels before plotting to avoid huge SVG.
            merged=[]
            for a,b in sorted(kernels):
                if not near(a,b):continue
                if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
                else:merged.append((a,b))
            ax.broken_barh([((a-row['trigger'])*1000,(b-a)*1000) for a,b in merged],(2.3,.5),facecolors='seagreen')
    ax.set_title(f"{row['target']} / {row['mode']} (profiler enabled)");ax.set_yticks([.5,1.35,1.95,2.55],['graph launch','main sync','control RPC','GPU event union']);ax.set_xlim(-30,250)
    out.append(res)
axes[-1].set_xlabel('ms relative to copy/no-copy trigger');fig.tight_layout();fig.savefig(P/'TIMELINE.png',dpi=150);fig.savefig(P/'TIMELINE.svg')
(P/'ANALYSIS.json').write_text(json.dumps(out,indent=2)+'\n')
for r in out:
 e=r['engines'][str(0 if r['target']=='source' else 1)];print(r['index'],r['target'],r['mode'],e['all_graph_gap']);print('controls',e['control_envelopes']);print('top-sync',sorted(e['trigger_sync_calls'],key=lambda s:s['duration_ms'],reverse=True)[:3]);print('top-gap',sorted(e['trigger_graph_gaps'],key=lambda s:s['duration_ms'],reverse=True)[:2])
