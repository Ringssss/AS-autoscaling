"""Offline descriptive analysis; RPC envelopes are not scheduler service time."""
import collections,gzip,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'experiments/agentic_profile/results/20260915'
OUT=ROOT/'experiments/local_pipeline/results/20260915'
def stats(values):
    return dict(zip(['count','p50_ms','p95_ms','p99_ms','max_ms'],[len(values),*map(float,np.percentile(values,[50,95,99,100]))])) if values else {'count':0}
def readrows(p): return [json.loads(x) for x in p.read_text().splitlines() if x]
def phase(e):
    action=e['path'].split('/')[4]
    return '/'.join([e['engine'],action,e.get('response',{}).get('state','unknown')])
result={'scope':'Saved traces only, PP1. RPC envelopes include queue/network/client; no causal or profiler-off gap claim.','datasets':{},'profile_rows':[]}
for name in ['state-main','state-profile']:
    rows=readrows(OLD/name/'records.jsonl'); groups=collections.defaultdict(list)
    for row in rows:
        if row['method']!='copy':continue
        for e in row.get('control_events',[]):
            groups[f"{row['foreground_target']}/c{row['concurrency']}/{row['mode']}/{phase(e)}"].append((e['finish']-e['start'])*1000)
    result['datasets'][name]={'rows':len(rows),'rpc_ms':{k:stats(v) for k,v in groups.items()}}
anchor=json.loads((OLD/'state-profile/PROFILE_ANALYSIS.json').read_text())['clock_anchor']
for row in readrows(OLD/'state-profile/records.jsonl'):
    report={'method':row['method'],'engines':{}}
    for engine in [0,1]:
        path=next((OLD/'state-profile'/f"profile-{row['index']}-engine-{engine}").glob('*.gz'))
        trace=json.load(gzip.open(path)); base=trace['baseTimeNanoseconds']/1e9-anchor['epoch']+anchor['monotonic']
        events=[]
        for e in trace['traceEvents']:
            if e.get('ph')!='X':continue
            start=base+e['ts']/1e6; end=start+e.get('dur',0)/1e6
            if row['start']<=start<=row['finish']:events.append((start,end,e))
        launches=sorted((a,b,e) for a,b,e in events if e['name']=='cudaGraphLaunch')
        gaps=[]
        for (a,b,e),(c,d,f) in zip(launches,launches[1:]):
            overlaps=[{'phase':phase(x),'rpc_start':x['start'],'rpc_finish':x['finish']} for x in row.get('control_events',[]) if x['engine']==('source' if engine==0 else 'destination') and x['start']<c and x['finish']>a]
            gaps.append({'start':a,'finish':c,'start_to_start_ms':(c-a)*1000,'rpc_overlap':overlaps})
        sync=collections.defaultdict(list)
        for a,b,e in events:
            if 'Synchronize' in e['name']:sync[e['name']].append(e.get('dur',0)/1000)
        report['engines'][str(engine)]={'launch_count':len(launches),'graph_launch_start_to_start':stats([g['start_to_start_ms'] for g in gaps]),'largest_graph_gaps':sorted(gaps,key=lambda x:x['start_to_start_ms'],reverse=True)[:10],'sync_api':{k:stats(v) for k,v in sync.items()}}
    result['profile_rows'].append(report)
result['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('/data-aisoft/zhujianian/cloud/sglang-hybrid-state/python/sglang/srt/mem_cache/hybrid_state_transfer.py'),Path('/data-aisoft/zhujianian/cloud/sglang-hybrid-state/python/sglang/srt/managers/scheduler.py')]}
OUT.mkdir(exist_ok=True,parents=True)
(OUT/'offline-review.json').write_text(json.dumps(result,indent=2)+'\n')
for r in result['profile_rows']:
 print(r['method'],[(k,v['graph_launch_start_to_start']) for k,v in r['engines'].items()])
for k,v in result['datasets']['state-profile']['rpc_ms'].items():print(k,v)
