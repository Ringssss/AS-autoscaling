"""Preregister structural strata without consulting any performance outcomes."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.weka_gravity.audit import flatten
from experiments.fluidkv.p07_prepare_cohorts import lcp


def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    previous=json.loads(args.exclude.read_text())
    excluded={c['conversation'] for c in previous['cohorts']}
    best={n:{} for n in args.streams}
    counts=defaultdict(int)
    digest=hashlib.sha256()
    for line,raw in enumerate(args.trace.open('rb'),1):
        digest.update(raw)
        trace=json.loads(raw)
        rows,_=flatten(trace)
        counts['conversations']+=1
        counts['requests']+=len(rows)
        if trace['id'] in excluded:
            counts['excluded_prior_conversations']+=1
            continue
        streams=defaultdict(list)
        for row in rows:
            streams[row['stream'],row['model']].append(row)
        eligible={}
        for rs in streams.values():
            for j in range(1,len(rs)-2):
                fragment=rs[j-1:j+3]
                if not all(32768<=r['in']<=110000 and r['out']>0
                           and r['in']+min(r['out'],512)<=131072 for r in fragment):
                    continue
                if any(b['t']<a['t']+a['api_time']-1e-6
                       or lcp(a['hash_ids'],b['hash_ids'])<32768
                       for a,b in zip(fragment,fragment[1:])):
                    continue
                eligible[rs[j]['location']]=fragment
        active=defaultdict(list)
        for r in rows:
            model=r['model']
            active[model]=[x for x in active[model] if x['t']+x['api_time']>r['t']]
            active[model].append(r)
            pool=list({x['stream']:x for x in active[model]
                if x['location'] in eligible}.values())
            for n in args.streams:
                if len(pool)<n:
                    continue
                chosen=sorted(pool,key=lambda x:(x['t'],x['location']))[-n:]
                fragments=[eligible[x['location']] for x in chosen]
                parents=[f[0]['hash_ids'] for f in fragments]
                common=min(lcp(parents[0],h) for h in parents)
                unique=len({h for f in fragments for x in f for h in x['hash_ids']})*64
                spread=max(x['t'] for x in chosen)-min(x['t'] for x in chosen)
                if common<32768 or unique>220000 or spread>20:
                    continue
                counts[f'eligible_windows_n{n}']+=1
                tag=':'.join([trace['id'],model]+[x['location'] for x in chosen])
                rank=hashlib.sha256((str(args.seed)+':'+tag).encode()).hexdigest()
                c=dict(rank=rank,line=line,conversation=trace['id'],model=model,
                    common_parent_tokens=common,unique_input_tokens=unique,
                    first_arrival_spread=spread,fragments=fragments)
                old=best[n].get(trace['id'])
                if old is None or rank<old['rank']:
                    best[n][trace['id']]=c
        if line%100==0:
            print('scan',line,{n:len(v) for n,v in best.items()},flush=True)
    selected=[]
    parts=[]
    offset=0
    for n,pool in best.items():
        counts[f'eligible_conversations_n{n}']=len(pool)
        for index,c in enumerate(sorted(pool.values(),key=lambda x:x['rank'])[:args.per_stratum]):
            cache={}
            namespace=c['conversation']+'/'+c['model']
            fragments=c.pop('fragments')
            first=min(f[1]['t'] for f in fragments)
            saved_streams=[]
            for fragment in fragments:
                requests=[]
                for row in fragment:
                    blocks=[]
                    for h in row['hash_ids']:
                        if h not in cache:
                            seed=int.from_bytes(hashlib.sha256(f'{namespace}:{h}'.encode()).digest()[:8],'little')
                            cache[h]=np.random.default_rng(seed).integers(1000,240000,64,dtype=np.int32)
                        blocks.append(cache[h])
                    ids=np.concatenate(blocks)
                    assert len(ids)==row['in']
                    parts.append(ids)
                    requests.append(dict(location=row['location'],native_t=row['t'],native_api_time=row['api_time'],
                        input_tokens=row['in'],native_output_tokens=row['out'],output_tokens=min(row['out'],128),
                        token_offset=offset,hash_ids=row['hash_ids'],
                        token_sha256=hashlib.sha256(ids.tobytes()).hexdigest()))
                    offset+=len(ids)
                saved_streams.append(dict(key=fragment[0]['stream'],requests=requests,
                    first_offset=fragment[1]['t']-first,
                    gaps=[max(0.,b['t']-a['t']-a['api_time']) for a,b in zip(fragment[1:],fragment[2:])]))
            selected.append(dict(c,name=f'holdout-n{n}-{index}',streams=saved_streams))
    if not selected:
        raise RuntimeError('no eligible holdout cohorts')
    np.save(args.out/'cohort-tokens.npy',np.concatenate(parts))
    report=dict(source=str(args.trace),sha256=digest.hexdigest(),seed=args.seed,
        counts=dict(counts),excluded_conversations=sorted(excluded),cohorts=selected,
        scope='Outcome-independent held-out conversation selection by SHA256 rank in 1/2/4/6-stream strata. '
        'Long shared-input/reuse structural slice, not random full-corpus serving. '
        'Conversation-local identities, model separation, input lengths, initial relative arrivals and '
        'completion-to-next-arrival gaps retained. Synthetic token contents, substitute Qwen3.6 model, '
        'capped outputs; no semantic task execution or recovered worker placements. '
        'API lifetime overlap used only for selecting cohorts, never as measured GPU queue load.')
    (args.out/'cohorts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(counts=counts,selected=[dict(name=c['name'],conversation=c['conversation'])
        for c in selected]),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',type=Path,default=Path('/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl'))
    p.add_argument('--exclude',type=Path,default=Path('results/routing-space-20261010/cohorts.json'))
    p.add_argument('--streams',nargs='+',type=int,default=[1,2,4,6])
    p.add_argument('--per-stratum',type=int,default=2)
    p.add_argument('--seed',type=int,default=20261011)
    p.add_argument('--out',type=Path,required=True)
    run(p.parse_args())
