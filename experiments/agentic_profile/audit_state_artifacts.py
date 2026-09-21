"""Independent saved-artifact checks; no serving requests."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics


def audit(root):
    rows=[json.loads(l) for l in (root/'records.jsonl').read_text().splitlines()]
    manifest=json.loads((root/'manifest.json').read_text());config=json.loads((root/'config.json').read_text())
    fields=('foreground_target','concurrency','repetition','method','mode')
    key=lambda r:tuple(r[k] for k in fields)
    errors=[];counts=collections.Counter();groups=collections.defaultdict(dict)
    if collections.Counter(map(key,rows))!=collections.Counter(map(key,manifest)):errors.append('matrix mismatch')
    if len(set(map(key,rows)))!=len(rows):errors.append('duplicate conditions')
    for r in rows:
        index=r['index'];counts['conditions']+=1
        def check(ok,message):
            if not ok: errors.append({'index':index,'error':message})
        check(len(r['foreground'])==(0 if r['mode']=='preparation-only' else r['concurrency']),'foreground count')
        check(len(r['preparations'])==(0 if r['method']=='none' else 4),'preparation count')
        check(len(r['validation'])==(0 if r['method']=='none' else 4),'validation count')
        check(abs(r['charged_gpu_seconds']-2*(r['finish']-r['start']))<1e-8,'GPU budget')
        for i,f in enumerate(r['foreground']):
            counts['foreground']+=1
            check(f['meta_info']['prompt_tokens']==1024 and f['meta_info']['completion_tokens']==len(f['output_ids'])==256,'foreground length')
            check(f['input_sha256']==config['foreground_hashes'][i],'foreground input identity')
            groups[(r['foreground_target'],r['concurrency'],r['repetition'],i)][(r['method'],r['mode'])]=tuple(f['output_ids'])
        if r['foreground']:
            check(abs(statistics.median(f['first']-r['start'] for f in r['foreground'])-r['foreground_ready_to_first_p50_seconds'])<1e-8,'common ready first')
            check(abs(statistics.median(f['finish']-r['start'] for f in r['foreground'])-r['foreground_ready_to_finish_p50_seconds'])<1e-8,'common ready finish')
        if r['mode']=='serial':check(max(p['finish'] for p in r['preparations'])<=min(f['start'] for f in r['foreground']),'serial order')
        if r['mode']=='concurrent':check(min(p['start'] for p in r['preparations'])>=min(f['first'] for f in r['foreground']),'concurrent trigger')
        for i,p in enumerate(r['preparations']):
            counts[r['method']+'_preparations']+=1
            check(p['prefix_tokens']==8192 and p['input_sha256']==config['parent_hashes'][i],'preparation identity')
            if r['method']=='replay':
                check(p['reconstructed_tokens']==8192 and p['generation']['meta_info']['cached_tokens']==0,'replay actual cold work')
            else:
                check(p['transfer_bytes']>0,'copy bytes')
                evidence=[e for e in r['control_events'] if '/cleanup/' in e['path'] and p['migration_id'] in e['path']]
                check({e['engine'] for e in evidence}=={'source','destination'},'two-engine cleanup evidence')
                check(all(e.get('response',{}).get('state')=='CLEANED' for e in evidence),'explicit cleanup success')
        for v in r['validation']:
            counts['validation_pairs']+=1
            check(v['destination']['meta_info']['cached_tokens']==8192,'actual consumed checkpoint')
            for endpoint in ('source','destination'):
                child=v[endpoint]
                check(child['meta_info']['prompt_tokens']==8256 and child['meta_info']['completion_tokens']==len(child['output_ids'])==8,'child length')
            check(v['source']['output_ids']==v['destination']['output_ids'],'cached continuation parity')
        for phase in ('loads_before','loads_after'):
            check(all(not load['num_running_reqs'] and not load['num_waiting_reqs'] for endpoint in r[phase] for load in endpoint['loads']),'drained '+phase)
    drift=[{'key':k,'variants':len(v),'distinct_outputs':len(set(v.values()))} for k,v in groups.items() if len(set(v.values()))>1]
    warm=json.loads((root/'warmup.json').read_text())
    warm_shapes=collections.Counter((w['endpoint'],w['concurrency']) for w in warm)
    infos=json.loads((root/'server_info.json').read_text())
    flag_names=('model_path','dtype','mamba_ssm_dtype','attention_backend','linear_attn_backend','mamba_backend','mamba_scheduler_strategy','disable_cuda_graph','disable_overlap_schedule','max_total_tokens','max_mamba_cache_size','enable_deterministic_inference')
    flags={k:[i.get(k) for i in infos] for k in flag_names}
    unequal={k:v for k,v in flags.items() if v[0]!=v[1]}
    freeze=json.loads((root.parent/'state-source-freeze.json').read_text())
    frozen_match=hashlib.sha256(Path(freeze['source']).read_bytes()).hexdigest()==freeze['sha256']
    result={'counts':dict(counts),'errors':errors,'paired_foreground_keys':len(groups),'foreground_output_drift':drift,'warmup_shapes':[{ 'endpoint':k[0],'concurrency':k[1],'completed':v} for k,v in warm_shapes.items()],'backend_flags':flags,'unequal_backend_flags':unequal,'source_freeze_matches':frozen_match,'scope':'Flags and lengths checked independently from row success; foreground output equality across methods/modes separate from migrated-child equality.'}
    (root/'independent-artifact-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'errors':errors,'foreground_drift_keys':len(drift),'paired_keys':len(groups),'freeze_matches':frozen_match,'unequal_flags':unequal}))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args();audit(args.root)
