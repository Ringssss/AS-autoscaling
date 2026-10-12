"""Deterministic native-hash, original-length AgentX trajectory cohort."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from experiments.weka_gravity.audit import flatten

ROOT=Path(__file__).parent
OUT=ROOT/'data'
TRACE=Path('/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl')


def token_block(namespace, block_id):
    # No special tokens; equal native prefix IDs yield equal 64-token blocks.
    seed=int.from_bytes(hashlib.sha256(f'{namespace}:{block_id}'.encode()).digest()[:8],'little')
    return np.random.default_rng(seed).integers(1000,240000,64,dtype=np.int32)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    candidates=[];digest=hashlib.sha256();total=fit=0
    with TRACE.open('rb') as f:
        for line,raw in enumerate(f,1):
            digest.update(raw);trace=json.loads(raw)
            flat,_=flatten(trace);streams=defaultdict(list)
            for r in flat:
                total+=1
                valid=r['in']+r['out']<=131072 and r['out']>0
                fit+=valid
                streams[r['stream'],r['model']].append((r,valid))
            for (stream,model),items in streams.items():
                window=[]
                for r,valid in items:
                    # Do not invent serial dependency for overlapping calls.
                    if not valid or (window and r['t']<window[-1]['t']+window[-1]['api_time']-1e-6):
                        window=[]
                    if valid:window.append(r)
                    if len(window)==9:
                        key=f"{trace['id']}/{stream}/{model}"
                        candidates.append(dict(key=key,namespace=trace['id']+'/'+model,line=line,
                            conversation=trace['id'],stream=stream,model=model,requests=window.copy(),
                            selection_hash=hashlib.sha256(('20260928:'+key).encode()).hexdigest()))
                        break
            if line%50==0:print('scanned',line,'candidate streams',len(candidates),flush=True)
    candidates.sort(key=lambda c:c['selection_hash'])
    assert len(candidates)>=128,len(candidates)
    chosen=candidates[:128];records=[];tokens=[];offset=0
    for index,c in enumerate(chosen):
        cache={}
        for turn,r in enumerate(c.pop('requests')):
            arrays=[]
            for h in r['hash_ids']:
                if h not in cache:cache[h]=token_block(c['namespace'],h)
                arrays.append(cache[h])
            ids=np.concatenate(arrays)
            assert len(ids)==r['in']
            # Fixed-width decimal keys avoid upstream join ambiguity; expand
            # 64-token native blocks into four 16-token adapter blocks.
            hashes=[]
            for h in r['hash_ids']:
                for part in range(4):
                    key=int.from_bytes(hashlib.sha256(f"{c['namespace']}:{h}:{part}".encode()).digest()[:8],'little')
                    hashes.append(100000000000000000+key%900000000000000000)
            row=dict(stream_index=index,turn=turn,key=c['key'],conversation=c['conversation'],
                     model=c['model'],line=c['line'],location=r['location'],native_t=r['t'],
                     native_api_time=r['api_time'],input_tokens=r['in'],output_tokens=r['out'],
                     token_offset=offset,token_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),hash_ids=hashes)
            records.append(row);tokens.append(ids);offset+=len(ids)
    np.save(OUT/'tokens.npy',np.concatenate(tokens))
    with (OUT/'requests.jsonl').open('w') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    manifest=dict(trace=str(TRACE),trace_sha256=digest.hexdigest(),total_native_requests=total,
        context_and_positive_output_fit=fit,candidate_streams=len(candidates),selected_streams=128,turns_per_stream=9,
        selected_requests=len(records),selected_output_tokens=sum(r['output_tokens'] for r in records),
        output_cap=None,input_cap=None,streams=chosen,
        scope='Seeded feasible contiguous substreams, not full AgentX corpus or recovered semantics. '
              'Same-model/conversation prefix identities and original input/output lengths retained. '
              'Cross-stream arrival schedules are controlled because absolute conversation start times are absent.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='streams'},indent=2),flush=True)


if __name__=='__main__':main()
