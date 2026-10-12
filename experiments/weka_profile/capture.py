"""Representative AgentX captures; profiling timings are never A/B timings."""
import asyncio
import hashlib
import json
from pathlib import Path
import time
import aiohttp
import numpy as np
from experiments.weka_baselines.replay import generate,write
from experiments.fluidkv.benchmark_native_pairs import flush

ROOT=Path(__file__).parent
OUT=ROOT/'results'
URL='http://12.11.3.27:31760'


async def run():
    rs=[json.loads(s) for s in (ROOT.parent/'weka_baselines/results/one-hour-20260929/selected-requests.jsonl').read_text().splitlines()]
    data=np.load(ROOT.parent/'weka_baselines/data/tokens.npy',mmap_mode='r')
    cold=next(r for r in rs if r['input_tokens']==116480 and r['output_tokens']==702)
    warm=next(r for r in rs if r['input_tokens']==57920 and r['output_tokens']==598)
    previous=next(r for r in rs if r['key']==warm['key'] and r['turn']==warm['turn']-1)
    def ids(r):return data[r['token_offset']:r['token_offset']+r['input_tokens']].tolist()
    cases=[('cold-critical',cold,None),('warm-critical',warm,previous)]
    write(OUT/'workloads.json',dict(cases=[dict(name=n,row={k:v for k,v in r.items() if k!='hash_ids'},
        parent={k:v for k,v in p.items() if k!='hash_ids'} if p else None) for n,r,p in cases],
        note='Exact selected AgentX symbol-token inputs and full recorded output lengths. Short warmup is separate. No production policy ranking from profiler durations.'))
    records=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900,sock_read=180)) as session:
        deadline=time.monotonic()+600
        while True:
            try:
                async with session.get(URL+'/health',timeout=aiohttp.ClientTimeout(total=5)) as r:
                    if r.status==200:break
            except Exception:pass
            if time.monotonic()>deadline:raise TimeoutError('readiness')
            await asyncio.sleep(5)
        async with session.get(URL+'/get_server_info') as r:r.raise_for_status();write(OUT/'server-info.json',await r.json())
        for name,row,parent in cases:
            token_ids=ids(row)
            await flush(session,[URL])
            if parent:await generate(session,URL,ids(parent),1)
            await generate(session,URL,token_ids,32) # shape warmup
            for mode in ['unprofiled-before','profiled','unprofiled-after']:
                await asyncio.sleep(.3);await flush(session,[URL])
                if parent:await generate(session,URL,ids(parent),1)
                if mode=='profiled':
                    payload=dict(output_dir='/profiles/'+name,activities=['CPU','GPU'],profile_by_stage=True,
                        num_steps=5,with_stack=True,record_shapes=True,profile_prefix=name)
                    async with session.post(URL+'/start_profile',json=payload) as r:
                        r.raise_for_status();ack=await r.text()
                    write(OUT/(name+'-profile-request.json'),dict(request=payload,response=ack))
                print('START',name,mode,flush=True)
                result=await generate(session,URL,token_ids,row['output_tokens'])
                records.append(dict(case=name,mode=mode,result=result))
                write(OUT/'records.json',records)
                print('DONE',name,mode,'ttft',result['ttft'],'tpot',result['mean_tpot'],flush=True)
        # A separate mixed-load capture retains the representative long context
        # with three independent 8K backgrounds. Outputs are controlled at 64
        # here to expose interference without labeling it native full replay.
        await asyncio.sleep(.3);await flush(session,[URL])
        long_ids=ids(warm);await generate(session,URL,ids(previous),1)
        async with session.post(URL+'/start_profile',json=dict(output_dir='/profiles/mixed',activities=['CPU','GPU'],
            profile_by_stage=True,num_steps=5,with_stack=True,record_shapes=True,profile_prefix='mixed')) as r:r.raise_for_status();await r.read()
        mixed=await asyncio.gather(generate(session,URL,long_ids,64),*(generate(session,URL,[21000+i]+list(range(1000,9191)),64) for i in range(3)))
        write(OUT/'mixed.json',dict(results=mixed,scope='Controlled 4-request mixed profile; 64-token outputs, not full native replay.'))
    print('CAPTURE_COMPLETE',flush=True)


if __name__=='__main__':asyncio.run(run())
