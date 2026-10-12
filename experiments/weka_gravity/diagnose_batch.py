"""Check output changes on one engine, with no migration or node change."""
import asyncio
import json
from pathlib import Path
import random
import aiohttp
import numpy as np
from experiments.fluidkv.benchmark_native_pairs import stream, flush


async def run():
    root=Path(__file__).parent/'results'
    out=root/'batch-diagnosis'
    out.mkdir(exist_ok=False)
    corpus=np.load(root/'component_tokens.npz')['tokens'].tolist()
    parents=[[20001+i]+corpus[:32767] for i in range(3)]
    children=[p+corpus[40000:40032] for p in parents]
    url='http://12.11.3.27:31720'
    rng=random.Random(20260928)
    rows=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        with (out/'requests.jsonl').open('w',buffering=1) as log:
            for repeat in range(2):
                for mode in rng.sample(['warm-single','warm-batch3','cold-single'],3):
                    await flush(session,[url])
                    if mode!='cold-single':
                        for p in parents:await stream(session,url,p,1)
                    if mode=='warm-batch3':
                        results=await asyncio.gather(*(stream(session,url,c,32) for c in children))
                        result=results[1]
                    else:
                        result=await stream(session,url,children[1],32)
                    row=dict(repeat=repeat,mode=mode,url=url,result=result)
                    rows.append(row);log.write(json.dumps(row)+'\n')
                    print(repeat,mode,result['output_ids'][:8],flush=True)
    previous=[json.loads(s) for s in (root/'three-node-burst/requests.jsonl').read_text().splitlines()]
    ref=next(r for r in previous if r['mode']=='copy-in-gap')['agents'][1]['generation']['output_ids']
    summary=dict(scope=__doc__,no_migration_performed=True,diagnostic_subject='history index 1',
        reference_output=ref,
        cases=[dict(repeat=r['repeat'],mode=r['mode'],matches_distributed_reference=r['result']['output_ids']==ref,
                    output_ids=r['result']['output_ids'],cached_tokens=r['result']['meta_info']['cached_tokens']) for r in rows])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':asyncio.run(run())
