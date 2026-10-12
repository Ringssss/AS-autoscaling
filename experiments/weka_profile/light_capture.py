"""Verify rank skew without Python stack/shape collection; not an A/B run."""
import asyncio
import json
from pathlib import Path
import aiohttp
import numpy as np
from experiments.weka_baselines.replay import generate,write
from experiments.fluidkv.benchmark_native_pairs import flush

ROOT=Path(__file__).parent
URL='http://12.11.3.27:31760'


async def run():
    rows=[json.loads(s) for s in (ROOT.parent/'weka_baselines/results/one-hour-20260929/selected-requests.jsonl').read_text().splitlines()]
    row=next(r for r in rows if r['input_tokens']==57920 and r['output_tokens']==598)
    a=np.load(ROOT.parent/'weka_baselines/data/tokens.npy',mmap_mode='r');ids=a[row['token_offset']:row['token_offset']+row['input_tokens']].tolist()
    results=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        await asyncio.sleep(.3);await flush(session,[URL]);await generate(session,URL,ids,16)
        for mode in ['off-before','light-profile','off-after']:
            if mode=='light-profile':
                async with session.post(URL+'/start_profile',json=dict(output_dir='/profiles/light',activities=['CPU','GPU'],
                    profile_by_stage=True,num_steps=8,with_stack=False,record_shapes=False,profile_prefix='light')) as r:r.raise_for_status();await r.read()
            result=await generate(session,URL,ids+[180001],64)
            results.append(dict(mode=mode,result=result));write(ROOT/'results/light-records.json',results)
    print('LIGHT_CAPTURE_COMPLETE',flush=True)


if __name__=='__main__':asyncio.run(run())
