#!/usr/bin/env python3
"""Short optional engine traces; these runs never serve as performance claims."""
import argparse,asyncio,json,time
from pathlib import Path
import aiohttp
from experiments.agentic_profile.run_profile_campaign import generate,pin_tree

async def main(a):
 a.output.mkdir(parents=True,exist_ok=False)
 pin_tree(a.pid,list(range(16)))
 results=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as s:
  ids=[31000]+[1700+i%200 for i in range(2047)]
  await generate(s,a.engine,ids,64,[],'warm')
  for index,mode in enumerate(a.modes):
   async with s.post(a.engine+'/flush_cache') as r:r.raise_for_status();await r.read()
   if mode!='off':
    activities=['CPU','GPU'] if mode=='torch' else ['CUDA_PROFILER']
    async with s.post(a.engine+'/start_profile',json={'activities':activities,'output_dir':str(a.output.resolve()),'with_stack':False,'record_shapes':False,'profile_prefix':f'{mode}-{index}'}) as r:r.raise_for_status();await r.read()
   events=[];start=time.perf_counter();epoch=time.time()
   batch=await asyncio.gather(*(generate(s,a.engine,[31000+i]+ids[1:],a.tokens,events,str(i)) for i in range(a.concurrency)))
   elapsed=time.perf_counter()-start
   if mode!='off':
    async with s.post(a.engine+'/stop_profile') as r:r.raise_for_status();await r.read()
   results.append({'mode':mode,'elapsed':elapsed,'batch':batch,'events':events,'epoch':epoch,'monotonic':start})
   (a.output/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
   print(json.dumps({'mode':mode,'elapsed':elapsed,'ttft':[x['ttft'] for x in batch]}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--pid',type=int,required=True);p.add_argument('--engine',default='http://127.0.0.1:31480');p.add_argument('--modes',nargs='+',default=['off','torch','off']);p.add_argument('--tokens',type=int,default=64);p.add_argument('--concurrency',type=int,default=2);asyncio.run(main(p.parse_args()))
