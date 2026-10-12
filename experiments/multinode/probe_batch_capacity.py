"""Measure offered concurrency versus observed scheduler batch and queue.

The engine's configured 128-request ceiling is recorded, not called a silicon
limit. Stop after two consecutive points grow a queue without increasing peak
running batch or improving throughput by 5%, or after an explicit failed run.
"""
import argparse
import asyncio
import json
from pathlib import Path
import time

import aiohttp
import numpy as np

from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.multinode.benchmark_highbs import telemetry


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    corpus=np.load(args.tokens)['tokens'].tolist()[:args.length]
    reports=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900),connector=aiohttp.TCPConnector(limit=0, force_close=True)) as session:
        async with session.get(args.engine+'/get_server_info') as response:
            response.raise_for_status();info=await response.json()
        (args.output/'server_info.json').write_text(json.dumps(info,indent=2)+'\n')
        for regime in args.regimes:
            previous=None;plateaus=0
            for batch in args.batches:
                await flush(session,[args.engine])
                if regime=='shared-hot':await stream(session,args.engine,corpus,1)
                stop=asyncio.Event();samples=[];records=[];began=time.perf_counter()
                monitor=asyncio.create_task(telemetry(session,[args.engine],stop,samples,began))
                async def send(i):
                    tokens=list(corpus)
                    if regime=='unique-cold':tokens[0]=1000+i
                    tokens += [170000+((i*79+j*31)%40000) for j in range(32)]
                    try:
                        result=await stream(session,args.engine,tokens,args.output_tokens)
                        records.append({'id':i,'success':True,'result':result})
                    except Exception as e:records.append({'id':i,'success':False,'error':repr(e)})
                await asyncio.gather(*(send(i) for i in range(batch)))
                elapsed=time.perf_counter()-began
                stop.set();await monitor
                good=[r for r in records if r['success']]
                summary={'regime':regime,'offered':batch,'successful':len(good),'seconds':elapsed,
                         'requests_per_second':len(good)/elapsed,
                         'peak_running':max((s['running'] for s in samples if 'running' in s),default=0),
                         'peak_waiting':max((s['waiting'] for s in samples if 'waiting' in s),default=0),
                         'ttft_p50_p95_p99':np.percentile([r['result']['ttft_seconds'] for r in good],[50,95,99]).tolist() if good else None}
                path=args.output/f'{regime}-{batch}.json'
                path.write_text(json.dumps({'summary':summary,'requests':records,'telemetry':samples},indent=2)+'\n')
                reports.append(summary);print(json.dumps(summary),flush=True)
                if len(good)!=batch:
                    summary['stop_reason']='failed requests';break
                if previous and summary['peak_waiting']>0 and summary['peak_running']<=previous['peak_running'] and summary['requests_per_second']<=previous['requests_per_second']*1.05:
                    plateaus+=1
                else:plateaus=0
                previous=summary
                if plateaus>=2:
                    summary['stop_reason']='two consecutive queue/throughput plateaus';break
        (args.output/'summary.json').write_text(json.dumps({'scope':'empirical capacity under the saved engine/model/context/cache configuration, not an unconditional per-node hardware maximum','runs':reports},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--engine',default='http://12.11.3.29:31600')
    p.add_argument('--tokens',type=Path,default=Path('experiments/multinode/data/longbench_tokens.npz'))
    p.add_argument('--length',type=int,default=32768)
    p.add_argument('--output-tokens',type=int,default=64)
    p.add_argument('--batches',type=int,nargs='+',default=[1,4,16,32,64,128,256])
    p.add_argument('--regimes',nargs='+',default=['shared-hot','unique-cold'])
    p.add_argument('--output',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
