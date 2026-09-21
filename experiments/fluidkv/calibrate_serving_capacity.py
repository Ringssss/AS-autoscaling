#!/usr/bin/env python3
"""Measured decode-capacity controls; synthetic input, not trace throughput."""
import argparse
import asyncio
import json
import time
from pathlib import Path

import aiohttp
import numpy as np

from experiments.fluidkv.benchmark_native_pairs import stream, flush


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180), connector=aiohttp.TCPConnector(limit=0)) as session:
        infos = []
        for url in args.engines:
            async with session.get(url+'/get_server_info') as response:
                response.raise_for_status()
                infos.append(await response.json())
            await stream(session, url, list(range(1000,1128)), 8)
        (args.output/'server_info.json').write_text(json.dumps(infos,indent=2)+'\n')
        for repetition in range(args.repetitions):
            for length in args.lengths:
                for concurrency in args.concurrency:
                    await flush(session,args.engines)
                    started = time.perf_counter()
                    async def one(engine, ordinal):
                        tokens = [1000+((i*17+ordinal*2371+engine*1999+repetition*10001)%100000) for i in range(length)]
                        result = await stream(session,args.engines[engine],tokens,args.outputs)
                        if result['meta_info']['completion_tokens'] != args.outputs or result['meta_info']['cached_tokens']:
                            raise RuntimeError('capacity control changed output count or unexpectedly hit cache')
                        return {'engine':engine,'ordinal':ordinal,**result}
                    rows=await asyncio.gather(*(one(engine,ordinal) for engine in range(len(args.engines)) for ordinal in range(concurrency)))
                    elapsed=time.perf_counter()-started
                    record={'repetition':repetition,'input_tokens':length,'per_engine_concurrency':concurrency,
                        'outputs_per_request':args.outputs,'elapsed_seconds':elapsed,
                        'aggregate_output_tokens_per_second':sum(r['meta_info']['completion_tokens'] for r in rows)/elapsed,
                        'ttft_p50_p90_seconds':np.percentile([r['ttft_seconds'] for r in rows],[50,90]).tolist(),'requests':rows}
                    results.append(record)
                    with (args.output/'records.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                    print(json.dumps({k:v for k,v in record.items() if k!='requests'}),flush=True)
        await flush(session,args.engines)
    (args.output/'summary.json').write_text(json.dumps({'scope':'Synthetic finite-batch capacity calibration, not sustainable trace goodput.','runs':results},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--engines',nargs='+',default=['http://127.0.0.1:31480','http://127.0.0.1:31481','http://127.0.0.1:31482'])
    p.add_argument('--lengths',nargs='+',type=int,default=[1024,4096])
    p.add_argument('--concurrency',nargs='+',type=int,default=[1,4,8])
    p.add_argument('--outputs',type=int,default=64)
    p.add_argument('--repetitions',type=int,default=1)
    asyncio.run(run(p.parse_args()))
