#!/usr/bin/env python3
"""Matched real-tool serving CPU-isolation experiment (constructed workflow)."""
import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import time

import aiohttp
import psutil

TOKENIZER = None


def worker_init(cpus, model, ready, tool_spec=None):
    global TOKENIZER
    os.sched_setaffinity(0, set(cpus))
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    if model:
        from transformers import AutoTokenizer
        TOKENIZER = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)
        TOKENIZER.encode('Warm tokenizer before timing.', add_special_tokens=False)
    if tool_spec:
        from experiments.agentic_profile.cpu_workloads import execute_tool
        result=execute_tool(tool_spec)
        if not result['success']: raise RuntimeError(result['error'])
    for thread in psutil.Process().threads():
        os.sched_setaffinity(thread.id,set(cpus))
    ready.put({'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0)), 'thread_affinity':{str(t.id):sorted(os.sched_getaffinity(t.id)) for t in psutil.Process().threads()}})


def tool_job(spec):
    from experiments.agentic_profile.cpu_workloads import execute_tool
    return execute_tool(spec)


def context_job(base_ids, output_ids, tool_text):
    started = time.perf_counter(); cpu = time.process_time()
    suffix = '\nTool execution result:\n' + tool_text + '\nSummarize the concrete findings concisely.\nAssistant:'
    # Preserve generated token identity and include the complete bounded tool output.
    suffix_ids = TOKENIZER.encode(suffix, add_special_tokens=False)
    ids = base_ids + output_ids + suffix_ids
    return {'input_ids': ids, 'input_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            'tool_sha256': hashlib.sha256(tool_text.encode()).hexdigest(),
            'tool_bytes': len(tool_text.encode()), 'suffix_tokens':len(suffix_ids),
            'start': started, 'finish':time.perf_counter(), 'cpu_seconds':time.process_time()-cpu}


def observer_job(pids, output, stop, ready):
    from experiments.agentic_profile.telemetry import Sampler
    sampler=Sampler(pids,output,interval=1.0,include_gpu=True,gpu_indices=[0])
    sampler.start();ready.set()
    try:stop.wait()
    finally:sampler.stop()


def ping():
    return {'pid':os.getpid(),'affinity':sorted(os.sched_getaffinity(0))}


def pin_tree(pid, cpus):
    root = psutil.Process(pid)
    records = []
    for p in [root,*root.children(recursive=True)]:
        try:
            exceptions=[]
            for tid in p.threads():
                try: os.sched_setaffinity(tid.id,set(cpus))
                except ProcessLookupError: pass
                except OSError as exc:
                    name=Path(f'/proc/{p.pid}/task/{tid.id}/comm').read_text().strip()
                    if exc.errno!=22 or not name.startswith('iou-sqp-'):raise
                    exceptions.append({'tid':tid.id,'name':name,'cpu_seconds':tid.user_time+tid.system_time,'reason':'kernel io_uring SQ poller rejects sched_setaffinity; separately recorded'})
            records.append({'pid':p.pid,'affinity':p.cpu_affinity(),'threads':len(p.threads()),'affinity_exceptions':exceptions,'thread_affinity':{str(t.id):sorted(os.sched_getaffinity(t.id)) for t in p.threads()}})
        except psutil.NoSuchProcess: pass
    return records


async def generate(session,url,ids,n,events,rid):
    start=time.perf_counter(); first=None; last_count=0; emissions=[]; final=None
    events.append({'name':'request_start','ts':start,'rid':rid,'input_tokens':len(ids)})
    async with session.post(url+'/generate',json={'input_ids':ids,'stream':True,
            'sampling_params':{'temperature':0,'ignore_eos':True,'max_new_tokens':n}}) as response:
        response.raise_for_status()
        while True:
            raw=await response.content.readline()
            if not raw:break
            if not raw.startswith(b'data:'):continue
            raw=raw[5:].strip()
            if raw==b'[DONE]':continue
            data=json.loads(raw);count=data.get('meta_info',{}).get('completion_tokens',len(data.get('output_ids',[])))
            if count>last_count:
                now=time.perf_counter();first=first or now;emissions.append([now,count-last_count]);last_count=count
            final=data
    finish=time.perf_counter()
    if first is None or final is None:raise RuntimeError('missing generated tokens')
    meta=final['meta_info']
    if meta['prompt_tokens']!=len(ids) or meta['completion_tokens']!=n:raise RuntimeError('input/output length changed')
    gaps=[emissions[i][0]-emissions[i-1][0] for i in range(1,len(emissions))]
    events.append({'name':'request','ts':start,'dur':finish-start,'rid':rid,'first':first})
    return {'start':start,'first':first,'finish':finish,'ttft':first-start,'e2e':finish-start,
            'output_ids':final['output_ids'],'cached_tokens':meta['cached_tokens'],
            'input_tokens':len(ids),'output_tokens':n,'emissions':emissions,
            'mean_decode_seconds_per_token':(finish-first)/(n-1) if n>1 else None,
            'stream_gap_p90':percentile(gaps,90) if gaps else None}


def percentile(a,p):
    if not a:return None
    a=sorted(a);k=(len(a)-1)*p/100;l=int(k);h=min(l+1,len(a)-1)
    return a[l]+(a[h]-a[l])*(k-l)


async def run_one(args,policy,concurrency,rep,manifest,base_ids,precomputed,folder):
    folder.mkdir(parents=True,exist_ok=False)
    cpus=list(range(args.cpu_start,args.cpu_start+args.cores));split=2 if policy=='frontend-reserved2' else args.cores//2
    engine_cpus=cpus if policy in ('shared','precomputed') else cpus[:split]
    tool_cpus=cpus if policy in ('shared','precomputed') else cpus[split:]
    context_cpus=cpus[split:] if policy in ('frontend-isolated','frontend-reserved2') else engine_cpus
    affinity=pin_tree(args.engine_pid,engine_cpus)
    ctx=multiprocessing.get_context('spawn')
    spec={'operation':'ast','manifest':manifest,'passes':args.tool_passes,'max_output_chars':args.tool_output_chars}
    tool_ready=ctx.Queue();context_ready=ctx.Queue()
    tool_pool=concurrent.futures.ProcessPoolExecutor(max_workers=args.tool_workers,mp_context=ctx,initializer=worker_init,initargs=(tool_cpus,None,tool_ready,spec))
    context_pool=concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=ctx,initializer=worker_init,initargs=(context_cpus,args.model,context_ready))
    loop=asyncio.get_running_loop();events=[];records=[]
    try:
        await asyncio.gather(*[loop.run_in_executor(tool_pool,ping) for _ in range(args.tool_workers)],*[loop.run_in_executor(context_pool,ping) for _ in range(2)])
        pings=[await asyncio.to_thread(tool_ready.get,True,120) for _ in range(args.tool_workers)]
        pings.extend([await asyncio.to_thread(context_ready.get,True,120) for _ in range(2)])
        if len({p['pid'] for p in pings})!=args.tool_workers+2:raise RuntimeError('worker readiness identity mismatch')
        # Finish startup and tokenizer warmup before timing.
        warm_context=await loop.run_in_executor(context_pool,context_job,base_ids,[],precomputed['text'])
        spec={'operation':'ast','manifest':manifest,'passes':args.tool_passes,'max_output_chars':args.tool_output_chars}
        if policy!='precomputed':
            warm_tool=await loop.run_in_executor(tool_pool,tool_job,spec)
            if not warm_tool['success']:raise RuntimeError(warm_tool)
        from experiments.agentic_profile.telemetry import Sampler
        sampler=None
        timeout=aiohttp.ClientTimeout(total=180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(args.engine+'/flush_cache') as r:r.raise_for_status();await r.read()
            # Warm graph path, then clear cache so all policies begin equally cold.
            await generate(session,args.engine,base_ids,args.output_tokens,[], 'warm')
            async with session.post(args.engine+'/flush_cache') as r:r.raise_for_status();await r.read()
            root_pids=[args.engine_pid,os.getpid()]+list(tool_pool._processes)+list(context_pool._processes)
            # Telemetry public API is supplied by the independent observer module.
            observer_stop=ctx.Event();observer_ready=ctx.Event()
            observer=ctx.Process(target=observer_job,args=(root_pids,folder/'telemetry.jsonl',observer_stop,observer_ready))
            observer.start()
            if not await asyncio.to_thread(observer_ready.wait,30):raise RuntimeError('observer did not start')
            begin=time.perf_counter();epoch=time.time();gate=asyncio.Semaphore(concurrency)
            async def agent(index):
                offered=begin+index*args.arrival_interval
                await asyncio.sleep(max(0,offered-time.perf_counter()))
                row={'agent':index,'policy':policy,'repetition':rep,'concurrency':concurrency,'offered':offered,'success':False}
                try:
                    async with gate:
                        row['admitted']=time.perf_counter()
                        # Same token count; unique first token prevents accidental across-agent prefix reuse.
                        ids=[30000+index]+base_ids[1:]
                        first=await generate(session,args.engine,ids,args.output_tokens,events,f'{index}-first')
                        row['first']=first
                        queued=time.perf_counter()
                        if policy=='precomputed':tool=precomputed
                        else:tool=await loop.run_in_executor(tool_pool,tool_job,spec)
                        returned=time.perf_counter()
                        if not tool['success']:raise RuntimeError(tool.get('error'))
                        row.update(tool=tool,tool_queued=queued,tool_returned=returned)
                        context_queued=time.perf_counter()
                        context=await loop.run_in_executor(context_pool,context_job,ids,first['output_ids'],tool['text'])
                        context_returned=time.perf_counter()
                        row.update(context=context,context_queued=context_queued,context_returned=context_returned)
                        if len(context['input_ids'])+args.output_tokens>32766:raise RuntimeError('context overflow')
                        second=await generate(session,args.engine,context['input_ids'],args.output_tokens,events,f'{index}-second')
                        row.update(success=True,first=first,second=second,tool=tool,tool_queued=queued,tool_returned=returned,
                                   context=context,context_queued=context_queued,context_returned=context_returned,
                                   finish=time.perf_counter())
                        row['e2e']=row['finish']-offered
                        events.extend([{'name':'tool_total','ts':queued,'dur':returned-queued,'rid':str(index)},
                                       {'name':'context_total','ts':context_queued,'dur':context_returned-context_queued,'rid':str(index)}])
                except Exception as exc:row.update(error=repr(exc),finish=time.perf_counter())
                records.append(row)
                with (folder/'requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            await asyncio.gather(*(agent(i) for i in range(args.agents)))
            elapsed=time.perf_counter()-begin
            observer_stop.set();await asyncio.to_thread(observer.join,30)
            if observer.is_alive() or observer.exitcode!=0:raise RuntimeError('observer failed to finish')
            successes=[r for r in records if r['success']]
            gen=[r[k] for r in successes for k in ('first','second')]
            summary={'policy':policy,'concurrency':concurrency,'repetition':rep,'offered':args.agents,'successful':len(successes),
                'seconds':elapsed,'agent_goodput':len(successes)/elapsed,'e2e_p50':percentile([r['e2e'] for r in successes],50),
                'e2e_p90':percentile([r['e2e'] for r in successes],90),'ttft_p50':percentile([r['ttft'] for r in gen],50),
                'ttft_p90':percentile([r['ttft'] for r in gen],90),'decode_s_per_token_p50':percentile([r['mean_decode_seconds_per_token'] for r in gen],50),
                'tool_total_p50':percentile([r['tool_returned']-r['tool_queued'] for r in successes],50),
                'context_total_p50':percentile([r['context_returned']-r['context_queued'] for r in successes],50),
                'engine_affinity':affinity,'engine_affinity_end':pin_tree(args.engine_pid,engine_cpus),'tool_cpus':tool_cpus,'context_cpus':context_cpus,'union_cpus':cpus,
                'controller_cpus':sorted(os.sched_getaffinity(0)),'worker_pings':pings,'begin_monotonic':begin,'begin_epoch':epoch,
                'gpu_budget':1,'cpu_budget_physical':len(cpus),'controller_cpu_budget_physical':4,'note':'Constructed finite-burst two-turn workflow, concurrency is whole-agent admission cap; fixed tool call, actual output text; local CPU partition, not remote disaggregation.'}
            (folder/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            trace=[{'name':e['name'],'ph':'X','ts':e['ts']*1e6,'dur':e.get('dur',0)*1e6,'pid':'application','tid':e['rid']} for e in events]
            (folder/'timeline.json').write_text(json.dumps({'traceEvents':trace,'clock':'host monotonic microseconds','begin_epoch':epoch,'begin_monotonic':begin})+'\n')
            return summary
    finally:
        if 'observer_stop' in locals():observer_stop.set()
        if 'observer' in locals():await asyncio.to_thread(observer.join,30)
        tool_pool.shutdown(wait=True,cancel_futures=True);context_pool.shutdown(wait=True,cancel_futures=True)


async def main_async(args):
    from experiments.agentic_profile.cpu_workloads import execute_tool,freeze_manifest
    from transformers import AutoTokenizer
    os.sched_setaffinity(0,{64,65,66,67})
    args.output.mkdir(parents=True,exist_ok=False)
    files=sorted(Path('agentshift').rglob('*.py'))
    manifest_path=args.output/'corpus-manifest.json'
    if args.tool_manifest:
        manifest_path.write_bytes(Path(args.tool_manifest).read_bytes())
    else:freeze_manifest(files,manifest_path)
    manifest=str(manifest_path.resolve())
    tool=execute_tool({'operation':'ast','manifest':manifest,'passes':args.tool_passes,'max_output_chars':args.tool_output_chars})
    if not tool['success']:raise RuntimeError(tool)
    (args.output/'precomputed-tool.json').write_text(json.dumps(tool,indent=2)+'\n')
    tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
    content='Analyze the Python repository after the planned AST inspection. We will run a repository analysis tool next.\n'+''.join(p.read_text() for p in files[:4])
    tokens=tokenizer.encode(content,add_special_tokens=False)
    while len(tokens)<args.prompt_tokens:tokens+=tokens
    base=tokens[:args.prompt_tokens]
    specs=[];rng=random.Random(args.seed)
    for concurrency in args.concurrency:
        for rep in range(args.repetitions):
            order=list(args.policies);rng.shuffle(order)
            specs.extend({'policy':p,'concurrency':concurrency,'repetition':rep} for p in order)
    contract={'args':vars(args)|{'output':str(args.output)},'specifications':specs,'runs':[],'complete':False,
              'base_input_sha256':hashlib.sha256(json.dumps(base).encode()).hexdigest()}
    path=args.output/'campaign.json'
    path.write_text(json.dumps(contract,indent=2)+'\n')
    for index,s in enumerate(specs):
        print(json.dumps({'event':'start','index':index,**s}),flush=True)
        result=await run_one(args,s['policy'],s['concurrency'],s['repetition'],manifest,base,tool,args.output/f'{index:03d}-{s["policy"]}-c{s["concurrency"]}-r{s["repetition"]}')
        contract['runs'].append(result);path.write_text(json.dumps(contract,indent=2)+'\n')
        print(json.dumps({'event':'complete','index':index,**{k:result[k] for k in ('policy','concurrency','repetition','successful','seconds','e2e_p50','ttft_p50','tool_total_p50','context_total_p50')}}),flush=True)
    contract['complete']=True;path.write_text(json.dumps(contract,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--engine',default='http://127.0.0.1:31480');p.add_argument('--engine-pid',type=int,required=True)
    p.add_argument('--model',default='/data-aisoft/zhujianian/Uneed/huggingface_download/Qwen3.5-35B-A3B');p.add_argument('--cores',type=int,default=8);p.add_argument('--cpu-start',type=int,default=0)
    p.add_argument('--agents',type=int,default=16);p.add_argument('--concurrency',type=int,nargs='+',default=[2,8]);p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--policies',nargs='+',choices=['shared','tools-isolated','frontend-isolated','frontend-reserved2','precomputed'],default=['shared','tools-isolated','frontend-isolated','precomputed'])
    p.add_argument('--tool-manifest');p.add_argument('--tool-workers',type=int,default=8);p.add_argument('--tool-passes',type=int,default=1);p.add_argument('--tool-output-chars',type=int,default=6000)
    p.add_argument('--prompt-tokens',type=int,default=2048);p.add_argument('--output-tokens',type=int,default=128);p.add_argument('--arrival-interval',type=float,default=0.0);p.add_argument('--seed',type=int,default=1702)
    args=p.parse_args()
    if args.cores<4 or args.cores%2: p.error('even cores >=4 required')
    if args.output_tokens<2 or args.agents<1 or min(args.concurrency)<1:p.error('output>=2, agents/concurrency>=1 required')
    asyncio.run(main_async(args))
if __name__=='__main__':main()
