"""Local PP1 hybrid-copy token-cadence comparison; no runtime changes."""
import argparse,asyncio,hashlib,json,random,statistics,time,uuid,os
from pathlib import Path
import aiohttp
from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.controller.snapshot_directory import validated_snapshots
from agentshift.state.schema import AgentContinuation
from experiments.fluidkv.benchmark_decisions import ObservedTransferClient
from experiments.fluidkv.benchmark_native_pairs import stream,flush
from experiments.fluidkv.calibrate_decision_costs import calibration_store,finish_store,digest
from experiments.agentic_profile.profile_state_work import idle,first_or_done
from experiments.local_pipeline.metrics import summarize_emissions,align_gaps_with_windows


async def observe_generate(session,url,tokens,outputs,first_event=None):
    row={'start':time.perf_counter(),'input_tokens':len(tokens),'input_sha256':digest(tokens),'requested_output_tokens':outputs,'emissions':[],'success':False}
    final=None
    try:
        async with session.post(url+'/generate',json={'input_ids':tokens,'stream':True,'sampling_params':{'temperature':0,'ignore_eos':True,'max_new_tokens':outputs}}) as response:
            response.raise_for_status()
            while True:
                raw=await response.content.readline();received=time.perf_counter()
                if not raw:break
                if not raw.startswith(b'data:'):continue
                payload=raw[5:].strip()
                if payload==b'[DONE]':continue
                final=json.loads(payload);count=final.get('meta_info',{}).get('completion_tokens',len(final.get('output_ids',[])))
                row['emissions'].append({'time':received,'tokens':count})
                if count>0 and 'first' not in row:
                    row['first']=received
                    if first_event is not None:first_event.set()
        if final is None or 'first' not in row:raise RuntimeError('missing output')
        row['meta_info']=final['meta_info'];row['output_ids']=final['output_ids'];row['output_sha256']=digest(row['output_ids'])
        if final['meta_info']['prompt_tokens']!=len(tokens) or final['meta_info']['completion_tokens']!=outputs or len(row['output_ids'])!=outputs:raise RuntimeError('length mismatch')
        row['success']=True
    except Exception as exc:row['error']=repr(exc)
    row['finish']=time.perf_counter();row['metrics']=summarize_emissions(row['emissions'],outputs)
    if not row['metrics']['complete']:
        row['success']=False;row.setdefault('error','stream count/timestamp integrity failure')
    if 'first' in row:
        row['ttft_seconds']=row['first']-row['start'];row['decode_mean_seconds']=(row['finish']-row['first'])/(outputs-1)
    return row


async def run(args):
    args.output.mkdir(parents=True,exist_ok=False);events=[]
    class Client(ObservedTransferClient):
        async def control_post(self,path,payload,**kwargs):
            event={'engine':self.engine_id,'path':path,'role':(payload or {}).get('role'),'prepare_only':(payload or {}).get('prepare_only'),'start':time.perf_counter()}
            try:
                result=await super().control_post(path,payload,**kwargs);event['response']=result;return result
            except Exception as exc:event['error']=repr(exc);raise
            finally:event['finish']=time.perf_counter();events.append(event)
    clients={name:Client(name,url) for name,url in zip(['source','destination'],args.engines)}
    store=calibration_store(args);coordinator=HybridMigrationCoordinator(store,clients,base_port=args.transfer_port)
    rng=random.Random(args.seed)
    parents={size:[2000+size//8192]+[rng.randrange(3000,19000) for _ in range(size-1)] for size in args.prefix_tokens}
    foregrounds=[[22000+i]+[rng.randrange(23000,29000) for _ in range(args.input_tokens-1)] for i in range(max(args.concurrency))]
    suffix=[rng.randrange(30000,31000) for _ in range(64)]
    manifest=[]
    for target in args.foreground_targets:
        for concurrency in args.concurrency:
            for size in args.prefix_tokens:
                for rep in range(args.repetitions):
                    modes=list(args.modes);rng.shuffle(modes)
                    manifest.extend({'target':target,'concurrency':concurrency,'prefix_tokens':size,'repetition':rep,'mode':mode} for mode in modes)
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output/'arguments.json').write_text(json.dumps(vars(args)|{'output':str(args.output),'metadata_directory':str(args.metadata_directory)},indent=2)+'\n')
    (args.output/'fixtures.json').write_text(json.dumps({'parent_hashes':{s:digest(t) for s,t in parents.items()},'foreground_hashes':[digest(t) for t in foregrounds],'suffix':suffix},indent=2)+'\n')
    (args.output/'clock.json').write_text(json.dumps({'epoch':time.time(),'monotonic':time.perf_counter()})+'\n')
    timeout=aiohttp.ClientTimeout(total=240)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        info=[]
        for url in args.engines:
            async with session.get(url+'/get_server_info') as response:response.raise_for_status();info.append(await response.json())
        (args.output/'server-info.json').write_text(json.dumps(info,indent=2)+'\n')
        if any(x.get('pp_size',1)!=1 or x.get('tp_size',1)!=1 for x in info):raise RuntimeError('requires PP1 TP1')
        await idle(session,args.engines);await coordinator._ensure_group('source','destination')
        warm=[]
        for url in args.engines:
            for c in args.concurrency:
                await flush(session,args.engines)
                warm.extend(await asyncio.gather(*(observe_generate(session,url,t,args.output_tokens) for t in foregrounds[:c])))
        for size,parent in parents.items():
            await flush(session,args.engines);await stream(session,args.engines[0],parent,1)
            observed=await clients['source'].inspect_hybrid_prefix(tuple(parent))
            if tuple(parent) not in validated_snapshots(parent,observed):raise RuntimeError(f'no exact checkpoint {size}: {observed}')
            agent=uuid.uuid4().hex;store.register_agent(AgentContinuation(agent,0,'source',1,tuple(parent)))
            result=await coordinator.migrate_async(agent,'destination');coordinator.acknowledge_async_migration(result.migration_id)
            if any(c.live_transfers for c in clients.values()):raise RuntimeError('warmup transfer not cleaned')
        (args.output/'warmup.json').write_text(json.dumps(warm)+'\n')
        if not all(x['success'] for x in warm):raise RuntimeError('warmup generation failed')
        with (args.output/'records.jsonl').open('w',buffering=1) as log:
            for index,spec in enumerate(manifest):
                row={'index':index,**spec,'success':False,'requests':[]};tasks=[];parent=parents[spec['prefix_tokens']];row['parent_input_sha256']=digest(parent)
                try:
                    if any(c.live_transfers for c in clients.values()):raise RuntimeError('unconfirmed transfer')
                    row['idle_before']=await idle(session,args.engines);await flush(session,args.engines)
                    row['source_build']=await stream(session,args.engines[0],parent,1)
                    observed=await clients['source'].inspect_hybrid_prefix(tuple(parent));row['source_inspection']=observed
                    if tuple(parent) not in validated_snapshots(parent,observed):raise RuntimeError('exact checkpoint absent')
                    agent=uuid.uuid4().hex;store.register_agent(AgentContinuation(agent,0,'source',1,tuple(parent)))
                    events.clear();first=asyncio.Event()
                    if args.profile:
                        for ei,url_profile in enumerate(args.engines):
                            async with session.post(url_profile+'/start_profile',json={'activities':['CPU','GPU'],'with_stack':False,'record_shapes':False,'output_dir':str((args.output/f'profile-{index}-engine-{ei}').resolve())}) as response:response.raise_for_status();await response.read()
                    row['profiled']=args.profile;row['start']=time.perf_counter()
                    url=clients[spec['target']].base_url
                    tasks=[asyncio.create_task(observe_generate(session,url,t,args.output_tokens,first)) for t in foregrounds[:spec['concurrency']]]
                    await first_or_done(first,tasks);row['first_observed']=time.perf_counter()
                    await asyncio.sleep(args.trigger_delay);row['trigger']=time.perf_counter();row['active_at_trigger']=sum(not t.done() for t in tasks)
                    if not row['active_at_trigger']:raise RuntimeError('foreground ended before injection')
                    if spec['mode']=='copy':
                        result=await coordinator.migrate_async(agent,'destination');coordinator.acknowledge_async_migration(result.migration_id)
                        row['migration']={'start':row['trigger'],'finish':time.perf_counter(),'bytes':result.bytes_transferred,'id':result.migration_id}
                    row['requests']=await asyncio.gather(*tasks);row['finish']=time.perf_counter();row['control_events']=list(events)
                    if args.profile:
                        for url_profile in args.engines:
                            async with session.post(url_profile+'/stop_profile') as response:response.raise_for_status();await response.read()
                    row['decoding_at_trigger']=sum('first' in r and r['first']<=row['trigger']<r['finish'] for r in row['requests'])
                    row['cleanup_confirmed']=not any(c.live_transfers for c in clients.values())
                    if not row['cleanup_confirmed']:raise RuntimeError('cleanup not confirmed')
                    row['idle_after']=await idle(session,args.engines)
                    if spec['mode']=='copy':
                        a=await stream(session,args.engines[0],parent+suffix,8);b=await stream(session,args.engines[1],parent+suffix,8)
                        row['validation']={'source':a,'destination':b,'cache_hit':b['meta_info']['cached_tokens']==len(parent),'cached_output_equal':a['output_ids']==b['output_ids'], 'source_hit':a['meta_info']['cached_tokens']>=len(parent),'lengths_preserved':all(x['meta_info']['prompt_tokens']==len(parent)+len(suffix) and x['meta_info']['completion_tokens']==8 and len(x['output_ids'])==8 for x in (a,b))}
                        if not all(row['validation'][key] for key in ('cache_hit','cached_output_equal','source_hit','lengths_preserved')):raise RuntimeError('copy validation failed')
                    window={'name':'fixed_injection_window','start':row['trigger']-.05,'end':row['trigger']+.35}
                    for r in row['requests']:
                        r['injection_window_gaps']=[g for g in r['metrics'].get('gaps',[]) if g['start']<window['end'] and g['end']>window['start']]
                    row['success']=all(r['success'] for r in row['requests'])
                except Exception as exc:
                    row['error']=repr(exc);values=await asyncio.gather(*tasks,return_exceptions=True);row['requests']=[x for x in values if isinstance(x,dict)]
                    row['control_events']=list(events);row['cleanup_confirmed']=not any(c.live_transfers for c in clients.values());row['finish']=time.perf_counter()
                log.write(json.dumps(row)+'\n')
                print(json.dumps({'index':index,**spec,'success':row['success'],'error':row.get('error'),'migration_seconds':(row.get('migration',{}).get('finish',0)-row.get('migration',{}).get('start',0)) or None}),flush=True)
                if not row['success']:
                    finish_store(store,args);raise RuntimeError('failed run preserved; stop without flushing')
    finish_store(store,args);(args.output/'complete.json').write_text(json.dumps({'complete':True,'runs':len(manifest)})+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--engines',nargs=2,default=['http://127.0.0.1:31480','http://127.0.0.1:31481'])
    p.add_argument('--prefix-tokens',type=int,nargs='+',default=[8192,24576]);p.add_argument('--concurrency',type=int,nargs='+',default=[2,8]);p.add_argument('--foreground-targets',nargs='+',choices=['source','destination'],default=['source','destination'])
    p.add_argument('--profile',action='store_true',help='Diagnostic only, excluded from performance comparisons');p.add_argument('--modes',nargs='+',choices=['no-copy','copy'],default=['no-copy','copy']);p.add_argument('--repetitions',type=int,default=3);p.add_argument('--trigger-delay',type=float,default=.1)
    p.add_argument('--input-tokens',type=int,default=1024);p.add_argument('--output-tokens',type=int,default=256);p.add_argument('--transfer-port',type=int,default=36000);p.add_argument('--metadata-directory',type=Path,default=Path('/tmp/local-pipeline-metadata'));p.add_argument('--seed',type=int,default=1515)
    args=p.parse_args();os.sched_setaffinity(0,{64,65,66,67})
    if max(args.prefix_tokens)>24576 or max(args.concurrency)>8 or args.output_tokens<2:p.error('bounded local experiment only')
    asyncio.run(run(args))
if __name__=='__main__':main()
