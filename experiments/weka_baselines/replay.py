"""Persistent-cache, original-length AgentX replay against dedicated engines.

Global offered-concurrency BS is a client limit, not asserted GPU batch size.
Only independent policy trajectories reset caches; no phase/round/scaleout flush.
Native hash identity/lengths are retained; content and fleet arrivals are modeled.
"""
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
import uuid

import aiohttp
import numpy as np
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.dualmap_adapter import DualMapAdapter
from experiments.fluidkv.benchmark_native_pairs import flush
from experiments.multinode.cacheroute import dispatch
from experiments.multinode.benchmark_highbs import settle_engines
from experiments.weka_baselines.common import schedule,rate_plan,expand_dualmap

ROOT=Path(__file__).parent
URLS=[f'http://12.11.3.{i}:31740' for i in [27,28,29]]


def write(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


async def generate(session,url,tokens,count,callback=None):
    start=time.perf_counter();first=last=None;final=None;chunks=[];previous_count=0
    async with session.post(url+'/generate',json={'input_ids':tokens,'stream':True,
        'sampling_params':{'temperature':0,'ignore_eos':True,'max_new_tokens':count}}) as response:
        if response.status!=200:raise RuntimeError(f'HTTP {response.status}: '+(await response.text())[:2000])
        async for line in response.content:
            if not line.startswith(b'data:'):continue
            data=line[5:].strip()
            if data==b'[DONE]':continue
            final=json.loads(data)
            n=final.get('meta_info',{}).get('completion_tokens',len(final.get('output_ids',[])))
            if n>previous_count:
                now=time.perf_counter()
                if first is None:
                    first=now
                    if callback:await callback(final,now-start)
                chunks.append([now-start,n]);last=now;previous_count=n
    if first is None or final is None:raise RuntimeError('no generated output')
    meta=final['meta_info']
    if meta['prompt_tokens']!=len(tokens) or meta['completion_tokens']!=count:
        raise RuntimeError('input/output length mismatch')
    return dict(meta=meta,output_ids=final['output_ids'],ttft=first-start,
        first_absolute=first,end_absolute=time.perf_counter(),
        mean_tpot=(last-first)/(count-1) if count>1 else 0,stream_chunks=chunks)


class Fleet:
    def __init__(self,out):
        self.out=out
        self.clients={i:SGLangAgentShiftClient(str(i),u,timeout=300) for i,u in enumerate(URLS)}
        self.store=SQLiteStateStore(out/'ownership.db');self.coordinators={}
        # Directed groups prevent role reversal and reserve four TP2 ports.
        for s in range(3):
            for d in range(3):
                if s==d:continue
                self.coordinators[s,d]=HybridMigrationCoordinator(self.store,
                    {'source':self.clients[s],'destination':self.clients[d]},
                    master_address=f'12.11.3.{27+s}',base_port=24100+8*len(self.coordinators))

    async def inspect(self,engine,tokens):
        return await self.clients[engine].inspect_hybrid_prefix(tuple(tokens))

    async def warm(self,session):
        info=[]
        for url in URLS:
            async with session.get(url+'/get_server_info') as r:r.raise_for_status();info.append(await r.json())
        write(self.out/'server-info.json',info)
        # Shared initialization, not charged as warm-model logical scaleout.
        for (s,d),coord in self.coordinators.items():
            await settle_engines(session,URLS)
            start=time.perf_counter();await coord._ensure_group('source','destination')
            print('communication group ready',s,d,flush=True)
            with (self.out/'initialization.jsonl').open('a') as f:f.write(json.dumps(dict(source=s,destination=d,group_seconds=time.perf_counter()-start))+'\n')
        await flush(session,URLS)
        parent=list(range(2000,3024))
        for (s,d),coord in self.coordinators.items():
            await settle_engines(session,URLS)
            await flush(session,URLS)
            await generate(session,URLS[s],parent,1)
            await settle_engines(session,URLS)
            agent='warm-'+str(s)+str(d)
            self.store.register_agent(AgentContinuation(agent,1,'source',1,tuple(parent)))
            moved=await coord.migrate(agent,'destination')
            child=parent+[1100,1101]
            a=await generate(session,URLS[s],child,8)
            b=await generate(session,URLS[d],child,8)
            checks=dict(output_equal=a['output_ids']==b['output_ids'],
                        source_cache=a['meta'].get('cached_tokens')==len(parent),
                        destination_cache=b['meta'].get('cached_tokens')==len(parent))
            with (self.out/'transfer-preflight.jsonl').open('a') as f:
                f.write(json.dumps(dict(source=s,destination=d,bytes=moved.bytes_transferred,checks=checks))+'\n')
            assert all(checks.values()),checks
        # Separate fixed calibration; no evaluated-future data used by routers.
        await settle_engines(session,URLS);await flush(session,URLS)
        ids=list(range(2000,34768))
        cold=await generate(session,URLS[0],ids,16)
        warm=await generate(session,URLS[0],ids+[1111],16)
        self.calibration=dict(prefill_seconds_per_token=max(1e-6,(cold['ttft']-warm['ttft'])/32768),
            qcap=1/max(.1,cold['end_absolute']-cold['first_absolute']+cold['ttft']),
            note='Fixed 32K/16-token local calibration; heterogeneous request costs are a limitation.')
        write(self.out/'calibration.json',self.calibration)


async def run_trial(fleet,session,rows,token_array,out,spec):
    out.mkdir(parents=True,exist_ok=False)
    workload=schedule(rows,spec['scenario'],spec.get('streams',128),spec.get('period',60))
    write(out/'spec.json',dict(spec,requests=len(workload),cache_reset='Once before independent trajectory; never between phases',
        workload_sha256=hashlib.sha256(json.dumps([(r['key'],r['turn'],r['offered_seconds'],r['token_sha256']) for r in workload]).encode()).hexdigest(),
        global_concurrency_limit=spec['bs'],server_max_running=128,output_cap=spec.get('output_cap')))
    await settle_engines(session,URLS)
    await flush(session,URLS)  # The ONLY reset in this trial.
    log=(out/'requests.jsonl').open('w',buffering=1)
    events=(out/'events.jsonl').open('w',buffering=1)
    samples=(out/'telemetry.jsonl').open('w',buffering=1)
    began=time.perf_counter();live=[0]*3;reserved=[False]*3
    active=[0,1] if spec['scenario']=='scaleout' else [0,1,2]
    base=spec['policy'].split('+')[0];action=spec['policy'].split('+')[1] if '+' in spec['policy'] else None
    adapter=DualMapAdapter(Path('/home/zhujianian/DualMap'),3,1048576,20480,10,
        fleet.calibration['prefill_seconds_per_token'],hybrid_snapshot_slots=384,inspect_snapshots=True,
        pending_prefills=128) if base=='dualmap' else None
    if adapter and len(active)==2:await adapter.retire_engine(2)
    mutex=asyncio.Lock();changed=asyncio.Event();stop=asyncio.Event();sem=asyncio.Semaphore(spec['bs'])
    done={r['ordinal']:asyncio.Event() for r in workload};futures={};owners={};records=[]
    prep_tasks=set();prep_lock=asyncio.Lock();rng=random.Random(8831)
    plan=rate_plan([],len(active),fleet.calibration['qcap'])
    arrived=[];pending_keys=Counter();prep_counts=Counter();last_hbm=0
    def event(kind,**values):events.write(json.dumps(dict(kind=kind,seconds=time.perf_counter()-began,**values))+'\n')
    event('initial_cache_reset',engines=URLS)

    async def telemetry():
        nonlocal last_hbm
        while not stop.is_set():
            write(out/'heartbeat.json',dict(elapsed_seconds=time.perf_counter()-began,completed=len(records),
                expected=len(workload),live=live,pending=sum(pending_keys.values()),preparation_counts=dict(prep_counts)))
            for i,url in enumerate(URLS):
                try:
                    async with session.get(url+'/get_load',timeout=aiohttp.ClientTimeout(total=5)) as r:
                        r.raise_for_status();data=await r.json()
                    sample=dict(seconds=time.perf_counter()-began,engine=i,load=data,client_live=live[i],reserved=reserved[i])
                    # Scalar pool occupancy from non-mutating snapshot metadata.
                    obs=await fleet.inspect(i,[999])
                    sample['pool']={k:obs.get(k) for k in ['available_token_slots','evictable_token_slots','available_snapshot_slots','evictable_snapshot_slots']}
                    samples.write(json.dumps(sample)+'\n')
                except Exception as e:samples.write(json.dumps(dict(engine=i,seconds=time.perf_counter()-began,error=repr(e)))+'\n')
            if time.perf_counter()-last_hbm>10:
                last_hbm=time.perf_counter()
                for node in ['node27','node28','node29']:
                    p=await asyncio.create_subprocess_exec('ssh','-o','BatchMode=yes',node,
                        '/usr/local/hyhal/bin/hy-smi --showmeminfo vram --showuse --json',stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                    try:
                        data,err=await asyncio.wait_for(p.communicate(),10)
                        samples.write(json.dumps(dict(node=node,seconds=time.perf_counter()-began,hbm=json.loads(data)))+'\n')
                    except Exception as e:
                        if p.returncode is None:p.kill();await p.wait()
                        samples.write(json.dumps(dict(node=node,error=repr(e)))+'\n')
            try:await asyncio.wait_for(stop.wait(),2)
            except asyncio.TimeoutError:pass

    async def epoch_events():
        nonlocal plan,active
        for phase in [1,2]:
            await asyncio.sleep(max(0,began+phase*spec.get('period',60)-time.perf_counter()))
            async with mutex:
                if phase==1 and spec['scenario']=='scaleout':
                    active=[0,1,2]
                    if adapter:await expand_dualmap(adapter)
                    event('logical_scaleout',active=active,cache_reset=False,model_preloaded=True)
                past=[key for t,key in arrived if t>=(phase-1)*spec.get('period',60)]
                plan=rate_plan(past,len(active),fleet.calibration['qcap'],spec.get('period',60))
                event('phase',phase=phase,active=active,training_requests=len(past),assignments=plan.assignments,cache_reset=False)
            changed.set()

    async def pump():
        while not stop.is_set():
            if adapter:
                async with mutex:
                    for request in await adapter.dispatch():
                        future=futures[request._id]
                        if not future.done():future.set_result(request)
            changed.clear()
            try:await asyncio.wait_for(changed.wait(),.05)
            except asyncio.TimeoutError:pass

    async def prepare(source,tokens,key):
        if prep_lock.locked():prep_counts['skip_other_preparation']+=1;return
        async with prep_lock:
            target=None;prefix=None;move=False
            async with mutex:
                pressure=[live[i]+sum(n for k,n in pending_keys.items() if owners.get(k)==i) for i in range(3)]
                target=min((i for i in active if i!=source),key=lambda i:pressure[i])
                if pressure[source]<=pressure[target]+1:
                    prep_counts['skip_no_load_relief']+=1;return
                if reserved[source] or reserved[target] or live[target]:
                    prep_counts['skip_target_busy']+=1;return
                if action=='move' and live[source]:
                    prep_counts['skip_source_busy']+=1;return
                src=await fleet.inspect(source,tokens)
                n=src.get('matched_tokens',0)
                if n<8192:prep_counts['skip_no_long_snapshot']+=1;return
                prefix=tokens[:n]
                dst=await fleet.inspect(target,prefix)
                if dst.get('full_prefix_available'):prep_counts['skip_already_warm']+=1;return
                if action=='move' and (dst.get('evictable_token_slots',1)>0 or dst.get('available_token_slots',0)<1048576):
                    prep_counts['skip_nonempty_destination']+=1;return
                if dst.get('available_token_slots',0)+dst.get('evictable_token_slots',0)<n:
                    prep_counts['skip_capacity']+=1;return
                move=action=='move';reserved[target]=True
                if move:reserved[source]=True
            started=time.perf_counter();moved=0;error=None
            try:
                if move:
                    agent=uuid.uuid4().hex
                    fleet.store.register_agent(AgentContinuation(agent,1,'source',1,tuple(prefix)))
                    result=await fleet.coordinators[source,target].migrate(agent,'destination')
                    moved=result.bytes_transferred
                    owners[key]=target
                else:await generate(session,URLS[target],prefix,1)
                prep_counts['completed']+=1
            except Exception as e:
                error=repr(e);prep_counts['failed']+=1
            finally:
                reserved[target]=False
                if move:reserved[source]=False
                event('preparation',method=action,source=source,target=target,key=key,tokens=len(prefix),
                      duration=time.perf_counter()-started,bytes=moved,error=error)
                changed.set()

    async def safe_prepare(source,tokens,key):
        try:await prepare(source,tokens,key)
        except Exception as e:
            prep_counts['observation_failed']+=1
            event('preparation_observation_error',source=source,key=key,error=repr(e))

    async def execute(row):
        ordinal=row['ordinal'];key=row['key'];target=None;result=None;error=None;request=None
        offered=began+row['offered_seconds'];first_at=None;admitted=dispatch_time=None;pending=True
        await asyncio.sleep(max(0,offered-time.perf_counter()))
        arrived.append((time.perf_counter()-began,key));pending_keys[key]+=1
        if row['predecessor'] is not None:await done[row['predecessor']].wait()
        eligible=time.perf_counter()
        async with sem:
            admitted=time.perf_counter()
            start=row['token_offset'];tokens=token_array[start:start+row['input_tokens']].tolist()
            try:
                async with mutex:
                    observations=await asyncio.gather(*(fleet.inspect(i,tokens) for i in active))
                    if adapter:
                        for i,obs in zip(active,observations):adapter.observe_snapshot_response(i,tokens,obs)
                        future=asyncio.get_running_loop().create_future();futures[ordinal]=future
                        await adapter.enqueue(ordinal,{'chat_id':key,'parent_chat_id':key,'turn':row['turn'],
                            'hash_ids':row['hash_ids']},tokens,row['output_tokens'])
                        changed.set()
                    elif base=='sticky':
                        target=owners.setdefault(key,active[int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%len(active)])
                    elif base=='p2c':target=min(rng.sample(active,min(2,len(active))),key=lambda i:live[i]+int(reserved[i]))
                    else:target=active[dispatch(plan,key,[live[i]+int(reserved[i]) for i in active],rng)]
                    if not adapter:live[target]+=1
                if adapter:
                    request=await asyncio.wait_for(future,7200)
                    target=adapter.physical_target(request);live[target]+=1
                while reserved[target]:
                    changed.clear()
                    try:await asyncio.wait_for(changed.wait(),.05)
                    except asyncio.TimeoutError:pass
                pending_keys[key]-=1;pending=False;owners[key]=target
                dispatch_time=time.perf_counter()
                async def first(payload,elapsed):
                    nonlocal first_at
                    first_at=time.perf_counter()
                    if adapter:
                        async with mutex:await adapter.first_token(request,elapsed)
                    changed.set()
                result=await generate(session,URLS[target],tokens,row['output_tokens'],first)
            except Exception as e:error=repr(e)
            finally:
                if pending:pending_keys[key]-=1
                if target is not None:live[target]-=1
                if adapter and request is None and ordinal in adapter.requests:
                    orphan=adapter.requests[ordinal]
                    async with mutex:
                        for slot in range(len(adapter.engine_map)):
                            adapter.scheduler.global_request_queue.del_req(slot,orphan)
                if adapter and request is not None:
                    async with mutex:await adapter.finished(request,result['output_ids'] if result else [],error is None)
                end=time.perf_counter()
                record={k:v for k,v in row.items() if k!='hash_ids'}
                record.update(success=error is None,error=error,target=target,
                    eligible_seconds=eligible-began,admitted_seconds=admitted-began,
                    dispatch_seconds=dispatch_time-began if dispatch_time else None,
                    completion_seconds=end-began,ttft_seconds=first_at-offered if first_at else None,
                    e2e_seconds=end-offered,dependency_delay_seconds=eligible-offered,
                    admission_delay_seconds=admitted-eligible,
                    tpot_seconds=result['mean_tpot'] if result else None,
                    cached_tokens=result['meta'].get('cached_tokens',0) if result else None,
                    engine_meta=result['meta'] if result else None,
                    output_sha256=hashlib.sha256(json.dumps(result['output_ids']).encode()).hexdigest() if result else None,
                    output_ids=result['output_ids'] if result else None,
                    stream_chunks=result['stream_chunks'] if result else None)
                log.write(json.dumps(record)+'\n');records.append(record);done[ordinal].set();changed.set()
                if len(records)%8==0:
                    write(out/'progress.json',dict(completed=len(records),total=len(workload),elapsed_seconds=end-began,
                        errors=sum(not r['success'] for r in records),live=live,queued_for_client=sum(pending_keys.values()),
                        preparation_counts=dict(prep_counts)))
                if action and result:
                    task=asyncio.create_task(safe_prepare(target,tokens,key));prep_tasks.add(task)
                    task.add_done_callback(prep_tasks.discard)
    background=[asyncio.create_task(telemetry()),asyncio.create_task(epoch_events()),asyncio.create_task(pump())]
    tasks=[asyncio.create_task(execute(r)) for r in workload]
    try:
        await asyncio.gather(*tasks)
        while prep_tasks:await asyncio.gather(*list(prep_tasks))
    finally:
        stop.set();changed.set()
        for task in background:task.cancel()
        await asyncio.gather(*background,return_exceptions=True)
        log.close();events.close();samples.close()
    ended=time.perf_counter()-began
    def quantile(field,rs):
        values=[r[field] for r in rs if r[field] is not None]
        return {str(p):float(np.percentile(values,p)) for p in [50,95,99]} if values else {}
    # Failed/no-first-token requests remain in the all-traffic CDF as infinity.
    all_ttft=sorted(r['ttft_seconds'] if r['ttft_seconds'] is not None else float('inf') for r in records)
    all_tails={str(p):(lambda v:v if np.isfinite(v) else None)(all_ttft[int(np.ceil(len(all_ttft)*p/100))-1]) for p in [50,95,99]}
    summary=dict(spec=spec,requests=len(records),expected=len(workload),errors=sum(not r['success'] for r in records),
        duration_seconds=ended,ttft=quantile('ttft_seconds',records),e2e=quantile('e2e_seconds',records),tpot=quantile('tpot_seconds',records),
        all_traffic_ttft=all_tails,unserved_first_token=sum(r['ttft_seconds'] is None for r in records),
        phases={str(p):dict(requests=sum(r['phase']==p for r in records),ttft=quantile('ttft_seconds',[r for r in records if r['phase']==p]),
            e2e=quantile('e2e_seconds',[r for r in records if r['phase']==p])) for p in range(3)},
        goodput=[dict(ttft_slo=ttft,tpot_slo=tpot,requests_per_second=sum(r['success'] and r['ttft_seconds']<=ttft and r['tpot_seconds']<=tpot for r in records)/ended,
            attainment=sum(r['success'] and r['ttft_seconds']<=ttft and r['tpot_seconds']<=tpot for r in records)/len(workload))
            for ttft in [1,5,10,30] for tpot in [.1,.2,.5]],
        task_completion_seconds={k:max(r['completion_seconds'] for r in records if r['key']==k)-min(r['offered_seconds'] for r in records if r['key']==k) for k in {r['key'] for r in records}},
        preparation_counts=dict(prep_counts),semantics='Symbolic original-length replay, not original Claude semantic execution. Batch-dependent output equality is not guaranteed and hashes are retained.')
    write(out/'summary.json',summary)
    return summary
