"""Time-bounded, paired AgentX logical-scaleout test with retained caches.

16 length-stratified native substreams, three original requests each, BS=16.
Only completed-history information is available to expansion preparation.
Two repetitions of each router with/without one admissible state placement.
"""
import asyncio
from collections import Counter,defaultdict
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import uuid

import aiohttp
import numpy as np
from agentshift.state.schema import AgentContinuation
from experiments.fluidkv.dualmap_adapter import DualMapAdapter
from experiments.fluidkv.benchmark_native_pairs import flush
from experiments.multinode.benchmark_highbs import settle_engines
from experiments.multinode.cacheroute import dispatch
from experiments.weka_baselines.common import expand_dualmap,rate_plan
from experiments.weka_baselines.replay import Fleet,ROOT,URLS,generate,write

OUT=ROOT/'results'/'one-hour-20260929'
POLICIES=['dualmap','dualmap+move','cacheroute','cacheroute+move']


def cohort(rows):
    groups=defaultdict(list)
    for r in rows:
        if r['turn']<3:groups[r['stream_index']].append(r)
    bins=defaultdict(list)
    for i,rs in sorted(groups.items()):
        if len(rs)!=3 or sum(r['output_tokens'] for r in rs)>1536:continue
        n=max(r['input_tokens'] for r in rs)
        b=0 if n<32768 else 1 if n<65536 else 2 if n<98304 else 3
        bins[b].append(i)
    allocation={0:5,1:5,2:5,3:1}
    assert all(len(bins[b])>=n for b,n in allocation.items())
    chosen=[i for b,n in allocation.items() for i in bins[b][:n]]
    selected=[r for i in chosen for r in sorted(groups[i],key=lambda r:r['turn'])]
    return selected,dict(streams=chosen,bins=dict(bins),allocation=allocation,
        selection='Existing seeded stream order; max context strata 0-32K/32-64K/64-96K/96-128K; sum of original outputs over 3 turns <=1536. No input/output truncation. No measured policy outcome used.',
        requests=len(selected),output_tokens=sum(r['output_tokens'] for r in selected),
        max_chain_output=max(sum(r['output_tokens'] for r in groups[i]) for i in chosen))


async def trial(fleet,session,rows,tokens,out,policy,repeat,status):
    out.mkdir(exist_ok=False);base=policy.split('+')[0];move=policy.endswith('+move')
    await settle_engines(session,URLS);await flush(session,URLS)
    log=(out/'requests.jsonl').open('w',buffering=1)
    ev=(out/'events.jsonl').open('w',buffering=1)
    tl=(out/'telemetry.jsonl').open('w',buffering=1)
    started=time.perf_counter();live=[0,0,0];active=[0,1];owners={};records=[];preps=[]
    adapter=DualMapAdapter(Path('/home/zhujianian/DualMap'),3,1048576,20480,10,
        fleet.calibration['prefill_seconds_per_token'],hybrid_snapshot_slots=384,
        inspect_snapshots=True,pending_prefills=16) if base=='dualmap' else None
    if adapter:await adapter.retire_engine(2)
    lock=asyncio.Lock();changed=asyncio.Event();stop=asyncio.Event();waiting={};phase=0
    plan=rate_plan([],2,fleet.calibration['qcap']);rng=random.Random(830+repeat)
    def event(kind,**values):
        ev.write(json.dumps(dict(kind=kind,elapsed=time.perf_counter()-started,**values))+'\n')
    event('initial_cache_reset',count=1)
    async def monitor():
        tick=0
        while not stop.is_set():
            for i,u in enumerate(URLS):
                try:
                    async with session.get(u+'/get_load',timeout=aiohttp.ClientTimeout(total=5)) as r:
                        r.raise_for_status();loads=await r.json()
                    observation=await fleet.inspect(i,[999])
                    tl.write(json.dumps(dict(elapsed=time.perf_counter()-started,engine=i,load=loads,
                        pool={k:observation.get(k) for k in ['available_token_slots','evictable_token_slots','available_snapshot_slots','evictable_snapshot_slots']}))+'\n')
                except Exception as e:tl.write(json.dumps(dict(elapsed=time.perf_counter()-started,engine=i,error=repr(e)))+'\n')
            if tick%5==0:
                for node in ['node27','node28','node29']:
                    p=await asyncio.create_subprocess_exec('ssh',node,'/usr/local/hyhal/bin/hy-smi --showmeminfo vram --showuse --json',
                        stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                    try:
                        output,_=await asyncio.wait_for(p.communicate(),10)
                        tl.write(json.dumps(dict(elapsed=time.perf_counter()-started,node=node,hbm=json.loads(output)))+'\n')
                    except Exception as e:
                        if p.returncode is None:p.kill();await p.wait()
                        tl.write(json.dumps(dict(node=node,error=repr(e)))+'\n')
            status.update(current_policy=policy,current_repeat=repeat,current_phase=phase,
                current_requests=len(records),current_expected=48,current_elapsed_s=time.perf_counter()-started,live=live)
            write(OUT/'status.json',status)
            tick+=1
            try:await asyncio.wait_for(stop.wait(),2)
            except asyncio.TimeoutError:pass
    async def pump():
        while not stop.is_set():
            if adapter:
                async with lock:
                    for r in await adapter.dispatch():
                        future=waiting[r._id]
                        if not future.done():future.set_result(r)
            changed.clear()
            try:await asyncio.wait_for(changed.wait(),.03)
            except asyncio.TimeoutError:pass
    async def invoke(row,offered):
        ordinal=row['stream_index']*3+row['turn'];target=None;request=None;result=None;error=None;first_at=None;sent=None
        ids=tokens[row['token_offset']:row['token_offset']+row['input_tokens']].tolist()
        try:
            async with lock:
                observations=await asyncio.gather(*(fleet.inspect(i,ids) for i in active))
                if adapter:
                    for i,obs in zip(active,observations):adapter.observe_snapshot_response(i,ids,obs)
                    future=asyncio.get_running_loop().create_future();waiting[ordinal]=future
                    await adapter.enqueue(ordinal,dict(chat_id=row['key'],parent_chat_id=row['key'],turn=phase,hash_ids=row['hash_ids']),ids,row['output_tokens'])
                    changed.set()
                else:
                    target=active[dispatch(plan,row['key'],[live[i] for i in active],rng)];live[target]+=1
            if adapter:
                request=await asyncio.wait_for(future,180)
                target=adapter.physical_target(request);live[target]+=1
            owners[row['key']]=target;sent=time.perf_counter()
            async def first(payload,delay):
                nonlocal first_at
                first_at=time.perf_counter()
                if adapter:
                    async with lock:await adapter.first_token(request,delay)
                changed.set()
            result=await generate(session,URLS[target],ids,row['output_tokens'],first)
        except Exception as e:error=repr(e)
        finally:
            if target is not None:live[target]-=1
            if adapter and request is not None:
                async with lock:await adapter.finished(request,result['output_ids'] if result else [],error is None)
            end=time.perf_counter()
            record={k:v for k,v in row.items() if k!='hash_ids'}
            record.update(ordinal=ordinal,phase=phase,target=target,success=error is None,error=error,
                offered_seconds=offered-started,dispatch_seconds=sent-started if sent else None,completion_seconds=end-started,
                ttft_seconds=first_at-offered if first_at else None,e2e_seconds=end-offered,
                tpot_seconds=result['mean_tpot'] if result else None,cached_tokens=result['meta'].get('cached_tokens',0) if result else 0,
                meta=result['meta'] if result else None,output_ids=result['output_ids'] if result else None,
                output_sha256=hashlib.sha256(json.dumps(result['output_ids']).encode()).hexdigest() if result else None)
            records.append(record);log.write(json.dumps(record)+'\n');changed.set()
    backgrounds=[asyncio.create_task(monitor()),asyncio.create_task(pump())]
    try:
        for phase in range(3):
            if phase:
                # All cases use the same closed-loop round barrier. Draining is
                # not a cache flush; its time remains in task completion.
                await settle_engines(session,URLS)
                if phase==1:
                    active=[0,1,2]
                    if adapter:
                        async with lock:await expand_dualmap(adapter)
                    event('logical_scaleout',active=active,cache_reset=False)
                previous=[r for r in rows if r['turn']==phase-1]
                plan=rate_plan([r['key'] for r in previous],3,fleet.calibration['qcap'],60)
                prep_start=time.perf_counter();offered=prep_start+1.0
                # One controlled second before tool return. Only previous
                # completed inputs and historical router anchors are consulted.
                if move:
                    candidates=[]
                    for r in previous:
                        source=owners[r['key']]
                        if source==2:continue
                        if adapter:
                            depth=adapter.prefix_table.lookup(r['hash_ids']);key=''.join(map(str,r['hash_ids'][:depth]))
                            eligible=2 in [adapter.scheduler.hash_function1(key,3),adapter.scheduler.hash_function2(key,3)]
                        else:eligible=2 in plan.assignments.get(r['key'],())
                        if eligible:candidates.append(r)
                    candidates.sort(key=lambda r:(-r['input_tokens'],r['key']))
                    prepared=False
                    for r in candidates:
                        source=owners[r['key']];ids=tokens[r['token_offset']:r['token_offset']+r['input_tokens']].tolist()
                        src=await fleet.inspect(source,ids);n=src.get('matched_tokens',0)
                        if n<8192:continue
                        prefix=ids[:n];dst=await fleet.inspect(2,prefix)
                        if dst.get('full_prefix_available'):
                            event('skip_preparation',phase=phase,reason='already_warm',key=r['key']);break
                        if dst.get('available_token_slots',0)<1048576 or dst.get('evictable_token_slots',0)>0:
                            event('skip_preparation',phase=phase,reason='nonempty_destination',key=r['key']);break
                        agent=uuid.uuid4().hex
                        fleet.store.register_agent(AgentContinuation(agent,1,'source',1,tuple(prefix)))
                        copy_start=time.perf_counter()
                        try:
                            copied=await fleet.coordinators[source,2].migrate(agent,'destination')
                            prep=dict(phase=phase,key=r['key'],source=source,target=2,tokens=n,bytes=copied.bytes_transferred,
                                transfer_s=time.perf_counter()-copy_start,preparation_s=time.perf_counter()-prep_start,
                                exposed_s=max(0,time.perf_counter()-offered),success=True)
                            preps.append(prep);event('migration',**prep);prepared=True
                        except Exception as e:
                            event('migration_failed',phase=phase,key=r['key'],error=repr(e));raise
                        break
                    if not candidates:event('skip_preparation',phase=phase,reason='no_new_node_router_anchor')
                    if candidates and not prepared:event('no_migration',phase=phase)
                await asyncio.sleep(max(0,offered-time.perf_counter()))
            else:offered=time.perf_counter()
            event('round_start',phase=phase,offered_seconds=offered-started,active=active,cache_reset=False)
            wave=[r for r in rows if r['turn']==phase]
            random.Random(977+repeat*3+phase).shuffle(wave)
            await asyncio.gather(*(invoke(r,offered) for r in wave))
            event('round_complete',phase=phase,requests=len(wave))
            if any(not r['success'] for r in records):raise RuntimeError('request failure; retain raw records')
        total=time.perf_counter()-started
        def q(values):
            a=sorted(values)
            return {str(p):a[max(0,math.ceil(len(a)*p/100)-1)] for p in [50,95,99]}
        def summarize(rs):
            return dict(requests=len(rs),ttft=q([r['ttft_seconds'] for r in rs]),e2e=q([r['e2e_seconds'] for r in rs]),
                tpot=q([r['tpot_seconds'] for r in rs]),cache_fraction=sum(r['cached_tokens'] for r in rs)/sum(r['input_tokens'] for r in rs),
                target_counts=dict(Counter(r['target'] for r in rs)))
        summary=dict(policy=policy,repeat=repeat,requests=len(records),errors=0,duration_seconds=total,
            total_output=sum(r['output_tokens'] for r in records),**{k:v for k,v in summarize(records).items() if k!='requests'},
            phases={str(i):summarize([r for r in records if r['phase']==i]) for i in range(3)},
            migration_bytes=sum(p['bytes'] for p in preps),migrations=preps,
            goodput=[dict(ttft_slo=a,tpot_slo=b,attainment=sum(r['ttft_seconds']<=a and r['tpot_seconds']<=b for r in records)/len(records),
                requests_per_second=sum(r['ttft_seconds']<=a and r['tpot_seconds']<=b for r in records)/total)
                for a in [1,5,10,30] for b in [.1,.2,.5]],
            tasks={key:max(r['completion_seconds'] for r in records if r['key']==key)-min(r['offered_seconds'] for r in records if r['key']==key) for key in owners})
        write(out/'summary.json',summary);return summary
    finally:
        stop.set();changed.set()
        for task in backgrounds:task.cancel()
        await asyncio.gather(*backgrounds,return_exceptions=True)
        log.close();ev.close();tl.close()


async def experiment(status):
    allrows=[json.loads(s) for s in (ROOT/'data/requests.jsonl').read_text().splitlines()]
    rows,selection=cohort(allrows);tokens=np.load(ROOT/'data/tokens.npy',mmap_mode='r')
    write(OUT/'cohort.json',selection)
    with (OUT/'selected-requests.jsonl').open('w') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    matrix=[]
    for repeat in range(2):
        # Paired blocks reverse treatment order in repetition two.
        order=['dualmap','dualmap+move','cacheroute+move','cacheroute'] if repeat==0 else ['cacheroute','cacheroute+move','dualmap+move','dualmap']
        matrix.extend(dict(policy=p,repeat=repeat) for p in order)
    write(OUT/'matrix.json',matrix)
    source=[Path(__file__),ROOT/'replay.py',ROOT/'common.py',ROOT.parent/'fluidkv/dualmap_adapter.py',ROOT.parent/'multinode/cacheroute.py']
    write(OUT/'provenance.json',dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
        source_dataset_sha256=json.loads((ROOT/'data/manifest.json').read_text())['trace_sha256'],
        control_runtime=json.loads((ROOT/'environment/runtime/control-isolation.json').read_text())))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900,connect=15,sock_read=120),connector=aiohttp.TCPConnector(limit=128)) as session:
        while True:
            ready=[]
            for u in URLS:
                try:
                    async with session.get(u+'/health',timeout=aiohttp.ClientTimeout(total=5)) as r:ready.append(r.status==200)
                except Exception:ready.append(False)
            if all(ready):break
            status.update(state='waiting_for_engines',healthy=ready);write(OUT/'status.json',status);await asyncio.sleep(5)
        fleet=Fleet(OUT);status.update(state='warming',healthy=[True]*3);write(OUT/'status.json',status)
        await fleet.warm(session)
        completed=[]
        for i,spec in enumerate(matrix):
            status.update(state='running',completed=i,total=len(matrix),current_policy=spec['policy'],current_repeat=spec['repeat'])
            write(OUT/'status.json',status);print('START',i,spec,flush=True)
            result=await trial(fleet,session,rows,tokens,OUT/'runs'/f"{i:02d}-{spec['policy']}-r{spec['repeat']}",spec['policy'],spec['repeat'],status)
            completed.append(result);write(OUT/'completed.json',completed)
            status['completed']=i+1;write(OUT/'status.json',status)
            print('FINISHED',i,'duration',result['duration_seconds'],'migrations',len(result['migrations']),flush=True)
        status['state']='complete';write(OUT/'status.json',status)


async def main():
    OUT.mkdir(parents=True,exist_ok=False);(OUT/'runs').mkdir()
    cancelled=json.loads((ROOT/'results/campaign-20260928-r3/status.json').read_text())['cancelled_at']
    deadline=datetime.datetime.fromisoformat(cancelled)+datetime.timedelta(hours=1)
    remaining=(deadline-datetime.datetime.now().astimezone()).total_seconds()
    status=dict(pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),deadline=deadline.isoformat(),
        state='starting',completed=0,total=8,budget_seconds_remaining_at_launch=remaining)
    write(OUT/'status.json',status)
    try:await asyncio.wait_for(experiment(status),timeout=max(1,remaining))
    except asyncio.TimeoutError:
        status.update(state='budget_expired',error='One-hour wall budget reached; partial cells excluded from paired claims.')
        write(OUT/'status.json',status);print(status,flush=True)
    except BaseException as e:
        status.update(state='failed',error=repr(e),traceback=traceback.format_exc());write(OUT/'status.json',status)
        print(traceback.format_exc(),flush=True)
    finally:
        for node in ['node27','node28','node29']:
            with (OUT/(node+'-engine.log')).open('wb') as f:
                subprocess.run(['ssh',node,'docker','logs','zhujianian-agentx-baselines'],stdout=f,stderr=subprocess.STDOUT)
            subprocess.run(['ssh',node,'docker','stop','--time','10','zhujianian-agentx-baselines'],timeout=45)
        status['ended']=datetime.datetime.now().astimezone().isoformat();write(OUT/'status.json',status)


if __name__=='__main__':asyncio.run(main())
