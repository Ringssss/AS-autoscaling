"""Real-engine Kimi timing/length replay with explicit constructed state identity.

One independent handoff per window. No cache flush within a window. Short-output
screening and original-output confirmation are separate stages, never pooled.
"""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import time
import uuid

import aiohttp
import numpy as np

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.fluidkv.dualmap_adapter import DualMapAdapter
from experiments.multinode.benchmark_highbs import blocks, settle_engines, telemetry
from experiments.multinode.cacheroute import build_plan, dispatch

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'experiments/multinode/results/kimi-trace-replay'
POLICIES = ['p2c', 'p2c+fluidkv', 'dualmap', 'dualmap+fluidkv', 'cacheroute', 'cacheroute+fluidkv']


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def capacity_plan(rate, qcap, replicas):
    # Division of qcap*replicas can round just ABOVE the integer replica count.
    # Keep the saturated rate one representable step inside the valid boundary.
    admitted_rate=min(rate,math.nextafter(qcap*replicas,0.0))
    return build_plan({'shared':admitted_rate},replicas,qcap,replicas)


def coalesced_task(pending, key, make_coroutine):
    previous=pending.get(key)
    if previous is not None and not previous.done():
        return None
    task=asyncio.create_task(make_coroutine())
    pending[key]=task
    return task


def select_windows(data, seed=20260924):
    arrival = data['arrival']; origin = float(arrival.min()); duration = 10
    bins = ((arrival-origin)//duration).astype(int)
    counts = np.bincount(bins)
    pool = np.arange(60, len(counts)-1)
    rng = np.random.default_rng(seed)
    windows = []
    for label, q in [('low', .1), ('medium', .5), ('high', .9)]:
        distance = abs(counts[pool] - np.quantile(counts[pool], q))
        index = int(rng.choice(pool[distance == distance.min()]))
        indices = np.flatnonzero(bins == index)
        indices = indices[np.argsort(arrival[indices], kind='stable')]
        start = origin + duration*index
        train = (arrival >= start-60) & (arrival < start)
        rows = [dict(ordinal=j, native_index=int(i), native_arrival=float(arrival[i]),
                     offset=float((arrival[i]-origin)-duration*index), input_tokens=int(data['input_tokens'][i]),
                     output_tokens=int(data['output_tokens'][i]), observed_cached_tokens=int(data['cached_tokens'][i]))
                for j, i in enumerate(indices)]
        windows.append(dict(label=label, quantile=q, bin=index, start=start, duration=duration,
                            requests=rows, training_inputs=data['input_tokens'][train].tolist(),
                            selection='Seeded tie selection at count quantile; no latency/output/performance selection.'))
    return windows


def construct(window, corpus, length, cap):
    extended = (corpus * ((max(r['input_tokens'] for r in window['requests']) // len(corpus)) + 1))
    requests = []
    for row in window['requests']:
        n = row['input_tokens']; eligible = n > length
        tokens = extended[:n]
        # Exact same constructed input for all policies, repetition and output stages.
        # A unique marker immediately after the assumed shared history prevents
        # accidental sharing of the remainder; short requests have unique roots.
        tokens[length if eligible else 0] = 170000 + row['ordinal']
        requests.append(row | dict(tokens=tokens, hashes=blocks(tokens), eligible=eligible,
            requested_output=min(row['output_tokens'], cap) if cap else row['output_tokens'],
            input_sha256=hashlib.sha256(np.asarray(tokens,dtype='<i4').tobytes()).hexdigest()))
    return requests


async def inspect(session, url, parent):
    async with session.post(url+'/agentshift/hybrid_state/inspect', json={'token_ids':parent}) as response:
        response.raise_for_status()
        result = await response.json()
    if not result.get('success') or 'snapshot_boundaries' not in result:
        raise RuntimeError('authoritative checkpoint inspection unavailable')
    return result


async def trial(args, session, window, corpus, coordinators, store, calibration, spec):
    length, policy, cap = spec['length'], spec['policy'], spec['output_cap']
    requests = construct(window, corpus, length, cap)
    parent = corpus[:length]
    urls = args.engines[1:] if spec['withdraw'] else args.engines
    await settle_engines(session, args.engines)
    await flush(session, args.engines)
    parent_start = time.perf_counter()
    await stream(session,args.engines[0],parent,1)
    parent_seconds = time.perf_counter()-parent_start
    await settle_engines(session,args.engines)
    initial = await asyncio.gather(*(inspect(session,u,parent) for u in urls))
    base = policy.split('+')[0]
    adapter = DualMapAdapter(Path('/home/zhujianian/DualMap'), len(urls),262144,20480,10,
        calibration['prefill_seconds_per_token'],hybrid_snapshot_slots=512,
        pending_prefills=32,inspect_snapshots=True) if base == 'dualmap' else None
    if adapter:
        for i, obs in enumerate(initial): adapter.observe_snapshot_response(i,parent,obs)
    # qcap measured before evaluated windows, with fixed output cap; saturation
    # clips replication to fleet size without dropping any offered request.
    rate = sum(x>length for x in window['training_inputs'])/60
    plan = capacity_plan(rate,calibration['qcap'],len(urls))
    live = [0]*len(urls); records=[]; observations=[]; copies=[]; samples=[]
    rng=random.Random(args.seed+spec['repeat'])
    changed=asyncio.Event(); gate=asyncio.Event(); active=set(); obs_tasks=set(); pending_observations={}
    router_seconds=0.; prep_started=time.perf_counter(); ready=prep_started+spec['tool_seconds']
    prep_error=None

    async def prepare():
        nonlocal prep_error
        try:
            if policy.endswith('+fluidkv'):
                for target,c in coordinators.items():
                    name=uuid.uuid4().hex
                    store.register_agent(AgentContinuation(name,1,'source',1,tuple(parent)))
                    began=time.perf_counter(); result=await c.migrate(name,'destination')
                    copies.append(dict(target=target,seconds=time.perf_counter()-began,
                        commit_from_tool=time.perf_counter()-prep_started,bytes=result.bytes_transferred))
            elif policy.endswith('+precompute'):
                async def precompute(target):
                    began=time.perf_counter(); await stream(session,args.engines[target],parent,1)
                    copies.append(dict(target=target,seconds=time.perf_counter()-began,
                        commit_from_tool=time.perf_counter()-prep_started,bytes=0))
                await asyncio.gather(*(precompute(i) for i in range(1,len(args.engines))))
            if '+' in policy:
                after=await asyncio.gather(*(inspect(session,u,parent) for u in urls))
                observations.extend(dict(target=i,phase='prepared',result=o) for i,o in enumerate(after))
                if adapter:
                    for i,o in enumerate(after): adapter.observe_snapshot_response(i,parent,o)
        except Exception as exc:
            prep_error=repr(exc)
        finally:
            gate.set(); changed.set()
    prep_task=asyncio.create_task(prepare())
    stop=asyncio.Event(); monitor=asyncio.create_task(telemetry(session,urls,stop,samples,ready))

    def observe_later(target):
        # Inspection refreshes availability after natural warming/eviction.
        async def observe():
            try:
                response=await inspect(session,urls[target],parent)
                observations.append(dict(target=target,phase='serving',result=response))
                if adapter: adapter.observe_snapshot_response(target,parent,response)
            except Exception as exc:
                observations.append(dict(target=target,error=repr(exc)))
                if adapter: adapter.invalidate_snapshot_engine(target)
            finally: changed.set()
        task=coalesced_task(pending_observations,target,observe)
        if task is None: return
        obs_tasks.add(task)
        task.add_done_callback(obs_tasks.discard)

    async def serve(row,target,request=None):
        nonlocal router_seconds
        live[target]+=1; start=time.perf_counter(); first_at=None; error=None; result=None
        async def first(payload,delay):
            nonlocal first_at
            first_at=time.perf_counter()
            if adapter: await adapter.first_token(request,delay)
            observe_later(target); changed.set()
        try:
            result=await stream(session,urls[target],row['tokens'],row['requested_output'],on_first_token=first)
            if result['meta_info']['prompt_tokens']!=row['input_tokens'] or result['meta_info']['completion_tokens']!=row['requested_output']:
                raise ValueError('input/output length changed')
        except Exception as exc: error=repr(exc)
        end=time.perf_counter()
        if adapter: await adapter.finished(request,result['output_ids'] if result else [],error is None)
        live[target]-=1; observe_later(target)
        arrival=ready+row['offset']/spec['rate_scale']
        records.append({k:v for k,v in row.items() if k not in ('tokens','hashes')} | dict(
            target=target,engine_url=urls[target],success=error is None,error=error,
            dispatch_seconds=start-ready,client_queue_seconds=start-arrival,
            ttft_seconds=first_at-arrival if first_at else None,e2e_seconds=end-arrival,
            first_seconds=first_at-ready if first_at else None,finished_seconds=end-ready,
            engine=result))
        changed.set()

    async def arrivals():
        nonlocal router_seconds
        for row in requests:
            await asyncio.sleep(max(0,ready+row['offset']/spec['rate_scale']-time.perf_counter()))
            # Offered times remain unchanged even if preparation/admission delays
            # dispatch. This gate is a charged prototype constraint for all traffic.
            await gate.wait()
            if prep_error: raise RuntimeError(prep_error)
            began=time.perf_counter()
            if adapter:
                await adapter.enqueue(row['ordinal'],dict(chat_id=row['ordinal'],hash_ids=row['hashes'],turn=2),row['tokens'],row['requested_output'])
            else:
                if base=='p2c':
                    candidates=rng.sample(range(len(urls)),min(2,len(urls))); target=min(candidates,key=lambda i:live[i])
                elif base=='sticky': target=0
                else: target=dispatch(plan,'shared' if row['eligible'] else 'cold-'+str(row['ordinal']),live,rng)
                active.add(asyncio.create_task(serve(row,target)))
            router_seconds+=time.perf_counter()-began
            changed.set(); await asyncio.sleep(0)
        changed.set()

    arrival_task=asyncio.create_task(arrivals())
    deadline=ready+args.timeout
    try:
        while True:
            changed.clear()
            if arrival_task.done(): arrival_task.result()
            if adapter and gate.is_set() and not prep_error:
                began=time.perf_counter(); batch=await adapter.dispatch(); router_seconds+=time.perf_counter()-began
                for req in batch: active.add(asyncio.create_task(serve(requests[req._id],adapter.physical_target(req),req)))
            done={t for t in active if t.done()}
            for task in done: task.result()
            active.difference_update(done)
            if arrival_task.done() and not active and (not adapter or not adapter.queued()): break
            remaining=deadline-time.perf_counter()
            if remaining<=0: raise TimeoutError('whole-window timeout')
            await asyncio.wait_for(changed.wait(),timeout=remaining)
        await prep_task
        if obs_tasks: await asyncio.gather(*list(obs_tasks))
    finally:
        stop.set(); await monitor
        for task in [arrival_task,prep_task,*active,*obs_tasks]:
            if not task.done(): task.cancel()
        await asyncio.gather(arrival_task,prep_task,*active,*obs_tasks,return_exceptions=True)
    good=[r for r in records if r['success']]
    ttft=[r['ttft_seconds'] for r in good]; e2e=[r['e2e_seconds'] for r in good]
    duration=max(window['duration']/spec['rate_scale'],max(r['finished_seconds'] for r in records))
    summary=spec | dict(window=window['label'],offered=len(requests),successful=len(good),errors=len(records)-len(good),
        eligible=sum(r['eligible'] for r in requests),offered_duration=window['duration']/spec['rate_scale'],
        drain_inclusive_duration=duration,parent_setup_seconds=parent_seconds,
        transfer_bytes=sum(c['bytes'] for c in copies),preparation=copies,
        training_shared_rate=rate,cacheroute_plan=asdict(plan),qcap=calibration['qcap'],
        router_seconds=router_seconds,ttft_mean=float(np.mean(ttft)),e2e_mean=float(np.mean(e2e)),
        ttft_quantiles=np.percentile(ttft,[50,95,99]).tolist(),e2e_quantiles=np.percentile(e2e,[50,95,99]).tolist(),
        ttft_slo_10s=sum(t<=10 for t in ttft)/len(requests),
        goodput_10s=sum(t<=10 for t in ttft)/duration,throughput=len(good)/duration,
        observed_cached_tokens=sum(r['engine']['meta_info']['cached_tokens'] for r in good),
        total_input_tokens=sum(r['input_tokens'] for r in requests),
        total_requested_output=sum(r['requested_output'] for r in requests),
        native_output_tokens=sum(r['output_tokens'] for r in requests),
        truncated_outputs=sum(r['requested_output']!=r['output_tokens'] for r in requests),
        checkpoint_errors=sum('error' in o for o in observations))
    return dict(summary=summary,requests=records,observations=observations,telemetry=samples)


def screen_specs(windows):
    specs=[]
    for window in windows:
        for length,withdraw in [(5760,True),(16384,True),(65536,True),(5760,False)]:
            for policy in POLICIES:
                specs.append(dict(stage='screen',window=window['label'],length=length,withdraw=withdraw,
                    tool_seconds=.5,rate_scale=1.,output_cap=64,policy=policy,repeat=0))
    for scale, tool, length in [(.25,.5,5760),(4.,.5,5760),(1.,0.,16384),(1.,10.,16384)]:
        for policy in POLICIES:
            specs.append(dict(stage='screen',window='high',length=length,withdraw=True,
                tool_seconds=tool,rate_scale=scale,output_cap=64,policy=policy,repeat=0))
    return specs


async def main(args):
    args.output.mkdir(parents=True,exist_ok=False)
    save(args.output/'command.json',dict(argv=__import__('sys').argv,cwd=str(Path.cwd())))
    data=np.load(ROOT/'experiments/multinode/results/kimi-opportunity/native_numeric_fields.npz')
    windows=json.loads(args.windows_file.read_text()) if args.windows_file else select_windows(data)
    save(args.output/'windows.json',windows)
    corpus=np.load(ROOT/'experiments/multinode/data/longbench_tokens.npz')['tokens'].tolist()
    sources=['experiments/multinode/run_kimi_trace_replay.py','experiments/multinode/cacheroute.py','experiments/fluidkv/dualmap_adapter.py','agentshift/controller/hybrid_migration.py']
    save(args.output/'source_manifest.json',{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources})
    for p in sources:
        destination=args.output/'source_snapshot'/p
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((ROOT/p).read_bytes())
    store=SQLiteStateStore(args.output/'state.db')
    coordinators={i:HybridMigrationCoordinator(store,{'source':SGLangAgentShiftClient('source',args.engines[0],timeout=300),
        'destination':SGLangAgentShiftClient('destination',url,timeout=300)},master_address='12.11.3.27',base_port=args.base_port+8*(i-1))
        for i,url in enumerate(args.engines[1:],1)}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout),connector=aiohttp.TCPConnector(limit=0,force_close=True)) as session:
        info=[]
        for u in args.engines:
            async with session.get(u+'/get_server_info') as r: r.raise_for_status(); info.append(await r.json())
        save(args.output/'server_info.json',info)
        if args.calibration_file:
            calibration=json.loads(args.calibration_file.read_text())
            calibration['reuse_source']=str(args.calibration_file)
            save(args.output/'calibration.json',calibration)
        else:
            print('CALIBRATION',flush=True)
            native_calibration = args.stage == 'native' or (args.spec_file and all(
                s['output_cap'] == 0 for s in json.loads(args.spec_file.read_text())))
            origin=float(data['arrival'].min())
            assert min(w['start'] for w in windows) >= origin+600
            prior=np.argsort(data['arrival'],kind='stable')
            prior=prior[(data['arrival'][prior]>=origin+300)&(data['arrival'][prior]<origin+600)][:16]
            assert len(prior)==16
            calibration_window={'requests':[dict(ordinal=j,native_index=int(i),
                input_tokens=int(data['input_tokens'][i]),output_tokens=int(data['output_tokens'][i]))
                for j,i in enumerate(prior)]}
            calibration_requests=construct(calibration_window,corpus,5760,0 if native_calibration else 64)
            times=[]
            for n in [1024,8192]:
                await settle_engines(session,args.engines); await flush(session,[args.engines[0]])
                result=await stream(session,args.engines[0],corpus[:n],1);times.append(result['ttft_seconds'])
            await settle_engines(session,args.engines); await flush(session,[args.engines[0]])
            await stream(session,args.engines[0],corpus[:5760],1)
            began=time.perf_counter()
            calibration_results=await asyncio.gather(*(stream(session,args.engines[0],r['tokens'],r['requested_output']) for r in calibration_requests))
            capacity_seconds=time.perf_counter()-began
            calibration=dict(prefill_seconds_per_token=max(1e-6,(times[1]-times[0])/7168),qcap=16/capacity_seconds,raw_prefill_times=times,
                capacity_seconds=capacity_seconds,
                calibration_rule='First 16 arrivals in [origin+300,origin+600), original input lengths, same constructed 5760 history. One-engine concurrent batch goodput including drain; not a steady-state capacity claim.',
                calibration_output_cap=0 if native_calibration else 64,
                calibration_requests=[{k:v for k,v in r.items() if k not in ('tokens','hashes')} for r in calibration_requests],
                calibration_results=calibration_results)
            save(args.output/'calibration.json',calibration)
        for i,c in coordinators.items():
            await settle_engines(session,args.engines); await flush(session,args.engines)
            await stream(session,args.engines[0],corpus[:1024],1);await settle_engines(session,args.engines)
            name=uuid.uuid4().hex;store.register_agent(AgentContinuation(name,1,'source',1,tuple(corpus[:1024])))
            began=time.perf_counter();await c.migrate(name,'destination')
            print(json.dumps(dict(group_warmup=i,seconds=time.perf_counter()-began)),flush=True)
        specs=screen_specs(windows)
        if args.smoke: specs=[s for s in specs if s['window']=='low' and s['length']==5760 and s['withdraw']]
        if args.stage=='native':
            specs=[dict(stage='native',window=w['label'],length=5760,withdraw=True,tool_seconds=.5,rate_scale=1.,output_cap=0,policy=p,repeat=0)
                   for w in windows for p in POLICIES]
        if args.spec_file: specs=json.loads(args.spec_file.read_text())
        if not args.preserve_job_order: random.Random(args.seed).shuffle(specs)
        save(args.output/'predeclared_jobs.json',specs)
        summaries=[]
        with (args.output/'summary.jsonl').open('w',buffering=1) as log:
            for index,spec in enumerate(specs):
                name=f'{index:03d}-{spec["stage"]}-{spec["window"]}-{spec["length"]}-{spec["policy"]}-r{spec["repeat"]}'
                print('START '+name,flush=True)
                try:
                    result=await trial(args,session,next(w for w in windows if w['label']==spec['window']),corpus,coordinators,store,calibration,spec)
                except Exception as exc:
                    save(args.output/(name+'-failure.json'),dict(spec=spec,error=repr(exc)))
                    raise
                save(args.output/(name+'.json'),result);summaries.append(result['summary'])
                log.write(json.dumps(result['summary'])+'\n')
                print('DONE '+json.dumps(result['summary']),flush=True)
        await settle_engines(session,args.engines)
        save(args.output/'summary.json',summaries)
        (args.output/'COMPLETE').write_text('All jobs complete; engines drained.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--engines',nargs='+',default=[f'http://12.11.3.{i}:31600' for i in (27,28,29)])
    p.add_argument('--base-port',type=int,default=25700)
    p.add_argument('--seed',type=int,default=20260924)
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--stage',choices=['screen','native'],default='screen')
    p.add_argument('--spec-file',type=Path)
    p.add_argument('--windows-file',type=Path)
    p.add_argument('--calibration-file',type=Path)
    p.add_argument('--preserve-job-order',action='store_true')
    p.add_argument('--smoke',action='store_true')
    asyncio.run(main(p.parse_args()))
