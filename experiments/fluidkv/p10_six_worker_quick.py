"""Bounded six-worker mechanism test; synthetic bursts, not native AgentX."""
import argparse
import asyncio
from collections import Counter
import contextlib
import hashlib
import json
from pathlib import Path
import random
import shlex
import time

import aiohttp
import numpy as np

from experiments.fluidkv import p07_routing_space as pilot
from experiments.fluidkv.p08_campaign import write
from experiments.fluidkv.p08_upstream_routers import UpstreamRouter
from experiments.fluidkv.p09_make_workload import make

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path('/tmp/fluidkv-blitz-upstream-20261010/target/release/fluidkv-policy-bridge')


class MappedFeedback:
    def __init__(self, router, logical_to_physical):
        self.router = router
        self.inverse = {p:l for l,p in enumerate(logical_to_physical)}

    def step(self, physical, event):
        self.router.step(self.inverse[physical], event)


async def trial(session, urls, case, tokens, policy, prep, rep, cal, telemetry, groups, out):
    await pilot.skew.reset(session, urls)
    def ids(row):
        off=row['token_offset']
        return tokens[off:off+row['input_tokens']].astype(int).tolist()
    parent=ids(case['streams'][0]['requests'][0])
    router=UpstreamRouter(policy,cal,[[] for _ in urls],telemetry,
        publish_prefill_prefix=len(parent),blitz_binary=BINARY)
    mapping=list(range(len(urls)))
    rows=[]
    preparation_tasks=[]
    try:
        # Let the actual policy choose the parent's logical worker. Normalize
        # physical labels BEFORE serving it so neither policy gets a foreign
        # cache placement that its own routing history could never create.
        target,parent_decision=await router.choose('parent',np.asarray(parent,dtype=np.int32),8,
            dict(case['streams'][0]['requests'][0],chat_id='parent',turn=0))
        mapping[0],mapping[target]=mapping[target],mapping[0]
        telemetry.router=MappedFeedback(router,mapping)
        r=await pilot.generate(session,urls[0],parent,8,
            callback=lambda p,n,o:router.progress(target,'parent',p,n,o))
        await router.completed(target,'parent',parent,r)
        state=await pilot.inspect(session,urls[0],parent)
        if state['matched_tokens']!=len(parent):
            raise RuntimeError('parent boundary missing')
        await pilot.skew.drain(session,urls)
        telemetry.peaks=[0 for _ in urls]
        before_events=list(telemetry.events)
        preparation=[]
        prep_start=time.perf_counter()
        arrival=prep_start+1.
        prep_done=prep_start
        async def recompute(logical,physical):
            nonlocal prep_done
            rid=f'prepare-{physical}'
            await router.admit_preparation(parent,logical,rid)
            result=await pilot.generate(session,urls[physical],parent,1,
                callback=lambda p,n,o:router.progress(logical,rid,p,n,o))
            await router.completed(logical,rid,parent,result)
            prep_done=time.perf_counter()
            preparation.append(dict(physical=physical,method='recompute',result=result))
        if prep=='copy':
            for physical in range(1,len(urls)):
                transfer=await pilot.skew.p03.copy_state(session,groups,urls,0,physical,parent)
                if not transfer.get('ok'):
                    raise RuntimeError(transfer)
                observed=await pilot.inspect(session,urls[physical],parent)
                if observed['matched_tokens']!=len(parent):
                    raise RuntimeError('copied prefix not executable')
                router.cache[mapping.index(physical)].append(np.asarray(parent,dtype=np.int32))
                preparation.append(dict(physical=physical,method='copy',**transfer))
            prep_done=time.perf_counter()
        elif prep in ('recompute','recompute_wait'):
            preparation_tasks=[asyncio.create_task(recompute(mapping.index(p),p))
                for p in range(1,len(urls))]
            if prep=='recompute_wait':
                # Strong control: the consumer waits for an executable copy
                # instead of entering an engine while its producer is still
                # prefilling. Arrival stays fixed, so the wait is fully paid.
                await asyncio.gather(*preparation_tasks)
        elif prep!='none':
            raise ValueError(prep)
        await asyncio.sleep(max(0,arrival-time.perf_counter()))
        tasks=[]
        async def consumer(index,stream):
            eligible=arrival
            for turn,row in enumerate(stream['requests'][1:]):
                await asyncio.sleep(max(0,eligible-time.perf_counter()))
                prompt=ids(row)
                rid=f'{index}:{turn}'
                logical,decision=await router.choose(rid,np.asarray(prompt,dtype=np.int32),128,
                    dict(row,chat_id=stream['key'],turn=turn+1))
                physical=mapping[logical]
                result=await pilot.generate(session,urls[physical],prompt,128,eligible,
                    callback=lambda p,n,o:router.progress(logical,rid,p,n,o))
                await router.completed(logical,rid,prompt,result)
                saved=dict(clients=len(case['streams']),policy=policy,prep=prep,repetition=rep,
                    consumer=index,turn=turn,target=physical,logical_target=logical,
                    input_tokens=len(prompt),decision=decision,**result)
                rows.append(saved)
                with (out/'requests.jsonl').open('a') as log:
                    log.write(json.dumps(saved)+'\n')
                eligible=result['completed_at']+.1
            tasks.append(result['completed_at']-arrival)
        await asyncio.gather(*(consumer(i,s) for i,s in enumerate(case['streams'])))
        await asyncio.gather(*preparation_tasks)
        ended=time.perf_counter()
        telemetry.router=None
        await pilot.skew.drain(session,urls)
        summary=pilot.skew.summarize(rows,ended-arrival,10.,.2)
        summary.update(task_p50=float(np.median(tasks)),task_p90=float(np.percentile(tasks,90)),
            paid_wall=ended-prep_start,offered_clients=len(case['streams']))
        mismatches=[c for c in router.corrections if c['predicted']>0 and
            (c['difference']<0 or 16<=c['difference']<=128)]
        audits=[]
        for physical,url in enumerate(urls):
            actual=await pilot.inspect(session,url,parent)
            audits.append(dict(physical=physical,actual=actual['matched_tokens'],
                predicted=router.hit(mapping.index(physical),np.asarray(parent,dtype=np.int32))))
        return dict(clients=len(case['streams']),policy=policy,prep=prep,repetition=rep,
            summary=summary,preparation=preparation,preparation_seconds=prep_done-prep_start,
            exposed_preparation=max(0,prep_done-arrival),logical_to_physical=mapping,
            parent_decision=parent_decision,peak_decode=list(telemetry.peaks),
            telemetry_events=[a-b for a,b in zip(telemetry.events,before_events)],
            cache_mismatches=mismatches,corrections=router.corrections,final_cache_audit=audits,
            dispatch_counts=dict(Counter(r['target'] for r in rows)),
            client_router_seconds=sum(r['decision']['router_seconds'] for r in rows))
    finally:
        telemetry.router=None
        for task in preparation_tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*preparation_tasks,return_exceptions=True)
        await router.close()


async def cleanup(fleet,out):
    records=[]
    for rec in fleet:
        remote=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',rec['node']]
        log_error=None
        with (out/(rec['name']+'.engine.log')).open('w') as log:
            p=await asyncio.create_subprocess_exec(*remote,shlex.join(['docker','logs',rec['container']]),
                stdout=log,stderr=asyncio.subprocess.STDOUT)
            try:
                await asyncio.wait_for(p.wait(),30)
            except asyncio.TimeoutError:
                log_error='log collection timed out; continue exact-container cleanup'
                with contextlib.suppress(ProcessLookupError):p.kill()
                await p.wait()
        p=await asyncio.create_subprocess_exec(*remote,shlex.join(['docker','rm','-f',rec['container']]),
            stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
        stdout,stderr=await asyncio.wait_for(p.communicate(),30)
        records.append(dict(name=rec['name'],code=p.returncode,out=stdout.decode(),err=stderr.decode(),
            log_error=log_error))
        write(out/'cleanup.json',records)
    write(out/'cleanup.json',records)


async def run(args):
    out=args.out
    fleet=json.loads((out/'fleet.json').read_text())
    assert len(fleet)==6
    urls=[r['url'] for r in fleet]
    data=make(out/'workload',1,65536,args.levels,seed=20261012)
    tokens=np.load(out/'workload/cohort-tokens.npy',mmap_mode='r')
    manifest=dict(status='starting',trials=[],levels=args.levels,repeats=args.reps,
        scope='Synthetic cold-expansion burst: 64K shared prefix, 64-token private growth/turn, '
        '128 output tokens, two turns/client, 0.1s turn gap, 1s preparation lead. '
        'Not AgentX, not steady-state workload evidence. Six identical TP2 engine settings. '
        'Copy/recompute prepare all five destinations; all paid cost is included. '
        'Only group connection setup is outside the service window and reported separately.',
        policy_scope='Pinned original Blitz Rust LmetricQ and original DualMap queues/dispatch '
        'on common SGLang backend, with common confirmed executable-prefix directory. '
        'Not comparisons of native complete stacks. Parent follows original policy.',
        upstream=dict(blitz='a005ceb3a6689d30514bd27e45ede0db933d0fea',
                      dualmap='24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d'),
        node28_scope='Existing idle resident GPU contexts retained; resource snapshots recorded.',
        hashes={})
    for file in [Path(__file__),Path(pilot.__file__),ROOT/'experiments/fluidkv/p08_upstream_routers.py',
                 ROOT/'experiments/fluidkv/dualmap_adapter.py',BINARY]:
        manifest['hashes'][str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
        if file.suffix=='.py':(out/file.name).write_bytes(file.read_bytes())
    def save():write(out/'summary.json',manifest)
    save()
    telemetry=pilot.Telemetry(fleet,out/'telemetry.jsonl')
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),
                connector=aiohttp.TCPConnector(limit=0,force_close=True)) as session:
            deadline=time.perf_counter()+900
            for url in urls:
                while True:
                    try:
                        async with session.get(url+'/health',timeout=aiohttp.ClientTimeout(total=3)) as r:
                            if r.status==200:break
                    except (aiohttp.ClientError,asyncio.TimeoutError):pass
                    if time.perf_counter()>deadline:raise RuntimeError('worker startup timeout: '+url)
                    await asyncio.sleep(3)
            await telemetry.start()
            manifest['status']='warmup_and_gates';save()
            parent=tokens[:65536].astype(int).tolist()
            warm_records=[]
            # Equal warmup, first-token visibility, completed-boundary gate and
            # per-node prefill calibration. All outside measured bursts.
            async def warm_worker(i,url):
                print('WARM_GATE',i,flush=True)
                await pilot.skew.reset(session,[url])
                await pilot.generate(session,url,parent,8)
                await pilot.skew.reset(session,[url])
                prompt=parent+np.random.default_rng(812).integers(1000,240000,64).tolist()
                first=[]
                async def observe(payload,n,old):
                    if old==0:first.append(await pilot.inspect(session,url,parent))
                result=await pilot.generate(session,url,prompt,16,callback=observe)
                completed=await pilot.inspect(session,url,prompt)
                if first[0]['matched_tokens']!=65536 or completed['matched_tokens']!=65600:
                    raise RuntimeError('cache publication gate failed')
                warm_records.append(dict(worker=i,result=result,first=first,completed=completed))
                # Warm a high batch on every physical worker. Include the
                # concentrated-owner shape as well as six-way spread shapes.
                for batch in (16,32,64):
                    await asyncio.gather(*(pilot.generate(session,url,parent+
                        np.random.default_rng(12000+j).integers(1000,240000,64).tolist(),8)
                        for j in range(batch)))
            await asyncio.gather(*(warm_worker(i,url) for i,url in enumerate(urls)))
            write(out/'warmup-gates.json',warm_records)
            cal=dict(prefill_points=[[65536,float(np.median([r['result']['ttft'] for r in warm_records]))]])
            manifest['calibration']=cal
            if min(telemetry.events)==0:raise RuntimeError('missing forward metrics')
            await pilot.skew.reset(session,urls)
            groups=pilot.skew.p03.GroupCache()
            init=[]
            for target in range(1,6):
                print('INIT_AND_COPY_GATE',target,flush=True)
                started=time.perf_counter()
                if await groups.ensure(session,urls,0,target) is None:raise RuntimeError(groups.last_error)
                init.append(dict(target=target,seconds=time.perf_counter()-started))
                gate=out/f'gate-{target}';gate.mkdir()
                # Reuse the initialized pair with copy_gate's two-worker keys.
                view=pilot.skew.p03.GroupCache()
                view.groups[(0,1)]=groups.groups[(0,target)]
                view.init_seconds[(0,1)]=groups.init_seconds[(0,target)]
                await pilot.copy_gate(session,[urls[0],urls[target]],data['cohorts'][0],tokens,view,gate)
                manifest['group_initialization']=init;save()
            arms=[('blitz','none'),('dualmap','none'),('blitz','copy'),('blitz','recompute')]
            rng=random.Random(20261012)
            schedule=[(case,rep,policy,prep) for case in data['cohorts'] for rep in range(args.reps)
                for policy,prep in rng.sample(arms,len(arms))]
            manifest['schedule']=[dict(clients=c['clients'],rep=r,policy=p,prep=a) for c,r,p,a in schedule]
            manifest['status']='running';save()
            for case,rep,policy,prep in schedule:
                print('TRIAL',case['clients'],rep,policy,prep,flush=True)
                result=await trial(session,urls,case,tokens,policy,prep,rep,cal,telemetry,groups,out)
                manifest['trials'].append(result);save()
                print('RESULT',json.dumps({k:result[k] for k in ('clients','policy','prep','repetition','summary','preparation_seconds','cache_mismatches')}),flush=True)
                if result['cache_mismatches']:raise RuntimeError('cache prediction invalid; comparison stopped')
                if telemetry.errors:raise RuntimeError('telemetry disconnected')
            manifest['status']='complete'
    except BaseException as exc:
        manifest['status']='failed';manifest['failure']=repr(exc)
        raise
    finally:
        save()
        await telemetry.close()
        await cleanup(fleet,out)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--levels',nargs='+',type=int,default=[32,96,192])
    parser.add_argument('--reps',type=int,default=2)
    asyncio.run(run(parser.parse_args()))
