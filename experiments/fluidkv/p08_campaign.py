"""Execute paid preparation versus pinned upstream routing, with auditable gates."""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time

import aiohttp
import numpy as np

from experiments.fluidkv import p07_routing_space as pilot
from experiments.fluidkv.p08_upstream_routers import UpstreamRouter, BLITZ_BINARY


ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


async def warm_case(session, urls, case, tokens):
    """Compile the tested prompt/batch shapes on both workers, then discard KV."""
    def ids(row):
        off=row['token_offset']
        return tokens[off:off+row['input_tokens']].astype(int).tolist()
    for url in urls:
        await pilot.skew.reset(session,[url])
        parents={s['requests'][0]['token_sha256']:s['requests'][0] for s in case['streams']}
        for row in parents.values():
            await pilot.generate(session,url,ids(row),8)
        for turn in range(1,len(case['streams'][0]['requests'])):
            await asyncio.gather(*(pilot.generate(session,url,ids(s['requests'][turn]),4)
                for s in case['streams']))
    await pilot.skew.reset(session,urls)


async def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    data = json.loads((args.cohorts/'cohorts.json').read_text())
    tokens = np.load(args.cohorts/'cohort-tokens.npy', mmap_mode='r')
    cases = [c for c in data['cohorts'] if not args.cases or c['name'] in args.cases]
    if not cases:
        raise ValueError('no selected cases')
    fleet = json.loads(args.fleet.read_text())
    if len(fleet) != 2:
        raise ValueError('this paired campaign requires exactly two workers')
    args.urls = [r['url'] for r in fleet]
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    manifest = dict(config=config, workload_scope=data['scope'], trials=[],
        status='starting',
        policy_scope='Pinned upstream routing algorithms on a shared SGLang transport. '
        'Blitz executes Rust LmetricQ and admission; DualMap executes queues, dispatch and rebalance. '
        'Both receive the same conservative cache estimate. Not native full-stack comparisons.',
        semantics='Skew is a declared initial placement, not measured native worker homes. '
        'Paid copy/recompute starts before the declared consumer arrival gap. '
        'All consumer and background requests enter the outcome, including failures. '
        'Communication initialization is recorded outside the warm-connection service window.',
        upstream={'dualmap':'24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d',
                  'blitz':'a005ceb3a6689d30514bd27e45ede0db933d0fea'},
        hashes={})
    for path in [Path(__file__),Path(pilot.__file__),
                 ROOT/'experiments/fluidkv/p08_upstream_routers.py',
                 ROOT/'experiments/weka_baselines/environment/runtime/managers/hygon_hybrid_control.py',
                 args.cohorts/'cohorts.json']:
        manifest['hashes'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix=='.py':
            (args.out/path.name).write_bytes(path.read_bytes())
    if 'blitz' in args.policies:
        manifest['hashes']['blitz_binary']=hashlib.sha256(args.blitz_binary.read_bytes()).hexdigest()
    write(args.out/'manifest.json',manifest)
    telemetry = pilot.Telemetry(fleet,args.out/'telemetry.jsonl')
    routers=[]
    def factory(*values):
        router=UpstreamRouter(*values,pending_prefills=args.dualmap_pending,
            publish_prefill_prefix=args.publish_prefill_prefix,blitz_binary=args.blitz_binary)
        routers.append(router)
        return router
    args.router_factory=factory
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),
            connector=aiohttp.TCPConnector(limit=0,force_close=True)) as session:
        manifest['http_connection_limit']=session.connector.limit
        manifest['http_force_close']=session.connector.force_close
        write(args.out/'manifest.json',manifest)
        try:
            manifest['server_info']=[]
            for url in args.urls:
                async with session.get(url+'/get_server_info') as response:
                    response.raise_for_status()
                    manifest['server_info'].append(await response.json())
            await telemetry.start()
            if args.calibration:
                cal=json.loads(args.calibration.read_text())
                # Equal warmup after server launch, independent of trial order.
                warm=tokens[:32768].astype(int).tolist()
                for url in args.urls:
                    await pilot.skew.reset(session,[url])
                    await pilot.generate(session,url,warm,8)
            else:
                source=np.load(ROOT/'experiments/weka_baselines/data/tokens.npy',mmap_mode='r')
                cal=await pilot.calibration(session,args.urls,source,args.out)
            manifest['calibration']=cal
            if args.publish_prefill_prefix:
                confirmations=[]
                firstcase=cases[0]
                for url in args.urls:
                    await pilot.skew.reset(session,[url])
                    row=firstcase['streams'][0]['requests'][1]
                    off=row['token_offset']
                    ids=tokens[off:off+row['input_tokens']].astype(int).tolist()
                    observed=[]
                    async def confirm(payload,count,old):
                        if old==0:
                            state=await pilot.inspect(session,url,ids[:args.publish_prefill_prefix])
                            observed.append(state)
                            if state['matched_tokens']<args.publish_prefill_prefix:
                                raise RuntimeError('cold producer did not publish the family prefix by first token')
                    r=await pilot.generate(session,url,ids,32,callback=confirm)
                    completed_state=await pilot.inspect(session,url,ids)
                    if completed_state['matched_tokens']<len(ids):
                        raise RuntimeError('completed input boundary is not executable; cache adapter must not publish it')
                    confirmations.append(dict(url=url,observation=observed,result=r,
                        completed_input_state=completed_state))
                write(args.out/'first-token-publication-gate.json',confirmations)
                await pilot.skew.reset(session,args.urls)
            await asyncio.sleep(.5)
            if min(telemetry.events)==0:
                raise RuntimeError(f'missing forward progress: {telemetry.errors}')
            group=pilot.skew.p03.GroupCache()
            if 'copy' in args.preps:
                await pilot.skew.reset(session,args.urls)
                start=time.perf_counter()
                if await group.ensure(session,args.urls,0,1) is None:
                    raise RuntimeError(group.last_error)
                manifest['group_init_seconds']=time.perf_counter()-start
                print('COPY_CORRECTNESS_GATE',flush=True)
                await pilot.copy_gate(session,args.urls,cases[0],tokens,group,args.out)
            random.seed(args.seed)
            cells=[(c,p,a) for c in cases for p in args.policies for a in args.preps]
            ordering=random.Random(args.seed)
            if args.case_blocks:
                schedule=[(rep,c,p,a) for c in cases for rep in range(args.reps)
                    for p,a in ordering.sample([(p,a) for p in args.policies for a in args.preps],
                                              len(args.policies)*len(args.preps))]
            else:
                schedule=[(rep,*cell) for rep in range(args.reps)
                    for cell in ordering.sample(cells,len(cells))]
            manifest['schedule']=[dict(repetition=r,case=c['name'],policy=p,prep=a)
                for r,c,p,a in schedule]
            manifest['status']='running'
            write(args.out/'summary.json',manifest)
            warmed=set()
            for rep,case,policy,prep in schedule:
                if args.warm_cases and case['name'] not in warmed:
                    print('WARM_CASE',case['name'],flush=True)
                    await warm_case(session,args.urls,case,tokens)
                    warmed.add(case['name'])
                print('TRIAL',rep,case['name'],policy,prep,flush=True)
                start=time.perf_counter()
                result=await pilot.trial(session,args,case,policy,prep,rep,tokens,cal,telemetry,group)
                result['total_trial_wall']=time.perf_counter()-start
                result['preparation_lateness_s']=result['exposed_preparation']
                result['preparation_blocking_s']=(result['exposed_preparation'] if prep=='copy' else 0.)
                result['offered_clients']=len(case['streams'])
                manifest['trials'].append(result)
                write(args.out/'summary.json',manifest)
                print('RESULT',json.dumps({k:result[k] for k in
                    ('case','policy','prep','summary','preparation_wall','exposed_preparation')}),flush=True)
                if result['summary']['errors']:
                    raise RuntimeError('request errors: retain failed cell, stop performance comparison')
                if args.require_exact_hits:
                    # A cold admission can legitimately hit state created by
                    # an earlier in-flight producer before it starts executing.
                    mismatches=[c for c in result['corrections']
                        if c['predicted']>0 and (c['difference']<0 or 16<=c['difference']<=128)]
                    if mismatches:
                        manifest['cache_mismatches']=mismatches
                        raise RuntimeError(f'cache estimate mismatch in {len(mismatches)} requests; no performance verdict')
            manifest['status']='complete'
        except BaseException as exc:
            manifest['status']='failed'
            manifest['failure']=repr(exc)
            raise
        finally:
            for router in routers:
                await router.close()
            await telemetry.close()
            manifest['telemetry_errors']=telemetry.errors
            write(args.out/'summary.json',manifest)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fleet',type=Path,required=True)
    ap.add_argument('--cohorts',type=Path,default=ROOT/'results/routing-space-20261010')
    ap.add_argument('--cases',nargs='+')
    ap.add_argument('--policies',nargs='+',choices=['blitz','dualmap'],default=['blitz','dualmap'])
    ap.add_argument('--preps',nargs='+',choices=['none','copy','recompute','ideal','ideal_common'],
        default=['none','copy','recompute'])
    ap.add_argument('--reps',type=int,default=3)
    ap.add_argument('--gap',type=float,default=1.)
    ap.add_argument('--placement',choices=['skew','balanced'],default='skew')
    ap.add_argument('--output-cap',type=int,default=128)
    ap.add_argument('--lookahead',type=int,default=128)
    ap.add_argument('--background-requests',type=int,default=0)
    ap.add_argument('--background-output',type=int,default=512)
    ap.add_argument('--ttft-slo',type=float,default=10.)
    ap.add_argument('--tpot-slo',type=float,default=.2)
    ap.add_argument('--seed',type=int,default=20261011)
    ap.add_argument('--dualmap-pending',type=int,default=1)
    ap.add_argument('--case-blocks',action='store_true')
    ap.add_argument('--warm-cases',action='store_true')
    ap.add_argument('--publish-prefill-prefix',type=int,default=0)
    ap.add_argument('--blitz-binary',type=Path,
        default=Path('/tmp/fluidkv-blitz-upstream-20261010/target/release/fluidkv-policy-bridge'))
    ap.add_argument('--require-exact-hits',action='store_true')
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    asyncio.run(run(ap.parse_args()))


if __name__=='__main__':
    main()
