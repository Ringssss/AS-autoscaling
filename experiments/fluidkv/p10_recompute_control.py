"""Reuse the completed campaign fleet for a dependency-aware recompute control."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import aiohttp
import numpy as np

from experiments.fluidkv import p07_routing_space as pilot
from experiments.fluidkv.p08_campaign import write
from experiments.fluidkv.p10_six_worker_quick import trial
from experiments.fluidkv import p10_six_worker_quick as driver
from experiments.fluidkv.p10_cached_observations import CachedObservationRouter


async def run(args):
    base=args.base
    out=base/'recompute-wait-control'
    out.mkdir(exist_ok=False)
    status=dict(status='waiting_for_main',trials=[],reason='The asynchronous recompute arm '
        'admitted some consumers before the shared prefix was ready, causing duplicate cold prefill. '
        'This control pays the readiness wait before dispatch. Same arrivals, model, fleet and tokens.',
        main_pid=args.pid,levels=[32,96,192],repeats=2,
        extra_baseline='DualMap with memoized exact cache observations; original policy unchanged.')
    write(out/'summary.json',status)
    paused=False
    telemetry=None
    try:
        # Freeze only OUR completed measurement process, during its finally
        # cleanup. No measurement is paused. This preserves the six ready
        # workers for the additional control without another model reload.
        deadline=time.monotonic()+7200
        modified=None
        while True:
            current=(base/'summary.json').stat().st_mtime_ns
            if current!=modified:
                d=json.loads((base/'summary.json').read_text())
                modified=current
            if d['status']=='failed' and 'TimeoutError' not in d.get('failure',''):
                raise RuntimeError('main campaign failed a non-timeout gate')
            if d['status'] in ('complete','failed'):
                os.kill(args.pid,signal.SIGSTOP)
                paused=True
                status['main_status']=d['status']
                status['main_failure']=d.get('failure')
                break
            if time.monotonic()>deadline:raise TimeoutError('main wait deadline')
            await asyncio.sleep(.01)
        fleet=json.loads((base/'fleet.json').read_text())
        urls=[r['url'] for r in fleet]
        tokens=np.load(base/'workload/cohort-tokens.npy',mmap_mode='r')
        data=json.loads((base/'workload/cohorts.json').read_text())
        telemetry=pilot.Telemetry(fleet,out/'telemetry.jsonl')
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),
                connector=aiohttp.TCPConnector(limit=0,force_close=True)) as session:
            for url in urls:
                async with session.get(url+'/health') as response:response.raise_for_status()
            await telemetry.start()
            status['status']='running'
            status['trial_driver_sha256']=hashlib.sha256(Path(trial.__code__.co_filename).read_bytes()).hexdigest()
            (out/'driver.py').write_bytes(Path(trial.__code__.co_filename).read_bytes())
            write(out/'summary.json',status)
            groups=pilot.skew.p03.GroupCache()
            driver.UpstreamRouter=CachedObservationRouter
            done={(t['clients'],t['repetition'],t['policy'],t['prep']) for t in d['trials']}
            missing=[(c,rep,p) for c in data['cohorts'] for rep in range(2)
                for p in ('none','copy','recompute') if (c['clients'],rep,'blitz',p) not in done]
            status['remaining_main_cells']=[dict(clients=c['clients'],rep=r,prep=p) for c,r,p in missing]
            if any(p=='copy' for _,_,p in missing):
                await pilot.skew.reset(session,urls)
                start=time.perf_counter()
                for target in range(1,6):
                    if await groups.ensure(session,urls,0,target) is None:raise RuntimeError(groups.last_error)
                status['additional_group_initialization_seconds']=time.perf_counter()-start
            for case,rep,prep in missing:
                print('FINISH_MAIN_CELL',case['clients'],rep,prep,flush=True)
                result=await trial(session,urls,case,tokens,'blitz',prep,rep,
                    d['calibration'],telemetry,groups,out)
                result['resumed_main_cell']=True
                status['trials'].append(result);write(out/'summary.json',status)
                if result['cache_mismatches']:raise RuntimeError('cache mismatch')
            for case in data['cohorts']:
                for rep in range(2):
                    print('CONTROL',case['clients'],rep,flush=True)
                    result=await trial(session,urls,case,tokens,'blitz','recompute_wait',rep,
                        d['calibration'],telemetry,groups,out)
                    status['trials'].append(result);write(out/'summary.json',status)
                    print('RESULT',case['clients'],rep,result['summary'],flush=True)
                    if result['cache_mismatches']:raise RuntimeError('cache mismatch')
                    if telemetry.errors:raise RuntimeError('telemetry disconnected')
                    print('OPTIMIZED_DUALMAP',case['clients'],rep,flush=True)
                    result=await trial(session,urls,case,tokens,'dualmap','none',rep,
                        d['calibration'],telemetry,groups,out)
                    result['observation_adapter']='memoized immutable input/cache-version lookup'
                    status['trials'].append(result);write(out/'summary.json',status)
                    print('RESULT',case['clients'],rep,'dualmap',result['summary'],flush=True)
                    if result['cache_mismatches']:raise RuntimeError('cache mismatch')
                    if telemetry.errors:raise RuntimeError('telemetry disconnected')
            status['status']='complete'
    except BaseException as exc:
        status['status']='failed';status['failure']=repr(exc)
        raise
    finally:
        if telemetry:await telemetry.close()
        write(out/'summary.json',status)
        if paused:
            # The main process resumes its original exact-container cleanup.
            os.kill(args.pid,signal.SIGCONT)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--pid',type=int,required=True)
    asyncio.run(run(parser.parse_args()))
