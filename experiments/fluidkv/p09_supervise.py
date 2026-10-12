"""Run a concurrency screen, confirm candidate neighborhoods, retain all data."""
import asyncio
from datetime import datetime,timezone
import json
from pathlib import Path
import shlex
import sys
import time

import aiohttp

from experiments.fluidkv.p08_campaign import write
from experiments.fluidkv.p09_report import report,comparisons


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/concurrency-sweep-20261011'
LEVELS=[4,8,16,32,64,128]


async def main():
    status=dict(started=datetime.now(timezone.utc).isoformat(),status='checking_workers',stages=[])
    write(OUT/'campaign-status.json',status)
    fleets={name:json.loads((OUT/f'fleet-{name}.json').read_text()) for name in ('local','node30')}
    jobs={}
    specs=[('shared','local',65536,1),('mixed','node30',32768,4)]
    def command(kind,fleet,length,name,extra):
        return [sys.executable,'-u','-m','experiments.fluidkv.p08_campaign',
            '--fleet',str(OUT/f'fleet-{fleet}.json'),'--cohorts',str(OUT/f'{kind}-data'),
            '--calibration',str(ROOT/f'results/state-preparation-20261011/holdout-{fleet}/calibration.json'),
            '--out',str(OUT/name),'--case-blocks','--warm-cases',
            '--publish-prefill-prefix',str(length),'--require-exact-hits',*extra]
    async def execute(name,cmd):
        rec=dict(name=name,status='running',started=datetime.now(timezone.utc).isoformat(),command=cmd)
        status['stages'].append(rec);write(OUT/'campaign-status.json',status)
        with (OUT/(name+'.log')).open('w') as log:
            proc=await asyncio.create_subprocess_exec(*cmd,cwd=ROOT,stdout=log,stderr=asyncio.subprocess.STDOUT)
            jobs[name]=proc
            try:
                deadline=time.monotonic()+7200
                while proc.returncode is None:
                    if time.monotonic()>deadline:raise TimeoutError(f'{name}: two-hour deadline')
                    try:await asyncio.wait_for(proc.wait(),30)
                    except asyncio.TimeoutError:report(OUT)
                rec['exit_code']=proc.returncode
                data=json.loads((OUT/name/'summary.json').read_text())
                if proc.returncode or data['status']!='complete':raise RuntimeError(f'{name}: {data.get("failure")}')
                report(OUT)
                audit=json.loads((OUT/'report.json').read_text())
                if next(b for b in audit['batches'] if b['batch']==name)['feedback_gaps']:
                    raise RuntimeError(f'{name}: missing progress events')
                rec['status']='complete'
            except BaseException:
                rec['status']='failed'
                raise
            finally:
                if proc.returncode is None:
                    proc.terminate()
                    try:await asyncio.wait_for(proc.wait(),10)
                    except asyncio.TimeoutError:proc.kill();await proc.wait()
                jobs.pop(name,None)
                rec['finished']=datetime.now(timezone.utc).isoformat()
                write(OUT/'campaign-status.json',status)
    async def track(kind,fleet,length,families):
        name=kind+'-screen'
        await execute(name,command(kind,fleet,length,name,['--preps','none','copy','--reps','1']))
        _,rows=comparisons(OUT/name/'summary.json')
        candidates={r['clients'] for r in rows if r['prep']=='copy' and r['reduction_pct']>=5
                    and r['target_requests']>0 and r['errors']==0}
        selected=set(candidates)
        for n in candidates:
            i=LEVELS.index(n)
            if i>0:selected.add(LEVELS[i-1])
            if i+1<len(LEVELS):selected.add(LEVELS[i+1])
        if not selected:selected={64,128}
        selected.add(4)
        cases=[f'f{families}-c{n:03d}' for n in sorted(selected)]
        write(OUT/f'{kind}-selection.json',dict(candidates=sorted(candidates),confirmation_levels=sorted(selected),
            rule='>=5% screen copy gain, plus adjacent boundary levels and low-load control=4; '
                 'if none qualify, check 64/128. Confirmation uses fresh run order and two repeats.'))
        name=kind+'-confirm'
        await execute(name,command(kind,fleet,length,name,['--cases',*cases,
            '--preps','none','copy','recompute','--reps','2','--seed','20261012']))
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            for rec in [r for fs in fleets.values() for r in fs]:
                deadline=time.monotonic()+900
                while True:
                    try:
                        async with session.get(rec['url']+'/health') as response:
                            if response.status==200:break
                    except (aiohttp.ClientError,asyncio.TimeoutError):pass
                    if time.monotonic()>deadline:raise TimeoutError('health: '+rec['name'])
                    await asyncio.sleep(5)
        status['status']='running';write(OUT/'campaign-status.json',status)
        tasks=[asyncio.create_task(track(*spec)) for spec in specs]
        try:await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
        status['status']='complete'
    except BaseException as exc:
        status['status']='failed';status['failure']=repr(exc);raise
    finally:
        cleanup=[]
        for rec in [r for fs in fleets.values() for r in fs]:
            remote=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',rec['node']]
            with (OUT/(rec['name']+'.engine.log')).open('w') as log:
                proc=await asyncio.create_subprocess_exec(*remote,shlex.join(['docker','logs',rec['container']]),
                    stdout=log,stderr=asyncio.subprocess.STDOUT)
                try:await asyncio.wait_for(proc.wait(),30)
                except asyncio.TimeoutError:proc.kill();await proc.wait()
            proc=await asyncio.create_subprocess_exec(*remote,shlex.join(['docker','rm','-f',rec['container']]),
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            try:
                stdout,stderr=await asyncio.wait_for(proc.communicate(),30)
                cleanup.append(dict(name=rec['name'],exit_code=proc.returncode,stdout=stdout.decode(),stderr=stderr.decode()))
            except asyncio.TimeoutError:proc.kill();await proc.wait();cleanup.append(dict(name=rec['name'],error='timeout'))
        write(OUT/'cleanup.json',cleanup)
        report(OUT)
        status['finished']=datetime.now(timezone.utc).isoformat()
        write(OUT/'campaign-status.json',status)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUT)
    OUT=parser.parse_args().out.resolve()
    asyncio.run(main())
