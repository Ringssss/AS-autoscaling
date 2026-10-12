"""Run the preregistered campaign; preserve outcomes and clean only owned workers."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import time

import aiohttp

from experiments.fluidkv.p08_campaign import write
from experiments.fluidkv.p08_report import report


ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/state-preparation-20261011'


async def main():
    status=dict(started=datetime.now(timezone.utc).isoformat(),status='waiting_for_pilot',stages=[])
    write(OUT/'campaign-status.json',status)
    fleets={name:json.loads((OUT/f'fleet-{name}.json').read_text()) for name in ('local','node30')}
    cohorts=json.loads((OUT/'holdout/cohorts.json').read_text())['cohorts']
    split={name:[c['name'] for c in cohorts if c['name'].endswith('-'+str(i))]
        for i,name in enumerate(('local','node30'))}
    selected='holdout-n4-0'
    if selected not in split['local']:
        raise RuntimeError('preregistered control cohort absent')
    cross=[fleets['local'][0],fleets['node30'][0]]
    write(OUT/'fleet-cross.json',cross)
    def command(name,fleet,cases,extra=(),calibration=None):
        cmd=[sys.executable,'-u','-m','experiments.fluidkv.p08_campaign',
            '--fleet',str(OUT/f'fleet-{fleet}.json'),
            '--cohorts',str(OUT/'holdout'),'--cases',*cases,
            '--out',str(OUT/name),*extra]
        if calibration:
            cmd += ['--calibration',str(calibration)]
        return dict(name=name,command=cmd)
    tasks=[command('holdout-local','local',split['local']),
           command('holdout-node30','node30',split['node30'])]
    cal=OUT/'holdout-local/calibration.json'
    controls=[
        command('balanced-control','local',[selected],['--placement','balanced','--preps','none'],cal),
        command('zero-gap','local',[selected],['--gap','0','--reps','2'],cal),
        command('long-output','local',[selected],['--output-cap','512','--reps','2'],cal),
        command('dualmap-admission4','local',[selected],['--policies','dualmap','--dualmap-pending','4'],cal),
        command('cross-node','cross',[selected]),
    ]
    write(OUT/'schedule.json',dict(main=tasks,controls=controls,
        policy='All eight outcome-independent cohorts, three repetitions; '
        'fixed structural n4-0 control, no performance-based selection. '
        'Separate hosts execute main batches in parallel; shared-worker stages serialize.',
        planned_trials=144+6+12+12+9+18,pilot_trials=6))
    running={}
    async def execute(rec):
        stage=dict(name=rec['name'],started=datetime.now(timezone.utc).isoformat(),status='running')
        status['stages'].append(stage)
        write(OUT/'campaign-status.json',status)
        with (OUT/(rec['name']+'.log')).open('w') as log:
            proc=await asyncio.create_subprocess_exec(*rec['command'],cwd=ROOT,
                stdout=log,stderr=asyncio.subprocess.STDOUT)
            running[rec['name']]=proc
            try:
                deadline=time.monotonic()+4*3600
                while proc.returncode is None:
                    if time.monotonic()>deadline:
                        raise TimeoutError('four-hour batch deadline')
                    try:
                        await asyncio.wait_for(proc.wait(),timeout=30)
                    except asyncio.TimeoutError:
                        report(OUT)
                        write(OUT/'campaign-status.json',status)
                stage['exit_code']=proc.returncode
                result=json.loads((OUT/rec['name']/'summary.json').read_text())
                if proc.returncode or result['status']!='complete':
                    raise RuntimeError(f'{rec["name"]} failed: {result.get("failure")}')
                report(OUT)
                audited=json.loads((OUT/'report.json').read_text())
                if next(r for r in audited if r['batch']==rec['name'])['feedback_gaps']:
                    raise RuntimeError(f'{rec["name"]} has missing progress events')
                stage['status']='complete'
            except BaseException:
                stage['status']='failed'
                raise
            finally:
                if proc.returncode is None:
                    proc.terminate()
                    try:await asyncio.wait_for(proc.wait(),timeout=10)
                    except asyncio.TimeoutError:
                        proc.kill(); await proc.wait()
                running.pop(rec['name'],None)
                stage['finished']=datetime.now(timezone.utc).isoformat()
                write(OUT/'campaign-status.json',status)
    try:
        deadline=time.monotonic()+1800
        while True:
            path=OUT/'pilot/summary.json'
            pilot=json.loads(path.read_text()) if path.exists() else {}
            if pilot.get('status')=='failed':
                raise RuntimeError(f'pilot failed: {pilot.get("failure")}')
            if pilot.get('status')=='complete':
                if len(pilot['trials'])!=6:
                    raise RuntimeError('pilot matrix incomplete')
                break
            if time.monotonic()>deadline:
                raise TimeoutError('pilot gate deadline')
            await asyncio.sleep(10)
        status['status']='checking_workers'
        write(OUT/'campaign-status.json',status)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            for rec in [r for rs in fleets.values() for r in rs]:
                deadline=time.monotonic()+900
                while True:
                    try:
                        async with session.get(rec['url']+'/health') as response:
                            if response.status==200:break
                    except (aiohttp.ClientError,asyncio.TimeoutError):pass
                    if time.monotonic()>deadline:
                        raise TimeoutError(f'worker health timeout: {rec["name"]}')
                    await asyncio.sleep(10)
        status['status']='running'
        write(OUT/'campaign-status.json',status)
        mains=[asyncio.create_task(execute(rec)) for rec in tasks]
        try:
            await asyncio.gather(*mains)
        finally:
            for task in mains:
                if not task.done():task.cancel()
            await asyncio.gather(*mains,return_exceptions=True)
        for rec in controls:
            await execute(rec)
        status['status']='complete'
    except BaseException as exc:
        status['status']='failed'
        status['failure']=repr(exc)
        raise
    finally:
        # These four exact container IDs were created by this campaign.
        # Existing containers, checkpoint projects and source datasets are untouched.
        cleanup=[]
        for rec in [r for rs in fleets.values() for r in rs]:
            remote=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5',rec['node']]
            with (OUT/(rec['name']+'.engine.log')).open('w') as log:
                p=await asyncio.create_subprocess_exec(*remote,
                    shlex.join(['docker','logs',rec['container']]),stdout=log,stderr=asyncio.subprocess.STDOUT)
                try:await asyncio.wait_for(p.wait(),30)
                except asyncio.TimeoutError:p.kill();await p.wait()
            p=await asyncio.create_subprocess_exec(*remote,
                shlex.join(['docker','rm','-f',rec['container']]),
                stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
            try:
                stdout,stderr=await asyncio.wait_for(p.communicate(),30)
                cleanup.append(dict(name=rec['name'],exit_code=p.returncode,
                    stdout=stdout.decode(),stderr=stderr.decode()))
            except asyncio.TimeoutError:
                p.kill();await p.wait()
                cleanup.append(dict(name=rec['name'],error='cleanup timeout'))
        write(OUT/'cleanup.json',cleanup)
        report(OUT)
        status['finished']=datetime.now(timezone.utc).isoformat()
        write(OUT/'campaign-status.json',status)


if __name__=='__main__':
    asyncio.run(main())
