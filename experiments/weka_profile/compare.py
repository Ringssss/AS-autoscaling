"""Held-out baseline/current/multiple-placement comparison, bounded to 35 min."""
import asyncio
import datetime
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import aiohttp
import numpy as np
from experiments.weka_baselines.replay import Fleet,write

ROOT=Path(__file__).parent
OUT=ROOT/'results/compare'
URLS=[f'http://12.11.3.{i}:31780' for i in [27,28,29]]


async def experiment(status):
    gate=json.loads((ROOT/'results/nonempty-validation/summary.json').read_text())
    assert gate['all_passed']
    spec=importlib.util.spec_from_file_location('nonempty_trial',ROOT/'environment/comparison_trial.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    import experiments.weka_baselines.replay as replay
    replay.URLS=URLS
    rows=[json.loads(s) for s in (OUT/'requests.jsonl').read_text().splitlines()]
    tokens=np.load(ROOT.parent/'weka_baselines/data/tokens.npy',mmap_mode='r')
    # Fixed predeclared order. One full, held-out trajectory each; exploratory,
    # no pooled confidence interval from the 3 dependent rounds.
    order=['dualmap','dualmap+multi','dualmap+move','cacheroute+move','cacheroute+multi','cacheroute']
    write(OUT/'matrix.json',order)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600,sock_read=180),connector=aiohttp.TCPConnector(limit=128)) as session:
        while True:
            ready=[]
            for u in URLS:
                try:
                    async with session.get(u+'/health',timeout=aiohttp.ClientTimeout(total=5)) as r:ready.append(r.status==200)
                except Exception:ready.append(False)
            if all(ready):break
            status.update(state='waiting',healthy=ready);write(OUT/'status.json',status);await asyncio.sleep(5)
        fleet=Fleet(OUT)
        # Existing node28->29 validation occupies 24500 only; baseline Fleet
        # groups remain at the checked 24100 range after server restarts.
        status['state']='warming';write(OUT/'status.json',status);await fleet.warm(session)
        summaries=[]
        for i,policy in enumerate(order):
            status.update(state='running',completed=i,policy=policy);write(OUT/'status.json',status)
            print('START',i,policy,flush=True)
            result=await mod.trial(fleet,session,rows,tokens,OUT/'runs'/f'{i:02d}-{policy}',policy,0,status)
            summaries.append(result);write(OUT/'completed.json',summaries)
            status['completed']=i+1;write(OUT/'status.json',status)
            print('DONE',policy,'seconds',result['duration_seconds'],'migrations',len(result['migrations']),flush=True)
        status['state']='complete';write(OUT/'status.json',status)


async def main():
    (OUT/'runs').mkdir(exist_ok=False)
    status=dict(pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),state='starting',completed=0,total=6)
    write(OUT/'status.json',status)
    try:await asyncio.wait_for(experiment(status),35*60)
    except asyncio.TimeoutError:status.update(state='budget_expired')
    except BaseException as e:status.update(state='failed',error=repr(e),traceback=traceback.format_exc());print(traceback.format_exc(),flush=True)
    finally:
        status['ended']=datetime.datetime.now().astimezone().isoformat();write(OUT/'status.json',status)
        for node in ['node27','node28','node29']:
            with (OUT/(node+'-engine.log')).open('wb') as f:
                subprocess.run(['ssh',node,'docker','logs','zhujianian-agentx-nonempty'],stdout=f,stderr=subprocess.STDOUT)
            subprocess.run(['ssh',node,'docker','stop','--time','10','zhujianian-agentx-nonempty'],timeout=45)


if __name__=='__main__':asyncio.run(main())
