"""Durable serial matrix driver; all policies use the same dedicated fleet."""
import asyncio
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import traceback
import time

import aiohttp
import numpy as np
from experiments.weka_baselines.common import POLICIES,SCENARIOS
from experiments.weka_baselines.replay import Fleet,ROOT,URLS,run_trial,write


def record_summary(root):
    summaries=[]
    for path in sorted(root.glob('runs/*/summary.json')):
        x=json.loads(path.read_text());d=path.parent
        events=[json.loads(s) for s in (d/'events.jsonl').read_text().splitlines()]
        x['migration_bytes']=sum(e.get('bytes',0) for e in events if e['kind']=='preparation')
        x['preparation_seconds']=sum(e.get('duration',0) for e in events if e['kind']=='preparation')
        x['cache_reset_count']=sum(e['kind']=='initial_cache_reset' for e in events)
        assert x['cache_reset_count']==1
        max_running=[0]*3;min_free=[1048576]*3;max_hbm={}
        for line in (d/'telemetry.jsonl').read_text().splitlines():
            r=json.loads(line)
            if 'load' in r:
                i=r['engine'];max_running[i]=max([max_running[i]]+[v['num_reqs']-v['num_waiting_reqs'] for v in r['load']])
                free=r.get('pool',{}).get('available_token_slots')
                if free is not None:min_free[i]=min(min_free[i],free)
            if 'hbm' in r:
                node=r['node'];devices={'node27':[6,7],'node28':[4,5],'node29':[0,1]}[node]
                usage=sum(int(r['hbm']['card'+str(i)]['vram Total Used Memory (MiB)']) for i in devices)
                max_hbm[node]=max(max_hbm.get(node,0),usage)
        x.update(observed_max_running_per_engine=max_running,max_used_token_slots=[1048576-v for v in min_free],
                 max_hbm_MiB_two_cards=max_hbm,artifact=str(d))
        summaries.append(x)
    write(root/'completed.json',summaries)
    lines=['# AgentX 三节点持续缓存实验进度','',
        'BS 为全局最大在途请求数；输入/输出保留所选轨迹原始长度。每个独立方案仅初始化一次缓存。',
        'goodput 主阈值为 TTFT≤10s、平均 TPOT≤0.2s；仅验证长度，语义/逐 token 一致性另行审计。', '',
        '| 场景 | BS | 策略 | 完成/失败 | 全流量 p95/p99 TTFT(s) | SLO goodput(req/s) | 迁移 MiB | 实际最大 running/节点 |',
        '|---|---:|---|---|---|---:|---:|---|']
    for x in summaries:
        s=x['spec'];g=next(g for g in x['goodput'] if g['ttft_slo']==10 and g['tpot_slo']==.2)
        tails=x['all_traffic_ttft']
        lines.append(f"| {s['scenario']} | {s['bs']} | {s['policy']} | {x['requests']}/{x['errors']} | {tails['95']} / {tails['99']} | {g['requests_per_second']:.5f} | {x['migration_bytes']/1048576:.2f} | {x['observed_max_running_per_engine']} |")
    (root/'PROGRESS.md').write_text('\n'.join(lines)+'\n')


async def main():
    root=ROOT/'results'/'campaign-20260928-r3'
    root.mkdir(parents=True,exist_ok=False);(root/'runs').mkdir();(root/'smoke').mkdir()
    status=dict(pid=os.getpid(),started=datetime.datetime.now().astimezone().isoformat(),state='starting',completed=0,total=160)
    write(root/'status.json',status)
    source_files=list(ROOT.glob('*.py'))+[ROOT.parent/'fluidkv/dualmap_adapter.py',ROOT.parent/'multinode/cacheroute.py']
    write(root/'provenance.json',dict(source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        data_manifest_sha256=hashlib.sha256((ROOT/'data/manifest.json').read_bytes()).hexdigest(),
        dualmap_commit=subprocess.check_output(['git','-C','/home/zhujianian/DualMap','rev-parse','HEAD'],text=True).strip(),
        control_runtime=json.loads((ROOT/'environment/runtime/control-isolation.json').read_text()),
        launch=json.loads((ROOT/'results/launch.json').read_text())))
    rows=[json.loads(line) for line in (ROOT/'data/requests.jsonl').read_text().splitlines()]
    tokens=np.load(ROOT/'data/tokens.npy',mmap_mode='r')
    # High-concurrency cells first provide early coverage; BS=1 uses full output
    # lengths as well and can run for days. No claim of completed matrix early.
    matrix=[]
    for bs in [128,64,16,1]:
        for scenario in SCENARIOS:
            policies=POLICIES.copy();random.Random(517+bs+SCENARIOS.index(scenario)).shuffle(policies)
            for policy in policies:matrix.append(dict(bs=bs,scenario=scenario,policy=policy,streams=128,period=60))
    write(root/'matrix.json',matrix)
    timeout=aiohttp.ClientTimeout(total=14400,connect=30,sock_read=900)
    try:
        async with aiohttp.ClientSession(timeout=timeout,connector=aiohttp.TCPConnector(limit=512)) as session:
            deadline=time.monotonic()+900
            while True:
                healthy=[]
                for url in URLS:
                    try:
                        async with session.get(url+'/health',timeout=aiohttp.ClientTimeout(total=5)) as r:
                            healthy.append(r.status==200)
                    except Exception:healthy.append(False)
                if all(healthy):break
                if time.monotonic()>deadline:raise RuntimeError('engine readiness timeout')
                status.update(state='waiting_for_engines',healthy=healthy);write(root/'status.json',status)
                await asyncio.sleep(5)
            fleet=Fleet(root)
            status['state']='warming';write(root/'status.json',status)
            await fleet.warm(session)
            # Transport/router smoke ONLY: caps explicit and never pooled with
            # the full-length matrix. Include membership expansion and both
            # preparation wrappers, then require zero inference failures.
            smoke=[]
            for r in rows:
                if r['stream_index']<4:
                    n=min(4096,r['input_tokens']);a=tokens[r['token_offset']:r['token_offset']+n]
                    smoke.append(dict(r,input_tokens=n,output_tokens=8,hash_ids=r['hash_ids'][:n//16],token_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
            for i,policy in enumerate(['p2c+move','dualmap+move','cacheroute+precompute','sticky']):
                spec=dict(bs=16,scenario='scaleout',policy=policy,streams=4,period=2,smoke_only=True,input_cap=4096,output_cap=8)
                status.update(state='smoke',current=spec);write(root/'status.json',status)
                s=await run_trial(fleet,session,smoke,tokens,root/'smoke'/str(i),spec)
                assert s['errors']==0 and s['requests']==12,s
                print('SMOKE PASSED',policy,flush=True)
            for index,spec in enumerate(matrix):
                path=root/'runs'/f"{index:03d}-{spec['scenario']}-bs{spec['bs']}-{spec['policy']}"
                status.update(state='running',current=spec,current_directory=str(path),completed=index)
                write(root/'status.json',status);print('START',index,spec,flush=True)
                result=await run_trial(fleet,session,rows,tokens,path,spec)
                record_summary(root)
                status['completed']=index+1;write(root/'status.json',status)
                print('FINISH',index,'errors',result['errors'],flush=True)
                if result['errors']:
                    # Retain all offered failures, but do not blindly continue
                    # hundreds of cells on an unhealthy or invalid backend.
                    raise RuntimeError('full-length cell had inference failures; audit before further runs')
            status['state']='complete';write(root/'status.json',status)
    except BaseException as e:
        status.update(state='failed',error=repr(e),traceback=traceback.format_exc())
        write(root/'status.json',status);print(traceback.format_exc(),flush=True)
        raise
    finally:
        # Only the containers created by this campaign are stopped.
        cleanup=[]
        for node in ['node27','node28','node29']:
            with (root/(node+'-engine.log')).open('wb') as f:
                subprocess.run(['ssh',node,'docker','logs','zhujianian-agentx-baselines'],stdout=f,stderr=subprocess.STDOUT)
            r=subprocess.run(['ssh',node,'docker','stop','zhujianian-agentx-baselines'],capture_output=True,text=True)
            cleanup.append(dict(node=node,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))
        write(root/'cleanup.json',cleanup)


if __name__=='__main__':asyncio.run(main())
