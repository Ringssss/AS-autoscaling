"""Run bounded, sequential followups once the initial paired screen completes."""
import json
import asyncio
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/routing-space-20261010'


async def extend_calibration():
    import aiohttp
    import numpy as np
    spec=importlib.util.spec_from_file_location('pilot',ROOT/'experiments/fluidkv/p07_routing_space.py')
    pilot=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pilot)
    data=json.loads((BASE/'phase1-skew/calibration.json').read_text())
    tokens=np.load(ROOT/'experiments/weka_baselines/data/tokens.npy',mmap_mode='r')
    urls=['http://12.11.3.27:31981','http://12.11.3.27:31982']
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        for length in (32768,65536):
            ids=tokens[:length].astype(int).tolist()
            for url in urls:
                await pilot.skew.reset(session,[url])
                await pilot.generate(session,url,ids,8)
                prompts=[ids+np.random.default_rng(710+i).integers(1000,240000,64).tolist() for i in range(16)]
                await asyncio.gather(*(pilot.generate(session,url,p,8) for p in prompts))
                rs=await asyncio.gather(*(pilot.generate(session,url,p,32) for p in prompts))
                value=float(np.median([r['tpot'] for r in rs]))
                data['decode_grid'].append(dict(length=length,bs=16,url=url,tpot=value))
                print('profile16',length,url,value,flush=True)
        await pilot.skew.reset(session,urls)
    data['extension']='Batch16 coverage added before mixed-background followups; grid router uses these measured points.'
    (BASE/'calibration-extended.json').write_text(json.dumps(data,indent=2)+'\n')


def main():
    # No concurrent GPU arms: host and interconnect interference would confound
    # an otherwise paired comparison. This process never touches unrelated jobs.
    deadline=time.monotonic()+7200
    while time.monotonic()<deadline:
        path=BASE/'phase1-skew/summary.json'
        data=json.loads(path.read_text()) if path.exists() else {}
        if len(data.get('trials',[]))==24:
            if any(t['summary']['errors'] for t in data['trials']):
                raise RuntimeError('initial screen contains request errors')
            break
        time.sleep(5)
    else:
        raise RuntimeError('initial screen did not complete')
    print('EXTEND CALIBRATION',flush=True)
    asyncio.run(extend_calibration())
    followups=[
        ('router-controls',['--policies','context_grid','lmetric_guard','--preps','none','ideal','--reps','2']),
        ('real-preparation',['--policies','context_grid','--preps','none','ideal_common','copy','recompute','--reps','2']),
        ('balanced-control',['--policies','context_grid','--preps','none','ideal','--placement','balanced','--reps','1']),
        ('pressure-screen',['--policies','context_grid','--preps','none','ideal','--reps','1',
            '--background-requests','6','--background-output','512','--lookahead','512','--gap','2']),
        ('long-decode',['--policies','context_grid','--preps','none','ideal','--reps','2',
            '--output-cap','512','--lookahead','512']),
    ]
    commands=[]
    for name,options in followups:
        command=[sys.executable,'-u','experiments/fluidkv/p07_routing_space.py',
            '--cases','native-n6','--calibration',str(BASE/'calibration-extended.json'),
            '--lookahead','512',*options,'--out',str(BASE/name)]
        commands.append(dict(name=name,command=command))
    (BASE/'FOLLOWUPS.json').write_text(json.dumps(commands,indent=2)+'\n')
    for rec in commands:
        print('FOLLOWUP',rec['name'],flush=True)
        with (BASE/(rec['name']+'.log')).open('w') as log:
            subprocess.run(rec['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('COMPLETE',rec['name'],flush=True)
    subprocess.run([sys.executable,'experiments/fluidkv/p07_report.py'],cwd=ROOT,check=True,
                   stdout=(BASE/'report-build.log').open('w'))


if __name__=='__main__':
    main()
