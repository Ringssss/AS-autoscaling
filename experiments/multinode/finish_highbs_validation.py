"""Post-campaign null controls, then independent byte-integrity validation."""
import concurrent.futures
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'experiments/multinode/results/highbs'


def wait_for(url, endpoint, predicate, timeout):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        try:
            with urllib.request.urlopen(url+endpoint,timeout=10) as response:
                data=json.load(response) if endpoint=='/get_load' else response.status
            if predicate(data):return
        except Exception:pass
        time.sleep(5)
    raise RuntimeError('readiness/drain timeout '+url)


def main():
    last=RESULTS/'three-node-cold-65536/summary.json'
    deadline=time.monotonic()+7200
    while not last.exists():
        if time.monotonic()>deadline:raise RuntimeError('cold campaign did not complete')
        time.sleep(5)
    for name,count in [('three-node-primary',144),('three-node-cold-32768',8),('three-node-cold-65536',8)]:
        data=json.loads((RESULTS/name/'summary.json').read_text())
        if len(data['runs'])!=count:raise ValueError('incomplete campaign '+name)
    print('Performance campaigns complete; fresh-process null controls begin',flush=True)
    control_root=RESULTS/'highest-point-null-control'
    control_root.mkdir(exist_ok=False)
    rows=[]
    rng=random.Random(20260925)
    for repeat in range(3):
        for policy in rng.sample(['dualmap','dualmap+fluidkv','cacheroute','cacheroute+fluidkv'],4):
            name=f'{repeat}-{policy}'
            command=[sys.executable,'experiments/multinode/benchmark_highbs.py','--lengths','65536','--batches','128',
                     '--regimes','all-hot','--policies',policy,'--repetitions','1','--output-tokens','64',
                     '--pending-prefills','32','--reuse-compiled-kernels',
                     '--calibration-file',str(RESULTS/'three-node-primary/calibration.json'),
                     '--seed',str(20260925+repeat),'--output',str(control_root/name)]
            with (control_root/(name+'.log')).open('w') as log:
                subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT),PYTHONHASHSEED='0'),
                               stdout=log,stderr=subprocess.STDOUT,check=True)
            summary=json.loads((control_root/name/'summary.json').read_text())['runs'][0]
            if summary['total_transfer_bytes']!=0:raise ValueError('null control performed a transfer')
            rows.append({'outer_repeat':repeat,'policy':policy,'path':str(control_root/name),'summary':summary})
            (control_root/'summary.json').write_text(json.dumps({'scope':'separate router process per no-op label; same persistent warmed engines and fixed primary calibration; GC before timing; not pooled with primary results','runs':rows},indent=2)+'\n')
            print('Null control complete',name,flush=True)
    def archive(node):
        wait_for(f'http://12.11.3.{node[-2:]}:31600','/get_load',lambda data:all(x['num_reqs']==0 for x in data),120)
        with (RESULTS/(node+'-performance-server.log')).open('w') as log:
            subprocess.run(['ssh',node,'docker logs zhujianian-as-multinode-pilot'],stdout=log,stderr=subprocess.STDOUT,check=True)
    nodes=['node27','node28','node29']
    with concurrent.futures.ThreadPoolExecutor() as executor:list(executor.map(archive,nodes))
    subprocess.run([sys.executable,'experiments/hygon_runtime/build_overlay.py','--source','/home/zhujianian/hygon-runtime-source/srt','--output','/home/zhujianian/hygon-runtime-overlay'],cwd=ROOT,check=True)
    def restart(node):subprocess.run(['ssh',node,'docker restart -t 15 zhujianian-as-multinode-pilot'],check=True)
    with concurrent.futures.ThreadPoolExecutor() as executor:list(executor.map(restart,nodes))
    for node in nodes:
        wait_for(f'http://12.11.3.{node[-2:]}:31600','/health',lambda status:status==200,900)
        print('Digest runtime ready',node,flush=True)
    for node,port in [(29,31820),(28,31840)]:
        name=f'state-byte-integrity-node{node}'
        command=[sys.executable,'experiments/multinode/validate_state_bytes.py','--destination',f'http://12.11.3.{node}:31600','--base-port',str(port),'--output',str(RESULTS/name)]
        with (RESULTS/(name+'.log')).open('w') as log:
            subprocess.run(command,cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Integrity complete',node,flush=True)
    print('Post-campaign validation complete',flush=True)


if __name__=='__main__':main()
