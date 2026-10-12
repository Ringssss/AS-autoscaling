"""Run preregistered followups only after the main campaign has drained."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/multinode/results/advantage-boundary'

def main():
    deadline=time.monotonic()+7200
    while not (OUT/'COMPLETE').exists():
        if time.monotonic()>deadline:raise SystemExit('Main campaign did not finish; no followup started')
        time.sleep(3)
    for spec in ('source-withdrawal.json','selected-validation.json','long-window-control.json'):
        job=json.loads((OUT/spec).read_text())
        print('START',job['name'],flush=True)
        (OUT/(job['name']+'-source.sha256')).write_text(hashlib.sha256((ROOT/'experiments/multinode/benchmark_highbs.py').read_bytes()).hexdigest()+'\n')
        with (OUT/(job['name']+'.log')).open('w') as log:
            result=subprocess.run(job['argv'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(f"failed {job['name']}: {result.returncode}")
        print('DONE',job['name'],flush=True)
    (OUT/'FOLLOWUPS_COMPLETE').write_text('Selected validation and long-window controls completed.\n')

if __name__=='__main__':main()
