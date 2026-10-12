"""Sequential, predeclared real-engine boundary and held-out persistence tests."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'experiments/multinode/results/advantage-boundary'
POLICIES = ['dualmap','dualmap+fluidkv','dualmap+precompute','cacheroute','cacheroute+fluidkv','cacheroute+precompute']

def main():
    source_files=['experiments/multinode/benchmark_highbs.py','experiments/multinode/run_advantage_boundary.py',
                  'experiments/multinode/cacheroute.py','experiments/fluidkv/dualmap_adapter.py',
                  'agentshift/controller/hybrid_migration.py']
    (OUT/'source-manifest.json').write_text(json.dumps({p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_files},indent=2))
    jobs=[]
    for window in (0,.5,5):
        jobs.append((f'screen-window-{window}', ['--lengths','8192','65536','--batches','4','16','--repetitions','1','--tool-seconds',str(window),'--seed','20260925']))
    jobs.append(('heldout-persistence', ['--lengths','65536','--batches','16','--repetitions','5','--waves','3','--tool-seconds','.5','--seed','20261001']))
    commands=[]
    for index,(name,extra) in enumerate(jobs):
        cmd=[sys.executable,'experiments/multinode/benchmark_highbs.py','--regimes','single-hot','--policies',*POLICIES,
             '--pending-prefills','32','--observe-checkpoints','--reuse-compiled-kernels','--settle-before-trial',
             '--calibration-file','experiments/multinode/results/highbs/three-node-primary/calibration.json',
             '--base-port',str(25100+index*40),'--output-tokens','64','--output',str(OUT/name),*extra]
        commands.append({'name':name,'argv':cmd})
        if index == 0:
            cmd.remove('--reuse-compiled-kernels')
    (OUT/'campaign.json').write_text(json.dumps({'jobs':commands,'claim_rule':'Held-out 64K/16/window=.5 chosen before screening. Assess both routers separately. Large stable TTFT gain requires >=1.2 paired median and run-bootstrap 95% CI lower bound >1, no errors, and retained-wave evidence. Five runs are limited evidence, never universal proof. Precompute has same prefix knowledge/window; baseline source prefill is outside return timing for all policies. Shared-prefix bursts cannot establish private-history or mixed-load superiority.'},indent=2)+'\n')
    for job in commands:
        print('START',job['name'],flush=True)
        with (OUT/(job['name']+'.log')).open('w') as log:
            result=subprocess.run(job['argv'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f"failed {job['name']}: {result.returncode}; retained log")
        print('DONE',job['name'],flush=True)
    (OUT/'COMPLETE').write_text('All predeclared jobs completed.\n')

if __name__=='__main__':main()
