"""Serialize predeclared native-output and continuous-window followups."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'experiments/multinode/results'
p=argparse.ArgumentParser();p.add_argument('--screen-pid',type=int,required=True)
p.add_argument('--screen-dir',type=Path,default=R/'kimi-trace-screen');a=p.parse_args()
while not (a.screen_dir/'COMPLETE').exists():
    if list(a.screen_dir.glob('*-failure.json')):
        raise SystemExit('screening failure; not starting further GPU work')
    try: os.kill(a.screen_pid,0)
    except ProcessLookupError: raise SystemExit('screening process stopped without COMPLETE')
    time.sleep(5)
# Recheck every affected comparison, selected only by observer errors, never by
# performance. The original screening data remain intact and flagged.
from experiments.multinode.summarize_kimi_trace_replay import PAIRS
condition_fields=['window','length','withdraw','tool_seconds','rate_scale','output_cap','repeat']
affected={}
for folder in [R/'kimi-trace-screen',a.screen_dir]:
    for line in (folder/'summary.jsonl').read_text().splitlines():
        s=json.loads(line)
        if not s['checkpoint_errors']: continue
        key=tuple(s[k] for k in condition_fields)
        policies=affected.setdefault(key,set())
        for baseline,treatment in PAIRS:
            if s['policy'] in (baseline,treatment): policies.update((baseline,treatment))
rechecks=[dict(zip(condition_fields,key))|{'stage':'observer-recheck','policy':policy}
    for key,policies in sorted(affected.items(),key=lambda item:tuple(map(str,item[0])))
    for policy in sorted(policies)]
(R/'kimi-trace-observer-jobs.json').write_text(json.dumps(rechecks,indent=2)+'\n')
counts={w['label']:len(w['requests']) for w in json.loads((R/'kimi-trace-screen/windows.json').read_text())}
progress=json.loads((R/'kimi-trace-TASK_PROGRESS.json').read_text())
progress['expected_formal_trials']=192+len(rechecks)
progress['expected_formal_requests']=8500+sum(counts[s['window']] for s in rechecks)
progress['pipeline_order'][1]=f'observer-recheck ({len(rechecks)})'
(R/'kimi-trace-TASK_PROGRESS.json').write_text(json.dumps(progress,indent=2)+'\n')
jobs=[
    ['--output',str(R/'kimi-trace-observer-recheck'),'--base-port','26020','--spec-file',str(R/'kimi-trace-observer-jobs.json'),'--calibration-file',str(R/'kimi-trace-screen/calibration.json')],
    ['--output',str(R/'kimi-trace-validation'),'--base-port','25860','--spec-file',str(R/'kimi-trace-validation-jobs.json')],
    ['--output',str(R/'kimi-trace-coverage'),'--base-port','25940','--spec-file',str(R/'kimi-trace-coverage-jobs.json'),'--windows-file',str(R/'kimi-trace-coverage-windows.json')],
    ['--output',str(R/'kimi-trace-native'),'--base-port','25780','--spec-file',str(R/'kimi-trace-native-jobs.json')],
    ['--output',str(R/'kimi-trace-long'),'--base-port','25820','--spec-file',str(R/'kimi-trace-long-jobs.json'),'--windows-file',str(R/'kimi-trace-long-windows.json')],
]
(R/'kimi-trace-followup-commands.json').write_text(json.dumps(jobs,indent=2)+'\n')
for extra in jobs:
    name=Path(extra[1]).name
    command=[sys.executable,'experiments/multinode/run_kimi_trace_replay.py',*extra]
    print('START '+name,flush=True)
    with (R/(name+'.log')).open('w') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise SystemExit(f'{name} failed, exit={result.returncode}')
    print('DONE '+name,flush=True)
print('ALL FOLLOWUPS COMPLETE',flush=True)
