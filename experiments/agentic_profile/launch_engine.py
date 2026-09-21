#!/usr/bin/env python3
"""Launch only a saved dedicated experiment engine, optionally under Nsight."""
import argparse,json,os,subprocess,time,urllib.request,shlex
from pathlib import Path
from experiments.fluidkv.engine_lifecycle import capture_owned_engine
p=argparse.ArgumentParser();p.add_argument('--index',type=int,default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--nsys',action='store_true');a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
old=json.loads(Path('experiments/fluidkv/results/decision-validation-20260914/checkpoint-native-paired-v2/post-experiment-cleanup.json').read_text())['identities'][a.index]
try:
 capture_owned_engine(old['url'])
except RuntimeError:pass
else:raise RuntimeError('owned engine already alive')
env=dict(old['environment']);env['UV_USE_IO_URING']='0'
argv=old['cmdline']
if a.nsys:
 script=a.output/'target.sh'
 script.write_text('exec '+shlex.join(argv)+'\n')
 argv=['nsys','profile','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','--cuda-graph-trace=graph',
       '--capture-range=cudaProfilerApi','--capture-range-end=stop-shutdown','--kill=none','--output',str((a.output/'engine').resolve()),'bash',str(script.resolve())]
with (a.output/'launch.log').open('wb') as f:proc=subprocess.Popen(argv,env=env,cwd=old['cwd'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(a.output/'launch.json').write_text(json.dumps({'pid':proc.pid,'argv':argv,'environment':env,'started_epoch':time.time()},indent=2)+'\n')
for _ in range(240):
 if proc.poll() is not None:raise RuntimeError(f'launcher exited {proc.returncode}')
 try:
  with urllib.request.urlopen(old['url']+'/health',timeout=1) as r:assert r.status==200
  identity=capture_owned_engine(old['url']);(a.output/'identity.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps({'ready':identity['pid'],'url':old['url']}),flush=True);break
 except (OSError,RuntimeError):time.sleep(1)
else:raise RuntimeError('readiness timeout, preserve launcher identity')
