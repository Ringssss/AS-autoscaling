"""Finish diagnostic capture, then start the gated same-backend comparison."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
from experiments.multinode.launch_pilot import ENGINES,command_for

ROOT=Path(__file__).parent.resolve()
OUT=ROOT/'results'
deadline=time.monotonic()+900
while True:
    log=(OUT/'capture.log').read_text()
    if 'CAPTURE_COMPLETE' in log:break
    if 'Traceback (most recent call last)' in log:raise RuntimeError('main profiler capture failed')
    if time.monotonic()>deadline:raise TimeoutError('main profiler capture timeout')
    time.sleep(5)
with (OUT/'light-capture.log').open('w') as f:
    subprocess.run([sys.executable,str(ROOT/'light_capture.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
with (OUT/'profile-engine.log').open('wb') as f:
    subprocess.run(['docker','logs','zhujianian-agentx-profile'],stdout=f,stderr=subprocess.STDOUT)
subprocess.run(['docker','stop','--time','10','zhujianian-agentx-profile'],check=True)
assert json.loads((OUT/'nonempty-validation/summary.json').read_text())['all_passed']
e=ENGINES[0]
mem=json.loads(subprocess.check_output(['ssh',e['node'],'/usr/local/hyhal/bin/hy-smi --showmeminfo vram --json'],text=True))
assert all(int(mem['card'+d]['vram Total Used Memory (MiB)'])<1024 for d in e['devices'].split(','))
cmd=command_for(e,max_running=128,mamba_slots=384)
cmd=[s.replace('zhujianian-as-multinode-pilot','zhujianian-agentx-nonempty').replace('/home/zhujianian/Qwen3.6-35B-A3B','/data/node28/shared/zhujianian/Qwen3.6-35B-A3B').replace('/home/zhujianian/hygon-runtime-overlay',str(ROOT/'environment/nonempty-runtime')).replace('--port 31600','--port 31780').replace('--max-total-tokens 262144','--max-total-tokens 1048576') for s in cmd]
i=cmd.index('--entrypoint');cmd[i:i]=['-e','AGENTSHIFT_NONEMPTY_IMPORT=1']
cid=subprocess.check_output(['ssh',e['node'],shlex.join(cmd)],text=True).strip()
(OUT/'nonempty-node27-launch.json').write_text(json.dumps(dict(engine=e,command=cmd,container_id=cid,memory_before=mem),indent=2)+'\n')
with (OUT/'compare.log').open('w') as f:
    subprocess.run([sys.executable,str(ROOT/'compare.py')],stdout=f,stderr=subprocess.STDOUT,check=True)
print('PROFILE_AND_COMPARISON_FINISHED',flush=True)
