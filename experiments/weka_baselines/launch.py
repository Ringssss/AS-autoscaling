"""Dedicated three-node servers; global BS is controlled by the replay client."""
import json
import shlex
import subprocess
from pathlib import Path
from experiments.multinode.launch_pilot import ENGINES,command_for

out=Path(__file__).parent/'results'
out.mkdir(parents=True,exist_ok=True)
records=[]
for e in ENGINES[:3]:
    mem=json.loads(subprocess.check_output(['ssh',e['node'],'/usr/local/hyhal/bin/hy-smi --showmeminfo vram --json'],text=True))
    assert all(int(mem['card'+d]['vram Total Used Memory (MiB)'])<1024 for d in e['devices'].split(','))
    cmd=command_for(e,max_running=128,mamba_slots=384)
    cmd=[s.replace('zhujianian-as-multinode-pilot','zhujianian-agentx-baselines')
          .replace('/home/zhujianian/hygon-runtime-overlay',str(Path(__file__).resolve().parent/'environment/runtime'))
          .replace('/home/zhujianian/Qwen3.6-35B-A3B','/data/node28/shared/zhujianian/Qwen3.6-35B-A3B')
          .replace('--port 31600','--port 31740').replace('--max-total-tokens 262144','--max-total-tokens 1048576')
          for s in cmd]
    cid=subprocess.check_output(['ssh',e['node'],shlex.join(cmd)],text=True).strip()
    records.append(dict(e,container_id=cid,command=cmd,free_memory_before=mem))
    (out/'launch.json').write_text(json.dumps(records,indent=2)+'\n')
    print(e['node'],cid,flush=True)
