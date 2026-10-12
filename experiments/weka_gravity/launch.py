"""Dedicated, free-device-only engines for the Weka applicability assessment."""
import json
import shlex
import subprocess
from pathlib import Path
from experiments.multinode.launch_pilot import ENGINES, command_for

OUT = Path(__file__).parent / 'results'
records = []
for engine in ENGINES[:3]:
    node = engine['node']
    memory = json.loads(subprocess.check_output([
        'ssh', '-o', 'BatchMode=yes', node,
        '/usr/local/hyhal/bin/hy-smi --showmeminfo vram --json'], text=True))
    assert all(int(memory['card'+d]['vram Total Used Memory (MiB)']) < 1024
               for d in engine['devices'].split(',')), (node, memory)
    cmd = command_for(engine)
    cmd = [s.replace('zhujianian-as-multinode-pilot', 'zhujianian-weka-gravity')
           .replace('/home/zhujianian/Qwen3.6-35B-A3B', '/data/node28/shared/zhujianian/Qwen3.6-35B-A3B')
           .replace('--port 31600', '--port 31720') for s in cmd]
    result = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', node, shlex.join(cmd)], text=True)
    records.append(dict(engine, command=cmd, container_id=result.strip()))
    (OUT/'launch.json').write_text(json.dumps(records, indent=2)+'\n')
    print(node, result.strip(), flush=True)
