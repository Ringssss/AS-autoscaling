"""Launch engines on the currently free DCU cards. Card choice is asserted
against live hy-smi, so nothing occupied can be taken by accident."""
import argparse, json, shlex, subprocess, sys
from pathlib import Path
sys.path.insert(0,'.')
from experiments.multinode.launch_pilot import command_for, IMAGE, ROOT

# Free-card plan from the 2026-09-29 scan. Each engine is TP2.
PLAN = {
  'node27': ['0,1','2,3','4,5'],
  'node28': ['2,3','6,7'],
  'node29': ['2,3','4,5','6,7'],
}
PORT = {('node27',i):31740+i for i in range(3)}
PORT.update({('node28',i):31750+i for i in range(2)})
PORT.update({('node29',i):31760+i for i in range(3)})
ADDR = {'node27':'12.11.3.27','node28':'12.11.3.28','node29':'12.11.3.29'}

def free(node, devices):
    mem=json.loads(subprocess.check_output(
        ['ssh','-o','BatchMode=yes',node,'/usr/local/hyhal/bin/hy-smi --showmeminfo vram --json'],text=True))
    return all(int(mem[f'card{d}']['vram Total Used Memory (MiB)'])<1024 for d in devices.split(',')), \
           {d:int(mem[f'card{d}']['vram Total Used Memory (MiB)']) for d in devices.split(',')}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--nodes',nargs='+',default=['node29'])
    ap.add_argument('--engines-per-node',type=int,default=1)
    ap.add_argument('--max-running',type=int,default=64)
    ap.add_argument('--mamba-slots',type=int,default=256)
    ap.add_argument('--tag',default='real')
    ap.add_argument('--start-index',type=int,default=0)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args()
    out=Path(__file__).parent/'results'; out.mkdir(parents=True,exist_ok=True)
    records=[]
    for node in a.nodes:
        sel=list(enumerate(PLAN[node]))[a.start_index:a.start_index+a.engines_per_node]
        for idx,dev in sel:
            ok,cur=free(node,dev)
            if not ok:
                print(f"SKIP {node} cards {dev} occupied {cur}"); continue
            port=PORT[(node,idx)]
            e=dict(node=node,address=ADDR[node],devices=dev)
            cmd=command_for(e,max_running=a.max_running,mamba_slots=a.mamba_slots)
            cmd=[s.replace('zhujianian-as-multinode-pilot',f'zhujianian-agentx-{a.tag}-{node}{idx}')
                   .replace('/home/zhujianian/hygon-runtime-overlay',str(ROOT/'experiments/weka_baselines/environment/runtime'))
                   .replace('/home/zhujianian/Qwen3.6-35B-A3B','/data/node28/shared/zhujianian/Qwen3.6-35B-A3B')
                   .replace('--port 31600',f'--port {port}')
                   for s in cmd]
            if a.dry_run:
                print(f"DRY {node} cards {dev} port {port}"); records.append(dict(e,port=port)); continue
            cid=subprocess.check_output(['ssh','-o','BatchMode=yes',node,shlex.join(cmd)],text=True).strip()
            rec=dict(e,port=port,container_id=cid,endpoint=f"http://{ADDR[node]}:{port}")
            records.append(rec); print(f"started {node} cards {dev} -> {rec['endpoint']} ({cid[:12]})",flush=True)
    (out/f'launch-{a.tag}.json').write_text(json.dumps(records,indent=2)+'\n')

if __name__=='__main__': main()
