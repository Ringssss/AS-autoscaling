"""Resume disjoint model shards with the user's hfd.sh on three nodes.

Public repository: no token passed or persisted. Separate .hfd directories
avoid concurrent resume-metadata races. Completed shards are published only
after every worker succeeds; tensor headers/index consistency are then checked.
"""
import concurrent.futures
import json
from pathlib import Path
import shlex
import struct
import subprocess


ROOT = Path('/data/node29/zhujianian/Qwen3.8-Flash-Next')
REVISION = 'de4b8e4d43b917e7706784d8bb445c9af86a3540'
LOGS = Path('/home/zhujianian/AS-autoscaling/experiments/multinode/results/highbs')


def main():
    index = json.loads((ROOT/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    tasks = []
    for offset, node in enumerate(['node27','node28','node29']):
        assigned = shards[offset::3]
        folder = ROOT.parent / ('Qwen3.8-Flash-Next-parts-'+node)
        folder.mkdir(exist_ok=True)
        for name in assigned:
            for suffix in ('','.aria2'):
                path=ROOT/(name+suffix)
                if path.exists():
                    target=folder/path.name
                    if target.exists():raise ValueError('conflicting partial '+str(target))
                    path.rename(target)
        command=['env','PATH=/home/zhujianian/miniconda3/envs/download-tools/bin:/usr/local/bin:/usr/bin:/bin',
                 'HF_ENDPOINT=https://hf-mirror.com','bash','/home/zhujianian/hfd.sh',
                 'Qwen/Qwen3.8-Flash-Next','--tool','aria2c','-x','16','-j','8',
                 '--revision',REVISION,'--local-dir',str(folder),'--include',*assigned]
        tasks.append((node,folder,assigned,command))
    (LOGS/'download-commands.json').write_text(json.dumps({node:command for node,_,_,command in tasks},indent=2)+'\n')
    def worker(task):
        node,folder,assigned,command=task
        with (LOGS/f'download-{node}.log').open('w') as log:
            result=subprocess.run(['ssh','-o','BatchMode=yes',node,shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
        print(node,'download exit',result.returncode,flush=True)
        return result.returncode
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        codes=list(executor.map(worker,tasks))
    if any(codes):raise SystemExit('At least one shard downloader failed; retained partials and logs')
    for _,folder,assigned,_ in tasks:
        for name in assigned:
            if (folder/(name+'.aria2')).exists():raise ValueError('unfinished shard '+name)
            (folder/name).rename(ROOT/name)
    tensor_bytes=0;keys={};checks=[]
    for name in shards:
        p=ROOT/name
        with p.open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0]
            header=json.loads(f.read(n))
        end=0
        for key,value in header.items():
            if key=='__metadata__':continue
            if key in keys:raise ValueError('duplicate tensor '+key)
            keys[key]=name
            start,stop=value['data_offsets'];tensor_bytes+=stop-start;end=max(end,stop)
        if p.stat().st_size!=8+n+end:raise ValueError('shard file size does not match header '+name)
        checks.append({'file':name,'size':p.stat().st_size,'header_bytes':n})
    assert keys==index['weight_map']
    assert tensor_bytes==int(index['metadata']['total_size'])
    report={'revision':REVISION,'shards':checks,'tensor_bytes':tensor_bytes,'index_keys_match':True,
            'scope':'all aria2 transfers complete; safetensors header/size/index validation, not cryptographic content verification'}
    (LOGS/'qwen38-download-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print('All model shards downloaded and validated',tensor_bytes,flush=True)


if __name__=='__main__':
    main()
