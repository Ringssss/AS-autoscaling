"""Launch a dedicated pinned FreeToken engine with explicit offload arguments."""
import argparse,json,os,subprocess,time,urllib.request
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--model',required=True);p.add_argument('--port',type=int,default=31580);p.add_argument('--memory-ratio',type=float,default=.85);p.add_argument('--kv-reserve',type=int,default=65536);p.add_argument('--timeout',type=float,default=1800);p.add_argument('--preservation-hook',action='store_true');p.add_argument('--policy-file',type=Path);p.add_argument('--hook-log',type=Path);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    envroot=Path('experiments/freetoken_elastic/environment').resolve();exe=envroot/'.venv/bin/ft'
    cmd=[str(exe),'serve','--model',a.model,'--host','127.0.0.1','--port',str(a.port),'--gpu','0',
         '--moe-strategy','offload','--moe-cache-auto','--kv-reserve-tokens',str(a.kv_reserve),
         '--memory-ratio',str(a.memory_ratio),'--max-running-requests','2','--graph','2',
         '--max-prefill-length','8192','--max-seq-len-override','32768','--enable-cache-report',
         '--text-model-only','--sampling-defaults','none','--enable-special-token-ckpt']
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUDA_DEVICE_ORDER='PCI_BUS_ID')
    if a.preservation_hook:
        if a.policy_file is None or a.hook_log is None or not a.policy_file.is_file():raise ValueError('hook requires existing policy and log path')
        a.hook_log.parent.mkdir(parents=True,exist_ok=True)
        env['FLUIDKV_ELASTIC_POLICY']=str(a.policy_file.resolve());env['FLUIDKV_ELASTIC_LOG']=str(a.hook_log.resolve())
        env['PYTHONPATH']=str(Path('experiments/freetoken_elastic/bootstrap').resolve())+os.pathsep+env.get('PYTHONPATH','')
    with (a.output/'engine.log').open('wb') as log:
        proc=subprocess.Popen(['taskset','-c','0-15',*cmd],cwd=str(Path.cwd()),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    whitelist=('CUDA_HOME','CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','LD_LIBRARY_PATH','PATH','TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','TMPDIR','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','PYTHONPATH','FLUIDKV_ELASTIC_POLICY','FLUIDKV_ELASTIC_LOG')
    record={'pid':proc.pid,'argv':cmd,'env':{k:env[k] for k in whitelist if k in env},'start_epoch':time.time(),
            'scope':'Actual offload on H100 under separate capacity ballast; no claim consumer GPU performance.'}
    (a.output/'launch.json').write_text(json.dumps(record,indent=2)+'\n')
    start=time.perf_counter();url=f'http://127.0.0.1:{a.port}'
    while time.perf_counter()-start<a.timeout:
        if proc.poll() is not None:raise RuntimeError(f'FreeToken process exited{proc.returncode}; inspect preserved log')
        try:
            with urllib.request.urlopen(url+'/v1/cache/status',timeout=2) as response:state=json.load(response)
            if state.get('state')=='serving':
                record.update(ready_epoch=time.time(),startup_seconds=time.perf_counter()-start,status=state)
                (a.output/'launch.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'ready':proc.pid,'seconds':record['startup_seconds'],'geometry':state.get('geometry')}),flush=True);return
        except (OSError,ValueError):pass
        time.sleep(1)
    raise RuntimeError('startup timeout; preserve live launch identity for diagnosis')
if __name__=='__main__':main()
