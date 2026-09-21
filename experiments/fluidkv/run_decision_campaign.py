#!/usr/bin/env python3
"""Sequential, resumable continuous comparisons with frozen run specifications."""
import argparse
import asyncio
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path


def build_runs(args):
    files={'A':'qwen_traceA_blksz_16.jsonl','Coder':'qwen_coder_blksz_16.jsonl','Thinking':'qwen_thinking_blksz_16.jsonl'}
    rng=random.Random(args.seed)
    specifications=[]
    for scale in args.time_scales:
        for split in args.splits:
            for start in args.starts:
                for repetition in range(args.repetitions):
                    for policy in rng.sample(args.policies,len(args.policies)):
                        specifications.append({'split':split,'trace':str(args.trace_root/files[split]),
                            'start':start,'duration':args.duration,'warmup':args.warmup,'time_scale':scale,
                            'policy':policy,'repetition':repetition,'retire_offset':None if args.no_retirement else args.retire_offset,
                            'retire_engine':args.retire_engine})
    return specifications


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--costs',type=Path,required=True)
    p.add_argument('--trace-root',type=Path,default=Path('/data-aisoft/zhujianian/cloud/data/qwen-bailian-usagetraces-anon'))
    p.add_argument('--splits',nargs='+',default=['Coder'])
    p.add_argument('--starts',nargs='+',type=float,default=[3000,5400])
    p.add_argument('--time-scales',nargs='+',type=float,default=[1])
    p.add_argument('--duration',type=float,default=60)
    p.add_argument('--warmup',type=float,default=60)
    p.add_argument('--retire-offset',type=float,default=0)
    p.add_argument('--no-retirement',action='store_true')
    p.add_argument('--retire-engine',type=int,default=0)
    p.add_argument('--physical-release',action='store_true')
    p.add_argument('--policies',nargs='+',default=['dualmap','full-copy','cost-aware'])
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--seed',type=int,default=1701)
    p.add_argument('--transfer-port',type=int,default=33000)
    p.add_argument('--slo',type=float,default=1)
    p.add_argument('--run-timeout',type=float,default=1800)
    p.add_argument('--busy-copy-multiplier',type=float,default=4)
    p.add_argument('--copy-interference-seconds',type=float,default=0.02)
    p.add_argument('--warm-ttft-seconds',type=float,default=0.1)
    p.add_argument('--prefill-seconds-per-token',type=float,default=0.000035)
    p.add_argument('--cache-tokens',type=int,default=65536)
    args=p.parse_args()
    if args.physical_release and args.no_retirement:
        p.error('physical release requires a retirement event')
    specs=build_runs(args)
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'campaign.json'
    contract={'specifications':specs,'costs':str(args.costs.resolve()),'costs_sha256':hashlib.sha256(args.costs.read_bytes()).hexdigest(),'seed':args.seed,
              'slo':args.slo,'busy_copy_multiplier':args.busy_copy_multiplier,
              'copy_interference_seconds':args.copy_interference_seconds,
              'warm_ttft_seconds':args.warm_ttft_seconds,'prefill_seconds_per_token':args.prefill_seconds_per_token,
              'physical_release':args.physical_release,'cache_tokens':args.cache_tokens}
    if path.exists():
        state=json.loads(path.read_text())
        if state['contract']!=contract:
            raise ValueError('resume arguments differ from frozen campaign')
    else:
        state={'contract':contract,'started_unix':time.time(),'complete':False,'runs':[]}
        path.write_text(json.dumps(state,indent=2)+'\n')
    byindex={run['index']:run for run in state['runs']}
    for index,spec in enumerate(specs):
        if index in byindex:
            if byindex[index]['exit_code']!=0:
                raise RuntimeError('previous failed run requires diagnosis; use a new explicit campaign for repaired results')
            continue
        out=args.output/f'{index:03d}-{spec["split"]}-s{spec["start"]:g}-x{spec["time_scale"]:g}-r{spec["repetition"]}-{spec["policy"]}'
        cmd=[sys.executable,'experiments/fluidkv/benchmark_decisions.py',
            '--trace',spec['trace'],'--output',str(out),'--costs',str(args.costs),
            '--start',str(spec['start']),'--duration',str(spec['duration']),'--warmup',str(spec['warmup']),
            '--time-scale',str(spec['time_scale']),
            '--retire-engine',str(spec['retire_engine']),'--policies',spec['policy'],
            '--repetitions','1','--repetition-start',str(spec['repetition']),
            '--seed',str(args.seed),'--transfer-port',str(args.transfer_port+index*3),
            '--slo',str(args.slo),'--run-timeout',str(args.run_timeout),
            '--busy-copy-multiplier',str(args.busy_copy_multiplier),
            '--copy-interference-seconds',str(args.copy_interference_seconds),
            '--warm-ttft-seconds',str(args.warm_ttft_seconds),
            '--prefill-seconds-per-token',str(args.prefill_seconds_per_token),
            '--cache-tokens',str(args.cache_tokens)]
        if spec['retire_offset'] is not None:
            cmd.extend(['--retire-offset',str(spec['retire_offset'])])
        identity=None
        if args.physical_release:
            from experiments.fluidkv.engine_lifecycle import capture_owned_engine
            identity=capture_owned_engine(f'http://127.0.0.1:{31480+args.retire_engine}')
            cmd.append('--physical-release')
            (args.output/f'{index:03d}-original-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
        started=time.time()
        print(json.dumps({'event':'run_start','index':index,'total':len(specs),'specification':spec}),flush=True)
        with (args.output/f'{index:03d}.log').open('x') as f:
            code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT,env=os.environ.copy())
        record={'index':index,'specification':spec,'command':cmd,'output':str(out),'started_unix':started,
                'elapsed_seconds':time.time()-started,'exit_code':code}
        if code==0 and identity is not None:
            summary=json.loads((out/'summary.json').read_text())['runs'][0]
            if not (summary.get('physical_release') or {}).get('released'):
                record['exit_code']=code=1
                record['error']='requested physical release was not confirmed'
            else:
                from experiments.fluidkv.engine_lifecycle import start_engine
                try:
                    restarted=asyncio.run(start_engine(identity,args.output/f'{index:03d}-restarted-engine.log'))
                    record['restart_seconds']=restarted['ready_monotonic']-restarted['started_monotonic']
                    (args.output/f'{index:03d}-restarted-identity.json').write_text(json.dumps(restarted,indent=2)+'\n')
                except Exception as exc:
                    record['exit_code']=code=1
                    record['error']=str(exc)
        state['runs'].append(record)
        path.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({'event':'run_complete',**record}),flush=True)
        if code:
            raise SystemExit(code)
    state['complete']=True
    state['finished_unix']=time.time()
    path.write_text(json.dumps(state,indent=2)+'\n')


if __name__=='__main__':
    main()
