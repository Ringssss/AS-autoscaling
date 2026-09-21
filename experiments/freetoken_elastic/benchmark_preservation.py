"""Matched-final-geometry drop versus longest-aligned-checkpoint preservation."""
import argparse,asyncio,json,os,random,time,uuid
from pathlib import Path
import aiohttp
from experiments.freetoken_elastic.client import FreeTokenClient
from experiments.freetoken_elastic.benchmark_repartition import parser as base_parser,load_manifest,run_trial


class PolicyClient(FreeTokenClient):
    def __init__(self,*args,policy_path,hook_log,verify_bytes=False,**kwargs):
        super().__init__(*args,**kwargs);self.policy_path=policy_path;self.hook_log=hook_log;self.preservation_mode='drop';self.verify_bytes=verify_bytes
    async def rebuild(self,**geometry):
        measured=len(geometry)==1 and any(k in geometry for k in ('num_pages','num_mamba_slots'))
        policy={'id':uuid.uuid4().hex,'mode':self.preservation_mode if measured else 'drop','verify_bytes':self.verify_bytes if measured else False,'host_budget_bytes':1024**3,'max_chunk_bytes':16*1024**2}
        temp=self.policy_path.with_suffix('.tmp');temp.write_text(json.dumps(policy));temp.replace(self.policy_path)
        response=await super().rebuild(**geometry);response['experimental_policy']=policy
        matches=[]
        if self.hook_log.exists():
            for line in self.hook_log.read_text().splitlines():
                record=json.loads(line)
                if record.get('id')==policy['id']:matches.append(record)
        response['hook_records']=matches
        if response['success'] and (len(matches)!=1 or not matches[0].get('success')):
            response.update(success=False,error='missing/failed/ambiguous experimental hook record')
        if response['success'] and policy['mode']=='preserve':
            if not matches[0].get('restore',{}).get('restored'):
                response.update(success=False,error='no checkpoint restored in preservation treatment')
            elif policy['verify_bytes'] and not matches[0].get('bytes_equal'):
                response.update(success=False,error='requested byte roundtrip verification did not pass')
        return response


async def run(args):
    manifest=load_manifest(args.manifest);args.output.mkdir(parents=True,exist_ok=False);matrix=[];rng=random.Random(args.seed)
    for rep in range(args.repetitions):
        for pair in manifest['pairs']:
            for condition in args.conditions:
                order=['drop','preserve'];rng.shuffle(order)
                matrix.extend([{'index':len(matrix)+i,'pair_id':pair['id'],'condition':condition,'repetition':rep,'policy':mode} for i,mode in enumerate(order)])
    campaign={'config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'manifest':manifest,'matrix':matrix,'runs':[],'complete':False,'scope':'Matched final geometry; one longest existing aligned checkpoint, not an optimized multi-branch planner.'}
    path=args.output/'campaign.json';path.write_text(json.dumps(campaign,indent=2)+'\n');pairs={p['id']:p for p in manifest['pairs']}
    async with aiohttp.ClientSession() as session:
        client=PolicyClient(session,args.engine,timeout=args.timeout,cache_report_enabled=True,policy_path=args.policy_file,hook_log=args.hook_log,verify_bytes=args.verify_bytes)
        for spec in matrix:
            client.preservation_mode=spec['policy']
            result=await run_trial(client,args,pairs[spec['pair_id']],spec,args.output)
            measured=[op for op in result['operations'] if op['label']=='measured-rebuild']
            hook=(measured[0].get('hook_records') or [None])[0] if measured else None
            if result['success']:
                expected=(hook or {}).get('export',{}).get('boundary',0) if spec['policy']=='preserve' else 0
                if (result.get('metrics',{}).get('first_child_cached_tokens')!=expected
                        or (spec['policy']=='preserve' and expected<=0)):
                    result.update(success=False,error='first child did not consume expected restored boundary')
                    (args.output/f'trial-{spec["index"]:03d}.json').write_text(json.dumps(result,indent=2)+'\n')
            summary={k:result.get(k) for k in ('index','pair_id','condition','repetition','policy','success','error','metrics')};summary['hook']=hook
            campaign['runs'].append(summary);path.write_text(json.dumps(campaign,indent=2)+'\n')
            print(json.dumps(summary),flush=True)
            if not result['success']:return 1
    campaign['complete']=True;path.write_text(json.dumps(campaign,indent=2)+'\n');return 0


if __name__=='__main__':
    p=base_parser();p.set_defaults(conditions=['kv-resize','mamba-resize'])
    p.add_argument('--policy-file',type=Path,required=True);p.add_argument('--hook-log',type=Path,required=True);p.add_argument('--verify-bytes',action='store_true')
    args=p.parse_args();os.sched_setaffinity(0,{64,65,66,67})
    if set(args.conditions)-{'kv-resize','mamba-resize'}:p.error('preservation comparison supports KV/Mamba resize')
    raise SystemExit(asyncio.run(run(args)))
