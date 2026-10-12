"""Independent full-byte integrity check, outside performance timing.

Requires the opt-in return_digest inspection added to the Hygon overlay.
Compare each TP shard's complete KV/conv/recurrent bytes, then a short warm
continuation on both endpoints. Digest materialization is not benchmark time.
"""
import argparse
import asyncio
import json
from pathlib import Path
import uuid

import numpy as np
import requests

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore


def post(url,path,data):
    r=requests.post(url+path,json=data,timeout=300)
    r.raise_for_status()
    return r.json()


def generate(url,ids,count):
    return post(url,'/generate',{'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':count,'ignore_eos':True}})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',default='http://12.11.3.27:31600')
    p.add_argument('--destination',default='http://12.11.3.29:31600')
    p.add_argument('--master-address',default='12.11.3.27')
    p.add_argument('--base-port',type=int,default=31820)
    p.add_argument('--tokens',type=Path,default=Path('experiments/multinode/data/longbench_tokens.npz'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    corpus=np.load(args.tokens)['tokens'].tolist()
    store=SQLiteStateStore(args.output/'ownership.db')
    clients={name:SGLangAgentShiftClient(name,url,timeout=300) for name,url in [('source',args.source),('destination',args.destination)]}
    coordinator=HybridMigrationCoordinator(store,clients,master_address=args.master_address,base_port=args.base_port)
    rows=[]
    for length in [32768,65536]:
        for url in [args.source,args.destination]:requests.post(url+'/flush_cache',timeout=30).raise_for_status()
        prefix=corpus[:length]
        generate(args.source,prefix,1)
        payload={'token_ids':prefix,'return_digest':True}
        before=post(args.source,'/agentshift/hybrid_state/inspect',payload)
        agent=uuid.uuid4().hex
        store.register_agent(AgentContinuation(agent,1,'source',1,tuple(prefix)))
        migration=asyncio.run(coordinator.migrate(agent,'destination'))
        dest=post(args.destination,'/agentshift/hybrid_state/inspect',payload)
        after=post(args.source,'/agentshift/hybrid_state/inspect',payload)
        def shards(r):return [(x['state_sha256'],x['state_bytes']) for x in r['tp_ranks']]
        expected=length*20480+64389120
        a=generate(args.source,corpus[:length+32],16)
        b=generate(args.destination,corpus[:length+32],16)
        checks={'every_tp_shard_byte_identical':shards(before)==shards(dest),
                'source_bytes_unchanged':shards(before)==shards(after),
                'aggregate_digest_equal':before['state_sha256']==dest['state_sha256'],
                'state_and_transfer_size_match':before['state_bytes']==dest['state_bytes']==migration.bytes_transferred==expected,
                'warm_continuation_output_match':a['output_ids']==b['output_ids'],
                'source_full_hit':a['meta_info']['cached_tokens']==length,
                'destination_full_hit':b['meta_info']['cached_tokens']==length,
                'owner_committed':store.get_agent(agent).owner_engine=='destination'}
        row={'prefix_tokens':length,'checks':checks,'before':before,'destination':dest,'after':after,
             'source_continuation':a,'destination_continuation':b}
        rows.append(row);print(json.dumps({'length':length,'checks':checks}),flush=True)
        (args.output/'result.json').write_text(json.dumps({'scope':'independent full-byte integrity and warm continuation checks, not timed performance','rows':rows},indent=2)+'\n')
    if not all(all(r['checks'].values()) for r in rows):raise SystemExit('integrity/correctness gate failed')


if __name__=='__main__':main()
