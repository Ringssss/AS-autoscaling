"""Three private histories, three nodes, one controlled return burst.

This is an isolated mechanism experiment, not a native trace replay or a
continuous load-balancing benchmark. The current importer needs empty targets.
Both copy and precompute receive the identical one-second preparation window.
"""
import asyncio
import json
from pathlib import Path
import random
import time
import uuid

import aiohttp
import numpy as np
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import stream, flush


async def run():
    out=Path(__file__).parent/'results'/'three-node-burst'
    out.mkdir(exist_ok=False)
    urls=[f'http://12.11.3.{n}:31720' for n in [27,28,29]]
    names=['source','target1','target2']
    clients={n:SGLangAgentShiftClient(n,u,timeout=300) for n,u in zip(names,urls)}
    store=SQLiteStateStore(out/'ownership.db')
    coordinator=HybridMigrationCoordinator(store,clients,master_address='12.11.3.27',base_port=32600)
    # The inherited coordinator allocates adjacent pair ports. TP2 reserves
    # four ports per pair, so leave eight ports between the used pair starts.
    coordinator._pair_indices={pair:i*8 for i,pair in enumerate(coordinator._pair_indices)}
    corpus=np.load(Path(__file__).parent/'results'/'component_tokens.npz')['tokens'].tolist()
    parents=[[20001+i]+corpus[:32767] for i in range(3)]
    children=[p+corpus[40000:40032] for p in parents]
    rows=[]
    rng=random.Random(20260928)
    gap=1.0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        # Warm both the three-request source batch and each destination shape.
        for url in urls:
            await flush(session,[url])
            for parent in parents:await stream(session,url,parent,1)
            await asyncio.gather(*(stream(session,url,c,32) for c in children))
        init_start=time.perf_counter()
        for n in names[1:]:await coordinator._ensure_group('source',n)
        group_init=time.perf_counter()-init_start
        for i,n in enumerate(names[1:],1):
            await flush(session,urls)
            await stream(session,urls[0],parents[i],1)
            agent='transport-'+n
            store.register_agent(AgentContinuation(agent,1,'source',1,tuple(parents[i])))
            await coordinator.migrate(agent,n)
        with (out/'requests.jsonl').open('w',buffering=1) as log:
            for repetition in range(3):
                paired=[]
                for mode in rng.sample(['sticky','cold-balanced','copy-in-gap','recompute-in-gap'],4):
                    await flush(session,urls)
                    for parent in parents:await stream(session,urls[0],parent,1)
                    # Parents differ at their very first token: no common
                    # global template is being replicated as private history.
                    agents=[]
                    for i in range(3):
                        agent=uuid.uuid4().hex;agents.append(agent)
                        store.register_agent(AgentContinuation(agent,1,'source',1,tuple(parents[i])))
                    start=time.perf_counter()
                    prep_ends=[start]*3
                    moved=[0]*3
                    ready=[asyncio.Event() for _ in range(3)]
                    ready[0].set()
                    async def prepare():
                        if mode=='copy-in-gap':
                            for i in [1,2]:
                                migration=await coordinator.migrate(agents[i],names[i])
                                moved[i]=migration.bytes_transferred
                                prep_ends[i]=time.perf_counter();ready[i].set()
                        elif mode=='recompute-in-gap':
                            async def precompute(i):
                                await stream(session,urls[i],parents[i],1)
                                prep_ends[i]=time.perf_counter();ready[i].set()
                            await asyncio.gather(*(precompute(i) for i in [1,2]))
                        else:
                            for i in [1,2]:ready[i].set()
                    preparation=asyncio.create_task(prepare())
                    async def continuation(i):
                        await asyncio.sleep(max(0,start+gap-time.perf_counter()))
                        await ready[i].wait()
                        target=0 if mode=='sticky' else i
                        first_at=None
                        async def first(*_):
                            nonlocal first_at
                            first_at=time.perf_counter()
                        generation=await stream(session,urls[target],children[i],32,on_first_token=first)
                        end=time.perf_counter()
                        return dict(agent_index=i,target=target,generation=generation,
                            return_to_first_s=first_at-(start+gap),return_to_end_s=end-(start+gap),
                            prep_seconds=prep_ends[i]-start,exposed_prep_s=max(0,prep_ends[i]-start-gap),
                            transfer_bytes=moved[i])
                    tasks=[asyncio.create_task(continuation(i)) for i in range(3)]
                    # Propagate preparation failures rather than leaving tasks
                    # waiting on a never-set readiness event.
                    try:
                        await preparation
                        results=await asyncio.gather(*tasks)
                    except BaseException:
                        for task in tasks:task.cancel()
                        await asyncio.gather(*tasks,return_exceptions=True)
                        raise
                    row=dict(repetition=repetition,mode=mode,controlled_gap_s=gap,
                        agents=results,makespan_s=max(r['return_to_end_s'] for r in results),
                        median_ttft_s=float(np.median([r['return_to_first_s'] for r in results])),
                        total_transfer_bytes=sum(moved))
                    paired.append(row)
                baseline=next(r for r in paired if r['mode']=='sticky')
                cold=next(r for r in paired if r['mode']=='cold-balanced')
                for row in paired:
                    checks=[]
                    for i,r in enumerate(row['agents']):
                        gen=r['generation'];meta=gen['meta_info']
                        expected=0 if row['mode']=='cold-balanced' and i>0 else 32768
                        checks.append(dict(input_length=meta['prompt_tokens']==32800,
                            output_length=meta['completion_tokens']==32,
                            cached_tokens=meta['cached_tokens']==expected,
                            exact_sticky_output=gen['output_ids']==baseline['agents'][i]['generation']['output_ids'],
                            exact_cold_output=gen['output_ids']==cold['agents'][i]['generation']['output_ids']))
                    row['checks']=checks
                    log.write(json.dumps(row)+'\n');rows.append(row)
                    print(json.dumps({k:row[k] for k in ['repetition','mode','median_ttft_s','makespan_s']}),flush=True)
        summary=dict(scope=__doc__,group_initialization_s=group_init,
            requests=len(rows)*3,all_checks_passed=all(all(c.values()) for r in rows for c in r['checks']),
            models='Qwen3.6-35B-A3B, TP2, 3 nodes / 6 cards',prefix_tokens=32768,
            output_tokens=32,controlled_gap_s=gap,
            modes={m:dict(repetitions=3,median_ttft_s=float(np.median([r['median_ttft_s'] for r in rows if r['mode']==m])),
                median_makespan_s=float(np.median([r['makespan_s'] for r in rows if r['mode']==m])),
                median_agent_e2e_s=float(np.median([a['return_to_end_s'] for r in rows if r['mode']==m for a in r['agents']])))
                for m in ['sticky','cold-balanced','copy-in-gap','recompute-in-gap']})
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(summary,indent=2),flush=True)
        if not summary['all_checks_passed']:raise RuntimeError('correctness gate failed')


if __name__=='__main__':asyncio.run(run())
