"""Live correctness gate for preserving nonempty destination cache."""
import asyncio
import json
import time
import uuid
from pathlib import Path
import aiohttp
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.weka_baselines.replay import generate,write
from experiments.fluidkv.benchmark_native_pairs import flush
from experiments.multinode.benchmark_highbs import settle_engines

OUT=Path(__file__).parent/'results/nonempty-validation'
URLS=['http://12.11.3.28:31780','http://12.11.3.29:31780']


async def run():
    OUT.mkdir(exist_ok=False)
    clients={n:SGLangAgentShiftClient(n,u,timeout=180) for n,u in zip(['source','destination'],URLS)}
    store=SQLiteStateStore(OUT/'ownership.db')
    coord=HybridMigrationCoordinator(store,clients,master_address='12.11.3.28',base_port=24500)
    rows=[]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        deadline=time.monotonic()+600
        while True:
            ready=[]
            for u in URLS:
                try:
                    async with session.get(u+'/health',timeout=aiohttp.ClientTimeout(total=3)) as r:ready.append(r.status==200)
                except Exception:ready.append(False)
            if all(ready):break
            if time.monotonic()>deadline:raise TimeoutError('engines not ready')
            await asyncio.sleep(5)
        async def inspect(index,tokens,digest=False):
            async with session.post(URLS[index]+'/agentshift/hybrid_state/inspect',json=dict(token_ids=tokens,return_digest=digest)) as r:
                r.raise_for_status();result=await r.json();assert result['success'];return result
        await settle_engines(session,URLS);await flush(session,URLS)
        unrelated=list(range(8000,9024));parent=list(range(1200,2224));extended=parent+list(range(10000,11024))
        await generate(session,URLS[1],unrelated,1)
        await settle_engines(session,URLS);reference=await inspect(1,unrelated,True)
        await coord._ensure_group('source','destination')
        for label,tokens in [('disjoint',parent),('shared-extension',extended),('duplicate',extended)]:
            await generate(session,URLS[0],tokens,1);await settle_engines(session,URLS)
            source=await inspect(0,tokens,True);before=await inspect(1,tokens)
            agent=uuid.uuid4().hex;store.register_agent(AgentContinuation(agent,1,'source',1,tuple(tokens)))
            moved=await coord.migrate(agent,'destination')
            after=await inspect(1,tokens);dest=await inspect(1,tokens,True);preserved=await inspect(1,unrelated,True)
            # Existing root/extension prompt boundary is executable; duplicate
            # free-list counters must return exactly to their previous values.
            expected_new=len(tokens)-before['matched_tokens']
            checks=dict(executable=dest['full_prefix_available'],state_digest_equal=source['state_sha256']==dest['state_sha256'],
                unrelated_preserved=preserved['state_sha256']==reference['state_sha256'],
                kv_capacity_delta=before['available_token_slots']-after['available_token_slots']==expected_new,
                snapshot_capacity_delta=before['available_snapshot_slots']-after['available_snapshot_slots']==(0 if before['full_prefix_available'] else 1),
                owner_committed=store.get_agent(agent).owner_engine=='destination')
            child=tokens+[180001,180002]
            a=await generate(session,URLS[0],child,16);b=await generate(session,URLS[1],child,16)
            checks['output_equal']=a['output_ids']==b['output_ids']
            checks['full_child_cache_hit']=b['meta']['cached_tokens']>=len(tokens)
            row=dict(case=label,checks=checks,source=source,destination=dest,before=before,after=after,
                transfer_bytes=moved.bytes_transferred,source_generation=a,destination_generation=b)
            rows.append(row);write(OUT/'checks.json',rows);print(label,checks,flush=True)
            assert all(checks.values()),checks
        # Missing source: destination still has unrelated warm cache. A prepare
        # failure must return all temporary KV + recurrent reservations.
        await settle_engines(session,URLS)
        missing=list(range(20000,21024));before=await inspect(1,missing)
        agent=uuid.uuid4().hex;store.register_agent(AgentContinuation(agent,1,'source',1,tuple(missing)))
        error=None
        try:await coord.migrate(agent,'destination')
        except Exception as e:error=repr(e)
        after=await inspect(1,missing);preserved=await inspect(1,unrelated,True)
        checks=dict(rejected=error is not None,kv_rollback=before['available_token_slots']==after['available_token_slots'],
            snapshot_rollback=before['available_snapshot_slots']==after['available_snapshot_slots'],
            owner_stays_source=store.get_agent(agent).owner_engine=='source',unrelated_preserved=preserved['state_sha256']==reference['state_sha256'])
        rows.append(dict(case='missing-source-rollback',checks=checks,error=error,before=before,after=after));write(OUT/'checks.json',rows)
        assert all(checks.values()),checks
        write(OUT/'summary.json',dict(all_passed=True,cases=len(rows),limitations='Idle engines only; no source release; full temporary allocation; no forced capacity eviction.'))
        print('NONEMPTY_VALIDATED',flush=True)


if __name__=='__main__':asyncio.run(run())
