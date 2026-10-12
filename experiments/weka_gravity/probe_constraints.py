"""Exercise the current runtime's constraints on dedicated engines only."""
import asyncio
import json
from pathlib import Path
import time

import aiohttp
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import stream, flush


async def run():
    out=Path(__file__).parent/'results'/'constraints'
    out.mkdir(exist_ok=False)
    urls={'source':'http://12.11.3.27:31720','destination':'http://12.11.3.29:31720'}
    clients={n:SGLangAgentShiftClient(n,u,timeout=150) for n,u in urls.items()}
    store=SQLiteStateStore(out/'ownership.db')
    coordinator=HybridMigrationCoordinator(store,clients,master_address='12.11.3.27',base_port=32520)
    parent=list(range(1200,2224))
    results={}
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180)) as session:
        async def inspect(name,digest=False):
            async with session.post(urls[name]+'/agentshift/hybrid_state/inspect',
                                    json={'token_ids':parent,'return_digest':digest}) as r:
                return {'http_status':r.status,'result':await r.json()}
        await flush(session,list(urls.values()))
        await stream(session,urls['source'],parent,1)
        before=await inspect('source',True)
        await coordinator._ensure_group('source','destination')
        store.register_agent(AgentContinuation('copy-check',1,'source',1,tuple(parent)))
        t=time.perf_counter()
        migration=await coordinator.migrate('copy-check','destination')
        results['successful_copy']={'seconds':time.perf_counter()-t,'bytes':migration.bytes_transferred,
            'source_before':before,'source_after':await inspect('source',True),
            'destination_after':await inspect('destination',True),
            'committed_owner':store.get_agent('copy-check').owner_engine}
        ready=asyncio.Event()
        async def first(*_):ready.set()
        busy=asyncio.create_task(stream(session,urls['source'],list(range(9000,9256)),64,on_first_token=first))
        await asyncio.wait_for(ready.wait(),90)
        results['busy_source_digest_attempt']=await inspect('source',True)
        results['foreground']=await busy
        # The destination still contains the just-transferred prefix. Even
        # an identical existing prefix is rejected by this empty-tree importer.
        store.register_agent(AgentContinuation('nonempty-check',1,'source',1,tuple(parent)))
        try:
            await coordinator.migrate('nonempty-check','destination')
            results['nonempty_destination']={'unexpected_success':True}
        except Exception as e:
            results['nonempty_destination']={'rejected':True,'error':str(e),
                'owner_after_rejection':store.get_agent('nonempty-check').owner_engine}
        (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps({'source_cache_retained':results['successful_copy']['source_after']['result'].get('full_prefix_available'),
            'busy_source_status':results['busy_source_digest_attempt']['http_status'],
            'nonempty_destination':results['nonempty_destination']},indent=2))


if __name__=='__main__':asyncio.run(run())
