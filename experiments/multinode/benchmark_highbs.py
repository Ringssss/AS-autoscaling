"""Real DualMap/CacheRoute decisions with/without controlled state preparation.

Known-prefix stress workload, not native Kimi replay. The preparation ablation
replicates one observed warm shared prefix during a controlled tool window;
the router itself is unchanged. Unique cold prefixes have no transferable
source state. All-hot prefixes need no preparation. Actual cache hits, failures,
router cost, queueing, engine running batch and copy costs are retained.
"""
import argparse
import asyncio
from collections import Counter
import hashlib
import gc
import json
from pathlib import Path
import random
import time
import uuid

import aiohttp
import numpy as np

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.fluidkv.dualmap_adapter import DualMapAdapter, UPSTREAM_COMMIT
from experiments.multinode.cacheroute import Plan, build_plan, dispatch


def blocks(tokens):
    return [int.from_bytes(hashlib.blake2b(np.asarray(tokens[i:i+16], dtype='<i4').tobytes(), digest_size=8).digest(), 'little') for i in range(0, len(tokens), 16)]


async def settle_engines(session, urls):
    # Streaming completion can precede overlap-scheduler cleanup. This is
    # common pre-trial setup, before the controlled tool window starts.
    await asyncio.sleep(.2)
    for url in urls:
        for _ in range(100):
            async with session.get(url+'/get_load') as response:
                response.raise_for_status()
                values=await response.json()
            if all(v['num_reqs']==0 for v in values):
                break
            await asyncio.sleep(.1)
        else:
            raise RuntimeError('engine did not drain before trial: '+url)
    await asyncio.sleep(.2)


async def telemetry(session, urls, stop, rows, began):
    async def read(i, url):
        try:
            async with session.get(url + '/get_load', timeout=aiohttp.ClientTimeout(total=5)) as response:
                response.raise_for_status()
                values = await response.json()
            for v in values:
                rows.append({'engine': i, 'seconds': time.perf_counter()-began,
                             'running': v['num_reqs']-v['num_waiting_reqs'],
                             'waiting': v['num_waiting_reqs'], 'raw': v})
        except Exception as exc:
            rows.append({'engine': i, 'seconds': time.perf_counter()-began, 'error': repr(exc)})
    while not stop.is_set():
        await asyncio.gather(*(read(i, u) for i, u in enumerate(urls)))
        try:
            await asyncio.wait_for(stop.wait(), timeout=.2)
        except asyncio.TimeoutError:
            pass


async def run_trial(args, session, corpus, coordinators, store, calibration,
                    length, per_engine, regime, policy, repeat):
    urls = args.engines
    withdraw_source = getattr(args, 'withdraw_source', False)
    n = per_engine * (len(urls) - int(withdraw_source))
    parent = corpus[:length]
    requests = []
    for i in range(n):
        prefix = list(parent)
        if regime == 'unique-cold':
            prefix[0] = 1000 + i  # disjoint first token: no cross-request prefix reuse
        suffix = [170000 + ((i*79+j*31) % 40000) for j in range(args.suffix_tokens)]
        requests.append(prefix + suffix)
    # Immutable-prefix hashes are workload metadata, as in the existing
    # DualMap trace adapter. Do not repeatedly rehash the entire long prefix on
    # only one router's return path; both policies receive prepared input keys.
    request_hashes = [blocks(tokens) for tokens in requests]
    retained = getattr(args, 'retain_cache', False)
    if not retained:
        await flush(session, urls)
    warm = list(range(len(urls))) if regime == 'all-hot' else ([0] if regime == 'single-hot' else [])
    for i in ([] if retained else warm):
        await stream(session, urls[i], parent, 1)
    if getattr(args, 'settle_before_trial', False):
        await settle_engines(session, urls)
    observe_enabled = getattr(args, 'observe_checkpoints', False) and regime != 'unique-cold'
    observations = []
    async def inspect_parent(target, phase):
        started = time.perf_counter()
        async with session.post(urls[target]+'/agentshift/hybrid_state/inspect', json={'token_ids':parent}) as response:
            response.raise_for_status()
            result = await response.json()
        if not result.get('success') or 'snapshot_boundaries' not in result:
            raise RuntimeError('complete checkpoint observation API required')
        observations.append({'target':target,'engine_url':urls[target],'phase':phase,'wall_seconds':time.perf_counter()-started,'result':result})
        return result
    initial_observations = await asyncio.gather(*(inspect_parent(i,'initial') for i in range(len(urls)))) if observe_enabled else []
    preparation = []
    tool_start = time.perf_counter()
    ready_at = tool_start + args.tool_seconds
    if policy.endswith('+fluidkv') and regime == 'single-hot':
        for target, coordinator in coordinators.items():
            if observe_enabled and initial_observations[target]['full_prefix_available']:
                continue
            agent = uuid.uuid4().hex
            store.register_agent(AgentContinuation(agent, 1, 'source', 1, tuple(parent)))
            started = time.perf_counter()
            result = await coordinator.migrate(agent, 'destination')
            preparation.append({'target': target, 'start_since_tool': started-tool_start,
                                'commit_since_tool': time.perf_counter()-tool_start,
                                'bytes': result.bytes_transferred,
                                'owner': store.get_agent(agent).owner_engine})
            warm.append(target)
            if observe_enabled:
                initial_observations[target] = await inspect_parent(target,'prepared')
    if policy.endswith('+precompute') and regime == 'single-hot':
        # Independent targets can prefill concurrently. Give this stronger
        # baseline the same known prefix and tool window as state transfer.
        async def prepare(target):
            if observe_enabled and initial_observations[target]['full_prefix_available']:
                return
            started = time.perf_counter()
            result = await stream(session, urls[target], parent, 1)
            preparation.append({'target': target, 'start_since_tool': started-tool_start,
                                'commit_since_tool': time.perf_counter()-tool_start,
                                'bytes': 0, 'kind': 'precompute', 'engine': result})
            warm.append(target)
            if observe_enabled:
                initial_observations[target] = await inspect_parent(target,'prepared')
        await asyncio.gather(*(prepare(i) for i in range(1,len(urls))))
    prep_done = time.perf_counter()
    await asyncio.sleep(max(0, ready_at-time.perf_counter()))
    if withdraw_source:
        # Logical membership change is identical for every policy. The source
        # remains model-resident for setup; it is not an eligible return route.
        urls = urls[1:]
        initial_observations = initial_observations[1:]
        warm = [i-1 for i in warm if i > 0]
    # Every policy is given the same metadata interface; transferred residency
    # changes the estimates only after real destination installation succeeds.
    router_start = time.perf_counter()
    base_policy = policy.split('+')[0]
    adapter = None
    if base_policy == 'dualmap':
        adapter = DualMapAdapter(args.dualmap_source, len(urls), args.cache_tokens,
                                 20480, args.slo, calibration['prefill_seconds_per_token'],
                                 hybrid_snapshot_slots=args.snapshot_slots,
                                 pending_prefills=getattr(args,'pending_prefills',1),
                                 inspect_snapshots=observe_enabled)
        if observe_enabled:
            for i, response in enumerate(initial_observations):
                adapter.observe_snapshot_response(i, parent, response)
        else:
            for i in warm:
                adapter.replica_for_engine(i).save_prefill_token_ids(-1, parent)
    plan = calibration['plan']
    if withdraw_source:
        plan = build_plan(calibration['training_rates'],len(urls),calibration['qcap'],args.keys*2)
    router_setup = time.perf_counter()-router_start
    rng = random.Random(args.seed + repeat)
    live = [0] * len(urls)
    records, samples, active = [], [], set()
    scheduling_event = asyncio.Event()
    known_full = {i for i, response in enumerate(initial_observations) if response['full_prefix_available']} if observe_enabled else set(warm)
    pending_observations = {}
    observation_tasks = []
    def schedule_observation(target):
        if not observe_enabled or target in known_full:
            return
        old = pending_observations.get(target)
        if old is not None and not old.done():
            return
        async def observe():
            try:
                response = await inspect_parent(target,'serving')
                if adapter is not None:
                    adapter.observe_snapshot_response(target,parent,response)
                if response['full_prefix_available']:
                    known_full.add(target)
            except Exception as exc:
                observations.append({'target':target,'phase':'serving','error':repr(exc)})
                if adapter is not None:
                    adapter.invalidate_snapshot_engine(target)
                known_full.discard(target)
            finally:
                scheduling_event.set()
        task = asyncio.create_task(observe())
        pending_observations[target] = task
        observation_tasks.append(task)
    stop = asyncio.Event()
    monitor = asyncio.create_task(telemetry(session, urls, stop, samples, ready_at))
    routing_cpu = 0.0
    async def serve(i, target, request=None):
        dispatched = time.perf_counter()
        live[target] += 1
        full_at_dispatch = target in known_full
        first_at = None
        async def first(payload, delay):
            nonlocal first_at
            first_at = time.perf_counter()
            if adapter is not None:
                await adapter.first_token(request, delay)
                scheduling_event.set()
            if observe_enabled:
                cached = payload.get('meta_info',{}).get('cached_tokens')
                if full_at_dispatch and cached is not None and cached < length:
                    known_full.discard(target)
                    if adapter is not None:
                        adapter.invalidate_snapshot_engine(target)
                schedule_observation(target)
        error = None
        result = None
        try:
            result = await stream(session, urls[target], requests[i], args.output_tokens, on_first_token=first)
            if result['meta_info']['prompt_tokens'] != len(requests[i]) or result['meta_info']['completion_tokens'] != args.output_tokens:
                raise ValueError('engine changed input/output length')
        except Exception as exc:
            error = repr(exc)
        finished = time.perf_counter()
        if adapter is not None:
            await adapter.finished(request, result['output_ids'] if result else [], error is None)
        live[target] -= 1
        schedule_observation(target)
        records.append({'request_id': i, 'key': str(i % args.keys), 'target': target,
                        'engine_url': urls[target],
                        'prompt_tokens': len(requests[i]), 'success': error is None, 'error': error,
                        'queue_and_preparation_seconds': dispatched-ready_at,
                        'ttft_seconds': first_at-ready_at if first_at is not None else None,
                        'e2e_seconds': finished-ready_at,
                        'engine': result, 'input_sha256': hashlib.sha256(np.asarray(requests[i],dtype='<i4').tobytes()).hexdigest()})
        scheduling_event.set()
    async def dispatch_ready():
        nonlocal routing_cpu
        t=time.perf_counter()
        batch=await adapter.dispatch()
        routing_cpu+=time.perf_counter()-t
        for request in batch:
            active.add(asyncio.create_task(serve(request._id,adapter.physical_target(request),request)))
    try:
        if adapter is not None:
            # Trigger on actual arrivals and state changes. A millisecond busy
            # poll repeatedly rebalanced an unchanged queue and manufactured a
            # large CPU tax; it does not represent the upstream event driver.
            for i, tokens in enumerate(requests):
                t = time.perf_counter()
                await adapter.enqueue(i, {'chat_id': i, 'hash_ids': request_hashes[i], 'turn': 2}, tokens, args.output_tokens)
                routing_cpu += time.perf_counter()-t
                await dispatch_ready()
                await asyncio.sleep(0)
            while adapter.queued() or active:
                if time.perf_counter()-ready_at > args.trial_timeout:
                    raise TimeoutError('trial deadline exceeded')
                scheduling_event.clear()
                await dispatch_ready()
                completed = {task for task in active if task.done()}
                for task in completed:
                    task.result()
                active.difference_update(completed)
                if active or adapter.queued():
                    if not active and not any(not task.done() for task in observation_tasks):
                        raise RuntimeError('router has queued requests but no pending work can release its budget')
                    await asyncio.wait_for(scheduling_event.wait(),timeout=max(.001,args.trial_timeout-(time.perf_counter()-ready_at)))
        else:
            for i in range(n):
                t = time.perf_counter()
                target = 0 if base_policy == 'sticky' else dispatch(plan, str(i % args.keys), live, rng)
                routing_cpu += time.perf_counter()-t
                active.add(asyncio.create_task(serve(i, target)))
                await asyncio.sleep(0)  # reflect actual dispatch in live load
            await asyncio.wait_for(asyncio.gather(*active), timeout=args.trial_timeout)
    finally:
        if observation_tasks:
            await asyncio.gather(*observation_tasks)
        stop.set()
        await monitor
    elapsed = time.perf_counter()-ready_at
    good = [r for r in records if r['success']]
    summary = {'policy': policy, 'regime': regime, 'repeat': repeat, 'prefix_tokens': length,
               'offered_per_engine': per_engine, 'offered_total': n, 'completed': len(records),
               'successful': len(good), 'errors': len(records)-len(good), 'duration_seconds': elapsed,
               'routing_cpu_seconds': routing_cpu, 'router_setup_seconds': router_setup,
               'dualmap_pending_prefills':getattr(args,'pending_prefills',1),
               'checkpoint_observation':observe_enabled,'checkpoint_observations':observations,
               'tool_window_seconds': args.tool_seconds,
               'retained_cache': retained,
               'source_withdrawn_from_routing': withdraw_source,
               'eligible_engine_urls': urls,
               'preparation_seconds': prep_done-tool_start, 'preparation_exposed_seconds': max(0,prep_done-ready_at),
               'preparation': preparation, 'total_transfer_bytes': sum(x['bytes'] for x in preparation),
               'target_counts': dict(Counter(r['target'] for r in records)),
               'ttft_p50_p95_p99': np.percentile([r['ttft_seconds'] for r in good],[50,95,99]).tolist() if good else None,
               'e2e_p50_p95_p99': np.percentile([r['e2e_seconds'] for r in good],[50,95,99]).tolist() if good else None,
               'requests_per_second': len(good)/elapsed, 'output_tokens_per_second': len(good)*args.output_tokens/elapsed,
               'slo_attainment': sum(r['ttft_seconds']<=args.slo for r in good)/n,
               'actual_cached_tokens': sum(r['engine']['meta_info']['cached_tokens'] for r in good),
               'prompt_tokens': n*(length+args.suffix_tokens),
               'observed_peak_running_by_engine': {str(i):max([s['running'] for s in samples if s['engine']==i and 'running' in s],default=0) for i in range(len(urls))},
               'observed_peak_waiting_by_engine': {str(i):max([s['waiting'] for s in samples if s['engine']==i and 'waiting' in s],default=0) for i in range(len(urls))},
               'upstream_rebalances': adapter.scheduler.rebalance_cnt if adapter else None}
    return summary, records, samples


async def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'command.json').write_text(json.dumps({'argv':__import__('sys').argv,'cwd':str(Path.cwd())},indent=2)+'\n')
    corpus = np.load(args.tokens)['tokens'].tolist()
    if max(args.lengths)>len(corpus):
        raise ValueError('insufficient corpus')
    store = SQLiteStateStore(args.output/'ownership.db')
    coordinators = {}
    for i, url in enumerate(args.engines[1:],1):
        coordinators[i] = HybridMigrationCoordinator(store,
            {'source':SGLangAgentShiftClient('source',args.engines[0],timeout=300),
             'destination':SGLangAgentShiftClient('destination',url,timeout=300)},
            master_address=args.master_address,base_port=args.base_port+8*(i-1))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.trial_timeout),connector=aiohttp.TCPConnector(limit=0, force_close=True)) as session:
        servers=[]
        for url in args.engines:
            async with session.get(url+'/get_server_info') as response:
                response.raise_for_status();servers.append(await response.json())
        (args.output/'server_info.json').write_text(json.dumps(servers,indent=2)+'\n')
        # Reuse fixed calibration for independent-process null controls; do not
        # confound a no-op label with a newly estimated routing cost or plan.
        if args.calibration_file:
            calibration=json.loads(args.calibration_file.read_text())
            spec=calibration['plan']
            plan=Plan(assignments={k:tuple(v) for k,v in spec['assignments'].items()},
                      expected_load=tuple(spec['expected_load']),slots_used=spec['slots_used'],
                      construction_seconds=spec['construction_seconds'])
            if len(plan.expected_load)!=len(args.engines):raise ValueError('calibration engine count mismatch')
            calibration['plan']=plan
            cost=calibration['prefill_seconds_per_token']
        else:
            times=[]
            for length in [1024,8192]:
                await flush(session,[args.engines[0]])
                await stream(session,args.engines[0],corpus[:length],1)
                await flush(session,[args.engines[0]])
                times.append((await stream(session,args.engines[0],corpus[:length],1))['ttft_seconds'])
            cost=max(1e-6,(times[1]-times[0])/7168)
            warm=await stream(session,args.engines[0],corpus[:8192]+[170000]*args.suffix_tokens,args.output_tokens)
            qcap=1/max(warm['e2e_seconds'],1e-6)
            rates={str(i):qcap/4 for i in range(args.keys)}
            plan=build_plan(rates,len(args.engines),qcap,args.keys*2)
            calibration={'prefill_seconds_per_token':cost,'qcap':qcap,'training_rates':rates,'plan':plan}
        # Load/validate the upstream implementation before all measured cases.
        DualMapAdapter(args.dualmap_source, len(args.engines), args.cache_tokens,
                       20480, args.slo, cost, hybrid_snapshot_slots=args.snapshot_slots,
                       pending_prefills=args.pending_prefills)
        (args.output/'calibration.json').write_text(json.dumps(calibration,default=lambda x:vars(x),indent=2)+'\n')
        # Warm concurrent decoding shapes on every engine, outside timed trials.
        for url in ([] if args.reuse_compiled_kernels else args.engines):
            for length in args.lengths:
                await flush(session,[url])
                await stream(session,url,corpus[:length],1)
                warm_n=min(max(args.batches),128)
                await asyncio.gather(*(stream(session,url,corpus[:length]+[170000+i]*args.suffix_tokens,args.output_tokens) for i in range(warm_n)))
        for i,c in (coordinators.items() if any(p.endswith('+fluidkv') for p in args.policies) and 'single-hot' in args.regimes else []):
            await flush(session,args.engines)
            await stream(session,args.engines[0],corpus[:1024],1)
            await settle_engines(session,args.engines)
            agent='warm-'+str(i)
            store.register_agent(AgentContinuation(agent,1,'source',1,tuple(corpus[:1024])))
            began=time.perf_counter();await c.migrate(agent,'destination')
            print(json.dumps({'group_warmup':i,'seconds':time.perf_counter()-began}),flush=True)
        rng=random.Random(args.seed)
        summaries=[]
        with (args.output/'summary.jsonl').open('w',buffering=1) as out:
            for length in args.lengths:
                for batch in args.batches:
                    for repeat in range(args.repetitions):
                        for regime in args.regimes:
                            for policy in rng.sample(args.policies,len(args.policies)):
                                if args.reuse_compiled_kernels:
                                    gc.collect()
                                print(json.dumps({'starting':policy,'regime':regime,'length':length,'bs':batch,'repeat':repeat}),flush=True)
                                for wave in range(args.waves):
                                    args.retain_cache = wave > 0
                                    summary,records,samples=await run_trial(args,session,corpus,coordinators,store,calibration,length,batch,regime,policy,repeat)
                                    summary['wave'] = wave
                                    name=f'{length}-{batch}-{repeat}-{regime}-{policy}' + (f'-wave{wave}' if args.waves > 1 else '')
                                    (args.output/(name+'.json')).write_text(json.dumps({'summary':summary,'requests':records,'telemetry':samples},indent=2)+'\n')
                                    summaries.append(summary);out.write(json.dumps(summary)+'\n');print(json.dumps({k:v for k,v in summary.items() if k != 'checkpoint_observations'}),flush=True)
        (args.output/'summary.json').write_text(json.dumps({'scope':'real routing algorithms with shared-prefix state-preparation ablation; known-prefix stress workload, not native Kimi replay',
            'upstream_dualmap_commit':UPSTREAM_COMMIT,'cacheroute':'paper Algorithm 1 reimplementation',
            'fluidkv_scope':'proactive shared-prefix replication during a controlled window, same router, no general mixed-cache async claim',
            'precompute_scope':'same known prefix/window; parallel target prefill, exposed tail charged',
            'wave_scope':'identical shared-prefix return bursts; caches retained after wave zero',
            'scheduler_driver':'arrival/first-token/completion events, no repeated polling of unchanged queues',
            'runs':summaries},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--engines',nargs='+',default=[f'http://12.11.3.{n}:31600' for n in (27,28,29)])
    p.add_argument('--master-address',default='12.11.3.27')
    p.add_argument('--base-port',type=int,default=30800)
    p.add_argument('--tokens',type=Path,default=Path('experiments/multinode/data/longbench_tokens.npz'))
    p.add_argument('--dualmap-source',type=Path,default=Path('/home/zhujianian/DualMap'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',type=int,nargs='+',default=[32768,65536])
    p.add_argument('--batches',type=int,nargs='+',default=[1,4,16,32,64,128])
    p.add_argument('--policies',nargs='+',default=['dualmap','dualmap+fluidkv','cacheroute','cacheroute+fluidkv'])
    p.add_argument('--regimes',nargs='+',default=['single-hot','all-hot','unique-cold'])
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--waves',type=int,default=1,help='Consecutive return bursts with caches retained within each policy/repetition; requires checkpoint observation.')
    p.add_argument('--settle-before-trial',action='store_true')
    p.add_argument('--withdraw-source',action='store_true',help='Logically remove the warm source from return routing; no physical GPU retirement claim.')
    p.add_argument('--output-tokens',type=int,default=64)
    p.add_argument('--suffix-tokens',type=int,default=32)
    p.add_argument('--keys',type=int,default=16)
    p.add_argument('--cache-tokens',type=int,default=262144)
    p.add_argument('--snapshot-slots',type=int,default=512)
    p.add_argument('--pending-prefills',type=int,default=1)
    p.add_argument('--reuse-compiled-kernels',action='store_true',help='Only for a same-engine follow-up after the full warmup campaign; still resets caches and calibrates. Collects unreachable Python objects before each measured trial.')
    p.add_argument('--calibration-file',type=Path)
    p.add_argument('--observe-checkpoints',action='store_true',help='Shared authoritative non-mutating checkpoint observations for naturally warmed targets; requires updated Hygon observation endpoint.')
    p.add_argument('--tool-seconds',type=float,default=.5)
    p.add_argument('--slo',type=float,default=10)
    p.add_argument('--trial-timeout',type=float,default=900)
    p.add_argument('--seed',type=int,default=20260924)
    args=p.parse_args()
    if args.waves < 1 or (args.waves > 1 and (not args.observe_checkpoints or args.regimes != ['single-hot'])):
        p.error('multiple waves require --observe-checkpoints --regimes single-hot')
    if args.withdraw_source and (len(args.engines)<2 or args.regimes!=['single-hot'] or not args.observe_checkpoints or 'sticky' in args.policies):
        p.error('source withdrawal requires at least two engines, observed single-hot regime, and no sticky-source control')
    asyncio.run(main(args))
