"""Matched serial/concurrent hybrid-state preparation microbenchmark.

This owns no engine lifecycle. Run only on two explicitly dedicated, idle engines.
Both GPUs are charged even in foreground-only/preparation-only diagnostic controls.
Source checkpoint construction and group setup precede the measured interval;
child consumption validation follows it. Inputs are deterministic token fixtures,
not claimed to represent native agent text or tool waiting distributions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import random
import statistics
import time
import uuid

import aiohttp

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.controller.snapshot_directory import validated_snapshots
from agentshift.state.schema import AgentContinuation
from experiments.fluidkv.benchmark_decisions import ObservedTransferClient
from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.fluidkv.calibrate_decision_costs import calibration_store, digest, finish_store


def overlap_seconds(preparations, foreground):
    """Request-lifetime overlap, including queue/CPU time; not CUDA overlap."""
    intervals = sorted((max(p['start'], f['start']), min(p['finish'], f['finish']))
                       for p in preparations for f in foreground
                       if max(p['start'], f['start']) < min(p['finish'], f['finish']))
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end-start for start, end in merged)


async def generation(session, url, tokens, outputs, first_event=None):
    row = {'start': time.perf_counter(), 'input_tokens': len(tokens),
           'requested_output_tokens': outputs, 'input_sha256': digest(tokens), 'success': False}
    async def first(_payload, _latency):
        row['first'] = time.perf_counter()
        if first_event is not None:
            first_event.set()
    try:
        result = await stream(session, url, tokens, outputs, first)
        row.update(result)
        row['success'] = result['meta_info']['prompt_tokens'] == len(tokens) and result['meta_info']['completion_tokens'] == outputs and len(result['output_ids']) == outputs
        if not row['success']:
            row['error'] = 'runtime input/output length mismatch'
    except Exception as exc:
        row['error'] = repr(exc)
    row['finish'] = time.perf_counter()
    row['mean_decode_seconds_per_token'] = (row['finish']-row['first'])/(outputs-1) if 'first' in row and outputs > 1 else None
    return row


async def idle(session, urls):
    records = []
    for url in urls:
        async with session.get(url+'/v1/loads?include=all') as response:
            response.raise_for_status()
            value = await response.json()
            records.append(value)
            if any(load['num_running_reqs'] or load['num_waiting_reqs'] for load in value['loads']):
                raise RuntimeError('engine not drained; refuse flush')
    return records


async def first_or_done(event, tasks):
    waiter = asyncio.create_task(event.wait())
    pending = set(tasks)
    try:
        while not waiter.done() and pending:
            done, _ = await asyncio.wait({waiter, *pending}, return_when=asyncio.FIRST_COMPLETED)
            pending.difference_update(done)
        if not event.is_set():
            raise RuntimeError('no first foreground token; cannot trigger preparation')
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)


def summarize(row):
    fg = row['foreground']
    successful = [f for f in fg if f['success']]
    row['foreground_offered'] = len(fg)
    row['foreground_successful'] = len(successful)
    row['foreground_ttft_p50_seconds'] = statistics.median(f['ttft_seconds'] for f in successful) if successful else None
    row['foreground_e2e_p50_seconds'] = statistics.median(f['e2e_seconds'] for f in successful) if successful else None
    decode = [f['mean_decode_seconds_per_token'] for f in successful if f['mean_decode_seconds_per_token'] is not None]
    row['foreground_decode_p50_seconds_per_token'] = statistics.median(decode) if decode else None
    row['observed_overlap_seconds'] = overlap_seconds(row['preparations'], fg)
    row['request_lifetime_overlap_seconds'] = row['observed_overlap_seconds']
    row['overlap_scope'] = 'Request lifetimes include queuing/CPU; not evidence of simultaneous CUDA execution'
    row['foreground_ready_to_first_p50_seconds'] = statistics.median(f['first']-row['start'] for f in successful) if successful else None
    row['foreground_ready_to_finish_p50_seconds'] = statistics.median(f['finish']-row['start'] for f in successful) if successful else None
    row['measured_makespan_seconds'] = row['finish'] - row['start']
    row['charged_gpu_seconds'] = 2*row['measured_makespan_seconds']
    row['preparation_total_seconds'] = sum(p['finish']-p['start'] for p in row['preparations'])
    return row


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    events = []
    class LoggingClient(ObservedTransferClient):
        async def control_post(self, path, payload, **kwargs):
            started = time.perf_counter()
            record = {'engine': self.engine_id, 'path': path, 'start': started}
            try:
                result = await super().control_post(path, payload, **kwargs)
                record['response'] = result
                return result
            except Exception as exc:
                record['error'] = repr(exc)
                raise
            finally:
                record['finish'] = time.perf_counter()
                events.append(record)
    clients = {name: LoggingClient(name, url) for name, url in zip(('source', 'destination'), args.engines)}
    store = calibration_store(args)
    coordinator = HybridMigrationCoordinator(store, clients, base_port=args.transfer_port)
    rng = random.Random(args.seed)
    parents = [[2000+i]+[rng.randrange(3000, 20000) for _ in range(args.prefix_tokens-1)] for i in range(args.preparations)]
    foregrounds = [[22000+i]+[rng.randrange(23000, 29000) for _ in range(args.foreground_input-1)] for i in range(max(args.concurrency))]
    suffixes = [[rng.randrange(30000, 31000) for _ in range(64)] for _ in parents]
    (args.output/'config.json').write_text(json.dumps({**vars(args), 'output': str(args.output), 'metadata_directory': str(args.metadata_directory),
        'parent_hashes': [digest(t) for t in parents], 'foreground_hashes': [digest(t) for t in foregrounds],
        'scope': __doc__, 'gpu_budget': 2}, indent=2))
    matrix = []
    for target in args.foreground_targets:
        for concurrency in args.concurrency:
            for repetition in range(args.repetitions):
                conditions = [('none', 'foreground-only')]
                for method in args.methods:
                    conditions += [(method, mode) for mode in ('preparation-only', 'serial', 'concurrent')]
                conditions = [(method, mode) for method, mode in conditions if mode in args.modes]
                random.Random(args.seed+repetition+concurrency+100*args.foreground_targets.index(target)).shuffle(conditions)
                for method, mode in conditions:
                    matrix.append({'foreground_target': target, 'concurrency': concurrency, 'repetition': repetition, 'method': method, 'mode': mode})
    (args.output/'manifest.json').write_text(json.dumps(matrix, indent=2))
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240)) as session:
        infos = []
        for url in args.engines:
            async with session.get(url+'/get_server_info') as response:
                response.raise_for_status(); infos.append(await response.json())
        (args.output/'server_info.json').write_text(json.dumps(infos, indent=2))
        await idle(session, args.engines)
        await coordinator._ensure_group('source', 'destination')
        # Warm the same foreground shape on each endpoint; warmups excluded.
        warmups = []
        for url in args.engines:
            for concurrency in sorted(set(args.concurrency)):
                await flush(session, args.engines)
                warmed = await asyncio.gather(*(generation(session, url, tokens, args.foreground_output) for tokens in foregrounds[:concurrency]))
                warmups.extend(dict(w, endpoint=url, concurrency=concurrency) for w in warmed)
        (args.output/'warmup.json').write_text(json.dumps(warmups, indent=2))
        if not all(w['success'] for w in warmups):
            raise RuntimeError('warmup failed')
        # Warm actual copy and replay paths once before randomized measurements.
        await flush(session, args.engines)
        preparation_warmup = {'source': await generation(session, args.engines[0], parents[0], 1)}
        observed = await clients['source'].inspect_hybrid_prefix(tuple(parents[0]))
        preparation_warmup['inspection'] = observed
        if tuple(parents[0]) not in validated_snapshots(parents[0], observed):
            (args.output/'preparation-warmup.json').write_text(json.dumps(preparation_warmup, indent=2))
            finish_store(store, args)
            raise RuntimeError('warmup lacks exact8192 checkpoint; no measured runs')
        warm_agent = uuid.uuid4().hex
        store.register_agent(AgentContinuation(warm_agent, 0, 'source', 1, tuple(parents[0])))
        if 'copy' in args.methods:
            copied = await coordinator.migrate_async(warm_agent, 'destination')
            coordinator.acknowledge_async_migration(copied.migration_id)
            preparation_warmup['copy_bytes'] = copied.bytes_transferred
            if any(client.live_transfers for client in clients.values()):
                raise RuntimeError('warmup cleanup unconfirmed; refuse flush')
        if 'replay' in args.methods:
            preparation_warmup['replay'] = await generation(session, args.engines[1], [2100]+parents[0][1:], 1)
        (args.output/'preparation-warmup.json').write_text(json.dumps(preparation_warmup, indent=2))
        with (args.output/'records.jsonl').open('w', buffering=1) as log:
            for index, condition in enumerate(matrix):
                row = {'index': index, **condition, 'success': False, 'foreground': [], 'preparations': [], 'validation': []}
                tasks = []
                try:
                    if any(client.live_transfers for client in clients.values()):
                        raise RuntimeError('unconfirmed transfers; no flush or new trial allowed')
                    row['loads_before'] = await idle(session, args.engines)
                    await flush(session, args.engines)
                    row['source_checkpoint_build'] = []
                    for tokens in parents:
                        built = await generation(session, args.engines[0], tokens, 1)
                        row['source_checkpoint_build'].append(built)
                        if not built['success']:
                            raise RuntimeError('source checkpoint construction failed')
                        inspection = await clients['source'].inspect_hybrid_prefix(tuple(tokens))
                        if tuple(tokens) not in validated_snapshots(tokens, inspection):
                            row['failed_checkpoint_inspection'] = inspection
                            raise RuntimeError('required exact source checkpoint absent')
                    agents = []
                    for tokens in parents:
                        agent = uuid.uuid4().hex
                        store.register_agent(AgentContinuation(agent, 0, 'source', 1, tuple(tokens)))
                        agents.append(agent)
                    events.clear()
                    first = asyncio.Event()
                    async def prepare():
                        for agent, tokens in zip(agents, parents):
                            record = {'start': time.perf_counter(), 'success': False, 'prefix_tokens': len(tokens), 'input_sha256': digest(tokens)}
                            try:
                                if condition['method'] == 'copy':
                                    migration = await coordinator.migrate_async(agent, 'destination')
                                    coordinator.acknowledge_async_migration(migration.migration_id)
                                    record.update(transfer_bytes=migration.bytes_transferred, migration_id=migration.migration_id)
                                    if any(client.live_transfers for client in clients.values()):
                                        raise RuntimeError('copy cleanup unconfirmed')
                                else:
                                    result = await generation(session, args.engines[1], tokens, 1)
                                    record['generation'] = result
                                    if not result['success']:
                                        raise RuntimeError('replay failed')
                                    record['reconstructed_tokens'] = len(tokens)-result['meta_info']['cached_tokens']
                                record['success'] = True
                            except Exception as exc:
                                record['error'] = repr(exc)
                                raise
                            finally:
                                record['finish'] = time.perf_counter()
                                row['preparations'].append(record)
                    def launch_foreground():
                        url = clients[condition['foreground_target']].base_url
                        return [asyncio.create_task(generation(session, url, tokens, args.foreground_output, first)) for tokens in foregrounds[:condition['concurrency']]]
                    if args.torch_profile:
                        for engine_index,url in enumerate(args.engines):
                            async with session.post(url+'/start_profile',json={'activities':['CPU','GPU'],'with_stack':False,'record_shapes':False,'output_dir':str((args.output/f'profile-{index}-engine-{engine_index}').resolve())}) as response:
                                response.raise_for_status();await response.read()
                        row['profiled']=True
                    row['start'] = time.perf_counter()
                    if condition['mode'] in ('serial', 'preparation-only'):
                        await prepare()
                    if condition['mode'] != 'preparation-only':
                        tasks = launch_foreground()
                    if condition['mode'] == 'concurrent':
                        await first_or_done(first, tasks)
                        row['active_foreground_at_preparation_trigger'] = sum(not t.done() for t in tasks)
                        await prepare()
                    row['foreground'] = await asyncio.gather(*tasks)
                    row['finish'] = time.perf_counter()
                    if args.torch_profile:
                        for url in args.engines:
                            async with session.post(url+'/stop_profile') as response:response.raise_for_status();await response.read()
                    row['control_events'] = list(events)
                    row['cleanup_confirmed'] = not any(client.live_transfers for client in clients.values())
                    if not row['cleanup_confirmed']:
                        raise RuntimeError('unconfirmed cleanup')
                    row['loads_after'] = await idle(session, args.engines)
                    # Outside interval: consuming real prepared prefixes, same suffix/output.
                    if condition['method'] != 'none':
                        for tokens, suffix in zip(parents, suffixes):
                            source_child = await generation(session, args.engines[0], tokens+suffix, 8)
                            target_child = await generation(session, args.engines[1], tokens+suffix, 8)
                            row['validation'].append({'source': source_child, 'destination': target_child,
                                'destination_hit': target_child.get('meta_info', {}).get('cached_tokens') == len(tokens),
                                'cached_continuation_equal': source_child.get('output_ids') == target_child.get('output_ids') and source_child['success'] and target_child['success']})
                    row['success'] = all(f['success'] for f in row['foreground']) and all(p['success'] for p in row['preparations'])
                    # Cold replay/cached numerical parity is a separate gate.
                    row['preparation_consumed'] = all(v['destination_hit'] for v in row['validation']) if row['validation'] else None
                    row['execution_success'] = row['success']
                    row['cached_continuation_equal'] = all(v['cached_continuation_equal'] for v in row['validation']) if row['validation'] else None
                    row['mechanism_valid'] = row['execution_success'] and (condition['method'] == 'none' or (row['preparation_consumed'] and all(v['source']['success'] and v['destination']['success'] for v in row['validation']) and (condition['method'] != 'copy' or row['cached_continuation_equal'])))
                    row['numerical_gate_scope'] = 'Copy must preserve source cached continuation; replay cold/cached equality is recorded separately'
                except Exception as exc:
                    row['error'] = repr(exc)
                    row['execution_success'] = False
                    row['mechanism_valid'] = False
                    drained = await asyncio.gather(*tasks, return_exceptions=True)
                    row['foreground'] = [x for x in drained if isinstance(x, dict)]
                    row.setdefault('start', time.perf_counter()); row['finish'] = time.perf_counter()
                    row['control_events'] = list(events)
                    row['cleanup_confirmed'] = not any(client.live_transfers for client in clients.values())
                log.write(json.dumps(summarize(row))+'\n')
                print(json.dumps({k: row.get(k) for k in ('index', 'method', 'mode', 'foreground_target', 'concurrency', 'success', 'error', 'observed_overlap_seconds', 'measured_makespan_seconds')}), flush=True)
                if not row['success'] or not row['cleanup_confirmed']:
                    finish_store(store, args)
                    raise RuntimeError('failed trial preserved; stop without flushing')
    finish_store(store, args)
    (args.output/'complete.json').write_text(json.dumps({'complete': True, 'runs': len(matrix)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engines', nargs=2, default=['http://127.0.0.1:31480', 'http://127.0.0.1:31481'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--metadata-directory', type=Path, default=Path('/tmp/agentic-state-profile-metadata'))
    parser.add_argument('--transfer-port', type=int, default=35800)
    parser.add_argument('--methods', nargs='+', choices=['copy', 'replay'], default=['copy', 'replay'])
    parser.add_argument('--modes', nargs='+', choices=['foreground-only', 'preparation-only', 'serial', 'concurrent'], default=['foreground-only', 'preparation-only', 'serial', 'concurrent'])
    parser.add_argument('--foreground-targets', nargs='+', choices=['source', 'destination'], default=['destination', 'source'])
    parser.add_argument('--concurrency', nargs='+', type=int, default=[2, 8])
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--prefix-tokens', type=int, default=8192)
    parser.add_argument('--preparations', type=int, default=4)
    parser.add_argument('--foreground-input', type=int, default=1024)
    parser.add_argument('--foreground-output', type=int, default=256)
    parser.add_argument('--torch-profile', action='store_true',help='Diagnostic only; exclude these runs from performance comparisons')
    parser.add_argument('--seed', type=int, default=1509)
    args = parser.parse_args()
    if not all(1 <= c <= 8 for c in args.concurrency) or not 1 <= args.preparations <= 4 or args.prefix_tokens != 8192:
        parser.error('bounded harness requires concurrency1..8, preparations1..4, prefix8192')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
