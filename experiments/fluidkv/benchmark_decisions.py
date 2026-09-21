#!/usr/bin/env python3
"""Continuous trace comparisons with explicit preparation and lifecycle timing.

This is a bounded inference experiment. It is not a full joint scheduler, nor
does it establish semantic output equivalence of cold and cached inference.
"""

import argparse
import asyncio
from collections import Counter, OrderedDict
import hashlib
import json
import random
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path

import aiohttp
import numpy as np

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.controller.preparation_decisions import CostCurve, PreparationEstimate, choose_preparation
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore
from agentshift.workloads.production import surrogate_tokens
from experiments.fluidkv.analyze_decision_workloads import select_continuous_window
from experiments.fluidkv.benchmark_native_pairs import flush, stream
from experiments.fluidkv.dualmap_adapter import DualMapAdapter


class JsonLog:
    def __init__(self, path):
        self.path = Path(path)

    def write(self, row):
        with self.path.open('a') as f:
            f.write(json.dumps(row, allow_nan=False) + '\n')


def quantiles(values):
    return np.percentile(values, [50, 90, 99]).tolist() if values else None


def summarize(records, *, phase, slo_seconds, observation_seconds):
    rows = [r for r in records if r['phase'] == phase]
    good = [r for r in rows if r['success']]
    passed = [r for r in good if r['ttft_seconds'] <= slo_seconds]
    return {
        'offered': len(rows), 'successful': len(good), 'failed': len(rows)-len(good),
        'slo_seconds': slo_seconds, 'slo_attainment': len(passed)/len(rows) if rows else None,
        'ttft_p50_p90_p99_seconds': quantiles([r['ttft_seconds'] for r in good]),
        'e2e_p50_p90_p99_seconds': quantiles([r['e2e_seconds'] for r in good]),
        'arrival_ttft_p50_p90_p99_seconds': quantiles([r['arrival_ttft_seconds'] for r in good]),
        'arrival_e2e_p50_p90_p99_seconds': quantiles([r['arrival_e2e_seconds'] for r in good]),
        'arrival_slo_attainment': sum(r['arrival_ttft_seconds'] <= slo_seconds for r in good)/len(rows) if rows else None,
        'slo_clock': 'Causal-eligibility TTFT; original-arrival attainment reported separately.',
        'router_wait_p50_p90_p99_seconds': quantiles([r['router_wait_seconds'] for r in good]),
        'actual_cached_tokens': sum(r['actual_cached_tokens'] for r in good),
        'cache_estimate_absolute_error_tokens': sum(abs(r.get('estimated_cached_tokens',0)-r['actual_cached_tokens']) for r in good),
        'cache_overestimated_requests': sum(r.get('estimated_cached_tokens',0)>r['actual_cached_tokens'] for r in good),
        'input_tokens': sum(r['input_tokens'] for r in rows),
        'output_tokens': sum(r['completion_tokens'] for r in good),
        'observation_seconds': observation_seconds,
        'completed_per_observation_second': len(good)/observation_seconds if observation_seconds > 0 else None,
        'slo_goodput_per_observation_second': len(passed)/observation_seconds if observation_seconds > 0 else None,
        'slo_sensitivity': {str(s): sum(r['ttft_seconds'] <= s for r in good)/len(rows) if rows else None
                            for s in (0.1, 0.25, 0.5, 1.0, 2.0, 5.0)},
    }


async def timed_stream(session, url, tokens, outputs, first_callback):
    start = time.perf_counter()
    first = None
    previous = None
    intervals = []
    chunks = 0
    final = None
    async with session.post(url+'/generate', json={
        'input_ids': tokens, 'stream': True,
        'sampling_params': {'max_new_tokens': outputs, 'temperature': 0, 'ignore_eos': True},
    }) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]':
                continue
            final = json.loads(line[5:])
            if final.get('output_ids') or final.get('meta_info', {}).get('completion_tokens', 0):
                now = time.perf_counter()
                if first is None:
                    first = now
                    await first_callback(final, first-start)
                elif previous is not None:
                    intervals.append(now-previous)
                previous = now
                chunks += 1
    if first is None or final is None:
        raise RuntimeError('generation returned no output')
    return {'first_monotonic': first, 'finish_monotonic': time.perf_counter(),
            'meta_info': final['meta_info'], 'output_ids': final['output_ids'],
            'stream_chunk_count': chunks, 'inter_chunk_p50_p90_p99_seconds': quantiles(intervals)}


class ObservedTransferClient(SGLangAgentShiftClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'live_transfers', set())

    async def start_async_hybrid_state_transfer(self, **kwargs):
        self.live_transfers.add(kwargs['transfer_id'])
        return await super().start_async_hybrid_state_transfer(**kwargs)

    async def cleanup_async_hybrid_state_transfer(self, transfer_id):
        result = await super().cleanup_async_hybrid_state_transfer(transfer_id)
        # PREPARING failure is rolled back inside the runtime before the
        # coordinator's idempotent cleanup. Its MISSING response confirms that
        # the job/reservation no longer exists; it is not an unknown HTTP error.
        if result.get('state') in ('CLEANED','MISSING'):
            self.live_transfers.discard(transfer_id)
        return result


class PreparationManager:
    def __init__(self, args, adapter, session, policy, output, curves, log, began):
        self.args, self.adapter, self.session, self.policy = args, adapter, session, policy
        self.curves = curves
        self.log, self.began = log, began
        self.output = output
        metadata_root = Path(getattr(args, 'metadata_directory', '/tmp/fluidkv-metadata-20260914'))
        metadata_root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = Path(tempfile.mkdtemp(prefix='decision-', dir=metadata_root))/'ownership.db'
        self.store = SQLiteStateStore(self.metadata_path)
        self.coordinator = HybridMigrationCoordinator(self.store, {
            str(i): ObservedTransferClient(str(i), url, timeout=args.request_timeout)
            for i, url in enumerate(args.engines)}, base_port=args.transfer_port)
        self.completed = OrderedDict()
        self.arrived = {}
        self.retired = None
        self.tasks = {}
        self.records = []
        self.lock = asyncio.Lock()
        self.source_closed = False
        self.inspection_inflight = Counter()
        self.inspection_failures = []

    async def inspect_engine(self, source, tokens, *, chat_id=None, phase='routing'):
        if not getattr(self.args, 'inspect_prefixes', False):
            return ()
        if self.source_closed and source == self.retired:
            return ()
        started = time.perf_counter()
        self.inspection_inflight[source] += 1
        try:
            response = await self.coordinator.engines[str(source)].inspect_hybrid_prefix(tuple(tokens))
            confirmed = self.adapter.observe_snapshot_response(source, tokens, response)
            inspected = tuple(tokens)
            confirmed_set = set(confirmed)
            for owner,prefix in list(self.completed):
                if owner==source and len(prefix)<=len(inspected) and inspected[:len(prefix)]==prefix and prefix not in confirmed_set:
                    del self.completed[(owner,prefix)]
            self.log.write({'event': 'prefix_inspection', 'engine': source, 'query_tokens': len(tokens),
                            'chat_id':chat_id, 'phase':phase,
                            'query_token_sha256':hashlib.sha256(json.dumps(list(tokens)).encode()).hexdigest(),
                            'seconds': time.perf_counter()-self.began, 'wall_seconds': time.perf_counter()-started,
                            'result': response})
            return confirmed
        except Exception as exc:
            record = {'event': 'prefix_inspection_failed', 'engine': source,
                      'query_tokens':len(tokens), 'seconds':time.perf_counter()-self.began,
                      'wall_seconds':time.perf_counter()-started,'error':str(exc)}
            self.inspection_failures.append(record)
            self.log.write(record)
            self.adapter.invalidate_snapshot_engine(source)
            for owner,prefix in list(self.completed):
                if owner==source:
                    del self.completed[(owner,prefix)]
            return ()
        finally:
            self.inspection_inflight[source] -= 1

    def source_inspection_idle(self, source):
        return self.inspection_inflight[source] == 0

    async def inspect_routing_targets(self, tokens, chat_id=None):
        await asyncio.gather(*(self.inspect_engine(source, tokens, chat_id=chat_id) for source in self.adapter.active_ids))

    async def observe_engine_state(self, row, source, tokens, phase='completion'):
        confirmed = await self.inspect_engine(source, tokens, chat_id=row['chat_id'],phase=phase)
        if confirmed:
            # Copy the longest currently runnable prefix of this prompt. Earlier
            # observations remain available for an arriving divergent child.
            for prefix in confirmed:
                self.completed[(source,prefix)] = row['chat_id']
            self.observe_completion(row, source, confirmed[-1])

    def observe_arrival(self, row, tokens):
        # Only causally eligible inputs may enter here. A replay child waiting
        # for its parent is not information available to the online controller.
        self.arrived[row['chat_id']] = tuple(tokens)
        if self.policy == 'cost-aware' and self.retired is not None and not self.source_closed:
            for (source, prefix), chat in list(self.completed.items()):
                if source == self.retired and len(prefix) < len(tokens) and tuple(tokens[:len(prefix)]) == prefix:
                    previous = self.tasks.get(prefix)
                    if previous and previous['task'].done() and not previous.get('record', {}).get('attempted', True):
                        del self.tasks[prefix]
                    self.schedule(source, prefix, chat)

    def observe_dispatch(self, row):
        self.arrived.pop(row['chat_id'], None)

    def observe_completion(self, row, source, tokens):
        key = tuple(tokens)
        self.completed[(source, key)] = row['chat_id']
        self.completed.move_to_end((source, key))
        while len(self.completed) > 4*self.args.snapshot_slots*len(self.args.engines):
            self.completed.popitem(last=False)
        if source == self.retired:
            self.schedule(source, key, row['chat_id'])

    def retire(self, source):
        self.retired = source
        snapshots = getattr(self.adapter.all_replicas[source], 'snapshots', {})
        candidates = [(tokens,chat) for (owner,tokens),chat in self.completed.items() if owner==source and tokens in snapshots]
        latest = {}
        for tokens,chat in candidates:
            if chat not in latest or len(tokens)>len(latest[chat]):
                latest[chat]=tokens
        for chat,tokens in latest.items():
            self.schedule(source, tokens, chat)

    def schedule(self, source, tokens, chat):
        if self.policy == 'dualmap' or self.source_closed or tokens in self.tasks:
            return
        task = asyncio.create_task(self.prepare(source, tokens, chat))
        self.tasks[tokens] = {'task': task, 'target': None, 'started': None, 'success': False}

    def queue_seconds(self, target):
        replica = self.adapter.all_replicas[target]
        pending = sum(r._actual_num_prefill_tokens for r in replica.pending_requests)
        return pending*self.args.prefill_seconds_per_token

    def copy_seconds(self, tokens):
        curve = self.curves['copy']
        # A recurrent snapshot and RPC overhead remain even for tiny prefixes.
        return curve.seconds(max(tokens, curve.points[0][0]))

    def cold_seconds(self, tokens):
        curve = self.curves['cold']
        return curve.seconds(max(tokens, curve.points[0][0]))

    def warm_seconds(self, prefix_tokens, suffix_tokens):
        warm = self.curves.get('warm')
        floor = warm.seconds(max(prefix_tokens, warm.points[0][0])) if warm else self.args.warm_ttft_seconds
        return max(floor, self.curves['cold'].seconds(suffix_tokens))

    async def prepare(self, source, tokens, chat):
        queued = time.perf_counter()
        async with self.lock:
            start = time.perf_counter()
            item = self.tasks[tokens]
            item['started'] = start
            targets = list(self.adapter.active_ids)
            target = min(targets, key=lambda i: (
                self.queue_seconds(i), self.adapter.all_replicas[i].get_num_running_req(), i))
            item['target'] = target
            matches = [(cid, ids) for cid, ids in self.arrived.items()
                       if len(ids) > len(tokens) and ids[:len(tokens)] == tokens]
            if matches:
                waiting_ids = {cid for cid, _ in matches}
                routed_matches = [request for request in self.adapter.requests.values()
                          if request._native_session_id in waiting_ids
                          and 0 <= request._primary_replica < len(self.adapter.engine_map)]
                if routed_matches:
                    # Use a currently known DualMap destination when available;
                    # this reveals neither future requests nor future routing.
                    request = max(routed_matches, key=lambda request: len(request._input_ids))
                    target = self.adapter.engine_map[request._primary_replica]
                    item['target'] = target
            busy = bool(self.adapter.all_replicas[target].pending_requests or self.adapter.all_replicas[target].running_requests
                        or self.adapter.all_replicas[source].pending_requests or self.adapter.all_replicas[source].running_requests)
            copy_seconds = self.copy_seconds(len(tokens))*(self.args.busy_copy_multiplier if busy else 1)
            item['estimate_seconds'] = copy_seconds
            replay_seconds = self.cold_seconds(len(tokens))
            record = {'chat_id': chat, 'source': source, 'target': target, 'input_tokens': len(tokens),
                      'queued_seconds': queued-self.began, 'start_seconds': start-self.began,
                      'queue_seconds': start-queued, 'known_waiting_matches': [cid for cid, _ in matches],
                      'estimated_copy_seconds': copy_seconds, 'estimated_reconstruction_seconds': replay_seconds,
                      'success': False, 'attempted': False, 'transfer_bytes': 0, 'reconstruction_tokens': 0,
                      'preparation_output_tokens': 0, 'method': 'copy'}
            if self.policy == 'selective-copy':
                # Proactive one-reuse cost proxy. Trigger, destination, order,
                # transport and dispatch match full-copy; only copy/skip differs.
                # The 64-token suffix belongs to the calibration workload, not
                # a prediction obtained from future trace requests.
                cold_proxy = self.cold_seconds(len(tokens)+64)
                warm_proxy = self.warm_seconds(len(tokens),64)
                interference = self.args.copy_interference_seconds if busy else 0
                benefit = cold_proxy-warm_proxy-copy_seconds-interference
                record.update({'selection_scope':'Optimistic one-future-reuse calibrated cost proxy; no future child or gap input.',
                               'estimated_cold_ttft_seconds':cold_proxy,'estimated_warm_ttft_seconds':warm_proxy,
                               'estimated_interference_seconds':interference,'estimated_net_gain_seconds':benefit})
                if benefit < self.args.minimum_gain_seconds:
                    record['reason']='preparation_exceeds_one_reuse_saving'
                    self.finish_record(record,item)
                    return
            if self.policy == 'cost-aware':
                # Conservative event-driven controller: speculative future reuse
                # is not assumed from a drain deadline or from future trace rows.
                if not matches:
                    record['reason'] = 'no_observed_waiting_reuse'
                    self.finish_record(record, item)
                    return
                ids = max(matches, key=lambda match: len(match[1]))[1]
                cold = PreparationEstimate('cold', 0, self.cold_seconds(len(ids)), self.queue_seconds(target))
                copy = PreparationEstimate('copy', copy_seconds,
                    self.warm_seconds(len(tokens), len(ids)-len(tokens)),
                    self.queue_seconds(target), interference_seconds=self.args.copy_interference_seconds if busy else 0)
                selected = choose_preparation([cold, copy], deadline_seconds=self.args.slo,
                                              minimum_gain_seconds=self.args.minimum_gain_seconds)
                record.update({'estimated_cold_ttft_seconds': cold.finish_seconds,
                               'estimated_copy_ttft_seconds': copy.finish_seconds,
                               'estimated_interference_seconds': copy.interference_seconds})
                if selected.method == 'cold':
                    record['reason'] = 'cold_execution_predicted_cheaper'
                    self.finish_record(record, item)
                    return
            record['attempted'] = True
            try:
                branch = uuid.uuid4().hex
                self.store.register_agent(AgentContinuation(branch, 0, str(source), 1, tokens))
                result = await self.coordinator.migrate_async(branch, str(target))
                record['transfer_bytes'] = result.bytes_transferred
                record['success'] = True
                if getattr(self.args,'inspect_prefixes',False):
                    await self.inspect_engine(target,tokens,chat_id=chat,phase='prepared')
                else:
                    self.adapter.all_replicas[target].save_token_ids(list(tokens))
            except Exception as exc:
                record['error'] = str(exc)
                record['reason'] = 'engine_rejected_or_transfer_failed'
            self.finish_record(record, item)

    def finish_record(self, record, item):
        record['finish_seconds'] = time.perf_counter()-self.began
        record['wall_seconds'] = record['finish_seconds']-record['start_seconds']
        item['success'] = record['success']
        item['record'] = record
        self.records.append(record)
        self.log.write({'event': 'preparation', **record})

    async def before_dispatch(self, row, tokens, target):
        if self.policy != 'cost-aware':
            return 0.0
        matches = [(prefix, item) for prefix, item in self.tasks.items()
                   if item['target'] == target and item['started'] is not None
                   and len(prefix) < len(tokens) and tuple(tokens[:len(prefix)]) == prefix
                   and not item['task'].done()]
        if not matches:
            return 0.0
        prefix, item = max(matches, key=lambda pair: len(pair[0]))
        residual = max(0, item.get('estimate_seconds', self.copy_seconds(len(prefix)))-(time.perf_counter()-item['started']))
        cold = PreparationEstimate('cold', 0, self.cold_seconds(len(tokens)), 0)
        copy = PreparationEstimate('copy', residual,
            self.warm_seconds(len(prefix), len(tokens)-len(prefix)), 0)
        choice = choose_preparation([cold, copy], deadline_seconds=self.args.slo,
                                    minimum_gain_seconds=self.args.minimum_gain_seconds)
        if choice.method == 'cold':
            return 0.0
        started = time.perf_counter()
        # Waiting is charged from eligibility and bounded by the predicted cold
        # alternative. Shield avoids cancelling a launched collective.
        try:
            await asyncio.wait_for(asyncio.shield(item['task']), timeout=max(0.001, cold.finish_seconds-copy.service_seconds))
        except asyncio.TimeoutError:
            pass
        waited = time.perf_counter()-started
        self.log.write({'event': 'request_waited_for_preparation', 'chat_id': row['chat_id'],
                        'seconds': waited, 'prepared': item['success'], 'target': target})
        return waited

    async def finish(self):
        outcomes = []
        if self.tasks:
            outcomes = await asyncio.gather(*(item['task'] for item in self.tasks.values()), return_exceptions=True)
        self.assert_transfers_clean()
        failures = [repr(value) for value in outcomes if isinstance(value,BaseException)]
        if failures:
            raise RuntimeError('preparation tasks failed after all jobs drained: '+'; '.join(failures))

    def assert_transfers_clean(self):
        pending = {engine: sorted(client.live_transfers) for engine, client in self.coordinator.engines.items() if client.live_transfers}
        if pending:
            raise RuntimeError(f'transfer cleanup unconfirmed; forbid pool flush or engine shutdown: {pending}')

    def archive(self):
        # Back up committed SQLite state after the measured interval. FULL/WAL
        # durability stays enabled during service on local ext4; no NFS fsync is
        # hidden inside a supposed GPU communication curve.
        with sqlite3.connect(self.output/'ownership.db') as target:
            self.store.conn.backup(target)
            if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise RuntimeError('ownership archive integrity check failed')
        return {'runtime_path': str(self.metadata_path), 'archived_path': str(self.output/'ownership.db'),
                'journal_mode': 'WAL', 'synchronous': 'FULL',
                'scope': 'Local durable control state; archive copying occurs after observation.'}


async def run_one(args, rows, manifest, policy, repetition, output, curves):
    output.mkdir(parents=True, exist_ok=False)
    log = JsonLog(output/'events.jsonl')
    request_log = JsonLog(output/'requests.jsonl')
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    async with aiohttp.ClientSession(timeout=timeout, connector=aiohttp.TCPConnector(limit=0)) as session:
        await flush(session, args.engines)
        adapter = DualMapAdapter(args.dualmap_source, len(args.engines), args.cache_tokens,
            20480, args.slo, args.prefill_seconds_per_token, mode='dualmap', hybrid_snapshot_slots=args.snapshot_slots,
            inspect_snapshots=getattr(args,'inspect_prefixes',False))
        setup_started = time.perf_counter()
        began = setup_started
        trace_origin = manifest.get('replay_start_timestamp', min(row['timestamp'] for row in rows))
        measure_offset = args.start+args.trace_origin-trace_origin
        measure_start = measure_offset/args.time_scale
        observation_end = (measure_offset+args.duration)/args.time_scale
        manager = PreparationManager(args, adapter, session, policy, output, curves, log, began)
        if policy != 'dualmap' and args.retire_offset is not None:
            for target in range(len(args.engines)):
                if target != args.retire_engine:
                    await manager.coordinator._ensure_group(str(args.retire_engine), str(target))
        setup_seconds = time.perf_counter()-setup_started
        began = time.perf_counter()
        manager.began = began
        source_rows = {i: row for i, row in enumerate(rows)}
        finished = {r['chat_id']: asyncio.Event() for r in rows}
        eligibility, records, tasks = {}, {}, set()
        event = asyncio.Event()
        retire_time = None if args.retire_offset is None else measure_start+args.retire_offset/args.time_scale
        retired, released, lifecycle_identity = None, None, None
        release_task = None
        if args.physical_release:
            from experiments.fluidkv.engine_lifecycle import capture_owned_engine
            lifecycle_identity = capture_owned_engine(args.engines[args.retire_engine])
        hard_deadline = began+args.run_timeout
        timed_out = False

        async def produce(ordinal, row):
            arrival = began+(row['timestamp']-trace_origin)/args.time_scale
            await asyncio.sleep(max(0, arrival-time.perf_counter()))
            tokens = surrogate_tokens(row)
            if row['parent_chat_id'] in finished:
                await finished[row['parent_chat_id']].wait()
            eligibility[ordinal] = time.perf_counter()
            if getattr(args,'inspect_prefixes',False):
                await manager.inspect_routing_targets(tokens,row['chat_id'])
            manager.observe_arrival(row, tokens)
            await adapter.enqueue(ordinal, row, tokens, row['output_length'])
            event.set()

        async def serve(request):
            row = source_rows[request._id]
            target = adapter.physical_target(request)
            eligible = eligibility[request._id]
            scheduled = began+(row['timestamp']-trace_origin)/args.time_scale
            prepare_wait = await manager.before_dispatch(row, request._input_ids, target)
            manager.observe_dispatch(row)
            dispatched = time.perf_counter()
            expected = len(request._input_ids)-adapter.all_replicas[target].get_num_recompute_token_ids(request._input_ids)
            result, error = None, None
            async def first(payload, delay):
                await adapter.first_token(request, delay)
                if getattr(args, 'state_observation', 'prefill') == 'prefill':
                    # Calibrated exact input-prompt snapshot only. The evolving
                    # generation still runs on its original engine.
                    if getattr(args,'inspect_prefixes',False):
                        await manager.observe_engine_state(row,target,request._input_ids,phase='prefill')
                    else:
                        manager.observe_completion(row, target, request._input_ids)
                event.set()
            try:
                result = await timed_stream(session, args.engines[target], request._input_ids, request._output_len, first)
                meta = result['meta_info']
                if meta['prompt_tokens'] != row['input_length'] or meta['completion_tokens'] != row['output_length']:
                    raise ValueError('engine changed input/output length')
            except Exception as exc:
                error = str(exc)
            await adapter.finished(request, result['output_ids'] if result else [], error is None)
            done = time.perf_counter()
            record = {'policy': policy, 'repetition': repetition, 'chat_id': row['chat_id'],
                'parent_chat_id': row['parent_chat_id'], 'phase': 'measurement' if scheduled-began >= measure_start-1e-6 else 'warmup',
                'trace_arrival_seconds': row['timestamp'], 'scheduled_seconds': scheduled-began,
                'eligibility_seconds': eligible-began, 'dispatched_seconds': dispatched-began,
                'finished_seconds': done-began, 'eligibility_delay_seconds': eligible-scheduled,
                'router_wait_seconds': dispatched-eligible, 'preparation_wait_seconds': prepare_wait,
                'target': target, 'input_tokens': row['input_length'], 'requested_output_tokens': row['output_length'],
                'prompt_token_sha256':hashlib.sha256(json.dumps(request._input_ids).encode()).hexdigest(),
                'estimated_cached_tokens': expected, 'success': error is None, 'error': error,
                'actual_cached_tokens': result['meta_info']['cached_tokens'] if result else 0,
                'completion_tokens': result['meta_info']['completion_tokens'] if result else 0,
                'ttft_seconds': result['first_monotonic']-eligible if result and error is None else None,
                'arrival_ttft_seconds': result['first_monotonic']-scheduled if result and error is None else None,
                'first_token_seconds': result['first_monotonic']-began if result else None,
                'e2e_seconds': done-eligible if error is None else None,
                'arrival_e2e_seconds': done-scheduled if error is None else None,
                'output_digest': hashlib.sha256(json.dumps(result['output_ids']).encode()).hexdigest() if result else None,
                'stream_chunk_count': result['stream_chunk_count'] if result else None,
                'inter_chunk_p50_p90_p99_seconds': result['inter_chunk_p50_p90_p99_seconds'] if result else None}
            records[row['chat_id']] = record
            request_log.write(record)
            # The parent has completed serving; metadata refresh must not add
            # an artificial causal dependency to a child's arrival.
            finished[row['chat_id']].set()
            if error is None:
                if getattr(args,'inspect_prefixes',False):
                    await manager.observe_engine_state(row,target,request._input_ids)
                else:
                    manager.observe_completion(row, target, request._input_ids)
            event.set()

        producers = [asyncio.create_task(produce(i, row)) for i, row in source_rows.items()]
        last_telemetry = -10
        fatal_error = None
        try:
            while (any(not p.done() for p in producers) or tasks or adapter.queued()
                   or time.perf_counter()-began < observation_end
                   or (release_task is not None and not release_task.done())):
                event.clear()
                now = time.perf_counter()
                if now > hard_deadline:
                    timed_out = True
                    break
                if retired is None and retire_time is not None and now-began >= retire_time:
                    retired = await adapter.retire_engine(args.retire_engine)
                    manager.retire(retired)
                    log.write({'event': 'logical_retirement', 'engine': retired, 'seconds': now-began})
                for request in await adapter.dispatch():
                    task = asyncio.create_task(serve(request))
                    tasks.add(task)
                    task.add_done_callback(lambda done: event.set())
                completed_tasks = {task for task in tasks if task.done()}
                for task in completed_tasks:
                    task.result()
                tasks.difference_update(completed_tasks)
                for task in producers:
                    if task.done() and not task.cancelled() and task.exception():
                        raise task.exception()
                if now-began-last_telemetry >= 5:
                    log.write({'event': 'load', 'seconds': now-began, 'completed': len(records),
                        'active_engines': adapter.active_ids, 'engines': {str(i): {
                            'pending': len(r.pending_requests), 'running': len(r.running_requests)} for i, r in adapter.all_replicas.items()},
                        'router_queued': adapter.queued()})
                    async def engine_load(engine):
                        try:
                            async with session.get(args.engines[engine]+'/v1/loads?include=all', timeout=aiohttp.ClientTimeout(total=2)) as response:
                                response.raise_for_status()
                                return {'engine': engine, 'response': await response.json()}
                        except Exception as exc:
                            return {'engine': engine, 'error': str(exc)}
                    live = [i for i in range(len(args.engines)) if not manager.source_closed or i != retired]
                    if getattr(args, 'engine_telemetry', False):
                        actual_loads = await asyncio.gather(*(engine_load(i) for i in live))
                        log.write({'event': 'engine_load', 'seconds': time.perf_counter()-began,
                                   'measurements': actual_loads,
                                   'capacity_note': 'Available-token fields can include evictable cache; not unreserved transfer capacity.'})
                    last_telemetry = now-began
                if release_task is not None and release_task.done() and released is None:
                    released = release_task.result()
                    if not released.get('released'):
                        log.write({'event': 'physical_release_failed', **released})
                        raise RuntimeError('physical GPU release was not confirmed')
                    released['confirmed_seconds'] = released['confirmed_monotonic']-began
                    log.write({'event': 'physical_release', **released})
                if args.physical_release and retired is not None and release_task is None:
                    replica = adapter.all_replicas[retired]
                    inspection_idle = manager.source_inspection_idle(retired) if hasattr(manager,'source_inspection_idle') else True
                    if not replica.pending_requests and not replica.running_requests and inspection_idle and all(item['task'].done() for item in manager.tasks.values()):
                        from experiments.fluidkv.engine_lifecycle import release_engine
                        manager.source_closed = True
                        manager.assert_transfers_clean()
                        release_task = asyncio.create_task(release_engine(lifecycle_identity, no_foreground=True, no_transfers=True))
                        log.write({'event': 'physical_release_started', 'engine': retired, 'seconds': time.perf_counter()-began})
                try:
                    await asyncio.wait_for(event.wait(), timeout=0.02)
                except asyncio.TimeoutError:
                    pass
        except Exception as exc:
            fatal_error = repr(exc)
            log.write({'event':'run_aborted','seconds':time.perf_counter()-began,'error':fatal_error})
        if timed_out or fatal_error is not None:
            # Do not cancel launched requests or NCCL jobs and immediately flush
            # underneath them. Stop new producers, drain admitted requests, and
            # retain every noncompleted offered request as a failure.
            for task in producers:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*producers, return_exceptions=True)
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.gather(*producers)
        try:
            await manager.finish()
        except Exception as exc:
            fatal_error = fatal_error or repr(exc)
            log.write({'event':'transfer_drain_unconfirmed','seconds':time.perf_counter()-began,'error':str(exc)})
        if release_task is not None and released is None:
            try:
                released = await release_task
            except Exception as exc:
                released = {'released':False,'error':repr(exc)}
            if not released.get('released'):
                log.write({'event': 'physical_release_failed', **released})
                fatal_error = fatal_error or 'physical GPU release was not confirmed'
            else:
                released['confirmed_seconds'] = released['confirmed_monotonic']-began
                log.write({'event': 'physical_release', **released})
        if args.physical_release and retired is not None and released is None and fatal_error is None:
            from experiments.fluidkv.engine_lifecycle import release_engine
            replica = adapter.all_replicas[retired]
            if replica.pending_requests or replica.running_requests:
                fatal_error = 'source still has admitted requests; cannot release'
            else:
                manager.source_closed = True
                try:
                    released = await release_engine(lifecycle_identity, no_foreground=True, no_transfers=True)
                except Exception as exc:
                    released = {'released':False,'error':repr(exc)}
                if not released.get('released'):
                    log.write({'event': 'physical_release_failed', **released})
                    fatal_error = 'physical GPU release was not confirmed'
                else:
                    released['confirmed_seconds'] = released['confirmed_monotonic']-began
                    log.write({'event': 'physical_release', **released})
        elapsed = time.perf_counter()-began
        for ordinal, row in source_rows.items():
            if row['chat_id'] in records:
                continue
            scheduled = (row['timestamp']-trace_origin)/args.time_scale
            record = {'policy': policy, 'repetition': repetition, 'chat_id': row['chat_id'],
                'parent_chat_id': row['parent_chat_id'], 'phase': 'measurement' if scheduled >= measure_start-1e-6 else 'warmup',
                'trace_arrival_seconds': row['timestamp'], 'scheduled_seconds': scheduled,
                'input_tokens': row['input_length'], 'requested_output_tokens': row['output_length'],
                'completion_tokens': 0, 'actual_cached_tokens': 0, 'target': None,
                'success': False, 'error': 'run_aborted_before_completion' if fatal_error else 'run_timeout_before_completion', 'ttft_seconds': None, 'e2e_seconds': None}
            records[row['chat_id']] = record
            request_log.write(record)
        observation = max(observation_end, elapsed)-measure_start
        all_records = list(records.values())
        released_seconds = max(0,elapsed-released['confirmed_seconds']) if released and released.get('released') else 0
        result = {'policy': policy, 'repetition': repetition, 'timed_out': timed_out, 'fatal_error':fatal_error,
            'duration_seconds': elapsed, 'group_setup_seconds': setup_seconds, 'measurement_start_seconds': measure_start,
            'measurement_observation_seconds': observation,
            'offered_measurement_span_seconds': args.duration/args.time_scale,
            'retired_engine': retired, 'physical_release': released,
            'model_occupied_gpu_seconds': len(args.engines)*elapsed-released_seconds,
            'model_occupied_gpu_seconds_including_group_setup': len(args.engines)*(elapsed+setup_seconds)-released_seconds,
            'gpu_accounting_scope': 'Model-resident TP1 engine occupancy; no cloud-billing or standby-free assumption.',
            'measurement': summarize(all_records, phase='measurement', slo_seconds=args.slo, observation_seconds=observation),
            'warmup': summarize(all_records, phase='warmup', slo_seconds=args.slo, observation_seconds=max(measure_start, 1e-9)),
            'upstream_rebalanced_requests': adapter.scheduler.rebalance_cnt,
            'preparation_attempts': sum(r['attempted'] for r in manager.records),
            'preparation_successes': sum(r['success'] for r in manager.records),
            'preparation_transfer_bytes': sum(r['transfer_bytes'] for r in manager.records),
            'prefix_inspection_failures': getattr(manager,'inspection_failures',[]),
            'preparations': manager.records,
            'limitations': ['Anonymous token surrogates; cold/cached semantic equivalence unresolved.',
                'Router snapshot metadata estimates availability; engine reservations are authoritative.',
                'Cost-aware is a conservative observed-demand controller, not a predictive global optimizer.',
                'Inter-chunk intervals are not asserted to be exact inter-token latency.']}
        if hasattr(manager, 'archive'):
            result['metadata_storage'] = manager.archive()
        (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
        return result


async def main_async(args):
    args.output.mkdir(parents=True, exist_ok=False)
    original = [json.loads(line) for line in args.trace.open()]
    args.trace_origin = min(row['timestamp'] for row in original)
    rows, manifest = select_continuous_window(original, start_seconds=args.start,
        duration_seconds=args.duration, warmup_seconds=args.warmup, context_length=args.context_length)
    if not rows:
        raise ValueError('empty continuous window')
    (args.output/'selection.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (args.output/'arguments.json').write_text(json.dumps({k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}, indent=2)+'\n')
    calibration = json.loads(args.costs.read_text())
    if 'lengths' in calibration:
        # Calibration cold children extend each measured parent by 64 tokens.
        calibration = {
            'cold': [(int(n)+64, item['cold_ttft_seconds']) for n,item in calibration['lengths'].items()],
            'copy': [(int(n), item['copy_idle_seconds']) for n,item in calibration['lengths'].items()],
            'warm': [(int(n), item['warm_ttft_seconds']) for n,item in calibration['lengths'].items()]}
    (args.output/'cost_curves.json').write_text(json.dumps(calibration, indent=2)+'\n')
    curves = {key: CostCurve(tuple((int(n), float(t)) for n,t in calibration[key])) for key in ('cold','copy','warm') if key in calibration}
    async with aiohttp.ClientSession() as session:
        infos = []
        for url in args.engines:
            async with session.get(url+'/get_server_info') as response:
                response.raise_for_status()
                infos.append(await response.json())
        (args.output/'server_info.json').write_text(json.dumps(infos, indent=2)+'\n')
        if len({info['model_path'] for info in infos}) != 1:
            raise ValueError('comparison engines must use the same model')
        if any(info['max_total_tokens'] != args.cache_tokens for info in infos):
            raise ValueError('router cache capacity must match every engine token pool')
        if any(args.context_length > info['context_length']-2 for info in infos):
            raise ValueError('runtime requires a two-token context reserve to preserve requested output length')
    rng = random.Random(args.seed)
    results = []
    base_port = args.transfer_port
    index = 0
    for repetition in range(args.repetition_start, args.repetition_start+args.repetitions):
        for policy in rng.sample(args.policies, len(args.policies)):
            args.transfer_port = base_port+index*3
            if args.physical_release and index:
                raise ValueError('physical-release runner must restart the released engine between invocations')
            result = await run_one(args, rows, manifest, policy, repetition,
                                   args.output/f'{repetition:02d}-{policy}', curves)
            results.append(result)
            (args.output/'summary.json').write_text(json.dumps({'runs': results}, indent=2)+'\n')
            print(json.dumps({'policy': policy, 'repetition': repetition, 'measurement': result['measurement'],
                              'preparation_attempts': result['preparation_attempts']}), flush=True)
            index += 1
            if result['timed_out'] or result.get('fatal_error'):
                raise RuntimeError('run incomplete or invalid; all offered requests retained; '+str(result.get('fatal_error') or 'run timeout'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--dualmap-source', type=Path, default=Path('/tmp/fluidkv-dualmap-review-20260914'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--costs', type=Path, required=True)
    p.add_argument('--engines', nargs='+', default=['http://127.0.0.1:31480','http://127.0.0.1:31481','http://127.0.0.1:31482'])
    p.add_argument('--policies', nargs='+', choices=['dualmap','full-copy','cost-aware','selective-copy'], default=['dualmap','full-copy','cost-aware'])
    p.add_argument('--start', type=float, default=600)
    p.add_argument('--duration', type=float, default=60)
    p.add_argument('--warmup', type=float, default=60)
    p.add_argument('--time-scale', type=float, default=1)
    p.add_argument('--retire-offset', type=float)
    p.add_argument('--retire-engine', type=int, default=0)
    p.add_argument('--physical-release', action='store_true')
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--repetition-start', type=int, default=0)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--slo', type=float, default=1)
    p.add_argument('--prefill-seconds-per-token', type=float, default=0.000025)
    p.add_argument('--warm-ttft-seconds', type=float, default=0.1)
    p.add_argument('--copy-interference-seconds', type=float, default=0.0)
    p.add_argument('--busy-copy-multiplier', type=float, default=1.0)
    p.add_argument('--minimum-gain-seconds', type=float, default=0.02)
    p.add_argument('--cache-tokens', type=int, default=65536)
    p.add_argument('--context-length', type=int, default=32766,
                   help='Maximum input plus requested output; must be below engine context limit.')
    p.add_argument('--snapshot-slots', type=int, default=48)
    p.add_argument('--state-observation', choices=['prefill','completion'], default='prefill')
    p.add_argument('--transfer-port', type=int, default=32000)
    p.add_argument('--request-timeout', type=float, default=900)
    p.add_argument('--engine-telemetry', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--inspect-prefixes', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--metadata-directory', type=Path, default=Path('/tmp/fluidkv-metadata-20260914'))
    p.add_argument('--run-timeout', type=float, default=1800)
    args = p.parse_args()
    if args.time_scale <= 0 or args.duration <= 0 or args.warmup < 0:
        p.error('positive time scale/duration and nonnegative warmup required')
    if args.physical_release and (args.retire_offset is None or len(args.policies)*args.repetitions != 1):
        p.error('physical release requires one policy/repetition and an explicit retirement event')
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
