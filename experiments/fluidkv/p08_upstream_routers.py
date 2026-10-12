"""Pinned policy execution on a common SGLang transport and observation adapter.

This compares original routing algorithms, not their native serving stacks.
Blitz executes its original Rust policy and admission logic. DualMap executes
its original queue, dispatch, rebalancing and request lifecycle code.
"""
import asyncio
import contextlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.fluidkv import p07_routing_space as pilot
from experiments.fluidkv.dualmap_adapter import DualMapAdapter


BLITZ_BINARY = Path('/tmp/fluidkv-blitz-upstream-20261010/target/debug/fluidkv-policy-bridge')


class UpstreamRouter(pilot.Router):
    def __init__(self, policy, calibration, initial, telemetry, lookahead=128, pending_prefills=1,
                 publish_prefill_prefix=0, blitz_binary=BLITZ_BINARY):
        super().__init__(policy, calibration, initial, telemetry, lookahead)
        self.serial = 0
        self.native = {}
        self.waiters = {}
        self.closed = False
        self.pump = None
        self.bridge = None
        self.publish_prefill_prefix = publish_prefill_prefix
        self.prompt_ids = {}
        self.workers=len(initial)
        self.preparation_ids=set()
        if policy == 'blitz':
            self.bridge = subprocess.Popen([str(blitz_binary)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, text=True, bufsize=1)
        elif policy == 'dualmap':
            points = calibration['prefill_points']
            slope = float(np.median([seconds / n for n, seconds in points]))
            self.dualmap = DualMapAdapter(Path('/home/zhujianian/DualMap'), self.workers,
                262144, 20480, 10., slope, pending_prefills=pending_prefills)
            for worker, prefixes in enumerate(initial):
                for prefix in prefixes:
                    self.dualmap.replica_for_engine(worker).save_token_ids(prefix.tolist())
                # Both algorithms consume the same backend cache estimate.
                # DualMap's queueing and selection logic remain upstream code.
                self.dualmap.replica_for_engine(worker).get_num_recompute_token_ids = (
                    lambda ids, w=worker: len(ids)-self.hit(w, np.asarray(ids,dtype=np.int32)))
            self.pump = asyncio.create_task(self.dispatch_loop())
        else:
            raise ValueError(policy)

    def hit(self, worker, ids):
        # The native Blitz bridge indexes 16-token blocks. A completed input
        # can expose an executable recurrent boundary between 512-token marks;
        # rounding it to 512 doubles a 64-token continuation's estimated work.
        return max((pilot.lcp(ids,prefix)//16*16 for prefix in self.cache[worker]),default=0)

    def book(self, target, rid, ids, decision):
        hits = [self.hit(i, ids) for i in range(self.workers)]
        self.active[target][rid] = dict(context=len(ids), predicted_hit=hits[target],
            initial_new=len(ids)-hits[target], first=False)
        self.pending[target] += len(ids)-hits[target]
        self.decisions.append(decision)
        if self.publish_prefill_prefix:
            self.prompt_ids[rid] = np.asarray(ids,dtype=np.int32)

    async def dispatch_loop(self):
        try:
            while not self.closed:
                for request in await self.dualmap.dispatch():
                    future = self.waiters.pop(request._id)
                    future.set_result(request)
                await asyncio.sleep(.005)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            for future in self.waiters.values():
                if not future.done():
                    future.set_exception(exc)
            raise

    async def choose(self, rid, ids, output, metadata):
        started = time.perf_counter()
        ordinal = self.serial
        self.serial += 1
        if self.policy == 'blitz':
            observations = [dict(bs=len(self.active[i]),
                waiting=sum(not r['first'] for r in self.active[i].values()),
                queued_tokens=int(round(self.pending[i])),
                all_tokens=sum(r['context'] for r in self.active[i].values()),
                hit_tokens=self.hit(i, ids)) for i in range(self.workers)]
            payload = dict(request_id=ordinal, tokens=ids.tolist(),
                max_new_tokens=output, observations=observations)
            self.bridge.stdin.write(json.dumps(payload)+'\n')
            self.bridge.stdin.flush()
            raw = self.bridge.stdout.readline()
            if not raw:
                raise RuntimeError('upstream Blitz bridge exited')
            result = json.loads(raw)
            target = result['target']
            if target not in range(self.workers):
                raise RuntimeError(result)
            decision = dict(result, observations=observations,
                router_seconds=time.perf_counter()-started)
            self.book(target, rid, ids, decision)
            # The original Rust admission update is the source of this counter.
            self.pending[target] = result['after_admission']['queued_tokens']
        else:
            if self.pump.done():
                await self.pump
            self.waiters[ordinal] = asyncio.get_running_loop().create_future()
            before = [self.dualmap.replica_for_engine(i).get_num_actual_pending_tokens()
                for i in range(self.workers)]
            await self.dualmap.enqueue(ordinal, metadata, ids.tolist(), output)
            request = await asyncio.wait_for(self.waiters[ordinal], timeout=180)
            self.native[rid] = request
            target = self.dualmap.physical_target(request)
            decision = dict(request_id=ordinal, target=target,
                primary=request._primary_replica, predicted_new=request._actual_num_prefill_tokens,
                before_pending=before, router_seconds=time.perf_counter()-started,
                implementation='upstream DualMap queues and dispatch')
            self.book(target, rid, ids, decision)
        return target, decision

    async def progress(self, target, rid, payload, count, old_count):
        first = not self.active[target][rid]['first']
        super().progress(target, rid, payload, count, old_count)
        if first and self.publish_prefill_prefix and rid not in self.preparation_ids:
            ids=self.prompt_ids[rid]
            # The p09 gate verifies that this exact family prefix is already
            # indexed while the cold producer is still decoding.
            prefix=ids[:self.publish_prefill_prefix]
            if self.hit(target,prefix)<len(prefix):
                self.cache[target].append(prefix.copy())
        if self.policy == 'dualmap' and first:
            request = self.native[rid]
            await self.dualmap.first_token(request, time.perf_counter()-request._arrived_at)

    async def admit_preparation(self, ids, target=1, request_id='prepare'):
        await super().admit_preparation(ids,target,request_id)
        self.preparation_ids.add(request_id)
        if self.publish_prefill_prefix:
            self.prompt_ids[request_id] = np.asarray(ids,dtype=np.int32)
        if self.policy == 'dualmap':
            # Fixed-target preparation is real prefill work, visible to the
            # upstream budget and pending-request limit until its first token.
            ordinal = self.serial
            self.serial += 1
            cls = self.dualmap.Request
            request = cls(request_id=ordinal, dataset_type='qwen-bailian-a',
                native_session_id='preparation', session_id='preparation',
                hash_session_id='preparation', round_id=0, prompts='',
                input_ids=list(ids), num_prefill_tokens=len(ids),
                actual_num_prefill_tokens=len(ids), output_len=8, over_flow=False,
                n=1, temperature=0, top_p=1, max_tokens=8, stream=True,
                arrived_at=time.perf_counter(), time_interval=0, hash_prefix_len=0)
            request._physical_engine = request._primary_replica = target
            await self.dualmap.replica_for_engine(target).add_request(request)
            self.native[request_id] = request

    async def completed(self, target, rid, ids, result):
        if self.policy == 'dualmap':
            await self.dualmap.finished(self.native.pop(rid), result.get('output_ids') or [],
                result['error'] is None)
        await super().completed(target, rid, ids, result)
        self.prompt_ids.pop(rid,None)
        self.preparation_ids.discard(rid)

    async def close(self):
        self.closed = True
        if self.pump:
            self.pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.pump
        if self.bridge:
            self.bridge.stdin.close()
            try:
                self.bridge.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.bridge.terminate()
                self.bridge.wait(timeout=3)
