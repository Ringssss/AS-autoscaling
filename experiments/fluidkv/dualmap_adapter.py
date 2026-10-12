"""Pinned upstream DualMap scheduling with an external serving transport."""

import logging
import subprocess
import sys
import time
from types import SimpleNamespace
from collections import OrderedDict

from agentshift.controller.snapshot_directory import SnapshotDirectory


UPSTREAM_COMMIT = "24816acc70b8e8f4f6b47bc47b7ce0fdddddf40d"


class DualMapAdapter:
    def __init__(self, source, replicas, cache_tokens, bytes_per_token, slo, prefill_seconds_per_token, mode="dualmap", hybrid_snapshot_slots=0, inspect_snapshots=False, pending_prefills=1):
        if type(pending_prefills) is not int or pending_prefills < 1:
            raise ValueError("pending_prefills must be a positive integer")
        commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if commit != UPSTREAM_COMMIT:
            raise ValueError(f"unreviewed DualMap commit {commit}")
        if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
            raise ValueError("DualMap source has local changes")
        sys.path.insert(0, str(source))
        from dualmap.entities.replica import Replica
        from dualmap.entities.request import Request
        from dualmap.scheduler.utils.double_hash_global_scheduler_utils import DoubleHashGlobalSchedulerUtils
        from dualmap.scheduler.utils.lazy_prefix_table import LazyPrefixTable, HotPrefixDetector, LazyExpansionController

        self.Request = Request
        self.mode = mode
        self.prefix_table = LazyPrefixTable()
        self.prefix_controller = LazyExpansionController(self.prefix_table, HotPrefixDetector())
        budget = max(1, int(slo / prefill_seconds_per_token))
        self.args = SimpleNamespace(
            balance_type="dualmap", cache_capacity=cache_tokens * bytes_per_token,
            block_size=16, kv_cache_size_per_token=bytes_per_token,
            replica_slo_budget=budget, dh_first_balance_ttft_thredhold=budget // 2,
            dh_rebalance_thredhold=budget, dh_replica_pending_req_threshold=pending_prefills,
            dh_rebalance_waiting_latency_thredhold=3, prefill_tpot=prefill_seconds_per_token,
            dh_recompute_punish_ratio=1, busy_prefill_interval=2,
        )
        self.hybrid_snapshot_slots = hybrid_snapshot_slots
        self.inspect_snapshots = inspect_snapshots
        self.snapshot_directory = SnapshotDirectory(max_snapshots_per_engine=max(1,hybrid_snapshot_slots),
            max_tokens_per_engine=cache_tokens) if inspect_snapshots else None
        if hybrid_snapshot_slots:
            class HybridReplica(Replica):
                def __init__(self, *arguments):
                    super().__init__(*arguments)
                    self.snapshots = OrderedDict()
                    self._snapshot_prefixes_by_length = {}

                def rebuild_snapshot_index(self):
                    self._snapshot_prefixes_by_length = {}
                    for prefix in self.snapshots:
                        self._snapshot_prefixes_by_length.setdefault(len(prefix), set()).add(prefix)

                def get_num_recompute_token_ids(self, ids):
                    # A matching KV block without a recurrent snapshot is not
                    # an executable hybrid prefix. This is backend metadata,
                    # shared by all policies; upstream route decisions remain.
                    # Check each boundary length once. Scanning every snapshot
                    # repeatedly copied the entire long input in high-BS runs;
                    # this index preserves exact token equality and LRU policy.
                    for length in sorted(self._snapshot_prefixes_by_length, reverse=True):
                        if length < len(ids) and tuple(ids[:length]) in self._snapshot_prefixes_by_length[length]:
                            return len(ids) - length
                    return len(ids)

                def remember(self, ids):
                    if inspect_snapshots:
                        return
                    key = tuple(ids)
                    if not key:
                        return
                    self.snapshots[key] = None
                    self._snapshot_prefixes_by_length.setdefault(len(key), set()).add(key)
                    self.snapshots.move_to_end(key)
                    while len(self.snapshots) > hybrid_snapshot_slots:
                        removed, _ = self.snapshots.popitem(last=False)
                        group = self._snapshot_prefixes_by_length[len(removed)]
                        group.remove(removed)
                        if not group:
                            del self._snapshot_prefixes_by_length[len(removed)]

                def save_token_ids(self, ids):
                    self.remember(ids)
                    return super().save_token_ids(ids)

                def save_prefill_token_ids(self, request_id, ids):
                    self.remember(ids)
                    return super().save_prefill_token_ids(request_id, ids)

            replica_class = HybridReplica
        else:
            replica_class = Replica
        self.replica_budgets = {i: replica_class(i, None, self.args) for i in range(replicas)}
        self.all_replicas = dict(self.replica_budgets)
        self.engine_map = list(range(replicas))
        self.retained_waiting = []
        self.scheduler = DoubleHashGlobalSchedulerUtils(replicas, self, self.args)
        self.requests = {}
        self.active_ids = list(range(replicas))
        self.chat_owners = {}
        for name in logging.root.manager.loggerDict:
            if name.startswith("dualmap"):
                logging.getLogger(name).setLevel(logging.WARNING)

    def observe_snapshot_response(self, physical_engine, token_ids, response):
        if self.snapshot_directory is None:
            raise RuntimeError('actual snapshot inspection is not enabled')
        confirmed = self.snapshot_directory.update(physical_engine, token_ids, response)
        self.all_replicas[physical_engine].snapshots = OrderedDict(
            (prefix,None) for prefix in self.snapshot_directory.prefixes(physical_engine))
        rebuild = getattr(self.all_replicas[physical_engine], "rebuild_snapshot_index", None)
        if rebuild is not None:
            rebuild()
        return confirmed

    def invalidate_snapshot_engine(self, physical_engine):
        if self.snapshot_directory is not None:
            self.snapshot_directory.discard_engine(physical_engine)
            self.all_replicas[physical_engine].snapshots.clear()
            rebuild = getattr(self.all_replicas[physical_engine], "rebuild_snapshot_index", None)
            if rebuild is not None:
                rebuild()

    def dump_replica_queue_info(self, replica):
        # Upstream invokes this only to emit diagnostic logging.
        return None

    def get_num_actual_pending_tokens_replica(self, replica):
        return self.replica_budgets[replica].get_num_actual_pending_tokens()

    def get_pending_input_tokens_replica(self, replica):
        return sum(len(r._input_ids) for r in self.replica_budgets[replica].pending_requests)

    async def enqueue(self, ordinal, row, token_ids, output_length):
        depth = self.prefix_table.lookup(row["hash_ids"])
        self.prefix_controller.process(row["hash_ids"])
        key = "".join(map(str, row["hash_ids"][:depth]))
        request = self.Request(
            request_id=ordinal, dataset_type="qwen-bailian-a", native_session_id=row["chat_id"],
            session_id=str(row["chat_id"]), hash_session_id="qwen@" + key,
            round_id=row.get("turn", 1), prompts="", input_ids=token_ids,
            num_prefill_tokens=len(token_ids), actual_num_prefill_tokens=len(token_ids),
            output_len=output_length, over_flow=False, n=1, temperature=0, top_p=1,
            max_tokens=output_length, stream=True, arrived_at=time.perf_counter(),
            time_interval=0, hash_prefix_len=depth,
        )
        self.requests[ordinal] = request
        if self.mode == "dualmap":
            await self.scheduler.add_request_to_best_global_queue(request)
        else:
            estimates = {i: replica.get_num_recompute_token_ids(token_ids) for i, replica in self.replica_budgets.items()}
            if self.mode == "sticky" and row.get("parent_chat_id") in self.chat_owners:
                physical = self.chat_owners[row["parent_chat_id"]]
                if physical not in self.engine_map:
                    request._physical_engine = physical
                    self.retained_waiting.append(request)
                    return
                target = self.engine_map.index(physical)
            elif self.mode in ("cache-affinity", "sticky"):
                target = min(estimates, key=lambda i: (estimates[i], self.get_num_actual_pending_tokens_replica(i), i))
            elif self.mode == "least-loaded":
                target = min(estimates, key=lambda i: (
                    self.get_pending_input_tokens_replica(i) + self.scheduler.global_request_queue.get_global_input_waiting_tokens_count(i),
                    self.replica_budgets[i].get_num_running_req(), i,
                ))
            else:
                raise ValueError(self.mode)
            request._primary_replica = target
            self.scheduler.global_request_queue.push(target, request, len(token_ids) - estimates[target])

    async def retire_last(self):
        return await self.retire_engine(self.active_ids[-1])

    async def retire_engine(self, physical_id):
        if len(self.active_ids) <= 1:
            raise ValueError("cannot retire the final instance")
        if physical_id not in self.active_ids:
            raise ValueError("instance is not active")
        from dualmap.scheduler.utils.double_hash_global_scheduler_utils import GlobalRequestQueue
        waiting = []
        for slot, engine in enumerate(self.engine_map):
            waiting.extend((engine, request) for request in self.scheduler.global_request_queue.get_all_requests(slot))
        self.active_ids.remove(physical_id)
        self.engine_map = list(self.active_ids)
        self.replica_budgets = {slot: self.all_replicas[engine] for slot, engine in enumerate(self.engine_map)}
        self.scheduler._num_replicas = len(self.engine_map)
        self.scheduler.global_request_queue = GlobalRequestQueue(len(self.engine_map))
        # Keep ring anchors tied to physical identities, even though upstream
        # queue arrays use compact slots. No inference request changes engine.
        import uhashring
        ring1 = uhashring.HashRing(nodes=[str(engine) for engine in self.engine_map])
        ring2 = uhashring.HashRing(nodes=[str(engine) for engine in self.engine_map], hash_fn="ketama")
        self.scheduler.hash_function1 = lambda key, count: self.engine_map.index(int(ring1.get_node(str(key))))
        self.scheduler.hash_function2 = lambda key, count: self.engine_map.index(int(ring2.get_node(str(key))))
        for engine, request in waiting:
            if self.mode == "dualmap":
                await self.scheduler.add_request_to_best_global_queue(request)
            elif engine in self.engine_map:
                slot = self.engine_map.index(engine)
                request._primary_replica = slot
                missing = self.replica_budgets[slot].get_num_recompute_token_ids(request._input_ids)
                self.scheduler.global_request_queue.push(slot, request, len(request._input_ids) - missing)
            else:
                request._physical_engine = engine
                self.retained_waiting.append(request)
        return physical_id

    def physical_target(self, request):
        return request._physical_engine

    def replica_for_engine(self, physical_id):
        return self.all_replicas[physical_id]

    def largest_observed_prefix_owner(self):
        def footprint(engine):
            replica = self.all_replicas[engine]
            requests = replica.pending_requests + replica.running_requests
            return (sum(len(request._input_ids) for request in requests),
                    sum(len(prefix) for prefix in getattr(replica, "snapshots", {})), -engine)
        return max(self.active_ids, key=footprint)

    async def dispatch(self):
        if self.mode == "dualmap":
            ready = await self.scheduler.get_schedulable_waiting_req_list()
        else:
            ready = []
            for i, replica in self.replica_budgets.items():
                if not replica.pending_requests:
                    ready.extend(self.scheduler.global_request_queue.pop_schedulable(i, replica.current_budget, 1))
        for request in ready:
            request._physical_engine = self.engine_map[request._primary_replica]
            await self.replica_budgets[request._primary_replica].add_request(request)
            self.chat_owners[request._native_session_id] = request._physical_engine
        for request in list(self.retained_waiting):
            replica = self.all_replicas[request._physical_engine]
            if not replica.pending_requests:
                self.retained_waiting.remove(request)
                await replica.add_request(request)
                self.chat_owners[request._native_session_id] = request._physical_engine
                ready.append(request)
        return ready

    async def first_token(self, request, elapsed):
        replica = self.all_replicas[request._physical_engine]
        await replica.complete_request_prefill(request._physical_engine, request, elapsed)

    async def finished(self, request, output_ids, success):
        replica = self.all_replicas[request._physical_engine]
        if not success:
            await replica.abort_request(request)
            return
        if request in replica.running_requests:
            replica.running_requests.remove(request)
        # /generate returns actual IDs, so no text re-tokenization is necessary.
        replica.save_token_ids(request._input_ids + (output_ids[:-1] if self.hybrid_snapshot_slots else output_ids))

    def queued(self):
        return sum(len(queue) for queue in self.scheduler.global_request_queue.queues) + len(self.retained_waiting)
