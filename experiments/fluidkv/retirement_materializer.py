"""Experimental drain materialization on observed completed request states.

This adapter exercises the existing planner and ownership store; it does not
implement physical engine shutdown or claim the full FluidKV scheduling design.
"""

import asyncio
import time
import uuid

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.controller.materialization import BranchMaterializationRequest, BranchStatePlanner, MaterializationMethod
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.descriptor import StateDescriptor
from agentshift.state.schema import AgentContinuation, MigrationRecord, MigrationState
from agentshift.state.store import SQLiteStateStore
from experiments.fluidkv.benchmark_native_pairs import stream


class RetirementMaterializer:
    def __init__(self, args, adapter, session, mode, replay_tps, repetition):
        self.args, self.adapter, self.session, self.mode = args, adapter, session, mode
        self.replay_tps = replay_tps
        self.store = SQLiteStateStore(args.output / f"{mode}-{repetition}-ownership.db")
        mode_index = {"dualmap-full-copy": 0, "fluidkv-planner-prototype": 1}[mode]
        pair_count = len(args.engines) * (len(args.engines) - 1) // 2
        transfer_port = args.transfer_port + (2 * repetition + mode_index) * pair_count
        self.coordinator = HybridMigrationCoordinator(self.store, {
            str(i): SGLangAgentShiftClient(str(i), url, timeout=args.request_timeout)
            for i, url in enumerate(args.engines)
        }, base_port=transfer_port)
        self.planner = BranchStatePlanner(bandwidth_bytes_per_second=args.bandwidth_gib * 1024**3,
                                          fixed_copy_seconds=args.copy_fixed_ms / 1000)
        self.completed = {}
        self.records = []
        self.lock = asyncio.Lock()
        self.retired = None
        self.retired_at = None
        self.tasks = set()

    async def initialize(self):
        if self.args.retire_engine == -1:
            return
        source = self.args.retire_engine if self.args.retire_engine is not None else len(self.args.engines) - 1
        for target in range(len(self.args.engines)):
            if target != source:
                await self.coordinator._ensure_group(str(source), str(target))

    def observe_completion(self, chat_id, source, token_ids):
        self.completed[chat_id] = (source, tuple(token_ids))
        if source == self.retired:
            self.schedule(chat_id)

    def retire(self, source):
        self.retired = source
        self.retired_at = time.perf_counter()
        for chat_id, (owner, _) in list(self.completed.items()):
            if owner == source:
                self.schedule(chat_id)

    def schedule(self, chat_id):
        task = asyncio.create_task(self.prepare(chat_id, time.perf_counter()))
        self.tasks.add(task)

    async def prepare(self, chat_id, queued_at):
        async with self.lock:
            source, token_ids = self.completed[chat_id]
            if source != self.retired:
                return
            started = time.perf_counter()
            branch = f"{self.mode}-{chat_id}-{uuid.uuid4().hex}"
            self.store.register_agent(AgentContinuation(agent_id=branch, committed_step=0,
                owner_engine=str(source), owner_epoch=1, token_ids=token_ids))
            candidates = self.adapter.active_ids
            target = min(candidates, key=lambda i: (
                self.adapter.replica_for_engine(i).get_num_running_req()
                + self.adapter.replica_for_engine(i).get_num_pending_req(), i))
            descriptor = StateDescriptor.hybrid_qwen35(descriptor_id=branch,
                model_identity="Qwen3.5-35B-A3B", branch_id=branch, prefix_tokens=len(token_ids))
            request = BranchMaterializationRequest(descriptor=descriptor,
                destination_engine=str(target), remaining_gap_seconds=max(0, self.args.drain_budget - (started - self.retired_at)),
                destination_free_token_slots=self.args.cache_tokens,
                destination_free_snapshot_slots=1, replay_tokens_per_second=self.replay_tps)
            # Estimates only select a method. Engine-side reservations still
            # enforce actual capacity; a failure is recorded, never a false hit.
            plan = self.planner.plan(request)
            method = MaterializationMethod.COPY_FULL_STATE if self.mode == "dualmap-full-copy" else plan.selected.method
            record = {"chat_id": chat_id, "source": source, "target": target,
                      "method": method.value, "input_tokens": len(token_ids),
                      "queue_seconds": started - queued_at,
                      "since_retirement_seconds": started - self.retired_at, "success": False,
                      "transfer_bytes": 0, "reconstruction_tokens": 0}
            record["remaining_drain_budget_seconds"] = request.remaining_gap_seconds
            try:
                migration_id = None
                if method == MaterializationMethod.COPY_FULL_STATE:
                    migration = await self.coordinator.migrate_async(branch, str(target))
                    record["transfer_bytes"] = migration.bytes_transferred
                else:
                    migration_id = uuid.uuid4().hex
                    self.store.start_migration(MigrationRecord(migration_id=migration_id,
                        agent_id=branch, source_engine=str(source), destination_engine=str(target),
                        source_epoch=1, state=MigrationState.PREPARING))
                    self.store.transition_migration(migration_id, MigrationState.PREPARING, MigrationState.COPYING)
                    record["requested_reconstruction_tokens"] = len(token_ids)
                    result = await stream(self.session, self.args.engines[target], list(token_ids), 1)
                    if result["meta_info"]["prompt_tokens"] != len(token_ids):
                        raise ValueError("reconstruction length mismatch")
                    record["reconstruction_tokens"] = len(token_ids) - result["meta_info"]["cached_tokens"]
                    record["preparation_output_tokens"] = result["meta_info"]["completion_tokens"]
                    self.store.transition_migration(migration_id, MigrationState.COPYING, MigrationState.DEST_READY)
                    self.store.commit_migration(migration_id)
                self.adapter.replica_for_engine(target).save_token_ids(list(token_ids))
                self.completed[chat_id] = (target, token_ids)
                record["success"] = self.store.get_agent(branch).owner_engine == str(target)
            except Exception as exc:
                record["error"] = str(exc)
                if migration_id is not None:
                    migration_state = self.store.get_migration(migration_id).state
                    if migration_state in (MigrationState.PREPARING, MigrationState.COPYING):
                        self.store.transition_migration(migration_id, migration_state, MigrationState.ABORTED, error=str(exc))
            record["wall_seconds"] = time.perf_counter() - started
            record["finished_since_retirement_seconds"] = time.perf_counter() - self.retired_at
            self.records.append(record)

    async def finish(self):
        if self.tasks:
            await asyncio.gather(*self.tasks)
        return {
            "scope": "Logical retirement preparation; no source eviction/physical GPU release or full transactional ingress claim.",
            "attempted": len(self.records), "successful": sum(r["success"] for r in self.records),
            "records": self.records,
        }
