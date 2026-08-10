from __future__ import annotations

import asyncio
import inspect
import time
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from agentshift.engine.sglang import SGLangAgentShiftClient, generate
from agentshift.state.schema import AgentContinuation, MigrationRecord, MigrationState
from agentshift.state.store import SQLiteStateStore


@dataclass(frozen=True, slots=True)
class MigrationResult:
    migration_id: str
    agent_id: str
    source_engine: str
    destination_engine: str
    old_epoch: int
    new_epoch: int
    token_count: int
    bytes_transferred: int
    transfer_seconds: float
    worker_transfer_seconds: float = 0.0
    queue_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class ProgressiveMigrationResult:
    migration: MigrationResult
    generation: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ProgressiveContinuation:
    token_ids: tuple[int, ...] | Awaitable[tuple[int, ...]]
    max_new_tokens: int
    rid: str
    layer_group_size: int


@dataclass(frozen=True, slots=True)
class RecoveryResult:
    continuation: AgentContinuation
    source_shadow_reused: bool


class MigrationCoordinator:
    def __init__(
        self,
        store: SQLiteStateStore,
        engines: dict[str, SGLangAgentShiftClient],
        *,
        master_address: str = "127.0.0.1",
        base_port: int = 29600,
        tp_size: int = 1,
        async_transfer: bool = True,
        transfer_poll_interval: float = 0.002,
        transfer_timeout: float = 120.0,
        fault_injector: Callable[[str, str], None] | None = None,
    ):
        self.store = store
        self.engines = engines
        self.master_address = master_address
        self.base_port = base_port
        self.tp_size = tp_size
        self.async_transfer = async_transfer
        self.transfer_poll_interval = transfer_poll_interval
        self.transfer_timeout = transfer_timeout
        self.fault_injector = fault_injector
        self._engine_order = tuple(sorted(engines))
        self._pair_groups: dict[tuple[str, str], tuple[str, tuple[int, ...]]] = {}
        self._group_lock = asyncio.Lock()
        # NCCL P2P messages are untagged. Serialize transfers that share either
        # endpoint, while allowing disjoint engine pairs to proceed in parallel.
        self._engine_locks = {
            engine_id: asyncio.Lock() for engine_id in engines
        }
        # SGLang's tokenizer control communicator accepts one in-flight RPC per
        # operation. First-token ACKs can arrive together, so serialize releases
        # for each source engine while allowing different engines to proceed.
        self._release_locks = {engine_id: asyncio.Lock() for engine_id in engines}

    def _inject_fault(self, point: str, migration_id: str) -> None:
        if self.fault_injector is not None:
            self.fault_injector(point, migration_id)

    async def _ensure_group(
        self, source: str, destination: str
    ) -> tuple[str, tuple[int, ...]]:
        pair = (source, destination)
        async with self._group_lock:
            if pair in self._pair_groups:
                return self._pair_groups[pair]
            source_index = self._engine_order.index(source)
            destination_index = self._engine_order.index(destination)
            destination_slot = (
                destination_index
                if destination_index < source_index
                else destination_index - 1
            )
            pair_index = source_index * (len(self._engine_order) - 1) + destination_slot
            ports = tuple(
                self.base_port + pair_index * self.tp_size + rank
                for rank in range(self.tp_size)
            )
            group_name = f"agentshift_{source}_{destination}"
            await asyncio.gather(
                self.engines[source].init_transfer_group(
                    master_address=self.master_address,
                    ports=ports,
                    group_rank=0,
                    group_name=group_name,
                ),
                self.engines[destination].init_transfer_group(
                    master_address=self.master_address,
                    ports=ports,
                    group_rank=1,
                    group_name=group_name,
                ),
            )
            self._pair_groups[pair] = (group_name, ports)
            return group_name, ports

    async def initialize_transfer_pair(
        self, source: str, destination: str
    ) -> tuple[str, tuple[int, ...]]:
        """Create persistent rank-pair transfer groups outside a timed handoff."""
        return await self._ensure_group(source, destination)

    async def _wait_for_transfer(
        self, engine: SGLangAgentShiftClient, migration_id: str
    ) -> dict:
        deadline = time.monotonic() + self.transfer_timeout
        while True:
            result = await engine.transfer_status(migration_id)
            if result["state"] == "COMPLETE":
                return result
            if result["state"] == "FAILED":
                raise RuntimeError(
                    f"transfer failed: {migration_id}: "
                    f"{result.get('message') or result.get('error') or 'unknown error'}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"transfer timed out: {migration_id}")
            await asyncio.sleep(self.transfer_poll_interval)

    async def _wait_for_transfer_states(
        self,
        engine: SGLangAgentShiftClient,
        migration_id: str,
        states: tuple[str, ...],
    ) -> dict:
        deadline = time.monotonic() + self.transfer_timeout
        while True:
            result = await engine.transfer_status(migration_id)
            if result["state"] in states or result["state"] == "FAILED":
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError(f"transfer timed out: {migration_id}")
            await asyncio.sleep(self.transfer_poll_interval)

    async def migrate(self, agent_id: str, destination_engine: str) -> MigrationResult:
        if destination_engine not in self.engines:
            raise KeyError(destination_engine)
        while True:
            source_engine = self.store.get_agent(agent_id).owner_engine
            if source_engine == destination_engine:
                raise ValueError("source and destination must differ")
            engine_ids = sorted((source_engine, destination_engine))
            locks = [self._engine_locks[engine_id] for engine_id in engine_ids]
            acquired: list[asyncio.Lock] = []
            try:
                for lock in locks:
                    await lock.acquire()
                    acquired.append(lock)
                # An earlier migration of the same agent may have committed
                # while this call waited. Retry with locks for the new owner.
                if self.store.get_agent(agent_id).owner_engine != source_engine:
                    continue
                return await self._migrate_serial(agent_id, destination_engine)
            finally:
                for lock in reversed(acquired):
                    lock.release()

    async def migrate_and_generate_progressive(
        self,
        agent_id: str,
        destination_engine: str,
        *,
        token_ids: tuple[int, ...] | Awaitable[tuple[int, ...]],
        max_new_tokens: int,
        rid: str,
        layer_group_size: int = 4,
    ) -> ProgressiveMigrationResult:
        """Overlap destination continuation with an in-flight prefix transfer.

        The generated response stays private to this controller call until the
        migration has committed, so callers cannot observe destination output
        under the old ownership epoch.
        """
        if not self.async_transfer:
            raise RuntimeError("progressive continuation requires async transfers")
        if layer_group_size <= 0:
            raise ValueError("layer_group_size must be positive")
        continuation = _ProgressiveContinuation(
            token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            rid=rid,
            layer_group_size=layer_group_size,
        )
        while True:
            source_engine = self.store.get_agent(agent_id).owner_engine
            if source_engine == destination_engine:
                raise ValueError("source and destination must differ")
            if destination_engine not in self.engines:
                raise KeyError(destination_engine)
            engine_ids = sorted((source_engine, destination_engine))
            locks = [self._engine_locks[engine_id] for engine_id in engine_ids]
            acquired: list[asyncio.Lock] = []
            try:
                for lock in locks:
                    await lock.acquire()
                    acquired.append(lock)
                if self.store.get_agent(agent_id).owner_engine != source_engine:
                    continue
                result = await self._migrate_serial(
                    agent_id,
                    destination_engine,
                    progressive_continuation=continuation,
                )
                assert isinstance(result, ProgressiveMigrationResult)
                return result
            finally:
                for lock in reversed(acquired):
                    lock.release()

    async def _migrate_serial(
        self,
        agent_id: str,
        destination_engine: str,
        progressive_continuation: _ProgressiveContinuation | None = None,
    ) -> MigrationResult | ProgressiveMigrationResult:
        continuation = self.store.get_agent(agent_id)
        if destination_engine == continuation.owner_engine:
            raise ValueError("source and destination must differ")
        if destination_engine not in self.engines:
            raise KeyError(destination_engine)

        migration_id = uuid.uuid4().hex
        record = MigrationRecord(
            migration_id=migration_id,
            agent_id=agent_id,
            source_engine=continuation.owner_engine,
            destination_engine=destination_engine,
            source_epoch=continuation.owner_epoch,
            state=MigrationState.PREPARING,
        )
        self.store.start_migration(record)
        source = self.engines[continuation.owner_engine]
        destination = self.engines[destination_engine]

        try:
            pin = await source.pin_prefix(
                agent_id, continuation.owner_epoch, continuation.token_ids
            )
            self._inject_fault("SOURCE_PINNED", migration_id)
            token_count = int(pin["token_count"])
            if token_count <= 0:
                raise RuntimeError("source has no completed prefix to migrate")
            token_ids = continuation.token_ids[:token_count]
            if (
                progressive_continuation is not None
                and isinstance(progressive_continuation.token_ids, tuple)
            ):
                if len(progressive_continuation.token_ids) <= token_count:
                    raise RuntimeError(
                        "progressive continuation must append at least one token"
                    )
                if progressive_continuation.token_ids[:token_count] != token_ids:
                    raise RuntimeError(
                        "progressive continuation does not extend the migrated prefix"
                    )
            group_name, ports = await self._ensure_group(
                continuation.owner_engine, destination_engine
            )
            self.store.transition_migration(
                migration_id,
                MigrationState.PREPARING,
                MigrationState.COPYING,
                token_count=token_count,
            )
            started = time.perf_counter()
            transfer_args = dict(
                migration_id=migration_id,
                agent_id=agent_id,
                token_ids=token_ids,
                group_name=group_name,
                ports=ports,
            )
            if self.async_transfer:
                await destination.transfer_prefix(
                    role="destination",
                    owner_epoch=continuation.owner_epoch + 1,
                    async_transfer=True,
                    progressive=progressive_continuation is not None,
                    layer_group_size=(
                        progressive_continuation.layer_group_size
                        if progressive_continuation is not None
                        else 4
                    ),
                    **transfer_args,
                )
                await source.transfer_prefix(
                    role="source",
                    owner_epoch=continuation.owner_epoch,
                    async_transfer=True,
                    **transfer_args,
                )
                progressive_token_ids = None
                if progressive_continuation is not None:
                    token_source = progressive_continuation.token_ids
                    try:
                        progressive_token_ids = (
                            await token_source
                            if inspect.isawaitable(token_source)
                            else token_source
                        )
                        if len(progressive_token_ids) <= token_count:
                            raise RuntimeError(
                                "progressive continuation must append at least one token"
                            )
                        if progressive_token_ids[:token_count] != token_ids:
                            raise RuntimeError(
                                "progressive continuation does not extend the migrated prefix"
                            )
                    except BaseException:
                        await asyncio.gather(
                            self._wait_for_transfer_states(
                                source, migration_id, ("COMPLETE",)
                            ),
                            self._wait_for_transfer_states(
                                destination, migration_id, ("COPY_DONE", "COMPLETE")
                            ),
                            return_exceptions=True,
                        )
                        await asyncio.gather(
                            source.cleanup_transfer(migration_id),
                            destination.cleanup_transfer(migration_id),
                            return_exceptions=True,
                        )
                        raise
                waiters = [
                    self._wait_for_transfer(source, migration_id),
                    self._wait_for_transfer(destination, migration_id),
                ]
                if progressive_continuation is not None:
                    waiters.append(
                        generate(
                            destination,
                            list(progressive_token_ids),
                            max_new_tokens=progressive_continuation.max_new_tokens,
                            rid=progressive_continuation.rid,
                            agentshift_progressive_migration_id=migration_id,
                        )
                    )
                transfer_results = await asyncio.gather(
                    *waiters, return_exceptions=True
                )
            else:
                transfer_results = await asyncio.gather(
                    source.transfer_prefix(
                        role="source",
                        owner_epoch=continuation.owner_epoch,
                        async_transfer=False,
                        **transfer_args,
                    ),
                    destination.transfer_prefix(
                        role="destination",
                        owner_epoch=continuation.owner_epoch + 1,
                        async_transfer=False,
                        **transfer_args,
                    ),
                    return_exceptions=True,
                )
            errors = [item for item in transfer_results if isinstance(item, Exception)]
            if errors:
                try:
                    await destination.release_prefix(
                        agent_id,
                        continuation.owner_epoch + 1,
                        allow_missing=True,
                    )
                except Exception:
                    pass
                await asyncio.gather(
                    source.cleanup_transfer(migration_id),
                    destination.cleanup_transfer(migration_id),
                    return_exceptions=True,
                )
                raise errors[0]
            source_result, destination_result = transfer_results[:2]
            generation_result = (
                transfer_results[2] if progressive_continuation is not None else None
            )
            transfer_seconds = time.perf_counter() - started
            worker_transfer_seconds = max(
                float(source_result.get("transfer_seconds", 0.0)),
                float(destination_result.get("transfer_seconds", 0.0)),
            )
            queue_seconds = max(
                float(source_result.get("queue_seconds", 0.0)),
                float(destination_result.get("queue_seconds", 0.0)),
            )
            bytes_transferred = int(
                max(
                    source_result.get("bytes_transferred", 0),
                    destination_result.get("bytes_transferred", 0),
                )
            )
            await asyncio.gather(
                source.cleanup_transfer(migration_id),
                destination.cleanup_transfer(migration_id),
                return_exceptions=True,
            )
            self.store.transition_migration(
                migration_id,
                MigrationState.COPYING,
                MigrationState.DEST_READY,
                bytes_transferred=bytes_transferred,
                transfer_seconds=transfer_seconds,
            )
            self._inject_fault("DEST_READY", migration_id)
            new_continuation = self.store.commit_migration(migration_id)
            self._inject_fault("COMMITTED", migration_id)
            migration_result = MigrationResult(
                migration_id=migration_id,
                agent_id=agent_id,
                source_engine=continuation.owner_engine,
                destination_engine=destination_engine,
                old_epoch=continuation.owner_epoch,
                new_epoch=new_continuation.owner_epoch,
                token_count=token_count,
                bytes_transferred=bytes_transferred,
                transfer_seconds=transfer_seconds,
                worker_transfer_seconds=worker_transfer_seconds,
                queue_seconds=queue_seconds,
            )
            if progressive_continuation is not None:
                assert isinstance(generation_result, dict)
                return ProgressiveMigrationResult(
                    migration=migration_result,
                    generation=generation_result,
                )
            return migration_result
        except Exception as exc:
            current = self.store.get_migration(migration_id)
            if current.state in (MigrationState.PREPARING, MigrationState.COPYING):
                self.store.transition_migration(
                    migration_id,
                    current.state,
                    MigrationState.ABORTED,
                    error=str(exc),
                )
            raise

    async def acknowledge_destination(self, migration_id: str) -> None:
        initial = self.store.get_migration(migration_id)
        async with self._release_locks[initial.source_engine]:
            migration = self.store.get_migration(migration_id)
            if migration.state == MigrationState.SOURCE_RELEASED:
                return
            if migration.state != MigrationState.COMMITTED:
                raise RuntimeError("migration has not committed")
            await self.engines[migration.source_engine].release_prefix(
                migration.agent_id, migration.source_epoch, allow_missing=True
            )
            self.store.transition_migration(
                migration_id,
                MigrationState.COMMITTED,
                MigrationState.SOURCE_RELEASED,
            )
            self._inject_fault("SOURCE_RELEASED", migration_id)

    async def finalize_destination(
        self, migration_id: str, *, keep_cached: bool = True
    ) -> None:
        """Finish first-token cleanup while preserving an extendable prefix."""
        migration = self.store.get_migration(migration_id)
        await self.acknowledge_destination(migration_id)
        destination_epoch = migration.source_epoch + 1
        async with self._release_locks[migration.destination_engine]:
            await self.engines[migration.destination_engine].release_prefix(
                migration.agent_id,
                destination_epoch,
                evict_after_release=not keep_cached,
                allow_missing=True,
            )

    async def recover_committed(self, migration_id: str):
        recovery = await self.recover_destination_failure(migration_id)
        return recovery.continuation

    async def recover_destination_failure(
        self, migration_id: str, recovery_engine: str | None = None
    ) -> RecoveryResult:
        migration = self.store.get_migration(migration_id)
        if migration.state == MigrationState.RECOVERED:
            return RecoveryResult(self.store.get_agent(migration.agent_id), False)
        if migration.state not in (
            MigrationState.COMMITTED,
            MigrationState.SOURCE_RELEASED,
        ):
            raise RuntimeError("migration has not committed")
        recovery_engine = recovery_engine or migration.source_engine
        recovery_epoch = migration.source_epoch + 2
        shadow_reused = False
        if (
            migration.state == MigrationState.COMMITTED
            and recovery_engine == migration.source_engine
        ):
            try:
                await self.engines[migration.source_engine].rebind_prefix(
                    migration.agent_id,
                    expected_owner_epoch=migration.source_epoch,
                    new_owner_epoch=recovery_epoch,
                )
            except Exception:
                pass
            else:
                shadow_reused = True
        continuation = self.store.recover_committed_migration(
            migration_id, recovery_engine
        )
        return RecoveryResult(continuation, shadow_reused)

    async def reconcile_after_restart(self) -> list[dict[str, str]]:
        actions = []
        live_states = (
            MigrationState.PREPARING,
            MigrationState.COPYING,
            MigrationState.DEST_READY,
            MigrationState.COMMITTED,
        )
        for migration in self.store.list_migrations(live_states):
            if migration.state == MigrationState.COMMITTED:
                actions.append(
                    {
                        "migration_id": migration.migration_id,
                        "action": "retain-source-shadow",
                    }
                )
                continue

            source = self.engines[migration.source_engine]
            destination = self.engines[migration.destination_engine]
            if migration.state == MigrationState.COPYING:
                statuses = await asyncio.gather(
                    source.transfer_status(migration.migration_id),
                    destination.transfer_status(migration.migration_id),
                    return_exceptions=True,
                )
                still_copying = any(
                    isinstance(status, dict)
                    and status.get("state") not in ("COMPLETE", "FAILED")
                    for status in statuses
                )
                if still_copying:
                    actions.append(
                        {
                            "migration_id": migration.migration_id,
                            "action": "wait-for-copy",
                        }
                    )
                    continue

            if migration.state == MigrationState.COPYING:
                await asyncio.gather(
                    source.cleanup_transfer(migration.migration_id),
                    destination.cleanup_transfer(migration.migration_id),
                    return_exceptions=True,
                )
            await destination.release_prefix(
                migration.agent_id,
                migration.source_epoch + 1,
                allow_missing=True,
            )
            self.store.transition_migration(
                migration.migration_id,
                migration.state,
                MigrationState.ABORTED,
                error="controller restart before ownership commit",
            )
            actions.append(
                {
                    "migration_id": migration.migration_id,
                    "action": "abort-to-source",
                }
            )
        return actions
