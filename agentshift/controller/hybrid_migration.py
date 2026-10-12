from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from agentshift.state.schema import AgentContinuation, MigrationRecord, MigrationState
from agentshift.state.store import SQLiteStateStore


@dataclass(frozen=True, slots=True)
class HybridMigrationResult:
    migration_id: str
    agent_id: str
    source_engine: str
    destination_engine: str
    old_epoch: int
    new_epoch: int
    continuation: AgentContinuation
    token_count: int
    bytes_transferred: int
    transfer_seconds: float
    group_name: str


class HybridMigrationCoordinator:
    """Commit durable ownership after a complete heterogeneous bundle is installed."""

    def __init__(
        self,
        store: SQLiteStateStore,
        engines: dict,
        *,
        master_address: str = "127.0.0.1",
        base_port: int = 29700,
    ):
        if not engines:
            raise ValueError("at least one engine is required")
        self.store = store
        self.engines = engines
        self.master_address = master_address
        self.base_port = base_port
        pairs = sorted(
            {
                frozenset((source, destination))
                for source in engines
                for destination in engines
                if source != destination
            },
            key=lambda pair: tuple(sorted(pair)),
        )
        self._pair_indices = {pair: index for index, pair in enumerate(pairs)}
        self._group_locks = {
            pair: asyncio.Lock()
            for pair in {
                frozenset((source, destination))
                for source in engines
                for destination in engines
                if source != destination
            }
        }
        self._groups: dict[frozenset[str], tuple[str, int]] = {}

    async def _ensure_group(self, source: str, destination: str) -> tuple[str, int]:
        pair = frozenset((source, destination))
        async with self._group_locks[pair]:
            if pair in self._groups:
                return self._groups[pair]

            port = self.base_port + self._pair_indices[pair]
            group_name = f"hybrid_{source}_{destination}_{port}_{uuid.uuid4().hex[:8]}"
            await asyncio.gather(
                self.engines[source].init_hybrid_state_group(
                    master_address=self.master_address,
                    port=port,
                    group_rank=0,
                    group_name=group_name,
                ),
                self.engines[destination].init_hybrid_state_group(
                    master_address=self.master_address,
                    port=port,
                    group_rank=1,
                    group_name=group_name,
                ),
            )
            self._groups[pair] = (group_name, port)
            return self._groups[pair]

    async def migrate(
        self,
        agent_id: str,
        destination_engine: str,
    ) -> HybridMigrationResult:
        continuation = self.store.get_agent(agent_id)
        source_engine = continuation.owner_engine
        if destination_engine == source_engine:
            raise ValueError("destination must differ from the source engine")
        if destination_engine not in self.engines or source_engine not in self.engines:
            raise KeyError(destination_engine)

        migration_id = uuid.uuid4().hex
        record = MigrationRecord(
            migration_id=migration_id,
            agent_id=agent_id,
            source_engine=source_engine,
            destination_engine=destination_engine,
            source_epoch=continuation.owner_epoch,
            state=MigrationState.PREPARING,
        )
        self.store.start_migration(record)
        current_state = MigrationState.PREPARING

        try:
            group_name, port = await self._ensure_group(
                source_engine, destination_engine
            )
            self.store.transition_migration(
                migration_id,
                MigrationState.PREPARING,
                MigrationState.COPYING,
                token_count=len(continuation.token_ids),
            )
            current_state = MigrationState.COPYING

            started = time.perf_counter()
            source_result, destination_result = await asyncio.gather(
                self.engines[source_engine].direct_transfer_hybrid_state(
                    role="source",
                    token_ids=continuation.token_ids,
                    group_name=group_name,
                    port=port,
                ),
                self.engines[destination_engine].direct_transfer_hybrid_state(
                    role="destination",
                    token_ids=continuation.token_ids,
                    group_name=group_name,
                    port=port,
                ),
            )
            elapsed = time.perf_counter() - started

            for engine, transfer_result in (
                (source_engine, source_result),
                (destination_engine, destination_result),
            ):
                if transfer_result.get("success") is False:
                    raise RuntimeError(
                        f"{engine} hybrid transfer failed: "
                        f"{transfer_result.get('message', 'request failed')}"
                    )

            token_count = int(source_result["token_count"])
            bytes_transferred = int(source_result["bytes_transferred"])
            if destination_result["token_count"] != token_count:
                raise RuntimeError("source and destination token counts disagree")
            if destination_result["bytes_transferred"] != bytes_transferred:
                raise RuntimeError(
                    "source and destination transfer byte counts disagree"
                )
            if token_count != len(continuation.token_ids):
                raise RuntimeError(
                    "transferred token count does not match continuation"
                )

            self.store.transition_migration(
                migration_id,
                MigrationState.COPYING,
                MigrationState.DEST_READY,
                bytes_transferred=bytes_transferred,
                transfer_seconds=elapsed,
            )
            current_state = MigrationState.DEST_READY
            new_continuation = self.store.commit_migration(migration_id)
        except Exception as exc:
            if current_state == MigrationState.PREPARING:
                self.store.transition_migration(
                    migration_id,
                    MigrationState.PREPARING,
                    MigrationState.ABORTED,
                    error=str(exc),
                )
            elif current_state == MigrationState.COPYING:
                self.store.transition_migration(
                    migration_id,
                    MigrationState.COPYING,
                    MigrationState.ABORTED,
                    error=str(exc),
                )
            raise

        return HybridMigrationResult(
            migration_id=migration_id,
            agent_id=agent_id,
            source_engine=source_engine,
            destination_engine=destination_engine,
            old_epoch=continuation.owner_epoch,
            new_epoch=new_continuation.owner_epoch,
            continuation=new_continuation,
            token_count=token_count,
            bytes_transferred=bytes_transferred,
            transfer_seconds=elapsed,
            group_name=group_name,
        )

    async def migrate_async(
        self,
        agent_id: str,
        destination_engine: str,
    ) -> HybridMigrationResult:
        continuation = self.store.get_agent(agent_id)
        source_engine = continuation.owner_engine
        if destination_engine == source_engine:
            raise ValueError("destination must differ from the source engine")
        if destination_engine not in self.engines or source_engine not in self.engines:
            raise KeyError(destination_engine)

        migration_id = uuid.uuid4().hex
        record = MigrationRecord(
            migration_id=migration_id,
            agent_id=agent_id,
            source_engine=source_engine,
            destination_engine=destination_engine,
            source_epoch=continuation.owner_epoch,
            state=MigrationState.PREPARING,
        )
        self.store.start_migration(record)
        current_state = MigrationState.PREPARING
        transfer_id = f"hybrid-{migration_id}"

        try:
            group_name, port = await self._ensure_group(
                source_engine, destination_engine
            )
            started = time.perf_counter()
            # No NCCL work may start until both engines own their reservations.
            for engine_id, role in ((source_engine, "source"), (destination_engine, "destination")):
                prepared = await self.engines[engine_id].start_async_hybrid_state_transfer(
                    transfer_id=transfer_id, role=role,
                    token_ids=continuation.token_ids, group_name=group_name,
                    port=port, prepare_only=True,
                )
                if prepared.get("state") != "PREPARED":
                    raise RuntimeError(prepared.get("error") or "engine did not prepare hybrid transfer")
            self.store.transition_migration(
                migration_id,
                MigrationState.PREPARING,
                MigrationState.COPYING,
                token_count=len(continuation.token_ids),
            )
            current_state = MigrationState.COPYING
            source_start, destination_start = await asyncio.gather(
                self.engines[source_engine].start_async_hybrid_state_transfer(
                    transfer_id=transfer_id,
                    role="source",
                    token_ids=continuation.token_ids,
                    group_name=group_name,
                    port=port,
                ),
                self.engines[destination_engine].start_async_hybrid_state_transfer(
                    transfer_id=transfer_id,
                    role="destination",
                    token_ids=continuation.token_ids,
                    group_name=group_name,
                    port=port,
                ),
            )
            for result in (source_start, destination_start):
                if result.get("state") == "FAILED":
                    raise RuntimeError(result.get("error", "async transfer failed"))

            deadline = time.monotonic() + 120
            while True:
                source_result, destination_result = await asyncio.gather(
                    self.engines[source_engine].async_hybrid_state_transfer_status(
                        transfer_id
                    ),
                    self.engines[destination_engine].async_hybrid_state_transfer_status(
                        transfer_id
                    ),
                )
                if any(
                    result.get("state") == "FAILED"
                    for result in (
                        source_result,
                        destination_result,
                    )
                ):
                    raise RuntimeError(
                        source_result.get("error")
                        or destination_result.get("error")
                        or "async transfer failed"
                    )
                if all(
                    result.get("state") == "COMPLETE"
                    for result in (source_result, destination_result)
                ):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("async hybrid transfer timed out")
                await asyncio.sleep(0.02)
            elapsed = time.perf_counter() - started

            token_count = int(source_result["token_count"])
            bytes_transferred = int(source_result["bytes_transferred"])
            if destination_result["token_count"] != token_count:
                raise RuntimeError("source and destination token counts disagree")
            if destination_result["bytes_transferred"] != bytes_transferred:
                raise RuntimeError("source and destination byte counts disagree")
            if token_count != len(continuation.token_ids):
                raise RuntimeError(
                    "transferred token count does not match continuation"
                )

            self.store.transition_migration(
                migration_id,
                MigrationState.COPYING,
                MigrationState.DEST_READY,
                bytes_transferred=bytes_transferred,
                transfer_seconds=elapsed,
            )
            current_state = MigrationState.DEST_READY
            new_continuation = self.store.commit_migration(migration_id)
            await asyncio.gather(
                self.engines[source_engine].cleanup_async_hybrid_state_transfer(
                    transfer_id
                ),
                self.engines[destination_engine].cleanup_async_hybrid_state_transfer(
                    transfer_id
                ),
            )
        except Exception as exc:
            cleanup = asyncio.gather(
                self.engines[source_engine].cleanup_async_hybrid_state_transfer(
                    transfer_id
                ),
                self.engines[destination_engine].cleanup_async_hybrid_state_transfer(
                    transfer_id
                ),
                return_exceptions=True,
            )
            await cleanup
            if current_state == MigrationState.PREPARING:
                self.store.transition_migration(
                    migration_id,
                    MigrationState.PREPARING,
                    MigrationState.ABORTED,
                    error=str(exc),
                )
            elif current_state == MigrationState.COPYING:
                self.store.transition_migration(
                    migration_id,
                    MigrationState.COPYING,
                    MigrationState.ABORTED,
                    error=str(exc),
                )
            raise

        return HybridMigrationResult(
            migration_id=migration_id,
            agent_id=agent_id,
            source_engine=source_engine,
            destination_engine=destination_engine,
            old_epoch=continuation.owner_epoch,
            new_epoch=new_continuation.owner_epoch,
            continuation=new_continuation,
            token_count=token_count,
            bytes_transferred=bytes_transferred,
            transfer_seconds=elapsed,
            group_name=group_name,
        )

    def acknowledge_async_migration(self, migration_id: str) -> None:
        """Close the live migration after async source cleanup has completed."""
        self.store.transition_migration(
            migration_id,
            MigrationState.COMMITTED,
            MigrationState.SOURCE_RELEASED,
        )
