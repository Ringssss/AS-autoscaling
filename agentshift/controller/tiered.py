from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

from agentshift.controller.migration import MigrationResult
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.state.schema import MigrationRecord, MigrationState
from agentshift.state.store import SQLiteStateStore


@dataclass(frozen=True, slots=True)
class TierOperationResult:
    operation_id: str
    checkpoint_id: str
    operation: str
    token_count: int
    bytes_transferred: int
    wall_seconds: float
    worker_seconds: float
    queue_seconds: float


class TieredPrefixCoordinator:
    def __init__(
        self,
        *,
        poll_interval: float = 0.002,
        operation_timeout: float = 300.0,
    ):
        self.poll_interval = poll_interval
        self.operation_timeout = operation_timeout

    async def _wait(
        self, client: SGLangAgentShiftClient, operation_id: str
    ) -> dict:
        deadline = time.monotonic() + self.operation_timeout
        while True:
            result = await client.tier_status(operation_id)
            if result["state"] == "COMPLETE":
                return result
            if result["state"] == "FAILED":
                raise RuntimeError(
                    f"tier operation failed: {operation_id}: "
                    f"{result.get('message') or result.get('error') or 'unknown error'}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"tier operation timed out: {operation_id}")
            await asyncio.sleep(self.poll_interval)

    async def run(
        self,
        client: SGLangAgentShiftClient,
        *,
        operation: str,
        checkpoint_id: str,
        agent_id: str,
        owner_epoch: int,
        token_ids: tuple[int, ...],
        release_gpu: bool = True,
    ) -> TierOperationResult:
        operation_id = uuid.uuid4().hex
        started = time.perf_counter()
        await client.start_tier_operation(
            operation=operation,
            operation_id=operation_id,
            checkpoint_id=checkpoint_id,
            agent_id=agent_id,
            owner_epoch=owner_epoch,
            token_ids=token_ids,
            release_gpu=release_gpu,
        )
        try:
            result = await self._wait(client, operation_id)
        except Exception:
            await client.cleanup_tier_operation(operation_id)
            raise
        wall_seconds = time.perf_counter() - started
        return TierOperationResult(
            operation_id=operation_id,
            checkpoint_id=checkpoint_id,
            operation=operation,
            token_count=int(result["token_count"]),
            bytes_transferred=int(result["bytes_transferred"]),
            wall_seconds=wall_seconds,
            worker_seconds=float(result["operation_seconds"]),
            queue_seconds=float(result["queue_seconds"]),
        )

    async def cleanup(
        self,
        client: SGLangAgentShiftClient,
        result: TierOperationResult,
        *,
        drop_checkpoint: bool = False,
    ) -> None:
        await client.cleanup_tier_operation(
            result.operation_id, drop_checkpoint=drop_checkpoint
        )


class SharedSemanticHandoffCoordinator:
    """Semantic handoff using a shared host tier instead of direct GPU P2P."""

    def __init__(
        self,
        store: SQLiteStateStore,
        engines: dict[str, SGLangAgentShiftClient],
        *,
        poll_interval: float = 0.002,
        operation_timeout: float = 300.0,
    ):
        self.store = store
        self.engines = engines
        self.tiered = TieredPrefixCoordinator(
            poll_interval=poll_interval,
            operation_timeout=operation_timeout,
        )
        self._handoff_lock = asyncio.Lock()

    async def handoff(
        self, agent_id: str, destination_engine: str
    ) -> MigrationResult:
        async with self._handoff_lock:
            return await self._handoff_serial(agent_id, destination_engine)

    async def _handoff_serial(
        self, agent_id: str, destination_engine: str
    ) -> MigrationResult:
        continuation = self.store.get_agent(agent_id)
        if destination_engine == continuation.owner_engine:
            raise ValueError("source and destination must differ")
        try:
            source = self.engines[continuation.owner_engine]
            destination = self.engines[destination_engine]
        except KeyError as exc:
            raise KeyError(f"unknown handoff engine: {exc.args[0]}") from exc

        migration_id = uuid.uuid4().hex
        checkpoint_id = f"shared-cas-{migration_id}"
        self.store.start_migration(
            MigrationRecord(
                migration_id=migration_id,
                agent_id=agent_id,
                source_engine=continuation.owner_engine,
                destination_engine=destination_engine,
                source_epoch=continuation.owner_epoch,
                state=MigrationState.PREPARING,
            )
        )
        export = None
        imported = None
        started = time.perf_counter()
        try:
            pin = await source.pin_prefix(
                agent_id, continuation.owner_epoch, continuation.token_ids
            )
            token_count = int(pin["token_count"])
            if token_count <= 0:
                raise RuntimeError("source has no completed prefix to export")
            token_ids = continuation.token_ids[:token_count]
            self.store.transition_migration(
                migration_id,
                MigrationState.PREPARING,
                MigrationState.COPYING,
                token_count=token_count,
            )
            export = await self.tiered.run(
                source,
                operation="shared_export",
                checkpoint_id=checkpoint_id,
                agent_id=agent_id,
                owner_epoch=continuation.owner_epoch,
                token_ids=token_ids,
                release_gpu=False,
            )
            imported = await self.tiered.run(
                destination,
                operation="shared_import",
                checkpoint_id=checkpoint_id,
                agent_id=agent_id,
                owner_epoch=continuation.owner_epoch + 1,
                token_ids=token_ids,
                release_gpu=False,
            )
            transfer_seconds = time.perf_counter() - started
            bytes_transferred = export.bytes_transferred + imported.bytes_transferred
            self.store.transition_migration(
                migration_id,
                MigrationState.COPYING,
                MigrationState.DEST_READY,
                bytes_transferred=bytes_transferred,
                transfer_seconds=transfer_seconds,
            )
            new_continuation = self.store.commit_migration(migration_id)
            await self.tiered.cleanup(source, export)
            await self.tiered.cleanup(
                destination, imported, drop_checkpoint=True
            )
            return MigrationResult(
                migration_id=migration_id,
                agent_id=agent_id,
                source_engine=continuation.owner_engine,
                destination_engine=destination_engine,
                old_epoch=continuation.owner_epoch,
                new_epoch=new_continuation.owner_epoch,
                token_count=token_count,
                bytes_transferred=bytes_transferred,
                transfer_seconds=transfer_seconds,
                worker_transfer_seconds=(
                    export.worker_seconds + imported.worker_seconds
                ),
                queue_seconds=export.queue_seconds + imported.queue_seconds,
            )
        except Exception as exc:
            if imported is not None:
                await destination.release_prefix(
                    agent_id,
                    continuation.owner_epoch + 1,
                    allow_missing=True,
                )
                await self.tiered.cleanup(
                    destination,
                    imported,
                    drop_checkpoint=True,
                )
            if export is not None:
                await self.tiered.cleanup(
                    source,
                    export,
                    drop_checkpoint=imported is None,
                )
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
        migration = self.store.get_migration(migration_id)
        if migration.state == MigrationState.SOURCE_RELEASED:
            return
        if migration.state != MigrationState.COMMITTED:
            raise RuntimeError("migration has not committed")
        await self.engines[migration.source_engine].release_prefix(
            migration.agent_id,
            migration.source_epoch,
            allow_missing=True,
        )
        self.store.transition_migration(
            migration_id,
            MigrationState.COMMITTED,
            MigrationState.SOURCE_RELEASED,
        )
