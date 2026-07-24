import asyncio

import pytest

from agentshift.controller.tiered import (
    SharedSemanticHandoffCoordinator,
    TieredPrefixCoordinator,
)
from agentshift.state.schema import AgentContinuation, MigrationState
from agentshift.state.store import SQLiteStateStore


class FakeTierClient:
    def __init__(self, *, failed: bool = False):
        self.failed = failed
        self.starts = []
        self.polls = 0
        self.cleanups = []

    async def start_tier_operation(self, **kwargs):
        self.starts.append(kwargs)
        return {"success": True, "state": "QUEUED"}

    async def tier_status(self, operation_id):
        self.polls += 1
        if self.failed:
            return {
                "success": False,
                "state": "FAILED",
                "error": "injected tier failure",
            }
        state = "COPYING" if self.polls == 1 else "COMPLETE"
        return {
            "success": True,
            "state": state,
            "token_count": 4,
            "bytes_transferred": 4096,
            "operation_seconds": 0.012,
            "queue_seconds": 0.003,
        }

    async def cleanup_tier_operation(self, operation_id, *, drop_checkpoint=False):
        self.cleanups.append((operation_id, drop_checkpoint))
        return {"success": True}


def test_tier_coordinator_runs_polls_and_cleans_up():
    client = FakeTierClient()
    coordinator = TieredPrefixCoordinator(poll_interval=0)
    result = asyncio.run(
        coordinator.run(
            client,
            operation="shared_export",
            checkpoint_id="checkpoint-1",
            agent_id="agent-1",
            owner_epoch=7,
            token_ids=(1, 2, 3, 4),
        )
    )
    assert result.operation == "shared_export"
    assert result.token_count == 4
    assert result.bytes_transferred == 4096
    assert result.worker_seconds == 0.012
    assert result.queue_seconds == 0.003
    assert client.polls == 2
    asyncio.run(coordinator.cleanup(client, result, drop_checkpoint=True))
    assert client.cleanups == [(result.operation_id, True)]


def test_tier_coordinator_fails_without_waiting_for_timeout():
    client = FakeTierClient(failed=True)
    coordinator = TieredPrefixCoordinator(poll_interval=0, operation_timeout=60)
    with pytest.raises(RuntimeError, match="injected tier failure"):
        asyncio.run(
            coordinator.run(
                client,
                operation="private_offload",
                checkpoint_id="checkpoint-1",
                agent_id="agent-1",
                owner_epoch=7,
                token_ids=(1, 2, 3, 4),
            )
        )


class FakeSharedClient(FakeTierClient):
    def __init__(self, engine_id, *, fail_import=False):
        super().__init__()
        self.engine_id = engine_id
        self.fail_import = fail_import
        self.operation_by_id = {}
        self.releases = []

    async def pin_prefix(self, agent_id, owner_epoch, token_ids):
        return {"success": True, "token_count": len(token_ids)}

    async def start_tier_operation(self, **kwargs):
        self.starts.append(kwargs)
        self.operation_by_id[kwargs["operation_id"]] = kwargs["operation"]
        return {"success": True, "state": "QUEUED"}

    async def tier_status(self, operation_id):
        operation = self.operation_by_id[operation_id]
        if self.fail_import and operation == "shared_import":
            return {
                "success": False,
                "state": "FAILED",
                "error": "injected shared import failure",
            }
        return {
            "success": True,
            "state": "COMPLETE",
            "token_count": 4,
            "bytes_transferred": 4096,
            "operation_seconds": 0.012,
            "queue_seconds": 0.003,
        }

    async def release_prefix(self, agent_id, owner_epoch, **kwargs):
        self.releases.append((agent_id, owner_epoch, kwargs))
        return {"success": True}


def test_shared_semantic_handoff_commits_before_source_release(tmp_path):
    store = SQLiteStateStore(tmp_path / "state.db")
    store.register_agent(AgentContinuation("a1", 1, "source", 3, (1, 2, 3, 4)))
    source = FakeSharedClient("source")
    destination = FakeSharedClient("destination")
    coordinator = SharedSemanticHandoffCoordinator(
        store,
        {"source": source, "destination": destination},
        poll_interval=0,
    )
    result = asyncio.run(coordinator.handoff("a1", "destination"))
    assert (store.get_agent("a1").owner_engine, result.new_epoch) == (
        "destination",
        4,
    )
    assert store.get_migration(result.migration_id).state == MigrationState.COMMITTED
    assert source.releases == []
    asyncio.run(coordinator.acknowledge_destination(result.migration_id))
    assert source.releases == [("a1", 3, {"allow_missing": True})]


def test_shared_semantic_handoff_import_failure_keeps_source_owner(tmp_path):
    store = SQLiteStateStore(tmp_path / "state.db")
    store.register_agent(AgentContinuation("a1", 1, "source", 3, (1, 2, 3, 4)))
    source = FakeSharedClient("source")
    destination = FakeSharedClient("destination", fail_import=True)
    coordinator = SharedSemanticHandoffCoordinator(
        store,
        {"source": source, "destination": destination},
        poll_interval=0,
    )
    with pytest.raises(RuntimeError, match="injected shared import failure"):
        asyncio.run(coordinator.handoff("a1", "destination"))
    assert store.get_agent("a1").owner_engine == "source"
    migration = store.conn.execute("SELECT state FROM migrations").fetchone()
    assert migration["state"] == MigrationState.ABORTED.value
