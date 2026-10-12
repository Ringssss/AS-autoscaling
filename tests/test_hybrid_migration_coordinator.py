import asyncio

import pytest

from agentshift.controller.hybrid_migration import HybridMigrationCoordinator
from agentshift.state.schema import AgentContinuation, MigrationState
from agentshift.state.store import SQLiteStateStore


class FakeHybridEngine:
    def __init__(
        self,
        engine_id: str,
        *,
        destination_bytes: int | None = None,
        fail: bool = False,
    ):
        self.engine_id = engine_id
        self.destination_bytes = destination_bytes
        self.fail = fail
        self.group_calls = []
        self.transfer_calls = []
        self.async_calls = []
        self.cleanup_calls = []

    async def init_hybrid_state_group(self, **kwargs):
        self.group_calls.append(kwargs)
        return {"success": True}

    async def direct_transfer_hybrid_state(self, **kwargs):
        self.transfer_calls.append(kwargs)
        if self.fail:
            raise RuntimeError("injected hybrid transfer failure")
        token_count = len(kwargs["token_ids"])
        bytes_transferred = token_count * 20480 + 64389120
        if kwargs["role"] == "destination" and self.destination_bytes is not None:
            bytes_transferred = self.destination_bytes
        return {
            "role": kwargs["role"],
            "token_count": token_count,
            "bytes_transferred": bytes_transferred,
            "elapsed_seconds": 0.01,
        }

    async def start_async_hybrid_state_transfer(self, **kwargs):
        self.async_calls.append(kwargs)
        if self.fail and kwargs.get("prepare_only"):
            raise RuntimeError("injected reservation failure")
        return {
            "transfer_id": kwargs["transfer_id"],
            "state": "PREPARED" if kwargs.get("prepare_only") else "COPYING",
            "token_count": len(kwargs["token_ids"]),
            "bytes_transferred": 0,
            "error": None,
        }

    async def async_hybrid_state_transfer_status(self, transfer_id):
        return {
            "transfer_id": transfer_id,
            "state": "COMPLETE",
            "token_count": 4,
            "bytes_transferred": 4 * 20480 + 64389120,
            "error": None,
        }

    async def cleanup_async_hybrid_state_transfer(self, transfer_id):
        self.cleanup_calls.append(transfer_id)
        return {"transfer_id": transfer_id, "state": "CLEANED"}


def make_store(tmp_path):
    store = SQLiteStateStore(tmp_path / "state.db")
    store.register_agent(AgentContinuation("a1", 3, "source", 7, (1, 2, 3, 4)))
    return store


def test_hybrid_migration_commits_bundle_then_ownership(tmp_path):
    store = make_store(tmp_path)
    source = FakeHybridEngine("source")
    destination = FakeHybridEngine("destination")
    coordinator = HybridMigrationCoordinator(
        store, {"source": source, "destination": destination}, base_port=29700
    )
    result = asyncio.run(coordinator.migrate("a1", "destination"))

    assert result.token_count == 4
    assert result.bytes_transferred == 4 * 20480 + 64389120
    assert result.new_epoch == 8
    assert store.get_agent("a1").owner_engine == "destination"
    assert store.get_migration(result.migration_id).state == MigrationState.COMMITTED
    assert [call["role"] for call in source.transfer_calls] == ["source"]
    assert [call["role"] for call in destination.transfer_calls] == ["destination"]


def test_hybrid_transfer_failure_preserves_source_owner(tmp_path):
    store = make_store(tmp_path)
    source = FakeHybridEngine("source")
    destination = FakeHybridEngine("destination", fail=True)
    coordinator = HybridMigrationCoordinator(
        store, {"source": source, "destination": destination}
    )
    with pytest.raises(RuntimeError, match="injected hybrid transfer failure"):
        asyncio.run(coordinator.migrate("a1", "destination"))

    agent = store.get_agent("a1")
    assert (agent.owner_engine, agent.owner_epoch) == ("source", 7)
    assert (
        store.get_migration(
            store.conn.execute("SELECT migration_id FROM migrations").fetchone()[0]
        ).state
        == MigrationState.ABORTED
    )


def test_hybrid_byte_count_mismatch_fails_closed(tmp_path):
    store = make_store(tmp_path)
    source = FakeHybridEngine("source")
    destination = FakeHybridEngine("destination", destination_bytes=1)
    coordinator = HybridMigrationCoordinator(
        store, {"source": source, "destination": destination}
    )
    with pytest.raises(RuntimeError, match="byte counts disagree"):
        asyncio.run(coordinator.migrate("a1", "destination"))
    assert store.get_agent("a1").owner_engine == "source"


@pytest.mark.parametrize("failed_engine", ["source", "destination"])
def test_explicit_failure_with_matching_counts_does_not_commit(tmp_path, failed_engine):
    class RejectedEngine(FakeHybridEngine):
        async def direct_transfer_hybrid_state(self, **kwargs):
            result = await super().direct_transfer_hybrid_state(**kwargs)
            return result | {"success": False, "message": "TP shard not installed"}

    store = make_store(tmp_path)
    engines = {
        name: (RejectedEngine(name) if name == failed_engine else FakeHybridEngine(name))
        for name in ("source", "destination")
    }
    coordinator = HybridMigrationCoordinator(store, engines)
    with pytest.raises(RuntimeError, match="TP shard not installed"):
        asyncio.run(coordinator.migrate("a1", "destination"))
    agent = store.get_agent("a1")
    assert (agent.owner_engine, agent.owner_epoch) == ("source", 7)
    migration_id = store.conn.execute("SELECT migration_id FROM migrations").fetchone()[0]
    assert store.get_migration(migration_id).state == MigrationState.ABORTED


def test_hybrid_group_is_initialized_once_per_pair(tmp_path):
    store = make_store(tmp_path)
    source = FakeHybridEngine("source")
    destination = FakeHybridEngine("destination")
    coordinator = HybridMigrationCoordinator(
        store, {"source": source, "destination": destination}
    )
    asyncio.run(coordinator.migrate("a1", "destination"))
    store.register_agent(AgentContinuation("a2", 1, "source", 9, (5, 6)))
    asyncio.run(coordinator.migrate("a2", "destination"))
    assert len(source.group_calls) == 1
    assert len(destination.group_calls) == 1


def test_hybrid_async_migration_commits_after_both_sides_complete(tmp_path):
    store = make_store(tmp_path)
    source = FakeHybridEngine("source")
    destination = FakeHybridEngine("destination")
    coordinator = HybridMigrationCoordinator(
        store, {"source": source, "destination": destination}
    )
    result = asyncio.run(coordinator.migrate_async("a1", "destination"))
    assert result.new_epoch == 8
    assert store.get_agent("a1").owner_engine == "destination"
    assert store.get_migration(result.migration_id).state == MigrationState.COMMITTED
    assert source.async_calls[0]["prepare_only"]
    assert destination.async_calls[0]["prepare_only"]


@pytest.mark.parametrize("failed_engine", ["source", "destination"])
def test_reservation_failure_starts_no_network_work_and_preserves_owner(tmp_path, failed_engine):
    store = make_store(tmp_path)
    engines = {name: FakeHybridEngine(name, fail=name == failed_engine) for name in ("source", "destination")}
    coordinator = HybridMigrationCoordinator(store, engines)
    with pytest.raises(RuntimeError, match="reservation failure"):
        asyncio.run(coordinator.migrate_async("a1", "destination"))
    assert (store.get_agent("a1").owner_engine, store.get_agent("a1").owner_epoch) == ("source", 7)
    for engine in engines.values():
        assert all(call.get("prepare_only") for call in engine.async_calls)
        assert engine.cleanup_calls
    migration_id = store.conn.execute("SELECT migration_id FROM migrations").fetchone()[0]
    assert store.get_migration(migration_id).state == MigrationState.ABORTED
