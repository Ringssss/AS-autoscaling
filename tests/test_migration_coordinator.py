import asyncio

import pytest

from agentshift.controller.migration import MigrationCoordinator
from agentshift.state.schema import AgentContinuation, MigrationRecord, MigrationState
from agentshift.state.store import SQLiteStateStore


class FakeEngine:
    def __init__(
        self,
        engine_id: str,
        fail_destination: bool = False,
        transfer_tracker: dict | None = None,
    ):
        self.engine_id = engine_id
        self.fail_destination = fail_destination
        self.releases = []
        self.rebinds = []
        self.cleanups = []
        self.active_releases = 0
        self.max_active_releases = 0
        self.release_options = []
        self.transfer_tracker = transfer_tracker
        self.transfer_calls = []
        self.generate_calls = []

    async def pin_prefix(self, agent_id, owner_epoch, token_ids):
        return {"success": True, "token_count": len(token_ids)}

    async def init_transfer_group(self, **kwargs):
        return {"success": True}

    async def transfer_prefix(self, **kwargs):
        self.transfer_calls.append(kwargs)
        if kwargs["role"] == "destination" and self.fail_destination:
            raise RuntimeError("injected destination failure")
        if self.transfer_tracker is not None and kwargs["role"] == "source":
            self.transfer_tracker["active"] += 1
            self.transfer_tracker["maximum"] = max(
                self.transfer_tracker["maximum"], self.transfer_tracker["active"]
            )
            await asyncio.sleep(0.01)
            self.transfer_tracker["active"] -= 1
        return {
            "success": True,
            "bytes_transferred": 4096,
            "state": "QUEUED" if kwargs.get("async_transfer") else "COMPLETE",
        }

    async def post(self, path, payload, require_success=True):
        self.generate_calls.append((path, payload))
        return {"success": True, "output_ids": [42]}

    async def transfer_status(self, migration_id):
        return {
            "success": True,
            "migration_id": migration_id,
            "state": "COMPLETE",
            "bytes_transferred": 4096,
            "transfer_seconds": 0.001,
            "queue_seconds": 0.0001,
        }

    async def cleanup_transfer(self, migration_id):
        self.cleanups.append(migration_id)
        return {"success": True}

    async def release_prefix(self, agent_id, owner_epoch, **kwargs):
        self.active_releases += 1
        self.max_active_releases = max(
            self.max_active_releases, self.active_releases
        )
        await asyncio.sleep(0.001)
        self.releases.append((agent_id, owner_epoch))
        self.release_options.append(kwargs)
        self.active_releases -= 1
        return {"success": True}

    async def rebind_prefix(
        self, agent_id, expected_owner_epoch, new_owner_epoch
    ):
        self.rebinds.append((agent_id, expected_owner_epoch, new_owner_epoch))
        return {"success": True}


def make_coordinator(tmp_path, fail_destination=False, fault_injector=None):
    store = SQLiteStateStore(tmp_path / "state.db")
    store.register_agent(AgentContinuation("a1", 1, "a", 3, (1, 2, 3, 4)))
    source = FakeEngine("a")
    destination = FakeEngine("b", fail_destination=fail_destination)
    coordinator = MigrationCoordinator(
        store,
        {"a": source, "b": destination},
        base_port=29900,
        fault_injector=fault_injector,
    )
    return store, source, destination, coordinator


def test_destination_failure_aborts_without_changing_owner(tmp_path):
    store, _, _, coordinator = make_coordinator(tmp_path, fail_destination=True)
    with pytest.raises(RuntimeError, match="injected destination failure"):
        asyncio.run(coordinator.migrate("a1", "b"))
    agent = store.get_agent("a1")
    assert (agent.owner_engine, agent.owner_epoch) == ("a", 3)
    row = store.conn.execute("SELECT state FROM migrations").fetchone()
    assert row["state"] == MigrationState.ABORTED.value


def test_source_released_only_after_destination_ack(tmp_path):
    store, source, _, coordinator = make_coordinator(tmp_path)
    result = asyncio.run(coordinator.migrate("a1", "b"))
    assert source.releases == []
    assert store.get_migration(result.migration_id).state == MigrationState.COMMITTED
    asyncio.run(coordinator.acknowledge_destination(result.migration_id))
    asyncio.run(coordinator.acknowledge_destination(result.migration_id))
    assert source.releases == [("a1", 3)]
    assert (
        store.get_migration(result.migration_id).state
        == MigrationState.SOURCE_RELEASED
    )


def test_finalize_unpins_destination_but_keeps_cache(tmp_path):
    store, source, destination, coordinator = make_coordinator(tmp_path)
    result = asyncio.run(coordinator.migrate("a1", "b"))
    asyncio.run(coordinator.finalize_destination(result.migration_id))
    assert source.releases == [("a1", 3)]
    assert destination.releases == [("a1", 4)]
    assert destination.release_options == [
        {"evict_after_release": False, "allow_missing": True}
    ]
    assert store.get_migration(result.migration_id).state == MigrationState.SOURCE_RELEASED


def test_fault_at_dest_ready_preserves_source_owner(tmp_path):
    injected = []

    def inject(point, migration_id):
        if point == "DEST_READY":
            injected.append(migration_id)
            raise RuntimeError("injected controller crash at DEST_READY")

    store, _, _, coordinator = make_coordinator(
        tmp_path, fault_injector=inject
    )
    with pytest.raises(RuntimeError, match="DEST_READY"):
        asyncio.run(coordinator.migrate("a1", "b"))
    assert len(injected) == 1
    assert store.get_migration(injected[0]).state == MigrationState.DEST_READY
    assert (store.get_agent("a1").owner_engine, store.get_agent("a1").owner_epoch) == (
        "a",
        3,
    )


def test_fault_after_commit_preserves_destination_owner(tmp_path):
    injected = []

    def inject(point, migration_id):
        if point == "COMMITTED":
            injected.append(migration_id)
            raise RuntimeError("injected controller crash after commit")

    store, _, _, coordinator = make_coordinator(
        tmp_path, fault_injector=inject
    )
    with pytest.raises(RuntimeError, match="after commit"):
        asyncio.run(coordinator.migrate("a1", "b"))
    assert len(injected) == 1
    assert store.get_migration(injected[0]).state == MigrationState.COMMITTED
    assert (store.get_agent("a1").owner_engine, store.get_agent("a1").owner_epoch) == (
        "b",
        4,
    )


def test_concurrent_acks_serialize_source_releases(tmp_path):
    store, source, _, coordinator = make_coordinator(tmp_path)
    store.register_agent(AgentContinuation("a2", 1, "a", 3, (5, 6, 7, 8)))
    first = asyncio.run(coordinator.migrate("a1", "b"))
    second = asyncio.run(coordinator.migrate("a2", "b"))

    async def acknowledge_both():
        await asyncio.gather(
            coordinator.acknowledge_destination(first.migration_id),
            coordinator.acknowledge_destination(second.migration_id),
        )

    asyncio.run(acknowledge_both())
    assert source.max_active_releases == 1
    assert sorted(source.releases) == [("a1", 3), ("a2", 3)]


def test_disjoint_engine_pairs_migrate_concurrently(tmp_path):
    tracker = {"active": 0, "maximum": 0}
    store = SQLiteStateStore(tmp_path / "parallel.db")
    store.register_agent(AgentContinuation("a1", 1, "a", 1, (1, 2, 3)))
    store.register_agent(AgentContinuation("c1", 1, "c", 1, (4, 5, 6)))
    engines = {
        engine_id: FakeEngine(engine_id, transfer_tracker=tracker)
        for engine_id in ("a", "b", "c", "d")
    }
    coordinator = MigrationCoordinator(store, engines, base_port=29900)

    async def migrate_both():
        return await asyncio.gather(
            coordinator.migrate("a1", "b"),
            coordinator.migrate("c1", "d"),
        )

    asyncio.run(migrate_both())
    assert tracker["maximum"] == 2


def test_shared_engine_serializes_concurrent_migrations(tmp_path):
    tracker = {"active": 0, "maximum": 0}
    store = SQLiteStateStore(tmp_path / "serialized.db")
    store.register_agent(AgentContinuation("a1", 1, "a", 1, (1, 2, 3)))
    store.register_agent(AgentContinuation("a2", 1, "a", 1, (4, 5, 6)))
    engines = {
        engine_id: FakeEngine(engine_id, transfer_tracker=tracker)
        for engine_id in ("a", "b", "c")
    }
    coordinator = MigrationCoordinator(store, engines, base_port=29900)

    async def migrate_both():
        return await asyncio.gather(
            coordinator.migrate("a1", "b"),
            coordinator.migrate("a2", "c"),
        )

    asyncio.run(migrate_both())
    assert tracker["maximum"] == 1


def test_transfer_ports_are_stable_across_pair_initialization_order(tmp_path):
    engines = {engine_id: FakeEngine(engine_id) for engine_id in ("a", "b", "c", "d")}
    first = MigrationCoordinator(
        SQLiteStateStore(tmp_path / "ports-first.db"), engines, base_port=29900, tp_size=2
    )
    second = MigrationCoordinator(
        SQLiteStateStore(tmp_path / "ports-second.db"), engines, base_port=29900, tp_size=2
    )

    async def first_order():
        ab = await first.initialize_transfer_pair("a", "b")
        cd = await first.initialize_transfer_pair("c", "d")
        return ab, cd

    async def second_order():
        cd = await second.initialize_transfer_pair("c", "d")
        ab = await second.initialize_transfer_pair("a", "b")
        return ab, cd

    assert asyncio.run(first_order()) == asyncio.run(second_order())


def test_sync_transfer_path_remains_available(tmp_path):
    store, _, _, coordinator = make_coordinator(tmp_path)
    coordinator.async_transfer = False
    result = asyncio.run(coordinator.migrate("a1", "b"))
    assert result.bytes_transferred == 4096
    assert store.get_agent("a1").owner_engine == "b"


def test_progressive_migration_is_opt_in_and_returns_after_commit(tmp_path):
    store, source, destination, coordinator = make_coordinator(tmp_path)
    result = asyncio.run(
        coordinator.migrate_and_generate_progressive(
            "a1",
            "b",
            token_ids=(1, 2, 3, 4, 9),
            max_new_tokens=1,
            rid="progressive-request",
            layer_group_size=2,
        )
    )
    assert result.generation["output_ids"] == [42]
    assert store.get_migration(result.migration.migration_id).state == (
        MigrationState.COMMITTED
    )
    assert store.get_agent("a1").owner_engine == "b"
    assert destination.transfer_calls[0]["progressive"] is True
    assert destination.transfer_calls[0]["layer_group_size"] == 2
    assert source.transfer_calls[0].get("progressive", False) is False
    assert destination.generate_calls[0][1][
        "agentshift_progressive_migration_id"
    ] == result.migration.migration_id


def test_progressive_migration_rejects_nonextending_continuation(tmp_path):
    store, _, _, coordinator = make_coordinator(tmp_path)
    with pytest.raises(RuntimeError, match="does not extend"):
        asyncio.run(
            coordinator.migrate_and_generate_progressive(
                "a1",
                "b",
                token_ids=(1, 2, 99, 4, 9),
                max_new_tokens=1,
                rid="bad-progressive-request",
            )
        )
    assert store.get_agent("a1").owner_engine == "a"


def test_progressive_awaitable_failure_cleans_transfer_reservations(tmp_path):
    store, source, destination, coordinator = make_coordinator(tmp_path)

    async def invalid_tokens():
        await asyncio.sleep(0)
        return (1, 2, 99, 4, 9)

    with pytest.raises(RuntimeError, match="does not extend"):
        asyncio.run(
            coordinator.migrate_and_generate_progressive(
                "a1",
                "b",
                token_ids=invalid_tokens(),
                max_new_tokens=1,
                rid="bad-progressive-request",
            )
        )
    assert len(source.cleanups) == 1
    assert destination.cleanups == source.cleanups
    assert store.get_agent("a1").owner_engine == "a"


def test_post_commit_failure_recovers_to_source_shadow(tmp_path):
    store, source, _, coordinator = make_coordinator(tmp_path)
    result = asyncio.run(coordinator.migrate("a1", "b"))
    restarted = MigrationCoordinator(
        store, coordinator.engines, base_port=coordinator.base_port
    )
    continuation = asyncio.run(restarted.recover_committed(result.migration_id))
    assert (continuation.owner_engine, continuation.owner_epoch) == ("a", 5)
    assert source.rebinds == [("a1", 3, 5)]
    assert store.get_migration(result.migration_id).state == MigrationState.RECOVERED
    with pytest.raises(RuntimeError, match="has not committed"):
        asyncio.run(coordinator.acknowledge_destination(result.migration_id))


def test_ack_loss_is_reconciled_by_restarted_controller(tmp_path):
    store, source, _, coordinator = make_coordinator(tmp_path)
    result = asyncio.run(coordinator.migrate("a1", "b"))
    restarted = MigrationCoordinator(store, coordinator.engines, base_port=29900)
    asyncio.run(restarted.acknowledge_destination(result.migration_id))
    assert source.releases == [("a1", 3)]
    assert store.get_migration(result.migration_id).state == MigrationState.SOURCE_RELEASED


def test_failure_after_source_release_recovers_cold(tmp_path):
    store, source, _, coordinator = make_coordinator(tmp_path)
    result = asyncio.run(coordinator.migrate("a1", "b"))
    asyncio.run(coordinator.acknowledge_destination(result.migration_id))
    recovery = asyncio.run(
        coordinator.recover_destination_failure(result.migration_id)
    )
    assert (recovery.continuation.owner_engine, recovery.continuation.owner_epoch) == (
        "a",
        5,
    )
    assert not recovery.source_shadow_reused
    assert source.rebinds == []


def test_restart_aborts_dest_ready_without_changing_owner(tmp_path):
    store, _, destination, coordinator = make_coordinator(tmp_path)
    store.start_migration(
        MigrationRecord("restart-m1", "a1", "a", "b", 3, MigrationState.PREPARING)
    )
    store.transition_migration(
        "restart-m1", MigrationState.PREPARING, MigrationState.COPYING
    )
    store.transition_migration(
        "restart-m1", MigrationState.COPYING, MigrationState.DEST_READY
    )
    actions = asyncio.run(coordinator.reconcile_after_restart())
    assert actions == [
        {"migration_id": "restart-m1", "action": "abort-to-source"}
    ]
    assert store.get_agent("a1").owner_engine == "a"
    assert store.get_migration("restart-m1").state == MigrationState.ABORTED
    assert destination.releases == [("a1", 4)]


def test_restart_preserves_committed_owner_and_shadow(tmp_path):
    store, source, _, coordinator = make_coordinator(tmp_path)
    result = asyncio.run(coordinator.migrate("a1", "b"))
    actions = asyncio.run(coordinator.reconcile_after_restart())
    assert actions == [
        {"migration_id": result.migration_id, "action": "retain-source-shadow"}
    ]
    assert (store.get_agent("a1").owner_engine, store.get_agent("a1").owner_epoch) == (
        "b",
        4,
    )
    assert source.releases == []
