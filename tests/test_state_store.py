from pathlib import Path

import pytest

from agentshift.runtime.effects import ManagedEffectProxy
from agentshift.state.schema import AgentContinuation, MigrationRecord, MigrationState, ToolResult
from agentshift.state.store import SQLiteStateStore, StaleLease, StateConflict


def make_store(tmp_path: Path) -> SQLiteStateStore:
    store = SQLiteStateStore(tmp_path / "state.db")
    store.register_agent(AgentContinuation("a1", 3, "engine-a", 7, (1, 2, 3)))
    return store


def test_migration_cas_fences_old_owner(tmp_path: Path):
    store = make_store(tmp_path)
    migration = MigrationRecord(
        "m1", "a1", "engine-a", "engine-b", 7, MigrationState.PREPARING
    )
    store.start_migration(migration)
    store.transition_migration("m1", MigrationState.PREPARING, MigrationState.COPYING)
    store.transition_migration("m1", MigrationState.COPYING, MigrationState.DEST_READY)
    continuation = store.commit_migration("m1")
    assert (continuation.owner_engine, continuation.owner_epoch) == ("engine-b", 8)
    with pytest.raises(StaleLease):
        store.assert_lease("a1", "engine-a", 7)


def test_recover_committed_migration_fences_failed_destination(tmp_path: Path):
    store = make_store(tmp_path)
    store.start_migration(
        MigrationRecord("m1", "a1", "engine-a", "engine-b", 7, MigrationState.PREPARING)
    )
    store.transition_migration("m1", MigrationState.PREPARING, MigrationState.COPYING)
    store.transition_migration("m1", MigrationState.COPYING, MigrationState.DEST_READY)
    store.commit_migration("m1")
    continuation = store.recover_committed_migration("m1", "engine-a")
    assert (continuation.owner_engine, continuation.owner_epoch) == ("engine-a", 9)
    with pytest.raises(StaleLease):
        store.assert_lease("a1", "engine-b", 8)


def test_only_one_live_migration(tmp_path: Path):
    store = make_store(tmp_path)
    store.start_migration(
        MigrationRecord("m1", "a1", "engine-a", "engine-b", 7, MigrationState.PREPARING)
    )
    with pytest.raises(StateConflict):
        store.start_migration(
            MigrationRecord("m2", "a1", "engine-a", "engine-c", 7, MigrationState.PREPARING)
        )


def test_tool_mailbox_is_idempotent(tmp_path: Path):
    store = make_store(tmp_path)
    result = ToolResult("a1", 3, "future-1", {"ok": True})
    assert store.put_tool_result(result)
    assert not store.put_tool_result(result)
    assert store.get_tool_result("a1", 3, "future-1") == result


def test_managed_effect_is_not_resubmitted(tmp_path: Path):
    store = make_store(tmp_path)
    proxy = ManagedEffectProxy(store)
    calls = []

    def submit(payload):
        calls.append(payload)
        return {"message_id": "42"}

    kwargs = dict(
        agent_id="a1",
        step_id=3,
        operation_id="send-email",
        owner_epoch=7,
        payload={"to": "test@example.com"},
        submit=submit,
    )
    assert proxy.execute(**kwargs) == {"message_id": "42"}
    assert proxy.execute(**kwargs) == {"message_id": "42"}
    assert len(calls) == 1


def test_managed_effect_updates_external_sqlite_once(tmp_path: Path):
    import sqlite3

    store = make_store(tmp_path)
    proxy = ManagedEffectProxy(store)
    external = sqlite3.connect(tmp_path / "external.db")
    external.execute("CREATE TABLE counter (value INTEGER NOT NULL)")
    external.execute("INSERT INTO counter VALUES (0)")
    external.commit()

    def submit(payload):
        external.execute("UPDATE counter SET value=value+?", (payload["delta"],))
        external.commit()
        return {"applied": True}

    kwargs = dict(
        agent_id="a1",
        step_id=3,
        operation_id="increment-counter",
        owner_epoch=7,
        payload={"delta": 1},
        submit=submit,
    )
    assert proxy.execute(**kwargs) == {"applied": True}
    assert proxy.execute(**kwargs) == {"applied": True}
    assert external.execute("SELECT value FROM counter").fetchone()[0] == 1
