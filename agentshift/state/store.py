from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from agentshift.state.schema import (
    AgentContinuation,
    EffectRecord,
    EffectStatus,
    MigrationRecord,
    MigrationState,
    StepStatus,
    ToolResult,
)


class StateConflict(RuntimeError):
    pass


class StaleLease(StateConflict):
    pass


class SQLiteStateStore:
    """Durable authority for progress, ownership, tools, and external effects."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        self._local = threading.local()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def _initialize(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS agents (
                    agent_id TEXT PRIMARY KEY,
                    committed_step INTEGER NOT NULL,
                    owner_engine TEXT NOT NULL,
                    owner_epoch INTEGER NOT NULL,
                    token_ids TEXT NOT NULL,
                    pending_tool_future TEXT,
                    workspace_ref TEXT,
                    stream_offset INTEGER NOT NULL,
                    metadata TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS migrations (
                    migration_id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL REFERENCES agents(agent_id),
                    source_engine TEXT NOT NULL,
                    destination_engine TEXT NOT NULL,
                    source_epoch INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    token_count INTEGER NOT NULL DEFAULT 0,
                    bytes_transferred INTEGER NOT NULL DEFAULT 0,
                    transfer_seconds REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS mailbox (
                    agent_id TEXT NOT NULL,
                    step_id INTEGER NOT NULL,
                    future_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (agent_id, step_id, future_id)
                );
                CREATE TABLE IF NOT EXISTS effects (
                    agent_id TEXT NOT NULL,
                    step_id INTEGER NOT NULL,
                    operation_id TEXT NOT NULL,
                    owner_epoch INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    result TEXT,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (agent_id, step_id, operation_id)
                );
                CREATE TABLE IF NOT EXISTS step_claims (
                    agent_id TEXT NOT NULL,
                    step_id INTEGER NOT NULL,
                    owner_epoch INTEGER NOT NULL,
                    rid TEXT NOT NULL,
                    status TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (agent_id, step_id)
                );
                DROP INDEX IF EXISTS one_live_migration_per_agent;
                CREATE UNIQUE INDEX one_live_migration_per_agent
                ON migrations(agent_id)
                WHERE state NOT IN ('SOURCE_RELEASED', 'RECOVERED', 'ABORTED');
                """
            )
        finally:
            conn.close()

    def register_agent(self, continuation: AgentContinuation) -> None:
        with self.transaction() as conn:
            try:
                conn.execute(
                    """INSERT INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        continuation.agent_id,
                        continuation.committed_step,
                        continuation.owner_engine,
                        continuation.owner_epoch,
                        json.dumps(continuation.token_ids),
                        continuation.pending_tool_future,
                        continuation.workspace_ref,
                        continuation.stream_offset,
                        json.dumps(continuation.metadata, sort_keys=True),
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StateConflict(f"agent already exists: {continuation.agent_id}") from exc

    def get_agent(self, agent_id: str) -> AgentContinuation:
        row = self.conn.execute(
            "SELECT * FROM agents WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if row is None:
            raise KeyError(agent_id)
        return AgentContinuation(
            agent_id=row["agent_id"],
            committed_step=row["committed_step"],
            owner_engine=row["owner_engine"],
            owner_epoch=row["owner_epoch"],
            token_ids=tuple(json.loads(row["token_ids"])),
            pending_tool_future=row["pending_tool_future"],
            workspace_ref=row["workspace_ref"],
            stream_offset=row["stream_offset"],
            metadata=json.loads(row["metadata"]),
        )

    def assert_lease(self, agent_id: str, owner_engine: str, owner_epoch: int) -> None:
        current = self.get_agent(agent_id)
        if (current.owner_engine, current.owner_epoch) != (owner_engine, owner_epoch):
            raise StaleLease(
                f"stale lease for {agent_id}: got {owner_engine}@{owner_epoch}, "
                f"current is {current.owner_engine}@{current.owner_epoch}"
            )

    def commit_step(self, continuation: AgentContinuation, expected_epoch: int) -> None:
        with self.transaction() as conn:
            current = conn.execute(
                "SELECT * FROM agents WHERE agent_id = ?", (continuation.agent_id,)
            ).fetchone()
            if current is None:
                raise KeyError(continuation.agent_id)
            if current["owner_epoch"] != expected_epoch:
                raise StaleLease(continuation.agent_id)
            if continuation.committed_step <= current["committed_step"]:
                raise StateConflict("committed_step must advance monotonically")
            conn.execute(
                """UPDATE agents SET committed_step=?, token_ids=?,
                   pending_tool_future=?, workspace_ref=?, stream_offset=?, metadata=?,
                   updated_at=? WHERE agent_id=?""",
                (
                    continuation.committed_step,
                    json.dumps(continuation.token_ids),
                    continuation.pending_tool_future,
                    continuation.workspace_ref,
                    continuation.stream_offset,
                    json.dumps(continuation.metadata, sort_keys=True),
                    time.time(),
                    continuation.agent_id,
                ),
            )

    def claim_step(
        self,
        *,
        agent_id: str,
        step_id: int,
        owner_engine: str,
        owner_epoch: int,
        rid: str,
    ) -> None:
        with self.transaction() as conn:
            agent = conn.execute(
                "SELECT * FROM agents WHERE agent_id=?", (agent_id,)
            ).fetchone()
            if agent is None:
                raise KeyError(agent_id)
            if (agent["owner_engine"], agent["owner_epoch"]) != (
                owner_engine,
                owner_epoch,
            ):
                raise StaleLease(agent_id)
            if step_id != agent["committed_step"] + 1:
                raise StateConflict(
                    f"step {step_id} cannot follow committed step {agent['committed_step']}"
                )
            existing = conn.execute(
                "SELECT * FROM step_claims WHERE agent_id=? AND step_id=?",
                (agent_id, step_id),
            ).fetchone()
            if existing is not None and existing["status"] == StepStatus.FAILED.value:
                conn.execute(
                    """UPDATE step_claims SET owner_epoch=?, rid=?, status=?, updated_at=?
                       WHERE agent_id=? AND step_id=?""",
                    (
                        owner_epoch,
                        rid,
                        StepStatus.CLAIMED.value,
                        time.time(),
                        agent_id,
                        step_id,
                    ),
                )
                return
            try:
                conn.execute(
                    "INSERT INTO step_claims VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        agent_id,
                        step_id,
                        owner_epoch,
                        rid,
                        StepStatus.CLAIMED.value,
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StateConflict(f"step is already claimed: {agent_id}/{step_id}") from exc

    def commit_claimed_step(
        self, continuation: AgentContinuation, *, expected_epoch: int, rid: str
    ) -> None:
        with self.transaction() as conn:
            claim = conn.execute(
                "SELECT * FROM step_claims WHERE agent_id=? AND step_id=?",
                (continuation.agent_id, continuation.committed_step),
            ).fetchone()
            agent = conn.execute(
                "SELECT * FROM agents WHERE agent_id=?", (continuation.agent_id,)
            ).fetchone()
            if claim is None or claim["status"] != StepStatus.CLAIMED.value:
                raise StateConflict("step has no live claim")
            if claim["rid"] != rid or claim["owner_epoch"] != expected_epoch:
                raise StaleLease(continuation.agent_id)
            if agent["owner_epoch"] != expected_epoch:
                raise StaleLease(continuation.agent_id)
            if continuation.committed_step != agent["committed_step"] + 1:
                raise StateConflict("committed_step must advance by one")
            conn.execute(
                """UPDATE agents SET committed_step=?, token_ids=?,
                   pending_tool_future=?, workspace_ref=?, stream_offset=?, metadata=?,
                   updated_at=? WHERE agent_id=?""",
                (
                    continuation.committed_step,
                    json.dumps(continuation.token_ids),
                    continuation.pending_tool_future,
                    continuation.workspace_ref,
                    continuation.stream_offset,
                    json.dumps(continuation.metadata, sort_keys=True),
                    time.time(),
                    continuation.agent_id,
                ),
            )
            conn.execute(
                """UPDATE step_claims SET status=?, updated_at=?
                   WHERE agent_id=? AND step_id=?""",
                (
                    StepStatus.COMPLETED.value,
                    time.time(),
                    continuation.agent_id,
                    continuation.committed_step,
                ),
            )

    def fail_claimed_step(
        self, agent_id: str, step_id: int, rid: str, *, outcome_unknown: bool
    ) -> None:
        target = StepStatus.UNKNOWN if outcome_unknown else StepStatus.FAILED
        with self.transaction() as conn:
            changed = conn.execute(
                """UPDATE step_claims SET status=?, updated_at=?
                   WHERE agent_id=? AND step_id=? AND rid=? AND status=?""",
                (
                    target.value,
                    time.time(),
                    agent_id,
                    step_id,
                    rid,
                    StepStatus.CLAIMED.value,
                ),
            ).rowcount
            if changed != 1:
                raise StateConflict("step claim is no longer active")

    def start_migration(self, record: MigrationRecord) -> None:
        self.assert_lease(record.agent_id, record.source_engine, record.source_epoch)
        with self.transaction() as conn:
            try:
                conn.execute(
                    """INSERT INTO migrations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.migration_id,
                        record.agent_id,
                        record.source_engine,
                        record.destination_engine,
                        record.source_epoch,
                        record.state.value,
                        record.token_count,
                        record.bytes_transferred,
                        record.transfer_seconds,
                        record.error,
                        time.time(),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise StateConflict(
                    f"agent already has a live migration: {record.agent_id}"
                ) from exc

    def transition_migration(
        self,
        migration_id: str,
        expected: MigrationState,
        target: MigrationState,
        *,
        token_count: int | None = None,
        bytes_transferred: int | None = None,
        transfer_seconds: float | None = None,
        error: str | None = None,
    ) -> None:
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT state FROM migrations WHERE migration_id=?", (migration_id,)
            ).fetchone()
            if row is None:
                raise KeyError(migration_id)
            if row["state"] != expected.value:
                raise StateConflict(
                    f"migration {migration_id} is {row['state']}, expected {expected.value}"
                )
            fields = ["state=?", "updated_at=?", "error=?"]
            values: list[object] = [target.value, time.time(), error]
            for name, value in (
                ("token_count", token_count),
                ("bytes_transferred", bytes_transferred),
                ("transfer_seconds", transfer_seconds),
            ):
                if value is not None:
                    fields.append(f"{name}=?")
                    values.append(value)
            values.append(migration_id)
            conn.execute(
                f"UPDATE migrations SET {', '.join(fields)} WHERE migration_id=?", values
            )

    def commit_migration(self, migration_id: str) -> AgentContinuation:
        with self.transaction() as conn:
            migration = conn.execute(
                "SELECT * FROM migrations WHERE migration_id=?", (migration_id,)
            ).fetchone()
            if migration is None:
                raise KeyError(migration_id)
            if migration["state"] != MigrationState.DEST_READY.value:
                raise StateConflict("destination is not ready")
            agent = conn.execute(
                "SELECT * FROM agents WHERE agent_id=?", (migration["agent_id"],)
            ).fetchone()
            if (
                agent["owner_engine"] != migration["source_engine"]
                or agent["owner_epoch"] != migration["source_epoch"]
            ):
                raise StaleLease(migration["agent_id"])
            new_epoch = migration["source_epoch"] + 1
            conn.execute(
                "UPDATE agents SET owner_engine=?, owner_epoch=?, updated_at=? WHERE agent_id=?",
                (
                    migration["destination_engine"],
                    new_epoch,
                    time.time(),
                    migration["agent_id"],
                ),
            )
            conn.execute(
                "UPDATE migrations SET state=?, updated_at=? WHERE migration_id=?",
                (MigrationState.COMMITTED.value, time.time(), migration_id),
            )
        return self.get_agent(migration["agent_id"])

    def get_migration(self, migration_id: str) -> MigrationRecord:
        row = self.conn.execute(
            "SELECT * FROM migrations WHERE migration_id=?", (migration_id,)
        ).fetchone()
        if row is None:
            raise KeyError(migration_id)
        return MigrationRecord(
            migration_id=row["migration_id"],
            agent_id=row["agent_id"],
            source_engine=row["source_engine"],
            destination_engine=row["destination_engine"],
            source_epoch=row["source_epoch"],
            state=MigrationState(row["state"]),
            token_count=row["token_count"],
            bytes_transferred=row["bytes_transferred"],
            transfer_seconds=row["transfer_seconds"],
            error=row["error"],
        )

    def list_migrations(
        self, states: tuple[MigrationState, ...] | None = None
    ) -> list[MigrationRecord]:
        if states:
            placeholders = ", ".join("?" for _ in states)
            rows = self.conn.execute(
                f"SELECT migration_id FROM migrations WHERE state IN ({placeholders}) "
                "ORDER BY updated_at, migration_id",
                tuple(state.value for state in states),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT migration_id FROM migrations ORDER BY updated_at, migration_id"
            ).fetchall()
        return [self.get_migration(row["migration_id"]) for row in rows]

    def recover_committed_migration(
        self, migration_id: str, recovery_engine: str
    ) -> AgentContinuation:
        with self.transaction() as conn:
            migration = conn.execute(
                "SELECT * FROM migrations WHERE migration_id=?", (migration_id,)
            ).fetchone()
            if migration is None:
                raise KeyError(migration_id)
            if migration["state"] == MigrationState.RECOVERED.value:
                return self.get_agent(migration["agent_id"])
            if migration["state"] not in (
                MigrationState.COMMITTED.value,
                MigrationState.SOURCE_RELEASED.value,
            ):
                raise StateConflict(
                    "only a committed or source-released migration can be recovered"
                )
            agent = conn.execute(
                "SELECT * FROM agents WHERE agent_id=?", (migration["agent_id"],)
            ).fetchone()
            destination_epoch = migration["source_epoch"] + 1
            if (agent["owner_engine"], agent["owner_epoch"]) != (
                migration["destination_engine"],
                destination_epoch,
            ):
                raise StaleLease(migration["agent_id"])
            recovery_epoch = destination_epoch + 1
            conn.execute(
                "UPDATE agents SET owner_engine=?, owner_epoch=?, updated_at=? WHERE agent_id=?",
                (
                    recovery_engine,
                    recovery_epoch,
                    time.time(),
                    migration["agent_id"],
                ),
            )
            conn.execute(
                "UPDATE migrations SET state=?, updated_at=? WHERE migration_id=?",
                (MigrationState.RECOVERED.value, time.time(), migration_id),
            )
        return self.get_agent(migration["agent_id"])

    def put_tool_result(self, result: ToolResult) -> bool:
        with self.transaction() as conn:
            cursor = conn.execute(
                """INSERT OR IGNORE INTO mailbox VALUES (?, ?, ?, ?, ?)""",
                (
                    result.agent_id,
                    result.step_id,
                    result.future_id,
                    json.dumps(result.payload, sort_keys=True),
                    time.time(),
                ),
            )
            return cursor.rowcount == 1

    def get_tool_result(self, agent_id: str, step_id: int, future_id: str) -> ToolResult | None:
        row = self.conn.execute(
            """SELECT * FROM mailbox WHERE agent_id=? AND step_id=? AND future_id=?""",
            (agent_id, step_id, future_id),
        ).fetchone()
        if row is None:
            return None
        return ToolResult(agent_id, step_id, future_id, json.loads(row["payload"]))

    def prepare_effect(self, record: EffectRecord) -> EffectRecord:
        owner_engine = self.get_agent(record.agent_id).owner_engine
        self.assert_lease(record.agent_id, owner_engine, record.owner_epoch)
        with self.transaction() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO effects VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.agent_id,
                    record.step_id,
                    record.operation_id,
                    record.owner_epoch,
                    EffectStatus.PREPARED.value,
                    json.dumps(record.payload, sort_keys=True),
                    None,
                    time.time(),
                ),
            )
        return self.get_effect(record.agent_id, record.step_id, record.operation_id)

    def transition_effect(
        self,
        agent_id: str,
        step_id: int,
        operation_id: str,
        expected: EffectStatus,
        target: EffectStatus,
        owner_epoch: int,
        result: dict | None = None,
    ) -> EffectRecord:
        current = self.get_effect(agent_id, step_id, operation_id)
        if current.status != expected:
            raise StateConflict(f"effect is {current.status}, expected {expected}")
        self.assert_lease(agent_id, self.get_agent(agent_id).owner_engine, owner_epoch)
        with self.transaction() as conn:
            changed = conn.execute(
                """UPDATE effects SET status=?, owner_epoch=?, result=?, updated_at=?
                   WHERE agent_id=? AND step_id=? AND operation_id=? AND status=?""",
                (
                    target.value,
                    owner_epoch,
                    json.dumps(result, sort_keys=True) if result is not None else None,
                    time.time(),
                    agent_id,
                    step_id,
                    operation_id,
                    expected.value,
                ),
            ).rowcount
            if changed != 1:
                raise StateConflict("effect transition raced")
        return self.get_effect(agent_id, step_id, operation_id)

    def get_effect(self, agent_id: str, step_id: int, operation_id: str) -> EffectRecord:
        row = self.conn.execute(
            """SELECT * FROM effects WHERE agent_id=? AND step_id=? AND operation_id=?""",
            (agent_id, step_id, operation_id),
        ).fetchone()
        if row is None:
            raise KeyError((agent_id, step_id, operation_id))
        return EffectRecord(
            agent_id=row["agent_id"],
            step_id=row["step_id"],
            operation_id=row["operation_id"],
            owner_epoch=row["owner_epoch"],
            status=EffectStatus(row["status"]),
            payload=json.loads(row["payload"]),
            result=json.loads(row["result"]) if row["result"] else None,
        )
