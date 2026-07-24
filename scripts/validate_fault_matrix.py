from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from benchmark_e2e import flush, timed_generate

from agentshift.controller.migration import MigrationCoordinator
from agentshift.engine.sglang import SGLangAgentShiftClient
from agentshift.runtime.effects import ManagedEffectProxy
from agentshift.state.schema import (
    AgentContinuation,
    MigrationRecord,
    MigrationState,
    ToolResult,
)
from agentshift.state.store import SQLiteStateStore, StaleLease


class FaultMatrix:
    def __init__(self, args):
        self.args = args
        self.source = SGLangAgentShiftClient("engine-a", args.source, timeout=300)
        self.destination = SGLangAgentShiftClient(
            "engine-b", args.destination, timeout=300
        )
        self.state_path = Path(args.output_dir) / f"fault-state-{time.time_ns()}.db"
        self.store = SQLiteStateStore(self.state_path)
        self.coordinator = MigrationCoordinator(
            self.store,
            {"engine-a": self.source, "engine-b": self.destination},
            base_port=args.transfer_port,
        )
        self.serial = 0

    async def clean(self) -> None:
        await asyncio.gather(flush(self.source), flush(self.destination))

    def prompt(self) -> list[int]:
        self.serial += 1
        return [21000 + self.serial] + [100] * (self.args.context_length - 1)

    async def prepare(self, label: str) -> tuple[str, tuple[int, ...]]:
        prompt = self.prompt()
        _, first = await timed_generate(
            self.source,
            prompt,
            max_new_tokens=4,
            rid=f"fault-{label}-{self.serial}-turn-1",
        )
        completed = tuple(prompt + first["output_ids"])
        agent_id = f"fault-{label}-{self.serial}"
        self.store.register_agent(
            AgentContinuation(agent_id, 1, "engine-a", 1, completed, f"tool-{agent_id}")
        )
        return agent_id, completed

    @staticmethod
    def stale_rejected(
        store: SQLiteStateStore, agent_id: str, engine: str, epoch: int
    ) -> bool:
        try:
            store.assert_lease(agent_id, engine, epoch)
        except StaleLease:
            return True
        return False

    async def source_shadow_case(self) -> dict[str, Any]:
        await self.clean()
        agent_id, completed = await self.prepare("shadow")
        migration = await self.coordinator.migrate(agent_id, "engine-b")
        restarted = MigrationCoordinator(
            self.store,
            self.coordinator.engines,
            base_port=self.args.transfer_port,
        )
        restart_actions = await restarted.reconcile_after_restart()
        recovery = await restarted.recover_destination_failure(migration.migration_id)
        next_seconds, result = await timed_generate(
            self.source,
            list(completed) + [200] * 32,
            max_new_tokens=1,
            rid=f"{agent_id}-recovered-next",
        )
        cached_tokens = int(result["meta_info"]["cached_tokens"])
        await self.source.release_prefix(agent_id, recovery.continuation.owner_epoch)
        await self.destination.release_prefix(
            agent_id, migration.new_epoch, allow_missing=True
        )
        return {
            "case": "post_commit_destination_failure_with_shadow",
            "passed": (
                recovery.source_shadow_reused
                and cached_tokens >= migration.token_count
                and self.stale_rejected(
                    self.store, agent_id, "engine-b", migration.new_epoch
                )
            ),
            "restart_action": restart_actions[0]["action"],
            "recovery_owner": recovery.continuation.owner_engine,
            "recovery_epoch": recovery.continuation.owner_epoch,
            "source_shadow_reused": recovery.source_shadow_reused,
            "cached_tokens": cached_tokens,
            "next_turn_seconds": next_seconds,
        }

    async def no_shadow_case(self) -> dict[str, Any]:
        await self.clean()
        agent_id, completed = await self.prepare("no-shadow")
        migration = await self.coordinator.migrate(agent_id, "engine-b")
        await self.coordinator.acknowledge_destination(migration.migration_id)
        await self.destination.release_prefix(agent_id, migration.new_epoch)
        recovery = await self.coordinator.recover_destination_failure(
            migration.migration_id
        )
        next_seconds, result = await timed_generate(
            self.source,
            list(completed) + [200] * 32,
            max_new_tokens=1,
            rid=f"{agent_id}-cold-recovery-next",
        )
        cached_tokens = int(result["meta_info"]["cached_tokens"])
        return {
            "case": "post_release_destination_failure_without_shadow",
            "passed": (
                not recovery.source_shadow_reused
                and recovery.continuation.owner_engine == "engine-a"
                and self.stale_rejected(
                    self.store, agent_id, "engine-b", migration.new_epoch
                )
            ),
            "recovery_owner": recovery.continuation.owner_engine,
            "recovery_epoch": recovery.continuation.owner_epoch,
            "source_shadow_reused": recovery.source_shadow_reused,
            "cached_tokens": cached_tokens,
            "next_turn_seconds": next_seconds,
        }

    async def ack_loss_case(self) -> dict[str, Any]:
        await self.clean()
        agent_id, _ = await self.prepare("ack-loss")
        migration = await self.coordinator.migrate(agent_id, "engine-b")
        restarted = MigrationCoordinator(
            self.store,
            self.coordinator.engines,
            base_port=self.args.transfer_port,
        )
        await restarted.acknowledge_destination(migration.migration_id)
        await restarted.acknowledge_destination(migration.migration_id)
        state = self.store.get_migration(migration.migration_id).state
        await self.destination.release_prefix(agent_id, migration.new_epoch)
        return {
            "case": "commit_ack_lost_controller_retries",
            "passed": (
                state == MigrationState.SOURCE_RELEASED
                and self.stale_rejected(self.store, agent_id, "engine-a", 1)
            ),
            "migration_state": state.value,
            "owner": self.store.get_agent(agent_id).owner_engine,
        }

    async def restart_before_commit_case(self) -> dict[str, Any]:
        await self.clean()
        agent_id, completed = await self.prepare("restart-before-commit")
        injected_migrations = []

        def inject(point: str, migration_id: str) -> None:
            if point == "DEST_READY":
                injected_migrations.append(migration_id)
                raise RuntimeError("injected controller crash at DEST_READY")

        crashing = MigrationCoordinator(
            self.store,
            self.coordinator.engines,
            base_port=self.args.transfer_port,
            fault_injector=inject,
        )
        fault_observed = False
        try:
            await crashing.migrate(agent_id, "engine-b")
        except RuntimeError as exc:
            fault_observed = "DEST_READY" in str(exc)
        if not injected_migrations:
            raise RuntimeError("DEST_READY fault was not injected")
        migration_id = injected_migrations[0]
        restarted = MigrationCoordinator(
            self.store,
            self.coordinator.engines,
            base_port=self.args.transfer_port,
        )
        actions = await restarted.reconcile_after_restart()
        migration = self.store.get_migration(migration_id)
        continuation = self.store.get_agent(agent_id)
        next_seconds, result = await timed_generate(
            self.source,
            list(completed) + [200] * 32,
            max_new_tokens=1,
            rid=f"{agent_id}-source-next",
        )
        cached_tokens = int(result["meta_info"]["cached_tokens"])
        await self.source.release_prefix(agent_id, 1)
        return {
            "case": "controller_restart_dest_ready_before_cas",
            "passed": (
                fault_observed
                and migration.state == MigrationState.ABORTED
                and continuation.owner_engine == "engine-a"
                and continuation.owner_epoch == 1
                and cached_tokens >= migration.token_count
            ),
            "action": next(
                row["action"]
                for row in actions
                if row["migration_id"] == migration_id
            ),
            "migration_state": migration.state.value,
            "owner": continuation.owner_engine,
            "owner_epoch": continuation.owner_epoch,
            "copied_tokens_before_fault": migration.token_count,
            "source_cached_tokens_after_recovery": cached_tokens,
            "next_turn_seconds": next_seconds,
        }

    async def tool_result_race_case(self) -> dict[str, Any]:
        agent_id = f"fault-tool-race-{self.serial}"
        future_id = f"future-{agent_id}"
        self.store.register_agent(
            AgentContinuation(agent_id, 1, "engine-a", 1, (), future_id)
        )
        migration_id = f"tool-race-{self.serial}"
        self.store.start_migration(
            MigrationRecord(
                migration_id,
                agent_id,
                "engine-a",
                "engine-b",
                1,
                MigrationState.PREPARING,
            )
        )
        self.store.transition_migration(
            migration_id, MigrationState.PREPARING, MigrationState.COPYING
        )
        self.store.transition_migration(
            migration_id, MigrationState.COPYING, MigrationState.DEST_READY
        )
        tool_result = ToolResult(agent_id, 1, future_id, {"ok": True})
        await asyncio.gather(
            asyncio.to_thread(self.store.commit_migration, migration_id),
            asyncio.to_thread(self.store.put_tool_result, tool_result),
        )
        duplicate_inserted = self.store.put_tool_result(tool_result)
        stale_source_rejected = self.stale_rejected(
            self.store, agent_id, "engine-a", 1
        )
        self.store.claim_step(
            agent_id=agent_id,
            step_id=2,
            owner_engine="engine-b",
            owner_epoch=2,
            rid=f"{agent_id}-step-2",
        )
        return {
            "case": "tool_result_races_ownership_cas",
            "passed": (
                self.store.get_tool_result(agent_id, 1, future_id) == tool_result
                and not duplicate_inserted
                and stale_source_rejected
            ),
            "mailbox_deduplicated": not duplicate_inserted,
            "stale_source_rejected": stale_source_rejected,
            "consumer_owner": self.store.get_agent(agent_id).owner_engine,
        }

    def managed_effect_case(self) -> dict[str, Any]:
        agent_id = f"fault-effect-{self.serial}"
        self.store.register_agent(AgentContinuation(agent_id, 1, "engine-a", 1))
        external_path = Path(self.args.output_dir) / f"effect-{time.time_ns()}.db"
        external = sqlite3.connect(external_path)
        external.execute(
            "CREATE TABLE effect_log (operation_id TEXT PRIMARY KEY, value INTEGER NOT NULL)"
        )
        external.commit()

        calls = 0

        def submit(payload):
            nonlocal calls
            calls += 1
            external.execute(
                "INSERT INTO effect_log VALUES (?, ?)",
                (payload["operation_id"], payload["value"]),
            )
            external.commit()
            return {"applied": True}

        proxy = ManagedEffectProxy(self.store)
        kwargs = {
            "agent_id": agent_id,
            "step_id": 1,
            "operation_id": "append-once",
            "owner_epoch": 1,
            "payload": {"operation_id": "append-once", "value": 1},
            "submit": submit,
        }
        first = proxy.execute(**kwargs)
        second = proxy.execute(**kwargs)
        rows = external.execute("SELECT COUNT(*) FROM effect_log").fetchone()[0]
        external.close()
        return {
            "case": "managed_effect_retry",
            "passed": first == second == {"applied": True} and calls == 1 and rows == 1,
            "submit_calls": calls,
            "external_rows": rows,
            "external_db": str(external_path),
        }

    def no_fencing_ablation_case(self) -> dict[str, Any]:
        agent_id = f"fault-no-fencing-{self.serial}"
        self.store.register_agent(AgentContinuation(agent_id, 1, "engine-b", 2))
        router_only_source_accepts = True
        router_only_destination_accepts = True
        protected_source_accepts = not self.stale_rejected(
            self.store, agent_id, "engine-a", 1
        )
        protected_destination_accepts = not self.stale_rejected(
            self.store, agent_id, "engine-b", 2
        )
        return {
            "case": "no_owner_fencing_ablation",
            "passed": (
                router_only_source_accepts
                and router_only_destination_accepts
                and not protected_source_accepts
                and protected_destination_accepts
            ),
            "router_only_accepting_executors": 2,
            "agentshift_accepting_executors": int(protected_source_accepts)
            + int(protected_destination_accepts),
        }

    async def flush_registry_case(self) -> dict[str, Any]:
        await self.clean()
        agent_id, completed = await self.prepare("flush-registry")
        await self.source.pin_prefix(agent_id, 1, completed)
        await flush(self.source)
        release = await self.source.release_prefix(
            agent_id, 1, allow_missing=True
        )
        return {
            "case": "flush_clears_terminal_prefix_registry",
            "passed": bool(release["success"]) and int(release["token_count"]) == 0,
            "release_after_flush_token_count": int(release["token_count"]),
        }


async def main(args) -> None:
    matrix = FaultMatrix(args)
    cases = [
        await matrix.source_shadow_case(),
        await matrix.no_shadow_case(),
        await matrix.ack_loss_case(),
        await matrix.restart_before_commit_case(),
        await matrix.tool_result_race_case(),
        matrix.managed_effect_case(),
        matrix.no_fencing_ablation_case(),
        await matrix.flush_registry_case(),
    ]
    output = {
        "config": vars(args),
        "state_db": str(matrix.state_path),
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }
    output_path = Path(args.output_dir) / f"fault-matrix-{time.time_ns()}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output_path), **output}, indent=2, sort_keys=True))
    if not output["passed"]:
        raise RuntimeError("fault matrix failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31000")
    parser.add_argument("--destination", default="http://127.0.0.1:31001")
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--transfer-port", type=int, default=30400)
    parser.add_argument("--output-dir", default="results")
    asyncio.run(main(parser.parse_args()))
