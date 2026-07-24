from __future__ import annotations

from dataclasses import replace

from agentshift.engine.sglang import SGLangAgentShiftClient, generate
from agentshift.state.store import SQLiteStateStore


class ManagedAgentExecutor:
    """Lease-fenced ingress for LLM turns managed by AgentShift."""

    def __init__(
        self,
        store: SQLiteStateStore,
        engines: dict[str, SGLangAgentShiftClient],
    ):
        self.store = store
        self.engines = engines

    async def run_turn(
        self,
        *,
        agent_id: str,
        owner_engine: str,
        owner_epoch: int,
        step_id: int,
        input_ids: list[int],
        max_new_tokens: int,
        rid: str,
        pending_tool_future: str | None = None,
    ) -> dict:
        self.store.claim_step(
            agent_id=agent_id,
            step_id=step_id,
            owner_engine=owner_engine,
            owner_epoch=owner_epoch,
            rid=rid,
        )
        try:
            result = await generate(
                self.engines[owner_engine],
                input_ids,
                max_new_tokens=max_new_tokens,
                rid=rid,
            )
        except Exception:
            self.store.fail_claimed_step(
                agent_id, step_id, rid, outcome_unknown=False
            )
            raise

        current = self.store.get_agent(agent_id)
        continuation = replace(
            current,
            committed_step=step_id,
            token_ids=tuple(input_ids + result["output_ids"]),
            pending_tool_future=pending_tool_future,
            stream_offset=current.stream_offset + len(result["output_ids"]),
        )
        self.store.commit_claimed_step(
            continuation, expected_epoch=owner_epoch, rid=rid
        )
        return result
