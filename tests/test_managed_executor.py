import asyncio

import pytest

from agentshift.runtime.executor import ManagedAgentExecutor
from agentshift.state.schema import AgentContinuation
from agentshift.state.store import SQLiteStateStore, StaleLease, StateConflict


class FakeClient:
    async def post(self, path, payload, *, require_success=True):
        assert path == "/generate"
        return {
            "output_ids": [7, 8],
            "meta_info": {"cached_tokens": len(payload["input_ids"])},
        }


def make_executor(tmp_path):
    store = SQLiteStateStore(tmp_path / "state.db")
    store.register_agent(AgentContinuation("a1", 1, "engine-b", 9, (1, 2)))
    return store, ManagedAgentExecutor(store, {"engine-b": FakeClient()})


def test_managed_turn_claims_and_commits_exactly_one_step(tmp_path):
    store, executor = make_executor(tmp_path)
    result = asyncio.run(
        executor.run_turn(
            agent_id="a1",
            owner_engine="engine-b",
            owner_epoch=9,
            step_id=2,
            input_ids=[1, 2, 3],
            max_new_tokens=2,
            rid="rid-1",
            pending_tool_future="tool-2",
        )
    )
    assert result["output_ids"] == [7, 8]
    continuation = store.get_agent("a1")
    assert continuation.committed_step == 2
    assert continuation.token_ids == (1, 2, 3, 7, 8)
    assert continuation.pending_tool_future == "tool-2"
    with pytest.raises(StateConflict):
        asyncio.run(
            executor.run_turn(
                agent_id="a1",
                owner_engine="engine-b",
                owner_epoch=9,
                step_id=2,
                input_ids=[1, 2, 3],
                max_new_tokens=2,
                rid="rid-replay",
            )
        )


def test_old_owner_epoch_is_rejected_before_inference(tmp_path):
    _, executor = make_executor(tmp_path)
    with pytest.raises(StaleLease):
        asyncio.run(
            executor.run_turn(
                agent_id="a1",
                owner_engine="engine-b",
                owner_epoch=8,
                step_id=2,
                input_ids=[1, 2, 3],
                max_new_tokens=2,
                rid="rid-stale",
            )
        )
