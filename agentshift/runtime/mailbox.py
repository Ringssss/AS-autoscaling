from agentshift.state.schema import ToolResult
from agentshift.state.store import SQLiteStateStore


class ToolMailbox:
    def __init__(self, store: SQLiteStateStore):
        self.store = store

    def publish(self, result: ToolResult) -> bool:
        return self.store.put_tool_result(result)

    def read(self, agent_id: str, step_id: int, future_id: str) -> ToolResult | None:
        return self.store.get_tool_result(agent_id, step_id, future_id)
