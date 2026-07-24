from dataclasses import dataclass

from agentshift.state.store import SQLiteStateStore


@dataclass(frozen=True, slots=True)
class LeaseGuard:
    store: SQLiteStateStore
    agent_id: str
    owner_engine: str
    owner_epoch: int

    def validate(self) -> None:
        self.store.assert_lease(self.agent_id, self.owner_engine, self.owner_epoch)
