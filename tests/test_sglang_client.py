import asyncio
import threading
import time

from agentshift.engine.sglang import SGLangAgentShiftClient


def test_control_requests_are_serialized_per_engine(monkeypatch):
    active = 0
    maximum = 0
    guard = threading.Lock()

    def fake_post(self, path, payload, require_success):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with guard:
            active -= 1
        return {"success": True, "token_count": len(payload.get("token_ids", ())) }

    monkeypatch.setattr(SGLangAgentShiftClient, "_post_sync", fake_post)
    client = SGLangAgentShiftClient("engine-0", "http://unused")

    async def pin_both():
        await asyncio.gather(
            client.pin_prefix("a", 1, (1, 2)),
            client.pin_prefix("b", 1, (3, 4)),
        )

    asyncio.run(pin_both())
    assert maximum == 1


def test_control_requests_to_different_engines_remain_concurrent(monkeypatch):
    active = 0
    maximum = 0
    guard = threading.Lock()

    def fake_post(self, path, payload, require_success):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.01)
        with guard:
            active -= 1
        return {"success": True, "token_count": len(payload.get("token_ids", ())) }

    monkeypatch.setattr(SGLangAgentShiftClient, "_post_sync", fake_post)
    first = SGLangAgentShiftClient("engine-0", "http://unused")
    second = SGLangAgentShiftClient("engine-1", "http://unused")

    async def pin_both():
        await asyncio.gather(
            first.pin_prefix("a", 1, (1, 2)),
            second.pin_prefix("b", 1, (3, 4)),
        )

    asyncio.run(pin_both())
    assert maximum == 2


def test_progressive_transfer_is_explicit_and_carries_group_size(monkeypatch):
    seen = {}

    def fake_post(self, path, payload, require_success):
        seen.update(path=path, payload=payload)
        return {"success": True, "state": "QUEUED"}

    monkeypatch.setattr(SGLangAgentShiftClient, "_post_sync", fake_post)
    client = SGLangAgentShiftClient("engine-0", "http://unused")
    asyncio.run(
        client.transfer_prefix(
            role="destination",
            migration_id="migration-1",
            agent_id="agent-1",
            owner_epoch=2,
            token_ids=(1, 2, 3),
            group_name="group",
            ports=(30000,),
            async_transfer=True,
            progressive=True,
            layer_group_size=6,
        )
    )
    assert seen["path"] == "/agentshift/prefix/transfer"
    assert seen["payload"]["progressive"] is True
    assert seen["payload"]["layer_group_size"] == 6


def test_baseline_transfer_defaults_progressive_off(monkeypatch):
    seen = {}

    def fake_post(self, path, payload, require_success):
        seen.update(payload)
        return {"success": True, "state": "COMPLETE"}

    monkeypatch.setattr(SGLangAgentShiftClient, "_post_sync", fake_post)
    client = SGLangAgentShiftClient("engine-0", "http://unused")
    asyncio.run(
        client.transfer_prefix(
            role="source",
            migration_id="migration-1",
            agent_id="agent-1",
            owner_epoch=1,
            token_ids=(1, 2, 3),
            group_name="group",
            ports=(30000,),
        )
    )
    assert seen["progressive"] is False
    assert seen["layer_group_size"] == 4
