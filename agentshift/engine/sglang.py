from __future__ import annotations

import asyncio
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class EngineRequestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SGLangAgentShiftClient:
    engine_id: str
    base_url: str
    timeout: float = 120.0

    def _post_sync(
        self, path: str, payload: dict[str, Any], require_success: bool
    ) -> dict[str, Any]:
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise EngineRequestError(f"{self.engine_id} {path}: {exc.code} {detail}") from exc
        except OSError as exc:
            raise EngineRequestError(f"{self.engine_id} {path}: {exc}") from exc
        if require_success and not result.get("success", False):
            raise EngineRequestError(
                f"{self.engine_id} {path}: {result.get('message', 'request failed')}"
            )
        return result

    async def post(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        require_success: bool = True,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._post_sync, path, payload, require_success
        )

    async def pin_prefix(
        self, agent_id: str, owner_epoch: int, token_ids: tuple[int, ...]
    ) -> dict[str, Any]:
        return await self.post(
            "/agentshift/prefix/pin",
            {
                "agent_id": agent_id,
                "owner_epoch": owner_epoch,
                "token_ids": token_ids,
            },
        )

    async def init_transfer_group(
        self,
        *,
        master_address: str,
        ports: tuple[int, ...],
        group_rank: int,
        group_name: str,
    ) -> dict[str, Any]:
        return await self.post(
            "/agentshift/transfer_group/init",
            {
                "master_address": master_address,
                "ports": ",".join(map(str, ports)),
                "group_rank": group_rank,
                "world_size": 2,
                "group_name": group_name,
                "backend": "nccl",
            },
        )

    async def transfer_prefix(
        self,
        *,
        role: str,
        migration_id: str,
        agent_id: str,
        owner_epoch: int,
        token_ids: tuple[int, ...],
        group_name: str,
        ports: tuple[int, ...],
        async_transfer: bool = False,
    ) -> dict[str, Any]:
        return await self.post(
            "/agentshift/prefix/transfer",
            {
                "role": role,
                "migration_id": migration_id,
                "agent_id": agent_id,
                "owner_epoch": owner_epoch,
                "token_ids": token_ids,
                "group_name": group_name,
                "ports": ",".join(map(str, ports)),
                "async_transfer": async_transfer,
            },
        )

    async def transfer_status(self, migration_id: str) -> dict[str, Any]:
        return await self.post(
            "/agentshift/prefix/transfer/status",
            {"migration_id": migration_id},
        )

    async def cleanup_transfer(self, migration_id: str) -> dict[str, Any]:
        return await self.post(
            "/agentshift/prefix/transfer/cleanup",
            {"migration_id": migration_id},
        )

    async def start_tier_operation(
        self,
        *,
        operation: str,
        operation_id: str,
        checkpoint_id: str,
        agent_id: str,
        owner_epoch: int,
        token_ids: tuple[int, ...],
        release_gpu: bool = True,
    ) -> dict[str, Any]:
        return await self.post(
            "/agentshift/prefix/tier/start",
            {
                "operation": operation,
                "operation_id": operation_id,
                "checkpoint_id": checkpoint_id,
                "agent_id": agent_id,
                "owner_epoch": owner_epoch,
                "token_ids": token_ids,
                "release_gpu": release_gpu,
            },
        )

    async def tier_status(self, operation_id: str) -> dict[str, Any]:
        return await self.post(
            "/agentshift/prefix/tier/status",
            {"operation_id": operation_id},
        )

    async def cleanup_tier_operation(
        self, operation_id: str, *, drop_checkpoint: bool = False
    ) -> dict[str, Any]:
        return await self.post(
            "/agentshift/prefix/tier/cleanup",
            {
                "operation_id": operation_id,
                "drop_checkpoint": drop_checkpoint,
            },
        )

    async def release_prefix(
        self,
        agent_id: str,
        owner_epoch: int,
        *,
        evict_after_release: bool = True,
        allow_missing: bool = False,
    ) -> dict[str, Any]:
        return await self.post(
            "/agentshift/prefix/release",
            {
                "agent_id": agent_id,
                "owner_epoch": owner_epoch,
                "evict_after_release": evict_after_release,
                "allow_missing": allow_missing,
            },
        )

    async def rebind_prefix(
        self, agent_id: str, expected_owner_epoch: int, new_owner_epoch: int
    ) -> dict[str, Any]:
        return await self.post(
            "/agentshift/prefix/rebind",
            {
                "agent_id": agent_id,
                "expected_owner_epoch": expected_owner_epoch,
                "new_owner_epoch": new_owner_epoch,
            },
        )


async def generate(
    client: SGLangAgentShiftClient,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    rid: str,
) -> dict[str, Any]:
    return await client.post(
        "/generate",
        {
            "input_ids": token_ids,
            "rid": rid,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0,
            },
        },
        require_success=False,
    )


def _stream_generate_sync(
    client: SGLangAgentShiftClient,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    rid: str,
    notify_first_token,
) -> dict[str, Any]:
    payload = {
        "input_ids": token_ids,
        "rid": rid,
        "stream": True,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0,
        },
    }
    request = urllib.request.Request(
        f"{client.base_url.rstrip('/')}/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    request_started = time.perf_counter()
    token_timestamps: list[float] = []
    final: dict[str, Any] | None = None
    try:
        with urllib.request.urlopen(request, timeout=client.timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                final = json.loads(data)
                token_timestamps.append(time.perf_counter())
                if len(token_timestamps) == 1:
                    notify_first_token()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise EngineRequestError(
            f"{client.engine_id} /generate: {exc.code} {detail}"
        ) from exc
    if final is None:
        raise EngineRequestError(f"{client.engine_id} /generate: empty stream")
    finished = time.perf_counter()
    return {
        "response": final,
        "request_started": request_started,
        "finished": finished,
        "token_timestamps": token_timestamps,
        "ttft_seconds": token_timestamps[0] - request_started,
        "e2e_seconds": finished - request_started,
        "token_intervals_seconds": [
            current - previous
            for previous, current in zip(token_timestamps, token_timestamps[1:])
        ],
    }


async def stream_generate(
    client: SGLangAgentShiftClient,
    token_ids: list[int],
    *,
    max_new_tokens: int,
    rid: str,
    first_token_event: asyncio.Event | None = None,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()

    def notify_first_token() -> None:
        if first_token_event is not None:
            loop.call_soon_threadsafe(first_token_event.set)

    return await asyncio.to_thread(
        _stream_generate_sync,
        client,
        token_ids,
        max_new_tokens=max_new_tokens,
        rid=rid,
        notify_first_token=notify_first_token,
    )
