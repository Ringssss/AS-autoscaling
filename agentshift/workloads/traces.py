from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class AgentTurn:
    agent_id: str
    turn: int
    arrival_seconds: float
    incremental_input_tokens: int
    cumulative_prefix_tokens: int
    output_tokens: int
    tool_gap_seconds: float


def iter_flowprefill_turns(path: str | Path, limit: int | None = None) -> Iterator[AgentTurn]:
    """Convert FlowPrefill's parent-linked chats into agent turns.

    The inter-arrival delta is a proxy for the blocked inter-turn window; the
    trace does not separately label tool execution time.
    """
    records: list[dict] = []
    roots: list[int] = []
    emitted = 0
    with Path(path).open() as handle:
        for line in handle:
            item = json.loads(line)
            parent = int(item["parent_chat_id"])
            if parent < 0:
                root = len(records)
                gap = 0.0
                incremental = int(item["input_length"])
            else:
                if parent >= len(records):
                    raise ValueError(f"parent {parent} appears after child")
                root = roots[parent]
                parent_item = records[parent]
                gap = max(0.0, float(item["timestamp"]) - float(parent_item["timestamp"]))
                incremental = max(
                    0,
                    (len(item["hash_ids"]) - len(parent_item["hash_ids"])) * 16,
                )
            records.append(item)
            roots.append(root)
            yield AgentTurn(
                agent_id=f"flow-{root}",
                turn=int(item.get("turn", 1)),
                arrival_seconds=float(item["timestamp"]),
                incremental_input_tokens=incremental,
                cumulative_prefix_tokens=len(item["hash_ids"]) * 16,
                output_tokens=int(item["output_length"]),
                tool_gap_seconds=gap,
            )
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def iter_kimi_requests(
    path: str | Path, limit: int | None = None
) -> Iterator[tuple[str, int, int]]:
    """Yield timestamp, context length, and output length from the Kimi trace."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            yield (
                row["TIMESTAMP"],
                int(row["ContextTokens"]),
                int(row["GeneratedTokens"]),
            )
            if limit is not None and index + 1 >= limit:
                return
