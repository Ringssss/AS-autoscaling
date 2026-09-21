"""Provenance-preserving readers for anonymous serving traces."""

from __future__ import annotations

import hashlib
import math
import re
import struct
from collections import Counter
from functools import lru_cache


_KIMI_FIELDS = {
    "input_length": (re.compile(r"Finish: input_ids_len=(\d+)"), int),
    "request_id": (re.compile(r"rid='([^']+)'"), str),
    "conversation_id": (re.compile(r"conversation_id='([^']+)'"), str),
    "output_length": (re.compile(r"'completion_tokens': (\d+)"), int),
    "cached_tokens": (re.compile(r"'cached_tokens': (\d+)"), int),
    "arrival_seconds": (re.compile(r"'request_received_ts': ([0-9.eE+-]+)"), float),
    "finished_seconds": (re.compile(r"'request_finished_ts': ([0-9.eE+-]+)"), float),
    "ttft_seconds": (re.compile(r"'ttft': ([0-9.eE+-]+)"), float),
    "dp_rank": (re.compile(r"'dp_rank': (-?\d+)"), int),
}


def parse_kimi_line(line: str) -> dict | None:
    if not line.startswith("Finish:"):
        return None
    result = {}
    for name, (pattern, cast) in _KIMI_FIELDS.items():
        match = pattern.search(line)
        result[name] = cast(match.group(1)) if match else None
    for required in ("request_id", "input_length", "output_length", "arrival_seconds"):
        if result[required] is None:
            raise ValueError(f"missing Kimi field {required}")
    result["input_ids_redacted"] = "<REDACTED_INPUT_IDS>" in line
    result["output_ids_redacted"] = "<REDACTED_OUTPUT_IDS>" in line
    return result


@lru_cache(maxsize=65536)
def block_tokens(block_hash: int) -> tuple[int, ...]:
    payload = hashlib.shake_256(f"fluidkv-block-v1:{block_hash}".encode()).digest(64)
    return tuple(1000 + number % 30000 for number in struct.unpack("<16I", payload))


def surrogate_tokens(row: dict) -> list[int]:
    length = int(row["input_length"])
    if length <= 0 or len(row["hash_ids"]) != math.ceil(length / 16):
        raise ValueError("input length and hash block count disagree")
    return [token for block in row["hash_ids"] for token in block_tokens(block)][:length]


def inspect_qwen_rows(rows: list[dict], max_pair_tokens: int = 8192) -> tuple[dict, list[dict]]:
    by_id = {}
    for row in rows:
        chat_id = row["chat_id"]
        if chat_id in by_id:
            raise ValueError(f"duplicate chat id {chat_id}")
        if row["input_length"] <= 0 or len(row["hash_ids"]) != math.ceil(row["input_length"] / 16):
            raise ValueError(f"invalid block count for {chat_id}")
        if not math.isfinite(row["timestamp"]) or row["output_length"] < 0:
            raise ValueError(f"invalid timestamp or output length for {chat_id}")
        by_id[chat_id] = row
    counts = Counter()
    types = Counter()
    pairs = []
    child_counts = Counter()
    for child in rows:
        types[child.get("type", "unknown")] += 1
        parent_id = child["parent_chat_id"]
        if parent_id < 0:
            counts["root_requests"] += 1
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            counts["missing_parents"] += 1
            continue
        if parent["timestamp"] >= child["timestamp"]:
            raise ValueError(f"non-increasing parent timestamp for {child['chat_id']}")
        counts["linked_requests"] += 1
        child_counts[parent_id] += 1
        matched = 0
        for left, right in zip(parent["hash_ids"], child["hash_ids"]):
            if left != right:
                break
            matched += 1
        overlap = min(16 * matched, parent["input_length"], child["input_length"])
        counts["verified_parent_input_overlap_tokens"] += overlap
        counts["linked_child_input_tokens"] += child["input_length"]
        if parent["input_length"] % 16:
            counts["partial_parent_boundary"] += 1
        if overlap != parent["input_length"]:
            counts["parent_input_not_fully_matched"] += 1
        if parent["input_length"] % 16 == 0 and overlap == parent["input_length"]:
            counts["exact_parent_prefix_pairs"] += 1
            if max(parent["input_length"], child["input_length"]) > max_pair_tokens:
                continue
            if child["input_length"] <= parent["input_length"]:
                continue
            pairs.append({
                "parent_id": parent_id,
                "child_id": child["chat_id"],
                "parent_tokens": parent["input_length"],
                "child_tokens": child["input_length"],
                "parent_arrival_seconds": parent["timestamp"],
                "child_arrival_seconds": child["timestamp"],
                "arrival_delta_seconds": child["timestamp"] - parent["timestamp"],
            })
    pairs.sort(key=lambda row: (row["child_arrival_seconds"], row["child_id"]))
    summary = {key: counts[key] for key in (
        "root_requests", "missing_parents", "linked_requests", "verified_parent_input_overlap_tokens",
        "linked_child_input_tokens", "partial_parent_boundary", "parent_input_not_fully_matched",
        "exact_parent_prefix_pairs",
    )}
    summary.update({
        "requests": len(rows), "types": dict(types),
        "max_children": max(child_counts.values(), default=0),
        "eligible_pair_count": len(pairs), "max_pair_tokens": max_pair_tokens,
        "input_tokens": sum(row["input_length"] for row in rows),
        "output_tokens": sum(row["output_length"] for row in rows),
        "span_seconds": max((r["timestamp"] for r in rows), default=0) - min((r["timestamp"] for r in rows), default=0),
        "timing_semantics": "request arrivals; no observed completion or tool duration",
        "content_semantics": "anonymous block hashes; no original tokens or multimedia assets",
    })
    return summary, pairs


def select_exact_parent_cohort(rows: list[dict], max_requests: int) -> list[dict]:
    """Select known parent-prefix relationships, retaining native timestamps.

    This deliberately selects a mechanism cohort rather than a representative
    traffic sample. The caller must report its coverage and any time scaling.
    """
    by_id = {row["chat_id"]: row for row in rows}
    selected = set()
    for child in sorted(rows, key=lambda row: (row["timestamp"], row["chat_id"])):
        parent = by_id.get(child["parent_chat_id"])
        if parent is None or parent["input_length"] % 16:
            continue
        if child["input_length"] <= parent["input_length"]:
            continue
        if child["hash_ids"][:len(parent["hash_ids"]) ] != parent["hash_ids"]:
            continue
        additions = {parent["chat_id"], child["chat_id"]} - selected
        if len(selected) + len(additions) > max_requests:
            continue
        selected.update(additions)
        if len(selected) == max_requests:
            break
    return sorted((by_id[key] for key in selected), key=lambda row: (row["timestamp"], row["chat_id"]))
