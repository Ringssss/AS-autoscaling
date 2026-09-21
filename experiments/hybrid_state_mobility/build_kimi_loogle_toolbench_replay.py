#!/usr/bin/env python3
"""Build a trace-shaped Kimi/LooGLE/ToolBench hybrid mobility replay."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def empirical_tool_durations(path: Path | None) -> list[float]:
    if path is None or not path.is_file():
        return [0.030, 0.080, 0.130, 0.180, 0.280, 0.430, 0.630, 1.000, 1.300, 1.800]
    data = json.loads(path.read_text())
    values = [float(row["tool_seconds"]) for row in data.get("records", [])]
    return values or [0.100]


def select_kimi_window(
    data: dict[str, np.ndarray],
    *,
    window_index: int | None,
    window_seconds: int,
    max_prompt_tokens: int,
) -> tuple[int, np.ndarray]:
    base = float(data["request_received_ts"].min())
    window = np.floor((data["request_received_ts"] - base) / window_seconds).astype(int)
    eligible = (data["input_len"] <= max_prompt_tokens) & (data["input_len"] > 0)
    if window_index is None:
        counts = collections.Counter(window[eligible].tolist())
        window_index = max(counts, key=lambda key: (counts[key], -key))
    mask = eligible & (window == window_index)
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise RuntimeError(f"no eligible Kimi records in window {window_index}")
    return window_index, indices


def stratified_sample(indices: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if indices.size <= count:
        return indices
    # Keep the real arrival order while evenly covering length quantiles.
    order = indices[np.argsort(indices, kind="stable")]
    return np.sort(rng.choice(order, size=count, replace=False))


def load_loogle_tokens(
    path: Path, tokenizer, needed_tokens: int
) -> tuple[list[int], list[str]]:
    tokens: list[int] = []
    doc_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            doc_id = row["doc_id"]
            if doc_id in doc_ids:
                continue
            ids = tokenizer.encode(row["context"], add_special_tokens=False)
            tokens.extend(ids)
            doc_ids.append(doc_id)
            if len(tokens) >= needed_tokens:
                break
    if len(tokens) < needed_tokens:
        raise RuntimeError(
            f"LooGLE corpus has {len(tokens)} tokens but {needed_tokens} are required"
        )
    return tokens, doc_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kimi-npz", type=Path, required=True)
    parser.add_argument("--loogle-jsonl", type=Path, required=True)
    parser.add_argument("--toolbench-comb", type=Path, required=True)
    parser.add_argument(
        "--toolbench-metadata",
        type=Path,
        default=None,
        help="Defaults to metadata.json beside the ToolBench-comb dataset root.",
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minute-index", type=int)
    parser.add_argument("--window-index", type=int)
    parser.add_argument("--window-seconds", type=int, default=60)
    parser.add_argument("--agents", type=int, default=128)
    parser.add_argument("--engine-count", type=int, default=8)
    parser.add_argument("--max-prompt-tokens", type=int, default=32000)
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument(
        "--tool-duration-source",
        type=Path,
        default=Path(
            "results/tool-gap-workloads/formal/agent-tool-gaps-1786347666986609309.json"
        ),
    )
    args = parser.parse_args()

    if args.agents <= 0 or args.engine_count <= 0:
        raise ValueError("agents and engine-count must be positive")
    rng = random.Random(args.seed)
    kimi = dict(np.load(args.kimi_npz))
    window_index, candidate_indices = select_kimi_window(
        kimi,
        window_index=(
            args.minute_index if args.minute_index is not None else args.window_index
        ),
        window_seconds=args.window_seconds,
        max_prompt_tokens=args.max_prompt_tokens,
    )
    agent_indices = stratified_sample(candidate_indices, args.agents, args.seed)
    if agent_indices.size < args.agents:
        raise RuntimeError(
            f"window {window_index} has only {agent_indices.size} eligible records"
        )

    metadata_path = (
        args.toolbench_metadata or args.toolbench_comb.parent.parent / "metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    table = pq.read_table(args.toolbench_comb).to_pylist()
    table_by_id = {row["id"]: row for row in table}
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for row_meta in metadata["rows"]:
        if row_meta["repeat"] != 0:
            continue
        groups[row_meta["comb_group_id"]].append(row_meta)
    selected_group_ids = sorted(groups)[: args.agents]
    if len(selected_group_ids) < args.agents:
        raise RuntimeError("ToolBench-comb has fewer groups than requested agents")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    group_toolbench: dict[str, list[dict]] = {}
    needed_tokens = 0
    group_base_lengths: dict[str, int] = {}
    group_meta_by_id = {g["comb_group_id"]: g for g in metadata["groups"]}
    for group_id, group_index in zip(selected_group_ids, agent_indices, strict=True):
        rows = sorted(groups[group_id], key=lambda row: row["target_index"])
        encoded = []
        for row_meta in rows:
            parquet_row = table_by_id[row_meta["id"]]
            conversation = parquet_row["conversations"]
            ids = tokenizer.encode(
                conversation["value"][0], add_special_tokens=False
            ) + tokenizer.encode(conversation["value"][1], add_special_tokens=False)
            encoded.append(
                {
                    **row_meta,
                    "token_ids": ids,
                }
            )
        # ToolBench-comb canonical prompts are text-nested, but independent BPE
        # tokenization can differ by a few boundary tokens. Normalize each deeper
        # branch to the previous token prefix plus its post-LCP suffix.
        normalized: list[list[int]] = []
        for row in encoded:
            ids = row["token_ids"]
            if not normalized:
                normalized.append(ids)
                continue
            previous = normalized[-1]
            lcp = next(
                (
                    index
                    for index, (left, right) in enumerate(zip(previous, ids))
                    if left != right
                ),
                min(len(previous), len(ids)),
            )
            normalized.append(previous + ids[lcp:])
        for row, ids in zip(encoded, normalized, strict=True):
            row["toolbench_raw_token_count"] = len(row["token_ids"])
            row["token_ids"] = ids
        common = min(
            next(
                (
                    index
                    for index, (left, right) in enumerate(
                        zip(encoded[0]["token_ids"], row["token_ids"])
                    )
                    if left != right
                ),
                min(len(encoded[0]["token_ids"]), len(row["token_ids"])),
            )
            for row in encoded[1:]
        )
        kimi_index = int(group_index)
        target_input = int(kimi["input_len"][kimi_index])
        desired_cached = int(kimi["cached_tokens"][kimi_index])
        max_tool_suffix = max(len(row["token_ids"]) for row in encoded)
        base_length = min(
            max(0, desired_cached - common),
            max(0, target_input - max_tool_suffix),
        )
        group_base_lengths[group_id] = base_length
        needed_tokens += base_length
        group_toolbench[group_id] = encoded

    # Add slack for deterministic non-overlap and future simulator use.
    loogle_tokens, loogle_doc_ids = load_loogle_tokens(
        args.loogle_jsonl, tokenizer, needed_tokens + args.max_prompt_tokens
    )
    durations = empirical_tool_durations(args.tool_duration_source)
    window_start = float(kimi["request_received_ts"][agent_indices].min())
    branch_template_pool = candidate_indices.tolist()
    if not branch_template_pool:
        raise RuntimeError("empty branch template pool")

    events = []
    token_arrays = {}
    cursor = 0
    for ordinal, (group_id, kimi_index) in enumerate(
        zip(selected_group_ids, agent_indices, strict=True)
    ):
        kimi_index = int(kimi_index)
        agent_id = f"klt-{ordinal:04d}-{group_id}"
        base_length = group_base_lengths[group_id]
        base_ids = loogle_tokens[cursor : cursor + base_length]
        cursor += base_length
        if len(base_ids) != base_length:
            raise RuntimeError("LooGLE token cursor exhausted")

        encoded = group_toolbench[group_id]
        group_metadata = group_meta_by_id[group_id]
        source_dp = int(kimi["dp_rank"][kimi_index])
        source_engine = f"engine-{source_dp % args.engine_count:02d}"
        arrival = float(kimi["request_received_ts"][kimi_index]) - window_start
        previous_prefix_len = 0

        for turn_index, row in enumerate(encoded):
            if turn_index == 0:
                template_index = kimi_index
            else:
                template_index = int(rng.choice(branch_template_pool))
            token_ids = base_ids + row["token_ids"]
            event_id = f"{agent_id}-t{turn_index:02d}"
            tool_name = group_metadata["tools"][
                min(turn_index, len(group_metadata["tools"]) - 1)
            ]
            duration = durations[
                int(hashlib.sha256(tool_name.encode()).hexdigest(), 16) % len(durations)
            ]
            e2e = float(kimi["e2e_latency"][template_index])
            tool_start = arrival + e2e
            tool_end = tool_start + duration
            event = {
                "event_id": event_id,
                "agent_id": agent_id,
                "turn_index": turn_index,
                "branch_id": event_id,
                "parent_branch_id": (
                    None if turn_index == 0 else events[-1]["branch_id"]
                ),
                "arrival_rel_s": arrival,
                "llm_e2e_template_s": e2e,
                "tool_start_rel_s": tool_start,
                "tool_end_rel_s": tool_end,
                "source_engine": source_engine,
                "source_dp_rank": source_dp,
                "prompt_tokens": len(token_ids),
                "parent_prefix_tokens": previous_prefix_len,
                "target_output_tokens": int(kimi["completion_tokens"][template_index]),
                "tool_name": tool_name,
                "tool_duration_s": duration,
                "tool_duration_source": str(args.tool_duration_source),
                "kimi_template_rid": str(kimi["rid"][template_index]),
                "kimi_template_input_len": int(kimi["input_len"][template_index]),
                "kimi_template_cached_tokens": int(
                    kimi["cached_tokens"][template_index]
                ),
                "loogle_base_tokens": base_length,
                "loogle_document_ids": loogle_doc_ids[
                    : max(1, min(len(loogle_doc_ids), base_length // 12000 + 1))
                ],
                "toolbench_group_id": group_id,
                "toolbench_row_id": row["id"],
                "toolbench_target_depth": row["target_token_depth"],
                "state_descriptor": {
                    "token_kv_slots": len(token_ids),
                    "recurrent_snapshot_slots": 1,
                    "full_kv_bytes": len(token_ids) * 20480,
                    "recurrent_state_bytes": 64389120,
                },
                "provenance": "kimi-timing+loogle-content+toolbench-boundary",
            }
            token_arrays[event_id] = np.asarray(token_ids, dtype=np.int32)
            events.append(event)
            previous_prefix_len = len(token_ids)
            arrival = tool_end

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "replay_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    np.savez_compressed(
        args.output_dir / "replay_tokens.npz",
        **token_arrays,
    )

    input_lengths = [event["prompt_tokens"] for event in events]
    parent_ratios = [
        event["parent_prefix_tokens"] / event["prompt_tokens"]
        for event in events
        if event["parent_prefix_tokens"]
    ]
    summary = {
        "schema_version": 1,
        "kimi_npz": str(args.kimi_npz),
        "kimi_window_index": window_index,
        "kimi_window_seconds": args.window_seconds,
        "kimi_window_start": window_start,
        "loogle_jsonl": str(args.loogle_jsonl),
        "toolbench_comb": str(args.toolbench_comb),
        "toolbench_metadata": str(metadata_path),
        "model_path": str(args.model_path),
        "agents": args.agents,
        "events": len(events),
        "engine_count": args.engine_count,
        "seed": args.seed,
        "prompt_p50_p90_p99_max": np.percentile(input_lengths, [50, 90, 99]).tolist()
        + [max(input_lengths)],
        "median_parent_prefix_ratio": float(np.median(parent_ratios)),
        "tool_duration_p50_p90_p99_max": np.percentile(
            [event["tool_duration_s"] for event in events], [50, 90, 99]
        ).tolist()
        + [max(event["tool_duration_s"] for event in events)],
        "event_span_seconds": [
            min(event["arrival_rel_s"] for event in events),
            max(event["tool_end_rel_s"] for event in events),
        ],
        "loogle_documents_tokenized": len(loogle_doc_ids),
        "loogle_tokens_available": len(loogle_tokens),
        "token_arrays": len(token_arrays),
        "artifact_manifest": str(manifest_path),
        "artifact_tokens": str(args.output_dir / "replay_tokens.npz"),
        "methodology": (
            "Kimi supplies only timing and length shape; LooGLE supplies deterministic "
            "long-context token content; ToolBench-comb supplies nested tool-boundary "
            "prompts and tool names. This is trace-shaped replay, not recovery of original IDs."
        ),
    }
    (args.output_dir / "replay_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
