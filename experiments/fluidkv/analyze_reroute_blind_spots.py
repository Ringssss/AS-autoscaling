#!/usr/bin/env python3
"""Quantify workload events that request routing alone cannot make warm.

The Qwen analysis is a conservative hash-ring projection, not a claim that we
replayed upstream DualMap's complete load-dependent decisions. A link is counted
only to the depth for which anonymous 16-token block hashes match. The final
partial block is not claimed.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from uhashring import HashRing


HYBRID_BYTES_PER_TOKEN = 20_480
HYBRID_SNAPSHOT_BYTES = 64_389_120
DENSE_QWEN3_8B_BYTES_PER_TOKEN = 147_456


def distribution(values):
    values = np.asarray(list(values), dtype=float)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "sum": float(values.sum()),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(values.max()),
    }


class VerifiedPrefixRing:
    """Two-choice projection using the same physical-node anchor idea as DualMap."""

    def __init__(self, engines):
        self.engines = list(engines)

    @staticmethod
    @lru_cache(maxsize=None)
    def _cached_ring(active_tuple, backend):
        kwargs = {"hash_fn": "ketama"} if backend == 2 else {}
        return HashRing(nodes=[str(engine) for engine in active_tuple], **kwargs)

    def candidates(self, key, active=None):
        active = list(self.engines if active is None else active)
        active_tuple = tuple(active)
        rings = {backend: self._cached_ring(active_tuple, backend) for backend in (1, 2)}
        first = int(rings[1].get_node(key))
        second = int(rings[2].get_node(key))
        if first == second:
            position = active.index(first)
            second = active[(position + 1) % len(active)]
        return [first, second]


def matched_prefix(parent, child):
    matched = 0
    for left, right in zip(parent["hash_ids"], child["hash_ids"]):
        if left != right:
            break
        matched += 1
    # The released hash list has ceil(length/16) entries. A matching final
    # partial-block hash still does not establish token identity, so count only
    # complete blocks when the final parent block is partial.
    full_blocks = matched
    if parent["input_length"] % 16 and matched == len(parent["hash_ids"]):
        full_blocks -= 1
    overlap = 16 * max(0, full_blocks)
    exact_parent = (
        parent["input_length"] % 16 == 0
        and matched == len(parent["hash_ids"])
        and overlap == parent["input_length"]
        and child["input_length"] > parent["input_length"]
    )
    return matched, overlap, exact_parent


def state_bytes(overlap_tokens, model):
    if overlap_tokens <= 0:
        return 0
    if model == "qwen3.5-35b-hybrid":
        return overlap_tokens * HYBRID_BYTES_PER_TOKEN + HYBRID_SNAPSHOT_BYTES
    if model == "qwen3-8b-dense":
        return overlap_tokens * DENSE_QWEN3_8B_BYTES_PER_TOKEN
    raise ValueError(model)


def analyze_qwen(path, engines=8):
    rows = [json.loads(line) for line in Path(path).open()]
    by_id = {row["chat_id"]: row for row in rows}
    ring = VerifiedPrefixRing(range(engines))
    links = []
    counts = Counter()
    for child in rows:
        parent = by_id.get(child["parent_chat_id"])
        if parent is None:
            continue
        matched_blocks, overlap, exact_parent = matched_prefix(parent, child)
        if matched_blocks <= 0:
            counts["no_verified_first_block"] += 1
            continue
        key = "fluidkv:" + ":".join(map(str, parent["hash_ids"][:matched_blocks]))
        before = ring.candidates(key)
        links.append({
            "parent_id": parent["chat_id"], "child_id": child["chat_id"],
            "matched_blocks": matched_blocks, "overlap_tokens": overlap,
            "exact_parent": exact_parent, "gap_seconds": child["timestamp"] - parent["timestamp"],
            "parent_candidates_before": before, "key": key,
            "hybrid_bytes": state_bytes(overlap, "qwen3.5-35b-hybrid"),
            "dense_qwen3_8b_bytes": state_bytes(overlap, "qwen3-8b-dense"),
        })

    single_retirements = []
    scenario_links = []
    for victim in range(engines):
        active = [engine for engine in range(engines) if engine != victim]
        selected = [link for link in links if victim in link["parent_candidates_before"]]
        primary = [link for link in selected if link["parent_candidates_before"][0] == victim]
        for link in selected:
            after = ring.candidates(link["key"], active)
            link = dict(link)
            link.update({"retired_engine": victim, "child_candidates_after": after,
                         "primary_hash_owner_retired": link["parent_candidates_before"][0] == victim})
            scenario_links.append(link)
        single_retirements.append({
            "retired_engine": victim,
            "candidate_owner_links": len(selected),
            "primary_hash_owner_links": len(primary),
            "verified_overlap_tokens_candidate_owner": sum(link["overlap_tokens"] for link in selected),
            "verified_overlap_tokens_primary_owner": sum(link["overlap_tokens"] for link in primary),
            "hybrid_transfer_gib_candidate_owner": sum(link["hybrid_bytes"] for link in selected) / 2**30,
            "dense_qwen3_8b_transfer_gib_candidate_owner": sum(link["dense_qwen3_8b_bytes"] for link in selected) / 2**30,
            "arrival_gap_seconds_candidate_owner": distribution(link["gap_seconds"] for link in selected),
            "exact_candidate_owner_links": sum(link["exact_parent"] for link in selected),
            "exact_primary_hash_owner_links": sum(link["exact_parent"] and link["parent_candidates_before"][0] == victim for link in selected),
            "exact_verified_overlap_tokens_primary_owner": sum(link["overlap_tokens"] for link in selected if link["exact_parent"] and link["parent_candidates_before"][0] == victim),
        })

    active_after_8_to_4 = [0, 1, 2, 3]
    retired_set = {4, 5, 6, 7}
    selected_8_to_4 = [link for link in links if retired_set.intersection(link["parent_candidates_before"])]
    primary_8_to_4 = [link for link in links if link["parent_candidates_before"][0] in retired_set]
    exact_selected_8_to_4 = [link for link in selected_8_to_4 if link["exact_parent"]]
    exact_primary_8_to_4 = [link for link in primary_8_to_4 if link["exact_parent"]]
    for link in selected_8_to_4:
        link["child_candidates_after_8_to_4"] = ring.candidates(link["key"], active_after_8_to_4)

    exact_links = [link for link in links if link["exact_parent"]]
    summary = {
        "path": str(path), "requests": len(rows), "linked_requests": len(links),
        "verified_parent_overlap_tokens": sum(link["overlap_tokens"] for link in links),
        "verified_parent_overlap_fraction_of_child_input": sum(link["overlap_tokens"] for link in links) / sum(by_id[link["child_id"]]["input_length"] for link in links),
        "exact_block_aligned_parent_links": len(exact_links),
        "exact_parent_overlap_tokens": sum(link["overlap_tokens"] for link in exact_links),
        "all_link_arrival_gap_seconds": distribution(link["gap_seconds"] for link in links),
        "single_retirements": single_retirements,
        "single_retirement_scenario_links_not_unique": len(scenario_links),
        "unique_links_with_some_candidate_retired": len({(link["parent_id"], link["child_id"]) for link in scenario_links}),
        "scale_in_8_to_4": {
            "retired_engines": sorted(retired_set),
            "links_with_a_candidate_owner_retired": len(selected_8_to_4),
            "links_with_primary_hash_owner_retired": len(primary_8_to_4),
            "verified_overlap_tokens_candidate_owner": sum(link["overlap_tokens"] for link in selected_8_to_4),
            "verified_overlap_tokens_primary_owner": sum(link["overlap_tokens"] for link in primary_8_to_4),
            "hybrid_transfer_gib_candidate_owner": sum(link["hybrid_bytes"] for link in selected_8_to_4) / 2**30,
            "dense_qwen3_8b_transfer_gib_candidate_owner": sum(link["dense_qwen3_8b_bytes"] for link in selected_8_to_4) / 2**30,
            "arrival_gap_seconds_candidate_owner": distribution(link["gap_seconds"] for link in selected_8_to_4),
            "exact_links_with_a_candidate_owner_retired": len(exact_selected_8_to_4),
            "exact_links_with_primary_hash_owner_retired": len(exact_primary_8_to_4),
            "exact_verified_overlap_tokens_primary_owner": sum(link["overlap_tokens"] for link in exact_primary_8_to_4),
            "exact_hybrid_transfer_gib_primary_owner": sum(link["hybrid_bytes"] for link in exact_primary_8_to_4) / 2**30,
            "exact_dense_qwen3_8b_transfer_gib_primary_owner": sum(link["dense_qwen3_8b_bytes"] for link in exact_primary_8_to_4) / 2**30,
            "exact_arrival_gap_seconds_primary_owner": distribution(link["gap_seconds"] for link in exact_primary_8_to_4),
        },
        "semantics": {
            "overlap": "Only full anonymous blocks that match positionally; partial final blocks are excluded.",
            "gap": "Parent arrival to child arrival, not an observed tool or idle window.",
            "owner": "A deterministic hash-owner projection over two candidate anchors; actual DualMap also uses live load and queue state.",
            "benefit": "Counting a link does not prove migration is profitable; it identifies state that routing cannot move after its owner disappears.",
        },
    }
    return summary, links, scenario_links


def analyze_kimi(path):
    rows = [json.loads(line) for line in Path(path).open()]
    rank = defaultdict(lambda: Counter())
    minute = defaultdict(lambda: Counter())
    base = min(row["arrival_seconds"] for row in rows)
    for row in rows:
        if row["dp_rank"] is None or row["dp_rank"] < 0:
            continue
        cached = row["cached_tokens"] or 0
        rank[int(row["dp_rank"])].update(requests=1, input_tokens=row["input_length"],
                                         cached_tokens=cached, uncached_tokens=row["input_length"] - cached)
        minute[int((row["arrival_seconds"] - base) // 60)][int(row["dp_rank"])] += cached
    ratios = []
    hour18_base = base + 18 * 3600
    hour18 = defaultdict(Counter)
    for index, counts in minute.items():
        values = np.asarray([counts.get(engine, 0) for engine in range(8)], dtype=float)
        if values.sum():
            ratios.append(values.max() / values.mean())
    for row in rows:
        if hour18_base <= row["arrival_seconds"] < hour18_base + 3600 and row["dp_rank"] is not None and row["dp_rank"] >= 0:
            hour18[int((row["arrival_seconds"] - hour18_base) // 60)][int(row["dp_rank"])] += row["cached_tokens"] or 0
    ratios18 = []
    for counts in hour18.values():
        values = np.asarray([counts.get(engine, 0) for engine in range(8)], dtype=float)
        if values.sum():
            ratios18.append(values.max() / values.mean())
    return {
        "path": str(path), "requests": len(rows),
        "rank_totals": {str(key): dict(value) for key, value in sorted(rank.items())},
        "full_day_minute_hot_over_mean": distribution(ratios),
        "hour18_minute_hot_over_mean": distribution(ratios18),
        "identity_limitation": "All token IDs are redacted and conversation IDs are empty; exact successor links and prefix keys cannot be recovered. Rank analysis is gravity evidence, not a successor-reuse replay.",
        "routing_implication": "If a rank with warm state must leave the serving set, a router can choose another rank but cannot move its unidentified warm execution state. No exact child-link count is claimable from this release.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-root", type=Path, required=True)
    parser.add_argument("--kimi-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = {"A": "qwen_traceA_blksz_16.jsonl", "Coder": "qwen_coder_blksz_16.jsonl", "Thinking": "qwen_thinking_blksz_16.jsonl"}
    result = {"kimi": analyze_kimi(args.kimi_jsonl), "qwen": {}}
    for split, filename in files.items():
        summary, _, _ = analyze_qwen(args.qwen_root / filename)
        result["qwen"][split] = summary
        print(json.dumps({"split": split, "linked_requests": summary["linked_requests"],
                          "verified_overlap_tokens": summary["verified_parent_overlap_tokens"],
                          "scale_in_primary_owner_links": summary["scale_in_8_to_4"]["links_with_primary_hash_owner_retired"]}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
