#!/usr/bin/env python3
"""Measure repeated hybrid-state handoffs over one persistent NCCL group."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def flush(url: str) -> None:
    response = requests.post(url.rstrip("/") + "/flush_cache", timeout=30)
    response.raise_for_status()


def post(url: str, path: str, payload: dict) -> dict:
    response = requests.post(url.rstrip("/") + path, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def generate(url: str, input_ids: list[int], output_tokens: int) -> dict:
    response = requests.post(
        url.rstrip("/") + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": output_tokens,
                "temperature": 0.0,
            },
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def concurrent_posts(
    source: str,
    destination: str,
    path: str,
    source_payload: dict,
    destination_payload: dict,
) -> tuple[dict, dict]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        source_future = executor.submit(post, source, path, source_payload)
        destination_future = executor.submit(
            post, destination, path, destination_payload
        )
        return source_future.result(), destination_future.result()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31080")
    parser.add_argument("--destination", default="http://127.0.0.1:31081")
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--rendezvous-port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    for url in (args.source, args.destination):
        requests.get(url.rstrip("/") + "/health", timeout=30).raise_for_status()
        flush(url)

    group_name = f"hybrid-direct-reuse-{uuid.uuid4().hex}"
    group_started = time.perf_counter()
    concurrent_posts(
        args.source,
        args.destination,
        "/agentshift/hybrid_state/direct_group/init",
        {
            "master_address": "127.0.0.1",
            "port": args.rendezvous_port,
            "group_rank": 0,
            "world_size": 2,
            "group_name": group_name,
            "backend": "nccl",
        },
        {
            "master_address": "127.0.0.1",
            "port": args.rendezvous_port,
            "group_rank": 1,
            "world_size": 2,
            "group_name": group_name,
            "backend": "nccl",
        },
    )
    group_wall_seconds = time.perf_counter() - group_started

    records = []
    for repetition in range(args.repetitions):
        flush(args.source)
        flush(args.destination)
        base = 100000 + repetition * 2000
        branch_ids = [
            base + ((index * 47 + repetition) % 120000)
            for index in range(args.prefix_tokens)
        ]
        continuation_ids = [
            200000 + ((index * 61 + repetition) % 40000)
            for index in range(args.continuation_tokens)
        ]
        continuation_input = branch_ids + continuation_ids
        generate(args.source, branch_ids, 1)

        transfer_started = time.perf_counter()
        source_transfer, destination_transfer = concurrent_posts(
            args.source,
            args.destination,
            "/agentshift/hybrid_state/direct_transfer",
            {
                "role": "source",
                "token_ids": branch_ids,
                "group_name": group_name,
                "port": args.rendezvous_port,
            },
            {
                "role": "destination",
                "token_ids": branch_ids,
                "group_name": group_name,
                "port": args.rendezvous_port,
            },
        )
        transfer_seconds = time.perf_counter() - transfer_started
        source_result = generate(
            args.source, continuation_input, args.continuation_tokens
        )
        destination_result = generate(
            args.destination, continuation_input, args.continuation_tokens
        )
        checks = {
            "exact_match": source_result["output_ids"]
            == destination_result["output_ids"],
            "source_full_hit": source_result["meta_info"]["cached_tokens"]
            == len(branch_ids),
            "destination_full_hit": destination_result["meta_info"]["cached_tokens"]
            == len(branch_ids),
            "bytes_match": source_transfer["bytes_transferred"]
            == len(branch_ids) * 20480 + 64389120,
        }
        records.append(
            {
                "repetition": repetition,
                "transfer_wall_seconds": transfer_seconds,
                "source_scheduler_seconds": source_transfer["elapsed_seconds"],
                "destination_scheduler_seconds": destination_transfer[
                    "elapsed_seconds"
                ],
                "bytes_transferred": source_transfer["bytes_transferred"],
                "effective_gib_s": source_transfer["bytes_transferred"]
                / 1024**3
                / transfer_seconds,
                "checks": checks,
            }
        )
        print(records[-1])
        if not all(checks.values()):
            raise SystemExit(f"repetition {repetition} failed checks")

    result = {
        "group_name": group_name,
        "group_init_wall_seconds": group_wall_seconds,
        "prefix_tokens": args.prefix_tokens,
        "repetitions": args.repetitions,
        "median_transfer_seconds": statistics.median(
            item["transfer_wall_seconds"] for item in records
        ),
        "median_effective_gib_s": statistics.median(
            item["effective_gib_s"] for item in records
        ),
        "records": records,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"direct-reuse-{args.prefix_tokens}.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(f"artifact: {output_path}")


if __name__ == "__main__":
    main()
