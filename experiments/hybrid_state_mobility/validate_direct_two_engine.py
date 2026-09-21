#!/usr/bin/env python3
"""Validate direct NCCL mobility of a Qwen3.5 hybrid-state bundle."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


def generate(url: str, input_ids: list[int], output_tokens: int) -> dict:
    started = time.perf_counter()
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
    result = response.json()
    return {
        "client_seconds": time.perf_counter() - started,
        "output_ids": result["output_ids"],
        "meta_info": result["meta_info"],
    }


def post(url: str, path: str, payload: dict) -> dict:
    response = requests.post(url.rstrip("/") + path, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def flush(url: str) -> None:
    response = requests.post(url.rstrip("/") + "/flush_cache", timeout=30)
    response.raise_for_status()


def call_both(
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
    parser.add_argument("--rendezvous-port", type=int, default=29710)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="direct_two_engine_result.json")
    args = parser.parse_args()

    for url in (args.source, args.destination):
        response = requests.get(url.rstrip("/") + "/health", timeout=30)
        response.raise_for_status()
        flush(url)

    branch_ids = [
        100000 + ((index * 41) % 120000) for index in range(args.prefix_tokens)
    ]
    continuation_ids = [
        200000 + ((index * 59) % 40000) for index in range(args.continuation_tokens)
    ]
    continuation_input = branch_ids + continuation_ids

    source_first = generate(args.source, branch_ids, 1)
    cold_destination = generate(
        args.destination, continuation_input, args.continuation_tokens
    )
    flush(args.destination)

    group_name = f"hybrid-direct-{uuid.uuid4().hex}"
    group_started = time.perf_counter()
    source_group, destination_group = call_both(
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

    transfer_started = time.perf_counter()
    source_transfer, destination_transfer = call_both(
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
    transfer_wall_seconds = time.perf_counter() - transfer_started

    source_continuation = generate(
        args.source, continuation_input, args.continuation_tokens
    )
    imported_continuation = generate(
        args.destination, continuation_input, args.continuation_tokens
    )

    checks = {
        "exact_import_match": source_continuation["output_ids"]
        == imported_continuation["output_ids"],
        "exact_cold_match": source_continuation["output_ids"]
        == cold_destination["output_ids"],
        "source_full_prefix_hit": source_continuation["meta_info"]["cached_tokens"]
        == len(branch_ids),
        "imported_full_prefix_hit": imported_continuation["meta_info"]["cached_tokens"]
        == len(branch_ids),
        "cold_zero_prefix_hit": cold_destination["meta_info"]["cached_tokens"] == 0,
        "source_transfer_bytes_match": source_transfer["bytes_transferred"]
        == len(branch_ids) * 20480 + 64389120,
        "destination_transfer_bytes_match": destination_transfer["bytes_transferred"]
        == len(branch_ids) * 20480 + 64389120,
    }
    result = {
        "source": args.source,
        "destination": args.destination,
        "prefix_tokens": len(branch_ids),
        "continuation_tokens": args.continuation_tokens,
        "group_name": group_name,
        "group_init": {
            "source": source_group,
            "destination": destination_group,
            "wall_seconds": group_wall_seconds,
        },
        "transfer": {
            "source": source_transfer,
            "destination": destination_transfer,
            "wall_seconds": transfer_wall_seconds,
            "effective_gib_s": source_transfer["bytes_transferred"]
            / 1024**3
            / transfer_wall_seconds,
        },
        "source_first": source_first,
        "cold_destination": cold_destination,
        "source_continuation": source_continuation,
        "imported_continuation": imported_continuation,
        "checks": checks,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(checks, indent=2))
    print(
        f"transfer={transfer_wall_seconds:.6f}s "
        f"effective={result['transfer']['effective_gib_s']:.2f} GiB/s"
    )
    print(f"source output:   {source_continuation['output_ids']}")
    print(f"imported output: {imported_continuation['output_ids']}")
    print(f"cold output:     {cold_destination['output_ids']}")
    print(f"artifact: {output_path}")
    if not all(checks.values()):
        raise SystemExit("direct hybrid-state checks failed")


if __name__ == "__main__":
    main()
