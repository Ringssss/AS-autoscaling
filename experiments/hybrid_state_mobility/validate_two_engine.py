#!/usr/bin/env python3
"""Validate a two-engine Qwen3.5 hybrid-state handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import uuid
from pathlib import Path

import requests


def post_json(url: str, payload: dict, timeout: float = 300.0) -> dict:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def generate(url: str, input_ids: list[int], output_tokens: int) -> dict:
    started = time.perf_counter()
    result = post_json(
        url.rstrip("/") + "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": output_tokens,
                "temperature": 0.0,
            },
        },
    )
    return {
        "client_seconds": time.perf_counter() - started,
        "output_ids": result["output_ids"],
        "meta_info": result["meta_info"],
    }


def flush(url: str) -> None:
    response = requests.post(url.rstrip("/") + "/flush_cache", timeout=30)
    response.raise_for_status()


def token_fingerprint(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="http://127.0.0.1:31080")
    parser.add_argument("--destination", default="http://127.0.0.1:31081")
    parser.add_argument("--prefix-tokens", type=int, default=2048)
    parser.add_argument("--continuation-tokens", type=int, default=16)
    parser.add_argument(
        "--ablate-mamba",
        action="store_true",
        help="Install full KV but leave the imported mamba snapshot zeroed.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-name", default="two_engine_result.json")
    args = parser.parse_args()

    if args.prefix_tokens < 2:
        raise ValueError("--prefix-tokens must be at least 2")
    if args.continuation_tokens < 1:
        raise ValueError("--continuation-tokens must be positive")

    for url in (args.source, args.destination):
        response = requests.get(url.rstrip("/") + "/health", timeout=30)
        response.raise_for_status()
        flush(url)

    # MambaRadixCache commits the prompt boundary in this configuration. Export
    # that exact branch and treat the generated first token only as a warmup.
    prompt_ids = [
        100000 + ((index * 37) % 120000) for index in range(args.prefix_tokens)
    ]
    source_first = generate(args.source, prompt_ids, 1)
    branch_ids = prompt_ids

    continuation_ids = [
        200000 + ((index * 53) % 40000) for index in range(args.continuation_tokens)
    ]
    continuation_input = branch_ids + continuation_ids

    cold_destination = generate(
        args.destination, continuation_input, args.continuation_tokens
    )
    flush(args.destination)

    checkpoint_id = f"hybrid-{uuid.uuid4().hex}"
    export_started = time.perf_counter()
    export_result = post_json(
        args.source.rstrip("/") + "/agentshift/export_hybrid_state",
        {"checkpoint_id": checkpoint_id, "token_ids": branch_ids},
    )
    export_wall_seconds = time.perf_counter() - export_started

    import_started = time.perf_counter()
    import_result = post_json(
        args.destination.rstrip("/") + "/agentshift/import_hybrid_state",
        {
            "checkpoint_id": checkpoint_id,
            "token_ids": branch_ids,
            "expected_sha256": export_result["checkpoint_sha256"],
            "ablate_mamba": args.ablate_mamba,
        },
    )
    import_wall_seconds = time.perf_counter() - import_started

    source_continuation = generate(
        args.source, continuation_input, args.continuation_tokens
    )
    imported_continuation = generate(
        args.destination, continuation_input, args.continuation_tokens
    )

    source_output = source_continuation["output_ids"]
    imported_output = imported_continuation["output_ids"]
    cold_output = cold_destination["output_ids"]
    exact_import_match = source_output == imported_output
    exact_cold_match = source_output == cold_output
    source_cached = int(source_continuation["meta_info"]["cached_tokens"])
    imported_cached = int(imported_continuation["meta_info"]["cached_tokens"])
    cold_cached = int(cold_destination["meta_info"]["cached_tokens"])

    result = {
        "source": args.source,
        "destination": args.destination,
        "checkpoint_id": checkpoint_id,
        "prefix_tokens": len(branch_ids),
        "continuation_input_tokens": len(continuation_input),
        "branch_token_sha256": token_fingerprint(branch_ids),
        "source_first_turn": source_first,
        "cold_destination": cold_destination,
        "source_continuation": source_continuation,
        "imported_continuation": imported_continuation,
        "export": export_result | {"wall_seconds": export_wall_seconds},
        "import": import_result | {"wall_seconds": import_wall_seconds},
        "checks": {
            "exact_import_match": exact_import_match,
            "exact_cold_destination_match": exact_cold_match,
            "source_full_prefix_hit": source_cached == len(branch_ids),
            "imported_full_prefix_hit": imported_cached == len(branch_ids),
            "cold_destination_zero_prefix_hit": cold_cached == 0,
            "state_bytes_formula_matches": export_result["state_bytes"]
            == len(branch_ids) * 20480 + 64389120,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / args.output_name
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["checks"], indent=2))
    print(f"source output:      {source_output}")
    print(f"imported output:    {imported_output}")
    print(f"cold output:        {cold_output}")
    print(f"artifact: {output_path}")

    if args.ablate_mamba:
        if exact_import_match:
            raise SystemExit("mamba ablation unexpectedly matched the source")
    elif not all(result["checks"].values()):
        raise SystemExit("hybrid-state correctness checks failed")


if __name__ == "__main__":
    main()
