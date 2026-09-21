#!/usr/bin/env python3
"""Separate transfer equivalence from cached-versus-cold backend behavior."""

import argparse
import json
from pathlib import Path

import numpy as np
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--pair", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:31280")
    args = parser.parse_args()
    pair = json.loads(args.manifest.read_text())["pairs"][args.pair]
    arrays = np.load(args.tokens)
    parent = arrays[str(pair["parent_id"])].tolist()
    child = arrays[str(pair["child_id"])].tolist()

    def flush():
        response = requests.post(args.url + "/flush_cache", timeout=30)
        response.raise_for_status()

    def generate(ids, outputs=8):
        response = requests.post(args.url + "/generate", json={
            "input_ids": ids, "return_logprob": True, "top_logprobs_num": 5,
            "logprob_start_len": -1,
            "sampling_params": {"max_new_tokens": outputs, "temperature": 0, "ignore_eos": True},
        }, timeout=180)
        response.raise_for_status()
        return response.json()

    records = []
    for repetition in range(3):
        flush()
        cold = generate(child)
        flush()
        generate(parent, 1)
        warm = generate(child)
        a, b = cold["output_ids"], warm["output_ids"]
        difference = next((i for i, (left, right) in enumerate(zip(a, b)) if left != right), None)
        records.append({"repetition": repetition, "cold": cold, "warm": warm,
                        "first_different_token": difference})
    result = {
        "scope": "Same engine, same exact input IDs, no transfer or ownership change; cold versus parent-prefilled continuation.",
        "pair": pair, "url": args.url,
        "equal_runs": sum(r["cold"]["output_ids"] == r["warm"]["output_ids"] for r in records),
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"equal_runs": result["equal_runs"], "first_differences": [r["first_different_token"] for r in records]}))


if __name__ == "__main__":
    main()
