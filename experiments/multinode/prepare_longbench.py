"""Prepare reproducible public-text filler; never label it native Kimi tokens."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=65568)
    args = parser.parse_args()
    if args.tokens <= 0:
        raise ValueError("positive token count required")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    with ZipFile(args.archive) as archive:
        member = "data/gov_report.jsonl"
        texts, rows, ids = [], [], []
        for index, raw in enumerate(archive.open(member)):
            texts.append(json.loads(raw)["context"])
            rows.append(index)
            ids = tokenizer.encode("\n\n".join(texts), add_special_tokens=False)
            if len(ids) >= args.tokens:
                break
        if len(ids) < args.tokens:
            raise ValueError("insufficient corpus tokens")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tokens=np.asarray(ids[:args.tokens], dtype=np.int32))
    provenance = {"dataset": "THUDM/LongBench", "archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                  "member": member, "rows": rows, "tokens": args.tokens, "tokenizer": str(args.model),
                  "provenance": "Concatenated public government-report text for component tests; not original Kimi token IDs or conversation lineage."}
    args.output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
