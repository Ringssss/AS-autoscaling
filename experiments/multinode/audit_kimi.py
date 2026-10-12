"""Audit the complete release without extracting or inventing prefix identities."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from agentshift.workloads.production import parse_kimi_line

EXPECTED_SHA256 = "e7d8fd50757208220b9bc5bbd111a9d491b94a89a6c42390ab345aab9a0c3513"


def audit(archive, output):
    digest = hashlib.sha256()
    with archive.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError("release SHA-256 mismatch; incomplete or different archive")
    output.mkdir(parents=True, exist_ok=False)
    counts, files, ids, values, ranks = Counter(), [], set(), {}, Counter()
    with tarfile.open(archive, "r|gz") as tar, gzip.open(output / "requests.jsonl.gz", "wt") as out:
        for member in tar:
            if not member.isfile() or not re.fullmatch(r"[0-2][0-9]\.redacted\.log", Path(member.name).name):
                continue
            source_hash, rows = hashlib.sha256(), 0
            for line_no, raw in enumerate(tar.extractfile(member), 1):
                source_hash.update(raw)
                line = raw.decode("utf-8")
                row = parse_kimi_line(line)
                if row is None:
                    counts["non_finish_lines"] += 1
                    continue
                rows += 1
                counts["requests"] += 1
                for name in ("input_ids_redacted", "output_ids_redacted"):
                    counts[name] += row[name]
                counts["nonempty_conversation_ids"] += bool(row["conversation_id"])
                counts["duplicate_request_ids"] += row["request_id"] in ids
                ids.add(row["request_id"])
                for field in ("routing_key", "extra_key", "session_params"):
                    match = re.search(r"\b" + field + r"=([^,)]+)", line)
                    counts[f"nonnull_{field}"] += bool(match and match.group(1) not in ("None", "''", '""'))
                for key in ("input_length", "output_length", "cached_tokens", "arrival_seconds", "finished_seconds", "ttft_seconds"):
                    if row[key] is not None:
                        values.setdefault(key, []).append(row[key])
                counts["cached_gt_input"] += row["cached_tokens"] is not None and row["cached_tokens"] > row["input_length"]
                if row["cached_tokens"] is not None and row["cached_tokens"] > row["input_length"]:
                    counts["cached_gt_input_with_one_input_one_output"] += row["input_length"] == row["output_length"] == 1
                elif row["cached_tokens"] is not None:
                    counts["records_with_plausible_cache_counts"] += 1
                    counts["plausible_cached_token_sum"] += row["cached_tokens"]
                ranks[str(row["dp_rank"])] += 1
                row.update(source_file=member.name, source_line=line_no)
                out.write(json.dumps(row) + "\n")
            files.append({"name": member.name, "uncompressed_bytes": member.size, "rows": rows, "sha256": source_hash.hexdigest()})
            print(f"{Path(member.name).name}: {rows}", flush=True)
    if len(files) != 24 or counts["requests"] == 0:
        raise ValueError("expected 24 nonempty hourly log files")
    arrivals = np.asarray(values["arrival_seconds"])
    lengths = np.asarray(values["input_length"])
    outputs = np.asarray(values["output_length"])
    per_minute = np.bincount(((arrivals - arrivals.min()) // 60).astype(int))
    percentiles = {
        key: dict(zip(("min", "p50", "p90", "p95", "p99", "max"),
                      np.percentile(value, [0, 50, 90, 95, 99, 100]).tolist()))
        for key, value in values.items() if key not in ("arrival_seconds", "finished_seconds")
    }
    report = {
        "archive": str(archive), "archive_sha256": digest.hexdigest(), "counts": dict(counts),
        "sources": files, "arrival_span_seconds": float(arrivals.max() - arrivals.min()),
        "request_rate_per_second": len(arrivals) / float(arrivals.max() - arrivals.min()),
        "requests_per_minute_quantiles": np.percentile(per_minute, [0, 50, 90, 99, 100]).tolist(),
        "percentiles": percentiles, "dp_ranks": dict(ranks),
        "observed_cached_tokens": int(sum(values["cached_tokens"])),
        "observed_input_tokens": int(sum(values["input_length"])),
        "input_plus_recorded_output_fits_context": {
            str(limit): {"requests": int(np.sum(lengths + outputs <= limit)),
                         "fraction": float(np.mean(lengths + outputs <= limit))}
            for limit in (4096, 16384, 32768, 65536, 131072, 262144)
        },
        "native_prefix_routing_replay_supported": False,
        "identifiability": (
            "Cached-token counts describe historical service, not prefix identity at another engine. "
            "Without token/hash identities and stable session/business keys, neither DualMap nor "
            "CacheRoute nor cross-turn migration can be faithfully replayed from these rows. "
            "Content substitution must label both content and reuse/lineage as reconstructed."
        ),
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sources"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.archive, args.output)
