#!/usr/bin/env python3
"""Native AgentX structure and conditional sticky-placement mismatch census.

The request lifetimes, local block hashes and stream/model identities are native.
Worker assignment is a declared hash sensitivity, not a recovered deployment.
Observed API intervals model concurrent load signals, never GPU service/queues.
Conversations retain their relative timelines and are analyzed independently:
there is no cross-conversation global arrival reconstruction or time compression.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import json
from pathlib import Path

import numpy as np

from experiments.weka_gravity.audit import flatten


THRESHOLDS = (8192, 32768, 65536)


def pct(values):
    if not values:
        return {}
    return dict(zip(("p10", "p50", "p90", "p99", "max"),
                    np.percentile(values, (10, 50, 90, 99, 100)).tolist()))


def prefix_count(mask):
    misses = np.flatnonzero(~mask)
    return int(misses[0]) if len(misses) else len(mask)


def placement_census(trace, reqs, hashes, sizes, counters, samples):
    """Replay ONLY recorded activity/completions on fixed hypothetical homes."""
    events = sorted((r["t"], i) for i, r in enumerate(reqs))
    for workers in (2, 4, 8):
        for seed in (0, 1, 2):
            homes = [int.from_bytes(hashlib.sha256(
                f"{seed}:{trace['id']}:{r['stream']}:{r['model']}".encode()).digest()[:8],
                "little") % workers for r in reqs]
            # Model separated, conversation-local hashes, timestamps of completed
            # input state. Input/output state formats and eviction are unobserved.
            residency = {m: np.full((workers, size), -np.inf) for m, size in sizes.items()}
            active = {m: np.zeros(workers, dtype=int) for m in sizes}
            finished = []
            for now, i in events:
                while finished and finished[0][0] <= now:
                    done, j = heapq.heappop(finished)
                    old = reqs[j]
                    m, home = old["model"], homes[j]
                    active[m][home] -= 1
                    if old["out"] > 0:
                        residency[m][home, hashes[j]] = done
                r, h = reqs[i], hashes[i]
                m, source = r["model"], homes[i]
                counts = active[m]
                target = min(range(workers), key=lambda j: (counts[j], j))
                for ttl in (300, None):
                    tag = f"n{workers}-seed{seed}-ttl{ttl if ttl else 'unlimited'}"
                    c = counters[tag]
                    c["requests"] += 1
                    c["context_fit_requests"] += r["in"] + r["out"] <= 131072 and r["out"] > 0
                    age = now - residency[m][source, h]
                    hit = prefix_count(age <= ttl if ttl else np.isfinite(age)) * 64
                    if hit == 0:
                        continue
                    dest_age = now - residency[m][target, h]
                    dest_hit = prefix_count(dest_age <= ttl if ttl else np.isfinite(dest_age)) * 64
                    for threshold in THRESHOLDS:
                        if hit < threshold:
                            continue
                        c[f"warm_source_{threshold}"] += 1
                        mismatch = target != source and counts[source] > counts[target] and hit > dest_hit
                        if mismatch:
                            c[f"load_mismatch_missing_state_{threshold}"] += 1
                            c[f"missing_state_tokens_{threshold}"] += hit - dest_hit
                            fit = r["in"] + r["out"] <= 131072 and r["out"] > 0
                            c[f"fit_load_mismatch_{threshold}"] += fit
                            if counts[target] == 0:
                                c[f"idle_target_mismatch_{threshold}"] += 1
                            if (workers == 4 and seed == 0 and ttl == 300 and threshold == 32768
                                    and len(samples) < 60):
                                samples.append({"conversation": trace["id"], "location": r["location"],
                                    "model": m, "stream": r["stream"], "arrival": now,
                                    "input_tokens": r["in"], "source": source, "target": target,
                                    "observed_active_by_hypothetical_worker": counts.tolist(),
                                    "completed_source_input_prefix": hit, "target_prefix": dest_hit,
                                    "fits_131k_input_plus_output": fit})
                if r["api_time"] > 0:
                    counts[source] += 1
                    heapq.heappush(finished, (now + r["api_time"], i))


def analyze(args):
    args.out.mkdir(parents=True, exist_ok=False)
    total = Counter()
    gaps, lcp_values, peak_values, run_lengths = [], [], [], []
    long_gap_counts = Counter()
    overlap_counts = Counter()
    per_model = defaultdict(Counter)
    placement = defaultdict(Counter)
    samples = []
    duration_examples = []
    source_sha = hashlib.sha256()
    for line, raw in enumerate(args.trace.open("rb"), 1):
        source_sha.update(raw)
        tr = json.loads(raw)
        if tr["block_size"] != 64 or tr["hash_id_scope"] != "local":
            raise ValueError("unexpected identity semantics")
        reqs, groups = flatten(tr)
        hashes = [np.asarray(r["hash_ids"], dtype=np.int32) for r in reqs]
        sizes = {m: max([int(h.max()) + 1 for r, h in zip(reqs, hashes)
                         if r["model"] == m and len(h)] or [1]) for m in tr["models"]}
        # Verify endpoint IDs imply a unique prefix chain before grouping shared
        # active prefixes by the block at a threshold.
        predecessor = np.full(max(sizes.values()), -2, dtype=np.int32)
        for h in hashes:
            prev = np.r_[-1, h[:-1]]
            known = predecessor[h]
            if np.any((known != -2) & (known != prev)):
                raise ValueError("prefix identity predecessor conflict")
            predecessor[h] = prev
        active = defaultdict(set)
        active_prefix = defaultdict(Counter)
        completion_heap = []
        completed = {}
        previous_arrival = {}
        runs = defaultdict(int)
        peak = 0
        for i, r in enumerate(reqs):
            now, m, key = r["t"], r["model"], (r["stream"], r["model"])
            while completion_heap and completion_heap[0][0] <= now:
                end, j = heapq.heappop(completion_heap)
                p = reqs[j]
                active[p["model"]].remove(j)
                for threshold in THRESHOLDS:
                    b = threshold // 64
                    if len(hashes[j]) >= b:
                        active_prefix[p["model"], threshold][int(hashes[j][b - 1])] -= 1
                if p["out"] > 0:
                    completed[p["stream"], p["model"]] = j
            total["requests"] += 1
            total["input_tokens"] += r["in"]
            total["output_tokens"] += r["out"]
            total["context_fit_requests"] += r["in"] + r["out"] <= 131072 and r["out"] > 0
            model = per_model[m]
            model["requests"] += 1
            if key in previous_arrival:
                p = reqs[previous_arrival[key]]
                total["subsequent_same_stream_model"] += 1
                if p["t"] + p["api_time"] > now:
                    total["overlapping_previous_same_stream_model"] += 1
            previous_arrival[key] = i
            prior = completed.get(key)
            lcp, gap = 0, None
            if prior is not None:
                p = reqs[prior]
                n = min(len(hashes[i]), len(hashes[prior]))
                lcp = prefix_count(hashes[i][:n] == hashes[prior][:n]) * 64
                gap = now - p["t"] - p["api_time"]
                gaps.append(gap)
                lcp_values.append(lcp)
                total["has_completed_same_stream_input"] += 1
                total["completed_previous_lcp_tokens"] += lcp
                model["completed_lcp_tokens"] += lcp
            other = [j for j in active[m] if reqs[j]["stream"] != r["stream"]]
            overlap_counts["arrivals_with_other_stream_same_model_active"] += bool(other)
            for threshold in THRESHOLDS:
                b = threshold // 64
                matching_other_streams = {reqs[j]["stream"] for j in other
                    if len(hashes[i]) >= b and len(hashes[j]) >= b
                    and hashes[i][b - 1] == hashes[j][b - 1]}
                total[f"arrivals_sharing_{threshold}_with_other_stream"] += bool(matching_other_streams)
                if len(hashes[i]) >= b:
                    total[f"max_simultaneous_distinct_streams_same_prefix_{threshold}"] = max(
                        total[f"max_simultaneous_distinct_streams_same_prefix_{threshold}"],
                        len(matching_other_streams) + 1)
                if lcp >= threshold:
                    total[f"completed_reuse_{threshold}"] += 1
                    model[f"completed_reuse_{threshold}"] += 1
                    total[f"completed_reuse_and_other_active_{threshold}"] += bool(other)
                    fit = r["in"] + r["out"] <= 131072 and r["out"] > 0
                    total[f"fit_completed_reuse_{threshold}"] += fit
                    total[f"fit_reuse_and_other_active_{threshold}"] += fit and bool(other)
                    for limit in (0.1, 1., 5., 30.):
                        long_gap_counts[f"reuse{threshold}_gap_ge{limit}"] += gap >= limit
                    # Offline byte/bandwidth sensitivity, not a measured serving
                    # transfer or evidence that migration is required.
                    for byte_per_token in (20480, 131072):
                        for bandwidth in (5e9, 25e9):
                            duration = .02 + (lcp * byte_per_token + 64 * 1024**2) / bandwidth
                            long_gap_counts[f"reuse{threshold}_copyfits_{byte_per_token}B_{int(bandwidth/1e9)}GBs"] += duration <= gap
                b = threshold // 64
                if len(hashes[i]) >= b:
                    sharing = active_prefix[m, threshold][int(hashes[i][b - 1])]
                    total[f"arrivals_sharing_{threshold}_with_active_request"] += sharing > 0
                    total[f"max_simultaneous_same_prefix_{threshold}"] = max(
                        total[f"max_simultaneous_same_prefix_{threshold}"], sharing + 1)
            if lcp >= 32768:
                runs[key] += 1
            elif runs[key]:
                run_lengths.append(runs[key])
                runs[key] = 0
            if r["api_time"] > 0:
                active[m].add(i)
                heapq.heappush(completion_heap, (now + r["api_time"], i))
                for threshold in THRESHOLDS:
                    b = threshold // 64
                    if len(hashes[i]) >= b:
                        active_prefix[m, threshold][int(hashes[i][b - 1])] += 1
                peak = max(peak, sum(len(v) for v in active.values()))
        run_lengths.extend(n for n in runs.values() if n)
        peak_values.append(peak)
        total["conversations"] += 1
        total["subagent_groups"] += len(groups)
        total["streams_model_pairs"] += len(previous_arrival)
        total["conversation_start_is_relative_zero"] += min(r["t"] for r in reqs) == 0
        if args.placement:
            placement_census(tr, reqs, hashes, sizes, placement, samples)
        if line % 25 == 0:
            print("conversations", line, "requests", total["requests"], flush=True)
    report = {"source": str(args.trace), "sha256": source_sha.hexdigest(), "native_counts": dict(total),
              "completed_same_stream_prefix_tokens": pct(lcp_values), "finish_to_next_arrival_gap": pct(gaps),
              "long_reuse_gap_counts": dict(long_gap_counts), "overlap_counts": dict(overlap_counts),
              "per_conversation_peak_logged_requests": pct(peak_values),
              "consecutive_long_reuse_runs": pct(run_lengths),
              "model_counts": {k: dict(v) for k, v in per_model.items()},
              "placement_sensitivity": {k: dict(v) for k, v in placement.items()},
              "interpretation": {
                  "native": "Same-conversation/model input block identities, completed-request reuse, observed overlapping API intervals.",
                  "placement": "Hypothetical stationary stream/model hash homes; cache is unlimited capacity with last-use TTL 300s or no expiry; states admitted only on observed completion. No migration performed.",
                  "mismatch": "Long completed input prefix at hypothetical home, more observed active requests there than on least-active compatible worker, which lacks some of that prefix. Structural candidate, not predicted beneficial migration.",
                  "missing": "Actual worker placement, global conversation starts, hardware service/capacity/batching, physical eviction, generated-output identities, tool semantics and migration interference.",
                  "model_compatibility": "Source model labels treated as distinct compatibility groups; actual checkpoints, precision and TP layouts unavailable.",
                  "timing": "api_time is upstream observed elapsed time, not calibrated GPU service. Relative within-conversation arrivals preserved; no timing compression.",
                  "prefix_scope": "Only completed input state is identifiable. Mamba snapshot boundaries and generated output state are unavailable."}}
    (args.out / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "candidate-examples.json").write_text(json.dumps(samples, indent=2) + "\n")
    print(json.dumps({"native_counts": report["native_counts"],
                      "gap": report["finish_to_next_arrival_gap"],
                      "peak": report["per_conversation_peak_logged_requests"]}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, default=Path("/home/zhujianian/datasets/cc-traces-weka-062126/traces.jsonl"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--placement", action="store_true")
    analyze(p.parse_args())
