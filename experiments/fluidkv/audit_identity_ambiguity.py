#!/usr/bin/env python3
"""Separate observed complete-block identity from unknowable final fragments."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from experiments.fluidkv.analyze_decision_workloads import QWEN_FILES, load_jsonl, select_continuous_window, validated_index


CATEGORIES = ("confirmed_complete_parent", "append_compatible_unknown_tail",
              "confirmed_complete_block_divergence", "shorter_child_no_confirmed_divergence")


def classify_pair(parent, child):
    plen, clen = parent["input_length"], child["input_length"]
    comparable = min(plen//16, clen//16)
    matched = 0
    for left, right in zip(parent["hash_ids"][:comparable], child["hash_ids"][:comparable]):
        if left != right:
            break
        matched += 1
    category = ("confirmed_complete_block_divergence" if matched < comparable else
                "shorter_child_no_confirmed_divergence" if clen < plen else
                "append_compatible_unknown_tail" if plen % 16 else "confirmed_complete_parent")
    floor64 = plen//64*64
    return {"parent_chat_id": parent["chat_id"], "child_chat_id": child["chat_id"],
            "parent_input_tokens": plen, "child_input_tokens": clen,
            "parent_block_aligned": plen % 16 == 0, "strict_child_extension": clen > plen,
            "category": category, "confirmed_shared_complete_tokens": matched*16,
            "unknown_parent_tail_tokens": plen % 16, "no_complete_parent_blocks": plen < 16,
            "complete_parent_equality_proven": category == "confirmed_complete_parent",
            "floor64_tokens": floor64,
            "floor64_identity_supported": floor64 > 0 and matched*16 >= floor64 and clen > floor64,
            "checkpoint_availability_proven": False}


def summarize_pairs(children, by_id):
    counts = Counter()
    strata = defaultdict(Counter)
    rows = []
    for child in children:
        parent = by_id.get(child["parent_chat_id"])
        if parent is None:
            counts["root_requests" if child["parent_chat_id"] < 0 else "missing_parent"] += 1
            continue
        result = classify_pair(parent, child)
        rows.append(result)
        counts[result["category"]] += 1
        strata["aligned" if result["parent_block_aligned"] else "unaligned"][result["category"]] += 1
    return {"requests": len(children), "linked": len(rows),
            "counts": {key: counts[key] for key in (*CATEGORIES, "root_requests", "missing_parent")},
            "parent_alignment": {key: {name: strata[key][name] for name in CATEGORIES} for key in ("aligned", "unaligned")},
            "confirmed_shared_complete_tokens_sum": sum(row["confirmed_shared_complete_tokens"] for row in rows),
            "linked_parent_input_tokens_sum": sum(row["parent_input_tokens"] for row in rows),
            "floor64_identity_supported_pairs": sum(row["floor64_identity_supported"] for row in rows),
            "no_complete_parent_block_pairs": sum(row["no_complete_parent_blocks"] for row in rows),
            "evidence": rows}


def runtime_notes(runtime_root, server_info):
    infos = json.loads(server_info.read_text())
    configs = []
    for info in infos if isinstance(infos, list) else [infos]:
        settings = info.get("server_args", info)
        configs.append({key: settings.get(key, info.get(key)) for key in (
            "mamba_scheduler_strategy", "mamba_track_interval", "page_size", "chunked_prefill_size",
            "disable_chunked_prefix_cache", "disable_radix_cache")})
    files = {
        "python/sglang/srt/server_args.py": "auto resolves to no_buffer; extra_buffer enables tracking; cache chunk size is max(FLA_CHUNK_SIZE,page_size).",
        "python/sglang/srt/layers/attention/fla/chunk_delta_h.py": "FLA_CHUNK_SIZE is 64.",
        "python/sglang/srt/managers/schedule_batch.py": "Only enable_mamba_extra_buffer calls tracking preparation. Tracked prefill position is prefix_len + floor(extend_len/chunk_size)*chunk_size, possibly replaced by an aligned branching point.",
        "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py": "Extra-buffer tracking extracts recurrent intermediate h and convolution state at the selected aligned boundary.",
        "python/sglang/srt/mem_cache/mamba_radix_cache.py": "no_buffer caches actual fill/committed boundaries; extra_buffer uses mamba_last_track_seqlen. Splitting KV nodes sets mamba_value=None. Matching returns only a valid earlier recurrent boundary.",
        "python/sglang/srt/managers/scheduler.py": "cache_unfinished_req(req,chunked=True) preserves actual chunked-prefill boundaries.",
        "python/sglang/srt/mem_cache/hybrid_state_transfer.py": "Source PREPARED requires exact requested prefix length plus existing mamba_value; it does not derive a past recurrent state from a later state.",
    }
    evidence = [{"path": str((runtime_root/name).resolve()),
                 "sha256": hashlib.sha256((runtime_root/name).read_bytes()).hexdigest(), "finding": finding}
                for name, finding in files.items()]
    return {"server_info": str(server_info.resolve()), "observed_configs": configs, "source_evidence": evidence,
            "conclusion": "Captured engines use no_buffer, page_size=1 and chunked_prefill_size=8192. They do not automatically retain floor(parent_length/64)*64 for every prompt. Actual chunk boundaries and earlier independently created prefix snapshots may exist, while the harness observes only prompt/final boundaries. Extra_buffer has a floor-aligned tracking path (relative to existing prefix/extend split), but switching strategy requires separate capacity, correctness and live-availability validation. No source checkpoint was inspected live in this read-only audit.",
            "live_snapshot_availability_proven": False}


def run_audit(trace_root, output, runtime_root=None, server_info=None):
    output.mkdir(parents=True, exist_ok=True)
    result = {"scope": "Anonymous hash identity audit, not inferred original-tail equality or measured runtime reuse.",
              "critical_caveat": "Low whole-parent certification primarily reflects a partial final hash block. It must not be interpreted as rare original prefix reuse. Complete blocks can match while a partial block hash differs after append; the final fragment remains unknown. No tail tokens are synthesized or equated.",
              "splits": {}}
    lines = ["# Parent identity ambiguity audit", "", result["critical_caveat"], "",
             "A differing hash proves content divergence only when both compared blocks are complete 16-token blocks. For an unaligned parent whose complete blocks all match a child at least as long, its final 1–15 original tokens are unknown. This is compatible with append-only history; it is not proof of that history.", "",
             "| Split | Linked | Confirmed whole parent | Complete blocks match; tail unknown | Confirmed complete-block divergence | Shorter child |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    window_lines = ["", "Fixed measurement windows, with equally long preceding warmup; all context-eligible background retained:", "",
                    "| Split | Start / duration (s) | Linked in source / parent in replay | Confirmed whole parent in replay | Tail-unknown compatible in replay | Complete-block divergence in replay | floor64 identity supported in replay |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for split, filename in QWEN_FILES.items():
        rows, source = load_jsonl(trace_root/filename)
        by_id = validated_index(rows)
        population = summarize_pairs(rows, by_id)
        population["examples"] = {name: [r for r in population["evidence"] if r["category"] == name][:4] for name in CATEGORIES}
        population.pop("evidence")
        windows = []
        for start in (600, 3000, 5400):
            for duration in (30, 60, 120, 300):
                selected, manifest = select_continuous_window(rows, start_seconds=start, duration_seconds=duration)
                selected_ids = {row["chat_id"] for row in selected}
                measured = [row for row in selected if row["timestamp"] >= manifest["measurement_start_timestamp"]]
                all_source = summarize_pairs(measured, by_id)
                retained_children = [row for row in measured if row["parent_chat_id"] in selected_ids]
                in_replay = summarize_pairs(retained_children, by_id)
                all_source.pop("evidence")
                window = {"start_seconds": start, "duration_seconds": duration, "warmup_seconds": duration,
                          "measurement_requests": len(measured), "context_exclusions": len(manifest["excluded"]),
                          "linked_against_full_source": all_source, "parent_retained_in_replay": in_replay}
                windows.append(window)
                c = in_replay["counts"]
                window_lines.append(f"| {split} | {start} / {duration} | {all_source['linked']} / {in_replay['linked']} | {c[CATEGORIES[0]]} | {c[CATEGORIES[1]]} | {c[CATEGORIES[2]]} | {in_replay['floor64_identity_supported_pairs']} |")
        result["splits"][split] = {"source": source, "population": population, "windows": windows}
        c = population["counts"]
        lines.append(f"| {split} | {population['linked']} | {c[CATEGORIES[0]]} | {c[CATEGORIES[1]]} | {c[CATEGORIES[2]]} | {c[CATEGORIES[3]]} |")
        print(json.dumps({"split": split, "counts": c, "floor64_identity_supported": population["floor64_identity_supported_pairs"]}), flush=True)
    lines += window_lines
    lines += ["", "Whole-parent certification is a conservative identity lower bound. Confirmed matching complete blocks are reusable input identity; a corresponding legal recurrent snapshot still needs to exist. The floor64 column tests only whether a hypothetical boundary would lie entirely within confirmed shared blocks. It does not prove that the runtime retained that checkpoint.", ""]
    if runtime_root and server_info:
        result["runtime"] = runtime_notes(runtime_root, server_info)
        lines += [result["runtime"]["conclusion"], "", "Source evidence:", ""]
        lines += [f"- `{item['path']}`: {item['finding']}" for item in result["runtime"]["source_evidence"]]
    lines += ["", "Implication: current surrogate/full-prompt mobility runs characterize a conservative replay implementation. Their low exact-parent count cannot falsify state mobility on original agent histories. Retain current runs as scoped controls; validate recovery at a confirmed earlier boundary or obtain a trace representation with original partial-block identities before drawing a general opportunity conclusion.", ""]
    (output/"identity-ambiguity.json").write_text(json.dumps(result, indent=2)+"\n")
    (output/"identity-ambiguity.md").write_text("\n".join(lines))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--server-info", type=Path)
    args = parser.parse_args()
    run_audit(args.trace_root, args.output, args.runtime_root, args.server_info)


if __name__ == "__main__":
    main()
