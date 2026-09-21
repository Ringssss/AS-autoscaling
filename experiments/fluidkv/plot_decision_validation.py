#!/usr/bin/env python3
"""Standalone figures from observed calibration, workload and paired run data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.fluidkv.report_decision_validation import ancestor_file, build_report, finite, read_jsonl


COLORS = {"cold": "#27629B", "warm": "#168270", "copy_plus_child": "#C66624"}
LABELS = {"cold": "Cold reroute", "warm": "Existing warm state",
          "copy_plus_child": "Copy + child TTFT (full cost)"}
NATIVE_TRACES = {"qwen_traceA_blksz_16.jsonl", "qwen_coder_blksz_16.jsonl", "qwen_thinking_blksz_16.jsonl"}


def provenance(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            digest.update(chunk)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def calibration_points(records):
    points, excluded = [], []
    for row in records:
        if row.get("phase") != "measured":
            continue
        if not row.get("success"):
            excluded.append({"prefix_tokens": row.get("prefix_tokens"), "repetition": row.get("repetition"),
                             "reason": "failed_measured_pair", "error": row.get("error")})
            continue
        modes = row.get("results", {})
        if not {"cold-reroute", "sticky", "prepared-full-copy"} <= modes.keys():
            excluded.append({"prefix_tokens": row.get("prefix_tokens"), "repetition": row.get("repetition"),
                             "reason": "incomplete_measured_pair"})
            continue
        copy = modes["prepared-full-copy"]
        point = {"prefix_tokens": row["prefix_tokens"], "suffix_tokens": row["suffix_tokens"],
                 "repetition": row["repetition"], "parent_sha256": row.get("parent_sha256"),
                 "cold": modes["cold-reroute"]["child"]["ttft_seconds"],
                 "warm": modes["sticky"]["child"]["ttft_seconds"],
                 "copy_prepare_seconds": copy["preparation_wall_seconds"],
                 "copy_child_ttft_seconds": copy["child"]["ttft_seconds"],
                 "copy_plus_child": copy["preparation_wall_seconds"]+copy["child"]["ttft_seconds"],
                 "included_in_median_guide": row["prefix_tokens"] >= 1024}
        if not all(finite(point[key]) for key in COLORS):
            excluded.append({"prefix_tokens": row["prefix_tokens"], "repetition": row["repetition"],
                             "reason": "nonfinite_measured_pair"})
            continue
        points.append(point)
    return points, excluded


def native_pairs(report):
    conditions = {run["condition_id"]: run["condition"] for run in report["runs"]}
    return [{**pair, "condition": conditions[pair["condition_id"]]}
            for pair in report["paired_comparisons"]
            if pair["condition_id"] in conditions
            and Path(conditions[pair["condition_id"]].get("trace") or "").name in NATIVE_TRACES]


def save_figure(fig, output, stem):
    paths = {}
    for extension in ("svg", "png"):
        path = output/f"{stem}.{extension}"
        fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
        paths[extension] = str(path.resolve())
    plt.close(fig)
    return paths


def calibration_figure(points, output, backend, storage):
    lengths = sorted({point["prefix_tokens"] for point in points})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), gridspec_kw={"width_ratios": [1.3, 1]})
    for axis, shown, title in ((axes[0], lengths, "All measured lengths"),
                               (axes[1], [n for n in lengths if n <= 8192], "Short prefixes and the noisy 256-token point")):
        for mode_index, (metric, color) in enumerate(COLORS.items()):
            medians, guide_lengths = [], []
            for length in shown:
                observations = sorted((p for p in points if p["prefix_tokens"] == length), key=lambda p: p["repetition"])
                values = [1000*p[metric] for p in observations]
                center = length*(1+(mode_index-1)*0.06)
                offsets = np.linspace(-0.015, 0.015, len(values))
                axis.scatter(center*(1+offsets), values, s=22, color=color, alpha=0.65, zorder=4,
                             edgecolors="white", linewidths=0.3)
                median = float(np.median(values))
                axis.errorbar(center, median, yerr=[[median-min(values)], [max(values)-median]],
                              fmt="_", markersize=10, capsize=3, color=color, linewidth=1.2, zorder=3)
                if length >= 1024:
                    medians.append(median)
                    guide_lengths.append(length)
            axis.plot(guide_lengths, medians, color=color, linewidth=1.5, label=LABELS[metric], zorder=2)
        axis.set_xscale("log", base=2)
        axis.set_xticks(shown, [f"{n/1024:g}k" if n % 1024 == 0 else str(n) for n in shown])
        axis.set_ylim(bottom=0)
        axis.set_xlabel("Parent input tokens (+64-token child suffix)")
        axis.set_ylabel("Time to first child token (ms)")
        axis.set_title(title, fontsize=11)
        axis.grid(axis="y", alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Synthetic service measurements: preparation is included in copy cost", fontsize=14, y=1.02)
    graph = "CUDA graphs enabled" if backend and backend.get("disable_cuda_graph") is False else "backend configuration in figure-data.json"
    durable = ("local ext4 metadata; SQLite FULL/WAL" if storage and storage.get("sqlite_synchronous") == 2
               and str(storage.get("sqlite_journal_mode", "")).lower() == "wal" and "ext4" in storage.get("filesystem", "")
               else "metadata provenance in figure-data.json")
    model = Path((backend or {}).get("model_path") or "model recorded in provenance").name
    backends = f"{(backend or {}).get('attention_backend', '?')}/{(backend or {}).get('mamba_backend', '?')}"
    counts = {length: sum(p["prefix_tokens"] == length for p in points) for length in lengths}
    count_text = "n=3 paired measured repetitions at every length" if set(counts.values()) == {3} else f"Measured pairs per length: {counts}"
    fig.text(0.02, -0.035, f"{model}; {backends}; {graph}; {durable}.\n"
             f"{count_text}. Dots: individual runs; whiskers: min–max; lines: median guides (≥1,024 tokens).\n"
             "Copy series = coordinator preparation + child TTFT; groups were preinitialized. Common source preparation is outside these intervals.\n"
             "Noisy 256-token measurements remain visible and are omitted only from the guides. No significance or native-cluster claim.",
             fontsize=9, ha="left", va="top")
    fig.tight_layout()
    return save_figure(fig, output, "synthetic-service-cost")


def workload_figure(workload, output, ambiguity=None):
    splits = list(workload["splits"])
    population = []
    windows = []
    for split, item in workload["splits"].items():
        pop = item["population"]
        population.append({"split": split, "eligible": pop["snapshot_eligible_children"],
                           "requests": pop["requests"], "linked": pop["linked_requests"],
                           "unknown_tail": (ambiguity or {}).get("splits", {}).get(split, {}).get("population", {}).get("counts", {}).get("append_compatible_unknown_tail")})
        for window in item["windows"]:
            if window["duration_seconds"] == 60 and window["warmup_seconds"] == 60:
                measured = window["measurement"]
                windows.append({"split": split, "start_seconds": window["start_seconds"],
                                "eligible": measured["snapshot_eligible_parent_in_replay"],
                                "requests": measured["requests"], "manifest": window.get("manifest"),
                                "unknown_tail": next((item["parent_retained_in_replay"]["counts"]["append_compatible_unknown_tail"]
                                    for item in (ambiguity or {}).get("splits", {}).get(split, {}).get("windows", [])
                                    if item["start_seconds"] == window["start_seconds"] and item["duration_seconds"] == 60), None)})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), gridspec_kw={"width_ratios": [1, 1.65]})
    values = [100*row["eligible"]/row["requests"] for row in population]
    bars = axes[0].bar(splits, values, color="#60768A", width=0.55)
    if ambiguity:
        axes[0].bar(splits, [100*row["unknown_tail"]/row["requests"] for row in population], bottom=values,
                    color="#DADADA", edgecolor="#777777", hatch="///", width=0.55,
                    label="Complete blocks match; final fragment unknown")
    for bar, row in zip(bars, population):
        if ambiguity:
            height = 100*(row["eligible"]+row["unknown_tail"])/row["requests"]
            label = f"{row['eligible']:,} certified\n{row['unknown_tail']:,} unknown"
        else:
            height, label = bar.get_height(), f"{row['eligible']:,}/{row['requests']:,}"
        axes[0].text(bar.get_x()+bar.get_width()/2, height+0.3, label, ha="center", fontsize=8)
    axes[0].set_title("Full original split", fontsize=12)
    starts = sorted({row["start_seconds"] for row in windows})
    width = 0.24
    for index, start in enumerate(starts):
        shown = [next((row for row in windows if row["split"] == split and row["start_seconds"] == start), None) for split in splits]
        position = np.arange(len(splits))+(index-(len(starts)-1)/2)*width
        valid = [(x, row) for x, row in zip(position, shown) if row and row["requests"]]
        if not valid:
            continue
        bars = axes[1].bar([x for x, _ in valid], [100*row["eligible"]/row["requests"] for _, row in valid],
                           width=width*0.92, color=["#27629B", "#168270", "#C66624"][index % 3],
                           label=f"Start {start:g}s")
        if ambiguity:
            axes[1].bar([x for x, _ in valid], [100*row["unknown_tail"]/row["requests"] for _, row in valid],
                        bottom=[100*row["eligible"]/row["requests"] for _, row in valid],
                        width=width*0.92, color="#DADADA", edgecolor="#777777", hatch="///", linewidth=0.6)
        for bar, (_, row) in zip(bars, valid):
            height = bar.get_height()+(100*row["unknown_tail"]/row["requests"] if ambiguity else 0)
            label = f"{row['eligible']} + {row['unknown_tail']}?" if ambiguity else f"{row['eligible']}/{row['requests']}"
            axes[1].text(bar.get_x()+bar.get_width()/2, height+0.3, label, ha="center", fontsize=8)
    axes[1].set_xticks(range(len(splits)), splits)
    axes[1].set_title("Prespecified 60s windows + 60s warmup", fontsize=12)
    axes[1].legend(loc="upper right", frameon=False, fontsize=9)
    max_y = max([100*(row["eligible"]+(row["unknown_tail"] or 0))/row["requests"] for row in population+windows if row["requests"]], default=0)
    for axis in axes:
        axis.set_ylabel("Parent relationship identity / all requests (%)")
        axis.set_ylim(0, max_y*1.2+1)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Strict whole-parent certification is limited by anonymous final fragments", fontsize=14, y=1.02)
    fig.text(0.02, -0.015, "Solid: whole parent certified. Hatched: every complete parent block matches, but final 1–15 original tokens are unknown (not presumed equal).\n"
             "Window labels: certified + unknown?; parents must occur in replay. All requests, including background, remain in percentage denominators.\n"
             "Low solid coverage does not show rare original reuse. Matching identity does not prove that a legal runtime checkpoint was retained.",
             fontsize=9, va="top")
    fig.tight_layout()
    return save_figure(fig, output, "exact-parent-coverage"), {"population": population, "windows": windows}


def paired_figure(pairs, output):
    metrics = [("population_slo_attainment", 100, "Eligible-TTFT SLO Δ (percentage points)"),
               ("population_slo_goodput", 1, "Eligible-TTFT goodput Δ (requests/s)"),
               ("model_occupied_gpu_seconds_including_group_setup", 1, "Resident GPU-seconds Δ (including setup)")]
    available = [(name, scale, label) for name, scale, label in metrics
                 if any(pair["metrics"].get(name, {}).get("n", 0) for pair in pairs)]
    if not available:
        return None
    fig, axes = plt.subplots(1, len(available), figsize=(5.3*len(available), max(4.0, 0.6*len(pairs)+1.8)), squeeze=False)
    for axis, (name, scale, label) in zip(axes[0], available):
        labels = []
        for position, pair in enumerate(pairs):
            condition = pair["condition"]
            split = Path(condition["trace"]).stem.replace("qwen_", "").replace("_blksz_16", "")
            data = pair["metrics"].get(name, {})
            labels.append(f"{split}, t={condition.get('start')}s, ×{condition.get('time_scale')}\n{pair['policy']} (n={data.get('n', 0)})")
            observations = [scale*row["delta"] for row in data.get("paired_deltas", [])]
            if not observations:
                continue
            color = "#C66624" if pair["policy"] == "full-copy" else "#168270"
            median = float(np.median(observations))
            axis.hlines(position, min(observations), max(observations), color=color, linewidth=2)
            axis.scatter(observations, position+np.linspace(-0.1, 0.1, len(observations)), s=25, color=color, alpha=0.6)
            axis.scatter([median], [position], marker="|", s=130, color=color)
        axis.axvline(0, color="#777777", linewidth=0.8, linestyle="--")
        axis.set_yticks(range(len(pairs)), labels, fontsize=8)
        axis.invert_yaxis()
        axis.set_xlabel(label, fontsize=10)
        axis.grid(axis="x", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Native continuous windows: paired run deltas relative to DualMap", fontsize=14, y=1.01)
    fig.text(0.02, -0.035, "Variant minus baseline within identical window, rate, backend and retirement settings. Dots: paired repetitions; bar: min–max; tick: median.\n"
             "Positive SLO/goodput and negative GPU-seconds favor the variant. Raw residency includes group setup, not a complete restart/demand cycle.\n"
             "No significance claim. Accounted timeouts remain flagged in figure data; missing metrics are omitted, not assigned zero.", fontsize=9, va="top")
    fig.tight_layout()
    return save_figure(fig, output, "native-paired-deltas")


def generate_figures(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    data = {"schema_version": 1, "generated_unix": report.get("generated_unix"),
            "report_root": report["root"], "report_status": report["status"],
            "figures": {}, "provenance": [provenance(__file__)], "skipped": [],
            "interpretation": "Descriptive observations; no significance, causal consumption, or universal cluster-performance inference."}
    if report.get("calibration"):
        profile_path = Path(report["calibration"]["path"])
        folder = profile_path.parent
        raw_path = folder/"paired-warmed"/"records.jsonl"
        if raw_path.exists():
            issues = []
            points, excluded = calibration_points(read_jsonl(raw_path, issues))
            backend_path, storage_path = folder/"backend_validation.json", folder/"paired-warmed"/"metadata_storage.json"
            backend = json.loads(backend_path.read_text()) if backend_path.exists() else {}
            storage = json.loads(storage_path.read_text()) if storage_path.exists() else {}
            data["calibration"] = {"points": points, "excluded_measured_pairs": excluded,
                                   "read_issues": issues, "backend": backend, "storage": storage,
                                   "copy_cost_scope": "Per-pair coordinator preparation_wall_seconds + migrated child ttft_seconds; common source preparation and preinitialized communication-group setup excluded.",
                                   "guide_scope": "Noisy 256-token raw points retained; guides connect medians only at >=1024. Not a controller-fit assertion."}
            data["provenance"].extend(provenance(path) for path in (profile_path, raw_path, backend_path, storage_path) if path.exists())
            if points:
                config = backend.get("configs", [{}])[0]
                data["figures"]["synthetic-service-cost"] = calibration_figure(points, output, config, storage)
        else:
            data["skipped"].append("Synthetic figure: no raw paired calibration records beside the selected profile.")
    if report.get("workload"):
        figure, coverage = workload_figure(report["workload"], output, report.get("identity_ambiguity"))
        data["coverage"] = coverage
        data["figures"]["exact-parent-coverage"] = figure
        data["provenance"].append(provenance(report["workload"]["path"]))
        if report.get("identity_ambiguity"):
            data["identity_ambiguity"] = report["identity_ambiguity"]
            data["provenance"].append(provenance(report["identity_ambiguity"]["path"]))
    pairs = native_pairs(report)
    data["native_paired_deltas"] = pairs
    paired = paired_figure(pairs, output) if pairs else None
    if paired:
        data["figures"]["native-paired-deltas"] = paired
        run_paths = {item[key] for pair in pairs for metric in pair["metrics"].values()
                     for item in metric.get("paired_deltas", []) for key in ("baseline_run", "variant_run")}
        raw_paths = set()
        root = Path(report["root"])
        for run in sorted(run_paths):
            folder = root/run
            raw_paths.update(path for path in (folder/"summary.json", folder/"requests.jsonl") if path.exists())
            for name in ("arguments.json", "selection.json", "cost_curves.json", "server_info.json"):
                path = ancestor_file(folder, name, root)
                if path:
                    raw_paths.add(path)
        data["provenance"].extend(provenance(path) for path in sorted(raw_paths))
    else:
        data["skipped"].append("Native paired delta figure: no matched native-trace run metrics yet; synthetic smoke and unmatched runs are not substituted.")
    path = output/"figure-data.json"
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    return {"data_path": str(path.resolve()), "figures": data["figures"], "skipped": data["skipped"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, help="Explicit calibration profile or directory; production-calibration-local preferred by default")
    parser.add_argument("--output", type=Path, help="Default ROOT/figures")
    args = parser.parse_args()
    report = build_report(args.root, args.calibration)
    print(json.dumps(generate_figures(report, args.output or args.root/"figures")))


if __name__ == "__main__":
    main()
