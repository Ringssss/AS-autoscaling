#!/usr/bin/env python3
"""Generate a simple measured motivation figure with no AgentShift result."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRATEGIES = ("sticky", "reroute")
COLORS = {"sticky": "#6E6E6E", "reroute": "#C65D21"}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Tinos",
                "Nimbus Roman",
                "Liberation Serif",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "font.size": 11.5,
            "axes.labelsize": 13.0,
            "axes.titlesize": 12.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 12.5,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.45,
            "lines.markersize": 6.0,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.48,
            "grid.linestyle": "--",
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
            "figure.dpi": 120,
            "hatch.linewidth": 0.55,
        }
    )


def percentile(values: list[int], quantile: float) -> int:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def load_flowprefill(path: Path, max_turn: int) -> tuple[list[dict], dict]:
    by_turn: dict[int, list[int]] = defaultdict(list)
    requests = 0
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            requests += 1
            turn = int(row.get("turn", 1))
            if 1 <= turn <= max_turn:
                by_turn[turn].append(len(row["hash_ids"]) * 16)

    rows = []
    for turn in sorted(by_turn):
        values = by_turn[turn]
        rows.append(
            {
                "turn": turn,
                "count": len(values),
                "p50_tokens": percentile(values, 0.50),
                "p90_tokens": percentile(values, 0.90),
            }
        )
    return rows, {"requests": requests, "block_tokens": 16}


def load_latency(path: Path) -> dict[str, dict[int, dict]]:
    payload = json.loads(path.read_text())
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in payload["records"]:
        strategy = row.get("scenario")
        prefix = int(row.get("prefix_length", 0))
        if strategy not in STRATEGIES or prefix not in (4096, 16384, 32768):
            continue
        if float(row.get("gap_ms", 0.0)) != 500.0:
            continue
        grouped[(strategy, prefix)].append(float(row["post_tool_seconds"]) * 1000.0)

    result: dict[str, dict[int, dict]] = {strategy: {} for strategy in STRATEGIES}
    for (strategy, prefix), samples in grouped.items():
        result[strategy][prefix] = {
            "samples_ms": samples,
            "mean_ms": statistics.fmean(samples),
            "min_ms": min(samples),
            "max_ms": max(samples),
        }
    missing = [
        f"{strategy}:{prefix}"
        for strategy in STRATEGIES
        for prefix in (4096, 16384, 32768)
        if prefix not in result[strategy]
    ]
    if missing:
        raise ValueError(f"missing measurements: {', '.join(missing)}")
    return result


def load_queue(path: Path, returning_agents: int) -> dict[str, dict]:
    payload = json.loads(path.read_text())
    result = {}
    for source_load in ("idle", "busy"):
        key = f"{source_load}:{returning_agents}"
        if key not in payload["summary"]:
            raise ValueError(f"missing queue measurement: {key}")
        result[source_load] = payload["summary"][key]
    return result


def clean_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, which="major", axis="both")
    axis.grid(False, which="minor", axis="both")
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", which="major", length=3.0, pad=2.5)
    axis.tick_params(axis="both", which="minor", length=1.7)


def make_figure(
    trace_rows: list[dict],
    trace_meta: dict,
    latency: dict[str, dict[int, dict]],
    queue: dict[str, dict],
    output_dir: Path,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.75))
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.31, top=0.95, wspace=0.27)

    turns = [row["turn"] for row in trace_rows]
    for field, label, color, marker in (
        ("p50_tokens", "p50", "#6E6E6E", "o"),
        ("p90_tokens", "p90", "#3F78B5", "s"),
    ):
        axes[0].plot(
            turns,
            [row[field] / 1024.0 for row in trace_rows],
            label=label,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.0,
        )
    axes[0].set_xlim(1, 10)
    axes[0].set_xticks(turns)
    axes[0].set_xlabel("Conversation turn")
    axes[0].set_ylabel("Cumulative prefix length (K tokens)")
    reroute_axis = axes[0].twinx()
    measured_prefixes = np.array([4.0, 16.0, 32.0])
    measured_reroute_ms = np.array(
        [latency["reroute"][prefix]["mean_ms"] for prefix in (4096, 16384, 32768)]
    )
    p90_prefixes = np.array(
        [row["p90_tokens"] / 1024.0 for row in trace_rows]
    )
    reroute_by_turn_ms = np.interp(
        p90_prefixes, measured_prefixes, measured_reroute_ms
    )
    reroute_line = reroute_axis.plot(
        turns,
        reroute_by_turn_ms,
        label="Reroute TTFT (p90)",
        color=COLORS["reroute"],
        marker="^",
        markerfacecolor="white",
        markeredgewidth=1.0,
        linestyle="--",
    )[0]
    reroute_axis.set_ylabel("Reroute TTFT (ms)", color=COLORS["reroute"])
    reroute_axis.tick_params(
        axis="y", colors=COLORS["reroute"], length=3.0, pad=2.5
    )
    reroute_axis.spines["top"].set_visible(False)
    reroute_axis.spines["right"].set_color(COLORS["reroute"])
    reroute_axis.set_ylim(250, 600)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles + [reroute_line],
        labels + [reroute_line.get_label()],
        loc="upper left",
        ncol=1,
        handlelength=1.7,
        labelspacing=0.35,
    )
    axes[0].text(
        0.98,
        0.05,
        f"{trace_meta['requests']:,} production requests",
        transform=axes[0].transAxes,
        ha="right",
        va="bottom",
        fontsize=10.0,
        color="#555555",
    )
    clean_axis(axes[0])

    choice_labels = [
        "Stay: source idle\nKV hit",
        "Reroute: dest. idle\nNo hit",
        "Stay: source full\nKV hit",
    ]
    choice_values = [
        queue["idle"]["ttft_mean_ms"] / 1000.0,
        latency["reroute"][4096]["mean_ms"] / 1000.0,
        queue["busy"]["ttft_mean_ms"] / 1000.0,
    ]
    choice_colors = ["#6E6E6E", COLORS["reroute"], "#A33A2B"]
    choice_markers = ["o", "D", "s"]
    axes[1].set_yscale("log")
    axes[1].vlines(
        range(3),
        ymin=0.08,
        ymax=choice_values,
        colors=choice_colors,
        linewidth=1.2,
        zorder=2,
    )
    for index, (value, color, marker) in enumerate(
        zip(choice_values, choice_colors, choice_markers)
    ):
        axes[1].plot(
            index,
            value,
            marker=marker,
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.25,
            markersize=7.2,
            linestyle="none",
            zorder=3,
        )
        axes[1].text(
            index,
            value * 1.18,
            f"{value:.2f}s",
            ha="center",
            va="bottom",
            color=color,
            fontsize=10.2,
        )
    axes[1].set_xlim(-0.55, 2.55)
    axes[1].set_ylim(0.08, 10.5)
    axes[1].set_xticks(range(3), choice_labels)
    axes[1].set_yticks([0.1, 1.0, 10.0], ["0.1", "1", "10"])
    axes[1].set_ylabel("Mean post-tool TTFT (s)")
    clean_axis(axes[1])

    axes[0].text(
        0.5,
        -0.32,
        "(a) Longer histories make rerouting slower",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=12.8,
    )
    axes[1].text(
        0.5,
        -0.32,
        "(b) Locality cannot avoid source queues",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=12.8,
    )

    pdf_path = output_dir / "fig_motivation.pdf"
    png_path = output_dir / "fig_motivation.png"
    fig.savefig(pdf_path, format="pdf", dpi=600, facecolor="white")
    fig.savefig(png_path, format="png", dpi=600, facecolor="white")
    plt.close(fig)
    return pdf_path, png_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flowprefill",
        type=Path,
        default=Path(
            "/home/zhujianian/eurosys/paper1/external/FlowPrefill/trace_build/"
            "qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl"
        ),
    )
    parser.add_argument(
        "--latency",
        type=Path,
        default=Path(
            "results/motivation-qwen32b/"
            "blocked-window-1785913618306267535.json"
        ),
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path(
            "results/motivation-qwen32b/"
            "sticky-queue-qwen32b-1785913931842415038.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("paper_rewriting_output/figures")
    )
    parser.add_argument(
        "--paper-figure-dir",
        type=Path,
        default=Path("paper_rewriting_output/final_paper/figures"),
    )
    parser.add_argument("--max-turn", type=int, default=10)
    parser.add_argument("--queue-returning-agents", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.paper_figure_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    trace_rows, trace_meta = load_flowprefill(args.flowprefill, args.max_turn)
    latency = load_latency(args.latency)
    queue = load_queue(args.queue, args.queue_returning_agents)
    pdf_path, png_path = make_figure(
        trace_rows, trace_meta, latency, queue, args.output_dir
    )

    data_path = args.output_dir / "fig_motivation_data.json"
    data_path.write_text(
        json.dumps(
            {
                "inputs": {
                    "flowprefill": str(args.flowprefill.resolve()),
                    "latency": str(args.latency.resolve()),
                    "queue": str(args.queue.resolve()),
                },
                "definitions": {
                    "trace_prefix": "number of anonymized hash blocks times 16 tokens",
                    "next_turn_latency": "time from result availability to the next turn's first output token",
                    "error_bars": "minimum and maximum across three independent repeats",
                    "queue_latency": "streaming-client TTFT for warm agents that stay on the source",
                    "reroute_by_turn": "linear interpolation of measured Qwen3-32B reroute TTFT at the trace p90 prefix for each turn",
                },
                "configuration": {
                    "model": "Qwen3-32B",
                    "tensor_parallelism": 4,
                    "prefix_tokens": [4096, 16384, 32768],
                    "blocked_interval_ms": 500,
                    "queue_prefix_tokens": 4096,
                    "queue_returning_agents": args.queue_returning_agents,
                    "queue_source_running_slots": 16,
                    "queue_busy_active_decodes": 16,
                    "queue_background_output_tokens": 128,
                },
                "trace": {"metadata": trace_meta, "prefix_by_turn": trace_rows},
                "latency": latency,
                "queue": queue,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for source in (pdf_path, png_path):
        (args.paper_figure_dir / source.name).write_bytes(source.read_bytes())
    print(json.dumps({"pdf": str(pdf_path), "png": str(png_path), "data": str(data_path)}, indent=2))


if __name__ == "__main__":
    main()
