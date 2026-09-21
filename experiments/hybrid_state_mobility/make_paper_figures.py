#!/usr/bin/env python3
"""Generate paper figures for the one-hour Kimi mobility replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


BG = "#fefefe"
BLACK = "#000000"
GRAY = "#7f7f7f"

POLICY_STYLE = {
    "sticky": {"color": "#f3caa3", "label": "Sticky"},
    "stateless-reroute": {"color": "#f9e59a", "label": "Stateless reroute"},
    "token-only": {"color": "#a8c7e9", "label": "Token-only migration"},
    "heterogeneous": {"color": "#5e8ec8", "label": "AgentShift (hybrid)"},
}

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "mathtext.fontset": "dejavusans",
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": BLACK,
        "axes.linewidth": 1.6,
        "xtick.color": BLACK,
        "ytick.color": BLACK,
        "xtick.major.width": 1.6,
        "ytick.major.width": 1.6,
        "xtick.major.size": 7.0,
        "ytick.major.size": 8.0,
        "hatch.linewidth": 1.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE_CMAP = LinearSegmentedColormap.from_list(
    "paper_blue", ["#ffffff", "#dae9f7", "#a8c7e9", "#5e8ec8", "#28527a"]
)


def edge(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_color(BLACK)


def load_inputs(kimi_path: Path, policy_path: Path):
    kimi = dict(np.load(kimi_path))
    base = float(kimi["request_received_ts"].min())
    hour = np.floor((kimi["request_received_ts"] - base) / 3600).astype(int)
    mask = hour == 18
    selected = {name: value[mask] for name, value in kimi.items()}
    selected["relative_time"] = (
        selected["request_received_ts"] - selected["request_received_ts"].min()
    )
    policy = json.loads(policy_path.read_text())
    return selected, policy


def workload_metrics(kimi: dict[str, np.ndarray]) -> dict:
    input_len = kimi["input_len"].astype(float)
    cached = kimi["cached_tokens"].astype(float)
    ratio = np.minimum(cached / np.maximum(input_len, 1.0), 1.0)
    minute = np.floor(kimi["relative_time"] / 60).astype(int)
    dp = kimi["dp_rank"].astype(int)
    matrix = np.zeros((8, 60), dtype=float)
    np.add.at(matrix, (dp, minute), cached)
    mean = matrix.mean(axis=0)
    nonzero_mean = np.where(mean > 0, mean, np.nan)
    max_over_mean = matrix.max(axis=0) / nonzero_mean
    max_over_min = matrix.max(axis=0) / np.where(
        matrix.min(axis=0) > 0, matrix.min(axis=0), np.nan
    )
    top2_share = np.sort(matrix, axis=0)[-2:].sum(axis=0) / np.where(
        matrix.sum(axis=0) > 0, matrix.sum(axis=0), np.nan
    )
    return {
        "requests": int(input_len.size),
        "input_token_p50_p90_p99": np.percentile(input_len, [50, 90, 99]).tolist(),
        "cached_token_p50_p90_p99": np.percentile(cached, [50, 90, 99]).tolist(),
        "cache_ratio_p50_p90": np.percentile(ratio, [50, 90]).tolist(),
        "fraction_at_least_half_cached": float(np.mean(ratio >= 0.5)),
        "cached_over_input_token_mass": float(cached.sum() / input_len.sum()),
        "cache_matrix": matrix,
        "max_over_mean": max_over_mean,
        "max_over_min": max_over_min,
        "top2_share": top2_share,
        "max_over_mean_p50_p90_max": np.nanpercentile(
            max_over_mean, [50, 90, 100]
        ).tolist(),
        "top2_share_p50_p90": np.nanpercentile(top2_share, [50, 90]).tolist(),
    }


def figure_workload(kimi: dict, metrics: dict, output: Path, dpi: int):
    fig = plt.figure(figsize=(12.0, 9.2), facecolor=BG)
    ax0 = fig.add_axes([0.09, 0.70, 0.36, 0.20])
    ax1 = fig.add_axes([0.60, 0.68, 0.34, 0.23])
    ax2 = fig.add_axes([0.09, 0.38, 0.84, 0.21])
    ax3 = fig.add_axes([0.09, 0.09, 0.84, 0.20])

    ratio = np.minimum(kimi["cached_tokens"] / np.maximum(kimi["input_len"], 1), 1.0)
    sorted_ratio = np.sort(ratio)
    ecdf = np.arange(1, sorted_ratio.size + 1) / sorted_ratio.size
    ax0.plot(
        sorted_ratio,
        ecdf,
        color="#3e76b7",
        linewidth=3.0,
        solid_capstyle="round",
    )
    p50, p90 = metrics["cache_ratio_p50_p90"]
    ax0.axvline(p50, color=GRAY, linewidth=1.5, linestyle=(0, (5, 3)))
    ax0.axvline(0.5, color="#a35b2f", linewidth=1.8, linestyle=(0, (4, 2, 1, 2)))
    ax0.text(
        0.03,
        0.08,
        f"{metrics['fraction_at_least_half_cached']:.1%} requests\nreuse ≥50% of input",
        transform=ax0.transAxes,
        fontsize=14,
        color=BLACK,
    )
    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax0.set_yticks([0, 0.5, 1.0])
    ax0.set_xlabel("Cached tokens / prompt tokens", fontsize=15)
    ax0.set_ylabel("CDF", fontsize=15)
    ax0.set_title("(a) Prefix-cache weight", fontsize=17, pad=8)
    ax0.tick_params(labelsize=13.5)
    edge(ax0)

    labels = ["0", "1", "2", "4", "8", "16", "32"]
    ax1.hist(
        kimi["input_len"],
        bins=np.logspace(2, np.log10(130000), 34),
        color="#a8c7e9",
        edgecolor=BLACK,
        linewidth=0.75,
    )
    ax1.set_xscale("log")
    ax1.set_xticks([10**2, 10**3, 10**4, 10**5])
    ax1.set_xticklabels(["100", "1K", "10K", "100K"], fontsize=13)
    ax1.set_yticks([0, 1000, 2000])
    ax1.set_xlabel("Prompt tokens", fontsize=15)
    ax1.set_ylabel("Requests", fontsize=15)
    ax1.set_title("(b) Context-length demand", fontsize=17, pad=8)
    ax1.tick_params(labelsize=13.5)
    edge(ax1)

    matrix_mtok = metrics["cache_matrix"] / 1e6
    im = ax2.imshow(
        matrix_mtok,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap=BLUE_CMAP,
    )
    ax2.set_xticks(np.arange(0, 61, 10))
    ax2.set_yticks(np.arange(8))
    ax2.set_xticklabels(np.arange(0, 61, 10), fontsize=13)
    ax2.set_yticklabels([f"E{i}" for i in range(8)], fontsize=13)
    ax2.set_ylabel("Engine", fontsize=15)
    ax2.set_title("(c) Cache-token arrivals per engine", fontsize=17, pad=8)
    cbar = fig.colorbar(im, ax=ax2, fraction=0.026, pad=0.015)
    cbar.set_label("M tokens / min", fontsize=13.5)
    cbar.ax.tick_params(labelsize=12, width=1.5, length=6)
    cbar.outline.set_linewidth(1.5)
    cbar.outline.set_color(BLACK)
    edge(ax2)

    x = np.arange(60)
    ax3.plot(
        x,
        metrics["max_over_mean"],
        color="#3e76b7",
        marker="o",
        markersize=4.8,
        linewidth=2.5,
        label="Hot / mean",
    )
    ax3.plot(
        x,
        metrics["max_over_min"],
        color="#a35b2f",
        marker="s",
        markersize=4.3,
        linewidth=2.1,
        label="Hot / cold",
    )
    p90 = metrics["max_over_mean_p50_p90_max"][1]
    ax3.axhline(
        p90,
        color=GRAY,
        linewidth=1.6,
        linestyle=(0, (5, 3)),
    )
    ax3.text(
        0.5,
        p90 + 0.12,
        f"p90 hot/mean = {p90:.2f}x",
        fontsize=13,
        color=BLACK,
    )
    ax3.set_xlim(0, 59)
    ax3.set_ylim(1, np.nanmax(metrics["max_over_min"]) * 1.08)
    ax3.set_xticks(np.arange(0, 61, 10))
    ax3.set_xlabel("Time in Kimi hour 18 (min)", fontsize=16)
    ax3.set_ylabel("Cache-load imbalance", fontsize=15)
    ax3.set_title("(d) Per-minute global hot/cold severity", fontsize=17, pad=8)
    ax3.tick_params(labelsize=13.5)
    ax3.grid(axis="y", color="#efefef", linewidth=0.8)
    ax3.legend(frameon=False, fontsize=13, loc="upper right", ncol=2)
    edge(ax3)

    for ext in ("png", "pdf"):
        fig.savefig(output.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)


def policy_metrics(policy: dict) -> tuple[dict, dict[str, dict]]:
    result = {}
    raw = {item["policy"]: item for item in policy["policies"]}
    for name, item in raw.items():
        after = [(t, loads) for t, loads in item["load_series"] if t >= 1800]
        imbalance = []
        for _, loads in after:
            loads = np.asarray(loads)
            active = loads[:4]
            imbalance.append(active.max() / max(active.mean(), 1e-9))
        result[name] = {
            "post_scale_imbalance_p50_p90": np.percentile(imbalance, [50, 90]).tolist(),
            "retiring_states_at_deadline": item["retirement_at_deadline"][
                "retiring_states"
            ],
            "drain_migrations": item["drain_migrations"],
            "migrations": item["migrations"],
            "cache_hit_tokens": item["cache_hit_tokens"],
            "recompute_tokens": item["recompute_tokens"],
            "transfer_gib": item["transfer_bytes"] / 1024**3,
        }
    return result, raw


def figure_policy(metrics: dict, output: Path, dpi: int):
    names = list(POLICY_STYLE)
    labels = [POLICY_STYLE[name]["label"] for name in names]
    colors = [POLICY_STYLE[name]["color"] for name in names]
    x = np.arange(len(names), dtype=float)

    fig = plt.figure(figsize=(12.0, 8.2), facecolor=BG)
    panels = [
        (
            [metrics[name]["post_scale_imbalance_p50_p90"][0] for name in names],
            "(a) Post-scale-in hot/mean",
            "x",
        ),
        (
            [metrics[name]["retiring_states_at_deadline"] for name in names],
            "(b) States left on retiring GPUs",
            "count",
        ),
        (
            [metrics[name]["cache_hit_tokens"] / 1e6 for name in names],
            "(c) Preserved prefix hits",
            "M tokens",
        ),
        (
            [metrics[name]["transfer_gib"] for name in names],
            "(d) Transferred hybrid state",
            "GiB",
        ),
    ]
    axes = []
    positions = [
        [0.08, 0.70, 0.36, 0.19],
        [0.59, 0.70, 0.36, 0.19],
        [0.08, 0.36, 0.36, 0.19],
        [0.59, 0.36, 0.36, 0.19],
    ]
    for (values, title, ylabel), pos in zip(panels, positions):
        ax = fig.add_axes(pos)
        ax.bar(
            x,
            values,
            width=0.60,
            color=colors,
            edgecolor=BLACK,
            linewidth=1.35,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=11.5)
        ax.set_title(title, fontsize=16.5, pad=7)
        ax.set_ylabel(ylabel, fontsize=14.5)
        ax.grid(axis="y", color="#efefef", linewidth=0.8)
        ax.tick_params(labelsize=12.5)
        max_value = max(values)
        ax.set_ylim(0, max_value * 1.22 if max_value else 1.0)
        for xi, value in zip(x, values):
            ax.text(
                xi,
                value + max_value * 0.035,
                (
                    f"{value:.2f}".rstrip("0").rstrip(".")
                    if value < 10
                    else f"{value:.0f}"
                ),
                ha="center",
                va="bottom",
                fontsize=12,
            )
        edge(ax)
        axes.append(ax)

    ax_note = fig.add_axes([0.08, 0.07, 0.87, 0.17])
    ax_note.axis("off")
    ax_note.text(
        0.01,
        0.82,
        "Existing policies:",
        fontsize=14.5,
        fontweight="bold",
        va="top",
    )
    ax_note.text(
        0.01,
        0.42,
        "Sticky preserves 6.86M cached tokens but leaves 11 warm states on retiring GPUs; "
        "stateless reroute drains them but reconstructs 6.25M tokens; token-only migration "
        "balances load but models state as a linear function of tokens.",
        fontsize=13.2,
        va="top",
        wrap=True,
    )
    ax_note.text(
        0.01,
        0.05,
        "AgentShift preserves all prefix hits, drains the retiring GPUs by the deadline, "
        "and accounts for full-KV plus recurrent-snapshot capacity.",
        fontsize=13.2,
        va="top",
        color="#28527a",
    )

    for ext in ("png", "pdf"):
        output.with_suffix(f".{ext}").unlink(missing_ok=True)
        fig.savefig(output.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)


def rolling_imbalance(series: list, window=240.0, step=15.0):
    times = np.asarray([item[0] for item in series], dtype=float)
    values = np.asarray([item[1] for item in series], dtype=float)
    out_t, out_v = [], []
    for now in np.arange(0, times[-1] + step, step):
        mask = (times >= now - window) & (times <= now)
        if not np.any(mask):
            continue
        maxima = values[mask].max(axis=1)
        means = values[mask].mean(axis=1)
        out_t.append(now)
        out_v.append(np.median(maxima / np.maximum(means, 1e-9)))
    return np.asarray(out_t), np.asarray(out_v)


def figure_load_timeseries(raw: dict, output: Path, dpi: int):
    fig = plt.figure(figsize=(11.8, 4.8), facecolor=BG)
    ax = fig.add_axes([0.09, 0.17, 0.88, 0.69])
    for name, style in POLICY_STYLE.items():
        x, y = rolling_imbalance(raw[name]["load_series"])
        ax.plot(
            x,
            y,
            color=style["color"],
            linewidth=3.0 if name == "heterogeneous" else 2.2,
            label=style["label"],
        )
    ax.axvline(
        1800,
        color=GRAY,
        linewidth=1.8,
        linestyle=(0, (5, 3)),
    )
    ax.text(1815, 1.62, "8 → 4 GPUs", fontsize=14, color=BLACK)
    ax.set_xlim(0, 3650)
    ax.set_ylim(1.0, 1.78)
    ax.set_xticks(np.arange(0, 3601, 600))
    ax.set_xlabel("One-hour replay time (s)", fontsize=16)
    ax.set_ylabel("Hot / mean load", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(color="#f0f0f0", linewidth=0.8)
    ax.legend(frameon=False, fontsize=12.5, ncol=2, loc="upper right")
    edge(ax)
    for ext in ("png", "pdf"):
        output.with_suffix(f".{ext}").unlink(missing_ok=True)
        fig.savefig(output.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)


def figure_migration_map(raw: dict, output: Path, dpi: int):
    hetero = raw["heterogeneous"]
    series = hetero["load_series"]
    bin_edges = np.arange(0, 3661, 30)
    load_matrix = np.zeros((8, len(bin_edges) - 1), dtype=float)
    for bi, right in enumerate(bin_edges[1:]):
        candidates = [(t, loads) for t, loads in series if t <= right]
        if candidates:
            load_matrix[:, bi] = np.asarray(candidates[-1][1])

    fig = plt.figure(figsize=(12.0, 11.0), facecolor=BG)
    ax0 = fig.add_axes([0.09, 0.70, 0.83, 0.19])
    im = ax0.imshow(
        load_matrix,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=BLUE_CMAP,
    )
    ax0.set_xticks(np.arange(0, 121, 20))
    ax0.set_xticklabels(np.arange(0, 3601, 600), fontsize=13)
    ax0.set_yticks(np.arange(8))
    ax0.set_yticklabels([f"E{i}" for i in range(8)], fontsize=13)
    ax0.axvline(60, color="#a35b2f", linewidth=2.4, linestyle=(0, (5, 3)))
    ax0.axhline(3.5, color=BLACK, linewidth=1.5, linestyle=":")
    ax0.text(61, 7.15, "8 → 4 GPUs", fontsize=13.5, color="#a35b2f")
    ax0.text(1, -0.95, "active", fontsize=12.5, color=GRAY)
    ax0.text(1, 3.72, "retiring after 1800s", fontsize=12.5, color=GRAY)
    ax0.set_title("(a) AgentShift engine load", fontsize=17, pad=8)
    ax0.set_ylabel("Engine", fontsize=15)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.014)
    cbar.set_label("Active + warm agents", fontsize=13)
    cbar.ax.tick_params(labelsize=11.5, width=1.4, length=5)
    cbar.outline.set_linewidth(1.5)
    edge(ax0)

    ax1 = fig.add_axes([0.09, 0.38, 0.83, 0.22], sharex=ax0)
    for event in hetero["migration_events"]:
        source = int(event["source_engine"].split("-")[1])
        destination = int(event["destination_engine"].split("-")[1])
        color = "#a35b2f" if event["drain"] else "#3e76b7"
        ax1.annotate(
            "",
            xy=(event["time"] + 18, destination),
            xytext=(event["time"], source),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=0.45,
                alpha=0.19,
                shrinkA=0,
                shrinkB=0,
            ),
        )
    ax1.axvline(60, color="#a35b2f", linewidth=2.2, linestyle=(0, (5, 3)))
    ax1.axhline(3.5, color=BLACK, linewidth=1.4, linestyle=":")
    ax1.set_ylim(-0.7, 7.7)
    ax1.set_yticks(np.arange(8))
    ax1.set_yticklabels([f"E{i}" for i in range(8)], fontsize=13)
    ax1.set_ylabel("Source → destination", fontsize=15)
    ax1.set_title("(b) Per-request mobility decisions", fontsize=17, pad=8)
    ax1.tick_params(labelsize=13)
    ax1.grid(axis="y", color="#f3f3f3", linewidth=0.7)
    handles = [
        Line2D([0], [0], color="#3e76b7", lw=3, label="Load balancing"),
        Line2D([0], [0], color="#a35b2f", lw=3, label="Drain urgent"),
    ]
    ax1.legend(handles=handles, frameon=False, fontsize=12.5, loc="upper right", ncol=2)
    edge(ax1)

    ax2 = fig.add_axes([0.11, 0.07, 0.72, 0.17])
    counts = np.zeros((8, 8), dtype=int)
    for event in hetero["migration_events"]:
        source = int(event["source_engine"].split("-")[1])
        destination = int(event["destination_engine"].split("-")[1])
        counts[source, destination] += 1
    im2 = ax2.imshow(counts, cmap=BLUE_CMAP, interpolation="nearest", origin="lower")
    ax2.set_xticks(np.arange(8))
    ax2.set_yticks(np.arange(8))
    ax2.set_xticklabels([f"E{i}" for i in range(8)], fontsize=12)
    ax2.set_yticklabels([f"E{i}" for i in range(8)], fontsize=12)
    ax2.set_xlabel("Destination engine", fontsize=14.5)
    ax2.set_ylabel("Source", fontsize=14.5)
    ax2.set_title("(c) Aggregated request directions", fontsize=16.5, pad=7)
    for i in range(8):
        for j in range(8):
            if counts[i, j]:
                ax2.text(
                    j,
                    i,
                    str(counts[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if counts[i, j] > counts.max() * 0.55 else BLACK,
                )
    edge(ax2)
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.035, pad=0.015)
    cbar2.set_label("Requests", fontsize=12)
    cbar2.ax.tick_params(labelsize=11, width=1.3, length=5)
    cbar2.outline.set_linewidth(1.5)

    for ext in ("png", "pdf"):
        output.with_suffix(f".{ext}").unlink(missing_ok=True)
        fig.savefig(output.with_suffix(f".{ext}"), dpi=dpi)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kimi-npz", type=Path, required=True)
    parser.add_argument("--policy-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    kimi, policy = load_inputs(args.kimi_npz, args.policy_json)
    work = workload_metrics(kimi)
    metrics, raw = policy_metrics(policy)
    summary = {
        "workload": {
            key: value for key, value in work.items() if key != "cache_matrix"
        },
        "policies": metrics,
        "methodology": (
            "Kimi hour 18 supplies timing, length, cached-token, and DP-placement shape. "
            "LooGLE and ToolBench supply replay content; policy plots are trace-shaped simulation."
        ),
    }
    (args.output_dir / "paper_figure_metrics.json").write_text(
        json.dumps(summary, indent=2, default=lambda value: value.tolist()) + "\n"
    )

    figure_workload(
        kimi,
        work,
        args.output_dir / "fig1_workload_state_gravity",
        args.dpi,
    )
    figure_policy(
        metrics,
        args.output_dir / "fig2_policy_comparison",
        args.dpi,
    )
    figure_load_timeseries(
        raw,
        args.output_dir / "fig3_one_hour_load_balance",
        args.dpi,
    )
    figure_migration_map(
        raw,
        args.output_dir / "fig4_migration_directions",
        args.dpi,
    )
    print(json.dumps(summary, default=lambda value: value.tolist(), indent=2))


if __name__ == "__main__":
    main()
