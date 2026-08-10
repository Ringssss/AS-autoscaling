#!/usr/bin/env python3
"""Plot the controlled AgentShift group-size sweep."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt


GROUP_SIZES = (1, 4, 8, 12, 18, 36)
REGIMES = (
    ("native", "Native NVLink", "#2F5597", "o"),
    ("rdma200", "Emulated 200 Gb/s", "#008B8B", "s"),
    ("rdma100", "Emulated 100 Gb/s", "#D55E00", "^"),
)


def configure_style() -> str:
    try:
        fm.findfont("Times New Roman", fallback_to_default=False)
        serif_font = "Times New Roman"
    except ValueError:
        # Liberation Serif is metrically compatible with Times New Roman.
        serif_font = "Liberation Serif"
        print("Times New Roman is unavailable; using Liberation Serif.")

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [serif_font],
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 15,
            "axes.linewidth": 1.1,
            "lines.linewidth": 2.3,
            "lines.markersize": 8,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    return serif_font


def latest_result(root: Path, regime: str, group_size: int) -> Path:
    directory = root / f"group-adaptive-{regime}-clean-g{group_size}"
    matches = sorted(directory.glob("real-tools-all-*.json"))
    if not matches:
        raise FileNotFoundError(f"no result JSON under {directory}")
    return matches[-1]


def load_medians(root: Path) -> dict[str, list[float]]:
    medians: dict[str, list[float]] = {}
    for regime, _, _, _ in REGIMES:
        values = []
        for group_size in GROUP_SIZES:
            result_path = latest_result(root, regime, group_size)
            result = json.loads(result_path.read_text())
            records = [
                record
                for record in result["records"]
                if record["scenario"] == "progressive"
            ]
            if len(records) != 3:
                raise ValueError(f"expected 3 repeats in {result_path}")
            if not all(record["full_prefix_hit"] for record in records):
                raise ValueError(f"incomplete prefix hit in {result_path}")
            if {record["tool_result_tokens"] for record in records} != {16}:
                raise ValueError(f"uncontrolled tool suffix in {result_path}")
            values.append(
                1000.0
                * statistics.median(
                    record["post_tool_seconds"] for record in records
                )
            )
        medians[regime] = values
    return medians


def plot(root: Path, output_dir: Path) -> None:
    font = configure_style()
    medians = load_medians(root)
    x = list(range(len(GROUP_SIZES)))

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.55))
    for regime, label, color, marker in REGIMES:
        axes[0].plot(
            x,
            medians[regime],
            color=color,
            marker=marker,
            label=label,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )
        best = min(medians[regime])
        regret = [(value / best - 1.0) * 100.0 for value in medians[regime]]
        axes[1].plot(
            x,
            regret,
            color=color,
            marker=marker,
            label=label,
            markeredgecolor="white",
            markeredgewidth=0.8,
        )

    for axis in axes:
        axis.set_xticks(x, GROUP_SIZES)
        axis.set_xlabel("Layer group size")
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    axes[0].set_title("(a) End-to-end latency")
    axes[0].set_ylabel("Median post-tool latency (ms)")
    axes[0].set_ylim(105, 315)
    axes[0].set_yticks((120, 160, 200, 240, 280))

    axes[1].set_title("(b) Cost of a fixed group size")
    axes[1].set_ylabel("Regret vs. per-link best (%)")
    axes[1].set_ylim(-0.8, 16.5)
    axes[1].set_yticks((0, 4, 8, 12, 16))
    axes[1].axhline(0, color="#666666", linewidth=1.0, zorder=0)
    axes[1].annotate(
        "slow links: g=4",
        xy=(1, 0),
        xytext=(1.15, 3.8),
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
        fontsize=13,
        color="#333333",
    )
    axes[1].annotate(
        "NVLink: g=36",
        xy=(5, 0),
        xytext=(3.55, 3.8),
        arrowprops={"arrowstyle": "->", "color": "#444444", "lw": 1.0},
        fontsize=13,
        color="#333333",
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.035),
        ncol=3,
        frameon=False,
        columnspacing=2.4,
        handletextpad=0.6,
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.18, top=0.78, wspace=0.27)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "adaptive_group_size_1x2"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"font={font}")
    print(f"wrote {stem}.pdf/.svg/.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results" / "figures",
    )
    args = parser.parse_args()
    plot(args.results_root, args.output_dir)


if __name__ == "__main__":
    main()
