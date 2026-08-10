#!/usr/bin/env python3
"""Generate measured and trace-projected AgentShift autoscaling figures.

The measured figure reads only completed Qwen3-8B telemetry.  The 30-minute
figure is deliberately labeled as a policy projection: it derives arrival
rates from existing manifests and does not claim to be a GPU measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


ENGINE_COLORS = ("#0072B2", "#D55E00", "#009E73", "#7A5195")
MODEL_COLORS = {"8B": "#0072B2", "32B": "#D55E00"}
PHASE_COLORS = ("#F2F4F5", "#FAFAFA", "#F2F4F5")


def configure_matplotlib() -> None:
    """Configure a compact systems-paper style with an Arial fallback."""

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "font.size": 8.5,
            "axes.labelsize": 9.0,
            "axes.titlesize": 9.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "legend.fontsize": 7.7,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.2,
            "grid.linewidth": 0.5,
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
            "figure.dpi": 150,
        }
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_axis(axis: plt.Axes, *, grid_axis: str = "y") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(True, which="major", axis=grid_axis)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", which="major", length=2.8, pad=2.2)


def add_phase_bands(
    axis: plt.Axes,
    boundaries: list[tuple[float, float, str]],
    *,
    label_y: float = 0.965,
) -> None:
    for index, (left, right, label) in enumerate(boundaries):
        axis.axvspan(left, right, color=PHASE_COLORS[index % 3], zorder=0)
        axis.text(
            (left + right) / 2.0,
            label_y,
            label,
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#606060",
            fontsize=7.4,
        )
        if index:
            axis.axvline(left, color="#A5A5A5", linewidth=0.65, linestyle=(0, (3, 3)))


def save_figure(fig: plt.Figure, stem: Path) -> tuple[Path, Path]:
    pdf = stem.with_suffix(".pdf")
    png = stem.parent / f"{stem.name}_600dpi.png"
    fig.savefig(pdf, format="pdf", dpi=600, facecolor="white")
    fig.savefig(png, format="png", dpi=600, facecolor="white")
    plt.close(fig)
    return pdf, png


def phase_boundaries_from_samples(
    samples: list[dict[str, Any]], duration_seconds: float
) -> list[tuple[float, float, str]]:
    phase_starts: dict[int, float] = {}
    for row in samples:
        phase = int(row["phase"])
        phase_starts.setdefault(phase, float(row["elapsed_seconds"]))
    starts = sorted(phase_starts.items())
    names = {0: "High", 1: "Idle", 2: "Medium"}
    boundaries = []
    for index, (phase, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else duration_seconds
        boundaries.append((start / 60.0, end / 60.0, names.get(phase, f"Phase {phase + 1}")))
    return boundaries


def make_measured_figure(
    summary_path: Path,
    telemetry_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    summary = read_json(summary_path)
    telemetry = read_jsonl(telemetry_path)
    samples = [row for row in telemetry if row.get("kind") == "sample"]
    migrations = [row for row in telemetry if row.get("kind") == "migration"]
    if not samples or not migrations:
        raise ValueError("measured telemetry lacks samples or migration records")

    duration = float(summary["duration_seconds"])
    sample_minutes = np.asarray([float(row["elapsed_seconds"]) / 60.0 for row in samples])
    active_engines = np.asarray(
        [sum(state == "ACTIVE" for state in row["states"].values()) for row in samples]
    )
    phase_bands = phase_boundaries_from_samples(samples, duration)

    bin_seconds = 2.0
    edges = np.arange(0.0, math.ceil(duration / bin_seconds) * bin_seconds + bin_seconds, bin_seconds)
    handoff_counts, _ = np.histogram(
        [float(row["elapsed_seconds"]) for row in migrations], bins=edges
    )
    handoff_minutes = (edges[:-1] + bin_seconds / 2.0) / 60.0

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=0.075, right=0.945, bottom=0.245, top=0.965, wspace=0.33)

    left = axes[0]
    add_phase_bands(left, phase_bands)
    left.step(
        sample_minutes,
        active_engines,
        where="post",
        color="#333333",
        linewidth=1.65,
        label="Serving engines",
        zorder=4,
    )
    left.set_xlim(0.0, duration / 60.0)
    left.set_ylim(0.8, 4.25)
    left.set_yticks([1, 2, 3, 4])
    left.set_xlabel("Elapsed time (min)")
    left.set_ylabel("Active TP=2 engines")
    clean_axis(left, grid_axis="y")

    handoff_axis = left.twinx()
    handoff_axis.bar(
        handoff_minutes,
        handoff_counts,
        width=bin_seconds / 60.0 * 0.78,
        color="#8FB9D7",
        edgecolor="#4D89B3",
        linewidth=0.45,
        alpha=0.72,
        label="KV handoffs",
        zorder=2,
    )
    handoff_axis.set_ylim(0, max(8, int(max(handoff_counts)) + 2))
    handoff_axis.set_ylabel("KV handoffs / 2 s", color="#4D7898")
    handoff_axis.tick_params(axis="y", colors="#4D7898", length=2.8, pad=2.2)
    handoff_axis.spines["top"].set_visible(False)
    handoff_axis.spines["right"].set_color("#4D7898")
    left.legend(
        handles=[
            Line2D([0], [0], color="#333333", linewidth=1.65, label="Serving engines"),
            Patch(facecolor="#8FB9D7", edgecolor="#4D89B3", label="KV handoffs"),
        ],
        loc="upper right",
        bbox_to_anchor=(0.99, 0.84),
        handlelength=1.6,
    )

    right = axes[1]
    add_phase_bands(right, phase_bands)
    engine_ids = sorted(samples[0]["owners"])
    for index, engine_id in enumerate(engine_ids):
        owner_counts = [int(row["owners"].get(engine_id, 0)) for row in samples]
        right.step(
            sample_minutes,
            owner_counts,
            where="post",
            color=ENGINE_COLORS[index],
            linewidth=1.35,
            label=engine_id.replace("engine-", "Engine "),
            zorder=3,
        )
    right.set_xlim(0.0, duration / 60.0)
    right.set_ylim(bottom=0)
    right.set_xlabel("Elapsed time (min)")
    right.set_ylabel("Live agents owned")
    right.yaxis.set_major_locator(MultipleLocator(5))
    clean_axis(right, grid_axis="y")
    right.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.84),
        ncol=2,
        columnspacing=0.8,
        handlelength=1.3,
    )

    axes[0].text(
        0.5,
        -0.25,
        "(a) Measured capacity and KV handoffs",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
    )
    axes[1].text(
        0.5,
        -0.25,
        "(b) Measured ownership distribution",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
    )

    paths = save_figure(fig, output_dir / "fig_autoscaling_measured")
    provenance = {
        "evidence_type": "measured",
        "model": "Qwen3-8B",
        "engine_layout": "4 TP=2 engines on 8 GPUs",
        "duration_seconds": duration,
        "successful_agents": int(summary["successful_agents"]),
        "failed_agents": int(summary["failed_agents"]),
        "migrations": int(summary["migrations"]),
        "migrated_full_prefix_hit_fraction": float(
            summary["migrated_full_prefix_hit_fraction"]
        ),
        "historical_reprefilled_tokens": int(summary["historical_reprefilled_tokens"]),
        "migration_hidden_fraction": float(summary["migration_hidden_fraction"]),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
        "telemetry_path": str(telemetry_path.resolve()),
        "telemetry_sha256": sha256(telemetry_path),
        "handoff_histogram_seconds": bin_seconds,
    }
    return paths[0], paths[1], provenance


def binned_arrival_rates(
    manifest: dict[str, Any], bin_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    duration = float(manifest["config"]["duration_seconds"])
    edges = np.arange(0.0, duration + bin_seconds, bin_seconds)
    counts, _ = np.histogram(
        [float(event["arrival_seconds"]) for event in manifest["events"]], bins=edges
    )
    centers = (edges[:-1] + bin_seconds / 2.0) / 60.0
    return centers, counts.astype(float) / bin_seconds


def replay_warm_pool(
    arrival_rates: np.ndarray,
    *,
    max_engines: int,
    target_arrivals_per_engine: float,
    scale_in_windows: int = 2,
) -> np.ndarray:
    """Replay a transparent rate policy, changing at most one engine per bin."""

    desired = np.clip(
        np.ceil(arrival_rates / target_arrivals_per_engine).astype(int),
        1,
        max_engines,
    )
    active = 1
    below_windows = 0
    replay: list[int] = []
    for target in desired:
        if target > active:
            active += 1
            below_windows = 0
        elif target < active:
            below_windows += 1
            if below_windows >= scale_in_windows:
                active -= 1
                below_windows = 0
        else:
            below_windows = 0
        replay.append(active)
    return np.asarray(replay)


def make_replay_figure(
    manifest_8b_path: Path,
    manifest_32b_path: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    manifests = {
        "8B": read_json(manifest_8b_path),
        "32B": read_json(manifest_32b_path),
    }
    bin_seconds = 30.0
    targets = {"8B": 2.5, "32B": 0.75}
    max_engines = {"8B": 4, "32B": 2}
    rates: dict[str, np.ndarray] = {}
    replays: dict[str, np.ndarray] = {}
    centers: np.ndarray | None = None
    for model, manifest in manifests.items():
        model_centers, rates[model] = binned_arrival_rates(manifest, bin_seconds)
        if centers is None:
            centers = model_centers
        elif not np.array_equal(centers, model_centers):
            raise ValueError("manifest durations do not align")
        replays[model] = replay_warm_pool(
            rates[model],
            max_engines=max_engines[model],
            target_arrivals_per_engine=targets[model],
        )
    assert centers is not None

    phase_bands = [
        (0.0, 10.0, "Azure"),
        (10.0, 20.0, "Kimi"),
        (20.0, 30.0, "FlowPrefill"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.245, top=0.965, wspace=0.28)

    left = axes[0]
    add_phase_bands(left, phase_bands)
    for model, marker in (("8B", "o"), ("32B", "s")):
        left.plot(
            centers,
            rates[model],
            color=MODEL_COLORS[model],
            marker=marker,
            markevery=4,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.45,
            label=f"Qwen3-{model} manifest",
            zorder=3,
        )
    left.set_xlim(0, 30)
    left.set_ylim(0, max(12.0, max(float(values.max()) for values in rates.values()) * 1.11))
    left.set_xlabel("Trace time (min)")
    left.set_ylabel("Arrival rate (requests/s)")
    left.xaxis.set_major_locator(MultipleLocator(5))
    clean_axis(left, grid_axis="y")
    left.legend(loc="upper right", bbox_to_anchor=(0.99, 0.84), handlelength=1.5)

    right = axes[1]
    add_phase_bands(right, phase_bands)
    for model, marker in (("8B", "o"), ("32B", "s")):
        label = f"Qwen3-{model}, TP={2 if model == '8B' else 4}"
        right.step(
            centers,
            replays[model],
            where="mid",
            color=MODEL_COLORS[model],
            linewidth=1.55,
            label=label,
            zorder=3,
        )
        right.plot(
            centers[::4],
            replays[model][::4],
            linestyle="none",
            marker=marker,
            markerfacecolor="white",
            markeredgecolor=MODEL_COLORS[model],
            markeredgewidth=0.8,
            zorder=4,
        )
    right.set_xlim(0, 30)
    right.set_ylim(0.8, 4.25)
    right.set_yticks([1, 2, 3, 4])
    right.set_xlabel("Trace time (min)")
    right.set_ylabel("Projected warm engines")
    right.xaxis.set_major_locator(MultipleLocator(5))
    clean_axis(right, grid_axis="y")
    right.legend(loc="upper right", bbox_to_anchor=(0.99, 0.84), handlelength=1.5)
    axes[0].text(
        0.5,
        -0.25,
        "(a) Thirty-minute trace input",
        transform=axes[0].transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
    )
    axes[1].text(
        0.5,
        -0.25,
        "(b) Policy projection (not a GPU run)",
        transform=axes[1].transAxes,
        ha="center",
        va="top",
        fontsize=8.6,
    )

    paths = save_figure(fig, output_dir / "fig_autoscaling_trace_replay")
    provenance = {
        "evidence_type": "trace-driven policy projection; not a GPU measurement",
        "bin_seconds": bin_seconds,
        "policy": (
            "desired=ceil(arrival_rate/target); one engine change per 30-s bin; "
            "scale out immediately; scale in after two consecutive lower-demand bins"
        ),
        "targets_arrivals_per_second_per_engine": targets,
        "max_warm_engines": max_engines,
        "manifests": {
            "8B": {
                "path": str(manifest_8b_path.resolve()),
                "sha256": sha256(manifest_8b_path),
                "events": len(manifests["8B"]["events"]),
                "phase_event_counts": manifests["8B"]["phase_event_counts"],
            },
            "32B": {
                "path": str(manifest_32b_path.resolve()),
                "sha256": sha256(manifest_32b_path),
                "events": len(manifests["32B"]["events"]),
                "phase_event_counts": manifests["32B"]["phase_event_counts"],
            },
        },
        "projected_engine_counts": {
            model: Counter(replay.tolist()) for model, replay in replays.items()
        },
    }
    return paths[0], paths[1], provenance


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    measured = (
        root
        / "results/autoscaling/runs/"
        "qwen8b-tp2-smoke-immediate-pin-1786185554439251023"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measured-run", type=Path, default=measured)
    parser.add_argument(
        "--manifest-8b",
        type=Path,
        default=root
        / "results/autoscaling/manifests/autoscaling-manifest-qwen8b-30m-v1.json",
    )
    parser.add_argument(
        "--manifest-32b",
        type=Path,
        default=root
        / "results/autoscaling/manifests/autoscaling-manifest-qwen32b-30m-v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "paper_rewriting_output/figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    measured_pdf, measured_png, measured_provenance = make_measured_figure(
        args.measured_run / "summary.json",
        args.measured_run / "telemetry.jsonl",
        args.output_dir,
    )
    replay_pdf, replay_png, replay_provenance = make_replay_figure(
        args.manifest_8b,
        args.manifest_32b,
        args.output_dir,
    )

    provenance_path = args.output_dir / "fig_autoscaling_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "measured_figure": measured_provenance,
                "trace_replay_figure": replay_provenance,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    for path in (
        measured_pdf,
        measured_png,
        replay_pdf,
        replay_png,
        provenance_path,
    ):
        print(path)


if __name__ == "__main__":
    main()
