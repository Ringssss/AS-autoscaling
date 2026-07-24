from __future__ import annotations

import argparse
import json
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Rectangle


COLORS = {
    "sticky": "#4D4D4D",
    "reroute": "#D55E00",
    "semantic-reroute": "#D55E00",
    "on-return": "#CC79A7",
    "agentshift": "#0072B2",
    "oracle": "#009E73",
    "sync": "#E69F00",
    "async": "#0072B2",
    "baseline": "#4D4D4D",
}
LABELS = {
    "sticky": "Sticky",
    "reroute": "Reroute",
    "semantic-reroute": "Semantic reroute",
    "on-return": "On-return",
    "agentshift": "AgentShift",
    "oracle": "Oracle",
}


def configure(font_dir: Path) -> None:
    for filename in ("Arial.TTF", "Arialbd.TTF", "Ariali.TTF", "Arialbi.TTF"):
        font_manager.fontManager.addfont(font_dir / filename)
    family = font_manager.FontProperties(fname=font_dir / "Arial.TTF").get_name()
    if family != "Arial":
        raise RuntimeError(f"expected Arial, found {family}")
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def clean_axis(axis, *, grid: bool = True) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid:
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.5, zorder=0)
    axis.set_axisbelow(True)


def save(fig, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(
        output_dir / f"{name}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def grouped_means(records, filters, value):
    groups = defaultdict(list)
    for row in records:
        if all(row.get(key) == expected for key, expected in filters.items()):
            groups[(row["prefix_length"], row["scenario"])].append(row[value])
    return {key: statistics.fmean(values) for key, values in groups.items()}


def context_latency(blocked: dict, output_dir: Path) -> None:
    records = blocked["records"]
    values = grouped_means(records, {"gap_ms": 500.0}, "post_tool_seconds")
    prefixes = sorted({prefix for prefix, _ in values})
    fig, axis = plt.subplots(figsize=(3.35, 2.25))
    for scenario, marker in (
        ("sticky", "o"),
        ("reroute", "s"),
        ("on-return", "D"),
        ("agentshift", "^"),
    ):
        axis.plot(
            [prefix / 1024 for prefix in prefixes],
            [values[(prefix, scenario)] * 1000 for prefix in prefixes],
            marker=marker,
            color=COLORS[scenario],
            label=LABELS[scenario],
        )
    axis.set_yscale("log")
    axis.set_xticks([prefix / 1024 for prefix in prefixes])
    axis.set_xlabel("Completed prefix (K tokens)")
    axis.set_ylabel("Post-tool latency (ms, log)")
    clean_axis(axis)
    axis.legend(frameon=False, ncol=2, loc="upper left")
    save(fig, output_dir, "fig_context_latency")


def gap_overlap(blocked: dict, output_dir: Path) -> None:
    groups = defaultdict(list)
    for row in blocked["records"]:
        if row["prefix_length"] != 32768:
            continue
        if row["scenario"] not in ("sticky", "reroute", "on-return", "agentshift"):
            continue
        groups[(row["gap_ms"], row["scenario"])].append(row["post_tool_seconds"])
    gaps = sorted({gap for gap, _ in groups})
    fig, axis = plt.subplots(figsize=(3.35, 2.25))
    for scenario, marker in (
        ("sticky", "o"),
        ("reroute", "s"),
        ("on-return", "D"),
        ("agentshift", "^"),
    ):
        axis.plot(
            gaps,
            [statistics.fmean(groups[(gap, scenario)]) * 1000 for gap in gaps],
            marker=marker,
            color=COLORS[scenario],
            label=LABELS[scenario],
        )
    axis.set_yscale("log")
    axis.set_xlabel("Blocked interval (ms)")
    axis.set_ylabel("Post-tool latency (ms, log)")
    clean_axis(axis)
    axis.legend(frameon=False, ncol=2, loc="upper right")
    save(fig, output_dir, "fig_gap_overlap")


def hotspot(hotspot_data: dict, output_dir: Path) -> None:
    groups = defaultdict(list)
    for row in hotspot_data["records"]:
        groups[row["scenario"]].append(row["makespan_seconds"])
    order = ["sticky", "tokencake-source", "reroute", "on-return", "agentshift"]
    labels = ["Sticky", "TokenCake-\nSource", "Reroute", "On-return", "AgentShift"]
    colors = [
        COLORS["sticky"],
        "#E69F00",
        COLORS["reroute"],
        COLORS["on-return"],
        COLORS["agentshift"],
    ]
    values = [statistics.fmean(groups[item]) for item in order]
    fig, axis = plt.subplots(figsize=(3.35, 2.25))
    bars = axis.bar(range(len(order)), values, color=colors, width=0.7, zorder=2)
    axis.set_xticks(range(len(order)), labels)
    axis.set_ylabel("Burst makespan (s)")
    clean_axis(axis)
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axis.set_ylim(0, max(values) * 1.16)
    save(fig, output_dir, "fig_hotspot")


def workloads(data: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
    kimi = data["length_traces"]["kimi_k25"]
    cdf = kimi["context_cdf_samples"]
    axes[0].plot(
        [row["tokens"] / 1024 for row in cdf],
        [row["quantile"] * 100 for row in cdf],
        marker="o",
        color=COLORS["agentshift"],
    )
    axes[0].axvline(32, color="#999999", linestyle="--", linewidth=0.8)
    axes[0].set_xlabel("Kimi context (K tokens)")
    axes[0].set_ylabel("Requests (percentile)")
    clean_axis(axes[0])

    turns = data["flowprefill"]["prefix_by_turn"][:10]
    axes[1].plot(
        [row["turn"] for row in turns],
        [row["p50"] / 1024 for row in turns],
        marker="o",
        color=COLORS["agentshift"],
        label="p50",
    )
    axes[1].plot(
        [row["turn"] for row in turns],
        [row["p90"] / 1024 for row in turns],
        marker="s",
        color=COLORS["on-return"],
        label="p90",
    )
    axes[1].set_xlabel("Agent turn")
    axes[1].set_ylabel("Cumulative prefix (K tokens)")
    axes[1].legend(frameon=False)
    clean_axis(axes[1])

    bursts = data["flowprefill"]["return_bursts"]
    windows = [10, 50, 100, 500]
    x = range(len(windows))
    axes[2].plot(
        x,
        [bursts[str(window)]["cluster"]["p99"] for window in windows],
        marker="o",
        color=COLORS["agentshift"],
        label="Cluster p99",
    )
    axes[2].plot(
        x,
        [bursts[str(window)]["cluster"]["max"] for window in windows],
        marker="s",
        color=COLORS["reroute"],
        label="Cluster max",
    )
    axes[2].set_xticks(x, [str(window) for window in windows])
    axes[2].set_xlabel("Return-proxy window (ms)")
    axes[2].set_ylabel("Arrivals per window")
    axes[2].legend(frameon=False)
    clean_axis(axes[2])
    fig.subplots_adjust(wspace=0.42)
    save(fig, output_dir, "fig_workload_characterization")


def elasticity(data: dict, output_dir: Path) -> None:
    summary = data["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    order = ["sticky", "semantic-reroute", "on-return", "agentshift"]
    colors = [COLORS[item] for item in order]
    labels = ["Sticky", "Semantic\nreroute", "On-return", "AgentShift"]
    for axis, mode in zip(axes, ("scale-out", "scale-in")):
        values = [summary[f"{mode}:{item}"]["post_tool_makespan_mean_seconds"] for item in order]
        bars = axis.bar(range(len(order)), values, color=colors, width=0.7, zorder=2)
        axis.set_xticks(range(len(order)), labels)
        axis.set_ylabel("Post-tool makespan (s)")
        clean_axis(axis)
        for index, (bar, value) in enumerate(zip(bars, values)):
            note = f"{value:.2f}"
            if mode == "scale-in" and index == 0:
                note += "\nno drain"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(values) * 0.025,
                note,
                ha="center",
                va="bottom",
                fontsize=7,
            )
        axis.set_ylim(0, max(values) * 1.2)
        axis.text(
            0.02,
            0.96,
            "Warm scale-out" if mode == "scale-out" else "Semantic scale-in",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    fig.subplots_adjust(wspace=0.3)
    save(fig, output_dir, "fig_elasticity")


def interference(data: dict, output_dir: Path) -> None:
    groups = defaultdict(list)
    for row in data["records"]:
        groups[row["mode"]].append(row)
    order = ["no-migration", "sync", "async"]
    labels = ["No migration", "Synchronous", "Asynchronous"]
    colors = [COLORS["baseline"], COLORS["sync"], COLORS["async"]]
    ttft = [statistics.fmean(row["arrival_probe_ttft_seconds"] for row in groups[item]) * 1000 for item in order]
    throughput = [statistics.fmean(row["foreground_throughput_tokens_per_second"] for row in groups[item]) for item in order]
    throughput_change = [
        (value / throughput[0] - 1.0) * 100 for value in throughput
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    axes[0].bar(range(3), ttft, color=colors, width=0.65, zorder=2)
    axes[0].set_xticks(range(3), labels)
    axes[0].set_ylabel("Arrival TTFT (ms)")
    clean_axis(axes[0])
    axes[1].bar(range(3), throughput_change, color=colors, width=0.65, zorder=2)
    axes[1].set_xticks(range(3), labels)
    axes[1].set_ylabel("Throughput change (percent)")
    axes[1].axhline(0, color="#666666", linewidth=0.7)
    margin = max(0.1, max(abs(value) for value in throughput_change) * 1.5)
    axes[1].set_ylim(-margin, margin)
    clean_axis(axes[1])
    fig.subplots_adjust(wspace=0.3)
    save(fig, output_dir, "fig_interference")


def control_plane(data: dict, output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(3.35, 2.25))
    for scale, marker in zip(data["scales"], ("o", "s", "^")):
        rows = [
            row
            for row in scale["operations"]
            if row["operation"] == "ownership_cas"
        ]
        axis.plot(
            [row["concurrency"] for row in rows],
            [row["p99_ms"] for row in rows],
            marker=marker,
            label=f"{scale['agent_count'] // 1000}K agents",
        )
    axis.set_yscale("log")
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 8, 32], ["1", "8", "32"])
    axis.set_xlabel("Concurrent controller clients")
    axis.set_ylabel("Ownership CAS p99 (ms, log)")
    clean_axis(axis)
    axis.legend(frameon=False, loc="upper left")
    save(fig, output_dir, "fig_control_plane")


def replay_and_policy(coding: dict, policy: dict, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))

    scenarios = ["sticky", "reroute", "on-return", "agentshift"]
    labels = [LABELS[item] for item in scenarios]
    colors = [COLORS[item] for item in scenarios]
    coding_groups = {
        scenario: [
            row["makespan_seconds"]
            for row in coding["records"]
            if row["scenario"] == scenario
        ]
        for scenario in scenarios
    }
    coding_means = [statistics.fmean(coding_groups[item]) for item in scenarios]
    axes[0].bar(range(len(scenarios)), coding_means, color=colors, width=0.68, zorder=2)
    for index, scenario in enumerate(scenarios):
        values = coding_groups[scenario]
        axes[0].scatter(
            [index - 0.10, index, index + 0.10][: len(values)],
            values,
            s=10,
            facecolors="white",
            edgecolors="#333333",
            linewidths=0.6,
            zorder=3,
        )
    axes[0].set_xticks(range(len(scenarios)), labels)
    axes[0].set_ylabel("Three-turn makespan (s)")
    axes[0].set_title("8-agent coding replay", loc="left", fontweight="bold")
    clean_axis(axes[0])

    policies = ["fifo", "shortest-kv", "earliest-return", "agentshift-score"]
    policy_labels = ["FIFO", "Shortest\nKV", "Earliest\nreturn", "AgentShift"]
    policy_colors = ["#777777", "#E69F00", "#CC79A7", COLORS["agentshift"]]
    policy_groups = {
        item: [
            row["completed_in_gap_fraction"] * 100
            for row in policy["records"]
            if row["policy"] == item
        ]
        for item in policies
    }
    policy_means = [statistics.fmean(policy_groups[item]) for item in policies]
    axes[1].bar(range(len(policies)), policy_means, color=policy_colors, width=0.68, zorder=2)
    for index, item in enumerate(policies):
        values = policy_groups[item]
        axes[1].scatter(
            [index - 0.10, index, index + 0.10][: len(values)],
            values,
            s=10,
            facecolors="white",
            edgecolors="#333333",
            linewidths=0.6,
            zorder=3,
        )
    axes[1].set_xticks(range(len(policies)), policy_labels)
    axes[1].set_ylabel("Handoffs completed in gap (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title(
        "Heterogeneous blocked agents", loc="left", fontweight="bold"
    )
    clean_axis(axes[1])
    fig.subplots_adjust(wspace=0.3)
    save(fig, output_dir, "fig_replay_policy")


def architecture(output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 2.45))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 5)
    axis.axis("off")

    def box(x, y, width, height, text, fill, edge="#555555", bold=False):
        axis.add_patch(Rectangle((x, y), width, height, facecolor=fill, edgecolor=edge, linewidth=1.0))
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontweight="bold" if bold else "normal",
            fontsize=8,
        )

    def arrow(x1, y1, x2, y2, text="", dashed=False):
        axis.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.0,
                linestyle="--" if dashed else "-",
                color="#555555",
            )
        )
        if text:
            axis.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, text, ha="center", fontsize=7)

    box(0.3, 3.55, 2.0, 0.75, "Agent runtime\nblocked on tool", "#E8E8E8", bold=True)
    box(3.0, 3.55, 2.2, 0.75, "Mobility controller\ngap-aware admission", "#FDE8D5", bold=True)
    box(6.0, 3.55, 2.2, 0.75, "Continuation store\nowner@epoch, step, effect", "#E1F1E8", bold=True)
    box(9.0, 3.55, 2.2, 0.75, "Tool mailbox\npending future", "#F3E8F1", bold=True)
    box(0.8, 0.8, 3.2, 1.15, "Source SGLang\npin completed prefix\nretain shadow", "#E5F0F8", bold=True)
    box(8.0, 0.8, 3.2, 1.15, "Destination SGLang\nreserve and install prefix\nfirst-token ACK", "#E5F0F8", bold=True)
    box(4.7, 0.95, 2.6, 0.85, "Rank-pair KV transfer\nindependent CUDA stream", "#FFF2CC", bold=True)

    arrow(2.3, 3.92, 3.0, 3.92, "tool start")
    arrow(5.2, 3.92, 6.0, 3.92, "CAS")
    arrow(8.2, 3.92, 9.0, 3.92, "rebind")
    arrow(4.0, 1.38, 4.7, 1.38, "K/V")
    arrow(7.3, 1.38, 8.0, 1.38, "install")
    arrow(4.1, 3.55, 2.7, 1.95, "pin/reserve", dashed=True)
    arrow(9.0, 1.95, 5.2, 3.55, "DEST_READY", dashed=True)
    axis.text(6.0, 0.18, "Completed-turn boundary: immutable prefix, no active decode request", ha="center", fontsize=8)
    save(fig, output_dir, "fig_architecture")


def main(args: argparse.Namespace) -> None:
    configure(Path(args.font_dir))
    output_dir = Path(args.output_dir)
    final_dir = Path(args.final_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)
    blocked = json.loads(Path(args.blocked).read_text())
    hotspot_data = json.loads(Path(args.hotspot).read_text())
    workload_data = json.loads(Path(args.workloads).read_text())
    elasticity_data = json.loads(Path(args.elasticity).read_text())
    interference_data = json.loads(Path(args.interference).read_text())
    control_data = json.loads(Path(args.control_plane).read_text())
    coding_data = json.loads(Path(args.coding_replay).read_text())
    policy_data = json.loads(Path(args.policy).read_text())

    context_latency(blocked, output_dir)
    gap_overlap(blocked, output_dir)
    hotspot(hotspot_data, output_dir)
    workloads(workload_data, output_dir)
    elasticity(elasticity_data, output_dir)
    interference(interference_data, output_dir)
    control_plane(control_data, output_dir)
    replay_and_policy(coding_data, policy_data, output_dir)
    architecture(output_dir)
    for source in output_dir.glob("fig_*.pdf"):
        shutil.copy2(source, final_dir / source.name)
    for source in output_dir.glob("fig_*.png"):
        shutil.copy2(source, final_dir / source.name)
    print(json.dumps({"output_dir": str(output_dir), "figures": sorted(path.name for path in output_dir.glob("fig_*"))}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--font-dir", default="paper_rewriting_output/fonts")
    parser.add_argument("--output-dir", default="paper_rewriting_output/figures")
    parser.add_argument("--final-dir", default="paper_rewriting_output/final_paper/figures")
    parser.add_argument("--blocked", default="results/blocked-window-1784565792789187724.json")
    parser.add_argument("--hotspot", default="results/hotspot-1784566696908767902.json")
    parser.add_argument("--workloads", default="results/agent-workloads-1784819110111467791.json")
    parser.add_argument("--elasticity", default="results/elasticity-1784820101002154448.json")
    parser.add_argument("--interference", default="results/interference-1784566871298243631.json")
    parser.add_argument("--control-plane", default="results/control-plane-1784819214062797714.json")
    parser.add_argument(
        "--coding-replay",
        default="results/coding-agent-replay-1784829317183895026.json",
    )
    parser.add_argument(
        "--policy", default="results/migration-policy-1784829520064490654.json"
    )
    main(parser.parse_args())
