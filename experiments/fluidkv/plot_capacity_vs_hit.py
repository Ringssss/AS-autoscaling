#!/usr/bin/env python3
"""Plot observed first-stream-event latency and allocator free slots.

Conditions use different engines and token regions. Equal initial free-slot
counts do not isolate a causal pressure effect. Missing survival observations
must remain missing; a free-slot decrease is not an eviction counter.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as _fs

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "results" / "capacity-vs-hit.json"
OUT = ROOT / "paper" / "figs" / "fig-capacity-vs-hit.pdf"

# Same three-hue system-conference palette used by plot_decision_validation.py.
COLORS = {"cold": "#27629B", "hit_tight": "#168270", "miss_tight": "#C66624"}
ORDER = ["cold", "hit_tight", "miss_tight"]
LABELS = {"cold": "Cold\n(engine empty)",
          "hit_tight": "Hit, occupied\n(7x32K resident)",
          "miss_tight": "Miss, occupied\n(7x32K resident)"}
TOTAL_SLOTS = 262144


def main() -> None:
    rows = json.loads(SRC.read_text())
    by = {c: [r for r in rows if r["cond"] == c] for c in ORDER}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.4, 3.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # --- left: TTFT, individual reps + median ---------------------------------
    for i, cond in enumerate(ORDER):
        vals = [r["ttft_s"] for r in by[cond]]
        med = statistics.median(vals)
        ax0.bar(i, med, width=0.52, color=COLORS[cond], zorder=3)
        ax0.scatter([i] * len(vals), vals, s=17, color="black", zorder=5,
                    linewidths=0, alpha=0.75)
        ax0.text(i, med + 0.28, f"{med:.2f}s", ha="center", fontsize=9,
                 fontweight="bold", zorder=6)
    ax0.set_xticks(range(len(ORDER)))
    ax0.set_xticklabels([LABELS[c] for c in ORDER], fontsize=8.5)
    ax0.set_ylabel("Time to first stream event (s)", fontsize=9)
    ax0.set_ylim(0, 11.6)
    ax0.grid(axis="y", alpha=0.22, zorder=0)
    ax0.set_axisbelow(True)
    ax0.set_title("(a) Latency", fontsize=9.5, loc="left")
    # the headline ratio, drawn where it reads
    hit = statistics.median(r["ttft_s"] for r in by["hit_tight"])
    miss = statistics.median(r["ttft_s"] for r in by["miss_tight"])
    ax0.annotate("", xy=(2, miss), xytext=(1, hit),
                 arrowprops=dict(arrowstyle="<->", color="#444444", lw=0.9))
    ax0.text(1.5, 6.0, f"{miss / hit:.1f}x", ha="center", fontsize=9.5, fontweight="bold",
             color="#444444")

    # Allocator availability only: these counts do not measure eviction.
    before = 32768
    width = 0.52
    for i, cond in enumerate(ORDER):
        if cond == "cold":
            continue
        avail_after = statistics.median(r["avail_after"] for r in by[cond])
        x = i
        ax1.bar(x, avail_after, width=width, color=COLORS[cond], zorder=3)
        ax1.text(x, avail_after + 1000, f"{avail_after:,.0f}", ha="center", fontsize=8)
        survivors = [r["survivors"] for r in by[cond]]
        if any(s is None for s in survivors):
            label = "Survival unmeasured"
        else:
            count = (str(survivors[0]) if len(set(survivors)) == 1
                     else f"{min(survivors)}-{max(survivors)}")
            label = f"{count}/7 retained\n(target included)"
        ax1.text(x, 38000, label,
                 ha="center", fontsize=7.5,
                 color="#1A1A1A", zorder=6)
    ax1.axhline(before, color="#666666", ls="--", lw=0.9, zorder=4)
    ax1.text(0.5, before - 1500, "  free before request", ha="left", va="top",
             fontsize=7.5, color="#444444", zorder=6,
             bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(["Hit, occupied", "Miss, occupied"], fontsize=8.5)
    ax1.set_xlim(0.4, 2.6)
    ax1.set_ylim(0, 44000)
    ax1.set_ylabel("KV slots free after request", fontsize=9)
    ax1.grid(axis="y", alpha=0.22, zorder=0)
    ax1.set_axisbelow(True)
    ax1.set_title("(b) Free slots after completion", fontsize=9.5, loc="left")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT} and {OUT.with_suffix('.png')}")


if __name__ == "__main__":
    _fs.configure(base_size=11.0)
    main()
