"""Unified figure style for the FluidKV paper.

One place for palette, typography and axes so every figure reads as one system.
Fonts: Arial if the TTFs are present locally, else Liberation Sans, which is
metric-compatible with Arial. Both are sans-serif so line breaks do not move.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

# --- palette -----------------------------------------------------------------
# Three-role system palette: a cool neutral for the baseline, a green for the
# good outcome, a warm orange for the bad outcome. Colour carries meaning:
# green = reuse served from cache, orange = recompute paid, blue = cost of
# moving the state itself.
COLORS = {
    "hit": "#1B7F5A",        # reused, resident  -> good
    "miss": "#D2691E",       # recompute, evicted -> bad
    "cold": "#2C5F9E",       # cold baseline
    "copy": "#7A4FA3",       # state transfer
    "sticky": "#4D4D4D",
    "dualmap": "#C46210",
    "cacheroute": "#1F77B4",
    "balanced": "#9E9E9E",
    "neutral": "#5A5A5A",
    "accent": "#B2182B",
}
# ordered categorical ramp for >=3 series
RAMP = ["#2C5F9E", "#1B7F5A", "#D2691E", "#7A4FA3", "#C46210", "#5A5A5A"]


def configure(font_dirs: "list[Path] | None" = None, base_size: float = 11.0) -> str:
    """Register fonts and set rcParams. Returns the family in use."""
    family = None
    for d in (font_dirs or []):
        d = Path(d)
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.[to]tf")):
            try:
                font_manager.fontManager.addfont(str(f))
            except Exception:
                continue
        for cand in ("Arial.ttf", "Arial.TTF", "arial.ttf"):
            if (d / cand).exists():
                family = font_manager.FontProperties(fname=str(d / cand)).get_name()
                break
        if family:
            break
    if not family:
        # metric-compatible stand-in; keeps the same metrics as Arial
        available = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ("Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"):
            if cand in available:
                family = cand
                break
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [family, "Liberation Sans", "DejaVu Sans"],
        "font.size": base_size,
        "axes.titlesize": base_size + 1.0,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "legend.fontsize": base_size - 1.5,
        "figure.dpi": 140,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
    })
    return family or "sans-serif"


def panel_title(ax, text: str) -> None:
    ax.set_title(text, loc="left", fontweight="bold")


def save(fig, out_pdf: Path, also_png: bool = True) -> None:
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf")
    if also_png:
        fig.savefig(out_pdf.with_suffix(".png"), format="png")
    plt.close(fig)
