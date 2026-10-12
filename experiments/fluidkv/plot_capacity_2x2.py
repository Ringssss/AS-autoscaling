#!/usr/bin/env python3
"""Figures for the paired 2x2 capacity experiment.

Panel (a): TTFT by condition, individual repetitions overlaid on the median.
Panel (b): free token slots after the request, with how many resident prefixes
survived, read per object rather than inferred from a slot delta.

The cold cells are drawn in their own panel-worth of space because their
free-slot count (a quarter of the pool) would otherwise flatten the occupied
cells, which are the comparison the figure exists to make.

No eviction *count* is drawn. The engine's `num_retractions` field counts
per-request retractions, not cache evictions.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results' / 'capacity-2x2.json'
OUT = ROOT / 'paper' / 'figs' / 'fig-capacity-2x2.pdf'

# only the occupied pair is compared in panel (b); the cold cells appear in (a)
OCC = ['full_hit', 'full_miss']
LAT = ['empty_miss', 'empty_hit', 'full_hit', 'full_miss']
LABEL = {
    'empty_miss': 'Cold\n(engine idle)',
    'empty_hit': 'Hit\n(engine idle)',
    'full_hit': 'Hit\n(engine full)',
    'full_miss': 'Miss\n(engine full)',
}
SHORT = {'full_hit': 'Hit\n(engine full)', 'full_miss': 'Miss\n(engine full)'}
COLOR = {
    'empty_miss': figstyle.COLORS['cold'],
    'empty_hit': figstyle.COLORS['hit'],
    'full_hit': figstyle.COLORS['hit'],
    'full_miss': figstyle.COLORS['miss'],
}


def main() -> None:
    figstyle.configure(base_size=11.0)
    rows = json.loads(SRC.read_text())
    by = {c: [r for r in rows if r['cond'] == c] for c in LAT}

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.8, 3.4),
                                   gridspec_kw={'width_ratios': [1.45, 1.0]})

    # ---------------- (a) TTFT, all four conditions ----------------
    for i, cond in enumerate(LAT):
        vals = [r['ttft_s'] for r in by[cond] if r['ttft_s']]
        med = statistics.median(vals)
        ax0.bar(i, med, width=0.58, color=COLOR[cond], zorder=3,
                edgecolor='white', linewidth=0.9)
        ax0.scatter([i] * len(vals), vals, s=26, color='#111111', zorder=5,
                    linewidths=0, alpha=0.85)
        ax0.text(i, med + 0.10, f'{med:.2f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold', zorder=6, color='#111111')
    ax0.set_xticks(range(len(LAT)))
    ax0.set_xticklabels([LABEL[c] for c in LAT], fontsize=10.5)
    ax0.set_ylabel('Time to first output event  (s)', fontsize=11)
    ax0.set_ylim(0, max(r['ttft_s'] for r in rows) * 1.20)
    figstyle.panel_title(ax0, '(a) Latency')

    h = statistics.median(r['ttft_s'] for r in by['full_hit'] if r['ttft_s'])
    m = statistics.median(r['ttft_s'] for r in by['full_miss'] if r['ttft_s'])
    ax0.annotate('', xy=(3, m), xytext=(2, h),
                 arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.1))
    ax0.text(2.5, (h + m) / 2 + 0.42, f'{m / h:.1f}x', ha='center', va='bottom',
             fontsize=12, fontweight='bold', color='#222222')

    # ---------------- (b) capacity, occupied cells only ----------------
    figstyle.panel_title(ax1, '(b) Capacity at equal occupancy')
    before = statistics.median(r['free_tok_before'] for r in by['full_miss']
                               if r['free_tok_before'] is not None)
    extra = {}
    for i, cond in enumerate(OCC):
        vals = [r['free_tok_after'] for r in by[cond] if r['free_tok_after'] is not None]
        med = statistics.median(vals)
        ax1.bar(i, med, width=0.50, color=COLOR[cond], zorder=3,
                edgecolor='white', linewidth=0.9)
        ax1.text(i, med + before * 0.035, f'{med:,.0f}', ha='center', va='bottom',
                 fontsize=11, fontweight='bold', zorder=6, color='#111111')
        surv = [r['survivors'] for r in by[cond] if r['survivors'] is not None]
        n = int(statistics.median(surv))
        lost = 7 - n
        # survival becomes a third line of the tick label: it can never be
        # clipped by a narrow bar and never collides with the value label
        extra[cond] = f'{n}/7 intact' if lost == 0 else f'{n}/7 intact, {lost} evicted'

    before = statistics.median(r['free_tok_before'] for r in by['full_miss']
                               if r['free_tok_before'] is not None)
    ax1.axhline(before, color='#444444', ls='--', lw=1.1, zorder=4)
    ax1.text(-0.44, before * 1.045, f'free before request: {before:,.0f}',
             ha='left', va='bottom', fontsize=10, color='#333333', zorder=8,
             bbox=dict(facecolor='white', edgecolor='none', pad=1.6))
    ax1.set_xticks(range(len(OCC)))
    ax1.set_xticklabels([f'{SHORT[c]}\n{extra.get(c, "")}' for c in OCC], fontsize=10.5)
    ax1.set_ylabel('KV token slots free after request', fontsize=11)
    ax1.set_ylim(0, before * 1.46)
    ax1.set_xlim(-0.5, len(OCC) - 0.5)

    fig.tight_layout()
    figstyle.save(fig, OUT)
    print(f'wrote {OUT} and {OUT.with_suffix(".png")}')

    print('\n=== summary ===')
    for cond in LAT:
        s = by[cond]
        if not s:
            continue
        print(f'{cond:11s} n={len(s)} '
              f'ttft_med={statistics.median(r["ttft_s"] for r in s):.3f}s '
              f'range=[{min(r["ttft_s"] for r in s):.3f},{max(r["ttft_s"] for r in s):.3f}] '
              f'free_after={sorted({r["free_tok_after"] for r in s})} '
              f'surv={sorted({r["survivors"] for r in s}, key=str)}')


if __name__ == '__main__':
    main()
