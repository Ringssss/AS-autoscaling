#!/usr/bin/env python3
"""F-timeline: the three actions compared at one common endpoint.

Every bar ends at the same event — the first streamed output token — so the
comparison is a serving comparison, not a phase-to-phase comparison. The
earlier exchange ratio divided a client cold TTFT by an engine-side transfer
phase, which are different endpoints and different clocks.

The result the figure has to carry: import is not currently a win. The prefix
arrives intact (inspect confirms a full runnable prefix) but the first request
that should reuse it pays a full prefill.
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
SRC = ROOT / 'results' / 'action-compare-130k.json'
OUT = ROOT / 'paper' / 'figs' / 'fig-timeline.pdf'

ARM = [
    ('cold_ttft_s', 'Cold\n(target recomputes)', 'cold'),
    ('local_ttft_s', 'Local\n(prefix resident)', 'hit'),
    ('import_ttft_s', 'Import\n(state moved first)', 'copy'),
]


def main() -> None:
    figstyle.configure(base_size=11.0)
    rows = json.loads(SRC.read_text())

    fig, ax = plt.subplots(figsize=(4.6, 3.2))

    for i, (key, label, color) in enumerate(ARM):
        vals = [r[key] for r in rows if r.get(key)]
        med = statistics.median(vals)
        ax.bar(i, med, width=0.58, color=figstyle.COLORS[color], zorder=3,
               edgecolor='white', linewidth=0.9)
        ax.scatter([i] * len(vals), vals, s=26, color='#111111', zorder=5,
                   linewidths=0, alpha=0.85)
        ax.text(i, med + 0.5, f'{med:.1f}', ha='center', va='bottom',
                fontsize=11, fontweight='bold', zorder=6)

    ax.set_xticks(range(len(ARM)))
    ax.set_xticklabels([a[1] for a in ARM], fontsize=10.5)
    ax.set_ylabel('Time to first output token  (s)', fontsize=11)
    ax.set_ylim(0, max(r['cold_ttft_s'] for r in rows) * 1.22)
    figstyle.panel_title(ax, 'One 130K prefix, three actions')

    cold = statistics.median(r['cold_ttft_s'] for r in rows if r.get('cold_ttft_s'))
    imp = statistics.median(r['import_ttft_s'] for r in rows if r.get('import_ttft_s'))
    ax.annotate('', xy=(2, imp), xytext=(0, cold),
                arrowprops=dict(arrowstyle='<->', color='#333333', lw=1.0))
    ax.text(1.0, max(cold, imp) * 0.66, f'{cold / imp:.2f}x\n(no gain)',
            ha='center', va='center', fontsize=11.5, fontweight='bold',
            color=figstyle.COLORS['accent'], linespacing=1.4)
    # the caption line sits under the axis label, where nothing can collide
    ax.text(0.5, -0.30, 'prefix verified runnable on the target before serving',
            transform=ax.transAxes, ha='center', va='top', fontsize=9,
            color='#555555')

    fig.tight_layout()
    figstyle.save(fig, OUT)
    print(f'wrote {OUT} and {OUT.with_suffix(".png")}')
    for key, label, _ in ARM:
        vals = [r[key] for r in rows if r.get(key)]
        print(f'  {label.replace(chr(10), " "):32s} median {statistics.median(vals):7.3f}s  n={len(vals)}')


if __name__ == '__main__':
    main()
