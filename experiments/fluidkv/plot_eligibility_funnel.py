#!/usr/bin/env python3
"""F-funnel: where requests drop out of the mobility-eligible set.

The funnel is the paper's honest scope statement: a mechanism can be cheap and
still apply to a small slice of traffic. Each stage is a necessary condition,
so the first stage that rejects a request is the one to report.

Reading: the dominant loss is `stays_on_home_engine`, i.e. under sticky
placement the prefix is already local and no migration is needed at all. That
is the reason a cheap transfer does not automatically become a system win.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'results' / 'eligibility-funnel.json'
OUT = ROOT / 'paper' / 'figs' / 'fig-eligibility-funnel.pdf'

STAGE_LABEL = {
    '1_arrive': 'Requests arriving',
    '2_has_prefix': 'Have a reusable prefix',
    '3_above_Lstar': 'Prefix above L*',
    '4_remote_candidate': 'Prefix lives elsewhere',
    '5_capacity_feasible': 'Destination has room',
    '6_net_positive': 'Net gain positive',
}
ORDER = list(STAGE_LABEL)


def main() -> None:
    figstyle.configure(base_size=11.0)
    d = json.loads(SRC.read_text())
    stages = d['stages']
    total = stages['1_arrive']

    fig, ax = plt.subplots(figsize=(7.2, 3.3))

    ys = list(range(len(ORDER)))[::-1]
    for y, key in zip(ys, ORDER):
        n = stages[key]
        share = n / total
        ax.barh(y, share, height=0.62, color=figstyle.COLORS['copy']
                if key == '6_net_positive' else figstyle.COLORS['cold'],
                alpha=1.0 if key == '6_net_positive' else 0.55,
                zorder=3, edgecolor='white', linewidth=0.9)
        ax.text(share + 0.012, y, f'{n:,}  ({share * 100:.1f}%)',
                va='center', ha='left', fontsize=10.5, zorder=6)

    ax.set_yticks(ys)
    ax.set_yticklabels([STAGE_LABEL[k] for k in ORDER], fontsize=10.5)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel('Share of arriving requests', fontsize=11)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], fontsize=10.5)
    figstyle.panel_title(ax, 'Mobility eligibility funnel')

    # annotate the dominant loss
    lost = d['rejection_reasons'].get('stays_on_home_engine')
    if lost:
        ax.text(0.99, 0.06,
                f'Main loss: {lost:,} requests ({lost / total * 100:.1f}%)\n'
                f'keep their prefix at home —\nno migration needed.',
                transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
                color='#222222', linespacing=1.4,
                bbox=dict(facecolor='white', edgecolor='#BBBBBB', pad=3.5))

    fig.tight_layout()
    figstyle.save(fig, OUT)
    print(f'wrote {OUT} and {OUT.with_suffix(".png")}')
    for k in ORDER:
        print(f'  {STAGE_LABEL[k]:28s} {stages[k]:>6,}  {stages[k] / total * 100:>5.1f}%')
    print('  rejections:', d['rejection_reasons'])


if __name__ == '__main__':
    main()
