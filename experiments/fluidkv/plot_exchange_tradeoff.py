#!/usr/bin/env python3
"""F-tradeoff: the exchange ratio between recomputing and copying a prefix.

Two series matter and they must not be conflated:
  - the engine-reported transfer phase (what the sweeps time)
  - the client-observed cold TTFT at the same length

The ratio grows monotonically, which is the paper's central asymmetry. What the
figure must NOT imply is that the ratio is a serving speedup: the numerator is a
cold prefill and the denominator is a transfer phase, and group init is excluded
from both. The caption states that.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle

ROOT = Path(__file__).resolve().parents[2]
SWEEP = ROOT / 'experiments/weka_baselines/results/real-20260929'
OUT = ROOT / 'paper' / 'figs' / 'fig-tradeoff.pdf'

LINES = [
    ('E', 'same-node, short', '#2C5F9E', 'o', '-'),
    ('A', 'same-node A', '#1B7F5A', 's', '-'),
    ('B', 'same-node B', '#5A5A5A', '^', '--'),
    ('D', 'cross-node D', '#C46210', 'D', '-'),
]


def load(line):
    rows = json.loads((SWEEP / f'sweep-{line}.json').read_text())
    by = defaultdict(list)
    for r in rows:
        if r.get('dst_verified_runnable') and r.get('prefill_s') and r.get('transfer_s'):
            by[r['n']].append(r)
    out = []
    for n in sorted(by):
        sel = by[n]
        out.append((n,
                    statistics.median(r['prefill_s'] for r in sel),
                    statistics.median(r['transfer_s'] for r in sel),
                    len(sel)))
    return out


def main() -> None:
    figstyle.configure(base_size=11.0)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.8, 3.3),
                                   gridspec_kw={'width_ratios': [1.05, 1.0]})

    all_ratio = []
    for line, label, color, marker, ls in LINES:
        data = load(line)
        ns = [d[0] for d in data]
        pre = [d[1] for d in data]
        xfr = [d[2] for d in data]
        ratio = [p / t for p, t in zip(pre, xfr)]

        ax0.plot(ns, ratio, marker=marker, color=color, ls=ls, label=label)
        ax1.plot(ns, xfr, marker=marker, color=color, ls=ls, label=label)
        all_ratio += ratio

    ax0.set_xscale('log'); ax0.set_yscale('log')
    ax0.set_xlabel('Prefix length  (tokens)', fontsize=11)
    ax0.set_ylabel('Recompute / copy  (x)', fontsize=11)
    figstyle.panel_title(ax0, '(a) Exchange ratio')
    ax0.axhline(1.0, color='#888888', ls=':', lw=1.0, zorder=1)

    # the crossover is the point of the left panel
    e = load('E')
    if e:
        n0, p0, t0, _ = e[0]
        ax0.annotate(f'L* ~ {n0:,}', xy=(n0, p0 / t0), xytext=(n0 * 1.1, p0 / t0 * 2.2),
                     fontsize=10, color='#333333',
                     arrowprops=dict(arrowstyle='->', color='#666666', lw=0.9))

    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('Prefix length  (tokens)', fontsize=11)
    ax1.set_ylabel('Transfer phase  (s)', fontsize=11)
    figstyle.panel_title(ax1, '(b) Copy cost stays nearly flat')

    ax0.legend(frameon=False, fontsize=9.5, loc='upper left')
    fig.tight_layout()
    figstyle.save(fig, OUT)
    print(f'wrote {OUT} and {OUT.with_suffix(".png")}')
    for line, label, _, _, _ in LINES:
        d = load(line)
        if d:
            print(f'  {label:18s} {len(d)} lengths, '
                  f'n={d[0][0]:,}..{d[-1][0]:,}, '
                  f'ratio {d[0][1] / d[0][2]:.2f}x..{d[-1][1] / d[-1][2]:.2f}x')


if __name__ == '__main__':
    main()
