"""Export standalone figures; exploratory and confirmation data stay separate."""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(root):
    data=json.loads((root/'report.json').read_text())
    for phase in ('screen','confirm'):
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
        plotted=False
        for column,kind in enumerate(('shared','mixed')):
            batch=next((r for r in data['batches'] if r['batch']==f'{kind}-{phase}'),None)
            if batch is None:continue
            groups=defaultdict(lambda:defaultdict(list))
            for row in batch['rows']:
                groups[row['policy'],row['prep']][row['clients']].append(row)
            for index,((policy,prep),points) in enumerate(sorted(groups.items())):
                xs=sorted(points)
                if not xs:continue
                plotted=True
                gains=[np.median([r['reduction_pct'] for r in points[x]]) for x in xs]
                lows=[min(r['reduction_pct'] for r in points[x]) for x in xs]
                highs=[max(r['reduction_pct'] for r in points[x]) for x in xs]
                label=policy+' + '+prep
                style='-' if prep=='copy' else '--'
                color=f'C{index}'
                axes[0,column].errorbar(xs,gains,yerr=[np.asarray(gains)-lows,np.asarray(highs)-gains],
                    marker='o',linestyle=style,color=color,capsize=3,label=label)
                axes[1,column].plot(xs,[100*np.median([r['slo'] for r in points[x]]) for x in xs],
                    marker='o',linestyle=style,color=color,label=label)
            axes[0,column].set_title(('Shared 64K prefix' if kind=='shared' else '4 independent 32K prefixes')+
                '\n'+phase+' / '+batch['status'])
            axes[0,column].axhline(0,color='black',linewidth=.8)
            axes[0,column].axhline(5,color='gray',linestyle=':',linewidth=.8)
            axes[1,column].axhline(95,color='gray',linestyle=':',linewidth=.8)
            axes[1,column].set_ylim(-2,103)
            for ax in axes[:,column]:
                ax.set_xscale('log',base=2)
                ax.set_xticks([4,8,16,32,64,128],[4,8,16,32,64,128])
                ax.grid(alpha=.2)
            if groups:axes[0,column].legend(fontsize=8)
            axes[1,column].set_xlabel('Concurrent closed-loop clients')
        axes[0,0].set_ylabel('Wall reduction vs stronger routing baseline (%)')
        axes[1,0].set_ylabel('Joint SLO attainment (%)')
        fig.text(.5,.01,'Fixed output: 128 tokens/turn; 2 turns; 1s preparation gap. '
            'Error bars: observed min/max, not confidence intervals.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,1))
        if plotted:
            for ext in ('png','pdf','svg'):
                fig.savefig(root/f'concurrency-{phase}.{ext}',dpi=180)
        plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path)
    plot(p.parse_args().root)
