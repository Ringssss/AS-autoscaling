"""Plot measured whole-window mean TTFT/E2E ratios; no simulated speedups."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def main(path,out):
    data=json.loads(path.read_text())
    comparisons=data['comparisons']
    pairs=[('p2c','p2c+fluidkv'),('dualmap','dualmap+fluidkv'),('cacheroute','cacheroute+fluidkv'),('dualmap','p2c+fluidkv'),('cacheroute','p2c+fluidkv')]
    fields=['stage','window','length','withdraw','tool_seconds','rate_scale','output_cap']
    keys=sorted({tuple(c[f] for f in fields) for c in comparisons},key=lambda k:tuple(map(str,k)))
    labels=[f'{k[0]} | {k[1]} | L={k[2]} | '+('exit' if k[3] else 'stay')+f' | W={k[4]} | rate={k[5]}' for k in keys]
    fig,axes=plt.subplots(1,2,figsize=(15,max(6,.34*len(keys)+2)),layout='constrained')
    for ax,metric,title in zip(axes,['mean_ttft_speedup','mean_e2e_speedup'],['Whole-window mean TTFT speedup','Whole-window mean completion-latency speedup']):
        values=np.full((len(keys),len(pairs)),np.nan)
        for i,k in enumerate(keys):
            for j,p in enumerate(pairs):
                v=[c[metric] for c in comparisons if tuple(c[f] for f in fields)==k and (c['baseline'],c['treatment'])==p]
                if v: values[i,j]=np.median(v)
        im=ax.imshow(np.log2(values),aspect='auto',cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-1,vcenter=0,vmax=1))
        ax.set_xticks(range(len(pairs)),['P2C+F / P2C','DM+F / DM','CR+F / CR','P2C+F / DM','P2C+F / CR'],rotation=40,ha='right')
        ax.set_yticks(range(len(keys)),labels if ax is axes[0] else ['']*len(keys))
        ax.set_title(title+'\n>1 better; repeats summarized by paired median')
        for i in range(len(keys)):
            for j in range(len(pairs)):
                affected=any(not c.get('observation_clean',True) for c in comparisons
                    if tuple(c[f] for f in fields)==keys[i] and (c['baseline'],c['treatment'])==pairs[j])
                if np.isfinite(values[i,j]):ax.text(j,i,f'{values[i,j]:.2f}'+('*' if affected else ''),ha='center',va='center',fontsize=8)
    fig.suptitle('Kimi timing/length-driven replay on Qwen3.6-35B-A3B; 3 nodes / 6 GPUs\nScreen: output cap64. Native: original outputs. Shared states and handoffs are controlled.\nF = FluidKV; DM = DualMap; CR = CacheRoute. * = checkpoint observation error.')
    out.mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'measured-speedups.png',dpi=170);fig.savefig(out/'measured-speedups.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.input,a.output)
