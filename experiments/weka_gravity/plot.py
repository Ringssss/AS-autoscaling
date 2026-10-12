"""Standalone figures for the full-trace applicability audit."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).parent/'results'
a=json.loads((root/'audit.json').read_text())
with (root/'requests.csv').open() as f: rows=list(csv.DictReader(f))
fig, axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
for sub,label in [('False','Main stream'),('True','Subagents')]:
    v=np.sort([int(r['input']) for r in rows if r['subagent']==sub])
    axes[0].plot(v,np.arange(1,len(v)+1)/len(v),label=label)
axes[0].axvline(131072,color='black',ls='--',alpha=.5,label='Local 128K context')
axes[0].set(xscale='log',xlabel='Input tokens',ylabel='Request CDF',title='Long per-conversation histories')
axes[0].legend(fontsize=8)
keys=['hit60','hit300','hit1800','hit']
axes[1].bar(['60 s','5 min','30 min','Unlimited'],[100*a['reuse_by_ttl'][k] for k in keys])
axes[1].set(ylim=(0,100),ylabel='Reusable input tokens (%)',title='Completed, same-model prefix reuse')
axes[1].text(.02,.03,'Capacity-unbounded analytical cache\nTTL is a sensitivity assumption',transform=axes[1].transAxes,fontsize=8)
for b,label in [(20480,'20 KiB/token (local hybrid)'),(131072,'128 KiB/token (scenario)'),(327680,'320 KiB/token (scenario)')]:
    s=[r for r in a['copy_window_sensitivity'] if r['bytes_per_token']==b]
    axes[2].plot([r['bandwidth_GB_s'] for r in s],[r['fit_fraction']*100 for r in s],marker='o',label=label)
axes[2].set(xlabel='Assumed effective bandwidth (GB/s)',ylabel='Fits observed gap (%)',ylim=(0,105),title='Migration-window opportunity, not speedup')
axes[2].legend(fontsize=7)
for ax in axes:ax.grid(alpha=.2,axis='y')
fig.savefig(root/'trace_evidence.png',dpi=180)
fig.savefig(root/'trace_evidence.pdf')
