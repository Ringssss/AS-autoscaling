"""Plot saved matched-geometry latency data; dots are independent paired repeats."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('experiments/freetoken_elastic/results/20260915')
sets=[('24 GiB / KV','preservation-main-24','kv-resize'),('24 GiB / recurrent','preservation-main-24','mamba-resize'),('32 GiB / KV','preservation-main-32','kv-resize')]
data={};fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,pair,title in zip(axes,['prefix-4096','prefix-16384'],['4K prefix: preservation adds cost','16K prefix: preservation avoids replay']):
 data[pair]={}
 for i,(label,folder,condition) in enumerate(sets):
  rows=json.loads((root/folder/'campaign.json').read_text())['runs'];data[pair][label]={}
  for mode,offset,color in [('drop',-.18,'#587ca5'),('preserve',.18,'#4a9479')]:
   values=[r['metrics']['intervention_to_completion_seconds'] for r in rows if r['pair_id']==pair and r['condition']==condition and r['policy']==mode]
   data[pair][label][mode]=values;med=statistics.median(values)
   ax.bar(i+offset,med,.32,color=color,label=mode if i==0 else None)
   ax.scatter([i+offset-.055,i+offset,i+offset+.055],values,color='black',s=15,zorder=3)
 ax.set_xticks(range(3),[s[0] for s in sets],rotation=10);ax.set_title(title);ax.set_ylabel('Resize to first resumed response completion (s)');ax.set_ylim(0,11);ax.grid(axis='y',alpha=.2);ax.legend()
fig.suptitle('Actual BF16 MoE offload on capacity-constrained H100',fontsize=13)
fig.text(.015,.02,'Three paired repeats; same final pool geometry within each pair. Save, rebuild, restore and response costs included.\nNot RTX measurements. Byte integrity and cache consumption pass; generated text/quality equivalence is not established.',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.94))
for ext in ('png','svg'):fig.savefig(root/f'overview.{ext}',dpi=180)
(root/'overview-data.json').write_text(json.dumps(data,indent=2)+'\n')
