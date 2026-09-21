"""Compact saved-artifact overview; raw paired points, no significance claims."""
import json,statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('experiments/agentic_profile/results/20260915')
light=json.loads((root/'cpu-staggered-8core/campaign.json').read_text())['runs']
heavy=json.loads((root/'cpu-heavy-8core/campaign.json').read_text())['runs']
state=[json.loads(l) for l in (root/'state-main/records.jsonl').read_text().splitlines()]
data={'cpu':{},'gpu2':{},'gpu8':{}}
for title,rows,policy in [('Light tool / 4+4',light,'frontend-isolated'),('Large tool / 4+4',heavy,'frontend-isolated'),('Large tool / 2+6',heavy,'frontend-reserved2')]:
 vals=[]
 for rep in range(3):
  a=next(r for r in rows if r['concurrency']==8 and r['repetition']==rep and r['policy']==policy)
  b=next(r for r in rows if r['concurrency']==8 and r['repetition']==rep and r['policy']=='shared')
  vals.append(100*(a['e2e_p50']/b['e2e_p50']-1))
 data['cpu'][title]=vals
for c in (2,8):
 for label,endpoint,method in [('Copy / same GPU','destination','copy'),('Replay / same GPU','destination','replay'),('Replay / other GPU','source','replay')]:
  vals=[]
  for rep in range(3):
   group=[r for r in state if r['concurrency']==c and r['repetition']==rep and r['foreground_target']==endpoint]
   a=next(r for r in group if r['method']==method and r['mode']=='concurrent');b=next(r for r in group if r['mode']=='foreground-only')
   vals.append(100*(a['foreground_decode_p50_seconds_per_token']/b['foreground_decode_p50_seconds_per_token']-1))
  data[f'gpu{c}'][label]=vals
fig,axes=plt.subplots(1,3,figsize=(14,4.7))
for ax,key,title,ylabel in zip(axes,['cpu','gpu2','gpu8'],['CPU partition: 8 concurrent agents','GPU preparation: 2 requests','GPU preparation: 8 requests'],['Agent E2E p50 change (%)','Decode time/token change (%)','Decode time/token change (%)']):
 for i,(label,values) in enumerate(data[key].items()):
  med=statistics.median(values)
  ax.bar(i,med,color=['#5683a8','#ce7947','#4b9a80'][i],alpha=.65,width=.6)
  ax.scatter([i-.08,i,i+.08],values,c='#222222',s=22,zorder=3)
 ax.axhline(0,c='black',lw=.8);ax.set_xticks(range(3),list(data[key]),rotation=15,ha='right');ax.set_title(title,fontsize=11);ax.set_ylabel(ylabel);ax.grid(axis='y',alpha=.2)
fig.suptitle('Tested static CPU partitions lose pooling; replay interferes with decode, copying mostly overlaps',fontsize=13)
fig.text(.01,.025,'Dots: three paired repeats. CPU: equal 8+4 cores / 1 GPU. State: fixed 2 GPUs; diagnostic foreground baseline adds less work.\nHeavy CPU has input/output identity caveats; state results do not establish dedicated-GPU or full-cluster economics.',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.93))
for ext in ('png','svg'):fig.savefig(root/f'decision-overview.{ext}',dpi=180)
(root/'decision-overview-data.json').write_text(json.dumps(data,indent=2)+'\n')
