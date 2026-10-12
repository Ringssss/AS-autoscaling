"""Freeze a held-out, original-length comparison and explicit harness delta."""
import ast
import hashlib
import json
from pathlib import Path
import difflib

ROOT=Path(__file__).parent
BASE=ROOT.parent/'weka_baselines'


def heldout():
    rows=[json.loads(s) for s in (BASE/'data/requests.jsonl').read_text().splitlines()]
    previous=set(json.loads((BASE/'results/one-hour-20260929/cohort.json').read_text())['streams'])
    groups={}
    for r in rows:
        if r['turn']<3:groups.setdefault(r['stream_index'],[]).append(r)
    bins={0:[],1:[],2:[]}
    for i,rs in sorted(groups.items()):
        if i in previous or len(rs)!=3 or sum(r['output_tokens'] for r in rs)>900:continue
        n=max(r['input_tokens'] for r in rs)
        if n>=98304:continue
        bins[0 if n<32768 else 1 if n<65536 else 2].append(i)
    chosen=[i for b in range(3) for i in bins[b][:4]]
    assert len(chosen)==12
    selected=[r for i in chosen for r in groups[i]]
    return selected,dict(streams=chosen,excluded_previous=sorted(previous),output_tokens=sum(r['output_tokens'] for r in selected),
        selection='Held-out relative to one-hour cohort. Four streams per max-input bin <32K,32-64K,64-96K; first in existing seeded order; first 3 original turns; <=900 total output tokens per stream; no outcome-based selection or truncation.')


def build():
    out=ROOT/'environment';out.mkdir(exist_ok=True)
    before=(BASE/'one_hour.py').read_text();text=before
    start=text.index('async def trial(');end=text.index('\n\nasync def experiment(',start)
    header='''# Generated from frozen one_hour.py; inspect harness.patch for changes.
import asyncio
from collections import Counter
import hashlib
import json
import math
import random
import time
import uuid
from agentshift.state.schema import AgentContinuation
from experiments.fluidkv.dualmap_adapter import DualMapAdapter
from experiments.fluidkv.benchmark_native_pairs import flush
from experiments.multinode.benchmark_highbs import settle_engines
from experiments.multinode.cacheroute import dispatch
from experiments.weka_baselines.common import expand_dualmap, rate_plan
from experiments.weka_baselines.replay import generate, write
from pathlib import Path
URLS=[f'http://12.11.3.{i}:31780' for i in [27,28,29]]
OUT=Path(__file__).resolve().parents[1]/'results/compare'
'''
    original_trial=text[start:end];text=original_trial
    text=text.replace("move=policy.endswith('+move')","move=policy.endswith(('+move','+multi')); multi=policy.endswith('+multi')")
    text=text.replace('current_expected=48','current_expected=len(rows)')
    text=text.replace('''                    prepared=False
                    for r in candidates:''','''                    prepared=False
                    copied_total=0
                    # Hard network/state budget applies only to the proposed
                    # multiple-placement optimization; current +move retained.
                    for r in candidates:
                        if multi and time.perf_counter() >= offered:
                            event('skip_preparation',phase=phase,reason='preparation_window_exhausted');break''')
    text=text.replace("event('skip_preparation',phase=phase,reason='already_warm',key=r['key']);break", "event('skip_preparation',phase=phase,reason='already_warm',key=r['key'])\n                            if multi:continue\n                            break")
    text=text.replace("if dst.get('available_token_slots',0)<1048576 or dst.get('evictable_token_slots',0)>0:","if not multi and (dst.get('available_token_slots',0)<1048576 or dst.get('evictable_token_slots',0)>0):")
    anchor="                        agent=uuid.uuid4().hex\n"
    assert text.count(anchor)==1
    text=text.replace(anchor,"""                        predicted_bytes=n*20480+64389120
                        if multi and (copied_total+predicted_bytes > 3*1024**3 or dst.get('available_token_slots',0)<n or dst.get('available_snapshot_slots',0)<1):
                            event('skip_preparation',phase=phase,reason='copy_budget_or_capacity',key=r['key']);continue
"""+anchor)
    text=text.replace("preps.append(prep);event('migration',**prep);prepared=True","preps.append(prep);event('migration',**prep);prepared=True;copied_total+=copied.bytes_transferred")
    anchor="""                            event('migration_failed',phase=phase,key=r['key'],error=repr(e));raise
                        break"""
    assert anchor in text
    text=text.replace(anchor,"""                            event('migration_failed',phase=phase,key=r['key'],error=repr(e));raise
                        if not multi:break""")
    ast.parse(header+text)
    (out/'comparison_trial.py').write_text(header+text+'\n')
    (out/'harness.patch').write_text(''.join(difflib.unified_diff(original_trial.splitlines(True),text.splitlines(True),fromfile='one_hour.trial',tofile='bounded_nonempty.trial')))
    (out/'harness-source.json').write_text(json.dumps(dict(original_sha256=hashlib.sha256(before.encode()).hexdigest(),
        candidate_sha256=hashlib.sha256((header+text+'\n').encode()).hexdigest()),indent=2)+'\n')
    selected,meta=heldout();target=ROOT/'results/compare';target.mkdir(exist_ok=True)
    (target/'cohort.json').write_text(json.dumps(meta,indent=2)+'\n')
    (target/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in selected))
    print(meta)


if __name__=='__main__':build()
