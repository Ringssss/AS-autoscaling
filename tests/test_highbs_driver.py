import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.multinode import benchmark_highbs as bench
from experiments.multinode.cacheroute import build_plan


@pytest.mark.parametrize('policy',['dualmap','cacheroute','dualmap+fluidkv','cacheroute+fluidkv','sticky'])
def test_event_driver_drains_burst_without_polling_or_losing_callbacks(monkeypatch,policy):
    source=os.environ.get('FLUIDKV_DUALMAP_SOURCE')
    if not source:pytest.skip('pinned DualMap checkout required')
    async def flush(*args):pass
    async def stream(session,url,tokens,count,on_first_token=None):
        await asyncio.sleep(.001)
        if on_first_token:await on_first_token({},.001)
        await asyncio.sleep(.001)
        return {'output_ids':[42]*count,'ttft_seconds':.001,'e2e_seconds':.002,
                'meta_info':{'prompt_tokens':len(tokens),'completion_tokens':count,'cached_tokens':max(0,len(tokens)-32)}}
    async def telemetry(session,urls,stop,rows,began):await stop.wait()
    monkeypatch.setattr(bench,'flush',flush)
    monkeypatch.setattr(bench,'stream',stream)
    monkeypatch.setattr(bench,'telemetry',telemetry)
    calls=[]
    original=bench.DualMapAdapter.dispatch
    async def counted(self):
        calls.append(1)
        return await original(self)
    monkeypatch.setattr(bench.DualMapAdapter,'dispatch',counted)
    args=SimpleNamespace(engines=['e0','e1'],dualmap_source=Path(source),cache_tokens=4096,
                         snapshot_slots=32,slo=1,tool_seconds=0,suffix_tokens=32,output_tokens=4,
                         seed=1,keys=8,trial_timeout=10)
    calibration={'prefill_seconds_per_token':.0001,'plan':build_plan({str(i):.1 for i in range(8)},2,1,16)}
    summary,rows,_=asyncio.run(bench.run_trial(args,None,list(range(128)),{},None,calibration,32,20,'all-hot',policy,0))
    assert summary['successful']==summary['offered_total']==40
    assert sorted(r['request_id'] for r in rows)==list(range(40))
    assert len(calls)<=160
    assert summary['total_transfer_bytes']==0
    if policy=='sticky':
        assert {r['target'] for r in rows}=={0}
    assert all(r['ttft_seconds']>=r['queue_and_preparation_seconds'] for r in rows)


@pytest.mark.parametrize('policy',['dualmap','dualmap+precompute'])
@pytest.mark.parametrize('withdraw',[False,True])
def test_naturally_warmed_target_is_observed_by_baseline(monkeypatch,policy,withdraw):
    source=os.environ.get('FLUIDKV_DUALMAP_SOURCE')
    if not source:pytest.skip('pinned DualMap checkout required')
    bank={'e0':False,'e1':False}
    flush_calls=[]
    async def flush(*args):
        flush_calls.append(1)
        bank.update(e0=False,e1=False)
    async def stream(session,url,tokens,count,on_first_token=None):
        old=bank[url]
        await asyncio.sleep(.002)
        bank[url]=True
        payload={'meta_info':{'cached_tokens':32 if old else 0}}
        if on_first_token:await on_first_token(payload,.002)
        await asyncio.sleep(.002)
        return {'output_ids':[42]*count,'ttft_seconds':.002,'e2e_seconds':.004,
                'meta_info':{'prompt_tokens':len(tokens),'completion_tokens':count,'cached_tokens':32 if old else 0}}
    async def monitor(session,urls,stop,rows,began):await stop.wait()
    class Response:
        def __init__(self,url):self.url=url
        async def __aenter__(self):return self
        async def __aexit__(self,*args):pass
        def raise_for_status(self):pass
        async def json(self):
            full=bank[self.url.split('/')[0]]
            return {'success':True,'query_tokens':32,'token_count':32,'matched_tokens':32 if full else 0,
                    'snapshot_boundaries':[32] if full else [],'full_prefix_available':full,
                    'available_token_slots':4096,'evictable_token_slots':0,
                    'available_snapshot_slots':32,'evictable_snapshot_slots':0}
    session=SimpleNamespace(post=lambda url,json:Response(url))
    monkeypatch.setattr(bench,'flush',flush);monkeypatch.setattr(bench,'stream',stream);monkeypatch.setattr(bench,'telemetry',monitor)
    args=SimpleNamespace(engines=['e0','e1'],dualmap_source=Path(source),cache_tokens=4096,
                         snapshot_slots=32,slo=.01,tool_seconds=0,suffix_tokens=32,output_tokens=4,
                         seed=1,keys=8,trial_timeout=10,observe_checkpoints=True,withdraw_source=withdraw)
    rates={str(i):.1 for i in range(8)}
    calibration={'prefill_seconds_per_token':.0001,'plan':build_plan(rates,2,1,16),'training_rates':rates,'qcap':1}
    summary,rows,_=asyncio.run(bench.run_trial(args,session,list(range(128)),{},None,calibration,32,20,'single-hot',policy,0))
    expected=20 if withdraw else 40
    assert summary['successful']==expected
    if withdraw:
        assert summary['eligible_engine_urls']==['e1']
        assert all(r['engine_url']=='e1' for r in rows)
    assert any(o['target']==1 and o['phase']=='initial' and not o['result']['full_prefix_available'] for o in summary['checkpoint_observations'])
    phase='prepared' if policy.endswith('+precompute') else 'serving'
    assert any(o['engine_url']=='e1' and o['phase']==phase and o['result']['full_prefix_available'] for o in summary['checkpoint_observations'])
    if policy.endswith('+precompute'):
        assert len(summary['preparation'])==1
        assert summary['preparation'][0]['kind']=='precompute'
        assert summary['total_transfer_bytes']==0
    args.retain_cache=True
    later,_,_=asyncio.run(bench.run_trial(args,session,list(range(128)),{},None,calibration,32,20,'single-hot',policy,0))
    assert len(flush_calls)==1
    assert later['successful']==expected
    assert later['preparation']==[]
    assert all(o['result']['full_prefix_available'] for o in later['checkpoint_observations'])
