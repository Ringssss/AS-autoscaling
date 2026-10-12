"""Exercise original policy execution and lifecycle through the serving adapter."""
import asyncio
from pathlib import Path

import numpy as np
import pytest

from experiments.fluidkv.p08_upstream_routers import UpstreamRouter, BLITZ_BINARY


@pytest.mark.parametrize('policy', ['blitz', 'dualmap'])
def test_confirmed_cache_admission_completion_and_abort(policy):
    if policy=='blitz' and not BLITZ_BINARY.exists():
        pytest.skip('build pinned Blitz bridge first')
    if policy=='dualmap' and not Path('/home/zhujianian/DualMap').exists():
        pytest.skip('pinned DualMap checkout unavailable')

    async def run():
        prefix=np.arange(32768,dtype=np.int32)
        ids=np.arange(33792,dtype=np.int32)
        r=UpstreamRouter(policy,dict(prefill_points=[[32768,4.]]),[[prefix],[]],None)
        try:
            target,decision=await r.choose('a',ids,8,dict(chat_id='a',hash_ids=[1,2],turn=1))
            assert target==0
            assert r.hit(1,ids)==0
            await r.progress(target,'a',{'meta_info':{'cached_tokens':32768}},1,0)
            await r.completed(target,'a',ids,dict(error=None,output_ids=[7]*8))
            assert r.hit(0,ids)==len(ids)
            target,_=await r.choose('b',ids+10,8,dict(chat_id='b',hash_ids=[2,3],turn=1))
            await r.completed(target,'b',ids+10,dict(error='cancelled'))
            assert not any(r.active)
            assert r.pending==[0.,0.]
            if policy=='dualmap':
                assert not r.dualmap.queued()
                for rep in r.dualmap.all_replicas.values():
                    assert not rep.pending_requests and not rep.running_requests
                    assert rep.current_budget==rep.replica_slo_budget
        finally:
            await r.close()
            await r.close()
    asyncio.run(run())


def test_dualmap_preparation_is_accounted_and_released():
    async def run():
        r=UpstreamRouter('dualmap',dict(prefill_points=[[32768,4.]]),[[],[]],None)
        ids=list(range(32768))
        try:
            await r.admit_preparation(ids)
            rep=r.dualmap.replica_for_engine(1)
            assert rep.pending_requests and rep.current_budget < rep.replica_slo_budget
            assert r.hit(1,np.asarray(ids))==0
            await r.progress(1,'prepare',{'meta_info':{'cached_tokens':0}},1,0)
            await r.completed(1,'prepare',ids,dict(error=None,output_ids=[7]*8))
            assert rep.current_budget==rep.replica_slo_budget
            assert not rep.pending_requests and not rep.running_requests
            assert r.hit(1,np.asarray(ids))==len(ids)
        finally:
            await r.close()
    asyncio.run(run())


def test_verified_family_prefix_is_published_at_first_token_not_admission():
    async def run():
        r=UpstreamRouter('dualmap',dict(prefill_points=[[32768,4.]]),[[],[]],None,
            publish_prefill_prefix=32768)
        ids=np.arange(32832,dtype=np.int32)
        try:
            target,_=await r.choose('consumer',ids,8,dict(chat_id='x',hash_ids=[1,2],turn=1))
            assert r.hit(target,ids)==0
            await r.progress(target,'consumer',{'meta_info':{'cached_tokens':0}},1,0)
            assert r.hit(target,ids)==32768
            assert r.active[target]['consumer']['first']
            await r.completed(target,'consumer',ids,dict(error=None,output_ids=[7]*8))
        finally:
            await r.close()
    asyncio.run(run())


def test_warm_second_turn_burst_does_not_route_cold_due_to_512_rounding():
    async def run():
        cached=np.arange(65600,dtype=np.int32)
        request=np.arange(65664,dtype=np.int32)
        r=UpstreamRouter('blitz',dict(prefill_points=[[65536,7.]]),[[cached],[]],None)
        try:
            decisions=[await r.choose(str(i),request,128,
                dict(chat_id=str(i),hash_ids=[1,2],turn=2)) for i in range(32)]
            assert [target for target,_ in decisions]==[0]*32
            assert r.pending==[2048,0]
        finally:
            await r.close()
    asyncio.run(run())


@pytest.mark.parametrize('policy', ['blitz', 'dualmap'])
def test_six_worker_parent_history_and_parallel_preparation(policy):
    async def run():
        ids=np.arange(65536,dtype=np.int32)
        r=UpstreamRouter(policy,dict(prefill_points=[[65536,8.]]),[[] for _ in range(6)],None,
            publish_prefill_prefix=65536)
        try:
            source,_=await r.choose('parent',ids,8,dict(chat_id='parent',hash_ids=[0,1],turn=0))
            await r.progress(source,'parent',{'meta_info':{'cached_tokens':0}},1,0)
            await r.completed(source,'parent',ids,dict(error=None,output_ids=[7]*8))
            for target in range(6):
                if target!=source:
                    await r.admit_preparation(ids.tolist(),target,f'prepare-{target}')
                    assert r.hit(target,ids)==0
            assert sum(len(x) for x in r.active)==5
            for target in range(6):
                if target!=source:
                    rid=f'prepare-{target}'
                    r.step(target,{'scheduled_requests':{'sum_prefill_tokens':65536}})
                    await r.progress(target,rid,{'meta_info':{'cached_tokens':0}},1,0)
                    await r.completed(target,rid,ids,dict(error=None,output_ids=[7]*8))
            assert not any(r.active)
            assert r.pending==[0]*6
            assert [r.hit(i,ids) for i in range(6)]==[65536]*6
        finally:
            await r.close()
    asyncio.run(run())


def test_memoized_observation_tracks_each_workers_confirmed_boundaries():
    from experiments.fluidkv.p10_cached_observations import CachedObservationRouter

    async def run():
        parent=np.arange(65536,dtype=np.int32)
        private=np.arange(65600,dtype=np.int32)
        r=CachedObservationRouter('dualmap',dict(prefill_points=[[65536,7.]]),
            [[parent],[],[],[],[],[]],None)
        ids=list(range(65664))
        try:
            for _ in range(3):
                assert [r.observed_hit(w,ids) for w in range(6)]==[65536,0,0,0,0,0]
            r.cache[0].append(private)
            r.cache[5].append(parent)
            assert [r.observed_hit(w,ids) for w in range(6)]==[65600,0,0,0,0,65536]
            other=list(ids)
            other[65536:]=[7]*128
            assert r.observed_hit(0,other)==65536
            assert r.observed_hit(0,ids)==65600
        finally:
            await r.close()
    asyncio.run(run())
