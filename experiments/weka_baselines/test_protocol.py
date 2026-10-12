import asyncio
from pathlib import Path
from experiments.weka_baselines.common import schedule,expand_dualmap
from experiments.weka_baselines.prepare import token_block
from experiments.fluidkv.dualmap_adapter import DualMapAdapter


def test_namespace_and_prefix_identity():
    assert (token_block('conversation/model',5)==token_block('conversation/model',5)).all()
    assert not (token_block('other/model',5)==token_block('conversation/model',5)).all()


def test_all_regimes_preserve_causal_order_and_common_cohort():
    rows=[dict(stream_index=i,turn=j,key=str(i),token_sha256=str(i*9+j)) for i in range(128) for j in range(9)]
    for scenario in ['balanced','hotspot','burst','scaleout']:
        rs=schedule(rows,scenario)
        assert len(rs)==384
        byid={r['ordinal']:r for r in rs}
        assert {r['phase'] for r in rs}=={0,1,2}
        for r in rs:
            if r['predecessor'] is not None:
                prev=byid[r['predecessor']]
                assert prev['key']==r['key'] and prev['turn']+1==r['turn']
                assert prev['offered_seconds']<=r['offered_seconds']
    assert schedule(rows,'balanced')==schedule(rows,'scaleout')


def test_dualmap_expansion_retains_inflight_and_dispatches_queued():
    async def check():
        a=DualMapAdapter(Path('/home/zhujianian/DualMap'),3,4096,20480,1,.0001)
        await a.retire_engine(2)
        for i in range(8):await a.enqueue(i,dict(chat_id=str(i),hash_ids=[i]),[1000+i]*32,2)
        flying=await a.dispatch();targets={r._id:a.physical_target(r) for r in flying}
        await expand_dualmap(a)
        assert a.active_ids==[0,1,2]
        for r in flying:
            assert a.physical_target(r)==targets[r._id]
            await a.first_token(r,.01);await a.finished(r,[2000,2001],True)
        for _ in range(20):
            for r in await a.dispatch():await a.first_token(r,.01);await a.finished(r,[2000,2001],True)
            if not a.queued():break
        assert not a.queued()
        assert all(not r.pending_requests and not r.running_requests for r in a.all_replicas.values())
    asyncio.run(check())
