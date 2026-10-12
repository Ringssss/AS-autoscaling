import numpy as np
from experiments.multinode.run_kimi_trace_replay import construct, select_windows


def test_construct_preserves_lengths_and_limits_shared_prefix():
    corpus=list(range(100,200))
    window={'requests':[dict(ordinal=i,input_tokens=n,output_tokens=o) for i,(n,o) in enumerate([(2,1),(6,80),(9,20),(205,90)])]}
    result=construct(window,corpus,4,64)
    assert [len(r['tokens']) for r in result]==[2,6,9,205]
    assert [r['requested_output'] for r in result]==[1,64,20,64]
    assert result[0]['tokens'][0]!=result[1]['tokens'][0]
    assert all(r['tokens'][:4]==corpus[:4] for r in result[1:])
    assert len({r['tokens'][4] for r in result[1:]})==3
    native=construct(window,corpus,4,0)
    assert [r['requested_output'] for r in native]==[1,80,20,90]
    assert [r['input_sha256'] for r in native]==[r['input_sha256'] for r in result]


def test_window_selection_independent_of_latency_and_output_and_uses_past_training():
    arrival=np.concatenate([1000+10*b+np.linspace(.1,9,1+b%7) for b in range(200)])
    data=dict(arrival=arrival,input_tokens=np.full(len(arrival),100),output_tokens=np.ones(len(arrival),int),cached_tokens=np.zeros(len(arrival),int))
    a=select_windows(data)
    data['output_tokens']=np.arange(len(arrival))+1
    b=select_windows(data)
    assert [w['bin'] for w in a]==[w['bin'] for w in b]
    for w in a:
        assert all(0<=r['offset']<10 for r in w['requests'])
        assert [r['offset'] for r in w['requests']]==sorted(r['offset'] for r in w['requests'])
        expected=((arrival>=w['start']-60)&(arrival<w['start'])).sum()
        assert len(w['training_inputs'])==expected


def test_capacity_clipping_does_not_round_three_replicas_up_to_four():
    from experiments.multinode.run_kimi_trace_replay import capacity_plan
    qcap=0.7307621427183835
    assert (qcap*3)/qcap > 3  # Actual failed calibration boundary.
    plan=capacity_plan(5.7,qcap,3)
    assert len(plan.assignments['shared'])==3
    assert plan.slots_used==3
    assert len(capacity_plan(qcap,qcap,3).assignments['shared'])==1


def test_observations_are_coalesced_per_engine_and_refresh_after_completion():
    import asyncio
    from experiments.multinode.run_kimi_trace_replay import coalesced_task
    async def check():
        pending={}; release=asyncio.Event(); started=[]
        async def work(key):
            started.append(key)
            await release.wait()
        first=coalesced_task(pending,0,lambda:work(0))
        assert all(coalesced_task(pending,0,lambda:work(0)) is None for _ in range(20))
        other=coalesced_task(pending,1,lambda:work(1))
        await asyncio.sleep(0)
        assert started==[0,1]
        release.set(); await asyncio.gather(first,other)
        fresh=coalesced_task(pending,0,lambda:work(0))
        assert fresh is not first
        await fresh
        assert started==[0,1,0]
    asyncio.run(check())
