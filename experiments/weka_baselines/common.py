"""Shared experiment semantics and persistent upstream router membership."""
import hashlib
import math
import random
from experiments.multinode.cacheroute import build_plan

POLICIES=['sticky','p2c','p2c+move','p2c+precompute','dualmap','dualmap+move',
          'dualmap+precompute','cacheroute','cacheroute+move','cacheroute+precompute']
SCENARIOS=['balanced','hotspot','burst','scaleout']


def schedule(rows,scenario,streams=128,period=60.):
    lookup={(r['stream_index'],r['turn']):r for r in rows}
    result=[]
    # Fixed hot identities, common across all policies. Selection uses no
    # future latency, cache placement, or outcome of a policy run.
    hot=sorted(range(streams),key=lambda i:hashlib.sha256(f'hot:{i}'.encode()).digest())[:max(1,streams//4)]
    for phase in range(3):
        pairs=[(i,phase) for i in range(streams)]
        if scenario=='hotspot' and phase>0:
            pairs=[(i,1+(phase-1)*4+j) for j in range(4) for i in hot]
        rng=random.Random(734+phase);rng.shuffle(pairs)
        # Causal order is preserved inside every stream even after shuffling.
        pairs.sort(key=lambda p:p[1] if scenario=='hotspot' and phase>0 else 0)
        for rank,(i,turn) in enumerate(pairs):
            spread=0 if scenario=='burst' else period*.5*rank/max(1,len(pairs))
            r=dict(lookup[i,turn],ordinal=len(result),phase=phase,offered_seconds=phase*period+spread)
            result.append(r)
    # Record predecessor explicitly, not inferred from completion order.
    prior={}
    for r in sorted(result,key=lambda r:(r['phase'],r['turn'],r['stream_index'])):
        r['predecessor']=prior.get(r['key']);prior[r['key']]=r['ordinal']
    return sorted(result,key=lambda r:(r['offered_seconds'],r['ordinal']))


def rate_plan(history,replicas,qcap,window=60):
    counts={}
    for key in history:counts[key]=counts.get(key,0)+1
    # Oversubscribed keys use all replicas; record this input saturation in
    # the harness. The underlying build_plan/dispatch algorithm is unchanged.
    rates={k:min(v/window,math.nextafter(qcap*replicas,0)) for k,v in counts.items()}
    return build_plan(rates,replicas,qcap,replicas*128)


async def expand_dualmap(adapter,physical=2):
    """Reverse initial logical removal; retain metadata and in-flight identities."""
    from dualmap.scheduler.utils.double_hash_global_scheduler_utils import GlobalRequestQueue
    import uhashring
    waiting=[(engine,r) for slot,engine in enumerate(adapter.engine_map)
             for r in adapter.scheduler.global_request_queue.get_all_requests(slot)]
    adapter.active_ids.append(physical);adapter.active_ids.sort()
    adapter.engine_map=list(adapter.active_ids)
    adapter.replica_budgets={slot:adapter.all_replicas[e] for slot,e in enumerate(adapter.engine_map)}
    adapter.scheduler._num_replicas=len(adapter.engine_map)
    adapter.scheduler.global_request_queue=GlobalRequestQueue(len(adapter.engine_map))
    ring1=uhashring.HashRing(nodes=[str(e) for e in adapter.engine_map])
    ring2=uhashring.HashRing(nodes=[str(e) for e in adapter.engine_map],hash_fn='ketama')
    adapter.scheduler.hash_function1=lambda key,count:adapter.engine_map.index(int(ring1.get_node(str(key))))
    adapter.scheduler.hash_function2=lambda key,count:adapter.engine_map.index(int(ring2.get_node(str(key))))
    for _,r in waiting:await adapter.scheduler.add_request_to_best_global_queue(r)
