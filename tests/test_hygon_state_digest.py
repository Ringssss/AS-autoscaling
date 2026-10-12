"""Exercise the overlay's actual digest methods on CPU tensors, without a server."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def control(monkeypatch):
    source=Path(__file__).resolve().parents[1]/'experiments/hygon_runtime/hybrid_control.py'
    parsed=ast.parse(source.read_text())
    cls=next(n for n in parsed.body if isinstance(n,ast.ClassDef) and n.name=='HybridControl')
    env={'torch':torch,'hashlib':hashlib,'json':json,
         'HygonHybridReqOutput':lambda result:SimpleNamespace(result=result),
         'RadixKey':lambda tokens:SimpleNamespace(child_key=lambda page:tokens[0])}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),str(source),'exec'),env)
    full=SimpleNamespace(k_buffer=[torch.arange(80,dtype=torch.float32).reshape(20,1,4).to(torch.bfloat16)],
                         v_buffer=[torch.arange(80,dtype=torch.float32).reshape(20,1,4).to(torch.bfloat16)])
    state=SimpleNamespace(conv=[torch.ones(2,5,4,3,dtype=torch.bfloat16)],temporal=torch.ones(2,5,2,4,4))
    pool=SimpleNamespace(full_kv_pool=full,mamba_pool=SimpleNamespace(mamba_cache=state,available_size=lambda:5))
    scheduler=SimpleNamespace(token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda:pool,available_size=lambda:20),model_config=SimpleNamespace(vocab_size=1000))
    c=env['HybridControl'](scheduler)
    c.match=lambda tokens:SimpleNamespace(device_indices=torch.tensor([2,3,4]),last_device_node=SimpleNamespace(mamba_value=torch.tensor([1])))
    c.validate_runtime=lambda **kwargs:None
    c.observe_prefix=lambda tokens:{'success':True,'query_tokens':len(tokens),'token_count':len(tokens),'matched_tokens':3,
                                   'full_prefix_available':len(tokens)==3,'has_mamba_state':True,'snapshot_boundaries':[3],
                                   'layout':c.layout(len(tokens))}
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    return c,pool


def test_digest_covers_selected_kv_conv_and_recurrent_state_only(control):
    c,pool=control
    payload={'token_ids':[11,12,13],'return_digest':True}
    first=c.inspect(payload)
    pool.full_kv_pool.k_buffer[0][10]+=1
    assert c.inspect(payload)['state_sha256']==first['state_sha256']
    for tensor,index in [(pool.full_kv_pool.k_buffer[0],2),
                         (pool.full_kv_pool.v_buffer[0],3)]:
        before=c.inspect(payload)['state_sha256']
        tensor[index]+=1
        assert c.inspect(payload)['state_sha256']!=before
    for tensor in [pool.mamba_pool.mamba_cache.conv[0],pool.mamba_pool.mamba_cache.temporal]:
        before=c.inspect(payload)['state_sha256']
        tensor[:,1]+=1
        assert c.inspect(payload)['state_sha256']!=before
    assert first['state_bytes']==2*3*4*2 + 2*4*3*2 + 2*2*4*4*4


def test_digest_rejects_incomplete_boundary(control):
    c,_=control
    with pytest.raises(ValueError,match='complete executable prefix'):
        c.inspect({'token_ids':[1,2,3,4],'return_digest':True})


def test_aggregate_digest_detects_rank_one_change(control):
    c,_=control
    req=SimpleNamespace(operation='inspect',payload={'token_ids':[1,2,3],'return_digest':True})
    second_hash='a'*64
    c.gather=lambda value:[None,None] if value is None else [value,dict(value,state_sha256=second_hash)]
    first=c.handle(req).result
    second_hash='b'*64
    second=c.handle(req).result
    assert first['success'] and second['success']
    assert first['state_sha256']!=second['state_sha256']
    assert first['state_bytes']==sum(r['state_bytes'] for r in first['tp_ranks'])


def test_metadata_walk_never_splits_or_touches_cache_and_crosses_tombstones(control):
    c,_=control
    def node(tokens,state=True):
        return SimpleNamespace(key=SimpleNamespace(token_ids=tokens),value=torch.arange(len(tokens)),
                               mamba_value=torch.tensor([1]) if state else None,children={},last_access_time=42)
    root=node([],False);a=node([11,12]);b=node([13,14],False);d=node([15,16])
    root.children[11]=a;a.children[13]=b;b.children[15]=d
    c.s.tree_cache=SimpleNamespace(root_node=root,page_size=1,full_evictable_size=lambda:6,mamba_evictable_size=lambda:2)
    c.match=lambda *args:pytest.fail('metadata inspection must not call mutating match')
    result=type(c).observe_prefix(c,[11,12,13,14,15,16])
    assert result['snapshot_boundaries']==[2,6]
    assert type(c).observe_prefix(c,[11,12,13])['snapshot_boundaries']==[2]
    assert all(x.last_access_time==42 for x in [root,a,b,d])
    assert b.key.token_ids==[13,14]
    b.value=None
    assert type(c).observe_prefix(c,[11,12,13,14,15,16])['snapshot_boundaries']==[2]


def test_busy_metadata_uses_only_tp_common_boundaries_and_minimum_capacity(control):
    c,_=control
    idle_requirements=[]
    c.validate_runtime=lambda **kwargs:idle_requirements.append(kwargs['require_idle'])
    c.observe_prefix=lambda tokens:{'success':True,'query_tokens':4,'token_count':4,
        'snapshot_boundaries':[2,4],'matched_tokens':4,'full_prefix_available':True,'has_mamba_state':True,
        'available_token_slots':20,'evictable_token_slots':10,'available_snapshot_slots':8,'evictable_snapshot_slots':3}
    c.gather=lambda value:[None,None] if value is None else [value,dict(value,snapshot_boundaries=[2],matched_tokens=2,
        full_prefix_available=False,available_token_slots=12,available_snapshot_slots=5)]
    result=c.handle(SimpleNamespace(operation='inspect',payload={'token_ids':[1,2,3,4]})).result
    assert idle_requirements==[False]
    assert result['success'] and result['snapshot_boundaries']==[2]
    assert result['matched_tokens']==2 and not result['full_prefix_available']
    assert result['available_token_slots']==12 and result['available_snapshot_slots']==5
