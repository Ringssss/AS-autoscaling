"""CPU tests of the isolated research helper; no engine or GPU workload."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

SOURCE = Path(__file__).resolve().parents[1] / 'experiments/freetoken_elastic/environment/FreeToken/python'
if not (SOURCE / 'freetoken').exists():
    pytest.skip('isolated FreeToken checkout unavailable', allow_module_level=True)
sys.path.insert(0, str(SOURCE))
from freetoken.kvcache.checkpoint_preservation import export_longest_checkpoint, restore_checkpoint
from freetoken.kvcache.mha_pool import MHAKVCache
import importlib.util
_spec=importlib.util.spec_from_file_location('preservation_test_cache', SOURCE/'freetoken/scheduler/cache.py')
_cache=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cache)
CacheManager=_cache.CacheManager


def make_scheduler():
    class Pool:
        def __init__(self):
            self.conv_states=torch.zeros(1,9,4,3)
            self.recurrent_states=torch.zeros(1,9,4,2,2)
            self.slot_states={}
            self._local_index={0:0}
            self.num_slots=9
            self.reclaim_all_slots()
        def reclaim_all_slots(self): self._free_slots=list(range(1,9))
        @property
        def num_free_slots(self): return len(self._free_slots)
        def alloc(self,n): return [self._free_slots.pop() for _ in range(n)]
        def free(self,slots): self._free_slots.extend([slots] if isinstance(slots,int) else slots)
    pool=Pool()
    pool.slot_states['test'] = torch.arange(9*3, dtype=torch.float32).reshape(1,9,3)
    kv = object.__new__(MHAKVCache)
    kv._kv_buffer = torch.arange(2*1*128*1*2*2, dtype=torch.float32).reshape(2,1,128,1,2,2)
    kv._layer_map = None
    cm = CacheManager(128, 1, torch.zeros(3,128,dtype=torch.int32), 'hybrid_radix', linear_state_pool=pool)
    engine = NS(kv_cache=kv, linear_state_pool=pool, config=NS(model_path='test',tp_info=NS(size=1)))
    return NS(engine=engine, cache_manager=cm,prefill_manager=NS(runnable=False),decode_manager=NS(runnable=False),_last_data=None)


def seed(s, n=64):
    cm,pool=s.cache_manager,s.engine.linear_state_pool
    pages=cm._allocate(n);slot=pool.alloc(1)[0]
    pool.conv_states[:,slot].fill_(2)
    pool.recurrent_states[:,slot].fill_(7)
    cm.prefix_cache.insert(torch.arange(n,dtype=torch.int32),pages,slot)


def test_roundtrip_fresh_indices_all_siblings():
    s=make_scheduler();seed(s)
    saved=export_longest_checkpoint(s,max_chunk_bytes=256)
    cm=s.cache_manager;cm.rebuild(128,cm.page_table)
    cm.free_slots=cm.free_slots.flip(0)
    s.engine.kv_cache._kv_buffer.zero_()
    for t in [s.engine.linear_state_pool.conv_states,s.engine.linear_state_pool.recurrent_states,*s.engine.linear_state_pool.slot_states.values()]: t.zero_()
    result=restore_checkpoint(s,saved)
    m=cm.prefix_cache.match_prefix(saved['tokens'])
    assert result['restored'] and m.cached_len==64
    assert m.kv_indices.tolist()==list(range(127,63,-1))
    assert torch.equal(s.engine.kv_cache._kv_buffer.index_select(2,m.kv_indices.long()),saved['kv'])
    slot=m.mamba_value
    assert torch.equal(s.engine.linear_state_pool.slot_states['test'][:,slot],saved['states']['sibling:test'])
    assert torch.equal(s.engine.linear_state_pool.recurrent_states[:,slot],saved['states']['recurrent'])
    cm.check_integrity()


def test_reject_budget_and_schema_before_allocation():
    s=make_scheduler();seed(s)
    with pytest.raises(ValueError,match='budget'): export_longest_checkpoint(s,host_budget_bytes=1)
    saved=export_longest_checkpoint(s)
    cm=s.cache_manager;cm.rebuild(128,cm.page_table)
    s.engine.config.model_path='different'
    with pytest.raises(ValueError,match='schema'): restore_checkpoint(s,saved)
    assert len(cm.free_slots)==128


def test_partial_publish_failure_cleans_cache(monkeypatch):
    s=make_scheduler();seed(s);saved=export_longest_checkpoint(s)
    cm=s.cache_manager;cm.rebuild(128,cm.page_table)
    original=cm.prefix_cache.insert
    def broken(*args):
        original(*args)
        raise RuntimeError('injected')
    monkeypatch.setattr(cm.prefix_cache,'insert',broken)
    with pytest.raises(RuntimeError,match='injected'):restore_checkpoint(s,saved)
    assert len(cm.free_slots)==128 and s.engine.linear_state_pool.num_free_slots==8
    assert not cm.prefix_cache._snapshot_nodes()
    cm.check_integrity()


def test_idle_gate_and_empty():
    s=make_scheduler();assert export_longest_checkpoint(s) is None
    s._last_data=object()
    with pytest.raises(ValueError,match='drained'):export_longest_checkpoint(s)


def test_reject_free_snapshot_slot_and_other_engine():
    s=make_scheduler();seed(s)
    saved=export_longest_checkpoint(s)
    other=make_scheduler()
    with pytest.raises(ValueError,match='schema'):restore_checkpoint(other,saved)
    node=s.cache_manager.prefix_cache._snapshot_nodes()[0]
    s.engine.linear_state_pool.free(node.mamba_value)
    with pytest.raises(ValueError,match='free state slot'):export_longest_checkpoint(s)


def test_allocation_failure_returns_pages(monkeypatch):
    s=make_scheduler();seed(s);saved=export_longest_checkpoint(s)
    cm=s.cache_manager;cm.rebuild(128,cm.page_table)
    def broken(n): raise RuntimeError('allocation fault')
    monkeypatch.setattr(s.engine.linear_state_pool,'alloc',broken)
    with pytest.raises(RuntimeError,match='allocation fault'):restore_checkpoint(s,saved)
    assert len(cm.free_slots)==128
    cm.check_integrity()


def test_nonempty_restore_rejects_without_mutation():
    s=make_scheduler();seed(s);saved=export_longest_checkpoint(s)
    before=len(s.cache_manager.free_slots)
    with pytest.raises(ValueError,match='empty'):restore_checkpoint(s,saved)
    assert len(s.cache_manager.free_slots)==before
