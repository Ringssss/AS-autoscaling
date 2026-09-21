"""Sequential experiment control only; the core checkpoint helper is independent."""
import functools,hashlib,json,os,time
from pathlib import Path


def digest(saved):
    import torch
    h=hashlib.sha256()
    for name,tensor in [('tokens',saved['tokens']),('kv',saved['kv']),*sorted(saved['states'].items())]:
        h.update(name.encode());h.update(str(tuple(tensor.shape)).encode());h.update(str(tensor.dtype).encode())
        h.update(memoryview(tensor.contiguous().view(torch.uint8).numpy()).cast('B'))
    return h.hexdigest()


def install(cls):
    if getattr(cls,'_fluidkv_elastic_wrapped',False):return
    if not os.environ.get('FLUIDKV_ELASTIC_POLICY') or not os.environ.get('FLUIDKV_ELASTIC_LOG'):
        raise RuntimeError('both policy and hook-log configuration are required')
    original=cls.rebuild_cache
    @functools.wraps(original)
    def wrapped(self,**geometry):
        from freetoken.kvcache.checkpoint_preservation import export_longest_checkpoint,restore_checkpoint
        from freetoken.engine.engine import CacheRebuildRejected
        policy=json.loads(Path(os.environ['FLUIDKV_ELASTIC_POLICY']).read_text())
        mode=policy.get('mode','drop')
        if mode not in ('drop','preserve'):raise CacheRebuildRejected('unknown experimental preservation policy')
        began=time.perf_counter();record={'id':policy.get('id'),'mode':mode,'geometry':geometry,'pid':os.getpid(),'start':began,'epoch':time.time(),'success':False}
        saved=None
        try:
            cm=self.cache_manager
            record['snapshots_before']=[{'length':cm.prefix_cache._path_len(n),'slot':n.mamba_value} for n in cm.prefix_cache._snapshot_nodes()] if cm.is_hybrid else []
            touches_state=any(geometry.get(k) is not None for k in ('num_pages','num_mamba_slots','num_swa_pages'))
            if mode=='preserve' and touches_state:
                try:
                    saved=export_longest_checkpoint(self,host_budget_bytes=int(policy.get('host_budget_bytes',1024**3)),max_chunk_bytes=int(policy.get('max_chunk_bytes',16*1024**2)))
                except ValueError as exc:raise CacheRebuildRejected(str(exc)) from exc
                if saved is not None:
                    record['export']={k:saved[k] for k in ('boundary','bytes','source_slot','max_chunk_bytes','host_budget_bytes','host_margin_bytes','export_seconds')}
                    if policy.get('verify_bytes'):record['export_sha256']=digest(saved)
            start=time.perf_counter();result=original(self,**geometry);record['original_rebuild_seconds']=time.perf_counter()-start
            if mode=='preserve' and touches_state:
                record['restore']=restore_checkpoint(self,saved)
                if saved is not None and policy.get('verify_bytes'):
                    # Release the first payload before roundtrip export; a digest retains identity.
                    del saved;saved=None
                    check=export_longest_checkpoint(self,host_budget_bytes=int(policy.get('host_budget_bytes',1024**3)),max_chunk_bytes=int(policy.get('max_chunk_bytes',16*1024**2)))
                    record['restored_sha256']=digest(check) if check is not None else None
                    record['bytes_equal']=record['export_sha256']==record['restored_sha256']
                    del check
                    if not record['bytes_equal']:
                        cm.rebuild(cm.num_pages,cm.page_table)
                        raise RuntimeError('restored state bytes differ; published cache cleared')
            record['snapshots_after']=[{'length':cm.prefix_cache._path_len(n),'slot':n.mamba_value} for n in cm.prefix_cache._snapshot_nodes()] if cm.is_hybrid else []
            record['success']=True
            return result
        except Exception as exc:record['error']=repr(exc);raise
        finally:
            saved=None;record['finish']=time.perf_counter();record['total_seconds']=record['finish']-began
            with Path(os.environ['FLUIDKV_ELASTIC_LOG']).open('a',buffering=1) as log:log.write(json.dumps(record)+'\n')
    cls.rebuild_cache=wrapped;cls._fluidkv_elastic_wrapped=True
