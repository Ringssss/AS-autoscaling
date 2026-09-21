"""Bounded, idle-only backup of aligned checkpoints for the TP=1 hybrid baseline.

The aligned-boundary choice is a retention policy for these prefix fixtures.
Non-aligned final/semantic snapshots can also be valid; they are not selected.
"""
from __future__ import annotations

import time

import torch


def _context(scheduler):
    engine, cm = scheduler.engine, scheduler.cache_manager
    if scheduler.prefill_manager.runnable or scheduler.decode_manager.runnable:
        raise ValueError('checkpoint maintenance requires an idle scheduler')
    if getattr(scheduler, '_last_data', None) is not None:
        raise ValueError('checkpoint maintenance requires all results drained')
    if engine.config.tp_info.size != 1 or cm.page_size != 1 or not cm.is_hybrid:
        raise ValueError('only TP=1 page-size-one hybrid checkpoints are supported')
    from freetoken.kvcache.mha_pool import MHAKVCache
    if type(engine.kv_cache) is not MHAKVCache:
        raise ValueError('checkpoint backup requires the exact MHA pool class')
    pool = engine.linear_state_pool
    if pool is None or cm.linear_state_pool is not pool:
        raise ValueError('inconsistent linear state pool')
    return engine, cm, pool


def _sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def _schema(engine, pool):
    kv = engine.kv_cache
    return (id(engine), str(engine.config.model_path), tuple(kv._kv_buffer.shape[:2]),
            tuple(kv._kv_buffer.shape[3:]), str(kv._kv_buffer.dtype),
            tuple(kv._layer_map) if kv._layer_map is not None else None,
            tuple(sorted(pool._local_index.items())),
            tuple((name, tuple(sorted(mapping.items()))) for name, mapping in
                  sorted(getattr(pool, '_state_layer_index', {}).items())),
            tuple((name, tuple(t.shape[:1]) + tuple(t.shape[2:]), str(t.dtype))
                  for name, t in _states(pool).items()))


def _states(pool):
    return {'conv': pool.conv_states, 'recurrent': pool.recurrent_states,
            **{'sibling:' + k: v for k, v in sorted(pool.slot_states.items())}}


def export_longest_checkpoint(scheduler, host_budget_bytes=1024**3,
                              max_chunk_bytes=16*1024**2):
    engine, cm, pool = _context(scheduler)
    if host_budget_bytes <= 0 or max_chunk_bytes <= 0:
        raise ValueError('budgets must be positive')
    nodes = [node for node in cm.prefix_cache._snapshot_nodes()
             if cm.prefix_cache._path_len(node) % 64 == 0]
    if not nodes:
        return None
    node = max(nodes, key=cm.prefix_cache._path_len)
    boundary = cm.prefix_cache._path_len(node)
    if boundary % 64:
        raise ValueError('checkpoint boundary must be a multiple of 64')
    if node.mamba_value in pool._free_slots or not 0 < node.mamba_value < pool.num_slots:
        raise ValueError('checkpoint references an invalid or free state slot')
    if any(t[:, 0][0].numel()*t.element_size() > max_chunk_bytes
           for t in _states(pool).values()):
        raise ValueError('state layer exceeds staging budget')
    parts, cur = [], node
    while not cur.is_root():
        parts.append(cur._key)
        cur = cur.parent
    tokens = torch.cat(parts[::-1]).cpu().clone()
    indices = cm.prefix_cache._collect_kv(node).long()
    buf = engine.kv_cache._kv_buffer
    if (indices.numel() != boundary or indices.unique().numel() != boundary
            or int(indices.min()) < 0 or int(indices.max()) >= cm.num_pages):
        raise ValueError('invalid checkpoint KV path')
    per_page = buf[:, :, 0].numel() * buf.element_size()
    total = per_page * boundary + sum(t[:, 0].numel()*t.element_size()
                                     for t in _states(pool).values())
    total += tokens.numel()*tokens.element_size()
    # Metadata and transient token assembly are charged separately from payload.
    host_margin = 1024**2 + 2*tokens.numel()*tokens.element_size() + 2*max_chunk_bytes
    if total + host_margin > host_budget_bytes or per_page > max_chunk_bytes:
        raise ValueError('checkpoint exceeds staging budget')
    cm.prefix_cache.inc_lock(node)
    began = time.perf_counter()
    try:
        _sync(buf.device)
        host = torch.empty((buf.shape[0], buf.shape[1], boundary, *buf.shape[3:]),
                           dtype=buf.dtype, device='cpu', pin_memory=buf.is_cuda)
        step = max_chunk_bytes // per_page
        for start in range(0, boundary, step):
            end = min(start+step, boundary)
            chunk = buf.index_select(2, indices[start:end])
            host[:, :, start:end].copy_(chunk)
            del chunk
        states = {}
        for name, tensor in _states(pool).items():
            view = tensor[:, node.mamba_value]
            out = torch.empty(view.shape, dtype=view.dtype, device='cpu',
                              pin_memory=view.is_cuda)
            for layer in range(view.shape[0]):
                out[layer].copy_(view[layer])
            states[name] = out
        _sync(buf.device)
        return {'schema': _schema(engine, pool), 'tokens': tokens, 'kv': host,
                'states': states, 'bytes': total, 'boundary': boundary,
                'source_slot': node.mamba_value, 'max_chunk_bytes': max_chunk_bytes,
                'host_budget_bytes': host_budget_bytes, 'host_margin_bytes': host_margin,
                'export_seconds': time.perf_counter()-began}
    finally:
        cm.prefix_cache.dec_lock(node)


def restore_checkpoint(scheduler, saved):
    engine, cm, pool = _context(scheduler)
    if saved is None:
        return {'restored': False, 'reason': 'no checkpoint'}
    if saved['schema'] != _schema(engine, pool):
        raise ValueError('checkpoint schema mismatch')
    if cm.prefix_cache.size_info.total_size or cm.prefix_cache._snapshot_nodes():
        raise ValueError('restore requires an empty rebuilt cache')
    n = saved['boundary']
    if n <= 0 or n % 64 or saved['max_chunk_bytes'] <= 0:
        raise ValueError('invalid checkpoint boundary or chunk budget')
    if n >= cm.num_pages or pool.num_free_slots < 4:
        raise ValueError('insufficient checkpoint and continuation capacity')
    buf = engine.kv_cache._kv_buffer
    expected = (buf.shape[0], buf.shape[1], n, *buf.shape[3:])
    if (tuple(saved['kv'].shape) != expected or saved['kv'].dtype != buf.dtype
            or saved['tokens'].numel() != n or saved['tokens'].dtype != torch.int32
            or saved['tokens'].ndim != 1 or not saved['tokens'].is_cpu
            or not saved['kv'].is_cpu
            or set(saved['states']) != set(_states(pool))):
        raise ValueError('invalid checkpoint payload')
    for name, tensor in _states(pool).items():
        host = saved['states'][name]
        if (host.shape != tensor[:, 0].shape or host.dtype != tensor.dtype
                or not host.is_cpu
                or host[0].numel()*host.element_size() > saved['max_chunk_bytes']):
            raise ValueError('invalid state payload')
    actual_bytes = saved['kv'].numel()*saved['kv'].element_size()
    actual_bytes += saved['tokens'].numel()*saved['tokens'].element_size()
    actual_bytes += sum(t.numel()*t.element_size() for t in saved['states'].values())
    if actual_bytes != saved['bytes']:
        raise ValueError('checkpoint byte accounting mismatch')
    began = time.perf_counter()
    pages = slot = None
    published = False
    try:
        pages = cm._allocate(n)
        slot = pool.alloc(1)[0]
        per_page = buf[:, :, 0].numel()*buf.element_size()
        step = max(1, saved['max_chunk_bytes']//per_page)
        for start in range(0, n, step):
            end = min(start+step, n)
            chunk = saved['kv'][:, :, start:end].to(buf.device)
            buf.index_copy_(2, pages[start:end].long(), chunk)
            del chunk
        for name, tensor in _states(pool).items():
            for layer in range(tensor.shape[0]):
                tensor[layer, slot].copy_(saved['states'][name][layer])
        _sync(buf.device)
        published = True
        matched, exists = cm.prefix_cache.insert(saved['tokens'], pages, slot)
        if matched or exists:
            raise RuntimeError('unexpected checkpoint deduplication')
        cm.check_integrity()
        if pool.num_free_slots + len(cm.prefix_cache._snapshot_nodes()) != pool.num_slots-1:
            raise RuntimeError('snapshot slot conservation failure')
        return {'restored': True, 'boundary': n, 'bytes': saved['bytes'],
                'source_slot': saved['source_slot'], 'destination_slot': slot,
                'restore_seconds': time.perf_counter()-began}
    except Exception:
        _sync(buf.device)
        if published:
            cm.rebuild(cm.num_pages, cm.page_table)
        else:
            if pages is not None:
                cm._free(pages)
            if slot is not None:
                pool.free(slot)
        raise
