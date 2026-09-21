"""CPU-only reproduction using pinned Req and scheduler drain methods, no engine import."""
import argparse
import ast
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import torch


def reproduce(source, requested, overlap, eos_at=None, stop_at=None):
    spec = importlib.util.spec_from_file_location('freetoken_length_audit_core', source / 'python/freetoken/core.py')
    core = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = core
    spec.loader.exec_module(core)
    tree = ast.parse((source / 'python/freetoken/scheduler/scheduler.py').read_text())
    scheduler = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Scheduler')
    method = next(n for n in scheduler.body if isinstance(n, ast.FunctionDef) and n.name == '_process_last_data')
    namespace = {'ChunkedReq': type('ChunkedReq', (), {}), 'DetokenizeMsg': NS}
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), method], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source / 'python/freetoken/scheduler/scheduler.py'), 'exec'), namespace)
    emitted = []
    req = core.Req(torch.tensor([1, 2, 3], dtype=torch.int32), 0, 0, requested, 1,
                   core.SamplingParams(ignore_eos=eos_at is None, max_tokens=requested, stop_strs=['STOP'] if stop_at else []), None)
    active = True
    freed_at_host_counts = []

    def remove_req(_req):
        nonlocal active
        active = False

    runner = NS(cache_manager=NS(lazy_free_region=nullcontext, cache_req=lambda *a, **k: None),
                decode_manager=NS(remove_req=remove_req, running_reqs=set()),
                prefill_manager=NS(pending_list=[]), finished_reqs=set(), eos_token_ids={99} if eos_at is not None else set(),
                toolcall_anchor_id=None, _free_req_resources=lambda r: freed_at_host_counts.append(r.input_ids.numel() - 3),
                _match_stop_str=lambda r: 'STOP' if r.input_ids.numel() - 3 == stop_at else None,
                _kv_usage_pages=lambda: (0, 0), _mamba_slot_usage=lambda: None,
                _swa_token_usage=lambda: None, _gpu_mem_bytes=lambda: 0,
                status_reporter=NS(report_batch=lambda *a, **k: None), config=NS(page_size=1),
                send_result=lambda replies: emitted.extend(replies))
    drain = namespace['_process_last_data']
    pending = None
    forwards = 0
    while (active and req.can_decode) or pending is not None:
        current = None
        if active and req.can_decode:
            req.complete_one()  # engine.forward_batch advances before host drain
            forwards += 1
            batch = NS(reqs=[req], is_prefill=forwards == 1)
            current = (NS(batch=batch), (None, torch.tensor([99 if forwards == eos_at else forwards + 10]), NS(synchronize=lambda: None)))
        if overlap:
            drain(runner, pending)
            pending = current
        else:
            drain(runner, current)
    return {'requested': requested, 'overlap': overlap, 'forward_calls': forwards,
            'emitted_token_replies': len(emitted), 'host_generated_tokens': req.input_ids.numel() - 3,
            'last_finish_reason': emitted[-1].finish_reason,
            'terminal_replies': sum(reply.finished for reply in emitted),
            'freed_at_host_counts': freed_at_host_counts,
            'emitted_token_ids': [reply.next_token for reply in emitted]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [reproduce(args.source, count, overlap) for count in (1, 2, 16, 64) for overlap in (False, True)]
    args.output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
