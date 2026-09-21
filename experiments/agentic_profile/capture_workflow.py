"""One diagnostic real-tool workflow with bounded engine Torch profiling.

Runs the existing workflow harness unchanged. This capture is never an input to
profiler-off performance comparisons. The telemetry baseline is emitted after
worker/tool/tokenizer/inference warmup and immediately before timed arrivals.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import aiohttp


def anchor():
    return {'epoch': time.time(), 'monotonic': time.perf_counter()}


def first_telemetry(path):
    if not path.exists():
        return None
    with path.open() as handle:
        for line in handle:
            if not line.endswith('\n'):
                return None
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get('event') == 'sample':
                return row
    return None


def explicit_already_stopped(text):
    return 'Profiling is not in progress. Call /start_profile first.' in text


async def control(session, engine, route, payload=None):
    record = {'route': route, 'payload': payload, 'requested': anchor()}
    try:
        async with session.post(engine+route, json=payload) as response:
            body = await response.text()
            record.update(status=response.status, body=body, returned=anchor())
            record['success'] = 200 <= response.status < 300
            record['explicit_already_stopped'] = route == '/stop_profile' and explicit_already_stopped(body)
            if not record['success'] and not record['explicit_already_stopped']:
                record['error'] = f'HTTP {response.status}: {body}'
    except Exception as exc:
        record.update(success=False, error=repr(exc), returned=anchor())
    return record


async def run(args):
    os.sched_setaffinity(0, {64, 65, 66, 67})
    args.output.mkdir(parents=True, exist_ok=False)
    workflow = args.output/'workflow'
    trace_dir = (args.output/'torch-profile').resolve()
    trace_dir.mkdir()
    runner = Path('experiments/agentic_profile/run_profile_campaign.py')
    command = [sys.executable, '-m', 'experiments.agentic_profile.run_profile_campaign',
               '--output', str(workflow), '--engine', args.engine, '--engine-pid', str(args.engine_pid),
               '--tool-manifest', str(args.tool_manifest.resolve()), '--arrival-interval', str(args.arrival_interval),
               '--agents', '16', '--concurrency', '8', '--repetitions', '1', '--policies', 'shared',
               '--tool-passes', '1', '--tool-output-chars', '6000', '--cores', '8', '--tool-workers', '8']
    record = {'scope': __doc__, 'command': command, 'driver_start': anchor(), 'driver_pid': os.getpid(),
              'driver_cpu_affinity': sorted(os.sched_getaffinity(0)), 'controller_cpu_budget_physical': 4,
              'driver_budget_scope': 'Shares existing controller cores64..67; no added worker/inference cores',
              'source_sha256': hashlib.sha256(runner.read_bytes()).hexdigest(), 'success': False,
              'excluded_from_performance_comparisons': True, 'profile_num_steps': args.num_steps,
              'delay_from_telemetry_detection_seconds': args.delay, 'control_calls': []}
    def save():
        (args.output/'capture.json').write_text(json.dumps(record, indent=2)+'\n')
    save()
    with (args.output/'workflow.log').open('wb') as log:
        process = await asyncio.create_subprocess_exec(*command, stdout=log, stderr=asyncio.subprocess.STDOUT)
        record['workload_pid'] = process.pid
        waiter = asyncio.create_task(process.wait())
        telemetry = workflow/'000-shared-c8-r0/telemetry.jsonl'
        deadline = time.perf_counter()+args.ready_timeout
        while not waiter.done():
            sample = first_telemetry(telemetry)
            if sample is not None:
                record.update(telemetry_detected=anchor(), telemetry_baseline=sample, telemetry_path=str(telemetry))
                break
            if time.perf_counter() > deadline:
                record['error'] = 'workflow telemetry readiness timeout; no profile started'
                break
            await asyncio.sleep(.1)
        if 'telemetry_detected' in record:
            await asyncio.sleep(args.delay)
        save()
        timeout = aiohttp.ClientTimeout(total=args.profile_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            if 'telemetry_detected' in record and not waiter.done():
                payload = {'activities': ['CPU', 'GPU'], 'with_stack': False, 'record_shapes': False,
                           'num_steps': args.num_steps, 'output_dir': str(trace_dir), 'profile_prefix': 'heavy-shared-c8'}
                record['profile_request_anchor'] = anchor()
                # num_steps may defer its response until the automatic stop; do
                # not block process completion observation on that response.
                request = asyncio.create_task(control(session, args.engine, '/start_profile', payload))
                await asyncio.wait({request, waiter}, return_when=asyncio.FIRST_COMPLETED)
                if waiter.done() and not request.done():
                    stopped = await control(session, args.engine, '/stop_profile')
                    record['control_calls'].append(stopped)
                record['control_calls'].append(await request)
                record['workload_exit_code'] = await waiter
                # If a completed bounded trace exists the automatic stop ran.
                # Otherwise explicitly stop and require either success or the
                # exact engine already-not-profiling message.
                if not list(trace_dir.glob('*.trace.json*')):
                    record['control_calls'].append(await control(session, args.engine, '/stop_profile'))
            else:
                record.setdefault('error', 'workflow ended before delayed capture could start')
                record['workload_exit_code'] = await waiter
    record['trace_files'] = [{'path': str(p), 'bytes': p.stat().st_size} for p in sorted(trace_dir.glob('*.trace.json*')) if p.stat().st_size]
    record['finished'] = anchor()
    record['success'] = (record['workload_exit_code'] == 0 and bool(record['trace_files']) and 'error' not in record
                         and bool(record['control_calls']) and all('error' not in call for call in record['control_calls']))
    if not record['trace_files']:
        record.setdefault('error', 'no nonempty engine trace artifact after workflow finish')
    save()
    print(json.dumps({'success': record['success'], 'workload_exit_code': record['workload_exit_code'], 'trace_files': record['trace_files'], 'error': record.get('error')}))
    if not record['success']:
        raise RuntimeError('diagnostic capture failed; artifacts retained')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engine', default='http://127.0.0.1:31480')
    parser.add_argument('--engine-pid', type=int, required=True)
    parser.add_argument('--tool-manifest', type=Path, default=Path('experiments/agentic_profile/results/20260915/cpu-large-corpus/ast-manifest.json'))
    parser.add_argument('--arrival-interval', type=float, default=.25)
    parser.add_argument('--delay', type=float, default=2)
    parser.add_argument('--num-steps', type=int, default=100)
    parser.add_argument('--ready-timeout', type=float, default=240)
    parser.add_argument('--profile-timeout', type=float, default=120)
    args = parser.parse_args()
    if not 1 <= args.num_steps <= 150 or args.delay < 0:
        parser.error('num_steps must be1..150 and delay nonnegative')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
