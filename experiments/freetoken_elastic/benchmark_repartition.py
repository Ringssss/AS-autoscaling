"""Sequential destructive-repartition baseline; diagnostic controls, no preservation claim."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import random
import time

import aiohttp

from experiments.freetoken_elastic.client import FreeTokenClient


CONDITIONS = ('no-resize', 'expert-only', 'kv-resize', 'mamba-resize')


def load_manifest(path):
    document = json.loads(Path(path).read_text())
    pairs = document['pairs']
    if not pairs or len({p['id'] for p in pairs}) != len(pairs):
        raise ValueError('manifest pairs must be nonempty with distinct IDs')
    for pair in pairs:
        for role in ('parent', 'child'):
            if not isinstance(pair[f'{role}_text'], str):
                raise ValueError('manifest requires exact prompt strings')
            count = pair[f'expected_{role}_tokens']
            if not isinstance(count, int) or count < 1:
                raise ValueError('manifest requires positive expected token counts')
    return document


def trial_matrix(pairs, conditions, repetitions, seed):
    rng = random.Random(seed)
    matrix = []
    for repetition in range(repetitions):
        for pair in pairs:
            order = list(conditions)
            rng.shuffle(order)
            for condition in order:
                matrix.append({'index': len(matrix), 'pair_id': pair['id'],
                               'condition': condition, 'repetition': repetition})
    return matrix


def geometry_for(args, condition=None):
    geometry = {'num_pages': args.initial_pages, 'moe_cache_size': args.initial_moe,
                'num_mamba_slots': args.initial_mamba}
    if condition == 'expert-only':
        geometry['moe_cache_size'] = args.expert_only_moe
    elif condition == 'kv-resize':
        geometry['num_pages'] = args.final_pages
    elif condition == 'mamba-resize':
        geometry['num_mamba_slots'] = args.final_mamba
    return geometry


def check_geometry(response, expected):
    if not response['success']:
        raise ValueError(f'cache status failed: {response.get("error")}')
    body = response.get('body') or {}
    if body.get('state') != 'serving':
        raise ValueError(f'engine not serving: {body.get("state")}')
    actual = body.get('geometry', {})
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f'geometry mismatch {key}: expected {value}, got {actual.get(key)}')
    return actual


def check_generation(response, expected_prompt, expected_output, cold=False):
    if not response['success']:
        raise ValueError(f'generation failed: {response.get("error")}')
    usage = response.get('usage') or {}
    if usage.get('prompt_tokens') != expected_prompt or usage.get('completion_tokens') != expected_output:
        raise ValueError(f'token length mismatch: {usage}')
    if response.get('cached_tokens') is None:
        raise ValueError('cache reporting is required')
    if cold and response['cached_tokens'] != 0:
        raise ValueError(f'reset did not produce cold parent: {response["cached_tokens"]} cached tokens')
    if response.get('first_text_seconds') is None:
        raise ValueError('generation emitted no decoded text')


async def run_trial(client, args, pair, spec, output):
    record = {**spec, 'success': False, 'operations': [], 'started_unix': time.time(),
              'control_scope': 'diagnostic; conditions have different final pool geometry',
              'second_child_scope': 'same child repeated; may reuse more than original parent prefix'}
    raw_path = output / f'trial-{spec["index"]:03d}.json'

    def save():
        temporary = raw_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(record, indent=2))
        temporary.replace(raw_path)

    async def operation(label, coroutine):
        response = await coroutine
        record['operations'].append({'label': label, **response})
        save()
        if not response['success']:
            raise ValueError(f'{label} failed: {response.get("error")}')
        return response

    try:
        # The intermediate geometry differs from the desired initial geometry even if the
        # prior run happened to finish there; it is outside the intervention clock.
        initial = geometry_for(args)
        await operation('reset-intermediate', client.rebuild(**{**initial, 'num_pages': args.initial_pages + args.reset_page_delta}))
        await operation('reset-initial', client.rebuild(**initial))
        state = await operation('initial-status', client.status())
        record['initial_geometry'] = check_geometry(state, initial)
        parent = await operation('cold-parent', client.generate(pair['parent_text'], args.parent_output, args.model))
        check_generation(parent, pair['expected_parent_tokens'], args.parent_output, cold=True)
        await operation('before-stats', client.stats())
        before = await operation('before-status', client.status())
        check_geometry(before, initial)
        final = geometry_for(args, spec['condition'])
        start = time.perf_counter()
        record['intervention_started_unix'] = time.time()
        if spec['condition'] != 'no-resize':
            changed = {key: value for key, value in final.items() if value != initial[key]}
            rebuild = await operation('measured-rebuild', client.rebuild(**changed))
            rebuild_seconds = rebuild['elapsed_seconds']
        else:
            rebuild_seconds = 0.0
        child = await operation('first-child', client.generate(pair['child_text'], args.child_output, args.model))
        completed = time.perf_counter()
        check_generation(child, pair['expected_child_tokens'], args.child_output)
        record['metrics'] = {
            'rebuild_seconds': rebuild_seconds,
            'first_child_text_seconds': child['first_text_seconds'],
            'first_child_e2e_seconds': child['elapsed_seconds'],
            'first_child_cached_tokens': child['cached_tokens'],
            'rebuild_plus_first_text_seconds': rebuild_seconds + child['first_text_seconds'],
            'rebuild_plus_child_e2e_seconds': rebuild_seconds + child['elapsed_seconds'],
            'intervention_to_first_text_seconds': child['started_monotonic'] - start + child['first_text_seconds'],
            'intervention_to_completion_seconds': completed - start,
            'client_journal_overhead_included_in_intervention_clock': True,
        }
        after = await operation('after-first-status', client.status())
        record['final_geometry'] = check_geometry(after, final)
        await operation('after-first-stats', client.stats())
        repeated = await operation('repeated-child', client.generate(pair['child_text'], args.child_output, args.model))
        check_generation(repeated, pair['expected_child_tokens'], args.child_output)
        record['metrics'].update(
            repeated_child_cached_tokens=repeated['cached_tokens'],
            repeated_child_first_text_seconds=repeated['first_text_seconds'],
            repeated_child_e2e_seconds=repeated['elapsed_seconds'],
            repeated_child_text_matches_first=child['text'] == repeated['text'])
        await operation('final-status', client.status())
        await operation('final-stats', client.stats())
        record['success'] = True
    except Exception as exc:
        record['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        record['finished_unix'] = time.time()
        save()
    return record


async def run(args):
    manifest = load_manifest(args.manifest)
    if args.repetitions < 1 or args.child_output < 1 or args.reset_page_delta == 0:
        raise ValueError('positive repetitions/output and nonzero reset delta required')
    if 'kv-resize' in args.conditions and args.final_pages == args.initial_pages:
        raise ValueError('kv-resize requires different initial/final pages')
    if 'mamba-resize' in args.conditions and args.final_mamba == args.initial_mamba:
        raise ValueError('mamba-resize requires different initial/final slots')
    if 'expert-only' in args.conditions and args.expert_only_moe == args.initial_moe:
        raise ValueError('expert-only requires different expert cache sizes')
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('output must be absent or empty; never overwrite a previous campaign')
    args.output.mkdir(parents=True, exist_ok=True)
    matrix = trial_matrix(manifest['pairs'], args.conditions, args.repetitions, args.seed)
    campaign = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                'manifest': manifest, 'manifest_sha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                'matrix': matrix, 'runs': [], 'complete': False}
    campaign_path = args.output / 'campaign.json'
    campaign_path.write_text(json.dumps(campaign, indent=2))
    by_id = {p['id']: p for p in manifest['pairs']}
    async with aiohttp.ClientSession() as session:
        client = FreeTokenClient(session, args.engine, timeout=args.timeout,
                                 cache_report_enabled=args.cache_report_enabled)
        for spec in matrix:
            result = await run_trial(client, args, by_id[spec['pair_id']], spec, args.output)
            summary = {key: result.get(key) for key in ('index', 'pair_id', 'condition', 'repetition', 'success', 'error', 'metrics')}
            campaign['runs'].append(summary)
            campaign_path.write_text(json.dumps(campaign, indent=2))
            print(json.dumps(summary), flush=True)
            if not result['success']:
                return 1
    campaign['complete'] = True
    campaign_path.write_text(json.dumps(campaign, indent=2))
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--engine', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--model', default='default')
    p.add_argument('--initial-pages', type=int, default=65536)
    p.add_argument('--final-pages', type=int, default=32768)
    p.add_argument('--initial-moe', type=int, default=1536)
    p.add_argument('--expert-only-moe', type=int, default=1430)
    p.add_argument('--initial-mamba', type=int, default=12)
    p.add_argument('--final-mamba', type=int, default=8)
    p.add_argument('--reset-page-delta', type=int, default=64)
    p.add_argument('--parent-output', type=int, choices=(1, 16), default=1)
    p.add_argument('--child-output', type=int, default=64)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--seed', type=int, default=1516)
    p.add_argument('--timeout', type=float, default=600)
    p.add_argument('--cache-report-enabled', action='store_true', required=True,
                   help='Assert the engine was launched with cache reporting enabled')
    p.add_argument('--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS))
    return p


if __name__ == '__main__':
    raise SystemExit(asyncio.run(run(parser().parse_args())))
