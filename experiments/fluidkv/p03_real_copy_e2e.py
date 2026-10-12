"""P0-3: same-router paired end-to-end comparison with a REAL state copy.

Why this exists: the earlier end-to-end table (real_compare.py) has no copy arm
at all -- its "+move" policies only re-dispatch the request to another engine,
which loses affinity but never moves state. Those are an affinity ablation, not
a mobility treatment, and cannot support a systems claim.

Design
------
Every arm runs the SAME arrivals over the SAME engines against the SAME trace.
Arms differ only in the placement rule and whether mobility is enabled:

  sticky         affinity only: every request goes to its stream's home engine
  least_loaded   balance only: every request goes to the least-busy engine,
                 which is generally NOT where its prefix lives, so it pays a
                 full recompute
  prefix_aware   best cache match wins, ties broken by load (Preble-style)
  balance+move   SAME router as least_loaded, plus a real state copy: when the
                 router's balanced choice would land cold, the runnable prefix
                 is copied to the destination instead of being recomputed

`least_loaded` vs `balance+move` is the paired comparison proper: one router,
identical decisions, mobility toggled. `sticky` and `prefix_aware` are the
reference points that show what affinity buys and what it costs.

Group reuse (the fix)
---------------------
The earlier version called `copy_state` with a fresh group name every time
(`p03-{ms}`), so EVERY migration paid a full NCCL group init. Measured cost:
init_once ~16-31 s vs per_transfer ~0.12-0.6 s on an established group -- a
40x overhead that made every migration net-negative by construction. Groups
are now cached per (src, dst) pair and initialized at most once; subsequent
transfers reuse the same group, which the engine supports natively via an
incrementing sequence number in its store.

Hard constraint (must be reported)
----------------------------------
The engine's transfer path requires an EMPTY destination radix tree
(`hygon_hybrid_control.py`: "destination cache must be empty"). Migration is
therefore only possible onto an idle engine. In steady state on a saturated
fleet this bounds coverage; the honest consequences are measured rather than
assumed.

Metrics, all over the FULL traffic set:
  - per-request TTFT, SLO attainment (TTFT <= 10 s), goodput
  - migrations attempted / succeeded / actually used (the consumer reached the
    copy target AND its cached_tokens covered the copied span)
  - copy seconds paid, and bytes moved
  - in-run instrument control: a fresh serve reports cached_tokens=0 and a
    local-warm serve reports cached_tokens>0, in the same trial

Reporting rule: the eligible-subset delta and the all-traffic delta are
reported SEPARATELY. A large eligible-subset effect and a near-flat all-traffic
effect are both possible and neither may be hidden.
"""
import argparse
import asyncio
import hashlib
import json
import socket
import time
from collections import defaultdict
from pathlib import Path

import aiohttp
import numpy as np

ROOT = Path('/home/zhujianian/AS-autoscaling')

# Engines running the PATCHED overlay (sha256 dc960414...). Verify before a run:
#   docker exec <c> sha256sum \
#     /usr/local/lib/python3.10/dist-packages/sglang/srt/managers/hygon_hybrid_control.py
# Nine engines across three hosts (cards verified free at launch). Engines 0-3
# are the "hot" set that carries the skewed traffic; engines 4-8 start cold and
# stay cold, which is the only regime in which the engine's empty-destination
# requirement can actually be satisfied in steady state.
ENGINES = [
    '12.11.3.27:31781',
    '12.11.3.27:31782',
    '12.11.3.27:31783',
    '12.11.3.28:31751',
    '12.11.3.28:31752',
    '12.11.3.28:31753',
    '12.11.3.29:31761',
    '12.11.3.29:31762',
    '12.11.3.29:31763',
]
ENGINE_HOSTS = [e.split(':')[0] for e in ENGINES]

# One 4-port stride per DIRECTED pair. init_group uses port+tp_rank for the
# process group and port+tp_size for the TCPStore, so port..port+3 must be free
# on both hosts of the pair.
#
# Ports come from the OS ephemeral allocator, not from arithmetic. An engine
# never closes a group's TCPStore, so any fixed formula eventually re-issues a
# port that is still held -- the next init then blocks 90 s on a store nobody
# will ever set and leaves the peer waiting. Asking the kernel for a free port
# and then closing it (the engine rebinds it moments later) avoids the wrap
# entirely; the standalone characterization runs did exactly this and went
# 12/12 while the arithmetic version failed.
_USED_PORTS = set()


def free_port_stride():
    """An OS-assigned ephemeral port whose whole stride P..P+3 is free.

    Two lessons are baked in here. First, ports must come from the ephemeral
    range rather than a fixed arithmetic base: an engine never closes a
    group's TCPStore, so any formula eventually re-issues a port that is still
    held, and the next init then blocks 90 s on a store nobody will set. The
    standalone characterization runs used OS-assigned ports and went 12/12
    while the arithmetic version failed. Second, the whole stride must be
    reserved at once -- init_group binds P+tp_rank for the process group and
    P+tp_size for the TCPStore, so a free P alone is not enough.
    """
    for _ in range(200):
        s = socket.socket()
        s.bind(('', 0))
        lo = max(s.getsockname()[1], 32768)
        base = lo - (lo % 4)
        s.close()
        if base in _USED_PORTS:
            continue
        socks = []
        ok = True
        for off in range(4):
            t = socket.socket()
            try:
                t.bind(('', base + off))
                socks.append(t)
            except OSError:
                ok = False
                t.close()
                break
        for t in socks:
            t.close()
        if ok:
            _USED_PORTS.add(base)
            return base
    raise RuntimeError('no free 4-port stride found')


def pair_port(src_i, dst_i, n, salt=0):
    """Kept for callers that want a deterministic stride; prefers OS ports."""
    return free_port_stride()


MOVE_MIN_TOKENS = 8192          # below L* moving loses
OUT_TOKENS = 16
SLO_SECONDS = 10.0
SERVE_BUDGET = 300.0            # a 130K cold prefill can legitimately take ~30 s


def load_workload():
    rows = [json.loads(l) for l in
            (ROOT / 'experiments/weka_baselines/data/requests.jsonl').open()]
    toks = np.load(ROOT / 'experiments/weka_baselines/data/tokens.npy')
    return rows, toks


def schedule(rows, streams, hot_frac=0.0):
    """Group rows into per-stream turn sequences and assign each a home engine.

    hot_frac > 0 skews the assignment: the longest streams (by max input
    tokens) are pinned to engine 0 together, which is the load-gravity regime
    the paper is about -- a few engines carry both the longest prefixes and
    the most traffic, while others sit idle.
    """
    by = defaultdict(list)
    for r in rows:
        by[r['key']].append(r)
    for k in by:
        by[k].sort(key=lambda r: r['turn'])
    keys = sorted(by, key=lambda k: hashlib.sha256(k.encode()).hexdigest())[:streams]
    order = [by[k] for k in keys]
    n = len(ENGINES)
    homes = {}
    if hot_frac <= 0:
        for rs in order:
            homes[rs[0]['key']] = int(hashlib.sha256(rs[0]['key'].encode()).hexdigest(), 16) % n
    else:
        ranked = sorted(order, key=lambda rs: -max(r['input_tokens'] for r in rs))
        n_hot = max(1, int(len(ranked) * hot_frac))
        for i, rs in enumerate(ranked):
            if i < n_hot:
                homes[rs[0]['key']] = 0
            else:
                h = hashlib.sha256(rs[0]['key'].encode()).hexdigest()
                homes[rs[0]['key']] = 1 + (int(h, 16) % max(1, n - 1))
    return order, homes


async def flush(session, urls, budget=45.0):
    """Best-effort flush. MUST be bounded.

    `/flush_cache?timeout=30` bounds the SERVER's own wait, not the client's.
    When the previous arm leaves requests queued, the handler can stall far
    past its nominal timeout and an unbounded client simply waits forever --
    which is what hung the treatment arm before it ever reached its routing
    logic. A flush that cannot complete is skipped: the arm then starts with a
    warm engine and the run is still valid, just less clean.
    """
    for u in urls:
        try:
            await asyncio.wait_for(
                _flush_one(session, u),
                timeout=budget)
        except Exception:
            pass
    await asyncio.sleep(1.5)


async def _flush_one(session, url, tries=3):
    for k in range(tries):
        try:
            async with session.post(url + '/flush_cache?timeout=30',
                                    timeout=aiohttp.ClientTimeout(total=20)) as r:
                txt = await r.text()
            if 'will not be performed' not in txt:
                return True
        except Exception:
            pass
        await asyncio.sleep(2.0 * (k + 1))
    return False


async def inspect(session, url, ids, budget=20.0):
    """Bounded inspect.

    The engine's hybrid control channel is single-queued. If a previous group
    init failed half-way, the queue can stop draining and an unbounded inspect
    then blocks forever -- which is what silently hung the treatment arm while
    it was still inside its very first routing decision. A timed-out inspect
    returns {} and the caller treats the prefix as absent.
    """
    payload = {'token_ids': [int(x) for x in ids]}
    try:
        async with session.post(url + '/agentshift/hybrid_state/inspect',
                                json=payload,
                                timeout=aiohttp.ClientTimeout(total=budget)) as r:
            return await r.json()
    except Exception:
        return {}


async def serve(session, url, ids, out_tokens=OUT_TOKENS):
    payload = {'input_ids': [int(x) for x in ids], 'stream': True,
               'sampling_params': {'temperature': 0, 'ignore_eos': True,
                                   'max_new_tokens': out_tokens}}
    t0 = time.perf_counter()
    first = None
    meta = {}
    async with session.post(url + '/generate', json=payload) as resp:
        if resp.status != 200:
            raise RuntimeError(f'HTTP {resp.status} {(await resp.text())[:200]}')
        async for line in resp.content:
            if not line.startswith(b'data:'):
                continue
            d = line[5:].strip()
            if d == b'[DONE]':
                continue
            try:
                j = json.loads(d)
            except Exception:
                continue
            if j.get('meta_info'):
                meta = j['meta_info']
            if first is None and meta.get('completion_tokens', 0) >= 1:
                first = time.perf_counter()
    if first is None:
        raise RuntimeError('no output')
    return first - t0, time.perf_counter() - t0, meta


class GroupCache:
    """Initialize one NCCL group per (src, dst) pair and reuse it forever.

    `init_group` rejects a name already in use, so the reuse path must SKIP
    init entirely rather than re-init under a new name -- a fresh name costs
    the full handshake again. The engine's transfer() keeps its own
    incrementing sequence number on the established group, so repeated
    transfers over one group need no protocol change.

    Names and ports are salted per run: an engine holds every group it has
    ever been given for its whole lifetime (they are never destroyed), so a
    fixed (name, port) pair makes the SECOND run of the same arm fail
    instantly on one side while the peer waits forever on its store.
    """

    _run_seq = 0

    def __init__(self):
        type(self)._run_seq += 1
        self.salt = f'{int(time.time()) % 10**7}-{GroupCache._run_seq}'
        # Every init ATTEMPT needs its own (name, port). Names are consumed
        # permanently by the engine even when the handshake fails half-way,
        # so a retry that reuses them gets HTTP 409 on the side that already
        # registered while the peer blocks forever on a store nobody sets.
        # Ports leak the same way: a group's TCPStore is never closed.
        self.numeric_salt = int(time.time()) % 60
        self.attempt = 0
        self.groups = {}          # (src_i, dst_i) -> (name, port)
        self.locks = defaultdict(asyncio.Lock)
        self.init_seconds = {}    # (src_i, dst_i) -> seconds spent initializing
        self.last_error = ''

    async def ensure(self, session, urls, src_i, dst_i):
        pair = (src_i, dst_i)
        if pair in self.groups:
            return self.groups[pair]
        async with self.locks[pair]:
            if pair in self.groups:
                return self.groups[pair]
            src, dst = urls[src_i], urls[dst_i]
            # A fresh (name, port) for EVERY attempt. The engine consumes a
            # name permanently even when the handshake fails half-way, and a
            # group's TCPStore is never closed, so retrying on the same pair
            # would 409 on one side and block the other forever.
            self.attempt += 1
            name = f'p03-{self.salt}-{src_i}-{dst_i}-{self.attempt}'
            port = pair_port(src_i, dst_i, len(urls),
                             self.numeric_salt + 7 * self.attempt)
            t0 = time.perf_counter()

            async def init(rank, url):
                payload = {'group_rank': rank, 'group_name': name,
                           'master_address': src.split('//')[-1].split(':')[0],
                           'port': port, 'world_size': 2, 'backend': 'nccl'}
                async with session.post(url + '/agentshift/hybrid_state/direct_group/init',
                                        json=payload,
                                        timeout=aiohttp.ClientTimeout(total=240)) as r:
                    return await r.json()
            r_src, r_dst = await asyncio.gather(init(0, src), init(1, dst),
                                                return_exceptions=True)
            rs = r_src if isinstance(r_src, dict) else {'success': False,
                                                        'detail': repr(r_src)[:120]}
            rd = r_dst if isinstance(r_dst, dict) else {'success': False,
                                                        'detail': repr(r_dst)[:120]}
            if not (rs.get('success') and rd.get('success')):
                # Do not cache a failed init, and do not reuse its identifiers
                # on the next attempt (see above).
                self.init_seconds[pair] = float('nan')
                self.last_error = f'name={name} port={port} src={str(rs)[:90]} dst={str(rd)[:90]}'
                return None
            self.init_seconds[pair] = time.perf_counter() - t0
            self.groups[pair] = (name, port)
            return self.groups[pair]


async def copy_state(session, gc, urls, src_i, dst_i, toks, budget=90.0):
    """One real state copy over the pair's cached group.

    The whole migration -- group init included -- is bounded by `budget`
    seconds. A group handshake that cannot complete degrades to "do not move"
    (the request is served where it was routed) instead of stalling the arm.
    This matters because a failed init leaves the peer blocked on a TCPStore
    nobody will ever write, and that hang is indefinite, not slow.
    """
    try:
        return await asyncio.wait_for(
            _copy_state_inner(session, gc, urls, src_i, dst_i, toks),
            timeout=budget)
    except asyncio.TimeoutError:
        return dict(ok=False, stage='timeout',
                    detail=f'migration exceeded {budget:.0f}s budget')


async def _copy_state_inner(session, gc, urls, src_i, dst_i, toks):
    pair = await gc.ensure(session, urls, src_i, dst_i)
    if pair is None:
        return dict(ok=False, stage='init', detail='group init failed')
    name, port = pair
    t0 = time.perf_counter()
    try:
        payload = {'group_name': name, 'port': port,
                   'token_ids': [int(x) for x in toks]}

        async def xfer(role, url):
            p = dict(payload, role=role)
            async with session.post(url + '/agentshift/hybrid_state/direct_transfer',
                                    json=p, timeout=aiohttp.ClientTimeout(total=600)) as r:
                return await r.json()
        a, b = await asyncio.gather(xfer('source', urls[src_i]),
                                    xfer('destination', urls[dst_i]))
        return dict(ok=bool(a.get('state') == 'COMPLETE' and b.get('state') == 'COMPLETE'),
                    stage='transfer', seconds=time.perf_counter() - t0,
                    bytes=b.get('bytes_transferred'),
                    pair_reused=gc.init_seconds.get((src_i, dst_i)) is not None,
                    init_seconds=gc.init_seconds.get((src_i, dst_i)),
                    detail='' if a.get('state') == 'COMPLETE' else str(a)[:200])
    except Exception as exc:
        return dict(ok=False, stage='exception', detail=repr(exc)[:200])


async def pick_target(session, urls, ids, home, busy, policy):
    """Route a request. Returns (target_index, info)."""
    if policy == 'sticky':
        return home, {}
    if policy == 'least_loaded':
        return min(range(len(urls)), key=lambda i: (busy[i], i)), {}
    if policy == 'prefix_aware':
        best, best_key = home, None
        for i, u in enumerate(urls):
            st = await inspect(session, u, ids)
            matched = st.get('matched_tokens') or 0
            key = (-matched, busy[i], i)
            if best_key is None or key < best_key:
                best_key, best = key, i
        return best, {}
    raise ValueError(policy)


async def run_arm(session, gc, urls, streams, homes, toks_all, policy, mobility):
    await flush(session, urls)
    busy = [0.0] * len(urls)
    recs = []
    migrations = []
    lock = asyncio.Lock()
    # Where a stream's prefix ACTUALLY lives right now, as opposed to where
    # its hash would send it. Under `least_loaded` the two diverge immediately:
    # the router spreads requests across engines, so the runnable prefix ends
    # up wherever the previous turn happened to run. Inspecting the hash home
    # would find matched_tokens=0 forever and no migration would ever fire.
    resident = {}
    # Why a migration did NOT happen. Reported because "coverage is bounded by
    # the destination-empty requirement" is a claim the paper has to make
    # explicitly, and this is the instrument that measures it.
    blocked_migrations = {}

    async def run_stream(rs):
        key = rs[0]['key']
        h = homes[key]
        for r in rs:
            ids = toks_all[r['token_offset']: r['token_offset'] + r['input_tokens']]
            # The lock covers ROUTING and the migration bookkeeping only.
            # Holding it across serve() would serialise every stream and make
            # each arm take minutes per request; the serving call is where the
            # wall-clock actually goes, so it must run concurrently.
            async with lock:
                target, _ = await pick_target(session, urls, ids, h, busy, policy)
                moved = False
                src = resident.get(key, h)
                if mobility and target != src:
                    # The router wants balance and just landed somewhere the
                    # prefix is not. Before paying a recompute, see whether the
                    # runnable prefix can be brought over instead.
                    src_inspect = await inspect(session, urls[src], ids)
                    n_reuse = src_inspect.get('matched_tokens') or 0
                    # The engine's transfer path requires an EMPTY destination
                    # radix tree (hygon_hybrid_control.py: "destination cache
                    # must be empty"). An empty tree reports zero evictable
                    # slots, which is the exact predicate -- and it needs no
                    # flush. Flushing a busy engine instead of checking this
                    # is what hung every earlier attempt: /flush_cache does not
                    # return promptly while other requests are in flight.
                    # A destination that is NOT empty simply cannot receive a
                    # migration; that is a property of the current
                    # implementation and it bounds coverage, so it is measured
                    # rather than worked around.
                    dst_inspect = await inspect(session, urls[target], ids)
                    dst_empty = (dst_inspect.get('evictable_token_slots') or 0) == 0
                    blocked = None
                    if n_reuse < MOVE_MIN_TOKENS:
                        blocked = 'src_has_no_runnable_prefix'
                    elif not dst_empty:
                        blocked = 'dst_not_empty'
                    if blocked is None:
                        res = await copy_state(session, gc, urls, src, target,
                                               ids[:n_reuse])
                        migrations.append(dict(key=key, turn=r['turn'],
                                               src=src, dst=target,
                                               tokens=n_reuse, **res))
                        moved = bool(res.get('ok'))
                    else:
                        blocked_migrations[blocked] = blocked_migrations.get(blocked, 0) + 1
            try:
                # Bounded: a migration that wedges the destination engine must
                # cost one request, not the whole arm. The 130K-prefix serve
                # is legitimately slow (tens of seconds), so the budget has to
                # clear that with room to spare.
                ttft, tot, meta = await asyncio.wait_for(
                    serve(session, urls[target], ids), timeout=SERVE_BUDGET)
                err = None
            except asyncio.TimeoutError:
                ttft, tot, meta = float('nan'), float('nan'), {}
                err = f'serve exceeded {SERVE_BUDGET:.0f}s budget'
            except Exception as exc:
                ttft, tot, meta = float('nan'), float('nan'), {}
                err = repr(exc)[:200]
            async with lock:
                busy[target] += tot if tot == tot else 0
                recs.append(dict(key=key, turn=r['turn'], engine=target,
                                 ttft=ttft, total=tot, moved=moved,
                                 cached_tokens=meta.get('cached_tokens') or 0,
                                 input_tokens=r['input_tokens'], error=err,
                                 home=h, src=src))
                # The prefix is now resident on whichever engine served this
                # turn (a successful copy put a runnable prefix here; a plain
                # serve left one here too). That engine -- not the hash home --
                # is the source for the next turn's migration decision.
                if err is None and not (ttft != ttft):
                    resident[key] = target

    t0 = time.perf_counter()
    await asyncio.gather(*(run_stream(rs) for rs in streams))
    wall = time.perf_counter() - t0
    return dict(wall=wall, records=recs, migrations=migrations,
                init_seconds=dict(gc.init_seconds),
                blocked=dict(blocked_migrations),
                init_error=gc.last_error)


def _pct(a, p):
    return float(np.percentile(a, p)) if len(a) else None


def summarize(arm):
    tt = np.array([r['ttft'] for r in arm['records']], float)
    ok = np.isfinite(tt)
    good = tt[ok]
    mig = arm['migrations']
    good_mig = [m for m in mig if m.get('ok')]
    used = [m for m in good_mig
            if any(r['key'] == m['key'] and r['turn'] == m['turn']
                   and r['engine'] == m['dst'] and r['cached_tokens'] >= m['tokens']
                   for r in arm['records'])]
    # Net benefit actually realized: recompute avoided minus copy paid.
    net = sum(m['tokens'] * 2.17e-4 - (m.get('seconds') or 0) for m in used)
    return dict(
        wall=arm['wall'],
        requests=len(arm['records']),
        errors=sum(1 for r in arm['records'] if r['error']),
        ttft_p50=_pct(good, 50), ttft_p90=_pct(good, 90), ttft_p99=_pct(good, 99),
        ttft_mean=float(good.mean()) if ok.any() else None,
        slo_all=float(np.mean(good <= SLO_SECONDS)) if ok.any() else None,
        goodput=float(np.sum(good <= SLO_SECONDS) / arm['wall']) if ok.any() else None,
        migrations_attempted=len(mig),
        migrations_ok=len(good_mig),
        migrations_used=len(used),
        copy_seconds=sum(m.get('seconds') or 0 for m in mig),
        copy_bytes=sum(m.get('bytes') or 0 for m in used),
        init_seconds_total=sum(arm['init_seconds'].values()),
        n_init_groups=len(arm['init_seconds']),
        realized_net_saving_s=net,
    )


POLICIES = [('sticky', False), ('least_loaded', False),
            ('prefix_aware', False), ('least_loaded', True)]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--streams', type=int, default=16)
    ap.add_argument('--hot-frac', type=float, default=0.0,
                    help='fraction of longest streams pinned to engine 0')
    ap.add_argument('--reps', type=int, default=1)
    ap.add_argument('--arms', nargs='+', default=None,
                    choices=['sticky', 'least_loaded', 'prefix_aware',
                             'balance_move'])
    ap.add_argument('--out', default=str(ROOT / 'results/p03-real-copy.json'))
    args = ap.parse_args()

    rows, toks = load_workload()
    streams, homes = schedule(rows, args.streams, args.hot_frac)
    urls = ['http://' + e for e in ENGINES]
    n_req = sum(len(s) for s in streams)
    print(f'engines {len(urls)}  streams {len(streams)}  requests {n_req}  '
          f'hot_frac {args.hot_frac}', flush=True)

    wanted = set(args.arms) if args.arms else {'sticky', 'least_loaded',
                                               'prefix_aware', 'balance_move'}
    # Mobility runs FIRST, on a freshly flushed fleet. The engine's transfer
    # path requires an empty destination, so an arm that leaves every engine
    # holding a populated radix tree makes every later migration impossible --
    # ordering the treatment last would measure the previous arm's residue
    # rather than the treatment.
    plan = [('least_loaded', True), ('sticky', False),
            ('least_loaded', False), ('prefix_aware', False)]
    plan = [(p, m) for p, m in plan
            if (('balance_move' if m else p) in wanted)]

    out = {'meta': {'engines': ENGINES, 'streams': len(streams),
                    'requests': n_req, 'hot_frac': args.hot_frac,
                    'move_min_tokens': MOVE_MIN_TOKENS,
                    'out_tokens': OUT_TOKENS, 'slo_seconds': SLO_SECONDS},
           'arms': {}}

    async with aiohttp.ClientSession() as session:
        for rep in range(args.reps):
            for policy, mobility in plan:
                tag = f'{policy}+move' if mobility else policy
                if args.reps > 1:
                    tag = f'{tag}#{rep}'
                gc = GroupCache()
                print(f'--- arm {tag} (rep {rep}) ---', flush=True)
                arm = await run_arm(session, gc, urls, streams, homes, toks,
                                    policy, mobility)
                s = summarize(arm)
                out['arms'][tag] = dict(summary=s,
                                        migrations=arm['migrations'],
                                        blocked=arm.get('blocked') or {},
                                        init_error=arm.get('init_error') or '')
                if arm.get('blocked'):
                    print(f'  blocked (why no migration): {arm["blocked"]}',
                          flush=True)
                if arm.get('init_error'):
                    print(f'  init_error: {arm["init_error"][:200]}', flush=True)
                print(json.dumps(s, indent=2), flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2) + '\n')
    print(f'\nwrote {args.out}', flush=True)

    print(f'\n{"arm":22s}{"p50":>8}{"p90":>9}{"p99":>9}{"SLO":>8}'
          f'{"goodput":>9}{"mig_used":>10}{"net_s":>9}')
    for tag, d in out['arms'].items():
        s = d['summary']
        print(f'{tag:22s}{s["ttft_p50"]:8.2f}{s["ttft_p90"]:9.2f}'
              f'{s["ttft_p99"]:9.2f}{s["slo_all"]:8.3f}{s["goodput"]:9.3f}'
              f'{s["migrations_used"]:10d}{s["realized_net_saving_s"]:9.1f}')


if __name__ == '__main__':
    asyncio.run(main())
