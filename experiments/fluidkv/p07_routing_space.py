"""Same-backend causal pilot: routing score vs executable-state preparation.

This is a LMetric-score port, NOT the complete upstream Blitz/vLLM system.
SGLang forward-step telemetry decrements admitted prefill work; first-token
feedback corrects cache predictions. Cache publication only follows confirmed
completed inputs. Actual engine cache matches are audited between trials.
"""
import argparse
import asyncio
from collections import Counter
import contextlib
import hashlib
import importlib.util
import inspect as pyinspect
import json
from pathlib import Path
import random
import shlex
import time

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/routing-space-20261010'
spec = importlib.util.spec_from_file_location('skew', Path(__file__).with_name('p06_skew_readiness.py'))
skew = importlib.util.module_from_spec(spec)
spec.loader.exec_module(skew)


def lcp(a, b):
    if not len(a) or not len(b):
        return 0
    n = min(len(a), len(b))
    bad = np.flatnonzero(a[:n] != b[:n])
    return int(bad[0]) if len(bad) else n


async def generate(session, url, ids, output, arrival=None, callback=None, rid=None):
    started = time.perf_counter()
    arrival = started if arrival is None else arrival
    first = last = None
    count = 0
    final = {}
    intervals = []
    payload={'input_ids':ids, 'stream':True,
            'sampling_params':{'temperature':0, 'ignore_eos':True, 'max_new_tokens':output}}
    if rid is not None:
        payload['rid']=rid
    async with session.post(url+'/generate', json=payload) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            if not raw.startswith(b'data:') or raw[5:].strip() == b'[DONE]':
                continue
            final = json.loads(raw[5:])
            new = final.get('meta_info', {}).get('completion_tokens', 0)
            if new > count:
                now = time.perf_counter()
                if first is None:
                    first = now
                elif last is not None:
                    intervals.append((now-last)/(new-count))
                if callback:
                    feedback = callback(final, new, count)
                    if pyinspect.isawaitable(feedback):
                        await feedback
                count, last = new, now
    done = time.perf_counter()
    if first is None or count != output:
        raise RuntimeError(f'output length {count} != {output}')
    return dict(error=None, ttft=first-arrival, e2e=done-arrival,
        arrival_at=arrival,dispatch_at=started,first_at=first,last_at=last,completed_at=done,
        service_seconds=done-started, router_wait=started-arrival,
        tpot=(last-first)/(count-1) if count>1 else None,
        token_intervals=intervals, coalesced_tokens=0,
        cached_tokens=final.get('meta_info', {}).get('cached_tokens', 0),
        completion_tokens=count, output_sha256=hashlib.sha256(json.dumps(final.get('output_ids')).encode()).hexdigest(),
        output_ids=final.get('output_ids'),
        meta_info=final.get('meta_info', {}))


class Router:
    def __init__(self, policy, calibration, initial, telemetry, lookahead=128):
        self.policy = policy
        self.calibration = calibration
        self.cache = [list(x) for x in initial]
        self.active = [{} for _ in initial]
        self.pending = [0. for _ in initial]
        self.telemetry = telemetry
        self.corrections = []
        self.decisions = []
        self.arrivals = []
        self.hotspot_streak = {}
        self.lookahead = lookahead

    async def choose(self, request_id, ids, output, metadata):
        return self.reserve(request_id, ids)

    async def completed(self, target, request_id, ids, result):
        self.release(target, request_id, ids, result['error'] is None)

    async def admit_preparation(self, ids, target=1, request_id='prepare'):
        self.active[target][request_id] = dict(context=len(ids), predicted_hit=0,
            initial_new=len(ids), first=False)
        self.pending[target] += len(ids)

    async def close(self):
        pass

    def hit(self, worker, ids):
        # Conservative 512-token rounding of the input-only cache mirror;
        # actual executable boundaries can differ and are corrected on feedback.
        return max((lcp(ids, x)//512*512 for x in self.cache[worker]), default=0)

    def step(self, worker, event):
        used = event.get('scheduled_requests', {}).get('sum_prefill_tokens', 0)
        self.pending[worker] = max(0., self.pending[worker]-used)

    def decode_cost(self, bs, contexts):
        if self.policy == 'context_grid':
            grid = self.calibration['decode_grid']
            lengths = sorted({r['length'] for r in grid})
            values = []
            for length in lengths:
                batches = sorted({r['bs'] for r in grid if r['length']==length})
                timings = [float(np.median([r['tpot'] for r in grid
                    if r['length']==length and r['bs']==batch])) for batch in batches]
                values.append(float(np.interp(bs,batches,timings)))
            return max(.001,float(np.interp(contexts/max(1,bs),lengths,values)))
        a, b, c = self.calibration['decode_coefficients']
        return max(.001, a+b*bs+c*contexts/65536)

    def prefill_cost(self, tokens):
        pts = self.calibration['prefill_points']
        if tokens <= pts[0][0]:
            return pts[0][1]*tokens/max(1, pts[0][0])
        return float(np.interp(tokens, [p[0] for p in pts], [p[1] for p in pts])) if tokens<=pts[-1][0] else pts[-1][1]*tokens/pts[-1][0]

    def reserve(self, request_id, ids):
        hits = [self.hit(i, ids) for i in range(2)]
        new = [len(ids)-h for h in hits]
        counts = [len(x) for x in self.active]
        contexts = [sum(x['context'] for x in rows.values()) for rows in self.active]
        if self.policy in ('lmetric','lmetric_guard'):
            scores = [(self.pending[i]+new[i])*(counts[i]+1) for i in range(2)]
        elif self.policy in ('context','context_grid'):
            # Declared fixed lookahead; no future output lengths or arrivals.
            # Profiled prefill cost + profile-based completion cost.
            scores = [self.prefill_cost(self.pending[i]+new[i])+
                self.lookahead*self.decode_cost(counts[i]+1, contexts[i]+len(ids)) for i in range(2)]
        elif self.policy == 'balanced':
            scores = counts
        elif self.policy == 'sticky':
            scores = [0, 1]
        else:
            raise ValueError(self.policy)
        target = min(range(2), key=lambda i:(scores[i], counts[i], i))
        filtered = []
        if self.policy == 'lmetric_guard':
            # Paper §5.2 two-stage fallback port, with a declared 32K prefix
            # class and trailing 60s demand window. This is a sensitivity,
            # not a claim of byte-for-byte upstream implementation equivalence.
            now = time.perf_counter()
            key = hashlib.sha256(ids[:32768].tobytes()).hexdigest()
            self.arrivals = [(t,k) for t,k in self.arrivals if now-t<=60]
            self.arrivals.append((now,key))
            fraction = sum(k==key for _,k in self.arrivals)/len(self.arrivals)
            warm = [i for i in range(2) if hits[i]>=32768]
            cold = [i for i in range(2) if i not in warm]
            alarm = bool(warm and cold and fraction>len(warm)/2)
            low_warm = alarm and min(scores[i] for i in warm)<=min(scores[i] for i in cold)
            self.hotspot_streak[key] = self.hotspot_streak.get(key,0)+1 if low_warm else 0
            if low_warm and self.hotspot_streak[key]>2*len(warm):
                filtered = warm
                target = min(cold,key=lambda i:(scores[i],counts[i],i))
        row = dict(request_id=request_id, target=target, scores=scores, hits=hits,
                   pending=list(self.pending), counts=counts, contexts=contexts,filtered=filtered)
        self.decisions.append(row)
        self.active[target][request_id] = dict(context=len(ids), predicted_hit=hits[target],
            initial_new=new[target], first=False)
        self.pending[target] += new[target]
        return target, row

    def progress(self, target, request_id, payload, count, old_count):
        entry = self.active[target][request_id]
        entry['context'] += count-old_count
        if not entry['first']:
            actual = payload.get('meta_info', {}).get('cached_tokens', 0)
            diff = actual-entry['predicted_hit']
            self.pending[target] = max(0., self.pending[target]-diff)
            self.corrections.append(dict(request_id=request_id, target=target,
                predicted=entry['predicted_hit'], actual=actual, difference=diff))
            entry['first'] = True

    def release(self, target, request_id, ids, success):
        self.active[target].pop(request_id)
        if success:
            self.cache[target].append(np.asarray(ids, dtype=np.int32))
        if not self.active[target]:
            self.pending[target] = 0.


class Telemetry:
    def __init__(self, fleet, path):
        self.fleet, self.path = fleet, path
        self.router = None
        self.events = [0 for _ in fleet]
        self.errors = []
        self.tasks, self.processes = [], []
        self.peaks = [0 for _ in fleet]

    async def start(self):
        for i, rec in enumerate(self.fleet):
            code = ('import json,zmq,msgspec; c=zmq.Context(); s=c.socket(zmq.SUB); '
                's.setsockopt(zmq.SUBSCRIBE,b""); s.connect('+repr(rec['fpm_ipc'])+');\n'
                'while True:\n'
                ' p=s.recv_multipart(); print(json.dumps(msgspec.msgpack.decode(p[-1])),flush=True)\n')
            command = shlex.join(['docker','exec',rec['name'],'python','-u','-c',code])
            proc = await asyncio.create_subprocess_exec('ssh','-o','BatchMode=yes',rec['node'],command,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            self.processes.append(proc)
            self.tasks.append(asyncio.create_task(self.read(i,proc)))

    async def read(self, i, proc):
        with self.path.with_name(f'telemetry-{i}.jsonl').open('w') as log:
            while raw := await proc.stdout.readline():
                event = json.loads(raw)
                self.events[i] += 1
                running = event.get('scheduled_requests', {}).get('num_decode_requests', 0)
                self.peaks[i] = max(self.peaks[i], running)
                log.write(json.dumps(dict(observed=time.perf_counter(),event=event))+'\n')
                if self.router:
                    self.router.step(i,event)
            stderr = await proc.stderr.read()
            self.errors.append(dict(worker=i, message=stderr.decode()))

    async def close(self):
        self.router = None
        for proc in self.processes:
            if proc.returncode is None:
                proc.terminate()
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        for proc in self.processes:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(proc.wait(),5)


async def inspect(session, url, ids):
    return await skew.json_post(session,url,'/agentshift/hybrid_state/inspect',{'token_ids':ids})


async def copy_gate(session,urls,case,tokens,gc,out):
    await skew.reset(session,urls)
    parents=[]
    seen = set()
    for stream in case['streams']:
        row=stream['requests'][0]
        parent=tokens[row['token_offset']:row['token_offset']+row['input_tokens']].astype(int).tolist()
        parents.append(parent)
        key = hashlib.sha256(np.asarray(parent,dtype=np.int32).tobytes()).hexdigest()
        if key not in seen:
            await generate(session,urls[0],parent,8)
            seen.add(key)
    span_to_prepare = case.get('preparation_tokens',case['common_parent_tokens'])
    available=await inspect(session,urls[0],parents[0][:span_to_prepare])
    span=available['matched_tokens']
    if span<32768:
        raise RuntimeError('copy gate has no long executable common boundary')
    common=parents[0][:span]
    payload={'token_ids':common,'return_digest':True}
    source=await skew.json_post(session,urls[0],'/agentshift/hybrid_state/inspect',payload)
    copied=await skew.p03.copy_state(session,gc,urls,0,1,common)
    if not copied.get('ok'):
        raise RuntimeError(copied)
    destination=await skew.json_post(session,urls[1],'/agentshift/hybrid_state/inspect',payload)
    prompt=common+np.random.default_rng(43210).integers(1000,240000,64).tolist()
    a=await generate(session,urls[0],prompt,16)
    b=await generate(session,urls[1],prompt,16)
    result=dict(prepared_tokens=span,source=source,destination=destination,copy=copied,
        source_continuation=a,destination_continuation=b,
        state_digest_equal=source['state_sha256']==destination['state_sha256'],
        same_shape_output_equal=a['output_sha256']==b['output_sha256'],
        reused=a['cached_tokens']>=span and b['cached_tokens']>=span,
        scope='Shared executable boundary from the multi-parent radix tree; full KV and final recurrent '
              'state digest, plus sequential same-shape continuation. Outside performance trials.')
    (out/'correctness.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['state_digest_equal'] or not result['same_shape_output_equal'] or not result['reused']:
        raise RuntimeError('copy correctness gate failed')
    await skew.reset(session,urls)


async def calibration(session, urls, tokens, out):
    from scipy.optimize import nnls
    ids = tokens[:98304].astype(int).tolist()
    records = []
    # Equal kernel warmup on both replicas, then cold cost on precompiled shapes.
    for url in urls:
        for length in (32768,65536,98304):
            await skew.reset(session,[url])
            await generate(session,url,ids[:length],8)
    prefill = []
    for length in (32768,65536,98304):
        await skew.reset(session,urls)
        r = await generate(session,urls[0],ids[:length],8)
        prefill.append([length,r['ttft']])
        records.append(dict(kind='cold',length=length,**r))
    grid = []
    for length in (32768,65536):
        for bs in (1,2,4,8):
            for url in urls:
                await skew.reset(session,[url])
                await generate(session,url,ids[:length],8)
                prompts = [ids[:length]+np.random.default_rng(710+i).integers(1000,240000,64).tolist()
                           for i in range(bs)]
                # Compile this exact shape before profiling.
                await asyncio.gather(*(generate(session,url,p,8) for p in prompts))
                rs = await asyncio.gather(*(generate(session,url,p,32) for p in prompts))
                value = float(np.median([r['tpot'] for r in rs]))
                grid.append(dict(length=length,bs=bs,url=url,tpot=value))
                print('profile',length,bs,url,round(value,5),flush=True)
                (out/'calibration-progress.json').write_text(json.dumps(grid,indent=2)+'\n')
    x = np.array([[1,r['bs'],r['bs']*r['length']/65536] for r in grid])
    coeff,residual = nnls(x,np.array([r['tpot'] for r in grid]))
    result = dict(prefill_points=prefill,decode_grid=grid,decode_coefficients=coeff.tolist(),
                  residual=residual,records=records,
                  semantics='Nonnegative model: step seconds = intercept + bs coefficient + total-context coefficient. '
                            'No output or future-arrival oracle; fixed 128-token scoring horizon.')
    (out/'calibration.json').write_text(json.dumps(result,indent=2)+'\n')
    await skew.reset(session,urls)
    return result


async def trial(session,args,case,policy,prep,rep,tokens,cal,telemetry,gc):
    urls = args.urls
    await skew.reset(session,urls)
    streams = case['streams']
    def ids(row):
        off = row['token_offset']
        return tokens[off:off+row['input_tokens']].astype(int).tolist()
    parents = [ids(s['requests'][0]) for s in streams]
    parent_keys = [hashlib.sha256(np.asarray(p,dtype=np.int32).tobytes()).hexdigest() for p in parents]
    # Source-only placement is a declared skew sensitivity, not recovered homes.
    homes = [0]*len(parents) if args.placement=='skew' else [i%2 for i in range(len(parents))]
    seeded = set()
    for parent,home,key in zip(parents,homes,parent_keys):
        if (home,key) not in seeded:
            await generate(session,urls[home],parent,8)
            seeded.add((home,key))
    initial = [[],[]]
    parent_audit = []
    observed = {}
    for parent,home,key in zip(parents,homes,parent_keys):
        if (home,key) not in observed:
            observed[home,key] = await inspect(session,urls[home],parent)
            span = observed[home,key]['matched_tokens']
            initial[home].append(np.array(parent[:span],dtype=np.int32))
        state = observed[home,key]
        matched = state['matched_tokens']
        if matched < 32768:
            raise RuntimeError(f'initial state lost: {matched}')
        parent_audit.append(dict(home=home,matched=matched,input_tokens=len(parent)))
    background_prefix = None
    if args.background_requests:
        background_prefix = np.random.default_rng(918271).integers(1000,240000,32768,dtype=np.int32).tolist()
        await generate(session,urls[0],background_prefix,8)
        state=await inspect(session,urls[0],background_prefix)
        initial[0].append(np.array(background_prefix[:state['matched_tokens']],dtype=np.int32))
    prep_start = time.perf_counter()
    arrival = prep_start+args.gap
    preparation = dict(method=prep,seconds=0.,bytes=0)
    recompute_common = None
    if prep=='ideal':
        # Actual finite-memory target population, timing removed from eligibility.
        # This deliberately generous counterfactual is not an achieved gain.
        for parent,home in zip(parents,homes):
            await generate(session,urls[1-home],parent,8)
        for parent,home in zip(parents,homes):
            state = await inspect(session,urls[1-home],parent)
            initial[1-home].append(np.array(parent[:state['matched_tokens']],dtype=np.int32))
        preparation['seconds'] = time.perf_counter()-prep_start
        arrival = time.perf_counter()+.25
    elif prep in ('ideal_common','copy','recompute'):
        if args.placement != 'skew':
            raise ValueError('real preparation pilot requires empty target setup')
        span_to_prepare = case.get('preparation_tokens',case['common_parent_tokens'])
        available = await inspect(session,urls[0],parents[0][:span_to_prepare])
        span = available['matched_tokens']
        if span<32768:
            raise RuntimeError(f'no sufficiently long executable common boundary: {span}')
        common = parents[0][:span]
        if prep=='copy':
            preparation = await skew.p03.copy_state(session,gc,urls,0,1,common)
            preparation['method']='copy'
            if not preparation.get('ok'):
                raise RuntimeError(preparation)
        elif prep=='ideal_common':
            r = await generate(session,urls[1],common,8)
            preparation = dict(method='recompute',seconds=r['service_seconds'],bytes=0)
        else:
            recompute_common=common
        if prep!='recompute':
            state = await inspect(session,urls[1],common)
            if state['matched_tokens'] < span-512:
                raise RuntimeError('preparation not executable')
            initial[1].append(np.array(common[:state['matched_tokens']],dtype=np.int32))
        preparation['prepared_tokens']=span
        if prep=='ideal_common':
            preparation['method']='ideal_common'
            arrival=time.perf_counter()+(.25 if not args.background_requests else 1.25)
    prep_done = time.perf_counter()
    service_start=arrival-1 if args.background_requests else arrival
    router_factory = getattr(args, 'router_factory', Router)
    router = router_factory(policy,cal,initial,telemetry,args.lookahead)
    telemetry.router=router
    telemetry.peaks=[0,0]
    preparation_task=None
    if recompute_common is not None:
        # Proactive recompute consumes target capacity while source requests
        # continue. Its work is registered before dispatch, like consumer work.
        await router.admit_preparation(recompute_common)
        async def prepare():
            nonlocal preparation,prep_done
            r=await generate(session,urls[1],recompute_common,8,
                callback=lambda payload,n,old:router.progress(1,'prepare',payload,n,old))
            await router.completed(1,'prepare',recompute_common,r)
            prep_done=time.perf_counter()
            preparation=dict(method='recompute',seconds=r['service_seconds'],bytes=0,
                prepared_tokens=len(recompute_common),mode='asynchronous',result=r)
        preparation_task=asyncio.create_task(prepare())
    if service_start>time.perf_counter():
        await asyncio.sleep(service_start-time.perf_counter())
    rows,tasks=[],[]
    async def consumer(index,s):
        eligible = arrival+s['first_offset']
        begin=eligible
        for turn,row in enumerate(s['requests'][1:]):
            if eligible>time.perf_counter():
                await asyncio.sleep(eligible-time.perf_counter())
            prompt=ids(row)
            rid=f'{index}:{turn}'
            output = min(row['native_output_tokens'],args.output_cap) if args.output_cap else row['output_tokens']
            target,decision=await router.choose(rid,np.array(prompt,dtype=np.int32),output,
                dict(row, chat_id=s['key'], turn=turn))
            try:
                result=await generate(session,urls[target],prompt,output,eligible,
                    lambda payload,n,old:router.progress(target,rid,payload,n,old))
            except Exception as exc:
                result=dict(error=repr(exc),ttft=None,e2e=None,tpot=None,
                            cached_tokens=0,token_intervals=[],coalesced_tokens=0)
            done=time.perf_counter()
            await router.completed(target,rid,prompt,result)
            saved=dict(case=case['name'],role='foreground',placement=args.placement,policy=policy,prep=prep,
                repetition=rep,consumer=index,turn=turn,target=target,input_tokens=len(prompt),
                output_tokens=output,native_output_tokens=row['native_output_tokens'],
                token_sha256=row['token_sha256'],decision=decision,**result)
            rows.append(saved)
            with (args.out/'requests.jsonl').open('a') as log:
                log.write(json.dumps(saved)+'\n')
            if turn<len(s['gaps']):
                eligible=done+s['gaps'][turn]
        tasks.append(dict(consumer=index,completion=time.perf_counter()-begin,
                          completion_from_service=time.perf_counter()-service_start))
    async def background(index):
        prompt=background_prefix+np.random.default_rng(831+index).integers(1000,240000,64).tolist()
        rid=f'bg:{index}'
        target,decision=await router.choose(rid,np.array(prompt,dtype=np.int32),args.background_output,
            dict(chat_id=rid, hash_ids=[918271,index], turn=0))
        try:
            result=await generate(session,urls[target],prompt,args.background_output,service_start,
                lambda payload,n,old:router.progress(target,rid,payload,n,old))
        except Exception as exc:
            result=dict(error=repr(exc),ttft=None,e2e=None,tpot=None,cached_tokens=0,
                        token_intervals=[],coalesced_tokens=0)
        await router.completed(target,rid,prompt,result)
        saved=dict(case=case['name'],role='background',placement=args.placement,policy=policy,prep=prep,
            repetition=rep,consumer=index,turn=0,target=target,input_tokens=len(prompt),
            output_tokens=args.background_output,native_output_tokens=None,
            token_sha256=hashlib.sha256(np.asarray(prompt,dtype=np.int32).tobytes()).hexdigest(),
            decision=decision,**result)
        rows.append(saved)
        with (args.out/'requests.jsonl').open('a') as log:
            log.write(json.dumps(saved)+'\n')
    await asyncio.gather(*(background(i) for i in range(args.background_requests)),
                         *(consumer(i,s) for i,s in enumerate(streams)))
    if preparation_task:
        await preparation_task
    wall=time.perf_counter()-service_start
    telemetry.router=None
    await skew.drain(session,urls)
    audit=[]
    for i,url in enumerate(urls):
        observed = {}
        for j,(parent,key) in enumerate(zip(parents,parent_keys)):
            if key not in observed:
                state=await inspect(session,url,parent)
                observed[key] = (state['matched_tokens'],router.hit(i,np.array(parent,dtype=np.int32)))
            actual,predicted=observed[key]
            audit.append(dict(worker=i,parent=j,actual_matched_tokens=actual,predicted=predicted))
    summary=skew.summarize(rows,wall,args.ttft_slo,args.tpot_slo)
    summary.update(task_p50=float(np.median([x['completion'] for x in tasks])),
        task_p90=float(np.percentile([x['completion'] for x in tasks],90)),
        background_requests=args.background_requests,
        foreground_wall=max(x['completion_from_service'] for x in tasks))
    await router.close()
    return dict(case=case['name'],placement=args.placement,policy=policy,prep=prep,repetition=rep,
        service_start=service_start,preparation_start=prep_start,preparation_done=prep_done,
        summary=summary,task_records=tasks,initial_cache_audit=parent_audit,final_cache_audit=audit,
        preparation=preparation,preparation_wall=prep_done-prep_start,
        exposed_preparation=max(0.,prep_done-arrival),corrections=router.corrections,
        telemetry_counts=list(telemetry.events),peak_decode=list(telemetry.peaks))


async def run(args):
    args.out.mkdir(exist_ok=False,parents=True)
    fleet=json.loads((BASE/'fleet.json').read_text())
    tokens=np.load(BASE/'cohort-tokens.npy',mmap_mode='r')
    data=json.loads((BASE/'cohorts.json').read_text())
    cases=[c for c in data['cohorts'] if c['name'] in args.cases]
    out=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        workload_scope=data['scope'],policy_scope='LMetric-score SGLang port with aggregate forward-step prefill feedback, '
        'admission reservations and first-token hit correction; not upstream Blitz/vLLM head-to-head.',trials=[])
    driver=Path(__file__).read_bytes()
    out['driver_sha256']=hashlib.sha256(driver).hexdigest()
    out['runtime_sha256']=hashlib.sha256((ROOT/'experiments/weka_baselines/environment/runtime/managers/hygon_hybrid_control.py').read_bytes()).hexdigest()
    out['cohorts_sha256']=hashlib.sha256((BASE/'cohorts.json').read_bytes()).hexdigest()
    (args.out/'driver.py').write_bytes(driver)
    (args.out/'manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    telemetry=Telemetry(fleet,args.out/'telemetry.jsonl')
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300)) as session:
        out['server_info']=[]
        for url in args.urls:
            async with session.get(url+'/get_server_info') as response:
                response.raise_for_status()
                out['server_info'].append(await response.json())
        await telemetry.start()
        try:
            if args.calibration:
                cal=json.loads(args.calibration.read_text())
            else:
                source=np.load(ROOT/'experiments/weka_baselines/data/tokens.npy',mmap_mode='r')
                cal=await calibration(session,args.urls,source,args.out)
            out['calibration']=cal
            await asyncio.sleep(2)
            if min(telemetry.events)==0:
                raise RuntimeError(f'no step telemetry: {telemetry.errors}')
            gc=skew.p03.GroupCache()
            if any(x=='copy' for x in args.preps):
                await skew.reset(session,args.urls)
                started=time.perf_counter()
                group=await gc.ensure(session,args.urls,0,1)
                if group is None:
                    raise RuntimeError(gc.last_error)
                out['group_init_seconds']=time.perf_counter()-started
                await copy_gate(session,args.urls,cases[0],tokens,gc,args.out)
            rng=random.Random(args.seed)
            cells=[(c,p,a) for c in cases for p in args.policies for a in args.preps]
            for rep in range(args.reps):
                for c,p,a in rng.sample(cells,len(cells)):
                    print('TRIAL',c['name'],args.placement,p,a,rep,flush=True)
                    result=await trial(session,args,c,p,a,rep,tokens,cal,telemetry,gc)
                    out['trials'].append(result)
                    (args.out/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
                    print('RESULT',json.dumps({k:result[k] for k in ('case','policy','prep','summary')}),flush=True)
                    if result['summary']['errors']:
                        raise RuntimeError('request errors; no performance verdict')
            out['telemetry_errors']=telemetry.errors
            (args.out/'summary.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
        finally:
            await telemetry.close()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--urls',nargs=2,default=['http://12.11.3.27:31981','http://12.11.3.27:31982'])
    ap.add_argument('--cases',nargs='+',default=['native-n2','native-n6'])
    ap.add_argument('--policies',nargs='+',choices=['lmetric','lmetric_guard','context','context_grid','balanced','sticky'],default=['lmetric','context'])
    ap.add_argument('--preps',nargs='+',choices=['none','ideal','ideal_common','copy','recompute'],default=['none','ideal'])
    ap.add_argument('--placement',choices=['skew','balanced'],default='skew')
    ap.add_argument('--reps',type=int,default=3)
    ap.add_argument('--gap',type=float,default=1.)
    ap.add_argument('--ttft-slo',type=float,default=10.)
    ap.add_argument('--tpot-slo',type=float,default=.2)
    ap.add_argument('--seed',type=int,default=20261010)
    ap.add_argument('--output-cap',type=int)
    ap.add_argument('--lookahead',type=int,default=128)
    ap.add_argument('--background-requests',type=int,default=0)
    ap.add_argument('--background-output',type=int,default=512)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--out',type=Path,required=True)
    asyncio.run(run(ap.parse_args()))


if __name__=='__main__':
    main()
