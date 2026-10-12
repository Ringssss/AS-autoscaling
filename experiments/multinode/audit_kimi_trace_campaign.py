"""Final completeness, input-length, clock-accounting and provenance audit."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent/'results'
FIELDS=('stage','window','length','withdraw','tool_seconds','rate_scale','output_cap','repeat','policy')
NAMES=['kimi-trace-screen','kimi-trace-screen-resume','kimi-trace-observer-recheck','kimi-trace-validation','kimi-trace-coverage','kimi-trace-native','kimi-trace-long']


def main(output):
    planned=[]; actual=[]; unique=set(); stages=defaultdict(lambda:Counter())
    core_hashes=defaultdict(set); full_hits=Counter(); native_hashes={}; mismatches=[]
    for name in NAMES:
        root=BASE/name
        if name!='kimi-trace-screen': assert (root/'COMPLETE').exists(),f'incomplete: {root}'
        # The original screen plan contains all 96; resume is a suffix of it.
        if name!='kimi-trace-screen-resume':
            planned.extend(json.loads((root/'predeclared_jobs.json').read_text()))
        manifest=json.loads((root/'source_manifest.json').read_text())
        for path,digest in manifest.items():
            snap=root/'source_snapshot'/path
            assert hashlib.sha256(snap.read_bytes()).hexdigest()==digest,(root,path)
            if not path.endswith('run_kimi_trace_replay.py'): core_hashes[path].add(digest)
        for path in sorted(root.glob('[0-9][0-9][0-9]-*.json')):
            record=json.loads(path.read_text())
            if 'summary' not in record: continue
            s=record['summary'];actual.append(s);stat=stages[s['stage']]
            stat.update(trials=1,requests=s['offered'],request_errors=s['errors'],observer_errors=s['checkpoint_errors'])
            stat['input_tokens']+=s['total_input_tokens'];stat['requested_output_tokens']+=s['total_requested_output']
            stat['native_output_tokens']+=s['native_output_tokens'];stat['truncated_outputs']+=s['truncated_outputs']
            assert len(record['requests'])==s['offered']
            assert s['transfer_bytes']==sum(c['bytes'] for c in s['preparation'])
            if not s['policy'].endswith('+fluidkv'): assert s['transfer_bytes']==0
            for r in record['requests']:
                unique.add(r['native_index'])
                key=(s['window'],s['length'],r['native_index'])
                old=native_hashes.setdefault(key,r['input_sha256'])
                assert old==r['input_sha256'],'input changed between policies/stages/repeats'
                arrival_offset=r['offset']/s['rate_scale']
                assert abs(r['client_queue_seconds']-(r['dispatch_seconds']-arrival_offset))<1e-5
                assert abs(r['e2e_seconds']-(r['finished_seconds']-arrival_offset))<1e-5
                if s['output_cap']==0: assert r['requested_output']==r['output_tokens']
                else: assert r['requested_output']==min(r['output_tokens'],s['output_cap'])
                if not r['success']: continue
                assert abs(r['ttft_seconds']-(r['first_seconds']-arrival_offset))<1e-5
                assert r['engine']['meta_info']['prompt_tokens']==r['input_tokens']
                assert r['engine']['meta_info']['completion_tokens']==r['requested_output']
                if s['withdraw']: assert '3.27:' not in r['engine_url']
                if s['policy'].endswith('+fluidkv') and r['eligible']:
                    full_hits['eligible_successful']+=1
                    full_hits['full_history_hits']+=r['engine']['meta_info']['cached_tokens']>=s['length']
    wanted=Counter(tuple(s[k] for k in FIELDS) for s in planned)
    measured=Counter(tuple(s[k] for k in FIELDS) for s in actual)
    assert wanted==measured,{'missing':list((wanted-measured).elements()),'extra':list((measured-wanted).elements())}
    assert all(n==1 for n in measured.values()),'duplicate trial'
    assert all(len(v)==1 for v in core_hashes.values()),'core routing/migration changed between stages'
    progress=json.loads((BASE/'kimi-trace-TASK_PROGRESS.json').read_text())
    assert len(actual)==progress['expected_formal_trials']
    assert sum(s['offered'] for s in actual)==progress['expected_formal_requests']
    result=dict(trials=len(actual),requests=sum(s['offered'] for s in actual),unique_native_rows=len(unique),
        request_errors=sum(s['errors'] for s in actual),observer_errors=sum(s['checkpoint_errors'] for s in actual),
        stages={k:dict(v) for k,v in stages.items()},fluidkv_full_history_hits=dict(full_hits),
        verified_core_hashes={k:list(v)[0] for k,v in core_hashes.items()},
        checks=['exact predeclared trial multiset covered','no duplicate complete trials','native input and requested output lengths preserved',
            'arrival-based clock identities verified','no native-output truncation','identical paired input hashes',
            'source excluded from all withdrawal routing','launched source snapshots match hashes','core router and migration implementations unchanged'],
        excluded_debug=['kimi-trace-smoke','kimi-trace-screen-warm-calibration-pilot'],
        retained_setup_failure='kimi-trace-screen/027-screen-high-5760-p2c-r0-failure.json')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.output)
