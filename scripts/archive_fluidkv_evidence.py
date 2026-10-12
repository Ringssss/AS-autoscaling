#!/usr/bin/env python3
"""Package the pinned public WekaTrace and FluidKV research records separately.

Requires Python 3 and the zstd executable. Never includes model checkpoints,
credential stores, chat histories, or the unrelated FluidCKPT project.
"""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

TRACE_SHA='29b6a19e751ff5230771519aab755f80a0f43a4ba9cf96b72d3a6a437ec99276'
TRACE_BYTES=1847151435
REVISION='23f152f6f0f9399a85901b89a6458def0ef16729'
SOURCE='https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126'
TOP_KEYS={'id','models','block_size','hash_id_scope','requests'}
REQUEST_KEYS={'t','model','in','out','hash_ids','api_time','type','ttft','think_time',
    'agent_id','subagent_type','duration_ms','total_tokens','tool_use_count','status','requests','models'}
SECRET=re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|'
    rb'hf_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''):h.update(part)
    return h.hexdigest()


def trace_audit(path):
    counts=Counter();fields=Counter();digest=hashlib.sha256()
    def request(row):
        unknown=set(row)-REQUEST_KEYS
        if unknown:raise ValueError(f'unreviewed request fields: {sorted(unknown)}')
        fields.update(row.keys())
        if 'requests' in row:
            counts['subagent_groups']+=1
            for child in row['requests']:
                counts['subagent_requests']+=1;request(child)
        else:
            assert len(row['hash_ids'])*64==row['in']
            assert all(type(x) is int for x in row['hash_ids'])
            counts['model_requests']+=1
            counts['input_tokens_block_estimate']+=row['in']
            counts['output_tokens']+=row['out']
    with path.open('rb') as f:
        for raw in f:
            digest.update(raw);row=json.loads(raw)
            unknown=set(row)-TOP_KEYS
            if unknown:raise ValueError(f'unreviewed conversation fields: {sorted(unknown)}')
            assert row['block_size']==64 and row['hash_id_scope']=='local'
            counts['conversations']+=1
            for r in row['requests']:
                if 'requests' not in r:counts['main_requests']+=1
                request(r)
    assert path.stat().st_size==TRACE_BYTES and digest.hexdigest()==TRACE_SHA
    assert counts['conversations']==393 and counts['model_requests']==98827
    return dict(counts=counts,request_fields=fields,trace_sha256=digest.hexdigest(),
        schema='Block hashes, lengths, relative times and agent/model metadata; no prompt/response text fields.',
        checks='All conversations checked; unknown schema fields rejected; upstream byte identity checked.')


def pack(out,name,files,generated,provenance):
    """Deterministic tar members; hashes refer to the bytes actually archived."""
    archive=out/(name+'.tar.zst')
    if archive.exists():raise FileExistsError(archive)
    entries=[]
    for rel,path in sorted(files.items()):
        if path.is_symlink() or not path.is_file():raise ValueError(f'not a regular file: {path}')
        if path.suffix in {'.py','.rs','.json','.jsonl','.md','.txt','.sh','.log','.yaml','.yml','.toml'}:
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):
                    if SECRET.search(block):raise ValueError(f'credential pattern in {rel}; review required')
        entries.append(dict(path=rel,bytes=path.stat().st_size,sha256=sha(path)))
    for rel,data in sorted(generated.items()):
        entries.append(dict(path=rel,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
    manifest=dict(provenance=provenance,files=entries)
    generated=dict(generated)
    generated['MANIFEST.json']=(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n').encode()
    generated['SHA256SUMS']=''.join(f'{e["sha256"]}  {e["path"]}\n' for e in entries).encode()
    with archive.open('wb') as dest:
        proc=subprocess.Popen(['zstd','-q','-T4','-6','-c'],stdin=subprocess.PIPE,stdout=dest)
        try:
            with tarfile.open(fileobj=proc.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for rel in sorted(set(files)|set(generated)):
                    info=tarfile.TarInfo(name+'/'+rel);info.mode=0o644;info.mtime=0
                    info.uid=info.gid=0;info.uname=info.gname=''
                    if rel in generated:
                        blob=generated[rel];info.size=len(blob);tar.addfile(info,io.BytesIO(blob))
                    else:
                        info.size=files[rel].stat().st_size
                        with files[rel].open('rb') as f:tar.addfile(info,f)
            proc.stdin.close()
            if proc.wait():raise RuntimeError('zstd failed')
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
    # Stream-decompress every archived file and compare it to the manifest.
    proc=subprocess.Popen(['zstd','-q','-d','-c',str(archive)],stdout=subprocess.PIPE)
    expected={e['path']:e for e in entries};seen=set()
    with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
        for member in tar:
            rel=member.name[len(name)+1:]
            assert member.isfile() and member.name.startswith(name+'/') and '..' not in Path(rel).parts
            h=hashlib.sha256();f=tar.extractfile(member)
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
            if rel in expected:
                assert h.hexdigest()==expected[rel]['sha256'] and member.size==expected[rel]['bytes']
                seen.add(rel)
    proc.stdout.close();assert proc.wait()==0 and seen==set(expected)
    result=dict(asset=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),
        uncompressed_file_bytes=sum(e['bytes'] for e in entries),file_count=len(entries)+2,
        provenance=provenance,verified_all_member_hashes=True)
    (out/(name+'.manifest.json')).write_text(json.dumps(dict(result,files=entries),indent=2,ensure_ascii=False)+'\n')
    (out/(name+'.sha256')).write_text(f'{result["sha256"]}  {archive.name}\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--trace-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--apache-license',type=Path,default=Path('/usr/share/common-licenses/Apache-2.0'))
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    audit=trace_audit(a.trace_dir/'traces.jsonl')
    (a.out/'agentx-schema-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print('TRACE_AUDIT',json.dumps(audit['counts']),flush=True)
    files={name:a.trace_dir/name for name in ['README.md','stats.txt','.gitattributes','traces.jsonl']}
    files.update({str(x.relative_to(a.trace_dir)):x for x in (a.trace_dir/'plots').glob('*.png')})
    notes=('This is a byte-preserving mirror of the public SemiAnalysis WekaTrace dataset, '
        'called AgentX/Weka trace in FluidKV experiment notes. It is not an original dataset '
        'collected by AS-autoscaling. Original README, statistics, plots and trace bytes are preserved.\n\n'
        f'Source: {SOURCE}\nRevision: {REVISION}\nLicense: Apache-2.0, as declared by the upstream dataset card.\n\n'
        'The LICENSE, MANIFEST.json, SHA256SUMS, SCHEMA-AUDIT.json and this attribution file '
        'were added for redistribution; upstream files were not modified.\n'
        'Hash IDs are conversation-local, not global token IDs. Input lengths are 64-token-block '
        'estimates. Timelines do not establish cross-conversation arrivals or worker placement.\n')
    trace=pack(a.out,'agentx-weka-trace-062126',files,
        {'LICENSE':a.apache_license.read_bytes(),'ATTRIBUTION.md':notes.encode(),
         'SCHEMA-AUDIT.json':(json.dumps(audit,indent=2)+'\n').encode()},
        dict(source=SOURCE,revision=REVISION,license='Apache-2.0',original_trace_sha256=TRACE_SHA))
    roots=['results/skew-readiness-20261009','results/routing-space-20261010',
        'results/native-relocation-space-20261010','results/state-preparation-20261011',
        'results/concurrency-sweep-20261011','results/six-worker-quick-20261012',
        'results/archive-20260930','experiments/fluidkv/results','experiments/hygon_runtime/results',
        'experiments/weka_baselines/results','experiments/weka_baselines/oracle/results',
        'experiments/weka_baselines/data',
        'experiments/weka_gravity/results','experiments/weka_profile/results',
        'experiments/multinode/results','experiments/multinode/archive','docs/review']
    files={}
    candidates=[x for folder in roots for x in (a.repo/folder).rglob('*')]
    candidates += list((a.repo/'results').glob('*'))
    for x in candidates:
        rel=x.relative_to(a.repo)
        if not x.is_file() or x.is_symlink() or '__pycache__' in rel.parts:continue
        if x.suffix in {'.pyc','.db','.sqlite','.sqlite3'} or '.db-' in x.name:continue
        if 'ckpt' in str(rel).lower():continue
        files[str(rel)]=x
    notes=('FluidKV historical research records, 2026-09-28 through 2026-10-12, with earlier '
        'supporting multinode records where already grouped with the experiments.\n\n'
        'Original raw results, failures, logs, inputs generated from public trace hashes, '
        'per-run source snapshots and historical interpretations are preserved. '
        'Historical positive claims are NOT all currently valid.\n\n'
        'Use docs/FLUIDKV-TECHNICAL-ARCHIVE.zh.md in the GitHub repository as the current index. '
        'The final October 12 comparison is results/six-worker-quick-20261012/v2/FINAL-REPORT.zh.md. '
        'The earlier 512-token-rounded comparison, including the apparent 18.4% gain, is withdrawn; '
        'see results/concurrency-sweep-20261011/v2/pause-reason.json. '
        'October 11 state-preparation results also use that old observation adapter.\n\n'
        'Extract with --strip-components=1 into an isolated repository checkout to restore original '
        'relative paths. Lab addresses, local paths and container commands are historical settings; '
        'review them before any GPU launch. No model weights, private chat histories or FluidCKPT '
        'project files are included.\n')
    third_party=('Derived WekaTrace inputs retain attribution to SemiAnalysis and the Apache-2.0 '
        'license declared at '+SOURCE+'. Original SGLang/Hygon headers in runtime snapshots '
        'are preserved. Added research records are from AS-autoscaling; this notice does not '
        'relicense unrelated repository content.\n')
    results=pack(a.out,'fluidkv-results-20260928-20261012',files,
        {'ARCHIVE-README.md':notes.encode(),'THIRD-PARTY-NOTICE.md':third_party.encode(),
         'THIRD-PARTY-LICENSES/Apache-2.0.txt':a.apache_license.read_bytes()},
        dict(repository='https://github.com/Ringssss/AS-autoscaling',
            source_base_commit='2e96af93f091808ed9e31f04eed800d0285c97b4',
            scope='Historical research archive, including superseded and failed runs; not all valid performance evidence.',
            derived_trace_source=SOURCE,derived_trace_license='Apache-2.0'))
    figures={str(x.relative_to(a.repo)):x for x in (a.repo/'paper/figs').rglob('*')
        if x.is_file() and x.suffix in {'.pdf','.png','.svg'}}
    figure_result=pack(a.out,'fluidkv-figures-20260930',figures,
        {'ARCHIVE-README.md':b'Historical FluidKV figures; see the technical archive index for evidence status.\n',
         'THIRD-PARTY-NOTICE.md':third_party.encode(),
         'THIRD-PARTY-LICENSES/Apache-2.0.txt':a.apache_license.read_bytes()},
        dict(repository='https://github.com/Ringssss/AS-autoscaling',scope='Historical derived figures; not current performance claims.'))
    (a.out/'assets.json').write_text(json.dumps([trace,results,figure_result],indent=2)+'\n')


if __name__=='__main__':main()
