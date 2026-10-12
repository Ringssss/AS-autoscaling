"""Supplemental identity/length/endpoint checks on raw redacted Kimi rows."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import tarfile
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/kimi-opportunity'
ARCHIVE=ROOT/'data/kimi-k25-workers-5p3d-all-decodes_finish_fullids_redacted_2026-04-07.tar.gz'
patterns={k:re.compile(r'\b'+k+r'=([^,)]+)') for k in ('custom_labels','external_trace_header','lora_id','routed_dp_rank','disagg_prefill_dp_rank')}
patterns['custom_params']=re.compile(r"'custom_params': ([^,}]+)")
PROMPT=re.compile(r"'prompt_tokens': (\d+)")
HOST=re.compile(r"bootstrap_host='([^']+)'")
def main():
    arrays=np.load(OUT/'native_numeric_fields.npz')
    counts={k:Counter() for k in patterns};values={k:Counter() for k in patterns}
    host_rows=[];prompt=[];row_count=0
    with tarfile.open(ARCHIVE,'r|gz') as tar:
        for member in tar:
            if not member.isfile() or not re.fullmatch(r'[0-2][0-9]\.redacted\.log',Path(member.name).name):continue
            for raw in tar.extractfile(member):
                line=raw.decode('utf-8')
                if not line.startswith('Finish:'):continue
                for k,pat in patterns.items():
                    m=pat.search(line);counts[k]['present']+=bool(m)
                    populated=bool(m and m.group(1).strip() not in ('None',"''",'""','[]','{}'))
                    counts[k]['nonnull']+=populated
                    if populated:values[k][hashlib.sha256(m.group(1).encode()).hexdigest()[:16]]+=1
                m=PROMPT.search(line);prompt.append(int(m.group(1)) if m else -1)
                h=HOST.search(line);host_rows.append(h.group(1) if h else '<missing>')
                row_count+=1
    assert row_count==len(arrays['input_tokens'])
    prompt=np.array(prompt);x=arrays['input_tokens'];c=arrays['cached_tokens']
    invalid=(c<0)|(c>x)
    classes={h:i for i,h in enumerate(sorted(set(host_rows)))}
    host_ids=np.array([classes[h] for h in host_rows]);arr=arrays['arrival'];t0=arr.min();bins=((arr-t0)//300).astype(int)
    activity=[]
    for b in range(int(bins.max())+1):
        mask=bins==b
        activity.append({'five_minute_bin':b,'logged_requests':int(mask.sum()),'observed_bootstrap_labels':np.unique(host_ids[mask]).tolist()})
    report={'rows':row_count,'extra_fields':{k:dict(counts[k],distinct_nonnull_values=len(values[k]),most_common_nonnull_fingerprint_counts=values[k].most_common(3)) for k in counts},
            'prompt_tokens_missing':int((prompt<0).sum()),'prompt_tokens_disagree_with_input_ids_len':int((prompt!=x).sum()),
            'invalid_cache_rows':int(invalid.sum()),'invalid_cache_rows_with_valid_meta_prompt':int((invalid&(prompt>=c)).sum()),
            'invalid_cache_meta_prompt_quantiles':np.percentile(prompt[invalid],[0,50,100]).tolist(),
            'bootstrap_label_count':len(classes),'missing_bootstrap_labels':host_rows.count('<missing>'),
            'bootstrap_semantics':'An observed disaggregation bootstrap address is not a proved serving engine, state owner, or membership event. Absence of requests in a bin does not prove shutdown.',
            'bootstrap_activity':activity}
    (OUT/'metadata-crosscheck.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(OUT/'metadata_fields.npz',prompt_tokens=prompt,bootstrap_label=host_ids)
    print(json.dumps({k:v for k,v in report.items() if k!='bootstrap_activity'},indent=2))

if __name__=='__main__':
    main()
