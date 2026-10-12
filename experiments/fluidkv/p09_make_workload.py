"""Fixed-length concurrency sweeps with explicit synthetic prefix families."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def make(out, families, prefix_tokens, levels, seed=20261011):
    out.mkdir(parents=True,exist_ok=False)
    assert prefix_tokens%64==0 and min(levels)>=families
    parts=[]
    offset=0
    def record(ids,hashes,location):
        nonlocal offset
        result=dict(location=location,input_tokens=len(ids),output_tokens=128,
            native_output_tokens=128,token_offset=offset,hash_ids=hashes,
            token_sha256=hashlib.sha256(ids.tobytes()).hexdigest())
        parts.append(ids);offset+=len(ids)
        return result
    parents=[]
    prefixes=[]
    for family in range(families):
        prefix=np.random.default_rng(seed+family).integers(1000,240000,prefix_tokens,dtype=np.int32)
        hashes=[family*1000000+i for i in range(prefix_tokens//64)]
        prefixes.append(prefix)
        parents.append(record(prefix,hashes,f'family-{family}/parent'))
    streams=[]
    for client in range(max(levels)):
        family=client%families
        private=np.random.default_rng(seed+10000+client).integers(1000,240000,128,dtype=np.int32)
        requests=[parents[family]]
        for turn in (1,2):
            ids=np.concatenate([prefixes[family],private[:64*turn]])
            hashes=parents[family]['hash_ids']+[10000000+2*client+i for i in range(turn)]
            requests.append(record(ids,hashes,f'client-{client}/turn-{turn}'))
        streams.append(dict(key=f'family-{family}/client-{client}',family=family,
            requests=requests,first_offset=0.,gaps=[.1]))
    cases=[dict(name=f'f{families}-c{n:03d}',conversation=f'synthetic-f{families}-seed{seed}',
        model='Qwen3.6-35B-A3B',clients=n,prefix_families=families,
        common_parent_tokens=prefix_tokens if families==1 else 0,
        preparation_tokens=prefix_tokens,prepared_family=0,
        prepared_family_fraction=1/families,streams=streams[:n]) for n in levels]
    np.save(out/'cohort-tokens.npy',np.concatenate(parts))
    d=dict(seed=seed,cohorts=cases,levels=levels,prefix_tokens=prefix_tokens,
        prefix_families=families,output_tokens=128,consumer_turns=2,gap_between_turns=.1,
        scope='Synthetic concurrency stress, not native AgentX concurrency or semantic agents. '
        'Fixed prefix length, suffix growth, output length, prefix family count and family proportions '
        'within each sweep. Only number of simultaneously active closed-loop clients changes. '
        'Each client has one outstanding request; two turns. All initial parent states on worker 0. '
        'Preparation copies/recomputes family 0 only; in four-family sweep this serves 25% of clients. '
        'Consumer arrival gap is separately declared; cache/engine capacity remains fixed.')
    (out/'cohorts.json').write_text(json.dumps(d,indent=2)+'\n')
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--families',type=int,required=True)
    p.add_argument('--prefix-tokens',type=int,required=True)
    p.add_argument('--levels',nargs='+',type=int,default=[4,8,16,32,64,128])
    a=p.parse_args()
    d=make(a.out,a.families,a.prefix_tokens,a.levels)
    print(json.dumps({k:v for k,v in d.items() if k not in ('cohorts','scope')},indent=2))
