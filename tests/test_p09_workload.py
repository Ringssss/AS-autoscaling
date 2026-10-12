import tempfile
from pathlib import Path

import numpy as np

from experiments.fluidkv.p09_make_workload import make


def test_concurrency_sweep_preserves_inputs_lengths_and_family_proportions():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)/'data'
        d=make(root,4,32768,[4,8,16])
        tokens=np.load(root/'cohort-tokens.npy')
        previous=[]
        for case in d['cohorts']:
            streams=case['streams']
            assert streams[:len(previous)]==previous
            previous=streams
            assert [sum(s['family']==f for s in streams) for f in range(4)]==[len(streams)//4]*4
            assert case['common_parent_tokens']==0
            for s in streams:
                parent,a,b=s['requests']
                extract=lambda r:tokens[r['token_offset']:r['token_offset']+r['input_tokens']]
                assert (a['input_tokens'],b['input_tokens'])==(32832,32896)
                assert a['output_tokens']==b['output_tokens']==128
                assert np.array_equal(extract(a)[:32768],extract(parent))
                assert np.array_equal(extract(b)[:32832],extract(a))
        first=[s['requests'][0] for s in previous[:4]]
        assert len({p['token_sha256'] for p in first})==4
