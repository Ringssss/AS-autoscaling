from experiments.weka_profile.analyze import union
from experiments.weka_profile.prepare_compare import heldout


def test_union_distinguishes_overlap_from_sum():
    assert union([])==0
    assert union([(0,10),(2,5),(8,12),(20,25)])==17
    assert union([(10,12),(0,10)])==12


def test_heldout_contains_no_previous_evaluated_streams_or_truncation():
    rows,metadata=heldout()
    assert len(rows)==36
    assert set(metadata['streams']).isdisjoint(metadata['excluded_previous'])
    for i in metadata['streams']:
        group=[r for r in rows if r['stream_index']==i]
        assert {r['turn'] for r in group}=={0,1,2}
        assert sum(r['output_tokens'] for r in group)<=900
        assert all(r['input_tokens']+r['output_tokens']<=131072 for r in group)
