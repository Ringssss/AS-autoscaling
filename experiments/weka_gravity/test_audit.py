import csv
import json
from argparse import Namespace

from experiments.weka_gravity.audit import run


def test_causal_model_separation_local_ids_and_expiry(tmp_path):
    def req(t, model='a', duration=1):
        return dict(t=t, model=model, api_time=duration, hash_ids=list(range(512)),
                    **{'in':32768, 'out':16})
    source=tmp_path/'input.jsonl'
    traces=[dict(id='one', block_size=64, hash_id_scope='local', requests=[
        req(0,duration=10), req(1), req(3,model='b'), req(4), req(700),
        dict(t=2.5, agent_id='child', requests=[req(2.5,duration=.25)])]),
        dict(id='two', block_size=64, hash_id_scope='local', requests=[req(0)])]
    source.write_text('\n'.join(json.dumps(t) for t in traces)+'\n')
    output=tmp_path/'out'
    run(Namespace(trace=source,output=output))
    with (output/'requests.csv').open() as f: rows=list(csv.DictReader(f))
    indexed={(r['conversation'],float(r['t'])):r for r in rows}
    assert int(indexed['one',1]['hit'])==0  # in-flight input must not count
    assert int(indexed['one',2.5]['hit'])==32768  # completed sibling can count
    assert int(indexed['one',3]['hit'])==0  # different model cannot share KV
    assert int(indexed['one',4]['hit'])==32768
    assert int(indexed['one',700]['hit'])==32768
    assert int(indexed['one',700]['hit300'])==0
    assert int(indexed['two',0]['hit'])==0  # local IDs cannot cross sessions
