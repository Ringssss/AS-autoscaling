import copy
import json

import pytest

from experiments.multinode.summarize_highbs import summarize


def trial(policy,repeat=0):
    return {'summary':{'prefix_tokens':32768,'offered_per_engine':1,'regime':'single-hot','repeat':repeat,
                      'policy':policy,'offered_total':3,'completed':3,'errors':0,
                      'ttft_p50_p95_p99':[2,3,4],'requests_per_second':1,
                      'routing_cpu_seconds':.1,'preparation_seconds':.2,'preparation_exposed_seconds':0,
                      'total_transfer_bytes':100,'slo_attainment':1,'actual_cached_tokens':96,'prompt_tokens':192},
            'requests':[{'request_id':i,'input_sha256':f'input-{i}','success':True,'engine':{'output_ids':[i,42]}} for i in range(3)]}


def test_failures_and_token_mismatches_are_not_hidden_by_latency_summary(tmp_path):
    baseline,prepared=trial('dualmap'),trial('dualmap+fluidkv')
    baseline['requests'][2].update(success=False,engine=None)
    baseline['summary']['errors']=1
    prepared['requests'][1]['engine']['output_ids']=[999,42]
    for name,data in [('a',baseline),('b',prepared)]:
        (tmp_path/f'{name}.json').write_text(json.dumps(data))
    result=summarize(tmp_path)
    assert result['strict_output_equivalence'] is False
    pair=result['pairs'][0]
    assert pair['output_compared']==2
    assert pair['exact_output_matches']==1
    assert pair['different_request_ids']==[1]
    assert result['summaries'][0]['baseline_errors']==1
    assert result['summaries'][0]['augmented_errors']==0


def test_different_workload_cannot_be_reported_as_a_paired_speedup(tmp_path):
    baseline,prepared=trial('cacheroute'),trial('cacheroute+fluidkv')
    prepared['requests'][0]['input_sha256']='different-prompt'
    (tmp_path/'a.json').write_text(json.dumps(baseline))
    (tmp_path/'b.json').write_text(json.dumps(prepared))
    with pytest.raises(ValueError,match='different input'):
        summarize(tmp_path)


def test_same_policy_repeat_variation_is_preserved_as_a_numerical_control(tmp_path):
    a=trial('dualmap',0)
    b=copy.deepcopy(a);b['summary']['repeat']=1
    b['requests'][0]['engine']['output_ids']=[123]
    (tmp_path/'a.json').write_text(json.dumps(a))
    (tmp_path/'b.json').write_text(json.dumps(b))
    control=summarize(tmp_path)['within_policy_repeat_output_checks'][0]
    assert control['compared']==3 and control['exact_matches']==2
    assert summarize(tmp_path)['strict_output_equivalence'] is False
