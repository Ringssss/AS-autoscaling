from copy import deepcopy
import json
from experiments.agentic_profile.report_state_work import evaluate_row, union_overlap, telemetry_summary


def generation(inputs, outputs, start, first, finish, ids=None):
    return {'success':True,'input_tokens':inputs,'requested_output_tokens':outputs,
            'meta_info':{'prompt_tokens':inputs,'completion_tokens':outputs,'cached_tokens':8192},
            'output_ids':ids or list(range(outputs)), 'start':start,'first':first,'finish':finish,
            'ttft_seconds':first-start,'e2e_seconds':finish-start,
            'mean_decode_seconds_per_token':(finish-first)/(outputs-1) if outputs>1 else None}


def fixture(method='copy'):
    config={'foreground_input':10,'foreground_output':3,'foreground_hashes':['fg'],
            'prefix_tokens':8192,'preparations':1,'parent_hashes':['parent']}
    fg=generation(10,3,0,1,3);fg['input_sha256']='fg'
    prep={'start':1,'finish':2,'success':True,'prefix_tokens':8192,'input_sha256':'parent','transfer_bytes':100}
    if method=='replay':prep['generation']=generation(8192,1,1,1.5,2)
    row={'index':0,'method':method,'mode':'concurrent','foreground_target':'source','concurrency':1,'repetition':0,
         'foreground':[fg],'preparations':[prep],'validation':[{'source':generation(8256,8,4,4.1,5),
            'destination':generation(8256,8,5,5.1,6),'destination_hit':True,'cached_continuation_equal':True}],
         'execution_success':True,'mechanism_valid':True,'cleanup_confirmed':True,'start':0,'finish':3,
         'measured_makespan_seconds':3,'charged_gpu_seconds':6,'foreground_ready_to_first_p50_seconds':1,
         'foreground_ready_to_finish_p50_seconds':3,'foreground_ttft_p50_seconds':1,'foreground_e2e_p50_seconds':3,
         'foreground_decode_p50_seconds_per_token':1,'preparation_total_seconds':1,'request_lifetime_overlap_seconds':1}
    return row,config


def test_independent_overlap_unions_concurrent_requests():
    assert union_overlap([{'start':1,'finish':3}], [{'start':0,'finish':2},{'start':1,'finish':4}])==2


def test_copy_requires_consumption_and_cached_parity():
    row,config=fixture()
    assert evaluate_row(row,config)['errors']==[]
    row['validation'][0]['destination']['output_ids']=[99]*8
    row['validation'][0]['cached_continuation_equal']=False
    row['mechanism_valid']=False
    result=evaluate_row(row,config)
    assert result['mechanism_valid'] is False
    assert result['errors']==[]


def test_replay_quality_separate_from_execution():
    row,config=fixture('replay')
    row['validation'][0]['destination']['output_ids']=[99]*8
    row['validation'][0]['cached_continuation_equal']=False
    result=evaluate_row(row,config)
    assert result['mechanism_valid'] is True
    assert result['replay_quality_gate_passed'] is False
    assert result['errors']==[]


def test_two_gpu_charge_and_hidden_serial_wait_audited():
    row,config=fixture()
    row['charged_gpu_seconds']=3
    row['foreground_ready_to_first_p50_seconds']=0.1
    errors=evaluate_row(row,config)['errors']
    assert any('charged_gpu_seconds' in e for e in errors)
    assert any('foreground_ready_to_first' in e for e in errors)


def test_memory_headroom_uses_same_sample_and_cpu_scope(tmp_path):
    records = [{'event':'metadata','root_pids':[1,2]},
               {'event':'sample','monotonic_seconds':1,'sampling_wall_seconds':.1,
                'gpu':{'devices':[{'index':0,'uuid':'gpu0','memory_used_mib':80,'memory_total_mib':100}]}},
               {'event':'sample','monotonic_seconds':2,'sampling_wall_seconds':.2,
                'gpu':{'devices':[{'index':0,'uuid':'gpu0','memory_used_mib':60,'memory_total_mib':70}]}},
               {'event':'summary','sampled_cpu_seconds':123,'error':None}]
    (tmp_path/'telemetry.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    result = telemetry_summary(tmp_path)
    assert result['gpu_memory']['0']['minimum_free_mib'] == 10
    assert result['sampled_cpu_seconds'] == 123
    assert result['complete']
    assert 'including excluded' in result['scope']
