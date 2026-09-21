import pytest

from test_freetoken_elastic_report import fixture
from experiments.freetoken_elastic.report_preservation import audit_preservation_trial, expected_matrix


def preservation_fixture():
    trial, spec, pair, config = fixture()
    spec.update(condition='kv-resize', policy='preserve');trial.update(spec)
    config['verify_bytes'] = False
    pair.update(expected_parent_tokens=128, expected_child_tokens=140)
    by_label = {o['label']:o for o in trial['operations']}
    for label in ['initial-status','before-status','after-first-status','final-status']:
        geo=by_label[label]['body']['geometry'];geo.update(moe_cache_size=20)
        geo['unit_bytes']=dict(kv_per_token=20480,mamba_per_slot=64389120)
        if label in ['after-first-status','final-status']:geo['num_pages']=128
    for label in ['cold-parent','first-child','repeated-child']:
        by_label[label]['usage']['prompt_tokens']=128 if label=='cold-parent' else 140
    by_label['first-child']['cached_tokens']=64;trial['metrics']['first_child_cached_tokens']=64
    for i,label in enumerate(['reset-intermediate','reset-initial','measured-rebuild']):
        op=by_label.get(label)
        if op is None:
            op=dict(label=label,success=True,elapsed_seconds=.5);trial['operations'].append(op)
        measured=label=='measured-rebuild';mode='preserve' if measured else 'drop'
        op['payload']=dict(mode='if_idle',num_pages=128 if measured else 256)
        op['experimental_policy']=dict(id=str(i),mode=mode)
        hook=dict(id=str(i),mode=mode,success=True,geometry=dict(num_pages=op['payload']['num_pages']),start=1.,finish=1.5,total_seconds=.5,original_rebuild_seconds=.3)
        if measured:
            nbytes=64*(20480+4)+64389120
            hook.update(snapshots_before=[dict(length=64,slot=1),dict(length=127,slot=2)],
                        snapshots_after=[dict(length=64,slot=3)],
                        export=dict(boundary=64,bytes=nbytes,host_margin_bytes=1024,host_budget_bytes=1024**3,export_seconds=.1),
                        restore=dict(boundary=64,bytes=nbytes,restored=True,destination_slot=3,restore_seconds=.05))
        op['hook_records']=[hook]
    return trial,spec,pair,config


def test_preservation_byte_math_consumption_and_hook_identity():
    trial,spec,pair,config=preservation_fixture()
    result=audit_preservation_trial(trial,spec,pair,config)
    assert result['audit_pass'],result['errors']
    assert result['byte_accounting']['token_bytes']==256


@pytest.mark.parametrize('defect,expected',[('bytes','byte mismatch'),('cache','exactly consumed'),('id','ID mismatch'),('selection','longest existing'),('timing','durations exceed')])
def test_preservation_audit_catches_invalid_mechanism(defect,expected):
    trial,spec,pair,config=preservation_fixture()
    ops={o['label']:o for o in trial['operations']};hook=ops['measured-rebuild']['hook_records'][0]
    if defect=='bytes':hook['export']['bytes']+=4
    elif defect=='cache':ops['first-child']['cached_tokens']=0
    elif defect=='id':hook['id']='wrong'
    elif defect=='selection':hook['snapshots_before'].append(dict(length=128,slot=4))
    else:hook['restore']['restore_seconds']=1.
    result=audit_preservation_trial(trial,spec,pair,config)
    assert not result['audit_pass'] and any(expected in e for e in result['errors'])


def test_policy_matrix_unique_and_balanced():
    c=dict(config=dict(seed=1516,repetitions=3,conditions=['kv-resize','mamba-resize']),manifest=dict(pairs=[dict(id='short'),dict(id='long')]))
    rows=expected_matrix(c)
    assert len(rows)==24 and [r['index'] for r in rows]==list(range(24))
    assert rows==expected_matrix(c)
    for n in range(0,24,2):assert {r['policy'] for r in rows[n:n+2]}=={'drop','preserve'}
