"""Audit complete campaign counts and retained-wave transfer invariants."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent/'results/advantage-boundary'
EXPECTED={'screen-window-0':24,'screen-window-0.5':24,'screen-window-5':24,
          'heldout-persistence':90,'heldout-low-concurrency':35,'screen-window-10':14,
          'heldout-source-withdrawal':30}

def main():
    result={'trials':0,'requests':0,'errors':0,'observation_errors':0,'retained_fluidkv_transfers':0,'campaigns':{}}
    for name,count in EXPECTED.items():
        root=OUT/name
        runs=json.loads((root/'summary.json').read_text())['runs']
        assert len(runs)==count,(name,len(runs),count)
        records=[]
        for p in root.glob('*.json'):
            data=json.loads(p.read_text())
            if isinstance(data,dict) and 'requests' in data:
                s=data['summary'];r=data['requests']
                assert len(r)==s['completed']==s['offered_total']
                assert s['errors']==sum(not x['success'] for x in r)
                assert len({x['request_id'] for x in r})==len(r)
                assert s['total_transfer_bytes']==sum(x['bytes'] for x in s['preparation'])
                if name=='heldout-source-withdrawal':
                    assert s['source_withdrawn_from_routing']
                    assert len(s['eligible_engine_urls'])==2
                    assert all(x['engine_url'] in s['eligible_engine_urls'] and x['engine_url']!='http://12.11.3.27:31600' for x in r)
                records.extend(r)
        assert len(records)==sum(s['offered_total'] for s in runs)
        result['campaigns'][name]={'trials':count,'requests':len(records),'errors':sum(s['errors'] for s in runs)}
        result['trials']+=count;result['requests']+=len(records)
        result['errors']+=sum(s['errors'] for s in runs)
        result['observation_errors']+=sum('error' in o for s in runs for o in s['checkpoint_observations'])
        result['retained_fluidkv_transfers']+=sum(bool(s['total_transfer_bytes']) for s in runs if s.get('wave',0)>0 and s['policy'].endswith('+fluidkv'))
    assert result['retained_fluidkv_transfers']==0
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
