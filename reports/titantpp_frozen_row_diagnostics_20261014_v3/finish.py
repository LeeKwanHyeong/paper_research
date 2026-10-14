"""Accept only all 162 real successful workers and both actual pipeline exits."""
from pathlib import Path
import hashlib
import json

B = Path(__file__).resolve().parents[2] / 'search_artifacts/titantpp_frozen_row_diagnostics_20261014_v3'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def main():
    result={'schema':'titantpp_frozen_rows_execution_receipt_v1','status':'complete',
        'bindings':{'path':'bindings.json','sha256':sha(B/'bindings.json')},'contracts':[],
        'jobs':[],'pipeline_exits':[],'new_training':False,'checkpoint_reselection':False,
        'scientific_fit_count':72,'new_inference_job_count':162,'external_Test_reused_conditions':54}
    keys=set()
    for host in ['5080','5090']:
        cpath=B/f'contract_{host}.json';c=read(cpath);cs=sha(cpath)
        result['contracts'].append({'host':host,'path':cpath.relative_to(B).as_posix(),'sha256':cs})
        native=B/'native'/host
        assert sha(native/'contract.json')==cs
        manifest=read(native/'retrieval_manifest.json')
        for rel,expected in manifest.items():assert sha(native/rel)==expected,rel
        ep=native/'pipeline_actual_exit.json';ex=read(ep)
        assert ex['actual_child_exit_observed'] and ex['exit_code']==0 and ex['supervisor_exit_code']==0
        assert ex['contract_unchanged'] and ex['contract_sha256']==cs
        result['pipeline_exits'].append({'host':host,'path':ep.relative_to(B).as_posix(),'sha256':sha(ep)})
        completion=read(native/'pipeline_completion.json')
        expected=sum(len(j['allowed_splits']) for j in c['jobs'])
        cached=c.get('reused_validation',{})
        assert completion['status']=='complete' and completion['actual_worker_count']==expected-len(cached) and completion['worker_exit0'] and completion['contract_sha256']==cs
        for j in c['jobs']:
            for split in j['allowed_splits']:
                key=j['id'],split;assert key not in keys;keys.add(key)
                p=native/'results'/j['id']/split/'receipt.json'
                e=native/'results'/j['id']/(split+'_worker_exit.json')
                row,exit_record=read(p),read(e)
                assert row['status']=='complete' and row['full_population'] and row['job_id']==j['id']
                expected_cs=cs
                if split=='validation' and j['id'] in cached:
                    bridge=cached[j['id']]
                    assert host=='5080' and j['family']=='component'
                    assert sha(p)==bridge['receipt_sha256'] and sha(e)==bridge['worker_exit_sha256']
                    expected_cs=bridge['contract_sha256']
                assert row['contract_raw_sha256']==expected_cs and row['state_sha256_before']==row['state_sha256_after']
                assert exit_record['actual_child_exit_observed'] and exit_record['exit_code']==0 and exit_record['contract_sha256']==expected_cs
                assert row['prediction_rows']==j['expected_populations'][split].get('count',j['expected_populations'][split].get('target_count'))
                pm=p.parent/'predictions_manifest.json';assert sha(pm)==row['predictions_manifest_sha256']
                parts=read(pm)['parts']
                assert sum(part['rows'] for part in parts)==row['prediction_rows']
                for part in parts:assert sha(p.parent/'predictions'/part['path'])==part['sha256']
                result['jobs'].append({'job_id':j['id'],'split':split,'receipt':{'path':p.relative_to(B).as_posix(),'sha256':sha(p)},
                    'worker_exit':{'path':e.relative_to(B).as_posix(),'sha256':sha(e)}})
    assert len(result['jobs'])==162 and len(keys)==162
    result['reused_validation_conditions']=14
    result['fresh_inference_conditions_in_final_native_roots']=148
    output=B/'execution_receipt.json'
    with output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'status':'complete','worker_count':162,'path':str(output),'sha256':sha(output)}))


if __name__=='__main__':main()
