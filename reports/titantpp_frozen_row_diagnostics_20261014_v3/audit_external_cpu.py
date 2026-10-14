"""Audit all external imports, checkpoint routes and strict CPU loads, no forward."""
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/'search_artifacts/titantpp_frozen_row_diagnostics_20261014_v3'


def main():
    jobs=[j for j in json.loads((B/'bindings.json').read_text())['jobs'] if j['family']=='external']
    if len(sys.argv)==1:
        groups=sorted({j['legacy_registry_row']['evaluator_source_bundle'] for j in jobs})
        rows=[]
        for group in groups:
            x=subprocess.run([sys.executable,str(Path(__file__)),group],capture_output=True,text=True)
            if x.returncode:
                print(x.stderr[-6000:]);raise SystemExit(x.returncode)
            rows.extend(json.loads(x.stdout)['rows'])
        assert len(rows)==54
        with (B/'external_cpu_import_and_load_audit.json').open('x') as f:
            json.dump({'status':'complete','model_count':54,'model_forward_calls':0,'rows':rows},f,indent=2)
        print(json.dumps({'status':'complete','CPU_strict_selected_loads':54,'model_forward_calls':0}))
        return
    spec=importlib.util.spec_from_file_location('row_adapter',ROOT/'paper/scripts/evaluate_titantpp_frozen_rows.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    group=sys.argv[1];subset=[j for j in jobs if j['legacy_registry_row']['evaluator_source_bundle']==group]
    first=dict(subset[0]);first['parent_root']=str(B/'external_parent');s=m.resolve_job(first)
    m.verify_science(s);m.isolate(s['source_root'])
    import torch
    factory=importlib.import_module('models.TPPs.CountAwareFactory')
    runner=importlib.import_module('simple_lab_test.search.common.runner')
    core=importlib.import_module('paper.scripts.count_aware_tpp_backbone.core')
    loader=importlib.import_module('paper.scripts.run_taxi_quantity_interface_ablation')
    rows=[]
    for original in subset:
        j=dict(original);j['parent_root']=str(B/'external_parent');s=m.resolve_job(j)
        cp_path=s['parent']/s['checkpoint']['path'];assert m.sha(cp_path)==s['checkpoint']['sha256']
        cp=runner.torch_load_checkpoint(cp_path,map_location='cpu');factory.validate_checkpoint_route(cp,j['arm'])
        d=s['definition'];config={k:v for k,v in d['model'].items() if k not in {'backbone','lambda_log_qty','lambda_tail','time_head_lr_multiplier'}}
        model,_=factory.build_count_aware_model(j['arm'],**config,train_log_mean=d['statistics']['train_log_mean'],train_log_std=d['statistics']['train_log_std'],max_seq_len=d['loader']['max_seq_len'])
        model.load_state_dict(cp['model_state_dict'],strict=True)
        assert runner.canonical_state_dict_sha256(model.state_dict())==s['state_sha']
        assert len(d['quantity_boundaries_all_train_rows'])==4
        assert cp['best_epoch']==s['selected_epoch'] and cp['seed']==j['seed'] and cp['backbone']==j['arm']
        m.verify_science(s,imports=True)
        rows.append({'id':j['id'],'state_sha256':s['state_sha'],'source_closure':s['source_closure'],'strict_load':'PASS'})
        del model,cp
    print(json.dumps({'rows':rows}))


if __name__=='__main__':main()
