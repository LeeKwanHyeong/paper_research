"""Reaggregate preserved predictions and bind original selected validation records.

No model loading, inference, training, raw dataset reads, or remote access.
"""
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TIME = 'recorded_positive_integer_lognormal_round_clamp_mass_nll; top-coded survival where configured'

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def load(p): return json.loads(Path(p).read_text())
def rel(p): return str(Path(p).resolve().relative_to(ROOT))
def write(n, x): (OUT/n).write_text(json.dumps(x, indent=2, ensure_ascii=False)+'\n')
def key(r): return r['dataset'],r['model'],int(r['seed'])
def near(a,b): return math.isclose(float(a),float(b),abs_tol=1e-5,rel_tol=1e-5)

def main():
    paths = [ROOT/'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json', ROOT/'reports/titantpp_b_test_evaluation_20261003_v1/evaluation_registry.json']
    bundles = [ROOT/'reports/titantpp_legacy_evaluation_20261003_v1', ROOT/'reports/titantpp_b_test_evaluation_20261003_v1']
    regs = [load(p) for p in paths]
    registered = {key(r):r for reg in regs for r in reg['rows']}
    assert len(registered)==117
    contract_path = ROOT/'reports/titantpp_test_stratified_analysis_20261003_v1/analysis_contract.json'
    thresholds = {r['dataset']:r['quantity_boundaries'][-1] for r in load(contract_path)['datasets']}
    original_csv = ROOT/'reports/titantpp_dataset_appendix_20261002_v1/selected_conditions.csv'
    original_val = {key(r):r for r in csv.DictReader(original_csv.open())}
    old_test_path = ROOT/'reports/titantpp_test_stratified_analysis_20261003_v1/results/seed_metrics.csv'
    old_test = {key(r):r for r in csv.DictReader(old_test_path.open()) if r['view']=='all' and r['seed']}
    metrics, source_checks, populations, prediction_checks = [], [], {}, []
    prediction_metrics = {}
    full_validation_keys = set()
    counters=dict(prediction_parts=0,prediction_bytes=0,checkpoints=0,endpoint_records=0)
    def add(r, split, view, m, source, provenance, receipt=None, simple=None):
        dataset=r['dataset']; family='simple_quantity_reference' if simple else ('historical_TitanTPP_B' if r['model']=='titantpp' else 'matched_head_main_comparison')
        row=dict(dataset=dataset, model=simple or r['model'],seed='' if simple else r['seed'],split=split,view=view,
                 threshold=thresholds[dataset] if view=='tail' else '',threshold_rule='raw_quantity > threshold' if view=='tail' else '',
                 count=int(m['count']),qty_mae=m['qty_mae'],qty_rmse=m['qty_rmse'],time_nll='' if simple else m['time_nll'],
                 selected_epoch='' if simple else r['selected_epoch'],source=rel(source),source_sha256=sha(source),
                 receipt_path=rel(receipt) if receipt else '',time_nll_definition='not_applicable_quantity_only' if simple else TIME,
                 comparison_family=family,provenance=provenance)
        metrics.append(row)
    def aggregate(frame, column='predicted_raw_quantity'):
        e=pl.col(column)-pl.col('raw_quantity')
        z=frame.select(pl.len().alias('count'),e.abs().mean().alias('qty_mae'),e.pow(2).mean().sqrt().alias('qty_rmse'),pl.col('time_nll').mean()).row(0,named=True)
        return z
    # Stored full-population predictions are preferred wherever available.
    for i,(rp,bundle) in enumerate(zip(paths,bundles)):
        for scope in ('validation_full','test'):
            receipts=sorted((bundle/'runs'/scope/'attempt1').glob('*/receipt.json'))
            assert len(receipts)==((4 if i==0 else 9) if scope=='validation_full' else (108 if i==0 else 9))
            for index,p in enumerate(receipts):
                receipt=load(p); k=key(receipt); r=registered[k]; split=receipt['split']
                assert receipt['status']=='complete' and receipt['full_population'] is True
                assert receipt['checkpoint_file_sha256']==r['checkpoint_file_sha256']
                assert receipt['state_tensor_sha256']==r['state_tensor_sha256']
                assert receipt['selected_epoch']==r['selected_epoch']
                assert receipt['source_closure_sha256']==r['source_closure_sha256']
                assert receipt['registry_sha256']==sha(rp)
                manifest=bundle/'dataset_manifest.json';assert receipt['dataset_manifest_sha256']==sha(manifest)
                ds=next(d for d in load(manifest)['datasets'] if d['dataset']==r['dataset'])
                assert receipt['data_file_sha256']==ds['sha256'] and receipt['loader']==ds['loader']
                pop={x:receipt[x] for x in ('prediction_rows','data_file_sha256','target_identity_sha256','truth_sha256','truth_digest_fields','loader')}
                pk=r['dataset']+'__'+split
                if pk in populations: assert pop==populations[pk],pk
                else: populations[pk]=pop
                parts=[]
                for part in receipt['parts']:
                    pp=p.parent/part['path'];assert sha(pp)==part['sha256'],str(pp)
                    counters['prediction_parts']+=1;counters['prediction_bytes']+=pp.stat().st_size;parts.append(pp)
                frame=pl.read_parquet(parts)
                assert frame.height==receipt['expected_target_count']==receipt['prediction_rows']
                assert frame['target_id'].n_unique()==frame.height
                assert frame['split'].unique().to_list()==[split]
                assert frame['raw_quantity'].is_finite().all() and frame['predicted_raw_quantity'].is_finite().all() and frame['time_nll'].is_finite().all()
                views={'overall':frame,'tail':frame.filter(pl.col('raw_quantity')>thresholds[r['dataset']])}
                for view,f in views.items():
                    m=aggregate(f);prediction_metrics[k+(split,view)]=m
                    add(r,split,view,m,p,'reaggregated_SHA_verified_full_population_predictions',p)
                    if r['model']=='titantpp_history_mlp' and r['seed']==42:
                        for col,name in [('last_quantity','last_observed_quantity'),('mean_quantity','mean_quantity_in_same_observed_window')]:
                            add(r,split,view,aggregate(f,col),p,'reaggregated_SHA_verified_full_population_predictions',p,name)
                if split=='test' and k in old_test:
                    for metric in ('qty_mae','qty_rmse','time_nll'):assert near(prediction_metrics[k+(split,'overall')][metric],old_test[k][metric]),(k,metric)
                if split=='validation': full_validation_keys.add(k)
                prediction_checks.append(dict(dataset=r['dataset'],model=r['model'],seed=r['seed'],split=split,receipt_path=rel(p),receipt_sha256=sha(p),parts=len(parts),rows=frame.height,checkpoint_binding=True,population_history_binding=True))
                if (index+1)%20==0: print(f'{bundle.name} {scope}: {index+1}/{len(receipts)} verified',flush=True)
    assert len(full_validation_keys)==13
    for k,r in registered.items():
        cp=ROOT/r['checkpoint_path'];assert sha(cp)==r['checkpoint_file_sha256'];counters['checkpoints']+=1
        p=ROOT/r['validation_replay_path'];ep=load(p);v=ep['selected']
        assert v['evaluation_scope']=='validation_only' and v['held_out_test_evaluated'] is False
        assert v['state_sha256']==r['state_tensor_sha256'] and ep['best_epoch']==r['selected_epoch']
        binding=''
        if k in original_val:
            prior=original_val[k];assert sha(p)==prior['source_sha256'];binding='dataset_appendix_selected_conditions_source_sha256'
            assert v['state_sha256']==prior['model_tensor_sha256']
            for m in ('qty_mae','qty_rmse','time_nll'): assert near(v[m],prior[m])
        elif 'validation_replay_sha256' in r:
            assert sha(p)==r['validation_replay_sha256'];binding='evaluation_registry_validation_replay_sha256'
        else:
            jobroot=Path(str(p).split('/runs/')[0]);tm=jobroot/'terminal_manifest.json';manifest=load(tm)
            assert manifest['scientific_success'] is True
            assert sha(p)==manifest['files'][str(p.relative_to(jobroot))];binding='original_terminal_manifest_record_sha256'
        for view,m in [('overall',v),('tail',v['tail'])]:
            if view=='tail':assert v['quantity_boundaries'][-1]==thresholds[r['dataset']]
            if k in full_validation_keys:
                fresh=prediction_metrics[k+('validation',view)]
                assert int(fresh['count'])==int(m['count'])
                for name in ('qty_mae','qty_rmse','time_nll'):assert near(fresh[name],m[name]),(k,view,name)
            else:add(r,'validation',view,m,p,'reused_SHA_verified_selected_endpoint_aggregate; no_full_validation_predictions_in_evaluation_bundle')
        source_checks.append(dict(dataset=r['dataset'],model=r['model'],seed=r['seed'],source=rel(p),source_sha256=sha(p),binding=binding,checkpoint_path=rel(cp),checkpoint_sha256=sha(cp),selected_state_binding=True,full_validation_predictions_reaggregated=k in full_validation_keys))
        counters['endpoint_records']+=1
    assert len(metrics)==500
    metrics.sort(key=lambda r:(r['dataset'],r['model'],str(r['seed']),r['split'],r['view']))
    with (OUT/'metrics_per_seed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    write('metrics_per_seed.json',dict(rows=metrics))
    write('source_bindings.json',dict(registries=[dict(path=rel(p),sha256=sha(p)) for p in paths],selected_endpoint_records=source_checks,prediction_receipts=prediction_checks,tail_contract=dict(path=rel(contract_path),sha256=sha(contract_path),thresholds=thresholds)))
    write('population_bindings.json',populations)
    write('verification.json',dict(status='passed',created_utc=datetime.now(timezone.utc).isoformat(),learned_conditions=117,validation_prediction_conditions=13,validation_endpoint_aggregate_conditions=104,test_prediction_conditions=117,simple_reference_rows_per_split=8,overall_metric_rows=250,tail_metric_rows=250,counters=counters,new_model_inference_calls=0,remote_calls=0,original_files_modified=False,metric_tolerance=dict(absolute=1e-5,relative=1e-5),source_validation_note='13 Validation full-prediction metrics checked against original selected replay; 104 selected aggregates checked against their recorded source SHA and state/checkpoint binding. All 108 legacy Test overall values match existing verified aggregate.'))
    print(json.dumps(dict(status='passed',rows=len(metrics),counters=counters)),flush=True)

if __name__=='__main__': main()
