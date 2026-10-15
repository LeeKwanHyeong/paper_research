#!/usr/bin/env python3
"""Stdlib-only, immutable saved Validation aggregate decomposition; no inference."""
from pathlib import Path
import json,hashlib,math,statistics,datetime,sys
ROOT=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
CAMPAIGN='titantpp_backbone_value_difference_raf3_20261016_v1'
B=ROOT/'search_artifacts'/CAMPAIGN;E=B/'evaluation_v1'
OUT=ROOT/'reports'/CAMPAIGN/'diagnostics_quantity_v1'
PINS={'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/execution_contract.json': 'c40a2921595bf60ac13d34d915f5a5b3fabe57db00b0eabf394d085c3c00ad99', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/preparation/contract.json': '1425944fc892f30434df3204dbcc831a7a4f7ced175b46d8e98b28d9638963cb', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/validation_gate.json': 'c3c1a58aa8c095145b974a24dec7f8147d66c4db064d2e336c8cf3685a76aa84', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/native/5090/validation_result_manifest.json': '90615f67a563a31c98326d6c3cddc75530c2894c42695fde173ae1675bf5aaaf', 'reports/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/THREE_SEED_VALIDATION.json': '59007ce031e3e6978f92b848cd8fd3a30f91788ec6412e1eb502667e63dd20cb', 'reports/titantpp_backbone_location_raf6_20261015_v1/diagnostics_v1/QUANTITY_TRADEOFF.json': 'a56d4715b1a5951144a57c2896caefdd672519ff0eee6c893993f2312a6c6ee8', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__42__titantpp_cnn_gru54_cnn_only/checkpoint_metadata.json': '2ec2561e790e0410fbc9b1cd91d08c92e49284f7cbd03626abba87768f09f279', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__42__titantpp_cnn_gru54_cnn_only/selected.pt': '182a16d4feb46b110088d6ed17c2f5d8d8d4b5b34c95ca104ab30beb3c0c0cd1', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__42__titantpp_cnn_gru54_cnn_only/selection_history.json': '18fec225b931ccd5addebae81b35b6198922cf8038cf36ae813dd54d51cf1b8e', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__42__titantpp_cnn_gru54_cnn_only/terminal_manifest.json': '74996833e00b45990ce33ad32aa624c0dc2a8b7bfcf3682450908bbd2e334caf', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__42__titantpp_cnn_gru54_cnn_only/worker_exit.json': '44a582557ae6366eef334e01251fcde7b6d502df5e674a4008d298d2d34a584f', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__52__titantpp_cnn_gru54_cnn_only/checkpoint_metadata.json': '80ab7d0cfe1fe3a3894bdc831d1c008e8521fd68d4090d6e906236b32cd902d1', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__52__titantpp_cnn_gru54_cnn_only/selected.pt': 'dceb5d346f750aeca93d063c17ef2f81d43ec3ced5867abbbeab10cd7146cf38', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__52__titantpp_cnn_gru54_cnn_only/selection_history.json': 'a0fd11427044296e52555fbbc4af96c4ab2968b3fd6987fff75bdc65f4aaaf4f', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__52__titantpp_cnn_gru54_cnn_only/terminal_manifest.json': 'ed09adfed5e6c628190455a8629966345e601e9362ea1a97a963c842a9cc81f8', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__52__titantpp_cnn_gru54_cnn_only/worker_exit.json': 'af9ecfeecd67462cf94a39dc326ec06aa78e75c404b4fdaa3dc62ffc98162317', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__62__titantpp_cnn_gru54_cnn_only/checkpoint_metadata.json': '4c5dae219ed2940fa873db542d3662b481043e128b19718ec80d1f7f8d980d5a', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__62__titantpp_cnn_gru54_cnn_only/selected.pt': '018c634f0db51054393af97f2b36b96b82da1d5cceb2d26c0e3a454fbedd9d7f', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__62__titantpp_cnn_gru54_cnn_only/selection_history.json': '7d58b8fdaff4f2b2adc81d7e9fe8981867bf9d21f1d6e36a3b16164b80fcbe0a', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__62__titantpp_cnn_gru54_cnn_only/terminal_manifest.json': '15346c412d3be00bd71d651a716e2fe5aef50b44f735d7d91b39471f475d2f71', 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/inputs/jobs/raf_spare_parts__62__titantpp_cnn_gru54_cnn_only/worker_exit.json': '820227bd657798d2c786fe5d42195f78623a28272986d61559c6049689116c3c', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/results/raf_spare_parts__42__titantpp_cnn_value_difference_v1/validation_replay.json': 'b81eaeb9910f72ade01f3c00b5e26471e58f5c9370f447530e75d1fc7319af2d', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/results/raf_spare_parts__42__titantpp_cnn_value_difference_v1/validation_worker_exit.json': 'd9914aac01b639fa0106108bff1f70fd972fdc590e2895bd8e119fc4b5b7be9e', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__42__titantpp_cnn_value_difference_v1/terminal_manifest.json': 'd80d316c7b8439cf45c0b312218e2270327d8553da20e0ba0a409e39d6cc31a9', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__42__titantpp_cnn_value_difference_v1/runs/titantpp_cnn_value_difference_v1/count_only_log_regression/seed_42/history.json': 'f37d468389b549d1f8474fddcf53d346685865e09ec10eb1db229ff1b70b766f', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__42__titantpp_cnn_value_difference_v1/runs/titantpp_cnn_value_difference_v1/count_only_log_regression/seed_42/endpoint_replays.json': '3b68b30cb78663db14b4a24e0d38aeca5ea233da537c76ace46e4bddc203612b', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__42__titantpp_cnn_value_difference_v1/runs/titantpp_cnn_value_difference_v1/count_only_log_regression/seed_42/best_val_qty_rmse_model.pt': 'd1d0c36e4e8eda3cba947bece753fbbe43310d41027b15bb7567e01041778839', 'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1/originals/run/raf_spare_parts__42__titantpp_cnn_gru54_cnn_only/runs/titantpp_cnn_gru54_cnn_only/count_only_log_regression/seed_42/endpoint_replays.json': '137004caf53a147fdc7e1b4851ab80cb5c8a7a7f5ec5e632d5dc6da66fd9f627', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/results/raf_spare_parts__52__titantpp_cnn_value_difference_v1/validation_replay.json': '59bde364625923ae0da899137bf82fdcc6f4291c95abd104e425677c72746940', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/results/raf_spare_parts__52__titantpp_cnn_value_difference_v1/validation_worker_exit.json': '04f1d79be12ea8912ca26643bde86933efcd87f67b2b91c01439b333c3b187e1', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__52__titantpp_cnn_value_difference_v1/terminal_manifest.json': '323e88f3184e98f9713c7c5c5e7f5a3c43b24a0a2867b5028280fdb3a33ea59d', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__52__titantpp_cnn_value_difference_v1/runs/titantpp_cnn_value_difference_v1/count_only_log_regression/seed_52/history.json': '7511a5006d01a8093b242ec9e4ec8158ac76a4ec76019f66a63a3ce1898b2f93', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__52__titantpp_cnn_value_difference_v1/runs/titantpp_cnn_value_difference_v1/count_only_log_regression/seed_52/endpoint_replays.json': 'a2406c2863cbaacab87e9ac621985992168a325b4b7b9bd7bd6f153ec060a23e', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__52__titantpp_cnn_value_difference_v1/runs/titantpp_cnn_value_difference_v1/count_only_log_regression/seed_52/best_val_qty_rmse_model.pt': 'f4deb6c52d1560c6671c6eb7a78d325f43aad5f86e217f03807141dff895a8c0', 'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1/originals/run/raf_spare_parts__52__titantpp_cnn_gru54_cnn_only/runs/titantpp_cnn_gru54_cnn_only/count_only_log_regression/seed_52/endpoint_replays.json': '972180b1036091fbe05d5c51ee1101415214937fb933ac7772c44a7aa84c01da', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/results/raf_spare_parts__62__titantpp_cnn_value_difference_v1/validation_replay.json': '40ff5a2b1ecedc04fea88b0f52830747248b0bf122cb3daddf9e22353efadf7b', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/results/raf_spare_parts__62__titantpp_cnn_value_difference_v1/validation_worker_exit.json': 'a390c6bf50509f4b8db210158f9594f95ac1aba0db102b2ff2177e8c08e3c84c', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__62__titantpp_cnn_value_difference_v1/terminal_manifest.json': '81d490b30d1fd0205b4952d82b58712a9e53866403d05a208ca1adda67676413', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__62__titantpp_cnn_value_difference_v1/runs/titantpp_cnn_value_difference_v1/count_only_log_regression/seed_62/history.json': '4f2a8f8718166f549e673ac33b2c126d2488e41a2f9dca0ce7c6a39efeb6308c', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__62__titantpp_cnn_value_difference_v1/runs/titantpp_cnn_value_difference_v1/count_only_log_regression/seed_62/endpoint_replays.json': 'ef48dd377ed1fe4ee96ddae5256597e555e4779441a091dd7939f20fab07a2c2', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/evaluation_v1/originals/raf_spare_parts__62__titantpp_cnn_value_difference_v1/runs/titantpp_cnn_value_difference_v1/count_only_log_regression/seed_62/best_val_qty_rmse_model.pt': '0995fb284437779d7b575d9fcfe30d9accfac4baae2c1d2b4dda5f090d855c8a', 'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1/originals/run/raf_spare_parts__62__titantpp_cnn_gru54_cnn_only/runs/titantpp_cnn_gru54_cnn_only/count_only_log_regression/seed_62/endpoint_replays.json': '86dd66ebdf823e41eca6a8674557ad02743920093ebc69e45c51e10c06162e3a', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/source/inputs/data/raf_spare_parts/split_manifest.json': '3cd6204667de4e44eda8787d32845bebff7624e1bb255ffd7677f810b8d7b149', 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1/source/inputs/data/raf_spare_parts/train_validation.parquet': 'd805cad0242078fedfbb87eed18bde9b73d37bfc9b93c6b75aac49087244efa2'}
CANDIDATE='titantpp_cnn_value_difference_v1'
SEEDS=(42,52,62); COUNTS=[3626,2555,235,224,50]; LABELS=['q≤2','2<q≤30','30<q≤60','60<q≤200','q>200']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def require(v,msg):
 if not v:raise ValueError(msg)
def read(p):
 p=Path(p);rel=str(p.relative_to(ROOT));require(rel in PINS,'Unpinned input '+rel)
 require(sha(p)==PINS[rel],'Changed input '+rel)
 return json.loads(p.read_text())
def close(a,b,msg):require(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-7),msg)
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def selected(history):
 best=None;value=math.inf
 for r in history['history']:
  x=r.get('val_qty_rmse')
  if isinstance(x,(int,float)) and math.isfinite(x) and x<value:best=int(r['epoch']);value=x
 require(best is not None,'No finite selected epoch')
 return best,value

def normalize(cell):
 n=cell['count'];require(n>0,'Empty unexpected quantity cell')
 sse=float(cell['qty_sse']);ae=float(cell.get('qty_absolute_error_sum',cell['qty_mae']*n))
 close(cell['qty_rmse']**2*n,sse,'SSE metric identity');close(cell['qty_mae']*n,ae,'AE metric identity')
 signed=cell.get('qty_signed_error_sum')
 if signed is not None:
  close(cell['qty_bias']*n,signed,'Signed error identity');require(abs(signed)<=ae+1e-7,'Signed error exceeds absolute error')
 return {'count':n,'qty_sse':sse,'qty_absolute_error_sum':ae,'qty_rmse':cell['qty_rmse'],'qty_mae':cell['qty_mae'],
  'qty_signed_error_sum':signed,'qty_bias':None if signed is None else signed/n,
  'bias_status':'unavailable_in_original_endpoint' if signed is None else 'saved_signed_aggregate',
  'ae_origin':'stored_sum' if 'qty_absolute_error_sum' in cell else 'stored_MAE_times_exact_count'}

def metric_checks(m):
 require(m['count']==6690 and m['quantity_boundaries']==[2.,30.,60.,200.],'Population/boundary drift')
 require([x['bin'] for x in m['quantity_cells']]==list(range(5)),'Bin order drift')
 require([x['count'] for x in m['quantity_cells']]==COUNTS,'Bin count drift')
 whole=normalize(m);cells=[normalize(x) for x in m['quantity_cells']];tail=normalize(m['tail'])
 require(sum(x['count'] for x in cells)==6690,'Whole count mismatch')
 for key in ['qty_sse','qty_absolute_error_sum']:
  close(sum(x[key] for x in cells),whole[key],'Nonadditive quantity total '+key)
  close(tail[key],cells[-1][key],'Tail cell mismatch '+key)
 if whole['qty_signed_error_sum'] is not None:
  close(sum(x['qty_signed_error_sum'] for x in cells),whole['qty_signed_error_sum'],'Signed total mismatch')
 return whole,cells

def delta(a,b):
 require(a['count']==b['count'],'Candidate/reference population count differs')
 n=a['count']
 return {'count':n,'candidate':a,'cnn':b,'delta_sse':a['qty_sse']-b['qty_sse'],
  'delta_absolute_error_sum':a['qty_absolute_error_sum']-b['qty_absolute_error_sum'],
  'delta_mse':(a['qty_sse']-b['qty_sse'])/n,'delta_rmse':a['qty_rmse']-b['qty_rmse'],
  'delta_mae':a['qty_mae']-b['qty_mae'],'delta_bias':None,'delta_bias_status':'CNN signed error is not saved; RMSE/MAE do not identify bias'}

def combine(cells):
 n=sum(x['count'] for x in cells);sse=sum(x['qty_sse'] for x in cells);ae=sum(x['qty_absolute_error_sum'] for x in cells)
 signed=None if any(x['qty_signed_error_sum'] is None for x in cells) else sum(x['qty_signed_error_sum'] for x in cells)
 return {'count':n,'qty_sse':sse,'qty_absolute_error_sum':ae,'qty_rmse':math.sqrt(sse/n),'qty_mae':ae/n,
  'qty_signed_error_sum':signed,'qty_bias':None if signed is None else signed/n}

def avg(rows):
 keys=['delta_sse','delta_absolute_error_sum','delta_rmse','delta_mae']
 z={k:statistics.mean(r[k] for r in rows) for k in keys}
 z.update({k+'_sample_sd':statistics.stdev(r[k] for r in rows) for k in keys})
 z['count_per_seed']=rows[0]['count'];z['candidate_bias_mean']=statistics.mean(r['candidate']['qty_bias'] for r in rows)
 z['cnn_bias']=None;z['delta_bias']=None
 return z

def main():
 for rel,h in PINS.items():require(sha(ROOT/rel)==h,'Input SHA drift '+rel)
 for name in ['RESULT.json','RESULT.md','RECEIPT.json']:require(not (OUT/name).exists(),'Exclusive output already exists '+name)
 c=read(B/'execution_contract.json');prep=read(B/'preparation/contract.json');gate=read(E/'validation_gate.json')
 native_manifest=read(E/'native/5090/validation_result_manifest.json')
 summary=read(ROOT/'reports'/CAMPAIGN/'evaluation_v1/THREE_SEED_VALIDATION.json')
 require(summary['split']=='validation' and summary['status']=='complete','Wrong summary split/status')
 require(gate['status']=='PASS' and gate['job_count']==3,'Validation gate missing')
 ds=c['datasets'][0];require(ds['dataset_id']=='raf_spare_parts','Dataset mismatch')
 for rel,h in prep['baseline_pins'].items():require(PINS.get(rel)==h,'Baseline chain mismatch')
 for rel,h in c['input_files'].items():
  if 'train_validation.parquet' in rel or 'split_manifest.json' in rel:require(PINS[str((B/'source'/rel).relative_to(ROOT))]==h,'Input identity changed')
 records=[];selection=[]
 for seed in SEEDS:
  ident=f'raf_spare_parts__{seed}__{CANDIDATE}';orig=E/'originals'/ident;run=orig/f'runs/{CANDIDATE}/count_only_log_regression/seed_{seed}'
  rp=E/f'results/{ident}/validation_replay.json';ep=E/f'results/{ident}/validation_worker_exit.json'
  a=read(rp);ex=read(ep)
  require(sha(rp)==gate['validation_receipts'][ident]==native_manifest[str(rp.relative_to(E))],'Native receipt binding')
  require(sha(ep)==gate['validation_worker_exits'][ident]==native_manifest[str(ep.relative_to(E))],'Native exit binding')
  require(ex['actual_child_exit_observed'] is True and ex['exit_code']==0 and ex['split']=='validation' and ex['job_id']==ident,'Native worker did not exit0')
  require(a['split']=='validation' and a['held_out_test_evaluated'] is False and a['validation_match']=='PASS' and a['full_population'] is True,'Candidate Validation scope')
  require(a['seed']==seed and a['arm']==CANDIDATE and a['prediction_rows']==6690,'Candidate identity')
  require(a['contract_sha256']==gate['contract_sha256']==ex['contract_sha256'],'Evaluation contract binding')
  require(a['original_population_identity']==ds['inherited_data_identity']['populations']['validation'],'Original Validation target fingerprint')
  require(a['data_sha256']==ds['data_sha256'] and a['loader']==ds['loader'],'Input/loader binding')
  tm=read(orig/'terminal_manifest.json');h=read(run/'history.json');end=read(run/'endpoint_replays.json')
  require(tm['scientific_success'] is True and tm['held_out_test_evaluated'] is False and tm['contract_sha256']==canonical(c),'Training terminal scope')
  for p in [run/'history.json',run/'endpoint_replays.json',run/'best_val_qty_rmse_model.pt']:
   require(tm['files'][str(p.relative_to(orig))]==sha(p),'Candidate terminal SHA binding')
  e,v=selected(h);require(e==a['selected_epoch']==end['best_epoch'],'Candidate strict selection epoch mismatch')
  require(a['checkpoint_sha256']==sha(run/'best_val_qty_rmse_model.pt') and a['state_sha256']==end['selected']['state_sha256'],'Candidate checkpoint binding')
  close(v,end['selected']['qty_rmse'],'Candidate first minimum metric')
  # Native reproduction may differ only within the already approved replay tolerance.
  for metric in ['qty_rmse','qty_mae','time_nll']:
   require(math.isclose(a['metrics'][metric],end['selected'][metric],rel_tol=1e-5,abs_tol=1e-5),'Candidate replay discrepancy')
  ref=next(x for x in c['baseline_jobs'] if x['seed']==seed)['original'];base=ROOT/'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3'/Path(ref['checkpoint_metadata_path']).parent
  bh=read(base/'selection_history.json');bm=read(base/'checkpoint_metadata.json');bt=read(base/'terminal_manifest.json')
  be,bv=selected(bh);require(be==ref['selected_epoch']==bm['best_epoch'],'Historical first strict selection mismatch')
  require(sha(base/'selected.pt')==ref['checkpoint']['sha256'] and bm['model_state_sha256']==ref['checkpoint_state_sha256'],'CNN checkpoint mismatch')
  args=bm['resume_identity']['arguments'];require(args['data_sha256']==ds['data_sha256'] and args['max_seq_len']==ds['loader']['max_seq_len'],'Historical dataset/loader mismatch')
  bp=ROOT/f'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1/originals/run/raf_spare_parts__{seed}__titantpp_cnn_gru54_cnn_only/runs/titantpp_cnn_gru54_cnn_only/count_only_log_regression/seed_{seed}/endpoint_replays.json'
  old=read(bp);rel=f'runs/titantpp_cnn_gru54_cnn_only/count_only_log_regression/seed_{seed}/endpoint_replays.json'
  require(sha(bp)==bt['files'][rel], 'Historical endpoint SHA mismatch')
  annotated_reference=dict(ref['validation_reference']);require(annotated_reference.pop('split',None)=='validation','Historical validation annotation changed')
  require(old['selected']==annotated_reference,'Historical endpoint differs beyond added split annotation')
  require(old['best_epoch']==be and old['held_out_test_evaluated'] is False,'Historical scope/selection drift');close(bv,old['selected']['qty_rmse'],'Historical first minimum metric')
  b=old['selected'];aw,ac=metric_checks(a['metrics']);bw,bc=metric_checks(b)
  cells=[dict(bin=i,label=LABELS[i],**delta(x,y)) for i,(x,y) in enumerate(zip(ac,bc))]
  full=delta(aw,bw);tail=delta(ac[-1],bc[-1]);body=delta(combine(ac[:-1]),combine(bc[:-1]))
  for k in ['delta_sse','delta_absolute_error_sum']:
   close(sum(z[k] for z in cells),full[k],'Delta additivity '+k);close(body[k]+tail[k],full[k],'Tail partition '+k)
  records.append({'seed':seed,'candidate_selected_epoch':e,'cnn_selected_epoch':be,'whole':full,'non_tail':body,'tail':tail,'quantity_cells':cells,
   'largest_sse_worsening_bin':max(cells,key=lambda x:x['delta_sse'])['label'],
   'largest_ae_worsening_bin':max(cells,key=lambda x:x['delta_absolute_error_sum'])['label']})
  selection.append({'seed':seed,'candidate_selected_epoch':e,'cnn_selected_epoch':be,'candidate_checkpoint_sha256':a['checkpoint_sha256'],
   'candidate_state_sha256':a['state_sha256'],'cnn_checkpoint_sha256':ref['checkpoint']['sha256'],'cnn_state_sha256':ref['checkpoint_state_sha256'],
   'candidate_first_strict_finite_minimum_verified':True,'cnn_first_strict_finite_minimum_verified':True,'target_count':6690,
   'input_sha256':ds['data_sha256'],'candidate_target_fingerprint':a['original_population_identity'],
   'historical_population_verification':'frozen same input/loader/full target count/bin counts; no per-bin target-ID hash in historical endpoint'})
 agg={k:avg([r[k] for r in records]) for k in ['whole','non_tail','tail']}
 agg['quantity_cells']=[dict(label=LABELS[i],bin=i,**avg([r['quantity_cells'][i] for r in records])) for i in range(5)]
 for group in ['whole','non_tail','tail']:
  for arm,key in [('candidate','candidate'),('CNN','cnn')]:
   agg[group][arm+'_rmse_mean']=statistics.mean(r[group][key]['qty_rmse'] for r in records)
   agg[group][arm+'_mae_mean']=statistics.mean(r[group][key]['qty_mae'] for r in records)
 agg['tail_target_fraction']=50/6690
 agg['tail_sse_fraction_candidate']=statistics.mean(r['tail']['candidate']['qty_sse']/r['whole']['candidate']['qty_sse'] for r in records)
 agg['tail_sse_fraction_CNN']=statistics.mean(r['tail']['cnn']['qty_sse']/r['whole']['cnn']['qty_sse'] for r in records)
 result={'schema':'raf_value_difference_quantity_diagnostics_v1','status':'complete','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'split':'validation','candidate':CANDIDATE,'reference':'historical_titantpp_cnn','seeds':list(SEEDS),'source_pins':PINS,'selection_and_population_binding':selection,
  'seed_results':records,'aggregate':agg,'bias_definition':'prediction minus target; target-defined strata are descriptive',
  'calculation':'Stored SSE and stored AE sum; where AE sum is absent, stored MAE*exact count. Sum over bins is additive; mean RMSE is not additive.',
  'confirmed':['Same five bin boundaries/counts across all6 fixed models; total6690, tail50, non-tail6640.',
   'Candidate absolute signed-error aggregates are present; historical CNN signed sums are absent.',
   'All3 candidate/oldCNN selected epochs are first strict finite Validation rawRMSE minima; raw binaries were SHA-verified without loading.'],
  'unconfirmed':['CNN bias and candidate−CNN bias cannot be recovered from stored RMSE/MAE/SSE.',
   'Per-row paired effects, entity/time/history composition and cause are not identified by target quantity bins.',
   'Fixed-model Train raw-error buckets are not present in the scoped candidate records; epoch training log1p loss cannot substitute.',
   'No per-bin historical target-ID hashes are present; matching counts do not alone establish row identity. Same frozen input/loader/full target population supports compatibility.'],
  'limitations':['New candidate5090/cuDNN92000 vs historicalCNN5080/cuDNN91900.',
   'The intervention combines576→512 local parameters, difference parameterization and short-history boundary behavior; no isolated causal attribution.',
   'Design follows prior Test observations; this diagnostic opens no Test result and does not establish independent generalization or significance.'],
  'new_inference':False,'new_training':False,'cnn_reevaluated':False,'test_opened':False,'remote_calls':0,'checkpoints_loaded':False}
 lines=['# RAF 루프3: 고정 Validation 수량 구간 오류 분해','',
  '상태: 완료. 기존 CNN3과 V 차분 후보3의 저장된 Validation 원본만 비교했다. 신규 추론·학습·Test 열람·원격 호출은 없다.',
  '',f"전체 3seed 평균 RMSE는 CNN {agg['whole']['CNN_rmse_mean']:.6f} → 후보 {agg['whole']['candidate_rmse_mean']:.6f}, MAE는 {agg['whole']['CNN_mae_mean']:.6f} → {agg['whole']['candidate_mae_mean']:.6f}다. 두 지표 모두 평균이 악화됐다.",
  '', '**오차 기여를 구간별로 나눈다 — 완료**',
  '- 경계는 Train에서 고정한 2·30·60·200이며 모든 모델의 구간 수는 3626·2555·235·224·50, 합계 6690이다.',
  '- Δ는 후보−CNN이다. 음수는 오류 감소다. SSE는 제곱오차합, AE합은 절대오차합이며, 세 seed의 합계 차이를 평균했다. RMSE 차이 자체를 가산 분해한 표가 아니다.',
  '', '| 수량 구간 | 개수/seed | 평균 ΔSSE | 평균 ΔAE합 | 후보 평균 편향 | CNN 편향·Δ편향 |','|---|---:|---:|---:|---:|---|']
 for z in agg['quantity_cells']:lines.append(f"| {z['label']} | {z['count_per_seed']} | {z['delta_sse']:+.6f} | {z['delta_absolute_error_sum']:+.6f} | {z['candidate_bias_mean']:+.6f} | 미확인 |")
 lines += ['', '| 영역 | 개수/seed | 평균 ΔSSE | 평균 ΔAE합 |','|---|---:|---:|---:|']
 for name,k in [('q≤200','non_tail'),('q>200','tail'),('전체','whole')]:
  z=agg[k];lines.append(f"| {name} | {z['count_per_seed']} | {z['delta_sse']:+.6f} | {z['delta_absolute_error_sum']:+.6f} |")
 lines += ['',f"꼬리 50개는 전체의 {agg['tail_target_fraction']*100:.4f}%지만 후보 SSE의 평균 {agg['tail_sse_fraction_candidate']*100:.3f}%, CNN SSE의 평균 {agg['tail_sse_fraction_CNN']*100:.3f}%를 차지한다. 꼬리 개선과 비꼬리 악화를 함께 기록해야 한다.",
  '', '**Seed별 차이와 예외를 확인한다 — 완료**','',
  '| Seed | 후보/CNN 선택 epoch | ΔRMSE | ΔMAE | 비꼬리 ΔSSE | 꼬리 ΔSSE | 전체 ΔAE합 |','|---|---|---:|---:|---:|---:|---:|']
 for r in records:lines.append(f"| {r['seed']} | {r['candidate_selected_epoch']}/{r['cnn_selected_epoch']} | {r['whole']['delta_rmse']:+.6f} | {r['whole']['delta_mae']:+.6f} | {r['non_tail']['delta_sse']:+.6f} | {r['tail']['delta_sse']:+.6f} | {r['whole']['delta_absolute_error_sum']:+.6f} |")
 lines += ['', '| Seed | 구간 | ΔSSE | ΔAE합 | 후보 편향 |','|---|---|---:|---:|---:|']
 for r in records:
  for z in r['quantity_cells']:lines.append(f"| {r['seed']} | {z['label']} | {z['delta_sse']:+.6f} | {z['delta_absolute_error_sum']:+.6f} | {z['candidate']['qty_bias']:+.6f} |")
 lines += ['',
  'seed42는 꼬리 SSE 감소가 비꼬리 증가를 넘어서 전체 RMSE만 소폭 개선됐지만 MAE는 악화됐다. seed52는 꼬리 RMSE·MAE도 악화됐다. seed62는 꼬리의 큰 이득을 비꼬리 증가가 상쇄했으며, MAE 악화의 가장 큰 양의 기여는 2<q≤30이다. q≤2에서는 seed62 AE합이 감소했다.',
  '', '**편향과 원인 해석의 범위를 구분한다 — 완료**',
  '- 후보는 q≤30 두 구간에서 평균 양의 편향(과대 예측), q>30에서 음의 편향(과소 예측)을 보인다. 이는 실제 목표 수량으로 나눈 사후 구간의 기술 통계다.',
  '- 기존 CNN 선택 endpoint에는 signed error/bias가 없으므로 CNN 편향과 Δ편향은 복원할 수 없다. RMSE·MAE만으로 예측 방향을 추정하지 않는다.',
  '- 같은 입력 SHA·loader·전체 대상 수와 구간 수를 결속했다. 과거 CNN endpoint에는 구간별 대상 ID 해시가 없어, 개별 행의 동일성은 이번 집계에서 직접 재검산하지 않았다.',
  '- 고정 모델의 Train raw 오류 구간값은 이번 후보 저장 원본에 없다. 학습 중 log1p 손실을 Train raw RMSE로 해석하지 않는다.',
  '- 수량 구간은 오류가 발생한 곳을 보여줄 뿐, 관측 이력·개체 구성·시간 간격·최적화가 원인인지 분리하지 못한다.',
  '- 후보5090/cuDNN92000과 기존CNN5080/cuDNN91900의 차이가 남는다. 576→512 용량 감소·차분 매개화·짧은 이력 경계 처리가 함께 바뀌어 단일 구조 효과를 확정할 수 없다.',
  '- 기존 Test 관찰 뒤 설계한 탐색적 후속 결과다. 이 진단은 Test를 읽지 않았으며 독립 일반화 개선이나 통계적 유의성을 주장하지 않는다.',
  '', '**확인된 오류 기여를 다음 가설의 근거로 사용한다 — 다음 작업**',
  '- 기존 CNN과 원본 결과를 보존한다. 비꼬리, 특히 seed62의 2<q≤30·60<q≤200 오류 증가를 다음 backbone 가설의 설명 대상에 포함한다.',
  '- 구간 오류만으로 새 방법의 효과나 원인을 확정하지 않는다. 다음 단일 변경과 대조군의 계약은 루트 세션에서 별도로 정한다.',
  '', '모든 입력 경로와 raw SHA, 선택 epoch 및 binary SHA는 RESULT.json에 기록했다.']
 for name,obj in [('RESULT.json',result)]:
  with (OUT/name).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
 with (OUT/'RESULT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 for rel,h in PINS.items():require(sha(ROOT/rel)==h,'Input changed during diagnostics '+rel)
 receipt={'status':'complete','schema':'raf_quantity_diagnostics_receipt_v1','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'script_sha256':sha(__file__),'source_count':len(PINS),'all_input_sha_verified_before_and_after':True,
  'population_count':6690,'seed_count':3,'comparison_model_count':6,'first_strict_selection_verified_count':6,
  'checks':['native gate3 and worker exit0 binding','historical endpoint equality and terminal SHA','selected binary SHA without deserialization','same frozen input/loader and6690 full targets','same five bin counts and sum6690','SSE/AE metric identities','bin/whole/tail additivity','signed candidate sum consistency','all inputs unchanged'],
  'result_files':{n:sha(OUT/n) for n in ['RESULT.json','RESULT.md']},'remote_calls':0,'new_inference':False,'new_training':False,'test_opened':False}
 with (OUT/'RECEIPT.json').open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps({'status':'complete','whole':agg['whole'],'non_tail':agg['non_tail'],'tail':agg['tail'],
  'receipt_sha256':sha(OUT/'RECEIPT.json'),'result_files':receipt['result_files']},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
