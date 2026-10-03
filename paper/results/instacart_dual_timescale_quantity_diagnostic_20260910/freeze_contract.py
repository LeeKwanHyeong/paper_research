from pathlib import Path
import json, hashlib, subprocess
from datetime import datetime, timezone
r=Path(__file__).resolve().parents[3];out=Path(__file__).resolve().parent
large=r/'search_artifacts/instacart_dual_timescale_quantity_diagnostic_20260910';campaign=r/'paper/results/final_backbone_and_baselines_20260909'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
bcp=r/'search_artifacts/hard_lmm_raw_rmse_checkpoint_alignment_seed42_5090_20260906_f752434/seed42_e300/insta_market_basket/quantile_checkpoint_alignment/runs/titantpp/count_only_log_regression/seed_42/best_val_qty_rmse_model.pt'
models={}
expected={'b':'e594f8df0dcd66249c3724063f44eee3de213bc83a780ca81fdf39a094bfed07','candidate':'a62f05ca648708672eed86cdaaa94ce6ac0748354436cf1bc4ca527f02f3f0c8','rmtpp':'64ec26afba13d3117e0d3cf0cc1c554d2a6de10b5a4181834dd0096fd290aacc','thp':'f52310f70e19bc2a9e7f7b63d256e684b9aee7daa8711e5505e4521f3582df37'}
for role in expected:
 cp=bcp if role=='b' else large/'checkpoints'/f'{role}.pt'
 if role=='b': sm=campaign/'B_reference/insta_market_basket/summary.json'
 elif role=='candidate': sm=next((campaign/'5090/instacart_completed').glob('**/summary.json'))
 else: sm=campaign/f'Instacart_existing/{role}/training/runs/{role}/count_only_log_regression/seed_42/summary.json'
 s=json.loads(sm.read_text())
 if sha(cp)!=expected[role]:raise ValueError('checkpoint digest drift')
 models[role]={'checkpoint':str(cp),'checkpoint_file_sha256':sha(cp),'checkpoint_state_sha256':s['checkpoint_state_sha256'],'summary':str(sm),'summary_file_sha256':sha(sm),'backbone':s['backbone'],'best_epoch':s['best_epoch'],'checkpoint_selection':s['checkpoint_selection']}
data=r/'sample_data/insta_market_basket/instacart_marked_target_with_split.parquet';split=r/'sample_data/insta_market_basket/instacart_marked_target_split_manifest.json'
train_cache=r/'search_artifacts/frozen_raw_affine_calibration_v1_1ba9a43_5080/full/train/insta_market_basket_B/train_predictions.npz';val_cache=r/'search_artifacts/hard_lmm_bounded_qk_error_diagnostic_20260908/paired_validation_predictions.parquet'
source=Path('/tmp/paper_research_dual_timescale');rev=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
if rev!='ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c':raise ValueError('source revision drift')
files=['paper/scripts/count_aware_tpp_backbone/core.py','paper/scripts/run_matched_frozen_lognormal_duration.py','paper/scripts/run_hard_lmm_bounded_qk_error_diagnostic.py','data_loader/event_seq_data_module.py','models/TPPs/CountAwareTitanDualTimescale.py','models/TPPs/CountAwareFactory.py']
c={'contract_id':'instacart_dual_timescale_quantity_v1','status':'frozen_before_row_level_inference','frozen_at':datetime.now(timezone.utc).isoformat(),'scope':{'dataset':'insta_market_basket','seed':42,'splits':['validation','train'],'held_out_test':False,'training':False,'checkpoint_selection':False,'parameter_updates':False,'calibration_fit':False,'primary_contrast':'candidate_minus_rmtpp','secondary_contrasts':['candidate_minus_b','candidate_minus_thp']},'models':models,'data':{'path':str(data),'sha256':sha(data),'split_manifest':str(split),'split_manifest_sha256':sha(split),'lookback_weeks':52,'max_seq_len':64,'train_quantity_boundaries':[8,20,25,35]},'populations':{'train':{'count':1991192,'identity_sha256':'c361f5e9904c25f18c91007f6e1209518a02fb755799e81c8300ee759b2a1210','quantity_sha256':'07e98693d5ac26caeaeb3c60d6d31fad21ab8f1e40ac649f231ac299963817d8'},'validation':{'count':503733,'identity_sha256':'28356570163221aa3eb13735076452886bdb471451e834e784afc3ad5c54bde8','quantity_sha256':'28ba2447505201e312d5049d20c0a2eeba109cb33be0c4413a0f69192c539184'}},'b_caches':{'train':{'path':str(train_cache),'sha256':sha(train_cache)},'validation':{'path':str(val_cache),'sha256':sha(val_cache)}},'execution':{'device':'cpu','batch_size':128,'threads':4,'shuffle':False,'frozen_source_root':str(source),'frozen_source_revision':rev,'source_file_sha256':{f:sha(source/f) for f in files},'validation_metric_absolute_tolerance':1e-5,'validation_before_train':True,'same_dataset_instance_for_every_model':True},'analysis':{'history_bins':['1','2-3','4-7','8-15','16-31','32-63'],'change_bins':['negative','zero','positive'],'series_fold':'int(SHA256(UTF8("instacart_dual_timescale_quantity_v1:20260910|"+str(series_id))),16)%2','fold_purpose':'Descriptive replication in disjoint training series partitions. Models saw both folds during fitting; not out-of-fold evidence.','decomposition':'MSE=bias^2+mean((error-bias)^2); group delta contributions weighted count/N','series_concentration_fractions':[0.001,0.01,0.05],'bootstrap_replicates':1000,'bootstrap_seed':20260910,'hypothesis_policy':'Fixed quantity/history/change cuts before inference. Narrow explanation on validation; freeze explanation before train comparison. Verify current candidate independently of earlier BOUNDED-QK findings; no causal claim from association.'}}
p=out/'contract.json'
if p.exists():raise FileExistsError('refuse to overwrite frozen contract')
p.write_text(json.dumps(c,ensure_ascii=False,indent=2)+'\n');print(sha(p))
