"""Audit frozen validation histories only; never reselect or load a checkpoint."""
import json,hashlib,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'paper/results/final_backbone_and_baselines_20260909'
EXT=ROOT/'paper/results/dual_timescale_intermittent_quantity_extension_20260910'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
reference=read(BASE/'seed42_validation_comparison.json')
campaign=read(BASE/'final_campaign_decision.json')
rows=[];sources={}
for dataset in ['yellow_trip_hourly','insta_market_basket','intermittent_frozen_5000']:
 b=next(r for r in reference['rows'] if r['dataset']==dataset and r['model']=='TitanTPP(B)')
 bp=(BASE/b['summary_file']).with_name('history.json');bh=read(bp)['history']
 assert sha(bp)==b['history_sha256']
 if dataset=='intermittent_frozen_5000':
  cp=next((EXT/'remote').rglob('history.json'));audit=read(EXT/'remote/audit.json')
  selected_epoch=audit['best_epoch'];metrics=audit['metrics']
 else:
  cp=next(p for p in (BASE/'5090').rglob('history.json') if f'seed42_screening_{dataset}' in str(p))
  frozen=next(r for r in campaign['rows'] if r['dataset']==dataset)
  selected_epoch=frozen['best_epoch'];metrics={k:v['candidate'] for k,v in frozen['metrics'].items()}
 ch=read(cp)['history'];best=min(ch,key=lambda r:r['val_qty_rmse'])
 assert best['epoch']==selected_epoch
 for metric,key in [('raw_rmse','val_qty_rmse'),('overall_mae','val_qty_mae'),('clamped_time_loss','val_time_nll')]:assert best[key]==metrics[metric]
 detail={'dataset':dataset,'selected_B_epoch':b['best_epoch'],'selected_candidate_epoch':selected_epoch,'history_points_with_RMSE_below_selected_B_and_legacy_time_within_B_plus_0_01':sum(r['val_qty_rmse']<b['raw_rmse'] and r['val_time_nll']<=b['legacy_clamped_time_loss']+.01 for r in ch)}
 if dataset=='yellow_trip_hourly':
  detail['same_epoch_examples']=[]
  for epoch in [45,83]:
   for model,h in [('B',bh),('candidate',ch)]:
    row=next(r for r in h if r['epoch']==epoch)
    detail['same_epoch_examples'].append({'model':model,**{key:row[key] for key in ['epoch','val_qty_rmse','val_qty_mae','val_time_nll']}})
 rows.append(detail)
 for p in [bp,cp]:sources[str(p.relative_to(ROOT))]=sha(p)
result={'status':'passed','scope':'retrospective_validation_history_audit','rows':rows,'source_sha256':sources,'checkpoint_loaded':False,'checkpoint_reselected':False,'original_campaign_decision_changed':False,'limitations':['Counts are retrospective explanatory diagnostics, not a new selection rule.','Only whole-population metrics exist at each history point; Body/tail and checkpoint availability are not established for epoch45.','Same-epoch differences describe training trajectories, not proof that encoder sharing is or is not causal.']}
(OUT/'selector_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: 3 frozen datasets, 6 history hashes, selected metrics match original receipts. No checkpoint selection changed.')
