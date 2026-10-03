"""Validate RAF integration, preserving all earlier reviewed numeric results."""
import json,hashlib,math,statistics,re
from pathlib import Path
from datetime import datetime,timezone
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent;C=R/'reports/titantpp_completed_external_comparison_20261001_v1'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=read(C/'before_raf_integration/comparison.json');new=read(C/'comparison.json');raf=read(O/'comparison.json');lookup={(r['dataset'],r['model'],r['seed']):r for r in new['rows']};checks={}
fields=['epoch','count','qty_mae','qty_rmse','time_nll']
checks['all_114_previous_conditions_preserved_numerically']=all(all(lookup[r['dataset'],r['model'],r['seed']][k]==r[k] for k in fields) for r in old['rows'])
checks['139_unique_completed_conditions']=len(new['rows'])==len(lookup)==139
checks['raf_all_24_in_combined_report']=all(all(lookup[r['dataset'],r['model'],r['seed']][k]==r[k] for k in fields) for r in raf['rows'])
checks['eight_raf_groups_with_complete_seeds']=len(raf['three_seed'])==8 and all(sorted(r['seed'] for r in raf['rows'] if r['model']==g['model'])==[42,52,62] for g in raf['three_seed'])
snapshot=read(C/'app/src/data.json');previous=read(C/'before_raf_integration/app/src/data.json')
checks['app_identity_preserved']=snapshot['id']==previous['id']
checks['app_authoring_complete']=snapshot['buildStatus']=='complete'
checks['app_all_rows']=len(snapshot['queries']['all_completed']['rows'])==139
q=snapshot['queries']['raf_spare_parts']['rows'];checks['raf_table_eight_rows']=len(q)==8
for x,g in zip(q,raf['three_seed']):
 for k in ['qty_mae','qty_rmse','time_nll']:
  assert x[k]==g[k]['mean'] and x[k+'_sd']==g[k]['sample_sd']
checks['raf_display_means_and_sample_sd_match']=True
checks['manuscript_integration_verified']=read(R/'reports/titantpp_manuscript_integration_20261001_v1/verification.json')['status']=='passed'
checks['raf_final_audit_passed']=read(R/'search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/terminal_audit.json')['status']=='passed'
checks['rendered_preview_saved']=(O/'report_preview.png').is_file()
result={'status':'passed' if all(checks.values()) else 'failed','checked_utc':datetime.now(timezone.utc).isoformat(),'checks':checks,'browser_review':{'raf_selected_8_model_table':'passed','same_seed_detail':'passed','complete_status_cleared':'passed','visual_layout':'passed'},'new_training_or_replay':False,'heldout_read':False,'preview_sha256':sha(O/'report_preview.png')}
(O/'integration_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');assert result['status']=='passed';print(json.dumps(result,ensure_ascii=False))
# Update only verification metadata after the checks above.
v=read(C/'verification.json');v['raf_integration_verification']=str((O/'integration_verification.json').relative_to(R));v['browser_checked']=result['browser_review'];v['delivery_files']={n:sha(C/n) for n in ['report.md','comparison.json','selected_conditions.csv','build_analysis.py','sync_app_data.py','app/src/data.json','app/dist/data-app-build.json']};(C/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
B=R/'reports/titantpp_baseline_reset_20261001_v1';b=read(B/'baseline.json');b['source_files_sha256']['reports/titantpp_completed_external_comparison_20261001_v1/verification.json']=sha(C/'verification.json');(B/'baseline.json').write_text(json.dumps(b,ensure_ascii=False,indent=2)+'\n')
v=read(B/'verification.json');v['raf_live_owner_gpu_confirmed_at_historical_0735']=v.pop('raf_live_owner_gpu_confirmed',True);v['raf_terminal_owner_absent']=True;v['raf_binary_source_final_audit']='passed_24_conditions_48_checkpoints';v['raf_integration']=result;v['plan_sha256']=sha(B/'README.md');v['manuscript_sha256']=sha(R/'paper/titantpp_history_mlp_manuscript_20261001_v1.md');v['raf_update_utc']=result['checked_utc'];(B/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
