"""Supplemental float32 environment-compatibility audit; preserves original failure."""
from pathlib import Path
import hashlib,json,math
HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 c=json.loads((HERE/'contract.json').read_text());p=json.loads((HERE/'numeric_replay_policy.json').read_text());a=json.loads((HERE/'validation_inference_audit.json').read_text());checks={};scalars={};baseline={}
 for role,spec in c['models'].items():
  sm=json.loads(Path(spec['summary']).read_text());baseline[role]={k:sm['best_val_'+k] for k in p['checks']['metrics']};scalars[role]={}
  for key,ref in baseline[role].items():
   actual=a['metrics'][role][key];limit=p['checks']['metric_atol']+p['checks']['metric_rtol']*abs(ref);ok=abs(actual-ref)<=limit
   scalars[role][key]={'original':ref,'cpu_replay':actual,'difference':actual-ref,'limit':limit,'passed':ok};checks[f'{role}.{key}']=ok
  checks[role+'.exact_state']=a['model_audits'][role]['before_state_sha256']==spec['checkpoint_state_sha256'] and a['model_audits'][role]['state_unchanged']
 checks['exact_population']=a['population']==c['populations']['validation'];contrasts={}
 for role in ['b','rmtpp','thp']:
  original=baseline['candidate']['qty_rmse']-baseline[role]['qty_rmse'];replay=a['metrics']['candidate']['qty_rmse']-a['metrics'][role]['qty_rmse'];relative=abs(replay-original)/abs(original);ok=original*replay>0 and relative<=p['checks']['raw_rmse_contrast_absolute_perturbation_at_most_fraction_of_original']
  contrasts[role]={'original_delta_rmse':original,'replay_delta_rmse':replay,'fractional_distortion':relative,'passed':ok};checks['contrast.'+role]=ok
 result={'status':'passed' if all(checks.values()) else 'failed','original_absolute_replay_passed':a['validation_replay_passed'],'policy_sha256':sha(HERE/'numeric_replay_policy.json'),'inference_audit_sha256':sha(HERE/'validation_inference_audit.json'),'scalars':scalars,'contrasts':contrasts,'checks':checks,'limitation':'Aggregate numerical equivalence under explicitly amended float32 CPU/CUDA tolerance; does not assert bitwise per-row equivalence. No training, selection, or predictions changed.'}
 (HERE/'numeric_compatibility_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
 if result['status']!='passed':raise ValueError('Numerical compatibility audit failed')
if __name__=='__main__':main()
