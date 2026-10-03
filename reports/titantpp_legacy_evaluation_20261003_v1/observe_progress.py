"""Read-only progress observation for the explicitly approved 5080 evaluation."""
import datetime,json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
SCRIPT=r'''from pathlib import Path
import datetime,json,subprocess
p=Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_legacy_evaluation_20261003_v1/reports/titantpp_legacy_evaluation_20261003_v1')
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pipeline':json.loads((p/'pipeline_status.json').read_text()),'phases':{}}
for phase in ['qualification_cpu','qualification_cuda','validation_full','test']:
 d=p/'runs'/phase/'attempt1'
 if not d.exists(): continue
 status=json.loads((d/'status.json').read_text()) if (d/'status.json').exists() else {}
 result['phases'][phase]={'status':status.get('status'),'current':status.get('current'),'completed':len(status.get('completed',[])),'failed':len(status.get('failures',[])),'total':status.get('total'),'updated_utc':status.get('updated_utc'),'unstarted':len(status['unstarted']) if 'unstarted' in status else None,'receipts':len(list(d.glob('*/receipt.json'))),'failure_records':len(list(d.glob('*/failure.json')))}
 if status.get('current'):
  progress=d/status['current']/'progress.json'
  if progress.exists():result['phases'][phase]['current_saved_predictions']=json.loads(progress.read_text()).get('prediction_rows')
 timing=[]
 for receipt_path in sorted(d.glob('*/receipt.json')):
  receipt=json.loads(receipt_path.read_text())
  timing.append({k:receipt.get(k) for k in ['dataset','model','seed','prediction_rows','elapsed_seconds','inference_seconds']})
 result['phases'][phase]['timing']=timing
 if status.get('failures'):result['phases'][phase]['failures']=status['failures']
 if (d/'terminal_manifest.json').exists():result['phases'][phase]['terminal']=json.loads((d/'terminal_manifest.json').read_text())
if (p/'qualification_gate.json').exists():result['gate']=json.loads((p/'qualification_gate.json').read_text())
result['processes']=[s for s in subprocess.check_output(['ps','-eo','pid,ppid,etime,args'],text=True).splitlines() if str(p.parent.parent) in s]
result['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True).strip()
result['log_tail']=(p/'pipeline.log').read_text()[-1500:]
print(json.dumps(result))
'''
def main():
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080','/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12','-'],input=SCRIPT,text=True,capture_output=True,timeout=40)
 d=json.loads(r.stdout) if r.returncode==0 else {'status':'observation_failed','stderr':r.stderr,'returncode':r.returncode}
 out=HERE/'observations'/stamp
 out.mkdir(parents=True)
 (out/'progress.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
 (HERE/'observations/latest.json').write_text(json.dumps({'path':str(out/'progress.json')})+'\n')
 shown={k:v for k,v in d.items() if k not in ('gate','log_tail','processes')}
 shown['phases']={phase:{k:v for k,v in state.items() if k not in ('timing','terminal')} for phase,state in d.get('phases',{}).items()}
 print(json.dumps(shown,ensure_ascii=False))
 if d.get('pipeline',{}).get('status')=='failed':print(d.get('log_tail',''))
if __name__=='__main__':main()
