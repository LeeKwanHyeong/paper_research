import pathlib,json,subprocess,time,hashlib,os
r=pathlib.Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_diagnostics_20261012_v2_5080')
def read(p):return json.loads(p.read_text()) if p.exists() else None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
state=read(r/'control/pipeline_latest.json')
files={str(p.relative_to(r)):{'sha256':sha(p),'value':read(p)} for p in [r/'control/pipeline_complete.json',r/'control/pipeline_failure.json',r/'control/pipeline_exit.json',r/'control/native_qualification.json'] if p.exists()}
active=(state or {}).get('active',[])
process=[]
for x in active+([{'pid':state['pid']}] if state else []):
 pid=x['pid'];base=pathlib.Path('/proc')/str(pid)
 process.append({'pid':pid,'exists':base.exists(),'argv':base.joinpath('cmdline').read_bytes().decode().replace('\0',' ').strip() if base.exists() else None,'status':base.joinpath('status').read_text().splitlines()[:8] if base.exists() else None})
def smi(args):
 x=subprocess.run(['nvidia-smi']+args,capture_output=True,text=True)
 return {'rc':x.returncode,'stdout':x.stdout.strip(),'stderr':x.stderr.strip()}
logs={}
if files.get('control/pipeline_failure.json'):
 for p in r.glob('results/*/*worker.log'):
  if not (p.parent/'validation_replay.json').exists() or not (p.parent/'test_metrics.json').exists():logs[str(p.relative_to(r))]=p.read_text()[-6000:]
 logs['control/pipeline.log']=(r/'control/pipeline.log').read_text()[-6000:]
result={'observed_unix':time.time(),'state':state,'terminal_files':files,'processes':process,'gpu':smi(['--query-gpu=uuid,name,utilization.gpu,memory.used','--format=csv,noheader']),'gpu_processes':smi(['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader']),'counts':{'native_probe_exits':len(list(r.glob('control/native_probes/*/worker_exit.json'))),'validation_receipts':len(list(r.glob('results/*/validation_replay.json'))),'test_receipts':len(list(r.glob('results/*/test_metrics.json'))),'combined_receipts':len(list(r.glob('results/*/receipt.json')))},'failure_logs':logs}
print(json.dumps(result,ensure_ascii=False))
