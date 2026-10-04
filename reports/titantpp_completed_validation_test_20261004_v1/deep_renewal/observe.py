"""One bounded read-only snapshot of the owned Deep Renewal evaluation."""
from pathlib import Path
import json
import shlex
import subprocess
from datetime import datetime, timezone

HERE=Path(__file__).resolve().parent

def main():
    contract=json.loads((HERE/'execution_contract.json').read_text())
    resources=contract['resources']
    code=r'''
from pathlib import Path
import json,subprocess,datetime
root=Path(REMOTE_ROOT);here=root/'reports/titantpp_completed_validation_test_20261004_v1/deep_renewal'
read=lambda p:json.loads(p.read_text()) if p.exists() else None
status=read(here/'pipeline_status.json')
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':status,
 'validation_gate':read(here/'validation_gate.json'),'inference_completion':read(here/'inference_completion.json')}
if status and status.get('status')=='running':
 folder=here/'runs'/status['split']/status['condition']
 result['progress']=read(folder/'progress.json')
 result['failure']=read(folder/'failure.json')
for name,args in [('gpu',['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used','--format=csv,noheader']),('gpu_compute',['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])]:
 result[name]=subprocess.check_output(args,text=True,timeout=15).strip()
processes=subprocess.check_output(['ps','-eo','pid=,ppid=,pgid=,args='],text=True,timeout=15)
result['owned_processes']=[line.strip() for line in processes.splitlines() if str(here/'pipeline.py') in line or str(here/'evaluate.py') in line]
result['tmux_running']=subprocess.run(['/usr/bin/tmux','has-session','-t','titantpp_deep_selected_vt_20261004'],capture_output=True).returncode==0
if status and status.get('status')=='failed':
 result['controller_log_tail']=(here/'controller.log').read_text()[-6000:]
 result['failures']={str(p.relative_to(here)):read(p) for p in (here/'runs').glob('*/*/failure.json')}
print(json.dumps(result))
'''.replace('REMOTE_ROOT',repr(resources['root']))
    value=subprocess.check_output(['ssh','5080',shlex.join([resources['python'],'-B','-c',code])],text=True,timeout=60)
    result=json.loads(value.strip().splitlines()[-1])
    directory=HERE/'observations';directory.mkdir(exist_ok=True)
    path=directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'latest_observation.json').write_text(json.dumps({'path':str(path),'observed_utc':result['observed_utc']},indent=2)+'\n')
    status=result.get('status') or {}
    compact={k:status.get(k) for k in ('status','split','condition','pid','exception','message') if k in status}
    compact['completed_split_count']=len(status.get('completed',[]))
    compact['validation_completed']=sum(x['split']=='validation' for x in status.get('completed',[]))
    compact['test_completed']=sum(x['split']=='test' for x in status.get('completed',[]))
    print(json.dumps({'path':str(path),'observed_utc':result['observed_utc'],'status':compact,
                      'gpu':result['gpu'],'gpu_compute':result['gpu_compute'],
                      'owned_process_count':len(result['owned_processes']),'tmux_running':result['tmux_running'],
                      'failure':result.get('failure'),'failures':result.get('failures')}))

if __name__=='__main__':main()
