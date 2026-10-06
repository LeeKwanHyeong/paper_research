"""One read-only SSH connection waits for the admitted evaluation supervisor exit."""
from pathlib import Path
import hashlib
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / 'search_artifacts/titantpp_cnn_gru54_test_3seed_20261006_v1'
HERE = Path(__file__).resolve().parent
REMOTE = r'''
from pathlib import Path
import json,hashlib,shlex,subprocess,shutil,time
from datetime import datetime,timezone
root=Path(REMOTE_ROOT)
assert hashlib.sha256((root/'execution_contract.json').read_bytes()).hexdigest()==CONTRACT_SHA
c=json.loads((root/'execution_contract.json').read_text())
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def processes():
 rows=[]
 for line in subprocess.check_output(['ps','-eo','pid=,ppid=,args='],text=True).splitlines():
  pid,ppid,args=line.strip().split(None,2)
  argv=shlex.split(args)
  if str(root/'operation/eval_supervisor.py') in argv or str(root/'operation/evaluate_titantpp_cnn_gru_frozen_test.py') in argv:
   rows.append({'pid':int(pid),'ppid':int(ppid),'argv':argv})
 return rows
def snapshot(stage):
 owned=processes();p=read(root/'pipeline_status.json');g=read(root/'validation_gate.json')
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 assert uuid==c['resources']['gpu_uuid']
 gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
 return {'stage':stage,'observed_utc':datetime.now(timezone.utc).isoformat(),'host':'5080','root':str(root),
  'contract_sha256':CONTRACT_SHA,'gpu_uuid':uuid,'gpu_processes':gpu,'owned_processes':owned,
  'pipeline_status':p.get('status','not_yet_written'),'active_split':p.get('split'),'active_condition':p.get('condition'),
  'completed_population_evaluations':len(p.get('completed',[])),
  'validation_gate_passed':g.get('passed',False),'validation_gate_conditions':g.get('conditions',0),
  'supervisor_exit':read(root/'supervisor_process_exit.json'),'performance_values_read':False}
initial=snapshot('initial');print(json.dumps(initial),flush=True)
supervisors=[p for p in initial['owned_processes'] if p['argv']==[c['resources']['python'],str(root/'operation/eval_supervisor.py')]]
assert len(supervisors)<=1
if supervisors:
 assert shutil.which('tail')
 subprocess.run(['tail','--pid='+str(supervisors[0]['pid']),'-f','/dev/null'],check=True,timeout=c['resources']['campaign_timeout_seconds']+30)
final=snapshot('terminal');assert not final['owned_processes'],'Owned evaluation processes remain'
exit=final['supervisor_exit'];assert exit and exit['contract_sha256']==CONTRACT_SHA
print(json.dumps(final),flush=True)
'''


def main():
    c = json.loads((BUNDLE / 'execution_contract.json').read_text())
    digest = hashlib.sha256((BUNDLE / 'execution_contract.json').read_bytes()).hexdigest()
    code = REMOTE.replace('REMOTE_ROOT', repr(c['resources']['root'])).replace('CONTRACT_SHA', repr(digest))
    process = subprocess.Popen(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080','python3 -c '+shlex.quote(code)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in process.stdout:
        result = json.loads(line)
        p = HERE / ('native_' + result['stage'] + '_observation.json')
        with p.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps(result), flush=True)
    error = process.stderr.read()
    returncode = process.wait()
    (HERE / 'native_wait_transport.log').write_text(error)
    assert returncode == 0, error


if __name__ == '__main__':
    main()
