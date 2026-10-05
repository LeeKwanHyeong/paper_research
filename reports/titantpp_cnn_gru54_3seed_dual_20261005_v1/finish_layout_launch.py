from pathlib import Path
import json,subprocess,concurrent.futures
s=Path('/private/tmp/cnn_gru54_3seed_switch');c=json.loads((s/'contract/execution_contract.json').read_text())
def run(h):
 program=r'''from pathlib import Path
import json,subprocess,shlex,datetime,hashlib
root=Path(ROOT);c=json.loads((root/'execution_contract.json').read_text());v=c['hosts'][HOST]
assert not (root/'qualification.claim').exists() and not (root/'qualification/receipt.json').exists() and not (root/'qualification/failure.json').exists() and not (root/'run').exists()
assert (root/'qualification_attempts/bootstrap_layout_failure/launch_exit_original.json').exists()
assert (root/'launch_qualification_attempt2.py').exists() and not (root/'logs/qualification_attempt2.log').exists()
ps=subprocess.check_output(['ps','-eo','args'],text=True);assert not any(str(root/'operation/campaign.py') in x for x in ps.splitlines())
assert all((root/'source'/n).is_dir() for n in ['models','utils','sample_data'])
for rel,d in c['source']['files'].items():assert hashlib.sha256((root/'source'/rel).read_bytes()).hexdigest()==d
for rel,d in c['operation']['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==d
session=v['tmux']+'_qualify2';subprocess.run([v['tmux_binary'],'new-session','-d','-s',session,'python3 '+shlex.quote(str(root/'launch_qualification_attempt2.py'))],check=True)
r={'host':HOST,'status':'layout_repaired_native_qualification_attempt2_launched','source_code_changed':False,'operation_code_changed':False,'scientific_training_started':False,'first_failure_preserved':str(root/'qualification_attempts/bootstrap_layout_failure'),'layout_marker_sha256':hashlib.sha256((root/'source/sample_data/.gitkeep').read_bytes()).hexdigest(),'purpose':'project root discovery only; no data file','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'layout_repair_receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
 program='HOST='+repr(h)+'\nROOT='+repr(c['hosts'][h]['root'])+'\n'+program
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',h,'python3 -'],input=program,text=True,capture_output=True,timeout=30);x={'host':h,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};(s/('layout_repair_launch_'+h+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2));return x
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
 for x in e.map(run,['5080','5090']):print(json.dumps(x,ensure_ascii=False))
