from pathlib import Path
import json,subprocess,concurrent.futures,shlex,datetime
s=Path('/private/tmp/cnn_gru54_3seed_switch');c=json.loads((s/'contract/execution_contract.json').read_text())
def run(h):
 v=c['hosts'][h];program=r'''from pathlib import Path
import json,subprocess,shutil,datetime,hashlib,shlex
root=Path(ROOT);c=json.loads((root/'execution_contract.json').read_text());v=c['hosts'][HOST]
assert json.loads((root/'qualification_process_exit.json').read_text())['returncode']==1
assert not (root/'qualification.claim').exists() and not (root/'qualification/receipt.json').exists() and not (root/'qualification/failure.json').exists() and not (root/'run').exists(),'Native qualification or training already entered; repair is not authorized retry'
log=(root/'logs/qualification.log').read_text();assert 'Could not locate the paper_research project root' in log
ps=subprocess.check_output(['ps','-eo','args'],text=True);assert not any(str(root/'operation/campaign.py') in x for x in ps.splitlines())
attempt=root/'qualification_attempts/bootstrap_layout_failure';attempt.mkdir(parents=True,exist_ok=False)
shutil.copy2(root/'logs/qualification.log',attempt/'qualification.log');shutil.copy2(root/'qualification_process_exit.json',attempt/'qualification_process_exit.json')
for name in ['models','utils','sample_data']:(root/'source'/name).mkdir(exist_ok=True)
(root/'source/sample_data/.gitkeep').write_text('Repository-root sentinel; no research data.\n')
for rel,digest in c['source']['files'].items():assert hashlib.sha256((root/'source'/rel).read_bytes()).hexdigest()==digest
for rel,digest in c['operation']['files'].items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest
(root/'logs/qualification.log').rename(attempt/'launch_log_original.log');(root/'qualification_process_exit.json').rename(attempt/'launch_exit_original.json')
# A distinct supervised verification attempt is reviewed after correcting the package layout.
script=(root/'launch_qualification.py').read_text().replace('logs/qualification.log','logs/qualification_attempt2.log').replace('qualification_process_exit.json','qualification_attempt2_process_exit.json')
(root/'launch_qualification_attempt2.py').write_text(script)
session=v['tmux']+'_qualify2';subprocess.run([v['tmux_binary'],'new-session','-d','-s',session,'python3 '+shlex.quote(str(root/'launch_qualification_attempt2.py'))],check=True)
r={'host':HOST,'status':'layout_only_repair_then_manual_qualification_attempt2','source_code_changed':False,'operation_code_changed':False,'scientific_training_started':False,'first_failure_preserved':str(attempt),'root_sentinels':['models','utils','sample_data'],'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(root/'layout_repair_receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
'''
 program='\n'.join(k+'='+repr(x) for k,x in {'ROOT':v['root'],'HOST':h}.items())+'\n'+program
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',h,'python3 -'],input=program,text=True,capture_output=True,timeout=30);x={'host':h,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};(s/('layout_repair_'+h+'.json')).write_text(json.dumps(x,ensure_ascii=False,indent=2));return x
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
 for x in e.map(run,['5080','5090']):print(json.dumps(x,ensure_ascii=False))
