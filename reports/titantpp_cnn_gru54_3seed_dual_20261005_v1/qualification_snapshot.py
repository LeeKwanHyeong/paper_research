import pathlib,json,subprocess,shlex,concurrent.futures
stage=pathlib.Path('/private/tmp/cnn_gru54_3seed_switch');c=json.loads((stage/'contract/execution_contract.json').read_text())
def run(h):
 root=c['hosts'][h]['root'];program='ROOT='+repr(root)+'\n'+r'''from pathlib import Path
import json,subprocess,datetime
root=Path(ROOT)
def read(p):return json.loads(p.read_text()) if p.exists() else None
ps=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
result={'host':HOST,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'receipt':read(root/'qualification/receipt.json'),'failure':read(root/'qualification/failure.json'),'exit':read(root/'qualification_attempt2_process_exit.json') if (root/'launch_qualification_attempt2.py').exists() else read(root/'qualification_process_exit.json'),'progress':read(root/'progress.json'),'processes':[l for l in ps.splitlines() if str(root/'operation/campaign.py') in l], 'log_tail':(root/'logs/qualification_attempt2.log' if (root/'logs/qualification_attempt2.log').exists() else root/'logs/qualification.log').read_text()[-6000:]}
print(json.dumps(result))
''';program='HOST='+repr(h)+'\n'+program
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',h,'python3 -'],input=program,text=True,capture_output=True,timeout=30)
 value={'host':h,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};(stage/('qualification_snapshot_'+h+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2));
 if not r.returncode:
  x=json.loads(r.stdout)
  if x['receipt']:(stage/'contract'/('qualification_'+h+'.json')).write_text(json.dumps(x['receipt'],ensure_ascii=False,indent=2)+'\n')
  return {'host':h,'status':'passed' if (x['receipt'] or {}).get('status')=='passed' else 'failed' if x['failure'] or (x['exit'] and x['exit']['returncode']) else 'checking','exit':x['exit'],'failure':x['failure'],'progress':x['progress'],'processes':x['processes'],'log_tail':x['log_tail']}
 return value
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
 for x in e.map(run,['5080','5090']):print(json.dumps(x,ensure_ascii=False))
