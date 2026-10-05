import pathlib,json,subprocess,shlex,hashlib,concurrent.futures,time,datetime
stage=pathlib.Path('/private/tmp/cnn_gru54_3seed_switch');c=json.loads((stage/'contract/execution_contract.json').read_text());start=json.loads((stage/'contract/start_permit.json').read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
qs={h:json.loads((stage/'contract'/('qualification_'+h+'.json')).read_text()) for h in ['5080','5090']}
for h,q in qs.items():
 assert q['status']=='passed' and q['host']==h and q['contract_sha256']==digest(c) and q['source_files_sha256']==c['source']['files_sha256'] and q['operation_files_sha256']==c['operation']['files_sha256'] and q['held_out_test_evaluated'] is False
 assert set(q['inputs'])==set(c['hosts'][h]['assigned_datasets']) and all(set(q['initialization'][d])=={'52','62'} and set(q['anchors42'][d])=={'selected','last'} for d in c['hosts'][h]['assigned_datasets'])
def run(h):
 v=c['hosts'][h];permit={'schema':'titantpp_cnn_gru54_3seed_training_permit_v1','host':h,'contract_sha256':digest(c),'start_permit_sha256':digest(start),'qualifications':qs,'approved_user_instruction':'지금 진행중인 실험 중지하고 cnn+gru54로 5080/5090 실험진행하자 승인할게','held_out_test_evaluated':False,'issued_unix':time.time(),'automatic_retry':False}
 p=stage/'contract'/('training_permit_'+h+'.json');p.write_text(json.dumps(permit,ensure_ascii=False,indent=2)+'\n')
 program=r'''import pathlib,json,subprocess,shlex,time
root=pathlib.Path(ROOT);c=json.loads((root/'execution_contract.json').read_text());v=c['hosts'][HOST]
assert json.loads((root/'qualification/receipt.json').read_text())==PERMIT['qualifications'][HOST]
assert not (root/'qualification/failure.json').exists() and not (root/'failure.json').exists() and not (root/'training_permit.json').exists()
assert time.time()<json.loads((root/'start_permit.json').read_text())['deadline_unix']
ps=subprocess.check_output(['ps','-eo','args'],text=True);assert not any(str(root/'operation/campaign.py') in l and ('--mode qualify' in l or '--mode dispatch' in l or '--mode fit' in l) for l in ps.splitlines())
assert json.loads((root/'qualification_process_exit.json').read_text())['returncode']==0
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
(root/'training_permit.json').write_text(json.dumps(PERMIT,ensure_ascii=False,indent=2)+'\n')
launcher=r"""import pathlib,json,subprocess,time,os
root=pathlib.Path(__file__).parent;c=json.loads((root/'execution_contract.json').read_text());v=c['hosts'][HOST];start=json.loads((root/'start_permit.json').read_text())
command=['timeout','--signal=TERM','--kill-after=15s',str(max(1,int(start['deadline_unix']-time.time()))),v['python'],str(root/'operation/campaign.py'),'--contract',str(root/'execution_contract.json'),'--host',HOST,'--mode','dispatch']
with (root/'logs/supervisor.log').open('xb') as stream:
 result=subprocess.run(command,cwd=v['source_root'],env={**os.environ,**v['environment']},stdout=stream,stderr=subprocess.STDOUT)
(root/'supervisor_process_exit.json').write_text(json.dumps({'returncode':result.returncode,'time':time.time(),'command':command}))
""".replace('HOST',repr(HOST))
(root/'launch_training.py').write_text(launcher);session=v['tmux']+'_train';subprocess.run([v['tmux_binary'],'new-session','-d','-s',session,'python3 '+shlex.quote(str(root/'launch_training.py'))],check=True)
receipt={'status':'training_supervisor_launched','host':HOST,'root':str(root),'tmux':session,'issued_unix':time.time(),'starts_new_seeds':[52,62],'seed42_retrained':False,'automatic_retry':False}
(root/'training_launch_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
 program='\n'.join(k+'='+repr(x) for k,x in {'ROOT':v['root'],'HOST':h,'PERMIT':permit}.items())+'\n'+program
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',h,'python3 -'],input=program,text=True,capture_output=True,timeout=45);value={'host':h,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};(stage/('training_launch_result_'+h+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2));return value
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
 for x in e.map(run,['5080','5090']):print(json.dumps(x,ensure_ascii=False))
