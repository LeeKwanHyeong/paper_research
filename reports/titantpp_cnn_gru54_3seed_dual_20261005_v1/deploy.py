import pathlib,json,hashlib,subprocess,shlex,tarfile,sys,concurrent.futures,time
STAGE=pathlib.Path('/private/tmp/cnn_gru54_3seed_switch');C=STAGE/'contract';c=json.loads((C/'execution_contract.json').read_text());old=json.loads(pathlib.Path('/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_gru_controls_3seed_dual_20261005_v1/execution_contract.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run(h):
 v=c['hosts'][h];archive=STAGE/('deployment_'+h+'.tar.gz')
 with tarfile.open(archive,'w:gz') as tf:
  for n in ['execution_contract.json','approval.json','start_permit.json','reuse_seed42_registry.json']:
   tf.add(C/n,arcname=n)
  for rel,digest in c['operation']['files'].items():
   p=STAGE/rel;assert sha(p)==digest;tf.add(p,arcname=rel)
  for rel,digest in c['input_files'].items():
   if rel in old['input_files'] and digest==old['input_files'][rel]:continue
   p=C/'payload/source'/rel;assert sha(p)==digest,(rel,p);tf.add(p,arcname='source/'+rel)
 digest=sha(archive);remote='/tmp/titantpp_cnn_gru54_3seed_'+h+'.tar.gz'
 subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',str(archive),h+':'+remote],check=True,timeout=120)
 program=r'''import pathlib,json,hashlib,tarfile,subprocess,shlex,shutil,time
archive=pathlib.Path(ARCHIVE);root=pathlib.Path(ROOT);oldroot=pathlib.Path(OLDROOT)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive)==ARCHIVESHA
assert not root.exists(),'New dedicated root already exists; no overwrite/retry'
oldc=json.loads((oldroot/'execution_contract.json').read_text());canon=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest();assert canon(oldc)==OLDSHA
ps=subprocess.check_output(['ps','-eo','args'],text=True)
assert not any(str(oldroot/'source/paper/scripts/run_titantpp_gru_control_campaign.py') in l and ('--mode fit' in l or '--mode dispatch' in l) for l in ps.splitlines()),'Old owner remains active'
root.mkdir();(root/'source').mkdir()
with tarfile.open(archive) as tf:
 for m in tf.getmembers():
  p=pathlib.Path(m.name);assert not p.is_absolute() and '..' not in p.parts and not m.issym() and not m.islnk()
 tf.extractall(root,filter='data')
c=json.loads((root/'execution_contract.json').read_text());v=c['hosts'][HOST];assert v['root']==str(root)
for rel,digest in c['source']['files'].items():
 src=oldroot/'source'/rel;assert sha(src)==digest;dest=root/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
for rel,digest in c['input_files'].items():
 dst=root/'source'/rel
 if not dst.exists():
  src=oldroot/'source'/rel;assert sha(src)==digest,(rel,'old input hash changed');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
 assert sha(dst)==digest,rel
for rel,digest in c['operation']['files'].items():assert sha(root/rel)==digest
(root/'logs').mkdir();(root/'cache').mkdir()
launcher=r"""import json,pathlib,subprocess,os,time
root=pathlib.Path(__file__).parent;c=json.loads((root/'execution_contract.json').read_text());v=c['hosts'][HOST]
command=['timeout','--signal=TERM','--kill-after=15s','5400',v['python'],str(root/'operation/campaign.py'),'--contract',str(root/'execution_contract.json'),'--host',HOST,'--mode','qualify']
with (root/'logs/qualification.log').open('xb') as stream:
 result=subprocess.run(command,cwd=v['source_root'],env={**os.environ,**v['environment']},stdout=stream,stderr=subprocess.STDOUT)
(root/'qualification_process_exit.json').write_text(json.dumps({'returncode':result.returncode,'time':time.time(),'command':command}))
""".replace('HOST',repr(HOST))
(root/'launch_qualification.py').write_text(launcher)
session=v['tmux']+'_qualify';subprocess.run([v['tmux_binary'],'new-session','-d','-s',session,'python3 '+shlex.quote(str(root/'launch_qualification.py'))],check=True)
receipt={'status':'qualification_launched','host':HOST,'root':str(root),'archive_sha256':ARCHIVESHA,'source_closure_sha256':c['source']['files_sha256'],'operation_closure_sha256':c['operation']['files_sha256'],'contract_sha256':canon(c),'tmux':session,'utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
(root/'deployment_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
'''
 program='\n'.join(k+'='+repr(x) for k,x in {'ARCHIVE':remote,'ROOT':v['root'],'OLDROOT':old['hosts'][h]['root'],'ARCHIVESHA':digest,'OLDSHA':'221e98d5f489842832a359d0c2d4d1ddeb4b22d0fbe08e5e564622a6c7f11b5f','HOST':h}.items())+'\n'+program
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',h,'python3 -'],input=program,text=True,capture_output=True,timeout=150)
 value={'host':h,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr};(STAGE/('deployment_result_'+h+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2));return value
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as e:
 for value in e.map(run,['5080','5090']):print(json.dumps(value,ensure_ascii=False))
