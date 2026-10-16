
import hashlib,io,json,os,pathlib,shlex,subprocess,sys,tarfile,time
root=pathlib.Path(sys.argv[1]);expected=sys.argv[2];archive_sha=sys.argv[3];host=sys.argv[4]
assert host in ('5080','5090') and root.name in ['titantpp_cnn_memory_sensitivity_20261015_'+v+'_'+host for v in ('v1','v2')]
assert root.parent==pathlib.Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts')
assert not root.exists(), 'Existing native root; no retry or overwrite'
raw=sys.stdin.buffer.read();assert hashlib.sha256(raw).hexdigest()==archive_sha
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 cbytes=tar.extractfile('contract.json').read();assert hashlib.sha256(cbytes).hexdigest()==expected
 c=json.loads(cbytes);assert c['remote_roots'][host]==str(root) and time.time()<c['deadline_unix']
 assert c['allowed_splits']==['train','validation'] and c['new_training'] is False
 if 'operational_repair' in c:
  old=root.with_name(root.name.replace('_v2_','_v1_'))
  assert hashlib.sha256((old/'contract.json').read_bytes()).hexdigest()==c['operational_repair']['prior_contract_raw_sha256']
  previous=json.loads((old/'runtime'/host/'receipt.json').read_text())
  assert previous['status']=='failed' and previous['actual_worker_exit0_count']==0
  assert json.loads((old/'launch/dispatcher_exit.json').read_text())['actual_dispatcher_os_exit']==1
  assert not list((old/'results').glob('*/*/receipt.json'))
  for worker in previous['workers']:
   err=old/'runtime'/host/'workers'/worker['job_id']/(worker['split']+'.stderr')
   assert 'Could not locate the paper_research project root' in err.read_text()
  assert c['deadline_unix']==json.loads((old/'contract.json').read_text())['deadline_unix']
 files={**c['input_files'],**c['operation_files']}
 assert not set(c['input_files'])&set(c['operation_files'])
 assert {m.name for m in tar.getmembers()}==set(files)|{'contract.json'}
 for m in tar.getmembers():
  p=pathlib.Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
  if m.name!='contract.json':assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==files[m.name]
 spec=c['runtimes'][host];env={**os.environ,**c['execution'][host]['environment']}
 assert pathlib.Path(spec['python']).is_file() and pathlib.Path(c['execution'][host]['tmux_binary']).is_file()
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],timeout=20).decode().strip().splitlines()
 assert gpu==[spec['gpu_uuid']], 'Wrong GPU identity'
 busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],timeout=20).decode().strip()
 assert not busy, 'Existing compute process; no changes made'
 probe="import json,platform,torch,numpy,polars;print(json.dumps(dict(python_version=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),numpy=numpy.__version__,polars=polars.__version__)))"
 actual=json.loads(subprocess.check_output([spec['python'],'-c',probe],env=env,timeout=45))
 assert all(actual[k]==spec[k] for k in actual), 'Native Runtime differs'
 root.mkdir()
 for m in tar.getmembers():
  path=root/m.name;path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(tar.extractfile(m).read())
 for name,digest in files.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 supervisor="""import json,os,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]);host=sys.argv[2];digest=sys.argv[3]
c=json.loads((root/'contract.json').read_text());d=root/'launch';d.mkdir(exist_ok=True)
argv=[sys.executable,str(root/'operation/dispatch.py'),'--root',str(root),'--contract-sha',digest,'--host',host]
with (d/'supervisor_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),argv=argv,started_unix=time.time()),f)
with (d/'dispatcher.stdout').open('xb') as out,(d/'dispatcher.stderr').open('xb') as err:
 p=subprocess.Popen(argv,stdout=out,stderr=err,stdin=subprocess.DEVNULL);code=p.wait()
with (d/'dispatcher_exit.json').open('x') as f:json.dump(dict(actual_dispatcher_os_exit=code,dispatcher_pid=p.pid,finished_unix=time.time(),supervisor_own_os_exit_observed=False),f)
sys.exit(code)
"""
 argv=[spec['python'],'-c',supervisor,str(root),host,expected]
 shell='cd '+shlex.quote(str(root))+' && exec env '+' '.join(shlex.quote(k+'='+v) for k,v in c['execution'][host]['environment'].items())+' '+' '.join(shlex.quote(x) for x in argv)
 subprocess.run([c['execution'][host]['tmux_binary'],'new-session','-d','-s',c['execution'][host]['tmux'],shell],check=True,capture_output=True,timeout=20)
 print(json.dumps(dict(status='started',host=host,root=str(root),contract_sha256=expected,archive_sha256=archive_sha,runtime=actual,started_unix=time.time(),max_parallel=3)))
