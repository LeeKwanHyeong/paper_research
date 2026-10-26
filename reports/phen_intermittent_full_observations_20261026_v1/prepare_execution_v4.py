import hashlib,json,pathlib,shutil
P=pathlib.Path(__file__).resolve().parent;old=P/'execution_v3';E=P/'execution_v4'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def remap(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return [remap(v,a,b) for v in x]
 if isinstance(x,dict):return {k:remap(v,a,b) for k,v in x.items()}
 return x
E.mkdir();shutil.copytree(old/'bundle',E/'bundle');shutil.copy2(old/'collector_v3.py',E/'collector_v3.py')
for name in ['remote_control.py','control_transport.py','test_remote_control.py','test_transport.py']:
 text=(old/name).read_text().replace('_SPEC_v3.json','_SPEC_v4.json')
 with (E/name).open('x') as f:f.write(text)
c=json.loads((P/'bundle/contracts/seed42/seed42.json').read_text())
for host in ['5080','5090']:
 g=json.loads((old/(host+'_GUARD.json')).read_text());previous=g['remote_root'];current=previous.replace('_execution_v3','_execution_v4');g['remote_root']=current
 diag=json.loads((P/'failed_gpu_qualification_v3'/host/'RUNTIME_DIAGNOSTIC.json').read_text())
 environment=dict(c['hosts'][host]['environment']);environment['SOURCE_REVISION']='b1455baec1c084e422ce044dec7cd454acd95631' if host=='5080' else '5d8ee1d64601a67d5ae0d0b4b97dd5547b63ebdd';environment['PYTHONPATH']=''
 environment.update(XDG_CACHE_HOME=current+'/cache',MPLCONFIGDIR=current+'/cache/matplotlib',TORCHINDUCTOR_CACHE_DIR=current+'/cache/torchinductor')
 g.update(process_environment=environment,runtime_libraries=diag['libraries']);write(E/(host+'_GUARD.json'),g)
 for mode in ['QUALIFICATION','FULL']:
  op=old/(host+'_'+mode+'_SPEC_v3.json');s=remap(json.loads(op.read_text()),previous,current)
  s.update(process_environment=environment,operational_revision=dict(previous_spec_sha256=sha(op),reason='Restore frozen per-process CUDA library and thread environment; collector/model/checkpoint/inputs/tolerances unchanged',failed_attempt_preserved='failed_gpu_qualification_v3/'+host))
  write(E/(host+'_'+mode+'_SPEC_v4.json'),s)
write(E/'REVISION.json',dict(collector_sha256=sha(E/'collector_v3.py'),previous_failed='failed_gpu_qualification_v3',changed='process-local runtime environment and new isolated paths only',public_runtime_modified=False))
print(str(E))
