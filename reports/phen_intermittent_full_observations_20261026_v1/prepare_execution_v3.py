"""New exclusive attempt after UUID representation diagnosis; previous attempts stay intact."""
import hashlib,json,pathlib,shutil
P=pathlib.Path(__file__).resolve().parent;E=P/'execution_v3'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2)
def remap(x,a,b):
 if isinstance(x,str):return x.replace(a,b)
 if isinstance(x,list):return [remap(v,a,b) for v in x]
 if isinstance(x,dict):return {k:remap(v,a,b) for k,v in x.items()}
 return x
assert (P/'collector_v3.py').is_file();E.mkdir();shutil.copytree(P/'bundle',E/'bundle')
shutil.copy2(P/'collector_v3.py',E/'collector_v3.py')
for name in ['remote_control.py','control_transport.py','test_remote_control.py','test_transport.py']:
 text=(P/name).read_text().replace('collector_v2.py','collector_v3.py').replace('_SPEC_v2.json','_SPEC_v3.json')
 with (E/name).open('x') as f:f.write(text)
for host in ['5080','5090']:
 g=json.loads((P/(host+'_GUARD.json')).read_text());oldroot=g['remote_root'];newroot=oldroot+'_execution_v3';g['remote_root']=newroot;write(E/(host+'_GUARD.json'),g)
 for mode in ['QUALIFICATION','FULL']:
  old=P/(host+'_'+mode+'_SPEC_v2.json');s=remap(json.loads(old.read_text()),oldroot,newroot)
  s.update(runner_sha256=sha(E/'collector_v3.py'),runner_filename='collector_v3.py',operational_revision=dict(previous_spec_sha256=sha(old),reason='Canonicalize optional NVIDIA GPU- UUID prefix; actual device UUID, boot, source, input, checkpoint and scientific tolerances unchanged',failed_attempt_preserved='failed_gpu_qualification_v2/'+host))
  write(E/(host+'_'+mode+'_SPEC_v3.json'),s)
write(E/'REVISION.json',dict(reason='Only UUID representation handling and isolated execution paths changed',collector_sha256=sha(E/'collector_v3.py'),previous_cpu_qualification='CPU_QUALIFICATION_02.receipt.json',previous_gpu_attempt='failed_gpu_qualification_v2',new_training=False))
print(json.dumps(dict(root=str(E),collector_sha256=sha(E/'collector_v3.py'))))
