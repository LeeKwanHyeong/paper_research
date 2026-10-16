import hashlib,io,json,shlex,subprocess,tarfile,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from paper.scripts.titantpp_inter62_transport_evidence import recorded_run
b=Path('search_artifacts/titantpp_components_inter62_test_20261014_v1');c=json.loads((b/'contract.json').read_text())
r=c['evaluation_root_mapping']['authorized_remote_root']
names=[]
for j in c['jobs']:
 for s in ('validation','test'):
  receipt='validation_replay' if s=='validation' else 'test_metrics'
  names += ['results/'+j['id']+'/'+receipt+'.json','results/'+j['id']+'/'+receipt+'_claim.json','results/'+j['id']+'/'+s+'_worker_exit.json','worker_launches/'+s+'_'+j['id']+'.json','logs/'+s+'_'+j['id']+'.log']
names+=['pipeline_complete.json','pipeline_actual_exit.json','pipeline_supervision_claim.json','pipeline.claim','validation_gate.json','evaluation_deployment_receipt.json']
p={'root':r,'names':names,'contract_raw_sha256':hashlib.sha256((b/'contract.json').read_bytes()).hexdigest(),'contract_sha256':c['contract_sha256']}
code="""import hashlib,io,json,sys,tarfile,time
from pathlib import Path
p=json.load(sys.stdin);r=Path(p['root']);assert r.name=='titantpp_components_inter62_test_20261014_v1_5080'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest();assert sha(r/'contract.json')==p['contract_raw_sha256'];c=json.loads((r/'contract.json').read_text())
assert c['contract_sha256']==p['contract_sha256'] and c['expected_jobs']==4 and c['evaluation_scope']=='test_seed62_followup'
actual=json.loads((r/'pipeline_actual_exit.json').read_text());assert actual['actual_child_exit_observed'] and actual['exit_code']==0
complete=json.loads((r/'pipeline_complete.json').read_text());assert complete['status']=='complete' and len(complete['completed'])==8
assert not (r/'pipeline_failure.json').exists()
for j in c['jobs']:
 for s in ('validation','test'):
  q='validation_replay' if s=='validation' else 'test_metrics';o=json.loads((r/'results'/j['id']/(q+'.json')).read_text());e=json.loads((r/'results'/j['id']/(s+'_worker_exit.json')).read_text())
  assert o['status']=='complete' and o['job_id']==j['id'] and o['split']==s and o['contract_sha256']==c['contract_sha256'] and not o['retrained'] and not o['reselected'] and o['state_sha256']==j['checkpoint_state_sha256'] and o['checkpoint_sha256']==j['checkpoint']['sha256']
  assert e['actual_child_exit_observed'] and e['exit_code']==0 and e['job_id']==j['id'] and e['contract_sha256']==c['contract_sha256']
  assert sha(r/j['checkpoint']['relative_path'])==j['checkpoint']['sha256']
for name,d in json.loads((r/'originals_manifest.json').read_text())['files'].items():assert sha(r/name)==d
for n,d in c['sources']['components']['files'].items():assert sha(r/c['sources']['components']['root']/n)==d
for n,d in c['evaluator_integrity']['files'].items():assert sha(r/n)==d
spec=c['datasets']['intermittent_frozen_5000'];assert sha(r/spec['data_path'])==spec['data_sha256']
raws={n:(r/n).read_bytes() for n in p['names']};manifest={'schema':'titantpp_inter62_evaluation_result_archive_v1','files':{n:hashlib.sha256(v).hexdigest() for n,v in raws.items()},'contract_sha256':c['contract_sha256'],'actual_pipeline_exit':0,'actual_native_worker_exit0_count':8,'all_selected_binary_SHA_verified':True,'all_original60_SHA_verified':True,'source131_SHA_verified':True,'completed_unix':time.time(),'new_training':False,'partial_seed_count':1}
raws['evaluation_results_manifest.json']=(json.dumps(manifest,sort_keys=True,ensure_ascii=False,allow_nan=False,indent=2)+'\\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as t:
 for n,v in raws.items():
  info=tarfile.TarInfo(n);info.size=len(v);info.mtime=int(time.time());t.addfile(info,io.BytesIO(v))
"""
archive=b/'evaluation_results_archive.tar.gz'
with archive.open('xb') as stream:
 x=recorded_run(b,'result_retrieval',['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080','/usr/bin/python3 -c '+shlex.quote(code)],input=json.dumps(p).encode(),stdout=stream,timeout=120)
assert x.returncode==0,x.stderr.decode()
with tarfile.open(archive) as t:
 manifest=json.load(t.extractfile('evaluation_results_manifest.json'));expected=manifest['files'];assert set(expected)==set(names)
 members=t.getmembers();assert len(members)==len(expected)+1 and len(set(m.name for m in members))==len(members)
 for m in members:
  assert m.isfile() and m.name in set(expected)|{'evaluation_results_manifest.json'} and '..' not in Path(m.name).parts and not Path(m.name).is_absolute()
  raw=t.extractfile(m).read();assert m.name=='evaluation_results_manifest.json' or hashlib.sha256(raw).hexdigest()==expected[m.name]
  path=b/m.name;assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
  with path.open('xb') as f:f.write(raw)
for n,d in expected.items():assert hashlib.sha256((b/n).read_bytes()).hexdigest()==d
with (b/'evaluation_results_recovery_receipt.json').open('x') as f:json.dump({'status':'complete','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((b/'evaluation_results_manifest.json').read_bytes()).hexdigest(),'files':len(expected),'all_local_result_SHA_verified':True,'actual_pipeline_exit0_verified':True,'actual_worker_exit0_count':8,'completed_unix':time.time()},f,indent=2)
print('Recovered8 native evaluations + actual exits; archive/manifest/all46localSHA verified')
