"""Execute only the approved fresh Taxi/RAF24 scope in an isolated native5080 root."""
from pathlib import Path
import argparse,base64,hashlib,json,os,shlex,subprocess,sys,tarfile,time,traceback
PROJECT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(PROJECT))
from paper.scripts.titantpp_inter62_transport_evidence import recorded_run
BUNDLE=PROJECT/'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1'
REPORT=PROJECT/'reports/titantpp_components_taxi_raf24_test_20261014_v1'
PARENT='e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074'
def require(ok,message):
 if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,ensure_ascii=False,allow_nan=False,indent=2)
def verify_base():
 c=read(BUNDLE/'provenance/training_contract.json');a=read(BUNDLE/'partial_approval.json');snapshot=read(BUNDLE/'provenance/training_terminal_snapshot.json')
 require(canonical(c)==PARENT and sha(BUNDLE/'provenance/training_contract.json')=='591a31ed862148c1f7343fef33a65da9d263ec9b80769a240d1ff43cdb6b49f4','Training contract seal mismatch')
 require(sha(BUNDLE/'provenance/training_terminal_snapshot.json')=='352b65c7f5c55a11fae7720b67c26c437b6999f662cfa0b8297fe0b2c33556f2','Historical terminal snapshot mismatch')
 require(sha(BUNDLE/'provenance/training_terminal_cache.json')=='a732d32a5b5976123da6fe2c6ae92844040f7d0cc6e4c05130e1b60e742b3ce1','Terminal cache mismatch')
 require(a['approved'] is True and not a['new_training'] and a['expected_condition_count']==24 and a['job_ids']==sorted(j['id'] for j in c['jobs'] if j['host']=='5080'),'Scope mismatch')
 require(snapshot['contract_sha256']==PARENT and snapshot['source_files_sha256']==c['source']['files_sha256'] and snapshot['root']==c['hosts']['5080']['root'] and snapshot['gpu_uuid']==c['hosts']['5080']['gpu_uuid'],'Snapshot identity mismatch')
 for path in ('phases/train_exit.json','phases/train_complete.json'):
  item=snapshot['files'][path];raw=base64.b64decode(item['base64']);require(hashlib.sha256(raw).hexdigest()==item['raw_sha256'],'Dispatcher evidence changed');v=json.loads(raw)
  key='os_exit_code' if path.endswith('train_exit.json') else 'subprocess_os_exit_code';require(type(v[key]) is int and v[key]==0 and v['contract_sha256']==PARENT,'Original dispatcher not terminal0')
 for n,d in c['source']['files'].items():require(sha(BUNDLE/'source'/n)==d,'Science seal mismatch '+n)
 for n,d in c['input_files'].items():require(sha(BUNDLE/'source'/n)==d,'Input seal mismatch '+n)
 return c,a,snapshot
def transport(label,code,payload,timeout=120,archive=None):
 write(BUNDLE/'control'/(label+'_intent.json'),{'started_unix':time.time(),'automatic_retry':False,'original_training_root_writes':0})
 argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080','/usr/bin/python3 -c '+shlex.quote(code)]
 if archive:
  with Path(archive).open('xb') as f:x=recorded_run(BUNDLE,label,argv,input=json.dumps(payload).encode(),stdout=f,timeout=timeout)
  out=b''
 else:
  x=recorded_run(BUNDLE,label,argv,input=json.dumps(payload).encode(),timeout=timeout);out=x.stdout
 require(x.returncode==0,'Transport failed '+label+': '+x.stderr.decode(errors='replace')[-3000:])
 return out


def recover_originals(c,snapshot):
 expected={}
 for j in (j for j in c['jobs'] if j['host']=='5080'):
  name='run/'+j['id']+'/terminal_manifest.json';item=snapshot['files'][name];raw=base64.b64decode(item['base64']);require(hashlib.sha256(raw).hexdigest()==item['raw_sha256'],'Terminal manifest snapshot changed')
  terminal=json.loads(raw);require(terminal['job']==j and terminal['contract_sha256']==PARENT and terminal['scientific_success'] is True and terminal['held_out_test_evaluated'] is False,'Foreign scientific terminal')
  expected[name]=item['raw_sha256'];exit_name='worker_exits/'+j['id']+'.json';expected[exit_name]=snapshot['files'][exit_name]['raw_sha256']
  for n,d in terminal['files'].items():expected['run/'+j['id']+'/'+n]=d
 require(len(expected)==360,'Expected360 originals only')
 archive=BUNDLE/'originals_archive_verified.tar.gz'
 payload={'remote_original_root':c['hosts']['5080']['root'],'training_contract_raw_sha256':sha(BUNDLE/'provenance/training_contract.json'),'expected_files':expected,'job_ids':sorted(j['id'] for j in c['jobs'] if j['host']=='5080')}
 transport('retrieval_schema_corrected',(REPORT/'remote_retrieval.py').read_text(),payload,archive=archive)
 with tarfile.open(archive) as t:
  members=t.getmembers();require(len(members)==360 and {m.name for m in members}=={'originals/'+n for n in expected},'Archive membership differs')
  for m in members:
   require(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,'Unsafe archive member')
   raw=t.extractfile(m).read();require(hashlib.sha256(raw).hexdigest()==expected[m.name.removeprefix('originals/')],'Original archive SHA differs')
   path=BUNDLE/m.name;path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as f:f.write(raw)
 files={'originals/'+n:d for n,d in expected.items()}
 for n,d in files.items():require(sha(BUNDLE/n)==d,'Local original SHA differs')
 write(BUNDLE/'originals_manifest.json',{'files':files,'all_file_SHA_verified':True,'source131_SHA_verified':True,'checkpoint_selected_last_pairs':24,'remote_originals_changed':False,'archive_raw_sha256':sha(archive)})


def deploy_inputs(c,approval):
 names=['partial_approval.json','originals_manifest.json','operation/audit_inputs.py']
 names+=['provenance/'+p.name for p in sorted((BUNDLE/'provenance').glob('*.json'))]
 names+=list(read(BUNDLE/'originals_manifest.json')['files'])
 names+=['source/'+n for n in c['source']['files']]
 names+=['source/'+n for n in c['input_files']]
 names+=[str(p.relative_to(BUNDLE)) for p in sorted((BUNDLE/'data').glob('*'))]
 require(len(names)==len(set(names)),'Duplicate deployment inputs')
 files={n:sha(BUNDLE/n) for n in names};manifest={'files':files,'approval_sha256':sha(BUNDLE/'partial_approval.json')};write(BUNDLE/'deployment_integrity.json',manifest)
 archive=BUNDLE/'input_deployment.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for n in names+['deployment_integrity.json']:t.add(BUNDLE/n,arcname=n,recursive=False)
 remote=approval['authorization_root'];payload={'root':remote,'archive_sha256':sha(archive),'approval_sha256':manifest['approval_sha256'],'count':len(files)}
 code="""import json,sys\nfrom pathlib import Path\np=json.load(sys.stdin);r=Path(p['root']);assert str(r)=='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_taxi_raf24_test_20261014_v1_5080';r.mkdir(exist_ok=False);(r/'incoming').mkdir()\nwith (r/'deployment_intent.json').open('x') as f:json.dump(p,f)\n"""
 transport('reserve',code,payload)
 x=recorded_run(BUNDLE,'upload',['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(archive),'5080:'+remote+'/incoming/input_deployment.tar.gz'],timeout=120)
 require(x.returncode==0,'Input upload failed')
 code="""import hashlib,json,sys,tarfile,time\nfrom pathlib import Path\np=json.load(sys.stdin);r=Path(p['root']);assert str(r)=='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_taxi_raf24_test_20261014_v1_5080';a=r/'incoming/input_deployment.tar.gz';assert hashlib.sha256(a.read_bytes()).hexdigest()==p['archive_sha256']\nwith tarfile.open(a) as t:\n m=json.load(t.extractfile('deployment_integrity.json'));files=m['files'];assert m['approval_sha256']==p['approval_sha256'] and len(files)==p['count'];members=t.getmembers();assert len(members)==len(files)+1 and len(set(x.name for x in members))==len(members)\n for x in members:\n  n=Path(x.name);target=r/n;assert x.isfile() and not n.is_absolute() and '..' not in n.parts and target.resolve().is_relative_to(r.resolve()) and not target.exists();raw=t.extractfile(x).read();assert x.name=='deployment_integrity.json' or hashlib.sha256(raw).hexdigest()==files[x.name]\n for x in members:\n  target=r/x.name;target.parent.mkdir(parents=True,exist_ok=True)\n  with target.open('xb') as f:f.write(t.extractfile(x).read())\nwith (r/'deployment_receipt.json').open('x') as f:json.dump({'status':'deployed','files':files,'archive_sha256':p['archive_sha256'],'original_root_writes':0,'completed_unix':time.time()},f)\nprint(json.dumps({'status':'deployed','files':len(files)}))\n"""
 transport('input_deploy',code,payload)


def native_audit(c,approval):
 root=approval['authorization_root'];env={**c['hosts']['5080']['environment'],'PYTHONPATH':root+'/source','MPLCONFIGDIR':root+'/audit/initial/runtime_cache/matplotlib','XDG_CACHE_HOME':root+'/audit/initial/runtime_cache'}
 argv=['env',*[k+'='+v for k,v in sorted(env.items())],c['hosts']['5080']['python'],root+'/operation/audit_inputs.py','--root',root,'--approval-sha',sha(BUNDLE/'partial_approval.json'),'--originals-manifest-sha',sha(BUNDLE/'originals_manifest.json'),'--audit-attempt','initial']
 write(BUNDLE/'control/native_audit_intent.json',{'argv':argv,'started_unix':time.time(),'model_predictions':0,'automatic_retry':False})
 x=recorded_run(BUNDLE,'native_audit',['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080',shlex.join(argv)],timeout=900)
 require(x.returncode==0,'Native input audit failed; no automatic retry: '+x.stderr.decode(errors='replace')[-2000:])
 value=json.loads(x.stdout);require(value['status']=='complete' and value['model_predictions']==0,'Input audit incomplete')
 with (BUNDLE/'native_input_receipt.json').open('xb') as f:f.write(x.stdout)


def retrieve_native_metadata(approval):
 receipt=read(BUNDLE/'native_input_receipt.json');refs={r['metadata_path']:r['metadata_sha256'] for r in receipt['metadata_refs'].values()};require(len(refs)==24,'Expected24 metadata')
 raw=transport('metadata_retrieval',(REPORT/'retrieve_native_metadata.py').read_text(),{'root':approval['authorization_root'],'refs':refs})
 blobs=json.loads(raw);require(set(blobs)==set(refs),'Metadata membership differs')
 for n,v in blobs.items():
  value=base64.b64decode(v);require(hashlib.sha256(value).hexdigest()==refs[n],'Native metadata byteSHA differs');target=BUNDLE/n;require(not Path(n).is_absolute() and '..' not in Path(n).parts and target.resolve().is_relative_to(BUNDLE.resolve()),'Metadata escapes root');target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as f:f.write(value)


def run_helper(name,*args):
 x=subprocess.run([sys.executable,str(REPORT/name),*args],cwd=PROJECT,timeout=43500)
 require(x.returncode==0,'Helper failed; automatic retry forbidden: '+name)



def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['recover','audit','evaluate'],required=True);a=p.parse_args();c,approval,snapshot=verify_base()
 if a.stage=='recover':recover_originals(c,snapshot)
 elif a.stage=='audit':deploy_inputs(c,approval);native_audit(c,approval);retrieve_native_metadata(approval)
 else:
  run_helper('build_partial_packet.py','--build');contract=read(BUNDLE/'contract.json');start=time.time()
  for split in ('validation','test'):
   write(BUNDLE/'permits'/(split+'.json'),{'schema':'titantpp_components_taxi_raf24_evaluation_permit_v1','authorized':True,'new_training':False,'frozen_selection':True,'stage':split,'contract_sha256':contract['contract_sha256'],'evaluator_sha256':sha(BUNDLE/'operation/evaluate.py'),'host':'5080','job_ids':sorted(j['id'] for j in contract['jobs']),'not_before_unix':start,'deadline_unix':start+43200,'automatic_retry':False,'explicit_user_instruction':approval['user_instruction']})
  run_helper('deploy_evaluation_packet.py');run_helper('run_native_evaluation.py');run_helper('retrieve_evaluation_results.py')
if __name__=='__main__':main()
