"""Advance exact seed62 Test4 only after a sealed terminal monitor round."""
from __future__ import annotations
import argparse,base64,hashlib,io,json,os,shlex,subprocess,sys,tarfile,time,traceback
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from paper.scripts.titantpp_inter62_transport_evidence import recorded_run
BUNDLE=PROJECT/'search_artifacts/titantpp_components_inter62_test_20261014_v1'
REPORT=PROJECT/'reports/titantpp_components_inter62_test_20261014_v1'
HANDOFF=PROJECT/'search_artifacts/titantpp_components_seed62_handoff_20261013_v1'
MONITOR=PROJECT/'reports/titantpp_components_seed62_handoff_20261013_v1/hourly_monitor_v2'
PARENT='81972d7b774c3b924d4bf6517252a0d0bed8291da9165100ab5b8932128ef1c1'
SOURCE='15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53'


def require(ok,message):
 if not ok:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def verify_bootstrap(digest):
 p=BUNDLE/'bootstrap_integrity_v2.json';require(sha(p)==digest,'Bootstrap seal SHA mismatch; remote0')
 seal=read(p);require(seal['closure']==canonical(seal['files']),'Bootstrap closure changed; remote0')
 require(sha(__file__) in seal['files'].values(),'Advance tool not sealed; remote0')
 for name,d in seal['files'].items():
  p=PROJECT/name;require(p.resolve().is_relative_to(PROJECT) and not p.is_symlink() and sha(p)==d,'Bootstrap file changed; remote0: '+name)
 return seal


def terminal_gate(round_path,round_sha):
 path=Path(round_path).resolve();require(path.is_relative_to(MONITOR/'rounds') and sha(path)==round_sha,'Foreign monitor round or SHA; remote0')
 r=read(path);require(r['monitor_integrity_raw_sha256']=='4f2ab78970d862cf146365de6a75999950964c2d809c27183d1a970f37fedb40','Foreign observation seal')
 g=r['groups']['handoff_5080']
 if r['all_training_terminal_verified'] is not True:return r,g,None
 c=read(BUNDLE/'provenance/training_contract.json');require(canonical(c)==PARENT,'Handoff contract changed')
 sp=Path(g['snapshot_path']).resolve();require(sp.is_relative_to(HANDOFF/'control/status/5080') and sha(sp)==g['snapshot_raw_sha256'],'Terminal snapshot SHA/root changed')
 snapshot=read(sp)
 from paper.scripts import titantpp_components_parallel_monitor_v2 as monitor
 from paper.scripts import titantpp_components_terminal_verifier as verifier
 # Repeat only local proof checks, never remote acquisition/observer execution.
 report=monitor.verify_destination_terminal(snapshot,c,verifier.training_status(snapshot,c,'5080'))
 bridge=verifier.qualification_bridge(HANDOFF,c,'5080',snapshot)
 require(bridge==g['qualification_bridge'],'Native qualification/permit bridge changed; remote0')
 require(g['status']==report['status']=='training_complete_remote_verified' and report['completed']==report['total']==4 and report['owner_process_absence_verified'] is True,'Exact4 terminal proof incomplete')
 require(snapshot['contract_sha256']==PARENT and snapshot['source_files_sha256']==SOURCE and not snapshot['gpu_pids'],'Terminal identity/GPU occupancy differs')
 return r,g,snapshot


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
 for j in c['jobs']:
  name='run/'+j['id']+'/terminal_manifest.json';item=snapshot['files'][name];raw=base64.b64decode(item['base64']);require(hashlib.sha256(raw).hexdigest()==item['raw_sha256'],'Terminal manifest snapshot changed')
  terminal=json.loads(raw);require(terminal['job']==j and terminal['contract_sha256']==PARENT and terminal['scientific_success'] is True and terminal['held_out_test_evaluated'] is False,'Foreign scientific terminal')
  expected[name]=item['raw_sha256'];exit_name='worker_exits/'+j['id']+'.json';expected[exit_name]=snapshot['files'][exit_name]['raw_sha256']
  for n,d in terminal['files'].items():expected['run/'+j['id']+'/'+n]=d
 require(len(expected)==60,'Expected60 originals only')
 archive=BUNDLE/'originals_archive.tar.gz'
 payload={'remote_original_root':c['hosts']['5080']['root'],'training_contract_raw_sha256':sha(BUNDLE/'provenance/training_contract.json'),'expected_files':expected}
 transport('retrieval',(REPORT/'remote_retrieval.py').read_text(),payload,archive=archive)
 with tarfile.open(archive) as t:
  members=t.getmembers();require(len(members)==60 and {m.name for m in members}=={'originals/'+n for n in expected},'Archive membership differs')
  for m in members:
   require(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts,'Unsafe archive member')
   raw=t.extractfile(m).read();require(hashlib.sha256(raw).hexdigest()==expected[m.name.removeprefix('originals/')],'Original archive SHA differs')
   path=BUNDLE/m.name;path.parent.mkdir(parents=True,exist_ok=True)
   with path.open('xb') as f:f.write(raw)
 files={'originals/'+n:d for n,d in expected.items()}
 for n,d in files.items():require(sha(BUNDLE/n)==d,'Local original SHA differs')
 write(BUNDLE/'originals_manifest.json',{'files':files,'all_file_SHA_verified':True,'source131_SHA_verified':True,'checkpoint_selected_last_pairs':4,'remote_originals_changed':False,'archive_raw_sha256':sha(archive)})


def deploy_inputs(c,approval):
 names=['partial_approval.json','originals_manifest.json','operation/audit_inputs.py']
 names+=['provenance/'+p.name for p in sorted((BUNDLE/'provenance').glob('*.json'))]
 names+=list(read(BUNDLE/'originals_manifest.json')['files'])
 names+=['source/'+n for n in c['source']['files']]
 names+=['source/sample_data/.frozen_root_sentinel','data/intermittent_frozen_5000_with_split.parquet','data/intermittent_frozen_5000_split_manifest.json']
 require(len(names)==len(set(names)),'Duplicate deployment inputs')
 files={n:sha(BUNDLE/n) for n in names};manifest={'files':files,'approval_sha256':sha(BUNDLE/'partial_approval.json')};write(BUNDLE/'deployment_integrity.json',manifest)
 archive=BUNDLE/'input_deployment.tar.gz'
 with tarfile.open(archive,'w:gz') as t:
  for n in names+['deployment_integrity.json']:t.add(BUNDLE/n,arcname=n,recursive=False)
 remote=approval['authorization_root'];payload={'root':remote,'archive_sha256':sha(archive),'approval_sha256':manifest['approval_sha256'],'count':len(files)}
 code="""import json,sys\nfrom pathlib import Path\np=json.load(sys.stdin);r=Path(p['root']);assert str(r)=='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter62_test_20261014_v1_5080';r.mkdir(exist_ok=False);(r/'incoming').mkdir()\nwith (r/'deployment_intent.json').open('x') as f:json.dump(p,f)\n"""
 transport('reserve',code,payload)
 x=recorded_run(BUNDLE,'upload',['scp','-o','BatchMode=yes','-o','ConnectTimeout=15',str(archive),'5080:'+remote+'/incoming/input_deployment.tar.gz'],timeout=120)
 require(x.returncode==0,'Input upload failed')
 code="""import hashlib,json,sys,tarfile,time\nfrom pathlib import Path\np=json.load(sys.stdin);r=Path(p['root']);assert str(r)=='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter62_test_20261014_v1_5080';a=r/'incoming/input_deployment.tar.gz';assert hashlib.sha256(a.read_bytes()).hexdigest()==p['archive_sha256']\nwith tarfile.open(a) as t:\n m=json.load(t.extractfile('deployment_integrity.json'));files=m['files'];assert m['approval_sha256']==p['approval_sha256'] and len(files)==p['count'];members=t.getmembers();assert len(members)==len(files)+1 and len(set(x.name for x in members))==len(members)\n for x in members:\n  n=Path(x.name);target=r/n;assert x.isfile() and not n.is_absolute() and '..' not in n.parts and target.resolve().is_relative_to(r.resolve()) and not target.exists();raw=t.extractfile(x).read();assert x.name=='deployment_integrity.json' or hashlib.sha256(raw).hexdigest()==files[x.name]\n for x in members:\n  target=r/x.name;target.parent.mkdir(parents=True,exist_ok=True)\n  with target.open('xb') as f:f.write(t.extractfile(x).read())\nwith (r/'deployment_receipt.json').open('x') as f:json.dump({'status':'deployed','files':files,'archive_sha256':p['archive_sha256'],'original_root_writes':0,'completed_unix':time.time()},f)\nprint(json.dumps({'status':'deployed','files':len(files)}))\n"""
 transport('input_deploy',code,payload)


def native_audit(c,approval):
 root=approval['authorization_root'];env={**c['hosts']['5080']['environment'],'PYTHONPATH':root+'/source','MPLCONFIGDIR':root+'/audit/initial/runtime_cache/matplotlib','XDG_CACHE_HOME':root+'/audit/initial/runtime_cache'}
 argv=['env',*[k+'='+v for k,v in sorted(env.items())],c['hosts']['5080']['python'],root+'/operation/audit_inputs.py','--root',root,'--approval-sha',sha(BUNDLE/'partial_approval.json'),'--originals-manifest-sha',sha(BUNDLE/'originals_manifest.json'),'--audit-attempt','initial']
 write(BUNDLE/'control/native_audit_intent.json',{'argv':argv,'started_unix':time.time(),'model_predictions':0,'automatic_retry':False})
 x=recorded_run(BUNDLE,'native_audit',['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5080',shlex.join(argv)],timeout=900)
 require(x.returncode==0,'Native input audit failed; no automatic retry: '+x.stderr.decode(errors='replace')[-2000:])
 value=json.loads(x.stdout);require(value['status']=='complete' and value['model_predictions']==0,'Input audit incomplete')
 with (BUNDLE/'native_input_receipt.json').open('xb') as f:f.write(x.stdout)


def run_helper(name,*args):
 x=subprocess.run([sys.executable,str(REPORT/name),*args],cwd=PROJECT,timeout=43200)
 require(x.returncode==0,'Helper failed; automatic retry forbidden: '+name)


def execute(round_path,round_sha,integrity_sha):
 verify_bootstrap(integrity_sha)
 if (BUNDLE/'final_receipt.json').exists():
  done=read(BUNDLE/'final_receipt.json');require(done['status']=='complete' and done['seed_count']==3 and done['test_job_count']==12,'Existing final receipt is not complete')
  require(sha(REPORT/'THREE_SEED_ANALYSIS.json')==done['analysis_sha256'] and sha(REPORT/'THREE_SEED_ANALYSIS.md')==done['report_sha256'],'Completed analysis changed')
  return done
 r,g,snapshot=terminal_gate(round_path,round_sha)
 if snapshot is None:
  status={'status':'waiting_for_FULL62_training_terminal','observed_KST':g['observed_KST'],'all_training_terminal_verified':False,'Test_executed':False,'originals_retrieved':False,'remote_calls':0}
  write(BUNDLE/'control'/('waiting_'+Path(round_path).stem+'.json'),status);return status
 require(not (BUNDLE/'advance.claim').exists() and not (BUNDLE/'advance_failure.json').exists(),'Existing advance claim/failure; no automatic retry')
 write(BUNDLE/'advance.claim',{'pid':os.getpid(),'ppid':os.getppid(),'started_unix':time.time(),'round_path':str(round_path),'round_raw_sha256':round_sha,'automatic_retry':False})
 try:
  for path,dest in [(Path(round_path),BUNDLE/'provenance/training_monitor.json'),(Path(g['snapshot_path']),BUNDLE/'provenance/training_terminal_snapshot.json')]:
   with dest.open('xb') as f:f.write(path.read_bytes())
  c=read(BUNDLE/'provenance/training_contract.json');approval=read(BUNDLE/'partial_approval.json')
  recover_originals(c,snapshot);deploy_inputs(c,approval);native_audit(c,approval)
  run_helper('retrieve_native_metadata.py');run_helper('build_partial_packet.py','--build')
  contract=read(BUNDLE/'contract.json');start=time.time()
  for split in ('validation','test'):
   write(BUNDLE/'permits'/(split+'.json'),{'schema':'titantpp_components_inter62_evaluation_permit_v1','authorized':True,'new_training':False,'frozen_selection':True,'stage':split,'contract_sha256':contract['contract_sha256'],'evaluator_sha256':sha(BUNDLE/'operation/evaluate.py'),'host':'5080','job_ids':sorted(j['id'] for j in contract['jobs']),'not_before_unix':start,'deadline_unix':start+43200,'automatic_retry':False,'explicit_user_instruction':approval['user_instruction']})
  run_helper('deploy_evaluation_packet.py');run_helper('run_native_evaluation.py');run_helper('retrieve_evaluation_results.py');run_helper('merge_three_seed_results.py')
  return read(BUNDLE/'final_receipt.json')
 except BaseException as error:
  write(BUNDLE/'advance_failure.json',{'status':'failed','error':type(error).__name__,'message':str(error),'traceback':traceback.format_exc(),'automatic_retry':False,'completed_unix':time.time()});raise


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--round',type=Path,required=True);p.add_argument('--round-sha',required=True);p.add_argument('--integrity-sha',required=True);a=p.parse_args();print(json.dumps(execute(a.round,a.round_sha,a.integrity_sha),ensure_ascii=False))
