"""New source-root packaging; preserve the failed setup and scientific contract."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,tarfile
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OLD=ROOT/'search_artifacts/phen_taxi_internal_holdout_20261027_v1/payload'
ART=OLD.parent/'execution_v2'
PAYLOAD=ART/'payload'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def prepare():
 assert not PAYLOAD.exists()
 PAYLOAD.mkdir(parents=True)
 for p in OLD.rglob('*'):
  if not p.is_file() or p.name in ['EXECUTION.json','native_smoke.py']:continue
  dest=PAYLOAD/p.relative_to(OLD);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
  assert sha(p)==sha(dest)
 marker=PAYLOAD/'source/sample_data/.SOURCE_ROOT_SENTINEL';marker.parent.mkdir()
 marker.write_text('Empty source-root sentinel. No dataset or Test data is stored here.\n')
 assert sha(HERE/'supervise.py')==sha(PAYLOAD/'supervise.py')
 shutil.copyfile(HERE/'native_smoke.py',PAYLOAD/'native_smoke.py')
 put(PAYLOAD/'SOURCE_INVENTORY.json',{str(p.relative_to(PAYLOAD/'source')):sha(p) for p in sorted((PAYLOAD/'source').rglob('*')) if p.is_file()})
 print(json.dumps({'staged':str(PAYLOAD),'source_files':len(json.loads((PAYLOAD/'SOURCE_INVENTORY.json').read_text())),'model_and_science_changed':False}))
def seal():
 qa=json.loads((HERE/'LOCAL_QA.json').read_text());assert qa['status']=='PASS'
 for n,h in qa['reviewed_sources'].items():assert sha(n)==h,n
 for p in OLD.rglob('*'):
  if p.is_file() and p.name not in ['EXECUTION.json','native_smoke.py']:assert sha(p)==sha(PAYLOAD/p.relative_to(OLD))
 c=json.loads((OLD/'EXECUTION.json').read_text());prior=c['remote_root'];remote=prior.replace('_v1_5090','_v1_execution_v2_5090')
 assert remote!=prior
 def replace(x):
  if isinstance(x,str):return x.replace(prior,remote)
  if isinstance(x,list):return [replace(v) for v in x]
  if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
  return x
 c=replace(c)
 c['tmux_session']='taxi_internal_holdout_seed42_20261027_exec2'
 c['packaging_revision']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 c['prior_setup_failure']={'remote_root':prior,'boot_id':c['boot_id'],'evidence':{}}
 diagnosis=json.loads((HERE.parent/'remote_prepare_failure_01/stdout.log').read_text())
 assert not any(diagnosis[n] for n in ['fit_claim_exists','fit_started_exists','prepared_exists'])
 for n in ['prelaunch_02.ACTUAL_EXIT.json','supervisor.actualexit.json','dispatcher.actualexit']:
  rec=diagnosis['files'][n];c['prior_setup_failure']['evidence'][n]={k:rec[k] for k in ['bytes','sha256']}
 assert c['source_commit']=='3a98264a5fbd261f364d5031fe42041f6203a462'
 assert c['environment']['SOURCE_REVISION']==c['source_commit']
 assert sha(PAYLOAD/'SCIENCE_SPEC.json')==c['science_sha256']
 c['input_files']={str(p.relative_to(PAYLOAD)):sha(p) for p in sorted(PAYLOAD.rglob('*')) if p.is_file()}
 put(PAYLOAD/'EXECUTION.json',c)
 members={str(p.relative_to(PAYLOAD)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(PAYLOAD.rglob('*')) if p.is_file()}
 with tarfile.open(ART/'payload.tar.gz','x:gz') as tar:
  for n in members:tar.add(PAYLOAD/n,arcname=n,recursive=False)
 package={'archive_sha256':sha(ART/'payload.tar.gz'),'archive_bytes':(ART/'payload.tar.gz').stat().st_size,'members':members}
 put(HERE/'PAYLOAD.json',package)
 put(HERE/'PREFLIGHT_QA.json',{'status':'PASS','local_qa_sha256':sha(HERE/'LOCAL_QA.json'),'archive_sha256':package['archive_sha256'],'execution_sha256':sha(PAYLOAD/'EXECUTION.json'),'reviewed_sources':qa['reviewed_sources'],'Test_access':False})
 print(json.dumps({'members':len(members),'archive_bytes':package['archive_bytes'],'remote_root':remote,'model_commit':c['source_commit'],'packaging_commit':c['packaging_revision']}))
p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','seal']);a=p.parse_args()
prepare() if a.mode=='prepare' else seal()
