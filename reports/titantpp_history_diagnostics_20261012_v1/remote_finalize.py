import pathlib,json,hashlib,time,subprocess,tarfile,os
import sys
root=pathlib.Path(sys.argv[1]).resolve()
expected_seal=sys.argv[2]
expected_contract=sys.argv[3]
assert root == pathlib.Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_diagnostics_20261012_v1_5080'), 'Foreign remote root'
def read(path):return json.loads(path.read_bytes())
def require(value,message):
 if not value:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def write(path,value):
 with path.open('x') as f:json.dump(value,f,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
require(sha(root/'operation_integrity.json')==expected_seal,'Operation seal rawSHA mismatch')
seal=read(root/'operation_integrity.json'); inputs=read(root/'input_integrity.json');c=read(root/'contract.json')
require(seal['contract_sha256']==inputs['contract_sha256']==c['contract_sha256']==expected_contract,'Contract binding mismatch')
complete=read(root/'control/pipeline_complete.json');exit=read(root/'control/pipeline_exit.json')
require(complete['status']=='complete' and complete['contract_sha256']==expected_contract and complete['validation_count']==complete['test_count']==27 and complete['actual_all_worker_exit0'] is True,'Pipeline not complete')
require(exit['exit_code']==0 and exit['actual_child_exit_observed'] is True,'Actual pipeline exit0 missing')
require(not (root/'control/pipeline_failure.json').exists() and not (root/'control/cleanup_failure.json').exists(),'Preserved failure present')
qualification=read(root/'control/native_qualification.json')
routes={job['study']+'__'+job['arm']:(job['study'],job['arm']) for job in c['jobs']}
require(qualification['status']=='PASS' and qualification['contract_sha256']==expected_contract and set(qualification['routes'])==set(routes),'Native qualification incomplete')
probe_pids=[]
for name,(study,arm) in routes.items():
 folder=root/'control/native_probes'/name;probe=read(folder/'receipt.json');probe_exit=read(folder/'worker_exit.json')
 require(sha(folder/'receipt.json')==qualification['routes'][name],'Native probe receipt raw SHA mismatch')
 require(probe['status']=='PASS' and probe['contract_sha256']==expected_contract and probe['study']==study and probe['arm']==arm and probe['device']=='cuda' and probe['source_closure']==c['sources'][study]['closure'],'Native probe identity mismatch')
 require(probe['native_pinned_contract_checked'] is True and probe['optimizer_updates']==0 and probe['checkpoint_reads']==0 and probe['real_data_loaded'] is False and probe['held_out_test_evaluated'] is False and probe['qualification'] and all(value is True for value in probe['qualification'].values()),'Native synthetic qualification incomplete')
 require(probe['runtime']['python']=='3.12.13' and probe['runtime']['gpu']['uuid']==c['runtime']['gpu_uuid'],'Native probe Python/GPU mismatch')
 for key in ['torch','cuda','cudnn','numpy','polars']:require(probe['runtime'][key]==c['runtime'][key],'Native probe numerical Runtime mismatch')
 require(probe_exit['exit_code']==0 and probe_exit['actual_child_exit_observed'] is True and probe_exit['contract_sha256']==expected_contract,'Actual native probe successful exit missing')
 probe_pids.append(probe_exit['pid'])
gate=read(root/'validation_gate.json');require(gate['status']=='PASS' and gate['contract_sha256']==expected_contract and len(gate['validation_receipts'])==27,'Validation gate incomplete')
gate_sha=sha(root/'validation_gate.json'); require(gate_sha==complete['validation_gate_sha256'],'Gate SHA mismatch')
owned=[read(root/'control/pipeline_claim.json')['pid']]+probe_pids
for job in c['jobs']:
 dest=root/'results'/job['id'];val=read(dest/'validation_replay.json');test=read(dest/'test_metrics.json')
 for n in ['validation_worker_exit.json','worker_exit.json']:
  e=read(dest/n);require(e['job_id']==job['id'] and e['contract_sha256']==expected_contract and e['exit_code']==0 and e['actual_child_exit_observed'] is True,'Worker successful exit missing');owned.append(e['pid'])
 require(sha(dest/'validation_replay.json')==gate['validation_receipts'][job['id']],'Val gate rawSHA mismatch')
 require(sha(dest/'validation_worker_exit.json')==gate['validation_worker_exits'][job['id']],'Val exit gate rawSHA mismatch')
 for output,split in [(val,'validation'),(test,'test')]:
  require(output['status']=='complete' and output['job_id']==job['id'] and output['split']==split and output['contract_sha256']==expected_contract and output['checkpoint_sha256']==job['checkpoint']['sha256'] and output['selected_epoch']==job['selected_epoch'] and output['frozen_source_closure']==job['frozen_source_closure'],'Native identity mismatch')
 require(test['validation_gate_sha256']==gate_sha and test['test_started_unix']>=gate['created_unix'],'Test gate order mismatch')
 for output in [val,test]:
  telemetry=root/output['telemetry_path']
  require(telemetry.resolve().is_relative_to(root) and sha(telemetry)==output['telemetry_sha256'],'Telemetry SHA mismatch')
  require(output['native_reference_match']=='PASS' and output['diagnostics_match']=='PASS' and output['new_fits']==0 and output['interventions']==0 and output['optimizer_updates']==0 and output['state_sha256']==output['state_after_sha256'],'Diagnostic observation or invariance failure')
 require((dest/'receipt.json').exists(),'Combinedreceiptmissing')
for pid in owned:
 proc=pathlib.Path('/proc')/str(pid)/'cmdline'
 require(not proc.exists() or str(root).encode() not in proc.read_bytes(),'Own evaluation process still alive')
smi=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
for row in smi.splitlines():
 parts=[x.strip() for x in row.split(',')]
 require(len(parts)<2 or int(parts[1]) not in owned,'Own GPU PID remains')
for rel,expected in {**inputs['files'],**seal['files']}.items():
 target=(root/rel).resolve();require(target.is_relative_to(root) and sha(target)==expected,'Binary/input/operation seal mismatch:'+rel)
claim=root/'retrieval_claim.json';write(claim,{'contract_sha256':expected_contract,'observed_unix':time.time(),'exclusive_retrieval':True,'automatic_retry':False})
files=set(inputs['files'])|set(seal['files'])|{'operation_integrity.json','validation_gate.json','retrieval_claim.json'}
for folder in ['results','control']:
 for path in (root/folder).rglob('*'):
  rel=path.relative_to(root)
  if path.is_file() and not any('cache' in part or part=='__pycache__' for part in rel.parts):files.add(rel.as_posix())
for rel in files:
 path=root/rel
 require(not path.is_symlink() and path.is_file() and path.resolve().is_relative_to(root),'Nonregular or escaping archive source:'+rel)
manifest={rel:sha(root/rel) for rel in sorted(files)}
dest=root/'retrieval';dest.mkdir(exist_ok=False)
write(dest/'manifest.json',{'schema':'immutable_diagnostic_originals_v1','contract_sha256':expected_contract,'files':manifest})
archive=dest/'originals.tar.gz'
with tarfile.open(archive,'x:gz') as t:
 for rel in sorted(files):t.add(root/rel,arcname=rel,recursive=False)
with tarfile.open(archive,'r:gz') as t:
 members=t.getmembers();require(len(members)==len(manifest) and {m.name for m in members}==set(manifest),'Archive member set differs')
 for member in members:
  require(member.isfile() and member.size==(root/member.name).stat().st_size,'Nonregular archive member or size mismatch')
  h=hashlib.sha256()
  with t.extractfile(member) as stream:
   for chunk in iter(lambda:stream.read(1048576),b''):h.update(chunk)
  require(h.hexdigest()==manifest[member.name],'Archive member rawSHA mismatch:'+member.name)
for rel,expected in manifest.items():require(sha(root/rel)==expected,'Source changed during archive creation:'+rel)
receipt={'status':'verified_terminal_archived','contract_sha256':expected_contract,'operation_integrity_sha256':expected_seal,'remote_root':str(root),'manifest_sha256':sha(dest/'manifest.json'),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'file_count':len(manifest),'actual_pipeline_exit0':True,'actual_worker_exit0_count':54,'native_qualification_pass':True,'all27_validation_gate_pass':True,'own_processes_absent':True,'own_gpu_pids_absent':True,'gpu_processes_observed':smi,'completed_unix':time.time(),'automatic_retry':False}
write(dest/'receipt.json',receipt);print(json.dumps(receipt))
