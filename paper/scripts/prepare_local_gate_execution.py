#!/usr/bin/env python3
"""Freeze two new gate arms; reuse audited same-host B/local reference evidence."""
from __future__ import annotations
import argparse
from copy import deepcopy
import io,json,math
from pathlib import Path
import subprocess,sys,tarfile,time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from paper.scripts.observed_slot_parallel_common import require,sha_file,sha_json,source_manifest,write_json
SCHEMA='local_gate_observed_time_execution_v1'
ARMS=('titantpp','titantpp_local_time_gate','titantpp_local_quantity_gate')
UNGATED='titantpp_local_detail'
ASSIGNMENTS={'5080':['yellow_trip_hourly','intermittent_frozen_5000'],'5090':['insta_market_basket']}
RUN_NAME='local_gate_observed_time_seed42_20260922_v1'
DESIGN='paper/contracts/hard_lmm_local_gate_design_v1.json'
PARENT='paper/contracts/multilag_detail_observed_time_execution_data_v3.json'
READINESS='search_artifacts/local_gate_preparation_20260922_v1/host_readiness.json'
ENTRYPOINTS=('paper/scripts/run_count_aware_tpp_backbone_control.py','paper/scripts/verify_local_gate_cost.py','paper/scripts/prepare_local_gate_execution.py','paper/scripts/run_local_gate_execution.py','paper/scripts/local_gate_reference_bridge.py')
EXTRA=(DESIGN,PARENT,READINESS)
LIMITS={'qualification_seconds_per_host':900,'total_wall_seconds':172800,'per_arm_wall_seconds':86400,'max_concurrent_gpu_jobs_per_host':1,'max_gpu_hours_aggregate':96,'per_host_output_bytes':8*1024**3,'min_free_bytes':10*1024**3}
COST_GATES={'parameter_ratio_max':1.10,'median_cuda_step_ratio_max':1.5,'peak_cuda_allocated_ratio_max':1.25,'device_memory_fraction_max':.8}
POLICY={'retry':False,'resume':False,'additional_seeds':False,'held_out':False,'runtime_mutation':False,'commit_push':False,'budget_extension':False,'other_gpu_processes':'observe_without_stopping','startup_min_free_gpu_bytes':2*1024**3,'qualification_per_host_before_its_training':True,'automatic_start_after_successful_dependency':True}
ARCHITECTURE={'hidden_dim':64,'rank':4,'local_lags':[1]*8,'availability_lags':[1,2,4,8,16,32,64,128],'residual_divisor':8,'gate':'2sigmoid','initial_multiplier':1.,'additional_gate_parameters':[3,5]}
def read(path):return json.loads(Path(path).read_text())
def parent():
 p=read(ROOT/PARENT);require(sha_json(p)=='b69ddb960ccacb65d987428e1b40cdbfb8f3a7493dd18138f1da2836c0588800','Wrong data/reference lineage');return p

def host_specs():
 p=parent();hosts=deepcopy(p['hosts']);process=deepcopy(p['process_environment']);ready=read(ROOT/READINESS)
 for h,s in hosts.items():
  observed=ready['hosts'][h];require(observed['source_verified'] and observed['disk_available_bytes']>=LIMITS['min_free_bytes'],'Missing readiness')
  s['root']=str(Path(s['root']).parent/RUN_NAME);s['source_root']=s['root']+'/source';s['output_dir']=s['root']+'/run';s['tmux']=RUN_NAME+'_'+h;s['assigned_datasets']=ASSIGNMENTS[h]
  process[h]['XDG_CACHE_HOME']=s['root']+'/cache'
 return hosts,process,p['library_sha256']

def dependency():
 return {'5090':{'root':'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/multilag_detail_instacart_shared_resume_20260921_v2','contract_sha256':'c55416f59386645b64cdaa3515ba8f7c9f3b91811225a274be674370ad840d86','required_status':'complete','arm_epochs':120,'own_worker_pid':633975,'tmux':'multilag_detail_instacart_shared_resume_20260921_v2_5090_train'}}

def build_contract(cpu_path,implementation_path):
 cost=read(cpu_path);receipt=read(implementation_path)
 require(cost['device']=='cpu' and cost['status']=='cpu_observation_complete' and cost['synthetic_optimizer_updates']==48 and all(cost['checks'].values()),'Passing synthetic CPU cost required')
 require(cost['verifier_source_sha256']==sha_file(ROOT/'paper/scripts/verify_local_gate_cost.py'),'Verifier changed')
 require(receipt['verification']['failed']==0 and receipt['verification']['passed']>0 and receipt['native_execution_verification']['status']=='passed_synthetic_cpu','Passing implementation/integration required')
 started=time.time()
 p=parent();hosts,env,libs=host_specs();ready=read(ROOT/READINESS);files=source_manifest(ROOT,entrypoints=ENTRYPOINTS,extra_files=EXTRA)
 return {'schema':SCHEMA,'date':'2026-09-22','status':'frozen_pending_native_qualification','model_role':'observed_time_local_gate_v1','arms':list(ARMS),'new_training_arms':list(ARMS[1:]),'reused_arms':['titantpp',UNGATED],'seed':42,'epochs':120,'datasets':deepcopy(p['datasets']),'architecture':ARCHITECTURE,'hosts':hosts,'process_environment':env,'library_sha256':libs,'limits':LIMITS,'policy':POLICY,'cost_gates':COST_GATES,'acceptance':read(ROOT/DESIGN)['acceptance'],'total_arms':6,'total_optimizer_steps':4544160,'endpoint_replays':12,'reused_reference_arms':6,'reused_reference_replays':12,'references':{h:ready['hosts'][h]['reference_files'] for h in ASSIGNMENTS},'dependency':dependency(),'parent_contract_sha256':sha_json(p),'design_sha256':sha_file(ROOT/DESIGN),'source':{'base_git_revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'working_tree':'dirty; isolated file closure authoritative','files':files,'files_sha256':sha_json(files)},'evidence':{'cpu_cost_sha256':sha_file(cpu_path),'implementation_sha256':sha_file(implementation_path),'host_readiness_sha256':sha_file(ROOT/READINESS)},'launch':{'fixed_started_at_unix':started,'fixed_deadline_unix':started+LIMITS['total_wall_seconds'],'server_assignment':ASSIGNMENTS,'root_must_be_new':True,'budget_origin':'frozen launch origin for this new six-fit gate comparison; shared48h ceiling including CPU preflight, qualification and5090 waiting; previous experiment clocks unchanged','deadline_action':'stop only owned new worker; preserve outputs; no auto retry or extension','estimated_completion_hours':None},'comparison':{'primary':'quantity_gate versus time_gate, exact2-parameter difference','secondary':'same-host audited B and ungated local references; actual batch exposure must match','reference_bridge':'same frozen data/settings and current source vs original source CPU initialization/forward/loss/preclip gradients/next optimizer update/RNG; historical wall-time not isolated','selector':'strict earliest validation raw-RMSE minimum; all selected metrics same checkpoint','scope':'single_seed_validation_exploration'},'approval_required':True,'approved':False}

def validate_contract(c,*,verify_source=True):
 p=parent();hosts,env,libs=host_specs();ready=read(ROOT/READINESS)
 expected={'schema':SCHEMA,'model_role':'observed_time_local_gate_v1','arms':list(ARMS),'new_training_arms':list(ARMS[1:]),'reused_arms':['titantpp',UNGATED],'seed':42,'epochs':120,'datasets':p['datasets'],'architecture':ARCHITECTURE,'hosts':hosts,'process_environment':env,'library_sha256':libs,'limits':LIMITS,'policy':POLICY,'cost_gates':COST_GATES,'acceptance':read(ROOT/DESIGN)['acceptance'],'total_arms':6,'total_optimizer_steps':4544160,'endpoint_replays':12,'dependency':dependency(),'references':{h:ready['hosts'][h]['reference_files'] for h in ASSIGNMENTS},'parent_contract_sha256':sha_json(p),'design_sha256':sha_file(ROOT/DESIGN),'approval_required':True,'approved':False}
 for k,v in expected.items():require(c.get(k)==v,'Frozen gate contract changed: '+k)
 require(type(c['launch'].get('fixed_started_at_unix')) in (int,float) and math.isfinite(c['launch']['fixed_started_at_unix']) and c['launch'].get('fixed_deadline_unix')==c['launch']['fixed_started_at_unix']+LIMITS['total_wall_seconds'],'Frozen deadline missing or extended')
 require(sha_json(c['source']['files'])==c['source']['files_sha256'],'Source digest inconsistent')
 if verify_source:
  require(source_manifest(ROOT,entrypoints=ENTRYPOINTS,extra_files=EXTRA)==c['source']['files'],'Frozen source changed')
  require(sha_file(ROOT/READINESS)==c['evidence']['host_readiness_sha256'],'Readiness changed')

def freeze(contract_path,output,cpu_path,implementation_path):
 require(not contract_path.exists() and not output.exists(),'New contract/output required')
 c=build_contract(cpu_path.resolve(),implementation_path.resolve());validate_contract(c)
 output.mkdir(parents=True);write_json(contract_path,c,exclusive=True)
 with tarfile.open(output/'source_bundle.tar.gz','w:gz') as tar:
  entry=tarfile.TarInfo('source/sample_data');entry.type=tarfile.DIRTYPE;entry.mode=0o755;tar.addfile(entry)
  for name in c['source']['files']:
   raw=(ROOT/name).read_bytes();entry=tarfile.TarInfo('source/'+name);entry.size=len(raw);entry.mode=0o644;tar.addfile(entry,io.BytesIO(raw))
  raw=(json.dumps(c,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode();entry=tarfile.TarInfo('frozen_execution/execution_contract.json');entry.size=len(raw);tar.addfile(entry,io.BytesIO(raw))
 receipt={'status':'contract_frozen','contract_sha256':sha_json(c),'bundle_sha256':sha_file(output/'source_bundle.tar.gz'),'gpu_executed':False,'source_files_sha256':c['source']['files_sha256']};write_json(output/'preparation_receipt.json',receipt,exclusive=True);return receipt

def main():
 a=argparse.ArgumentParser();a.add_argument('--contract',type=Path,required=True);a.add_argument('--output',type=Path,required=True);a.add_argument('--cpu-cost',type=Path,required=True);a.add_argument('--implementation',type=Path,required=True);v=a.parse_args();print(json.dumps(freeze(v.contract,v.output,v.cpu_cost,v.implementation)))
if __name__=='__main__':main()
