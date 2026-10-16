import importlib.util,json,sys,time,unittest,tempfile,os,copy,hashlib,subprocess
from pathlib import Path
from unittest.mock import patch
PROJECT=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,PROJECT/'paper/scripts'/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
s=load('resume_supervisor','titantpp_cnn_gru_resume_supervisor.py');o=load('resume_observer','observe_titantpp_cnn_gru_resume_hourly.py')
def write(path,v):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(v))
def fixture(root):
 job={'id':'intermittent_frozen_5000__42__titantpp_gru54','host':'5080','dataset':'intermittent_frozen_5000','seed':42,'arm':'titantpp_gru54'}
 c={'schema':s.SCHEMA,'evaluation_scope':'validation_only','held_out_test_evaluated':False,'source':{'files_sha256':'sealed'},'jobs':[job],'hosts':{'5080':{'root':str(root),'source_root':str(root/'source'),'python':sys.executable,'gpu_uuid':'GPU-test','environment':{}},'5090':{}},'artifacts':{job['id']:{'source_epoch':2}},'limits':{'workers_per_host':1,'automatic_retry':False,'manual_resume_once':True,'per_condition_seconds':129600,'total_wall_seconds':604800},'parent':{'contract_sha256':'parent-sha'}}
 (root/'source').mkdir();(root/'operation').mkdir();(root/'operation/titantpp_cnn_gru_resume_runtime.py').write_text('# fake adapter\n');c['operation_adapter_sha256']=s.sha(root/'operation/titantpp_cnn_gru_resume_runtime.py')
 write(root/'execution_contract.json',c)
 a={'approved':True,'manual_resume_once':True,'contract_sha256':s.canonical(c),'parent_contract_sha256':'parent-sha','hosts':['5080','5090'],'job_ids':[job['id']],'user_instruction':'approved'};write(root/'approval.json',a)
 start={'contract_sha256':s.canonical(c),'approval_sha256':s.canonical(a),'started_at_unix':time.time()-1,'deadline_unix':time.time()+100};write(root/'start_permit.json',start)
 q={'status':'passed','host':'5080','job':job,'contract_sha256':s.canonical(c),'full_validation_replay':True,'held_out_test_evaluated':False};write(root/'qualification'/job['id']/'receipt.json',q)
 p={'host':'5080','contract_sha256':s.canonical(c),'start_permit_sha256':s.canonical(start),'manual_resume_once':True,'automatic_retry':False,'job_ids':[job['id']],'qualification_files':{job['id']:s.sha(root/'qualification'/job['id']/'receipt.json')}};write(root/'training_permit_5080.json',p)
 return c,job
def snapshot(c,job):
 hist=[{'epoch':i,'val_qty_rmse':4-i,'val_qty_mae':1,'val_time_nll':.2} for i in (1,2,3)]
 r={'job':job,'folder_exists':True,'history':{'history':hist},'checkpoint_receipt':{'epoch':3,'contract_sha256':s.canonical(c),'files':{name:'recordsha' for name in ('history.json','exposure.json','epoch_timing.json')}},'last_checkpoint_bytes':12,'timing':{'epochs':[{'epoch':1,'elapsed_seconds':1000},{'epoch':2,'elapsed_seconds':2000},{'epoch':3,'elapsed_seconds':10}]},'small_record_checks':{name:True for name in ['status.json','history.json','exposure.json','epoch_timing.json','endpoint_replays.json']}}
 argv=s.fit_argv(c,Path(c['hosts']['5080']['root']),'5080',job)
 return {'host':'5080','root':c['hosts']['5080']['root'],'observed_unix':time.time(),'files':{'execution_contract.json':c,'supervisor_status.json':{'active_job':job}},'file_sha256':{'run/'+job['id']+'/runs/'+job['arm']+'/'+s.VARIANT+'/seed_42/'+name:'recordsha' for name in ('history.json','exposure.json','epoch_timing.json')},'runs':[r],'gpu':{'returncode':0,'rows':[['GPU-test','RTX5080']]},'compute':{'returncode':0,'rows':[['GPU-test','123','12']]},'processes':[{'pid':123,'argv':argv,'cwd':c['hosts']['5080']['source_root']}]}
class Tests(unittest.TestCase):
 def setUp(self):self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.c,self.job=fixture(self.root)
 def tearDown(self):self.tmp.cleanup()
 def test_approved_permits_and_fit_argv(self):
  c,root,jobs,start=s.permits(self.root/'execution_contract.json','5080');self.assertEqual(jobs,[self.job]);self.assertNotIn('--deadline',s.fit_argv(c,root,'5080',self.job))
 def test_changed_approval_blocked(self):
  p=self.root/'approval.json';a=s.read(p);a['approved']=False;write(p,a)
  with self.assertRaises(ValueError):s.permits(self.root/'execution_contract.json','5080')
 def test_changed_qualification_blocked(self):
  p=self.root/'qualification'/self.job['id']/'receipt.json';q=s.read(p);q['status']='failed';write(p,q)
  with self.assertRaises(ValueError):s.permits(self.root/'execution_contract.json','5080')
 def test_expired_deadline_blocked(self):
  with self.assertRaises(ValueError):s.permits(self.root/'execution_contract.json','5080',now=time.time()+1000)
 def test_duplicate_claim_never_spawns(self):
  write(self.root/'claims'/(self.job['id']+'.json'),{})
  old=os.getcwd();os.chdir(self.root/'source')
  try:
   with patch.object(s.subprocess,'Popen') as popen:
    with self.assertRaises(ValueError):s.supervise(self.root/'execution_contract.json','5080')
    popen.assert_not_called();self.assertEqual(s.read(self.root/'supervisor_failure.json')['automatic_retry'],False)
  finally:os.chdir(old)
 def test_group_timeout_stops_only_child_group(self):
  child=unittest.mock.Mock();child.pid=543;child.poll.return_value=None;child.wait.side_effect=[subprocess.TimeoutExpired('fit',15),0]
  with patch.object(s.os,'killpg') as kill:s.stop_owned(child)
  self.assertEqual([x.args[0] for x in kill.call_args_list],[543,543]);self.assertEqual(kill.call_args_list[0].args[1],s.signal.SIGTERM);self.assertEqual(kill.call_args_list[1].args[1],s.signal.SIGKILL)
 def test_terminal_rejects_empty_replay(self):
  folder=self.root/'run'/self.job['id'];e={'status':'complete','evaluation_scope':'validation_only','held_out_test_evaluated':False,'replay_checks':{'selected':{},'last':{}}};path=folder/'runs'/self.job['arm']/s.VARIANT/'seed_42/endpoint_replays.json';write(path,e)
  write(folder/'terminal_manifest.json',{'status':'complete','scientific_success':True,'job':self.job,'contract_sha256':s.canonical(self.c),'held_out_test_evaluated':False,'files':{str(path.relative_to(folder)):s.sha(path)}})
  with self.assertRaises(ValueError):s.terminal(self.root,self.c,self.job)
 def test_observer_own_target_times_only(self):
  a=o.analyse(snapshot(self.c,self.job),self.c,'5080');self.assertEqual(a['status_counts']['in_progress'],1);self.assertEqual(a['rows'][0]['recent_target_epoch_seconds_median'],10);self.assertEqual(a['rows'][0]['eta']['target_epoch'],43)
 def test_worker_absence_not_restart_or_new_failure(self):
  snap=snapshot(self.c,self.job);snap['processes']=[];snap['compute']['rows']=[];a=o.analyse(snap,self.c,'5080');self.assertEqual(a['status_counts']['unknown'],1);self.assertEqual(a['status_counts']['failed'],0)
 def test_query_failure_is_not_fit_failure(self):
  def runner(*args,**kwargs):return subprocess.CompletedProcess([],255,'','secret stderr')
  c=self.c;write(self.root/'current.json',{'contract_sha256':s.canonical(c)})
  r=o.run_once('5080',self.root,runner);self.assertEqual(r['ssh_calls'],1);self.assertEqual(r['status'],'observation_failed_not_training_failure');self.assertIsNone(r['observed_unix'])
 def test_remote_reader_stdlib_and_single_ssh(self):
  script=o.remote_script(self.c,'5080');self.assertNotIn('import torch',script);self.assertNotIn('torch.load',script)
  def runner(argv,**kwargs):self.assertEqual(argv[0],'ssh');self.assertEqual(argv[-1],'python3 -');return subprocess.CompletedProcess(argv,0,json.dumps(snapshot(self.c,self.job)),'')
  self.assertEqual(o.snapshot_once(self.c,'5080',runner)['host'],'5080')
 def test_remote_reader_does_not_open_test_rows(self):
  folder=self.root/'run'/self.job['id'];run=folder/'runs'/self.job['arm']/s.VARIANT/'seed_42'
  write(run/'history.json',{'history':[{'epoch':1,'test_qty_rmse':1}]})
  p=subprocess.run([sys.executable,'-'],input=o.remote_script(self.c,'5080'),text=True,capture_output=True)
  self.assertEqual(p.returncode,0,p.stderr);snap=json.loads(p.stdout);self.assertIsNone(snap['runs'][0]['history']);self.assertTrue(snap['record_errors'])
 def test_pending_remains_pending(self):
  snap=snapshot(self.c,self.job);snap['runs'][0]={'job':self.job,'folder_exists':False};snap['processes']=[];snap['compute']['rows']=[];snap['files'].pop('supervisor_status.json');a=o.analyse(snap,self.c,'5080');self.assertEqual(a['status_counts']['pending'],1)
 def test_foreign_remote_contract_refused(self):
  snap=snapshot(self.c,self.job);snap['files']['execution_contract.json']={}
  with self.assertRaises(ValueError):o.analyse(snap,self.c,'5080')
 def test_changed_checkpoint_json_is_unknown(self):
  snap=snapshot(self.c,self.job);name=next(iter(snap['file_sha256']));snap['file_sha256'][name]='changed';a=o.analyse(snap,self.c,'5080');self.assertEqual(a['status_counts']['unknown'],1)
 def test_terminal_cache_zero_ssh_and_original_time(self):
  snap=snapshot(self.c,self.job);r=snap['runs'][0];snap['processes']=[];snap['compute']['rows']=[]
  r['manifest']={'status':'complete','scientific_success':True,'job':self.job,'contract_sha256':s.canonical(self.c),'held_out_test_evaluated':False}
  r['endpoints']={'status':'complete','job':self.job,'evaluation_scope':'validation_only','held_out_test_evaluated':False,'best_epoch':3,'completed_epochs':3,'selected':{'qty_rmse':1},'last':{'qty_rmse':1},'replay_checks':{label:{metric:True for metric in ['qty_rmse','qty_mae','time_nll']} for label in ['selected','last']}}
  snap['file_sha256']['run/'+self.job['id']+'/terminal_manifest.json']='manifestsha'
  snap['files']['supervisor_terminal.json']={'status':'complete','host':'5080','contract_sha256':s.canonical(self.c),'all_assigned_jobs_terminal_verified':True,'outcomes':{self.job['id']:{'terminal_manifest_sha256':'manifestsha'}}}
  def runner(argv,**kw):return subprocess.CompletedProcess(argv,0,json.dumps(snap),'')
  first=o.run_once('5080',self.root,runner);self.assertTrue(first['server_terminal_verified']);self.assertEqual(first['ssh_calls'],1)
  def fail(*a,**k):raise AssertionError('cached terminal must not SSH')
  second=o.run_once('5080',self.root,fail);self.assertEqual(second['ssh_calls'],0);self.assertTrue(second['reused_terminal_evidence']);self.assertEqual(first['observed_unix'],second['observed_unix'])
 def test_parent_inventory_nine_ids_once(self):
  bundle=PROJECT/'search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1';inv=json.loads((bundle/'recovery_inventory.json').read_text());parent=json.loads((PROJECT/'search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1/execution_contract.json').read_text());local=self.root/'parent_execution_contract.json';write(local,parent)
  c=copy.deepcopy(self.c);c['parent']={'contract':'parent_execution_contract.json','contract_sha256':s.canonical(parent)};c['jobs']=[{**r['job'],'host':r['host']} for r in inv['remaining_jobs']];write(self.root/'recovery_inventory.json',inv)
  analysis={'host':'5080','status_counts':{'in_progress':1}};a=o.aggregate_originals(self.root,c,analysis);self.assertEqual(a['parent_conditions'],9);self.assertEqual(a['original_completed_count'],7)
  inv['original_completed_jobs'].append(inv['original_completed_jobs'][0]);write(self.root/'recovery_inventory.json',inv)
  with self.assertRaises(ValueError):o.aggregate_originals(self.root,c,analysis)
 def revision_pointer(self,revision):
  write(self.root/'execution_contract_5090.json',revision)
  write(self.root/'current.json',{'bundle':str(self.root),'contract_sha256':s.canonical(self.c),'host_contracts':{'5080':{'path':'execution_contract.json','contract_sha256':s.canonical(self.c)},'5090':{'path':'execution_contract_5090.json','contract_sha256':s.canonical(revision)}}})
 def test_primary_api_and_per_host_revision(self):
  revision=copy.deepcopy(self.c);revision['hosts']['5090'].update(python='/dedicated/runtime_env/bin/python',runtime_expected={'numpy':'2.4.4'},environment={'NUMPY_RUNTIME':'dedicated'})
  self.revision_pointer(revision)
  self.assertEqual(o.load_contract(self.root),self.c);self.assertEqual(o.load_contract(self.root,'5080'),self.c);self.assertEqual(o.load_contract(self.root,'5090'),revision)
 def test_revision_cannot_change_other_owner(self):
  revision=copy.deepcopy(self.c);revision['hosts']['5080']['python']='/wrong/bin/python';self.revision_pointer(revision)
  with self.assertRaises(ValueError):o.load_contract(self.root,'5090')
 def test_revision_cannot_change_gpu_or_root(self):
  revision=copy.deepcopy(self.c);revision['hosts']['5090']['gpu_uuid']='wrong';self.revision_pointer(revision)
  with self.assertRaises(ValueError):o.load_contract(self.root,'5090')
 def test_revision_cannot_change_source(self):
  revision=copy.deepcopy(self.c);revision['source']['files_sha256']='changed';self.revision_pointer(revision)
  with self.assertRaises(ValueError):o.load_contract(self.root,'5090')
 def test_revision_pointer_seal_changed(self):
  revision=copy.deepcopy(self.c);self.revision_pointer(revision);revision['hosts']['5090']['python']='changed';write(self.root/'execution_contract_5090.json',revision)
  with self.assertRaises(ValueError):o.load_contract(self.root,'5090')
 def test_revision_path_cannot_escape_bundle(self):
  self.revision_pointer(copy.deepcopy(self.c));current=o.read(self.root/'current.json');current['host_contracts']['5090']['path']='../escape.json';write(self.root/'current.json',current)
  with self.assertRaises(ValueError):o.load_contract(self.root,'5090')
 def initial_snapshot(self):
  snap=snapshot(self.c,self.job);r=snap['runs'][0];r['history']['history']=r['history']['history'][:2];r['checkpoint_receipt']=None
  a=self.c['artifacts'][self.job['id']];a.update(files={'history.json':'recordsha'},source_best_epoch=2,source_checkpoint_sha256='binarysha',source_best_state_sha256='bestsha',source_current_state_sha256='statesha',source_resume_identity_sha256='identitysha',origin_contract_sha256='parent-sha')
  r['initial_checkpoint_sha256']='binarysha';digest=s.canonical(self.c)
  r['source_continuation']={'status':'verified','contract_sha256':digest,'job':self.job,'host':'5080','source_host':'a100','target_host':'5080','optimizer_rng_loader_generator_present':True,'segment_boundary_epoch':2,'next_epoch':3,**{key:a[key] for key in ['source_epoch','source_best_epoch','source_checkpoint_sha256','source_best_state_sha256','source_current_state_sha256','source_resume_identity_sha256']}}
  r['claim']={'job':self.job,'contract_sha256':digest,'host':'5080','pid':123,'source_checkpoint_sha256':'binarysha','manual_resume_once':True,'automatic_retry':False,'deadline_unix':time.time()+1000}
  r['progress']={'job':self.job,'contract_sha256':digest,'pid':123,'updated_unix':snap['observed_unix']}
  r['worker_started']={'pid':123,'job':self.job,'contract_sha256':digest,'command':snap['processes'][0]['argv'],'supervisor_pid':122}
  snap['processes'][0]['ppid']=122
  snap['processes'].append({'pid':122,'ppid':1,'argv':[self.c['hosts']['5080']['python'],str(self.root/'operation/titantpp_cnn_gru_resume_supervisor.py'),'--contract',str(self.root/'execution_contract.json'),'--host','5080'],'cwd':str(self.root/'source')})
  snap['files']['supervisor_status.json']={'status':'running','contract_sha256':digest,'active_job':self.job,'worker_pid':123}
  return snap
 def test_live_initial_resume_before_first_save(self):
  snap=self.initial_snapshot();a=o.analyse(snap,self.c,'5080');row=a['rows'][0];self.assertEqual(row['state'],'in_progress');self.assertEqual(row['phase'],'awaiting_first_resumed_checkpoint');self.assertEqual(row['saved_epoch'],2);self.assertEqual(row['new_completed_epochs'],0);self.assertIsNone(row['eta']);self.assertTrue(row['initial_source_checkpoint_verified'])
 def test_initial_resume_requires_every_proof(self):
  base=self.initial_snapshot()
  for kind in ['binary','history','progress','claim','source','supervisor']:
   snap=copy.deepcopy(base);r=snap['runs'][0]
   if kind=='binary':r['initial_checkpoint_sha256']='changed'
   if kind=='history':snap['file_sha256'][next(iter(snap['file_sha256']))]='changed'
   if kind=='progress':r['progress']['updated_unix']-=91
   if kind=='claim':r['claim']['pid']=0
   if kind=='source':r['source_continuation']['source_current_state_sha256']='changed'
   if kind=='supervisor':snap['processes']=snap['processes'][:1]
   with self.subTest(proof=kind):self.assertEqual(o.analyse(snap,self.c,'5080')['rows'][0]['state'],'unknown')
 def test_initial_binary_is_sha_streamed_without_loading(self):
  folder=self.root/'run'/self.job['id'];run=folder/'runs'/self.job['arm']/s.VARIANT/'seed_42';write(run/'history.json',{'history':[{'epoch':1},{'epoch':2}]});raw=b'checkpoint fixture only';(run/'last_epoch_state.pt').write_bytes(raw)
  p=subprocess.run([sys.executable,'-'],input=o.remote_script(self.c,'5080'),text=True,capture_output=True);self.assertEqual(p.returncode,0,p.stderr)
  row=json.loads(p.stdout)['runs'][0];self.assertEqual(row['initial_checkpoint_sha256'],hashlib.sha256(raw).hexdigest())
if __name__=='__main__':unittest.main(verbosity=2)
