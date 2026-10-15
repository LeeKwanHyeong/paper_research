"""No GPU/data/SSH: guards, immutable archives, and a local fabricated pipeline."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch
from paper.scripts import prepare_titantpp_value_branch_probe as p
from paper.scripts import titantpp_value_branch_probe_runtime as r
from paper.scripts import advance_titantpp_value_branch_probe as a

FAKE_WORKER=r'''
import argparse,hashlib,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for k in ('root','contract-sha','job-id','stage','permit-path','permit-sha'):p.add_argument('--'+k)
x=p.parse_args();root=Path(x.root);c=json.loads((root/'contract.json').read_text());job=next(j for j in c['jobs'] if j['id']==x.job_id);stage=x.stage
split='qualification' if stage=='qualification' else 'train' if stage=='off_train' else 'validation'
condition='synthetic_both' if stage=='qualification' else 'on' if stage=='on_validation' else 'off'
if stage=='on_validation':assert (root/'qualification_gate.json').is_file()
if stage in ('off_validation','off_train'):assert (root/'on_validation_gate.json').is_file()
mode=c.get('fake_mode')
if mode=='fail_on_validation' and stage=='on_validation' and job['seed']==52:time.sleep(.12);raise SystemExit(7)
if mode=='fail_off_validation' and stage=='off_validation' and job['seed']==52:time.sleep(.12);raise SystemExit(8)
if mode=='timeout' and stage=='qualification':time.sleep(10)
time.sleep(.08)
def sha(z):return hashlib.sha256(z.read_bytes()).hexdigest()
def write(z,v):
 z.parent.mkdir(parents=True,exist_ok=True)
 with z.open('x') as f:json.dump(v,f)
v={'status':'complete','job_id':job['id'],'stage':stage,'split':split,'condition':condition,'contract_sha256':c['contract_sha256'],'contract_raw_sha256':x.contract_sha,'parameters_unchanged':True,'state_including_buffers_unchanged':True,'branch_function_restored':True,'state_sha256_before':job['checkpoint_state_sha256'],'state_sha256_after':job['checkpoint_state_sha256'],'checkpoint_sha256':job['checkpoint']['sha256'],'selected_epoch':job['selected_epoch'],'no_training':True,'native_runtime_checked':True,'held_out_test_evaluated':False,'pid':os.getpid(),'argv':sys.argv,'source_closure':'fake','source_revision':'fake','runtime':{},'runtime_after':{}}
if stage=='qualification':v.update(qualification_passed=True,real_data_loaded=False)
else:
 n=c['datasets']['raf_spare_parts']['populations'][split]['target_count']
 row=root/'rows'/job['id']/stage;row.mkdir(parents=True);part=row/'part-00000.parquet';part.write_bytes(b'fabricated-not-real-data');manifest={'prediction_rows':n,'parts':[{'path':part.name,'sha256':sha(part),'bytes':part.stat().st_size,'rows':n}]}
 write(row/'manifest.json',{**manifest,'job_id':job['id'],'stage':stage,'split':split,'condition':condition,'contract_sha256':c['contract_sha256']})
 v.update(population={'target_count':n},prediction_rows=n,prediction_manifest=manifest,manifest_path=str((row/'manifest.json').relative_to(root)),manifest_sha256=sha(row/'manifest.json'),metrics={'count':n,'qty_rmse':1.,'qty_mae':1.,'time_nll':1.,'tail':{'count':0,'qty_rmse':None,'qty_mae':None,'time_nll':None}})
 if stage=='on_validation':
  v.update(validation_match='PASS',validation_replay_check={'six_metrics':True,'relative_tolerance':1e-5,'absolute_tolerance':1e-5},rowwise_replay={'status':'PASS','target_count':n,'exact_targets':True,'exact_truth_and_history':True,'prediction_match':True,'time_loss_match':True,'tolerance':{'relative':1e-5,'absolute':1e-5},'max_absolute_error':{'prediction':0.,'time_nll':0.}})
write(root/'results'/job['id']/(stage+'_metrics.json'),v)
'''


def fake_contract(root,mode=None):
    c={'schema':r.SCHEMA,'allowed_splits':['train','validation'],'stages':list(r.STAGES),'conditions':['on','off'],'row_replay_tolerance':{'relative':1e-5,'absolute':1e-5},'max_parallel_per_host':3,
       'new_training':False,'frozen_selection':True,'checkpoint_reselection':False,'held_out_test_evaluated':False,
       'historical_CNN_inference':False,'automatic_retry':False,'host':'5090','remote_root':r.REMOTE,
       'deadline_unix':r.DEADLINE,'worker_timeout_seconds':10800,'pipeline_timeout_seconds':43200,
       'source':{'closure':'fake','revision':'fake'},'datasets':{'raf_spare_parts':{'populations':{'train':{'target_count':25779},'validation':{'target_count':6690}}}},
       'runtimes':{'5090':{'python':sys.executable,'gpu_uuid':'fake'}},'environment':{},'input_files':{},'operation_files':{},'fake_mode':mode,
       'jobs':[{'id':f'raf_spare_parts__{s}__{r.ARM}','seed':s,'arm':r.ARM,'dataset':'raf_spare_parts','evaluation_host':'5090',
                'checkpoint_state_sha256':'state'+str(s),'checkpoint':{'sha256':'bin'+str(s)},'selected_epoch':{42:9,52:5,62:22}[s]} for s in (42,52,62)]}
    c['contract_sha256']=p.canonical(c);p.write(root/'contract.json',c);return c


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name).resolve()
    def tearDown(self):self.temp.cleanup()
    def test_safe_rejects_escapes_absolute_and_symlink(self):
        (self.root/'link').symlink_to('/tmp')
        for name in ('../x','/tmp/x','link/x'):
            with self.subTest(name=name),self.assertRaises(ValueError):p.safe(self.root,name)
    def test_exclusive_write_preserves_original(self):
        path=self.root/'receipt.json';p.write(path,{'original':1});digest=p.sha(path)
        with self.assertRaises(FileExistsError):p.write(path,{'original':2})
        self.assertEqual(p.sha(path),digest)
    def runtime_fixture(self):
        c=fake_contract(self.root);(self.root/'operation').mkdir();(self.root/'source').mkdir()
        (self.root/'source/a').write_bytes(b'pinned')
        (self.root/'operation/runtime.py').write_bytes(Path(r.__file__).read_bytes())
        (self.root/'operation/evaluate.py').write_text('import json\ndef verify_contract(root,expected):return json.loads((root/"contract.json").read_text())\n')
        c['input_files']={'source/a':p.sha(self.root/'source/a')}
        c['operation_files']={f'operation/{n}':p.sha(self.root/'operation'/n) for n in ('runtime.py','evaluate.py')}
        self.reseal(c);return c
    def reseal(self,c):
        c['contract_sha256']=p.canonical({k:v for k,v in c.items() if k!='contract_sha256'})
        (self.root/'contract.json').write_text(json.dumps(c));return p.sha(self.root/'contract.json')
    def test_runtime_guard_valid_and_raw_mismatch(self):
        c=self.runtime_fixture();self.assertEqual(r.verify_contract(self.root,p.sha(self.root/'contract.json')),c)
        with self.assertRaises(ValueError):r.verify_contract(self.root,'0'*64)
    def test_runtime_forbids_changed_science_scope(self):
        original=self.runtime_fixture()
        mutations={'allowed_splits':['train','test'],'new_training':True,'historical_CNN_inference':True,
                   'checkpoint_reselection':True,'frozen_selection':False,'automatic_retry':True,'max_parallel_per_host':4,
                   'host':'5080','remote_root':'/tmp/foreign','deadline_unix':r.DEADLINE+1,
                   'stages':['train','validation'],'conditions':['off','on'],'row_replay_tolerance':{'relative':1.,'absolute':1.},'pipeline_timeout_seconds':999999}
        for key,value in mutations.items():
            c=copy.deepcopy(original);c[key]=value;digest=self.reseal(c)
            with self.subTest(key=key),self.assertRaises(ValueError):r.verify_contract(self.root,digest)
    def test_runtime_rejects_foreign_model_and_seed(self):
        original=self.runtime_fixture()
        for key,value in (('seed',99),('arm','titantpp_cnn'),('evaluation_host','5080')):
            c=copy.deepcopy(original);c['jobs'][0][key]=value;digest=self.reseal(c)
            with self.subTest(key=key),self.assertRaises(ValueError):r.verify_contract(self.root,digest)
    def test_operation_and_source_drift_rejected(self):
        self.runtime_fixture();raw=p.sha(self.root/'contract.json');(self.root/'source/a').write_bytes(b'changed')
        with self.assertRaises(ValueError):r.verify_contract(self.root,raw)
    def test_on_permit_requires_qualification_and_both_off_require_on_gate(self):
        c=fake_contract(self.root);c['operation_files']={'operation/evaluate.py':'fake'}
        cases=(('on_validation','qualification_gate_sha256','qualification_gate.json'),('off_validation','on_validation_gate_sha256','on_validation_gate.json'),('off_train','on_validation_gate_sha256','on_validation_gate.json'))
        for stage,key,name in cases:
            with self.subTest(stage=stage):
                if (self.root/name).exists():(self.root/name).unlink()
                with self.assertRaises(FileNotFoundError):r.stage_permit(self.root,c,'raw',stage)
                p.write(self.root/name,{'approved':True})
                rel,digest=r.stage_permit(self.root,c,'raw',stage);v=p.read(self.root/rel)
                self.assertEqual(v[key],p.sha(self.root/name));self.assertEqual(p.sha(self.root/rel),digest)
                self.assertLessEqual(v['deadline_unix'],r.DEADLINE)
    def test_qualification_permit_does_not_need_realdata_gate(self):
        c=fake_contract(self.root);c['operation_files']={'operation/evaluate.py':'fake'}
        rel,_=r.stage_permit(self.root,c,'raw','qualification');v=p.read(self.root/rel)
        self.assertFalse(v['new_training']);self.assertEqual(v['stage'],'qualification')
    def test_stable_identity_retries_preexec_not_pid_reuse(self):
        class Process:
            pid=123
            def poll(self):return None
        seq=[{'state':'present','start_ticks':9,'argv':['parent']},{'state':'present','start_ticks':9,'argv':['python','worker']}]
        with patch.object(r,'process_probe',side_effect=seq):self.assertEqual(r.stable_child_identity(Process(),['python','worker'])['start_ticks'],9)
        seq=[{'state':'present','start_ticks':9,'argv':['parent']},{'state':'present','start_ticks':10,'argv':['python','worker']}]
        with patch.object(r,'process_probe',side_effect=seq),self.assertRaises(ValueError):r.stable_child_identity(Process(),['python','worker'])
    def pipeline(self,mode=None):
        c=fake_contract(self.root,mode);(self.root/'operation').mkdir();(self.root/'operation/evaluate.py').write_text(FAKE_WORKER)
        c['operation_files']={'operation/evaluate.py':p.sha(self.root/'operation/evaluate.py')}
        if mode=='timeout':c['worker_timeout_seconds']=.18
        self.reseal(c)
        probe=lambda pid,expected_start=None:{'pid':pid,'state':'present','start_ticks':pid+1000,'argv':['fabricated']}
        stable=lambda process,argv:{'pid':process.pid,'state':'present','start_ticks':process.pid+1000,'argv':argv}
        with patch.object(r,'REMOTE',str(self.root)),patch.object(r,'verify_contract',return_value=c),patch.object(r,'process_probe',side_effect=probe),patch.object(r,'stable_child_identity',side_effect=stable):
            result=r.dispatch(self.root,p.sha(self.root/'contract.json'))
        return c,result
    def test_fake_pipeline_all_on_validation_before_off_validation_then_off_train(self):
        c,result=self.pipeline();self.assertEqual(result['status'],'complete');self.assertEqual(result['qualification_count'],3);self.assertEqual(result['data_worker_count'],9)
        self.assertEqual([x['stage'] for x in result['workers']],['qualification']*3+['on_validation']*3+['off_validation']*3+['off_train']*3)
        launches=[p.read(x) for x in (self.root/'runtime/worker_launches').glob('*.json')]
        self.assertEqual(len(launches),12);self.assertEqual(max(x['active_owned_children'] for x in launches),3)
        self.assertTrue(all(x['exit_code']==0 and x['actual_child_exit_observed'] for x in result['workers']))
        for stage in ('qualification','on_validation'):self.assertEqual(len(p.read(self.root/(stage+'_gate.json'))['receipts']),3)
    def test_failed_on_validation_never_starts_any_off(self):
        c,result=self.pipeline('fail_on_validation');self.assertEqual(result['status'],'failed')
        self.assertFalse((self.root/'on_validation_gate.json').exists())
        for stage in ('off_validation','off_train'):self.assertFalse((self.root/f'permits/{stage}.json').exists())
        self.assertTrue(any(x['exit_code']==7 for x in result['workers']));self.assertTrue((self.root/'runtime/failure.json').exists())
    def test_failed_off_validation_never_starts_off_train(self):
        c,result=self.pipeline('fail_off_validation');self.assertEqual(result['status'],'failed')
        self.assertTrue((self.root/'on_validation_gate.json').exists());self.assertFalse((self.root/'permits/off_train.json').exists())
        self.assertTrue(any(x['exit_code']==8 for x in result['workers']))
    def test_replay_and_state_guards_reject_fabricated_success(self):
        c,result=self.pipeline();job=c['jobs'][0];path=self.root/r.result_path(job['id'],'on_validation');original=p.read(path)
        changes=[('branch_function_restored',False),('state_sha256_after','drift'),('condition','off'),('stage','off_validation'),('population',{'target_count':1}),('validation_match','FAIL')]
        for key,value in changes:
            record=copy.deepcopy(original);record[key]=value;path.write_text(json.dumps(record))
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_result(self.root,c,job,'on_validation')
        for key,value in [('time_loss_match',False),('prediction_match',False),('exact_targets',False),('exact_truth_and_history',False),('target_count',6689),('tolerance',{'relative':1.,'absolute':1.}),('max_absolute_error',{'prediction':float('nan'),'time_nll':0.})]:
            record=copy.deepcopy(original);record['rowwise_replay'][key]=value;path.write_text(json.dumps(record))
            with self.subTest(replay=key),self.assertRaises(ValueError):r.validate_result(self.root,c,job,'on_validation')
        path.write_text(json.dumps(original));self.assertEqual(r.validate_result(self.root,c,job,'on_validation')['sha256'],p.sha(path))
    def test_timeout_terminates_only_owned_children_and_records_exits(self):
        c,result=self.pipeline('timeout');self.assertEqual(result['status'],'failed')
        self.assertFalse((self.root/'qualification_gate.json').exists());self.assertTrue(all(x['actual_child_exit_observed'] for x in result['workers']))
        self.assertTrue(all(x['exit_code']!=0 for x in result['workers']))
        for x in result['workers']:
            with self.assertRaises(ProcessLookupError):os.kill(x['pid'],0)
    def test_existing_pipeline_claim_prevents_second_launch(self):
        c,result=self.pipeline();before=p.sha(self.root/'runtime/pipeline.claim')
        with patch.object(r,'REMOTE',str(self.root)),patch.object(r,'verify_contract',return_value=c),patch.object(r.subprocess,'Popen') as launch,self.assertRaises(FileExistsError):
            r.dispatch(self.root,p.sha(self.root/'contract.json'))
        launch.assert_not_called();self.assertEqual(p.sha(self.root/'runtime/pipeline.claim'),before)
    def archive(self,entries=None,mutate=False):
        entries=entries or {'results/fake/validation_metrics.json':b'{}','rows/fake/validation/part-00000.parquet':b'fake'}
        manifest={'schema':'current_candidate_branch_probe_archive_v1','contract_raw_sha256':'raw','files':{n:{'sha256':hashlib.sha256(v).hexdigest(),'bytes':len(v)} for n,v in entries.items()},'supervisor_actual_exit':0,'error':None}
        if mutate:next(iter(manifest['files'].values()))['sha256']='f'*64
        path=self.root/'archive.tar.gz'
        with tarfile.open(path,'w:gz') as tar:
            for n,data in {**entries,'ARCHIVE_MANIFEST.json':json.dumps(manifest).encode()}.items():
                info=tarfile.TarInfo(n);info.size=len(data);tar.addfile(info,io.BytesIO(data))
        return path
    def test_archive_roundtrip_and_no_overwrite(self):
        path=self.archive();dest=self.root/'originals';m=a.extract_archive(path,dest,'raw');self.assertEqual(len(m['files']),2)
        with self.assertRaises(ValueError):a.extract_archive(path,dest,'raw')
    def test_archive_hash_mismatch_writes_nothing(self):
        path=self.archive(mutate=True);dest=self.root/'originals'
        with self.assertRaises(ValueError):a.extract_archive(path,dest,'raw')
        self.assertFalse(dest.exists())
    def test_archive_escape_and_foreign_scope_rejected(self):
        for name in ('../escape','source/old.py','/tmp/escape'):
            with self.subTest(name=name):
                path=self.archive({name:b'bad'});dest=self.root/'originals'
                with self.assertRaises(ValueError):a.extract_archive(path,dest,'raw')
                self.assertFalse(dest.exists())
    def test_installer_compile_and_scope(self):
        b={'contract_raw_sha256':'raw','native_packet_files':{'contract.json':'fake'}};c={'remote_root':p.REMOTE,'pipeline_timeout_seconds':43200}
        code=a.installer_code(b,c,'packet');compile(code,'<installer>','exec')
        self.assertIn("root.mkdir(parents=True,exist_ok=False)",code);self.assertIn("supervisor_child_exit.json",code)
    def test_advance_stops_prior_attempt_before_any_remote(self):
        c=fake_contract(self.root);p.write(self.root/'control/attempt_intent.json',{'original':True})
        with patch.object(a,'BUNDLE',self.root),patch.object(a,'verify_integrity',return_value=({},c)),patch.object(a,'recorded_run') as remote,self.assertRaises(ValueError):a.advance('pin')
        remote.assert_not_called()
    def test_advance_integrity_failure_is_remote_zero(self):
        with patch.object(a,'verify_integrity',side_effect=ValueError('tamper')),patch.object(a,'recorded_run') as remote,self.assertRaises(ValueError):a.advance('pin')
        remote.assert_not_called()
    def test_completion_cannot_bless_empty_files_with_forged_counts(self):
        archive=self.archive();originals=self.root/'retrieval/originals';a.extract_archive(archive,originals,'raw')
        archive.rename(self.root/'retrieval/originals.tar.gz');archive=self.root/'retrieval/originals.tar.gz'
        c={'contract_sha256':'canonical'};b={'contract_raw_sha256':'raw'}
        receipt={'status':'complete','candidate_count':3,'data_worker_count':9,'qualification_count':3,
          'seed_count':3,'stages':list(r.STAGES),'actual_supervisor_child_exit0_count':1,'automatic_retry':False,
          'actual_native_worker_exit0_count':12,'actual_data_worker_exit0_count':9,'actual_pipeline_exit0_count':1,
          'validation_replay_count':3,'on_replay_count':3,'off_worker_count':6,'historical_CNN_inference':False,'held_out_test_evaluated':False,'new_training':False,
          'bootstrap_integrity_sha256':'pin','contract_sha256':'canonical','contract_raw_sha256':'raw',
          'originals_path':'retrieval/originals','archive_sha256':p.sha(archive),'manifest_sha256':p.sha(originals/'ARCHIVE_MANIFEST.json'),'files':{}}
        p.write(self.root/'retrieval/final_receipt.json',receipt)
        with patch.object(a,'BUNDLE',self.root),patch.object(a,'PROJECT',self.root),patch.object(a,'verify_integrity',return_value=(b,c)),patch.object(a,'check_receipts') as scientific,self.assertRaisesRegex(ValueError,'coverage'):
            a.verify_complete('pin')
        scientific.assert_not_called()
    def test_local_prepare_rejects_existing_bundle_before_validation(self):
        with patch.object(p,'BUNDLE',self.root),patch.object(p,'verify_parent') as verify,self.assertRaises(ValueError):p.prepare()
        verify.assert_not_called()
    def test_only_current_code_module_is_packed(self):
        (self.root/'contract.json').write_text('{}');(self.root/'baseline_local_references.json').write_text('{"old":"reference"}')
        b={'native_packet_files':{'contract.json':p.sha(self.root/'contract.json')}}
        with patch.object(a,'BUNDLE',self.root):raw=a.make_packet(b)
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:self.assertEqual(set(tar.getnames()),{'contract.json','PACKET_MANIFEST.json'})


if __name__=='__main__':unittest.main()
