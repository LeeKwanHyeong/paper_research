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
from types import SimpleNamespace
from unittest.mock import patch
from paper.scripts import prepare_titantpp_qk_branch_probe as p
from paper.scripts import titantpp_qk_branch_probe_runtime as r
from paper.scripts import advance_titantpp_qk_branch_probe as a

FAKE_WORKER=r'''
import argparse,hashlib,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for k in ('root','contract-sha','job-id','stage','permit-path','permit-sha'):p.add_argument('--'+k)
x=p.parse_args();root=Path(x.root);c=json.loads((root/'contract.json').read_text());job=next(j for j in c['jobs'] if j['id']==x.job_id);stage=x.stage
split='qualification' if stage=='qualification' else 'train' if stage in ('on_train','off_train') else 'validation'
condition='synthetic_both' if stage=='qualification' else 'on' if stage in ('on_validation','on_train') else 'off'
if stage=='on_validation':assert (root/'qualification_gate.json').is_file()
if stage in ('on_train','off_validation','off_train'):assert (root/'on_validation_gate.json').is_file()
if stage in ('off_validation','off_train'):assert (root/'on_train_gate.json').is_file()
mode=c.get('fake_mode')
if mode=='fail_on_validation' and stage=='on_validation' and job['seed']==52:time.sleep(.12);raise SystemExit(7)
if mode=='fail_on_train' and stage=='on_train' and job['seed']==52:time.sleep(.12);raise SystemExit(9)
if mode=='fail_off_validation' and stage=='off_validation' and job['seed']==52:time.sleep(.12);raise SystemExit(8)
if mode=='timeout' and stage=='qualification':time.sleep(10)
time.sleep(.08)
def sha(z):return hashlib.sha256(z.read_bytes()).hexdigest()
def write(z,v):
 z.parent.mkdir(parents=True,exist_ok=True)
 with z.open('x') as f:json.dump(v,f)
v={'status':'complete','job_id':job['id'],'stage':stage,'split':split,'condition':condition,'contract_sha256':c['contract_sha256'],'contract_raw_sha256':x.contract_sha,'parameters_unchanged':True,'state_including_buffers_unchanged':True,'branch_function_restored':True,'state_sha256_before':job['checkpoint_state_sha256'],'state_sha256_after':job['checkpoint_state_sha256'],'checkpoint_sha256':job['checkpoint']['sha256'],'selected_epoch':job['selected_epoch'],'no_training':True,'native_runtime_checked':True,'held_out_test_evaluated':False,'pid':os.getpid(),'argv':sys.argv,'source_closure':'fake','source_revision':'fake','runtime':{},'runtime_after':{}}
state={'state_sha256':job['checkpoint_state_sha256'],'all_named_buffers_sha256':hashlib.sha256(('buffers'+str(job['seed'])).encode()).hexdigest(),'context_cache_sha256':hashlib.sha256(b'empty-context').hexdigest()}
v.update(predictive_state_before=state,predictive_state_after=dict(state))
if stage=='qualification':v.update(qualification_passed=True,real_data_loaded=False)
else:
 n=c['datasets']['raf_spare_parts']['populations'][split]['target_count']
 row=root/'rows'/job['id']/stage;row.mkdir(parents=True);part=row/'part-00000.parquet';part.write_bytes(b'fabricated-not-real-data');manifest={'prediction_rows':n,'parts':[{'path':part.name,'sha256':sha(part),'bytes':part.stat().st_size,'rows':n}]}
 write(row/'manifest.json',{**manifest,'job_id':job['id'],'stage':stage,'split':split,'condition':condition,'contract_sha256':c['contract_sha256']})
 v.update(population={'target_count':n},prediction_rows=n,prediction_manifest=manifest,manifest_path=str((row/'manifest.json').relative_to(root)),manifest_sha256=sha(row/'manifest.json'),metrics={'count':n,'qty_rmse':1.,'qty_mae':1.,'time_nll':1.,'tail':{'count':0,'qty_rmse':None,'qty_mae':None,'time_nll':None}})
 if stage=='on_validation':
  v.update(validation_match='PASS',validation_replay_check={'six_metrics':True,'relative_tolerance':1e-5,'absolute_tolerance':1e-5},original_population_match=True,stored_prediction_rows_available=False,rowwise_replay=None)
write(root/'results'/job['id']/(stage+'_metrics.json'),v)
'''


def fake_contract(root,mode=None):
    c={'schema':r.SCHEMA,'parent_evaluation_receipt_sha256':'parent-pin','operation_code_commit':'synthetic-commit','allowed_splits':['train','validation'],'stages':list(r.STAGES),'conditions':['on','off'],'validation_tolerance':{'rtol':1e-5,'atol':1e-5},'max_parallel_per_host':3,
       'new_training':False,'frozen_selection':True,'checkpoint_reselection':False,'held_out_test_evaluated':False,
       'historical_CNN_inference':False,'automatic_retry':False,'host':'5090','remote_root':r.REMOTE,
       'deadline_unix':r.DEADLINE,'worker_timeout_seconds':10800,'pipeline_timeout_seconds':43200,
       'source':{'closure':'fake','revision':'fake'},'datasets':{'raf_spare_parts':{'sha256':'fake-input','populations':{'train':{'target_count':25779},'validation':{'target_count':6690}}}},
       'runtimes':{'5090':{'python':sys.executable,'gpu_uuid':'fake'}},'environment':{},'input_files':{},'operation_files':{},'fake_mode':mode,
       'jobs':[{'id':f'raf_spare_parts__{s}__{r.ARM}','seed':s,'arm':r.ARM,'dataset':'raf_spare_parts','evaluation_host':'5090',
                'checkpoint_metadata_sha256':'metadata'+str(s),'selection_history_sha256':'history'+str(s),'validation_reference':{'count':6690},'checkpoint_state_sha256':hashlib.sha256(('state'+str(s)).encode()).hexdigest(),'checkpoint':{'sha256':'bin'+str(s)},'selected_epoch':{42:5,52:5,62:22}[s]} for s in (42,52,62)]}
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
                   'stages':['train','validation'],'conditions':['off','on'],'validation_tolerance':{'rtol':1.,'atol':1.},'pipeline_timeout_seconds':999999}
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
        cases=(('on_validation','qualification_gate_sha256','qualification_gate.json'),('on_train','on_validation_gate_sha256','on_validation_gate.json'),('off_validation','on_validation_gate_sha256','on_validation_gate.json'),('off_train','on_validation_gate_sha256','on_validation_gate.json'))
        for stage,key,name in cases:
            with self.subTest(stage=stage):
                if (self.root/name).exists():(self.root/name).unlink()
                with self.assertRaises(FileNotFoundError):r.stage_permit(self.root,c,'raw',stage)
                p.write(self.root/name,{'approved':True})
                if stage in ('off_validation','off_train') and not (self.root/'on_train_gate.json').exists():p.write(self.root/'on_train_gate.json',{'approved':True})
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
        c,result=self.pipeline();self.assertEqual(result['status'],'complete');self.assertEqual(result['qualification_count'],3);self.assertEqual(result['data_worker_count'],12)
        self.assertEqual([x['stage'] for x in result['workers']],['qualification']*3+['on_validation']*3+['on_train']*3+['off_validation']*3+['off_train']*3)
        launches=[p.read(x) for x in (self.root/'runtime/worker_launches').glob('*.json')]
        self.assertEqual(len(launches),15);self.assertEqual(max(x['active_owned_children'] for x in launches),3)
        self.assertTrue(all(x['exit_code']==0 and x['actual_child_exit_observed'] for x in result['workers']))
        for stage in ('qualification','on_validation','on_train'):self.assertEqual(len(p.read(self.root/(stage+'_gate.json'))['receipts']),3)
    def test_failed_on_validation_never_starts_any_off(self):
        c,result=self.pipeline('fail_on_validation');self.assertEqual(result['status'],'failed')
        self.assertFalse((self.root/'on_validation_gate.json').exists())
        for stage in ('on_train','off_validation','off_train'):self.assertFalse((self.root/f'permits/{stage}.json').exists())
        self.assertTrue(any(x['exit_code']==7 for x in result['workers']));self.assertTrue((self.root/'runtime/failure.json').exists())
    def test_failed_off_validation_never_starts_off_train(self):
        c,result=self.pipeline('fail_off_validation');self.assertEqual(result['status'],'failed')
        self.assertTrue((self.root/'on_validation_gate.json').exists());self.assertFalse((self.root/'permits/off_train.json').exists())
        self.assertTrue(any(x['exit_code']==8 for x in result['workers']))
    def test_failed_on_train_never_starts_any_off(self):
        c,result=self.pipeline('fail_on_train')
        self.assertEqual(result['status'],'failed')
        self.assertTrue((self.root/'on_validation_gate.json').exists())
        self.assertFalse((self.root/'on_train_gate.json').exists())
        for stage in ('off_validation','off_train'):
            self.assertFalse((self.root/f'permits/{stage}.json').exists())
        self.assertTrue(any(x['exit_code']==9 for x in result['workers']))
    def test_off_permit_requires_on_train_gate(self):
        c=fake_contract(self.root);c['operation_files']={'operation/evaluate.py':'fake'}
        p.write(self.root/'on_validation_gate.json',{'approved':True})
        for stage in ('off_validation','off_train'):
            with self.subTest(stage=stage),self.assertRaises(FileNotFoundError):
                r.stage_permit(self.root,c,'raw',stage)
            self.assertFalse((self.root/f'permits/{stage}.json').exists())
    def test_fifteen_fresh_processes_and_stage_barriers(self):
        c,result=self.pipeline();workers=result['workers']
        self.assertEqual(len({x['pid'] for x in workers}),15)
        self.assertEqual(len({(x['stage'],x['job_id']) for x in workers}),15)
        for before,after in zip(r.STAGES,r.STAGES[1:]):
            self.assertLessEqual(max(x['finished_unix'] for x in workers if x['stage']==before),
                                 min(x['started_unix'] for x in workers if x['stage']==after))
        for seed in (42,52,62):
            job=next(j for j in c['jobs'] if j['seed']==seed)
            on=p.read(self.root/r.result_path(job['id'],'on_train'))
            self.assertEqual(on['condition'],'on');self.assertEqual(on['split'],'train')
            self.assertEqual(on['prediction_rows'],25779)
    def test_replay_and_state_guards_reject_fabricated_success(self):
        c,result=self.pipeline();job=c['jobs'][0];path=self.root/r.result_path(job['id'],'on_validation');original=p.read(path)
        changes=[('branch_function_restored',False),('state_sha256_after','drift'),('condition','off'),('stage','off_validation'),('population',{'target_count':1}),('validation_match','FAIL')]
        for key,value in changes:
            record=copy.deepcopy(original);record[key]=value;path.write_text(json.dumps(record))
            with self.subTest(key=key),self.assertRaises(ValueError):r.validate_result(self.root,c,job,'on_validation')
        for key,value in [('original_population_match',False),('stored_prediction_rows_available',True),('rowwise_replay',{'status':'PASS'})]:
            record=copy.deepcopy(original);record[key]=value;path.write_text(json.dumps(record))
            with self.subTest(proof=key),self.assertRaises(ValueError):r.validate_result(self.root,c,job,'on_validation')
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
        receipt={'status':'complete','candidate_count':3,'data_worker_count':12,'qualification_count':3,
          'seed_count':3,'stages':list(r.STAGES),'actual_supervisor_child_exit0_count':1,'automatic_retry':False,
          'actual_native_worker_exit0_count':15,'actual_data_worker_exit0_count':12,'actual_pipeline_exit0_count':1,
          'validation_replay_count':3,'on_replay_count':3,'on_train_count':3,'off_worker_count':6,'historical_CNN_inference':False,'held_out_test_evaluated':False,'new_training':False,
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

    def collector_modules(self):
        # Isolate the already separately tested scientific worker. Operational
        # fixtures contain fabricated parquet bytes, never research predictions.
        def verify_gate(root,c,stage,digest):
            path=root/(stage+'_gate.json');self.assertEqual(p.sha(path),digest)
            gate=p.read(path);ids={j['id'] for j in c['jobs']}
            if set(gate['receipts'])!=ids or set(gate['worker_exits'])!=ids:
                raise ValueError('Incomplete synthetic gate')
            for j in c['jobs']:
                for label in ('receipts','worker_exits'):
                    entry=gate[label][j['id']]
                    if p.sha(root/entry['path'])!=entry['sha256']:raise ValueError('Gate hash differs')
                ex=p.read(root/gate['worker_exits'][j['id']]['path'])
                if ex['exit_code']!=0:raise ValueError('Gate child failed')
            return gate
        worker=SimpleNamespace(verify_gate=verify_gate,validate_six_metrics=lambda left,right:None)
        helper=SimpleNamespace(check_runtime=lambda *args,**kwargs:None)
        return lambda path,name:helper if str(path).endswith('helper.py') else worker

    def completed_originals(self):
        c,result=self.pipeline();raw=p.sha(self.root/'contract.json')
        digest=hashlib.sha256(b'').hexdigest();pipeline_pid=99999999
        p.write(self.root/'runtime/pipeline_actual_exit.json',{'actual_child_exit_observed':True,'exit_code':0,
            'contract_sha256':c['contract_sha256'],'pid':pipeline_pid,'stdout_sha256':digest,'stderr_sha256':digest})
        (self.root/'runtime/pipeline.stdout').write_bytes(b'');(self.root/'runtime/pipeline.stderr').write_bytes(b'')
        p.write(self.root/'runtime/supervisor_child_exit.json',{'actual_child_exit_observed':True,'exit_code':0})
        p.write(self.root/'runtime/terminal.json',{'status':'complete','all_owned_absent':True,
            'contract_sha256':c['contract_sha256'],'pipeline_actual_exit':0,
            'root_owners':{'owners':[],'query_errors':[]},'gpu':{'gpu_uuid':'fake','compute_pids':[]},
            'owner_probes':[{'pid':pid,'state':'absent','explicit_absence':True} for pid in
                            [x['pid'] for x in result['workers']]+[pipeline_pid]],'observed_unix':time.time()})
        return c,raw,result

    def test_positive_terminal_requires_fifteen_workers_sixteen_owners_and_twelve_receipts(self):
        c,raw,result=self.completed_originals()
        with patch.object(a,'module',side_effect=self.collector_modules()):
            receipts,terminal=a.check_receipts(self.root,c,raw)
        self.assertEqual(len(receipts),12)
        self.assertEqual(len(terminal['owner_probes']),16)
        self.assertEqual(len({x['pid'] for x in terminal['owner_probes']}),16)
        self.assertEqual(sum('/on_train' in k for k in receipts),3)

    def test_missing_on_train_exit_or_receipt_blocks_terminal(self):
        c,raw,_=self.completed_originals();job=c['jobs'][0]
        paths=[self.root/r.result_path(job['id'],'on_train'),
               self.root/f"runtime/worker_exits/on_train_{job['id']}.json"]
        for path in paths:
            original=path.read_bytes();path.unlink()
            try:
                with self.subTest(path=str(path)),patch.object(a,'module',side_effect=self.collector_modules()),self.assertRaises((ValueError,FileNotFoundError)):
                    a.check_receipts(self.root,c,raw)
            finally:path.write_bytes(original)

    def test_terminal_owner_and_gpu_mutations_rejected(self):
        c,raw,_=self.completed_originals();path=self.root/'runtime/terminal.json';original=p.read(path)
        changes=[]
        x=copy.deepcopy(original);x['owner_probes'].pop();changes.append(x)
        x=copy.deepcopy(original);x['owner_probes'][0]=copy.deepcopy(x['owner_probes'][1]);changes.append(x)
        x=copy.deepcopy(original);x['owner_probes'][0]['state']='present';changes.append(x)
        x=copy.deepcopy(original);x['owner_probes'][0]['explicit_absence']=False;changes.append(x)
        x=copy.deepcopy(original);x['gpu']['compute_pids']=[123];changes.append(x)
        x=copy.deepcopy(original);x['root_owners']['query_errors']=['unreadable'];changes.append(x)
        for number,value in enumerate(changes):
            path.write_text(json.dumps(value))
            with self.subTest(number=number),patch.object(a,'module',side_effect=self.collector_modules()),self.assertRaises(ValueError):
                a.check_receipts(self.root,c,raw)
        path.write_text(json.dumps(original))

    def test_on_validation_population_and_legacy_row_claim_rejected(self):
        c,result=self.pipeline();job=c['jobs'][0];path=self.root/r.result_path(job['id'],'on_validation');original=p.read(path)
        for mutation in ({'original_population_match':False},{'stored_prediction_rows_available':True},
                         {'rowwise_replay':{'status':'PASS'}},{'validation_match':'FAIL'},
                         {'population':{'target_count':6690,'foreign_target_digest':'x'}},
                         {'validation_replay_check':{'six_metrics':True,'relative_tolerance':1e-4,'absolute_tolerance':1e-5}}):
            path.write_text(json.dumps({**original,**mutation}))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):r.validate_result(self.root,c,job,'on_validation')
        path.write_text(json.dumps(original))

    def test_predictive_state_dict_required_not_just_unchanged_flags(self):
        c,result=self.pipeline();job=c['jobs'][0];path=self.root/r.result_path(job['id'],'off_train');original=p.read(path)
        changes=[]
        for field in ('predictive_state_before','predictive_state_after'):
            for value in (None,{},'not-a-dict'):
                x=copy.deepcopy(original);x[field]=value;changes.append(x)
        for key in ('state_sha256','all_named_buffers_sha256','context_cache_sha256'):
            x=copy.deepcopy(original);x['predictive_state_after'][key]='f'*64;changes.append(x)
        for value in ('not-a-sha','F'*64,'a'*63,17):
            x=copy.deepcopy(original)
            for field in ('predictive_state_before','predictive_state_after'):x[field]['all_named_buffers_sha256']=value
            changes.append(x)
        x=copy.deepcopy(original)
        for field in ('predictive_state_before','predictive_state_after'):x[field]['extra']='a'*64
        changes.append(x)
        for number,value in enumerate(changes):
            path.write_text(json.dumps(value))
            with self.subTest(number=number),self.assertRaises(ValueError):r.validate_result(self.root,c,job,'off_train')
        path.write_text(json.dumps(original));self.assertTrue(r.validate_result(self.root,c,job,'off_train'))

    def test_each_seed_fresh_initial_predictive_state_matches_all_stages(self):
        c,raw,result=self.completed_originals();job=c['jobs'][0];stage='off_train'
        receipt_path=self.root/r.result_path(job['id'],stage);receipt=p.read(receipt_path)
        for field in ('predictive_state_before','predictive_state_after'):receipt[field]['context_cache_sha256']='f'*64
        receipt_path.write_text(json.dumps(receipt))
        # Individually unchanged state remains valid; its fresh initial context
        # must nevertheless equal the same seed's other four stage processes.
        self.assertTrue(r.validate_result(self.root,c,job,stage))
        exit_path=self.root/f'runtime/worker_exits/{stage}_{job["id"]}.json';ex=p.read(exit_path)
        ex['receipt']['sha256']=p.sha(receipt_path);exit_path.write_text(json.dumps(ex))
        path=self.root/'runtime/pipeline_complete.json';pipeline=p.read(path)
        pipeline['workers']=[ex if x['stage']==stage and x['job_id']==job['id'] else x for x in pipeline['workers']]
        path.write_text(json.dumps(pipeline))
        with patch.object(a,'module',side_effect=self.collector_modules()),self.assertRaises(ValueError):a.check_receipts(self.root,c,raw)

    def test_coordinated_duplicate_worker_and_owner_pids_rejected(self):
        c,raw,result=self.completed_originals();job=c['jobs'][0]
        first=next(x for x in result['workers'] if x['stage']=='on_validation' and x['job_id']==job['id'])
        ex_path=self.root/f"runtime/worker_exits/on_train_{job['id']}.json"
        ex=p.read(ex_path);old_pid=ex['pid'];ex['pid']=first['pid'];ex['launch_identity']['pid']=first['pid']
        receipt_path=self.root/r.result_path(job['id'],'on_train');receipt=p.read(receipt_path)
        receipt['pid']=first['pid'];receipt_path.write_text(json.dumps(receipt))
        ex['receipt']['sha256']=p.sha(receipt_path);ex_path.write_text(json.dumps(ex))
        gate_path=self.root/'on_train_gate.json';gate=p.read(gate_path)
        gate['receipts'][job['id']]['sha256']=p.sha(receipt_path)
        gate['worker_exits'][job['id']]['sha256']=p.sha(ex_path);gate_path.write_text(json.dumps(gate))
        pipeline_path=self.root/'runtime/pipeline_complete.json';pipeline=p.read(pipeline_path)
        pipeline['workers']=[ex if x['stage']=='on_train' and x['job_id']==job['id'] else x for x in pipeline['workers']]
        pipeline_path.write_text(json.dumps(pipeline))
        terminal_path=self.root/'runtime/terminal.json';terminal=p.read(terminal_path)
        for probe in terminal['owner_probes']:
            if probe['pid']==old_pid:probe['pid']=first['pid']
        terminal_path.write_text(json.dumps(terminal))
        with patch.object(a,'module',side_effect=self.collector_modules()),self.assertRaisesRegex(ValueError,'PID identities overlap'):
            a.check_receipts(self.root,c,raw)

    def test_pipeline_pid_cannot_alias_a_worker_pid(self):
        c,raw,result=self.completed_originals();path=self.root/'runtime/pipeline_actual_exit.json';direct=p.read(path)
        direct['pid']=result['workers'][0]['pid'];path.write_text(json.dumps(direct))
        with patch.object(a,'module',side_effect=self.collector_modules()),self.assertRaisesRegex(ValueError,'PID identities overlap'):
            a.check_receipts(self.root,c,raw)

    def test_pipeline_exact_stage_membership_rejects_duplicate_or_omitted_on_train(self):
        c,raw,result=self.completed_originals();path=self.root/'runtime/pipeline_complete.json';original=p.read(path)
        for mode in ('duplicate','omit'):
            changed=copy.deepcopy(original);index=next(i for i,x in enumerate(changed['workers']) if x['stage']=='on_train')
            if mode=='duplicate':changed['workers'][index]=copy.deepcopy(changed['workers'][0])
            else:changed['workers'].pop(index)
            path.write_text(json.dumps(changed))
            with self.subTest(mode=mode),patch.object(a,'module',side_effect=self.collector_modules()),self.assertRaisesRegex(ValueError,'Pipeline stage proof'):
                a.check_receipts(self.root,c,raw)
        path.write_text(json.dumps(original))

    def completed_cache_fixture(self):
        import shutil
        c,raw,result=self.completed_originals()
        bundle=self.root/'new_bundle';originals=bundle/'retrieval/originals';originals.mkdir(parents=True)
        names=['runtime','logs','results','rows','permits','qualification_gate.json','on_validation_gate.json','on_train_gate.json']
        for name in names:
            source=self.root/name
            if source.is_dir():shutil.copytree(source,originals/name)
            else:shutil.copyfile(source,originals/name)
        rawfiles={str(x.relative_to(originals)):x.read_bytes() for x in originals.rglob('*') if x.is_file()}
        manifest={'schema':'current_candidate_branch_probe_archive_v1','contract_raw_sha256':raw,
            'files':{name:{'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)} for name,data in rawfiles.items()},
            'supervisor_actual_exit':0,'error':None}
        p.write(originals/'ARCHIVE_MANIFEST.json',manifest)
        archive=bundle/'retrieval/originals.tar.gz'
        with tarfile.open(archive,'w:gz') as tar:
            for path in sorted(originals.rglob('*')):
                if path.is_file():tar.add(path,arcname=str(path.relative_to(originals)),recursive=False)
        p.write(bundle/'bootstrap_integrity.json',{'fabricated':True});pin=p.sha(bundle/'bootstrap_integrity.json')
        p.write(bundle/'control/owned_attempt_exit.json',{'actual_transport_exit_observed':True,'actual_transport_exit':0,'timeout':False})
        b={'contract_raw_sha256':raw}
        with patch.object(a,'BUNDLE',bundle),patch.object(a,'PROJECT',self.root),patch.object(a,'verify_integrity',return_value=(b,c)),patch.object(a,'module',side_effect=self.collector_modules()):
            receipt=a.complete_receipt(b,c,pin,archive,manifest)
        return bundle,b,c,pin,receipt

    def verify_fixture_cache(self,bundle,b,c,pin):
        with patch.object(a,'BUNDLE',bundle),patch.object(a,'PROJECT',self.root),patch.object(a,'verify_integrity',return_value=(b,c)),patch.object(a,'module',side_effect=self.collector_modules()):
            return a.verify_complete(pin)

    def test_complete_cache_positive_with_all_native_originals(self):
        bundle,b,c,pin,receipt=self.completed_cache_fixture()
        self.assertEqual(self.verify_fixture_cache(bundle,b,c,pin),receipt)
        self.assertEqual(receipt['actual_native_worker_exit0_count'],15)
        self.assertEqual(receipt['data_worker_count'],12)
        self.assertEqual(receipt['parent_evaluation_receipt_sha256'],'parent-pin')

    def test_complete_cache_rejects_empty_files_missing_on_train_and_wrong_parent(self):
        bundle,b,c,pin,receipt=self.completed_cache_fixture();path=bundle/'retrieval/final_receipt.json'
        for mutation in ({'files':{}},{'on_train_count':0},{'parent_evaluation_receipt_sha256':'foreign'},
                         {'stages':['qualification','on_validation','off_validation','off_train']},
                         {'actual_native_worker_exit0_count':12},{'data_worker_count':9}):
            path.write_text(json.dumps({**receipt,**mutation}))
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.verify_fixture_cache(bundle,b,c,pin)
        path.write_text(json.dumps(receipt))

    def test_complete_cache_rehashed_part_cannot_override_archived_original(self):
        bundle,b,c,pin,receipt=self.completed_cache_fixture()
        part=next((bundle/'retrieval/originals/rows').rglob('part-00000.parquet'));part.write_bytes(b'changed rows')
        receipt['files'][str(part.relative_to(self.root))]=p.sha(part)
        (bundle/'retrieval/final_receipt.json').write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):self.verify_fixture_cache(bundle,b,c,pin)


if __name__=='__main__':unittest.main()
