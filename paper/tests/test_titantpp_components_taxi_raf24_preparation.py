"""Exact24 fixed-checkpoint contract tests; synthetic opaque artifacts only."""
from __future__ import annotations
import ast
import base64
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from paper.scripts import prepare_titantpp_components_taxi_raf24_frozen_evaluation as prepare
from paper.scripts import evaluate_titantpp_components_taxi_raf24_frozen as evaluator
from paper.tests.test_titantpp_components_evaluation_preparation import BindingFixture


class TaxiRAF24Fixture(BindingFixture):
    def __init__(self, root):
        super().__init__(root)
        original = self.bindings['jobs']
        self.bindings['jobs'] = [j for j in original if j['dataset'] in prepare.DATASETS]
        self.bindings['schema'] = 'titantpp_components_taxi_raf24_checkpoint_bindings_v1'
        self.bindings['datasets'] = {d: self.bindings['datasets'][d] for d in prepare.DATASETS}
        self.runtimes = {'5080': self.runtimes['5080']}
        self.design_closure = prepare.canonical(self.design)
        training = prepare.read(self.root / self.bindings['training_contract_path'])
        training.update(hosts={'5080': {'root': '/synthetic/original5080', 'python': '/synthetic/bin/python'}},
                        jobs=[{'id': j['id'], 'dataset': j['dataset'], 'seed': j['seed'], 'arm': j['arm'],
                               'host': '5080' if j['dataset'] in prepare.DATASETS else '5090'} for j in original])
        self.parent = prepare.canonical(training)
        self.bindings['training_contract_sha256'] = self.write_json(self.bindings['training_contract_path'], training)
        records, rows = {}, []
        for j in self.bindings['jobs']:
            j['cohort'] = prepare.COHORT
            self.refresh_json(j, 'terminal_manifest_path', 'terminal_manifest_sha256', lambda v: v.update(contract_sha256=self.parent))
            self.refresh_json(j, 'worker_exit_path', 'worker_exit_sha256', lambda v: v.update(contract_sha256=self.parent,supervisor_pid=100))
            records['worker_exits/'+j['id']+'.json'] = prepare.read(self.root/j['worker_exit_path'])
            records['run/'+j['id']+'/terminal_manifest.json'] = prepare.read(self.root/j['terminal_manifest_path'])
            rows.append({'job': records['worker_exits/'+j['id']+'.json']['job'], 'status':'complete',
                'selected_checkpoint_file_present':True, 'remote_all_terminal_file_hashes_verified':True,
                'terminal_manifest_raw_sha256':j['terminal_manifest_sha256'], 'worker_exit_raw_sha256':j['worker_exit_sha256']})
        owner = {'pid':100,'ppid':200,'contract_sha256':self.parent,
                 'argv':['/synthetic/bin/python','/synthetic/original5080/operation/campaign.py','--contract',
                         '/synthetic/original5080/execution_contract.json','--host','5080','--mode','dispatch']}
        records.update({n:deepcopy(owner) for n in ('supervisor.claim','phases/train_launch.json','phases/train_exit.json')})
        records['phases/train_exit.json'].update(phase='train',os_exit_code=0)
        records['phases/train_complete.json']={'phase':'train','host':'5080','contract_sha256':self.parent,
                                               'subprocess_os_exit_code':0,'gpu_compute_pids':[]}
        records['status.json']={'status':'complete','completed_conditions':24,'held_out_test_evaluated':False,
          'completed':{j['id']:{'worker_exit_sha256':j['worker_exit_sha256'],'terminal_manifest_sha256':j['terminal_manifest_sha256']} for j in self.bindings['jobs']}}
        self.snapshot={'host':'5080','root':'/synthetic/original5080','contract_sha256':self.parent,
            'source_files_sha256':self.source_closure,'gpu_uuid':prepare.base.GPU_UUIDS['5080'],
            'gpu_pids':[],'processes':{},'observed_unix':100.,'files':{},
            'terminal_hashes':{j['id']:{'all_files_verified':True,
                              'files':dict.fromkeys(prepare.read(self.root/j['terminal_manifest_path'])['files'],True)} for j in self.bindings['jobs']}}
        for name,value in records.items():
            raw=json.dumps(value,sort_keys=True).encode()
            self.snapshot['files'][name]={'base64':base64.b64encode(raw).decode(),'raw_sha256':hashlib.sha256(raw).hexdigest()}
        self.bindings['parentterminal_evidence_path']='evidence/original5080_snapshot.json'
        self.bindings['parentterminal_evidence_sha256']=self.write_json(self.bindings['parentterminal_evidence_path'],self.snapshot)
        self.monitor={'parent_contract_sha256':self.parent,'all_training_terminal_verified':True,
            'totals':{'total':36,'completed':36,'running':0,'waiting':0,'unconfirmed':0,'all_training_terminal_verified':True},
            'groups':{'original_5080':{'status':'training_complete_remote_verified','completed':24,'total':24,
               'pipeline_os_exit_directly_observed':False,'observed_unix':100.,'jobs':rows,
               'snapshot_raw_sha256':self.bindings['parentterminal_evidence_sha256']}}}
        self.bindings['training_monitor_sha256']=self.write_json(self.bindings['training_monitor_path'],self.monitor)
        self.local_root=str(self.root.resolve());self.remote_root=self.local_root
        instruction='Taxi/RAF24 고정 선택 모델 Validation 재현 뒤 Test 평가를 승인한다.'
        self.approval={'schema':'titantpp_components_taxi_raf24_partial_approval_v1',
            'partial_cohort_explicitly_authorized':True,'evaluation_scope':'test_taxi_raf24_followup',
            'cohort':prepare.COHORT,'host':'5080','expected_condition_count':24,
            'job_ids':sorted(prepare.expected_ids()),'parent_contract_sha256':self.parent,
            'frozen_source_closure':self.source_closure,'new_training':False,'original36_global_gate_modified':False,
            'user_instruction':instruction,'user_instruction_sha256':hashlib.sha256(instruction.encode()).hexdigest(),
            'authorization_root':self.remote_root,'allowed_local_binding_root':self.local_root}
        self.bindings['partial_approval_path']='evidence/approval.json'
        self.bindings['partial_approval_sha256']=self.write_json('evidence/approval.json',self.approval)
        files={'operation/evaluate.py':self.write_bytes('operation/evaluate.py',Path(evaluator.__file__).read_bytes())}
        self.bindings['evaluator_integrity']={'files':files,'closure':prepare.canonical(files)}

    def pins(self):
        stack=ExitStack()
        stack.enter_context(patch.multiple(prepare,SOURCE=self.source_closure,PARENT=self.parent,
            ORIGINAL_DESIGN_CLOSURE=self.design_closure,AUTHORIZED_LOCAL_ROOT=self.local_root,AUTHORIZED_REMOTE_ROOT=self.remote_root))
        stack.enter_context(patch.multiple(prepare.base,SOURCE=self.source_closure,PARENT=self.parent,
            TEST_COUNTS={d:5 for d in prepare.DATASETS}))
        stack.enter_context(patch.multiple(evaluator,SOURCE_CLOSURE=self.source_closure,TRAINING_CONTRACT=self.parent))
        return stack

    def build(self):
        with self.pins(): return prepare.build_contract(self.root,self.design,self.bindings,self.runtimes)

    def refresh(self):
        self.bindings['parentterminal_evidence_sha256']=self.write_json(self.bindings['parentterminal_evidence_path'],self.snapshot)
        self.monitor['groups']['original_5080']['snapshot_raw_sha256']=self.bindings['parentterminal_evidence_sha256']
        self.bindings['training_monitor_sha256']=self.write_json(self.bindings['training_monitor_path'],self.monitor)

    def mutate_record(self,name,change):
        value=json.loads(base64.b64decode(self.snapshot['files'][name]['base64']));change(value)
        raw=json.dumps(value,sort_keys=True).encode()
        self.snapshot['files'][name]={'base64':base64.b64encode(raw).decode(),'raw_sha256':hashlib.sha256(raw).hexdigest()}
        self.refresh()


class TaxiRAF24PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir='/private/tmp');self.addCleanup(self.temp.cleanup)
        self.f=TaxiRAF24Fixture(self.temp.name)

    def test_exact24_builder_and_static_evaluator(self):
        c=self.f.build();self.assertEqual(len(c['jobs']),24)
        self.assertEqual(set(c['datasets']),set(prepare.DATASETS));self.assertEqual(c['expected_seeds'],[42,52,62])
        self.assertFalse(c['new_training']);self.assertFalse(c['checkpoint_reselection'])
        self.f.write_json('contract.json',c)
        with self.f.pins(): self.assertEqual(evaluator.verify_contract(self.f.root)['expected_jobs'],24)

    def test_missing_duplicate_or_foreign_dataset_block(self):
        jobs=deepcopy(self.f.bindings['jobs'])
        for altered in (jobs[:-1],jobs[:-1]+[jobs[0]],jobs[:-1]+[{**jobs[-1],'id':'intermittent_frozen_5000__62__foreign'}]):
            self.f.bindings['jobs']=altered
            with self.assertRaises(ValueError):self.f.build()

    def test_selected_binary_history_and_reference_are_bound(self):
        job=self.f.bindings['jobs'][0]
        for key,value in (('selected_epoch',3),('checkpoint_state_sha256','f'*64),('training_host','5090')):
            before=job[key];job[key]=value
            with self.assertRaises(ValueError):self.f.build()
            job[key]=before
        (self.f.root/job['checkpoint']['relative_path']).write_bytes(b'altered')
        with self.assertRaises(ValueError):self.f.build()

    def test_approval_scope_no_training_and_no_root_escape(self):
        for key,value in (('new_training',True),('job_ids',[]),('expected_condition_count',12),('authorization_root','/foreign'),('user_instruction_sha256','f'*64)):
            self.f.bindings['partial_approval_sha256']=self.f.write_json('evidence/approval.json',{**self.f.approval,key:value})
            with self.assertRaises(ValueError):self.f.build()

    def test_dispatcher_or_worker_failure_and_modified_snapshot_block(self):
        self.f.mutate_record('phases/train_exit.json',lambda v:v.update(os_exit_code=1))
        with self.assertRaises(ValueError):self.f.build()
        self.f.mutate_record('phases/train_exit.json',lambda v:v.update(os_exit_code=0))
        self.f.mutate_record('worker_exits/'+self.f.bindings['jobs'][0]['id']+'.json',lambda v:v.update(returncode=1))
        with self.assertRaises(ValueError):self.f.build()

    def test_partial_training_or_binary_witness_blocks(self):
        self.f.monitor['all_training_terminal_verified']=False;self.f.refresh()
        with self.assertRaises(ValueError):self.f.build()
        self.f.monitor['all_training_terminal_verified']=True
        self.f.snapshot['terminal_hashes'][self.f.bindings['jobs'][0]['id']]['all_files_verified']=False;self.f.refresh()
        with self.assertRaises(ValueError):self.f.build()

    def test_raf_full_parent_and_training_subset_are_distinct_required_bytes(self):
        spec=self.f.bindings['datasets']['raf_spare_parts'];self.assertNotEqual(spec['data_sha256'],spec['training_data_sha256'])
        spec['training_data_sha256']=spec['data_sha256']
        with self.assertRaises(ValueError):self.f.build()

    def test_scientific_asts_equal_completed_inter62(self):
        old=ast.parse(Path(evaluator.__file__).with_name('evaluate_titantpp_components_inter62_frozen.py').read_text())
        new=ast.parse(Path(evaluator.__file__).read_text())
        for name in ('worker','validate_payload','validate_frame','require_raf_training_slice','population_identity','qualify','compare_validation_metrics','strict_selected_epoch','verify_training_metadata','qualify_synthetic'):
            a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==name)
            b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==name)
            self.assertEqual(ast.dump(a,include_attributes=False),ast.dump(b,include_attributes=False),name)

if __name__=='__main__':unittest.main()
