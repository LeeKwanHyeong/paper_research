"""Synthetic exact8 partial-scope qualification; no real inference/remote work."""
from __future__ import annotations

import ast
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'paper/scripts/evaluate_titantpp_components_inter8_frozen.py'
OLD_SCRIPT = ROOT / 'paper/scripts/evaluate_titantpp_components_frozen.py'
PARENT = ROOT / 'search_artifacts/titantpp_cnn_gru_components_20261012_v5'
SPEC = importlib.util.spec_from_file_location('_inter8_eval_unit', SCRIPT)
eval = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(eval)
FIXTURE_SPEC = importlib.util.spec_from_file_location('_old_component_contract_fixtures',
    ROOT / 'paper/tests/test_titantpp_components_frozen_evaluator.py')
fixtures = importlib.util.module_from_spec(FIXTURE_SPEC); FIXTURE_SPEC.loader.exec_module(fixtures)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def seal(c):
    c = deepcopy(c); c.pop('contract_sha256', None)
    c['contract_sha256'] = eval.canonical(c)
    return c


class Inter8EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='inter8-eval-synthetic-', dir='/private/tmp')
        self.root = Path(self.temp.name)
        parent = json.loads((PARENT / 'execution_contract.json').read_text())
        old = fixtures.fabricated_contract()
        self.c = {**old, 'schema': eval.KIND, 'cohort': eval.COHORT, 'evaluation_scope': 'test_partial_followup',
            'expected_seeds': [42, 52], 'expected_jobs': 8, 'source_training_cohort_count': 36,
            'original36_all_training_terminal_verified': False, 'original36_global_gate_modified': False,
            'partial_cohort_explicitly_authorized': True,
            'datasets': {'intermittent_frozen_5000': old['datasets']['intermittent_frozen_5000']},
            'jobs': [{**j, 'cohort': eval.COHORT} for j in old['jobs']
                if j['dataset'] == 'intermittent_frozen_5000' and j['seed'] in (42, 52)]}
        self.expected_runtime, self.runtime = fixtures.ComponentEvaluatorContracts.host_runtime(self, '5090')
        self.c['runtimes'] = {'5090': self.expected_runtime}
        training_path = self.root / 'metadata/training_contract.json'
        training_path.parent.mkdir(parents=True)
        shutil.copyfile(PARENT / 'execution_contract.json', training_path)
        evaluator_path = self.root / 'operation/evaluator.py'; evaluator_path.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, evaluator_path)
        files = {'operation/evaluator.py': eval.sha(evaluator_path)}
        self.c.update(training_contract_path='metadata/training_contract.json',
            training_contract_sha256=eval.sha(training_path), evaluator_integrity={'files': files, 'closure': eval.canonical(files)})
        ids = sorted(j['id'] for j in self.c['jobs'])
        instruction = 'Intermittent42/52 component8 partial Test followup is explicitly authorized for this fabricated fixture.'
        self.approval = {'schema': 'titantpp_components_inter8_partial_approval_v1',
            'partial_cohort_explicitly_authorized': True, 'user_instruction': instruction,
            'user_instruction_sha256': hashlib.sha256(instruction.encode()).hexdigest(),
            'evaluation_scope': 'test_partial_followup', 'host': '5090', 'cohort': eval.COHORT,
            'expected_condition_count': 8, 'job_ids': ids, 'parent_contract_sha256': eval.TRAINING_CONTRACT,
            'frozen_source_closure': eval.SOURCE_CLOSURE, 'authorization_root': str(self.root),
            'allowed_local_binding_root': str(self.root), 'new_training': False, 'original36_global_gate_modified': False}
        dump(self.root / 'metadata/partial_approval.json', self.approval)
        self.c.update(partial_approval_path='metadata/partial_approval.json',
                      partial_approval_sha256=eval.sha(self.root / 'metadata/partial_approval.json'))
        spec = parent['hosts']['5090']
        dispatcher = [spec['python'],spec['root']+'/operation/campaign.py','--contract',
            spec['root']+'/execution_contract.json','--host','5090','--mode','dispatch']
        owner = {'contract_sha256': eval.TRAINING_CONTRACT, 'pid': 100, 'ppid': 200, 'argv': dispatcher}
        raw_files = {name: deepcopy(owner) for name in ('supervisor.claim','phases/train_launch.json','phases/train_exit.json')}
        raw_files['phases/train_exit.json']['os_exit_code'] = 1
        observations = {'100': {'pid': 100, 'state': 'absent', 'evidence_sources': list(raw_files)},
                        '200': {'pid': 200, 'state': 'absent', 'evidence_sources': list(raw_files)}}
        for index, job in enumerate(self.c['jobs']):
            pid = 300 + index
            name = 'worker_exits/' + job['id'] + '.json'
            raw_files[name] = {'contract_sha256': eval.TRAINING_CONTRACT, 'worker_group_pid': pid,
                'supervisor_pid': 100, 'job': {'id': job['id']}, 'returncode': 0, 'actual_child_exit_observed': True}
            observations[str(pid)] = {'pid': pid, 'state': 'absent', 'evidence_sources': [name]}
            observations['100']['evidence_sources'].append(name)
        encoded = {}
        for name, value in raw_files.items():
            raw = json.dumps(value).encode()
            encoded[name] = {'base64': base64.b64encode(raw).decode(), 'raw_sha256': hashlib.sha256(raw).hexdigest()}
        self.snapshot = {'host': '5090', 'root': spec['root'], 'contract_sha256': eval.TRAINING_CONTRACT,
            'source_files_sha256': eval.SOURCE_CLOSURE, 'gpu_uuid': eval.GPU_UUIDS['5090'],
            'gpu_inventory_uuids': [eval.GPU_UUIDS['5090']], 'gpu_pids': [], 'processes': {},
            'observed_unix': 100., 'files': encoded, 'process_observations': observations,
            'observed_owner_pids': [100, 200, *range(300, 308)],
            'terminal_hashes': {i: {'all_files_verified': True} for i in ids}}
        dump(self.root / 'metadata/parent_terminal_snapshot.json', self.snapshot)
        self.c.update(parentterminal_evidence_path='metadata/parent_terminal_snapshot.json',
            parentterminal_evidence_sha256=eval.sha(self.root / 'metadata/parent_terminal_snapshot.json'))
        self.monitor = {'parent_contract_sha256': eval.TRAINING_CONTRACT, 'all_training_terminal_verified': False,
            'groups': {'parent_5090': {'status': 'planned_queue_exclusion_terminal_verified', 'host': '5090',
                'completed': 8, 'total': 8, 'identity_verified': True, 'owner_process_absence_verified': True,
                'administrative_dispatcher_actual_exit': 1, 'parent_original12_complete': False,
                'original_failure_and_claim_preserved': True, 'jobs': [{'job': {'id': i}} for i in ids],
                'snapshot_raw_sha256': self.c['parentterminal_evidence_sha256'], 'root': spec['root'], 'observed_unix': 100.}}}
        dump(self.root / 'metadata/training_monitor.json', self.monitor)
        self.c.update(training_monitor_path='metadata/training_monitor.json',
            training_monitor_sha256=eval.sha(self.root / 'metadata/training_monitor.json'))
        self.c = seal(self.c); dump(self.root / 'contract.json', self.c)

    def tearDown(self):
        self.temp.cleanup()

    def refresh(self):
        dump(self.root / 'metadata/parent_terminal_snapshot.json', self.snapshot)
        self.c['parentterminal_evidence_sha256'] = eval.sha(self.root / 'metadata/parent_terminal_snapshot.json')
        self.monitor['groups']['parent_5090']['snapshot_raw_sha256'] = self.c['parentterminal_evidence_sha256']
        dump(self.root / 'metadata/training_monitor.json', self.monitor)
        self.c['training_monitor_sha256'] = eval.sha(self.root / 'metadata/training_monitor.json')
        self.c = seal(self.c); dump(self.root / 'contract.json', self.c)

    def test_exact8_static_contract_with_old36_remains_unmodified(self):
        old_sha = eval.sha(OLD_SCRIPT)
        self.assertEqual(len(eval.verify_contract(self.root)['jobs']), 8)
        r = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root), '--mode', 'check-contract'],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)['job_count'], 8)
        self.assertFalse(json.loads(r.stdout)['real_data_loaded'])
        self.assertEqual(eval.sha(OLD_SCRIPT), old_sha)

    def test_worker_and_scientific_function_asts_equal_original36(self):
        a, b = ast.parse(OLD_SCRIPT.read_text()), ast.parse(SCRIPT.read_text())
        for name in ('worker','validate_payload','validate_frame','population_identity','qualify',
                     'compare_validation_metrics','strict_selected_epoch','verify_training_metadata','qualify_synthetic'):
            left = next(n for n in a.body if isinstance(n,ast.FunctionDef) and n.name==name)
            right = next(n for n in b.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertEqual(ast.dump(left,include_attributes=False),ast.dump(right,include_attributes=False),name)

    def test_foreign_seed_dataset_cohort_or_host_rejected(self):
        for key,value in (('schema','titantpp_components_frozen_evaluation_v1'),('cohort','components36_source131'),
                           ('expected_seeds',[42,52,62]),('evaluation_scope','test_final_comparison')):
            dump(self.root/'contract.json',seal({**self.c,key:value}))
            with self.assertRaises(ValueError): eval.verify_contract(self.root)
        c = deepcopy(self.c); c['jobs'][0]['seed']=62
        dump(self.root/'contract.json',seal(c))
        with self.assertRaises(ValueError): eval.verify_contract(self.root)
        c = deepcopy(self.c); c['runtimes']['5080'] = self.expected_runtime
        dump(self.root/'contract.json',seal(c))
        with self.assertRaises(ValueError): eval.verify_contract(self.root)

    def test_partial_approval_root_sha_exact_scope_and_no_training_enforced(self):
        for key,value in (('partial_cohort_explicitly_authorized',False),('new_training',True),
                         ('original36_global_gate_modified',True),('job_ids',[]),('authorization_root','/foreign/root'),
                         ('user_instruction_sha256','a'*64)):
            p={**self.approval,key:value}
            if key=='authorization_root': p['allowed_local_binding_root']='/foreign/local'
            dump(self.root/'metadata/partial_approval.json',p)
            c=deepcopy(self.c); c['partial_approval_sha256']=eval.sha(self.root/'metadata/partial_approval.json')
            dump(self.root/'contract.json',seal(c))
            with self.assertRaises(ValueError): eval.verify_contract(self.root)

    def test_terminal_owner_presence_and_bad_worker_exit_block(self):
        for bad in ('live','query_error','reused_pid'):
            self.snapshot['process_observations']['100']['state']=bad; self.refresh()
            with self.assertRaises(ValueError): eval.verify_contract(self.root)
        self.snapshot['process_observations']['100']['state']='absent'
        name='worker_exits/'+self.c['jobs'][0]['id']+'.json'
        value=json.loads(base64.b64decode(self.snapshot['files'][name]['base64']))
        value['returncode']=1; raw=json.dumps(value).encode()
        self.snapshot['files'][name]={'base64':base64.b64encode(raw).decode(),'raw_sha256':hashlib.sha256(raw).hexdigest()}
        self.refresh()
        with self.assertRaises(ValueError): eval.verify_contract(self.root)

    def test_permit_missing_wrong_stage_and_local_only_execution_block(self):
        job=self.c['jobs'][0]
        with patch.object(eval,'verify_source',side_effect=AssertionError('scientific data touched')):
            with self.assertRaisesRegex(ValueError,'permit required'):
                eval.worker(self.root,job['id'],'test','cuda')
        p={'schema':'titantpp_components_inter8_evaluation_permit_v1','authorized':True,'new_training':False,
            'frozen_selection':True,'stage':'validation','contract_sha256':self.c['contract_sha256'],
            'evaluator_sha256':eval.sha(SCRIPT),'host':'5090','job_ids':[job['id']],
            'not_before_unix':100.,'deadline_unix':200.}
        dump(self.root/'permit.json',p)
        eval.verify_permit(self.root,self.c,job,'validation','permit.json',eval.sha(self.root/'permit.json'),now=150.)
        with self.assertRaises(ValueError):
            eval.verify_permit(self.root,self.c,job,'test','permit.json',eval.sha(self.root/'permit.json'),now=150.)

    def test_gate_requires_schema_exact8_and_each_actual_exit0(self):
        bindings,exits={},{}
        for job in self.c['jobs']:
            path=self.root/'results'/job['id']
            receipt={'schema':eval.KIND,'cohort':eval.COHORT,'status':'complete','split':'validation',
                'contract_sha256':self.c['contract_sha256'],
                **{k:job[k] for k in ('id','study','dataset','arm','seed','selected_epoch','initial_state_sha256')},
                'checkpoint_sha256':job['checkpoint']['sha256'],'state_sha256':job['checkpoint_state_sha256'],
                'source_closure':eval.SOURCE_CLOSURE,'retrained':False,'reselected':False,'raw_predictions_written':False,
                'full_population':True,'validation_match':'PASS','native_runtime_checked':True,
                'qualification':dict.fromkeys(eval.QUALIFICATION_KEYS,True),'metrics':job['validation_reference'],
                **{k:self.c['datasets'][job['dataset']]['validation_population'][k]
                   for k in ('target_identity_sha256','truth_sha256','time_sha256')},
                'runtime':self.runtime,'runtime_after':self.runtime,'completed_unix':100.}
            dump(path/'validation_replay.json',receipt)
            dump(path/'validation_worker_exit.json',{'job_id':job['id'],'contract_sha256':self.c['contract_sha256'],
                'exit_code':0,'actual_child_exit_observed':True,'completed_unix':100.})
            bindings[job['id']]=eval.sha(path/'validation_replay.json');exits[job['id']]=eval.sha(path/'validation_worker_exit.json')
        gate={'schema':'titantpp_components_inter8_validation_gate_v1','cohort':eval.COHORT,'status':'PASS',
            'contract_sha256':self.c['contract_sha256'],'job_count':8,'validation_receipts':bindings,
            'validation_worker_exits':exits,'created_unix':150.}
        dump(self.root/'validation_gate.json',gate)
        self.assertEqual(eval.verify_global_validation_gate(self.root,self.c)['job_count'],8)
        for change in ({'job_count':36},{'schema':'whole36gate'},{'validation_worker_exits':{}}):
            dump(self.root/'validation_gate.json',{**gate,**change})
            with self.assertRaises(ValueError): eval.verify_global_validation_gate(self.root,self.c)

    def test_selector_metric_tolerance_and_population_contract_unchanged(self):
        self.assertEqual(eval.strict_selected_epoch([{'epoch':1,'val_qty_rmse':3.},{'epoch':2,'val_qty_rmse':3.}]),(1,3.))
        ref=self.c['jobs'][0]['validation_reference']
        eval.compare_validation_metrics(ref,ref)
        for group in (None,'tail'):
            for key in eval.METRICS:
                bad=deepcopy(ref);cell=bad if group is None else bad[group];cell[key]+=.01
                with self.assertRaises(ValueError): eval.compare_validation_metrics(bad,ref)
        self.assertEqual(eval.TAILS,{'intermittent_frozen_5000':187.})

    def test_source131_synthetic_qualification_and_physical_budgets(self):
        for relative in self.c['sources']['components']['files']:
            path=self.root/'source'/relative;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(PARENT/'source'/relative,path)
        (self.root/'source/sample_data').mkdir()
        r=subprocess.run([sys.executable,str(SCRIPT),'--root',str(self.root),'--mode','qualify-synthetic'],
            capture_output=True,text=True,timeout=120,env={**__import__('os').environ,'PYTHONDONTWRITEBYTECODE':'1'})
        self.assertEqual(r.returncode,0,r.stderr)
        value=json.loads(r.stdout)
        self.assertFalse(value['real_data_loaded']);self.assertFalse(value['held_out_test_evaluated'])
        self.assertEqual(value['optimizer_updates'],0)
        counts=value['dataset_parameter_counts']['intermittent_frozen_5000']
        active=(89859,90435,114591,115167)
        for arm,n in zip(eval.ARMS,active):
            self.assertEqual(counts[arm]['actual_registered_parameters'],115167)
            self.assertEqual(counts[arm]['actual_active_parameters'],n)
            self.assertTrue(all(value['arms'][arm]['qualification'].values()))


if __name__=='__main__':
    unittest.main()
