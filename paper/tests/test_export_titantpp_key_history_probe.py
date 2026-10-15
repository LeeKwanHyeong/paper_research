"""Synthetic-only current-candidate row diagnostics; no remote or scientific data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import export_titantpp_key_history_probe as w


def metric_reference():
    return {'count': 6690, 'qty_rmse': 4., 'qty_mae': 2., 'time_nll': 1.,
            'tail': {'count': 50, 'qty_rmse': 8., 'qty_mae': 6., 'time_nll': 2.}}


def job(seed):
    ident = f'raf_spare_parts__{seed}__{w.ARM}'
    return {'id': ident, 'dataset': 'raf_spare_parts', 'arm': w.ARM, 'seed': seed,
            'cohort': w.COHORT, 'evaluation_host': '5090', 'selected_epoch': w.EPOCHS[seed],
            'checkpoint': {'relative_path': f'inputs/{seed}.pt', 'sha256': 'd' * 64},
            'checkpoint_state_sha256': 'a' * 64,
            'checkpoint_metadata_path': f'inputs/{seed}.metadata.json', 'checkpoint_metadata_sha256': 'b' * 64,
            'selection_history_path': f'inputs/{seed}.history.json', 'selection_history_sha256': 'c' * 64,
            'validation_reference': metric_reference()}


class RowsBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp', prefix='key-history-probe-synthetic-')
        self.root = Path(self.tmp.name)
        self.jobs = [job(s) for s in w.SEEDS]
        self.population = {'target_count': 6690, 'target_identity_sha256': 'p' * 64}
        self.contract = {'contract_sha256': 'c' * 64, 'jobs': self.jobs, 'deadline_unix': 200.,
                         'datasets': {'raf_spare_parts': {'definition': {'inherited_data_identity': {'populations': {'validation': self.population,'train':{'target_count':25779,'target_identity_sha256':'t'*64}}}}}}}

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
        return w.sha(path)

    def fixture_contract(self):
        inputs, operations, source = {}, {}, {}
        for i in range(145):
            rel = f'file_{i:03d}.py'; path = self.root / 'source' / rel
            path.parent.mkdir(exist_ok=True); path.write_text('# fabricated scientific file\n')
            source[rel] = w.sha(path)
        for rel in ('operation/evaluate.py', 'operation/helper.py', 'operation/row_utils.py', 'operation/lineage.json'):
            path = self.root / rel; path.parent.mkdir(exist_ok=True)
            path.write_bytes(Path(w.__file__).read_bytes() if rel.endswith('evaluate.py') else b'{}')
            operations[rel] = w.sha(path)
        data = self.root / 'inputs/raf_train_validation.parquet'; data.parent.mkdir(exist_ok=True)
        data.write_bytes(b'fabricated input, never opened as research parquet')
        inputs[str(data.relative_to(self.root))] = w.sha(data)
        jobs = deepcopy(self.jobs)
        for j in jobs:
            for field, hfield in [('checkpoint_metadata_path', 'checkpoint_metadata_sha256'), ('selection_history_path', 'selection_history_sha256')]:
                inputs[j[field]] = self.save(j[field], {'fabricated': True}); j[hfield] = inputs[j[field]]
            path = self.root / j['checkpoint']['relative_path']; path.write_bytes(b'fabricated checkpoint, never deserialized')
            j['checkpoint']['sha256'] = inputs[j['checkpoint']['relative_path']] = w.sha(path)
        spec = {'fabricated': True}; inputs['spec.json'] = self.save('spec.json', spec)
        approval = {'authorized': True, 'new_training': False, 'frozen_selection': True, 'allowed_splits': ['train', 'validation'], 'historical_CNN_inference':False, 'held_out_test_evaluated':False}
        inputs['approval.json'] = self.save('approval.json', approval)
        definition = {'data_sha256': w.sha(data), 'inherited_data_identity': {'populations': {s: {'target_count': n} for s, n in w.COUNTS.items()}},
                      'loader': {'batch_size': 128, 'drop_last': False, 'lookback_weeks': 84, 'max_seq_len': 84,
                                 'num_workers': 0, 'train_shuffle': True, 'validation_shuffle': False}}
        c = {'schema': w.SCHEMA, 'allowed_splits': ['train', 'validation'], 'new_training': False, 'frozen_selection': True,
             'max_parallel_per_host': 3, 'host': '5090', 'remote_root': w.REMOTE_ROOT, 'deadline_unix': 200., 'jobs': jobs,
             'source': {'root': 'source', 'files': source, 'closure': w.canonical(source), 'revision': 'synthetic'},
             'datasets': {'raf_spare_parts': {'path': str(data.relative_to(self.root)), 'sha256': w.sha(data), 'definition': definition}},
             'runtimes': {'5090': {'host': '5090', 'gpu_uuid': w.GPU_UUID, 'python_version': '3.12.13', 'torch': '2.11.0+cu130', 'cudnn': 92000, 'torch_num_threads': 4}},
             'operation_files': operations, 'input_files': inputs, 'spec_path': 'spec.json', 'spec_sha256': inputs['spec.json'],
             'acceptance_path': 'acceptance.json', 'acceptance_sha256': self.save('acceptance.json', {'fabricated':True}),
             'intervention':w.INTERVENTION,'history_tap_multipliers':[1,0],'tap_masks':deepcopy(w.TAP_MASKS),
             'stages':list(w.STAGES),'conditions':['on','off'],'validation_tolerance':dict(w.VALIDATION_TOLERANCE),
             'approval_path': 'approval.json', 'approval_sha256': inputs['approval.json']}
        inputs['acceptance.json']=c['acceptance_sha256']
        return c

    def seal(self, c):
        c['contract_sha256'] = w.canonical({k: v for k, v in c.items() if k != 'contract_sha256'})
        return self.save('contract.json', c)

    def verify_fixture(self, c):
        raw = self.seal(c)
        with patch.object(w, 'SOURCE_CLOSURE', c['source']['closure']), patch.object(w, 'DATA_SHA', c['datasets']['raf_spare_parts']['sha256']), \
             patch.object(w, 'SPEC_SHA', c['spec_sha256']), patch.object(w,'ACCEPTANCE_SHA',c['acceptance_sha256']), patch.object(w.time, 'time', return_value=100.):
            return w.verify_contract(self.root, raw)

    def gate(self, stage):
        g = {'schema': f'titantpp_key_history_probe_{stage}_gate_v1', 'contract_sha256': self.contract['contract_sha256'],
             'all_native_exit0': True, 'job_ids': [j['id'] for j in self.jobs], 'receipts': {}, 'worker_exits': {}}
        for j in self.jobs:
            ident = j['id']; receipt = {'status': 'complete', 'job_id': ident, 'stage':stage, 'split':w.stage_identity(stage)[1], 'condition':w.stage_identity(stage)[0], 'branch_function_restored':True, 'contract_sha256': self.contract['contract_sha256'],
                    'parameters_unchanged': True, 'state_including_buffers_unchanged': True,
                    'state_sha256_before': j['checkpoint_state_sha256'], 'state_sha256_after': j['checkpoint_state_sha256'],
                    'native_runtime_checked': True, 'no_training': True, 'held_out_test_evaluated': False}
            snap={'state_sha256':j['checkpoint_state_sha256'],'all_named_buffers_sha256':'b'*64,'context_cache_sha256':'c'*64}
            receipt.update(**w.intervention_fields(w.stage_identity(stage)[0]))
            receipt.update(predictive_state_before=deepcopy(snap),predictive_state_after=deepcopy(snap))
            if stage == 'qualification':
                receipt.update(qualification_passed=True, real_data_loaded=False)
            else:
                split=w.stage_identity(stage)[1];count=w.COUNTS[split]
                part=f'rows/{ident}/{stage}/part-00000.parquet';path=self.root/part;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fabricated shard')
                manifest={'prediction_rows':count,'job_id':ident,'contract_sha256':self.contract['contract_sha256'],'split':split,'stage':stage,'condition':'on',
                          'input_sha256':w.DATA_SHA,'source_closure':w.SOURCE_CLOSURE,'checkpoint_sha256':j['checkpoint']['sha256'],
                          'parts':[{'path':'part-00000.parquet','rows':count,'sha256':w.sha(path)}]}
                manifest_path=f'rows/{ident}/{stage}/manifest.json';manifest_sha=self.save(manifest_path,manifest)
                metrics=metric_reference();metrics['count']=count
                receipt.update(rowwise_replay=None,stored_prediction_rows_available=False,original_population_match=True,
                    validation_match='PASS' if stage=='on_validation' else None,
                    validation_replay_check={'six_metrics':stage=='on_validation','relative_tolerance':1e-5,'absolute_tolerance':1e-5},
                    selected_epoch=j['selected_epoch'],checkpoint_sha256=j['checkpoint']['sha256'],source_closure=w.SOURCE_CLOSURE,data_sha256=w.DATA_SHA,
                    metrics=metrics,prediction_rows=count,population=self.contract['datasets']['raf_spare_parts']['definition']['inherited_data_identity']['populations'][split],
                    manifest_path=manifest_path,manifest_sha256=manifest_sha)
            rel = f'results/{ident}/{stage}.json'; g['receipts'][ident] = {'path': rel, 'sha256': self.save(rel, receipt)}
            actual = {'job_id': ident, 'stage':stage, 'split':w.stage_identity(stage)[1], 'condition':w.stage_identity(stage)[0], 'branch_function_restored':True, 'contract_sha256': self.contract['contract_sha256'], 'actual_child_exit_observed': True, 'exit_code': 0}
            rel = f'exits/{stage}_{ident}.json'; g['worker_exits'][ident] = {'path': rel, 'sha256': self.save(rel, actual)}
        return g

    def permit(self, stage):
        return {'schema': w.PERMIT_SCHEMA, 'authorized': True, 'stage': stage, 'host': '5090', 'contract_sha256': self.contract['contract_sha256'],
                'contract_raw_sha256': 'r' * 64, 'evaluator_sha256': w.sha(w.__file__), 'new_training': False,
                'checkpoint_reselection': False, 'automatic_retry': False, 'job_ids': [j['id'] for j in self.jobs],
                'not_before_unix': 90., 'deadline_unix': 150.}

    def test_valid_contract_hashes_all_pins_without_loading_models_or_data(self):
        c = self.fixture_contract()
        with patch.object(w, 'prepare_model') as model, patch.object(w, 'load_observed_frame') as data:
            self.assertEqual(self.verify_fixture(c), c)
        model.assert_not_called(); data.assert_not_called()

    def test_contract_rejects_foreign_arm_seed_epoch_source_loader_and_runtime(self):
        base = self.fixture_contract()
        changes = [lambda c: c['jobs'][0].update(arm='titantpp_cnn_gru54_cnn_only'),
                   lambda c: c['jobs'][0].update(seed=7), lambda c: c['jobs'][0].update(selected_epoch=10),
                   lambda c: c.update(allowed_splits=['train', 'validation', 'test']), lambda c: c.update(new_training=True),
                   lambda c: c['source']['files'].pop(next(iter(c['source']['files']))),
                   lambda c: c['datasets']['raf_spare_parts']['definition']['loader'].update(max_seq_len=83),
                   lambda c: c['runtimes']['5090'].update(cudnn=91900), lambda c: c['runtimes']['5090'].update(gpu_uuid='foreign')]
        for change in changes:
            c = deepcopy(base); change(c)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                self.verify_fixture(c)

    def test_modified_frozen_bytes_rejected(self):
        c = self.fixture_contract(); self.verify_fixture(c)
        (self.root / 'source/file_000.py').write_text('# tampered')
        with self.assertRaisesRegex(ValueError, 'scientific source changed'):
            self.verify_fixture(c)

    def test_contract_canonical_and_raw_sha_both_required(self):
        c = self.fixture_contract(); raw = self.seal(c)
        with self.assertRaisesRegex(ValueError, 'raw SHA'):
            w.verify_contract(self.root, '0' * 64)
        c['new_training'] = True; changed = self.save('contract.json', c)
        with self.assertRaisesRegex(ValueError, 'canonical'):
            w.verify_contract(self.root, changed)
        self.assertNotEqual(raw, changed)

    def test_path_escape_and_absolute_path_rejected(self):
        for path in ('../outside', '/tmp/outside'):
            with self.subTest(path=path), self.assertRaises(ValueError): w.checked(self.root, path)

    def test_disallowed_stage_or_foreign_root_fails_before_io(self):
        for stage in ('test', 'on_test', '0.5', None):
            with self.subTest(stage=stage), patch.object(w, 'verify_contract') as c, self.assertRaises(ValueError):
                w.worker(self.root, 'raw', self.jobs[0]['id'], stage)
            c.assert_not_called()
        with patch.object(w, 'verify_contract') as c, self.assertRaisesRegex(ValueError, 'native diagnostic root'):
            w.worker(self.root, 'raw', self.jobs[0]['id'], 'off_train')
        c.assert_not_called()

    def test_permit_expiry_budget_extension_foreign_stage_and_missing_candidate_rejected(self):
        base = self.permit('qualification')
        for changes in ({'deadline_unix': 99.}, {'deadline_unix': 201.}, {'not_before_unix': 101.}, {'host': '5080'},
                        {'stage': 'test'}, {'checkpoint_reselection': True}, {'automatic_retry': True}, {'job_ids': [self.jobs[0]['id']]}):
            p = {**base, **changes}; digest = self.save('permit.json', p)
            with self.subTest(changes=changes), patch.object(w.time, 'time', return_value=100.), self.assertRaises(ValueError):
                w.verify_permit(self.root, self.contract, 'r' * 64, self.jobs[0], 'qualification', 'permit.json', digest)
        digest = self.save('permit.json', base)
        with patch.object(w.time, 'time', return_value=100.):
            self.assertEqual(w.verify_permit(self.root, self.contract, 'r' * 64, self.jobs[0], 'qualification', 'permit.json', digest), base)

    def test_qualification_gate_requires_all_three_native_exits(self):
        gate = self.gate('qualification'); digest = self.save('qualification_gate.json', gate)
        self.assertEqual(w.verify_gate(self.root, self.contract, 'qualification', digest), gate)
        for mutate in ('exit1', 'false_observed', 'missing', 'foreign_state', 'buffers_changed'):
            changed = deepcopy(gate); ident = self.jobs[0]['id']
            if mutate == 'missing': del changed['worker_exits'][ident]
            elif mutate in ('exit1', 'false_observed'):
                rel = changed['worker_exits'][ident]['path']; actual = w.read(self.root / rel)
                actual['exit_code' if mutate == 'exit1' else 'actual_child_exit_observed'] = 1 if mutate == 'exit1' else False
                changed['worker_exits'][ident]['sha256'] = self.save(rel, actual)
            else:
                rel = changed['receipts'][ident]['path']; receipt = w.read(self.root / rel)
                receipt['state_sha256_after' if mutate == 'foreign_state' else 'state_including_buffers_unchanged'] = 'z' * 64 if mutate == 'foreign_state' else False
                changed['receipts'][ident]['sha256'] = self.save(rel, receipt)
            digest = self.save('qualification_gate.json', changed)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): w.verify_gate(self.root, self.contract, 'qualification', digest)

    def test_validation_gate_checks_six_metrics_and_shard_bytes(self):
        gate = self.gate('on_validation'); digest = self.save('on_validation_gate.json', gate)
        self.assertEqual(w.verify_gate(self.root, self.contract, 'on_validation', digest), gate)
        rel = gate['receipts'][self.jobs[0]['id']]['path']; receipt = w.read(self.root / rel)
        receipt['metrics']['tail']['qty_mae'] += .1
        changed = deepcopy(gate); changed['receipts'][self.jobs[0]['id']]['sha256'] = self.save(rel, receipt)
        digest = self.save('on_validation_gate.json', changed)
        with self.assertRaisesRegex(ValueError, 'six-metric'): w.verify_gate(self.root, self.contract, 'on_validation', digest)
        receipt['metrics'] = metric_reference(); gate['receipts'][self.jobs[0]['id']]['sha256'] = self.save(rel, receipt)
        digest = self.save('on_validation_gate.json', gate)
        (self.root / f'rows/{self.jobs[0]["id"]}/on_validation/part-00000.parquet').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'shard changed'): w.verify_gate(self.root, self.contract, 'on_validation', digest)

    def test_stages_require_complete_prior_gates(self):
        cases=[('on_validation',[('qualification','qualification_gate_sha256')]),
               ('on_train',[('on_validation','on_validation_gate_sha256')]),
               ('off_validation',[('on_validation','on_validation_gate_sha256'),('on_train','on_train_gate_sha256')]),
               ('off_train',[('on_validation','on_validation_gate_sha256'),('on_train','on_train_gate_sha256')])]
        for stage,priors in cases:
            p=self.permit(stage)
            for prior,key in priors:p[key]=prior+'digest'
            digest=self.save('permit.json',p)
            with self.subTest(stage=stage),patch.object(w.time,'time',return_value=100.),patch.object(w,'verify_gate') as gate:
                w.verify_permit(self.root,self.contract,'r'*64,self.jobs[0],stage,'permit.json',digest)
                self.assertEqual(gate.call_count,len(priors))
                for prior,key in priors:gate.assert_any_call(self.root,self.contract,prior,p[key])

    def test_on_train_gate_and_fresh_reference_match_current_selection(self):
        gate=self.gate('on_train');digest=self.save('on_train_gate.json',gate)
        self.assertEqual(w.verify_gate(self.root,self.contract,'on_train',digest),gate)
        r,m,pin=w.verify_reference(self.root,self.contract,self.jobs[0],'train',digest)
        self.assertEqual(r['prediction_rows'],25779);self.assertEqual(m['stage'],'on_train');self.assertEqual(pin,gate['receipts'][self.jobs[0]['id']])
        rel=gate['receipts'][self.jobs[0]['id']]['path'];r['validation_match']='PASS'
        gate['receipts'][self.jobs[0]['id']]['sha256']=self.save(rel,r)
        digest=self.save('on_train_gate.json',gate)
        with self.assertRaisesRegex(ValueError,'Train must not'):w.verify_gate(self.root,self.contract,'on_train',digest)

    def test_gate_predictive_buffer_context_snapshots_required_and_bound(self):
        for change in ('missing','extra','bad_hex','different_after','wrong_state'):
            gate=self.gate('on_validation');ident=self.jobs[0]['id'];rel=gate['receipts'][ident]['path'];r=w.read(self.root/rel)
            if change=='missing':del r['predictive_state_before']['context_cache_sha256']
            elif change=='extra':r['predictive_state_before']['extra']='d'*64
            elif change=='bad_hex':r['predictive_state_before']['all_named_buffers_sha256']='Z'*64;r['predictive_state_after']=deepcopy(r['predictive_state_before'])
            elif change=='different_after':r['predictive_state_after']['context_cache_sha256']='d'*64
            else:
                r['predictive_state_before']['state_sha256']='d'*64;r['predictive_state_after']=deepcopy(r['predictive_state_before'])
            gate['receipts'][ident]['sha256']=self.save(rel,r);digest=self.save('on_validation_gate.json',gate)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'Predictive'):w.verify_gate(self.root,self.contract,'on_validation',digest)

    def test_historical_row_replay_claim_or_missing_population_proof_rejected(self):
        for field,value in [('rowwise_replay',{'status':'PASS'}),('stored_prediction_rows_available',True),('original_population_match',False)]:
            gate=self.gate('on_validation');ident=self.jobs[0]['id'];rel=gate['receipts'][ident]['path'];r=w.read(self.root/rel);r[field]=value
            gate['receipts'][ident]['sha256']=self.save(rel,r);digest=self.save('on_validation_gate.json',gate)
            with self.subTest(field=field),self.assertRaises(ValueError):w.verify_gate(self.root,self.contract,'on_validation',digest)

    def test_six_metrics_finite_counts_and_empty_population(self):
        ref = metric_reference(); w.validate_six_metrics(deepcopy(ref), ref)
        for group in (None, 'tail'):
            for key in ('qty_rmse', 'qty_mae', 'time_nll', 'count'):
                a = deepcopy(ref); cell = a if group is None else a[group]; cell[key] += 1
                with self.subTest(group=group, key=key), self.assertRaises(ValueError): w.validate_six_metrics(a, ref)
        a = deepcopy(ref); a['qty_mae'] = float('nan')
        with self.assertRaises(ValueError): w.validate_six_metrics(a, ref)
        empty = {'count': 0, 'qty_rmse': None, 'qty_mae': None, 'time_nll': None}
        ref['tail'] = empty; w.validate_six_metrics(deepcopy(ref), ref)

    def mock_context(self):
        j = self.jobs[0]; self.contract['source'] = {'closure': 's' * 64, 'revision': 'synthetic'}
        return {'state': j['checkpoint_state_sha256'], 'counts': {}, 'runtime': {}, 'predictive_state': {}}

    def test_qualification_never_materializes_real_data(self):
        ctx = self.mock_context()
        with patch.object(w, 'REMOTE_ROOT', str(self.root)), patch.object(w, 'verify_contract', return_value=self.contract), \
             patch.object(w, 'verify_permit', return_value={}), patch.object(w, 'prepare_model', return_value=ctx), \
             patch.object(w, 'synthetic_checks', return_value={}), patch.object(w, 'final_integrity', return_value=([], {})), \
             patch.object(w, 'load_observed_frame') as data:
            result = w.worker(self.root, 'raw', self.jobs[0]['id'], 'qualification')
        data.assert_not_called(); self.assertFalse(result['real_data_loaded']); self.assertTrue(result['qualification_passed'])
        self.assertTrue(result['state_including_buffers_unchanged']); self.assertFalse(result['held_out_test_evaluated'])

    def test_original_failure_and_claim_preserved_no_second_attempt(self):
        with patch.object(w, 'REMOTE_ROOT', str(self.root)), patch.object(w, 'verify_contract', return_value=self.contract), \
             patch.object(w, 'verify_permit', return_value={}), patch.object(w, 'prepare_model', side_effect=ValueError('original synthetic failure')) as model:
            with self.assertRaisesRegex(ValueError, 'original synthetic failure'):
                w.worker(self.root, 'raw', self.jobs[0]['id'], 'qualification')
            p = self.root / f'results/{self.jobs[0]["id"]}/qualification_metrics_failure.json'; before = p.read_bytes()
            self.assertIn('original synthetic failure', w.read(p)['traceback'])
            with self.assertRaisesRegex(ValueError, 'Existing attempt'):
                w.worker(self.root, 'raw', self.jobs[0]['id'], 'qualification')
            self.assertEqual(p.read_bytes(), before); self.assertEqual(model.call_count, 1)

    def test_intervention_contract_masks_scope_and_stage_order_are_fixed(self):
        base=self.fixture_contract()
        changes=[{'intervention':'all_key_residual'}, {'history_tap_multipliers':[1,.5]},
                 {'history_tap_multipliers':[True,False]}, {'tap_masks':{'on':[1,1,1],'off':[0,0,0]}},
                 {'tap_masks':{'on':[1,1,1],'off':[True,False,False]}},
                 {'stages':['qualification','on_validation','on_train','off_validation','off_train']}]
        for change in changes:
            c=deepcopy(base);c.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):self.verify_fixture(c)

    def test_gate_rejects_rehashed_missing_or_wrong_current_history_proof(self):
        for change in ({'intervention':'whole_key'}, {'current_tap_preserved':False}, {'history_taps_disabled':True},
                       {'original_full_function_checks_preserved':False}, {'current_tap_preserved':1}):
            gate=self.gate('on_validation');ident=self.jobs[0]['id'];rel=gate['receipts'][ident]['path'];r=w.read(self.root/rel)
            r.update(change);gate['receipts'][ident]['sha256']=self.save(rel,r);digest=self.save('on_validation_gate.json',gate)
            with self.subTest(change=change),self.assertRaisesRegex(ValueError,'tap intervention proof'):
                w.verify_gate(self.root,self.contract,'on_validation',digest)



class RowNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/key-history-probe-synthetic-mpl')
        os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/key-history-probe-synthetic-cache')
        import numpy as np
        import torch
        import polars as pl
        from paper.scripts import evaluate_titantpp_frozen_rows as utils
        from paper.scripts import evaluate_titantpp_qk_location_frozen as helper
        from paper.scripts import run_titantpp_qk_location_cnn as engine
        from paper.scripts.count_aware_tpp_backbone import core
        from simple_lab_test.search.common import runner
        cls.np, cls.torch, cls.pl, cls.utils, cls.helper = np, torch, pl, utils, helper
        cls.engine, cls.core, cls.runner = engine, core, runner
        torch.set_num_threads(1)

    def test_history_statistics_exclude_target_and_padding_before_transform(self):
        np, torch = self.np, self.torch
        q = torch.tensor([[float('nan'), 1., 3., 8., 999.], [0., 0., 0., 7., 111.]])
        d = torch.tensor([[float('nan'), 2., 3., 4., 99.], [0., 0., 0., 2., 91.]])
        mask = torch.tensor([[False, True, True, True, True], [False, False, False, True, True]])
        a, hashes = w.observed_features(q, mask, d, np)
        q[:, -1] = float('nan'); d[:, -1] = float('nan'); b, changed = w.observed_features(q, mask, d, np)
        self.assertEqual(hashes, changed)
        for key in a: np.testing.assert_equal(a[key], b[key])
        self.assertEqual(a['history_length'].tolist(), [3, 1]); self.assertEqual(a['last_quantity'].tolist(), [8., 7.])
        np.testing.assert_allclose(a['mean_quantity'], [4., 7.])
        self.assertAlmostEqual(a['history_log_std'][0], np.std(np.log1p([1., 3., 8.])))
        self.assertEqual(a['previous_observation_available'].tolist(), [True, False]); self.assertEqual(a['last_observed_log_change'][1], 0.)

    def test_history_fingerprint_detects_order_and_gap_even_same_mean(self):
        torch, np = self.torch, self.np
        q = torch.tensor([[1., 3., 2., 9.]]); d = torch.tensor([[1., 2., 3., 8.]]); m = torch.ones_like(q, dtype=torch.bool)
        _, base = w.observed_features(q, m, d, np)
        q2 = q.clone(); q2[0, :2] = q2[0, [1, 0]]; _, reordered = w.observed_features(q2, m, d, np)
        d2 = d.clone(); d2[0, 0] = 4.; _, gap = w.observed_features(q, m, d2, np)
        self.assertNotEqual(base, reordered); self.assertNotEqual(base, gap)

    def test_invalid_observed_context_rejected(self):
        torch, np = self.torch, self.np
        for q, mask in [(torch.tensor([[1., 2.]]), torch.tensor([[False, True]])),
                        (torch.tensor([[-1., 2.]]), torch.tensor([[True, True]]))]:
            with self.assertRaises(ValueError): w.observed_features(q, mask, torch.ones_like(q), np)

    def test_actual_candidate_synthetic_checks_and_all_state_unchanged(self):
        torch = self.torch; torch.manual_seed(17)
        data = self.engine.fabricated_data(); data['loader']['max_seq_len'] = 84
        with patch.object(self.pl, 'read_parquet', side_effect=AssertionError('No real data')), patch.object(self.pl, 'scan_parquet', side_effect=AssertionError('No real data')):
            model, _ = self.engine.build_model(data, w.ARM)
            self.assertEqual(sum(p.numel() for p in model.parameters()), 103775)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 79043)
            self.assertFalse(any('causal_v_kernel' in k for k in model.state_dict()))
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if 'causal_k_kernel' in name: parameter.normal_(0., .03)
            model.eval().requires_grad_(False)
            ctx = {'torch': torch, 'model': model, 'core': self.core, 'runner': self.runner, 'helper': self.helper,
                   'definition': data, 'state': self.runner.canonical_state_dict_sha256(model.state_dict()), 'predictive_state': w.predictive_state(model, self.runner)}
            result = w.synthetic_checks(ctx)
            self.assertTrue(result['branch_function_restored']); self.assertEqual(w.predictive_state(model, self.runner), ctx['predictive_state'])

    def test_parameter_buffer_nonpersistent_and_context_changes_all_rejected(self):
        torch = self.torch
        for mutation in ('parameter', 'persistent', 'nonpersistent', 'context'):
            model = torch.nn.Linear(2, 2); model.register_buffer('persistent', torch.zeros(1)); model.register_buffer('nonpersistent', torch.zeros(1), persistent=False)
            model._ctx_mem = None; model.eval().requires_grad_(False)
            ctx = {'model': model, 'runner': self.runner, 'state': self.runner.canonical_state_dict_sha256(model.state_dict()), 'predictive_state': w.predictive_state(model, self.runner)}
            if mutation == 'parameter': model.weight.add_(1.)
            elif mutation == 'context': model._ctx_mem = torch.ones(1)
            else: getattr(model, mutation).add_(1.)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'buffers, or contextual memory'): w.require_fixed_state(ctx)

    def test_six_metrics_hand_calculation_and_tail_boundary(self):
        metric = self.utils.RawMetrics([2., 4., 6., 8.], self.np)
        metric.add([1., 3., 5., 7., 10.], [2., 1., 8., 8., 6.], [.1, .2, .3, .4, .5]); actual = metric.finish()
        self.assertAlmostEqual(actual['qty_rmse'], (31. / 5.) ** .5); self.assertAlmostEqual(actual['qty_mae'], 11. / 5.)
        self.assertAlmostEqual(actual['time_nll'], .3); self.assertAlmostEqual(actual['qty_bias'], -.2)
        self.assertEqual(actual['tail']['count'], 1); self.assertEqual(actual['tail']['qty_rmse'], 4.); self.assertEqual(actual['tail']['qty_mae'], 4.); self.assertEqual(actual['tail']['time_nll'], .5)
        boundary = self.utils.RawMetrics([2., 4., 6., 8.], self.np); boundary.add([8.], [8.], [1.]); self.assertEqual(boundary.finish()['tail']['count'], 0)
        with self.assertRaises(ValueError): metric.add([1.], [float('nan')], [1.])

    def test_shards_exclusive_count_sha_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            path = Path(temp) / 'rows'; writer = self.utils.PredictionWriter(path, self.pl, part_rows=2)
            writer.add([{'seq': n, 'prediction': float(n)} for n in range(5)]); manifest = writer.finish()
            self.assertEqual([p['rows'] for p in manifest['parts']], [2, 2, 1]); self.assertEqual(manifest['prediction_rows'], 5)
            for part in manifest['parts']: self.assertEqual(w.sha(path / part['path']), part['sha256'])
            with self.assertRaises(FileExistsError): self.utils.PredictionWriter(path, self.pl)
            (path / manifest['parts'][0]['path']).write_bytes(b'tamper')
            with self.assertRaisesRegex(ValueError, 'bytes changed'): writer.finish()

    def test_synthetic_frame_train_filter_and_mixed_split_rejection(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temp:
            p = Path(temp) / 'fabricated.parquet'
            self.pl.DataFrame({'oper_part_no': ['x', 'x'], 'seq': [1, 2], 'chronological_split': ['train', 'validation']}).write_parquet(p)
            self.assertEqual(w.load_observed_frame(p, 'train', self.pl).height, 1)
            self.assertEqual(w.load_observed_frame(p, 'validation', self.pl).height, 2)
            with self.assertRaises(ValueError): w.load_observed_frame(p, 'test', self.pl)
            self.pl.DataFrame({'oper_part_no': ['x'], 'seq': [1], 'chronological_split': ['test']}).write_parquet(p)
            with self.assertRaisesRegex(ValueError, 'unauthorized split'): w.load_observed_frame(p, 'validation', self.pl)





class BranchInterventionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from models.TPPs import CountAwareTitanQKLocationCNN as branch, CountAwareTitanCNNGRU as base
        cls.torch,cls.branch,cls.base=torch,branch,base

    def inputs(self):
        t=self.torch
        return (t.tensor([[[2.,5.],[4.,3.],[7.,8.]]]),t.tensor([[.3,-.2],[.1,.4],[-.2,.5]]),
                t.tensor([[[0,0,0],[1,0,0],[2,1,0]]],dtype=t.long),
                t.tensor([[[True,False,False],[True,True,False],[True,True,True]]]))

    def test_zero_one_exact_delegation_and_all_inputs_unchanged(self):
        t,b=self.torch,self.branch;args=self.inputs();original=b.causal_depthwise_residual;baseoriginal=self.base.causal_depthwise_residual
        before=[x.clone() for x in args];expected=original(*args)
        current_kernel=args[1].clone();current_kernel[1:]=0
        expected_off=original(args[0],current_kernel,*args[2:])
        for scale in (1,0,1.,0.):
            with w.branch_scope(b,scale,t) as audit:
                actual=b.causal_depthwise_residual(*args)
                self.assertTrue(t.equal(actual,expected if scale else expected_off))
                self.assertIs(self.base.causal_depthwise_residual,baseoriginal)
                self.assertEqual(audit['calls'],1)
            self.assertTrue(audit['restored']);self.assertIs(b.causal_depthwise_residual,original)
        self.assertFalse(t.equal(expected[:,0],t.zeros_like(expected[:,0])))
        for a,x in zip(args,before):self.assertTrue(t.equal(a,x))

    def test_both_modes_preserve_guards_and_reject_nonfinite_original_result(self):
        t,b=self.torch,self.branch;original=b.causal_depthwise_residual;args=self.inputs();bad=[]
        bad.append((args[0].squeeze(0),*args[1:]));bad.append((args[0],args[1][:1],*args[2:]))
        bad.append((*args[:2],args[2].float(),args[3]));bad.append((*args[:3],args[3].float()))
        x=args[0].clone();x[0,1,0]=float('nan');bad.append((x,*args[1:]))
        for value in (float('inf'),float('nan')):
            k=args[1].clone();k[0,0]=value;bad.append((args[0],k,*args[2:]))
        x=args[0].clone();x[:]=1e30;k=args[1].clone();k[:]=1e30;bad.append((x,k,*args[2:]))
        for scale in (0,1):
            for i,badargs in enumerate(bad):
                with self.subTest(scale=scale,case=i),self.assertRaises((ValueError,TypeError)):
                    with w.branch_scope(b,scale,t):b.causal_depthwise_residual(*badargs)
                self.assertIs(b.causal_depthwise_residual,original)

    def test_padding_sanitized_withheld_reset_retains_current_tap(self):
        t,b=self.torch,self.branch;values,k,s,a=self.inputs();values[0,0]=float('nan');a[0,0]=False;a[0,1,1:]=False;a[0,2,2]=False
        for scale in (0,1):
            with w.branch_scope(b,scale,t):actual=b.causal_depthwise_residual(values,k,s,a)
            self.assertTrue(t.isfinite(actual).all());self.assertTrue(t.equal(actual[:,0],t.zeros_like(actual[:,0])))
            self.assertTrue(t.equal(actual[:,1],values[:,1]*k[0]))

    def test_exception_nesting_invalid_scales_and_tampering_restore(self):
        t,b=self.torch,self.branch;original=b.causal_depthwise_residual
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with w.branch_scope(b,0,t):raise RuntimeError('synthetic')
        self.assertIs(b.causal_depthwise_residual,original)
        with w.branch_scope(b,1,t):
            with self.assertRaisesRegex(ValueError,'Nested'):
                with w.branch_scope(b,0,t):pass
        for scale in (.5,-1,2,'on','off','0',True,False,None,float('nan'),float('inf'),t.tensor(0)):
            with self.subTest(scale=str(scale)),self.assertRaises(ValueError):
                with w.branch_scope(b,scale,t):pass
        with self.assertRaisesRegex(ValueError,'identity changed'):
            with w.branch_scope(b,0,t):b.causal_depthwise_residual=lambda *a:None
        self.assertIs(b.causal_depthwise_residual,original)

    def test_foreign_original_not_permitted(self):
        t,b=self.torch,self.branch;original=b.causal_depthwise_residual
        b.causal_depthwise_residual=lambda *a:None
        try:
            with self.assertRaisesRegex(ValueError,'original residual binding'):
                with w.branch_scope(b,0,t):pass
        finally:b.causal_depthwise_residual=original

    def test_off_retains_nonzero_current_tap_not_whole_key_zero(self):
        t,b=self.torch,self.branch;v,k,s,a=self.inputs();original=b.causal_depthwise_residual
        with w.branch_scope(b,0,t) as audit:actual=b.causal_depthwise_residual(v,k,s,a)
        expected=v*k[0]
        self.assertTrue(t.equal(actual,expected));self.assertFalse(t.equal(actual,t.zeros_like(actual)))
        self.assertTrue(t.equal(actual[:,0],original(v,k,s,a)[:,0]))
        self.assertEqual(audit['tap_mask'],[1,0,0]);self.assertTrue(audit['current_tap_preserved'])
        self.assertTrue(audit['history_taps_disabled']);self.assertTrue(audit['original_full_function_checks_preserved'])

    def test_masked_past_kernel_nonfinite_is_not_hidden(self):
        t,b=self.torch,self.branch;args=self.inputs();original=b.causal_depthwise_residual
        for lag in (1,2):
            for value in (float('inf'),float('nan')):
                k=args[1].clone();k[lag,0]=value
                with self.subTest(lag=lag,value=value),self.assertRaisesRegex(ValueError,'Nonfinite original residual'):
                    with w.branch_scope(b,0,t):b.causal_depthwise_residual(args[0],k,*args[2:])
                self.assertIs(b.causal_depthwise_residual,original)

    def test_masked_past_invalid_index_still_rejected(self):
        t,b=self.torch,self.branch;v,k,s,a=self.inputs();s[0,2,2]=1000;original=b.causal_depthwise_residual
        for scale in (0,1):
            with self.subTest(scale=scale),self.assertRaises(RuntimeError):
                with w.branch_scope(b,scale,t):b.causal_depthwise_residual(v,k,s,a)
            self.assertIs(b.causal_depthwise_residual,original)

    def test_one_observation_padding_and_reset_on_off_identical(self):
        t,b=self.torch,self.branch;v,k,_,_=self.inputs()
        valid=t.tensor([[True,False,True]]);write=valid.clone()
        _,sources,available=self.base.observed_context(valid,memory_write_mask=write)
        available[:,:,1:]=False
        outputs=[]
        for scale in (1,0):
            with w.branch_scope(b,scale,t):outputs.append(b.causal_depthwise_residual(v,k,sources,available))
        self.assertTrue(t.equal(*outputs));self.assertTrue(t.equal(outputs[0][:,[0,2]],v[:,[0,2]]*k[0]))

    def test_keyword_arguments_preserve_current_tap_and_original_identity(self):
        t,b=self.torch,self.branch;v,k,s,a=self.inputs();original=b.causal_depthwise_residual
        before=k.clone()
        with w.branch_scope(b,0,t):out=b.causal_depthwise_residual(values=v,kernel=k,sources=s,available=a)
        self.assertTrue(t.equal(out,v*k[0]));self.assertTrue(t.equal(k,before));self.assertIs(b.causal_depthwise_residual,original)



class PairedRowsTests(unittest.TestCase):
    def rows(self,condition='on'):
        return [dict(condition=condition,stage=condition+'_validation',target_id=str(i),entity_id='a',seq=i,split='validation',recorded_date='x',raw_quantity=1.,recorded_gap=2.,
            history_length=3,last_quantity=4.,mean_quantity=5.,dataset='raf_spare_parts',seed=42,history_log_mean=1.,history_log_std=2.,
            last_log_deviation=3.,last_observed_log_change=4.,previous_observation_available=True,observed_history_sha256='h',
            prediction=float(i),time_nll=1.) for i in range(3)]

    def test_reordered_exact_targets_with_different_predictions_and_losses_pass(self):
        on=self.rows();off=self.rows('off');off[0]['prediction']=99.;off[2]['time_nll']=4.
        result=w.validate_paired_rows(off,list(reversed(on)),3)
        self.assertEqual(result['status'],'PASS');self.assertFalse(result['prediction_equality_required']);self.assertFalse(result['time_loss_equality_required'])

    def test_truth_history_identity_coverage_duplicates_and_nonfinite_rejected(self):
        rows=self.rows()
        for key,value in [('raw_quantity',2.),('recorded_gap',3.),('target_id','foreign'),('observed_history_sha256','foreign'),('seed',52),('history_log_std',999.),('prediction',float('nan')),('time_nll',float('inf'))]:
            actual=self.rows('off');actual[0][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):w.validate_paired_rows(actual,rows,3)
        off=self.rows('off')
        for actual in (off[:2],off[:2]+[off[0]]):
            with self.assertRaises(ValueError):w.validate_paired_rows(actual,rows,3)

    def test_row_condition_and_stage_cannot_be_relabelled(self):
        on=self.rows();off=self.rows('off')
        for role in ('on','off'):
            for field,value in [('condition','on' if role=='off' else 'off'),('stage','off_train'),('stage','on_test')]:
                a,b=deepcopy(off),deepcopy(on);(a if role=='off' else b)[0][field]=value
                with self.subTest(role=role,field=field,value=value),self.assertRaisesRegex(ValueError,'condition/stage'):
                    w.validate_paired_rows(a,b,3)


if __name__=='__main__':unittest.main()
