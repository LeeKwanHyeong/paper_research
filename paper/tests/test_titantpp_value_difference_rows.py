"""Synthetic-only current-candidate row diagnostics; no remote or scientific data."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import export_titantpp_value_difference_rows as w


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
        self.tmp = tempfile.TemporaryDirectory(dir='/private/tmp', prefix='value-rows-synthetic-')
        self.root = Path(self.tmp.name)
        self.jobs = [job(s) for s in w.SEEDS]
        self.population = {'target_count': 6690, 'target_identity_sha256': 'p' * 64}
        self.contract = {'contract_sha256': 'c' * 64, 'jobs': self.jobs, 'deadline_unix': 200.,
                         'datasets': {'raf_spare_parts': {'definition': {'inherited_data_identity': {'populations': {'validation': self.population}}}}}}

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
        return w.sha(path)

    def fixture_contract(self):
        inputs, operations, source = {}, {}, {}
        for i in range(141):
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
        approval = {'authorized': True, 'new_training': False, 'frozen_selection': True, 'allowed_splits': ['train', 'validation']}
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
             'approval_path': 'approval.json', 'approval_sha256': inputs['approval.json']}
        return c

    def seal(self, c):
        c['contract_sha256'] = w.canonical({k: v for k, v in c.items() if k != 'contract_sha256'})
        return self.save('contract.json', c)

    def verify_fixture(self, c):
        raw = self.seal(c)
        with patch.object(w, 'SOURCE_CLOSURE', c['source']['closure']), patch.object(w, 'DATA_SHA', c['datasets']['raf_spare_parts']['sha256']), \
             patch.object(w, 'SPEC_SHA', c['spec_sha256']), patch.object(w.time, 'time', return_value=100.):
            return w.verify_contract(self.root, raw)

    def gate(self, stage):
        g = {'schema': f'titantpp_value_difference_rows_{stage}_gate_v1', 'contract_sha256': self.contract['contract_sha256'],
             'all_native_exit0': True, 'job_ids': [j['id'] for j in self.jobs], 'receipts': {}, 'worker_exits': {}}
        for j in self.jobs:
            ident = j['id']; receipt = {'status': 'complete', 'job_id': ident, 'split': stage, 'contract_sha256': self.contract['contract_sha256'],
                    'parameters_unchanged': True, 'state_including_buffers_unchanged': True,
                    'state_sha256_before': j['checkpoint_state_sha256'], 'state_sha256_after': j['checkpoint_state_sha256'],
                    'native_runtime_checked': True, 'no_training': True, 'held_out_test_evaluated': False}
            if stage == 'qualification':
                receipt.update(qualification_passed=True, real_data_loaded=False)
            else:
                part = f'rows/{ident}/validation/part-00000.parquet'; path = self.root / part; path.parent.mkdir(parents=True); path.write_bytes(b'fabricated shard')
                manifest = {'prediction_rows': 6690, 'job_id': ident, 'contract_sha256': self.contract['contract_sha256'], 'split': 'validation',
                            'parts': [{'path': 'part-00000.parquet', 'rows': 6690, 'sha256': w.sha(path)}]}
                manifest_path = f'rows/{ident}/validation/manifest.json'; manifest_sha = self.save(manifest_path, manifest)
                receipt.update(validation_match='PASS', selected_epoch=j['selected_epoch'], checkpoint_sha256=j['checkpoint']['sha256'],
                               metrics=metric_reference(), population=self.population, manifest_path=manifest_path, manifest_sha256=manifest_sha)
            rel = f'results/{ident}/{stage}.json'; g['receipts'][ident] = {'path': rel, 'sha256': self.save(rel, receipt)}
            actual = {'job_id': ident, 'split': stage, 'contract_sha256': self.contract['contract_sha256'], 'actual_child_exit_observed': True, 'exit_code': 0}
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

    def test_disallowed_split_cpu_or_foreign_root_fails_before_io(self):
        for mode, split, device in [('worker', 'test', 'cuda'), ('worker', 'all', 'cuda'), ('qualify', 'train', 'cuda'), ('worker', 'train', 'cpu')]:
            with self.subTest((mode, split, device)), patch.object(w, 'verify_contract') as c, self.assertRaises(ValueError):
                w.worker(self.root, 'raw', self.jobs[0]['id'], split, device, mode=mode)
            c.assert_not_called()
        with patch.object(w, 'verify_contract') as c, self.assertRaisesRegex(ValueError, 'native diagnostic root'):
            w.worker(self.root, 'raw', self.jobs[0]['id'], 'train')
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
        gate = self.gate('validation'); digest = self.save('validation_gate.json', gate)
        self.assertEqual(w.verify_gate(self.root, self.contract, 'validation', digest), gate)
        rel = gate['receipts'][self.jobs[0]['id']]['path']; receipt = w.read(self.root / rel)
        receipt['metrics']['tail']['qty_mae'] += .1
        changed = deepcopy(gate); changed['receipts'][self.jobs[0]['id']]['sha256'] = self.save(rel, receipt)
        digest = self.save('validation_gate.json', changed)
        with self.assertRaisesRegex(ValueError, 'six-metric'): w.verify_gate(self.root, self.contract, 'validation', digest)
        receipt['metrics'] = metric_reference(); gate['receipts'][self.jobs[0]['id']]['sha256'] = self.save(rel, receipt)
        digest = self.save('validation_gate.json', gate)
        (self.root / f'rows/{self.jobs[0]["id"]}/validation/part-00000.parquet').write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError, 'shard changed'): w.verify_gate(self.root, self.contract, 'validation', digest)

    def test_validation_requires_qualification_and_train_requires_validation(self):
        for stage, prior, key in [('validation', 'qualification', 'qualification_gate_sha256'), ('train', 'validation', 'validation_gate_sha256')]:
            p = self.permit(stage); p[key] = 'g' * 64; digest = self.save('permit.json', p)
            with self.subTest(stage=stage), patch.object(w.time, 'time', return_value=100.), patch.object(w, 'verify_gate') as gate:
                w.verify_permit(self.root, self.contract, 'r' * 64, self.jobs[0], stage, 'permit.json', digest)
                gate.assert_called_once_with(self.root, self.contract, prior, 'g' * 64)

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
            result = w.worker(self.root, 'raw', self.jobs[0]['id'], None, mode='qualify')
        data.assert_not_called(); self.assertFalse(result['real_data_loaded']); self.assertTrue(result['qualification_passed'])
        self.assertTrue(result['state_including_buffers_unchanged']); self.assertFalse(result['held_out_test_evaluated'])

    def test_original_failure_and_claim_preserved_no_second_attempt(self):
        with patch.object(w, 'REMOTE_ROOT', str(self.root)), patch.object(w, 'verify_contract', return_value=self.contract), \
             patch.object(w, 'verify_permit', return_value={}), patch.object(w, 'prepare_model', side_effect=ValueError('original synthetic failure')) as model:
            with self.assertRaisesRegex(ValueError, 'original synthetic failure'):
                w.worker(self.root, 'raw', self.jobs[0]['id'], None, mode='qualify')
            p = self.root / f'results/{self.jobs[0]["id"]}/qualification_failure.json'; before = p.read_bytes()
            self.assertIn('original synthetic failure', w.read(p)['traceback'])
            with self.assertRaisesRegex(ValueError, 'Existing attempt'):
                w.worker(self.root, 'raw', self.jobs[0]['id'], None, mode='qualify')
            self.assertEqual(p.read_bytes(), before); self.assertEqual(model.call_count, 1)


class RowNumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/value-rows-synthetic-mpl')
        os.environ.setdefault('XDG_CACHE_HOME', '/private/tmp/value-rows-synthetic-cache')
        import numpy as np
        import torch
        import polars as pl
        from paper.scripts import evaluate_titantpp_frozen_rows as utils
        from paper.scripts import evaluate_titantpp_value_difference_raf3_frozen as helper
        from paper.scripts import run_titantpp_value_difference_cnn as engine
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
            self.assertEqual(sum(p.numel() for p in model.parameters()), 104095)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), 79363)
            self.assertFalse(any('causal_v_kernel' in k for k in model.state_dict()))
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if any(k in name for k in ('causal_q_kernel', 'causal_k_kernel', 'value_difference_kernel')): parameter.normal_(0., .03)
            model.eval().requires_grad_(False)
            ctx = {'torch': torch, 'model': model, 'core': self.core, 'runner': self.runner, 'helper': self.helper,
                   'definition': data, 'state': self.runner.canonical_state_dict_sha256(model.state_dict()), 'predictive_state': w.predictive_state(model, self.runner)}
            result = w.synthetic_checks(ctx)
            self.assertTrue(all(result.values())); self.assertEqual(w.predictive_state(model, self.runner), ctx['predictive_state'])

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


if __name__ == '__main__':
    unittest.main()
