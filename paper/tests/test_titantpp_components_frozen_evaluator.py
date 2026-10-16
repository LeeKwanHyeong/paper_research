"""Local contracts and fabricated inference only; no scientific data reads."""
from __future__ import annotations

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
SCRIPT = ROOT / 'paper/scripts/evaluate_titantpp_components_frozen.py'
SPEC = importlib.util.spec_from_file_location('_components_eval_unit', SCRIPT)
eval = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval)
PARENT = ROOT / 'search_artifacts/titantpp_cnn_gru_components_20261012_v5'


def seal(c):
    c = deepcopy(c)
    c.pop('contract_sha256', None)
    c['contract_sha256'] = eval.canonical(c)
    return c


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def fabricated_contract():
    # Only frozen metadata/source manifest is read, never a research dataset.
    parent = json.loads((PARENT / 'execution_contract.json').read_text())
    source = parent['source']
    datasets = {}
    for d in parent['datasets']:
        name = d['dataset_id']
        datasets[name] = {'definition': deepcopy(d), 'tail_threshold': eval.TAILS[name],
            'data_path': 'fabricated/full.parquet', 'data_sha256': 'a' * 64,
            'training_data_path': 'fabricated/validation.parquet', 'training_data_sha256': 'b' * 64,
            'expected_test_count': 2, 'validation_population': {'count': 2,
                'target_identity_sha256': '1' * 64, 'truth_sha256': '2' * 64, 'time_sha256': '3' * 64},
            'test_population': {'count': 2, 'target_identity_sha256': '4' * 64,
                'truth_sha256': '5' * 64, 'time_sha256': '6' * 64}}
    jobs = []
    for dataset, arm, seed in sorted(eval.expected_jobs()):
        host = '5090' if dataset == 'intermittent_frozen_5000' and seed in (42, 52) else '5080'
        jobs.append({'id': f'{dataset}__{seed}__{arm}', 'cohort': eval.COHORT, 'study': 'components',
            'dataset': dataset, 'arm': arm, 'seed': seed, 'evaluation_host': host, 'training_host': host,
            'frozen_source_closure': eval.SOURCE_CLOSURE, 'binding_status': 'bound_terminal_immutable_originals',
            'selected_epoch': 4, 'selected_metric_value': 3., 'initial_state_sha256': '7' * 64,
            'checkpoint_state_sha256': '8' * 64, 'checkpoint_metadata_sha256': '9' * 64,
            'checkpoint_metadata_path': 'fabricated/metadata.json', 'selection_history_sha256': '0' * 64,
            'selection_history_path': 'fabricated/history.json', 'source_revision_history': [eval.SOURCE_REVISION],
            'checkpoint': {'relative_path': 'fabricated/best_val_qty_rmse_model.pt', 'sha256': 'a' * 64},
            'validation_reference': {'split': 'validation', 'count': 2, 'qty_rmse': 3., 'qty_mae': 2.,
                'time_nll': 1., 'tail': {'count': 1, 'qty_rmse': 4., 'qty_mae': 4., 'time_nll': 2.}}})
    return seal({'schema': eval.KIND, 'cohort': eval.COHORT, 'metrics_schema': 'quantity_time_tail_v1',
        'evaluation_scope': 'test_final_comparison', 'expected_seeds': list(eval.SEEDS), 'jobs': jobs,
        'sources': {'components': {'root': 'source', 'files': source['files'],
            'closure': source['files_sha256'], 'revision': source['git_revision']}},
        'datasets': datasets, 'runtimes': {'5080': {}, '5090': {}}})


class ComponentEvaluatorContracts(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='component-eval-synthetic-', dir='/private/tmp')
        self.root = Path(self.folder.name)
        self.c = fabricated_contract()
        training_path = self.root / 'metadata/training_contract.json'
        training_path.parent.mkdir(parents=True)
        shutil.copyfile(PARENT / 'execution_contract.json', training_path)
        evaluator_path = self.root / 'operation/evaluator.py'
        evaluator_path.parent.mkdir(parents=True)
        shutil.copyfile(SCRIPT, evaluator_path)
        self.c['training_contract_path'] = 'metadata/training_contract.json'
        self.c['training_contract_sha256'] = eval.sha(training_path)
        files = {'operation/evaluator.py': eval.sha(evaluator_path)}
        self.c['evaluator_integrity'] = {'files': files, 'closure': eval.canonical(files)}
        self.c['runtimes'] = {host: self.host_runtime(host)[0] for host in ('5080', '5090')}
        self.c = seal(self.c)
        dump(self.root / 'contract.json', self.c)

    def tearDown(self):
        self.folder.cleanup()

    def test_exact_component36_and_static_check_loads_no_data(self):
        self.assertEqual(len(eval.verify_contract(self.root)['jobs']), 36)
        r = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root), '--mode', 'check-contract'],
            capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertFalse(json.loads(r.stdout)['real_data_loaded'])
        self.assertFalse(json.loads(r.stdout)['execution_ready'])

    def test_foreign_history_instacart36_is_not_component36(self):
        for changes in ({'schema': 'titantpp_history_instacart_frozen_test_20261012_v1'}, {'cohort': 'history27'},
                        {'jobs': self.c['jobs'][:-1]}, {'expected_seeds': [42, 52]},
                        {'sources': {'components': {**self.c['sources']['components'], 'closure': 'b' * 64}}}):
            dump(self.root / 'contract.json', seal({**self.c, **changes}))
            with self.assertRaises(ValueError):
                eval.verify_contract(self.root)

    def test_foreign_job_route_cohort_host_and_tail_are_rejected(self):
        for key, value in (('cohort', 'instacart9'), ('arm', 'titantpp_cnn_gru54_current_only'),
                           ('evaluation_host', 'unapproved'), ('id', '../../escape')):
            c = deepcopy(self.c)
            c['jobs'][0][key] = value
            dump(self.root / 'contract.json', seal(c))
            with self.assertRaises(ValueError):
                eval.verify_contract(self.root)
        c = deepcopy(self.c)
        c['datasets']['raf_spare_parts']['tail_threshold'] = 199.
        dump(self.root / 'contract.json', seal(c))
        with self.assertRaises(ValueError):
            eval.verify_contract(self.root)

    def test_pending_null_binary_cannot_be_executed(self):
        c = deepcopy(self.c)
        c['jobs'][0]['selected_epoch'] = None
        c['jobs'][0]['binding_status'] = 'pending_terminal_immutable_originals'
        dump(self.root / 'contract.json', seal(c))
        self.assertEqual(len(eval.verify_contract(self.root, require_bound=False)['jobs']), 36)
        with self.assertRaisesRegex(ValueError, 'not bound'):
            eval.verify_contract(self.root)

    def test_hash_mismatch_and_artifact_escape(self):
        c = deepcopy(self.c)
        c['cohort'] = 'other'
        dump(self.root / 'contract.json', c)
        with self.assertRaisesRegex(ValueError, 'canonical SHA'):
            eval.verify_contract(self.root)
        for path in ('../else', '/private/tmp/absolute'):
            with self.assertRaises(ValueError):
                eval.confined(self.root, path)

    def test_selector_finite_strict_earliest_not_last_or_test(self):
        rows = [{'epoch': 1, 'val_qty_rmse': float('nan')}, {'epoch': 2, 'val_qty_rmse': 4.},
            {'epoch': 3, 'val_qty_rmse': 3.}, {'epoch': 4, 'val_qty_rmse': 3.},
            {'epoch': 5, 'val_qty_rmse': float('inf')}]
        self.assertEqual(eval.strict_selected_epoch(rows), (3, 3.))
        with self.assertRaises(ValueError):
            eval.strict_selected_epoch([{'epoch': 1, 'val_qty_rmse': float('nan')}])
        with self.assertRaises(ValueError):
            eval.strict_selected_epoch([{'epoch': 2, 'val_qty_rmse': 1.}, {'epoch': 1, 'val_qty_rmse': 0.}])

    def test_selection_original_csv_bytes_nan_ties_and_json(self):
        path = self.root / 'history.csv'
        path.write_text('epoch,val_qty_rmse\n1,nan\n2,4\n3,3\n4,3\n5,inf\n')
        before = eval.sha(path)
        self.assertEqual(eval.strict_selected_epoch(eval.read_selection_history(path)), (3, 3.))
        self.assertEqual(eval.sha(path), before)
        path = self.root / 'history.json'
        dump(path, {'history': [{'epoch': 1, 'val_qty_rmse': 4.}, {'epoch': 2, 'val_qty_rmse': 3.}]})
        self.assertEqual(eval.strict_selected_epoch(eval.read_selection_history(path)), (2, 3.))

    def test_training_statistics_and_operational_bytes_are_sealed(self):
        c = deepcopy(self.c)
        c['datasets']['raf_spare_parts']['definition']['statistics']['train_log_mean'] += .1
        dump(self.root / 'contract.json', seal(c))
        with self.assertRaisesRegex(ValueError, 'statistics/model'):
            eval.verify_contract(self.root)
        dump(self.root / 'contract.json', self.c)
        (self.root / 'operation/evaluator.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'Operational evaluator file changed'):
            eval.verify_contract(self.root)

    def test_six_metrics_exact_counts_and_fixed_tolerance(self):
        ref = self.c['jobs'][0]['validation_reference']
        eval.compare_validation_metrics(deepcopy(ref), ref)
        for group in (None, 'tail'):
            for metric in eval.METRICS:
                bad = deepcopy(ref)
                cell = bad if group is None else bad[group]
                cell[metric] += .01
                with self.assertRaises(ValueError):
                    eval.compare_validation_metrics(bad, ref)
        bad = deepcopy(ref)
        bad['count'] += 1
        with self.assertRaises(ValueError):
            eval.compare_validation_metrics(bad, ref)
        empty = {'count': 0, **dict.fromkeys(eval.METRICS), 'tail': {'count': 0, **dict.fromkeys(eval.METRICS)}}
        eval.compare_validation_metrics(empty, empty)
        bad = deepcopy(empty)
        bad['qty_rmse'] = 0.
        with self.assertRaises(ValueError):
            eval.compare_validation_metrics(bad, empty)

    def test_no_permit_rejects_before_source_or_data_touch(self):
        job = self.c['jobs'][0]
        with patch.object(eval, 'verify_source', side_effect=AssertionError('source opened')), \
                patch.object(eval, 'write_exclusive', side_effect=AssertionError('claim written')):
            for split in ('validation', 'test'):
                with self.assertRaisesRegex(ValueError, 'permit required'):
                    eval.worker(self.root, job['id'], split, 'cuda')
        r = subprocess.run([sys.executable, str(SCRIPT), '--root', '/does/not/exist',
            '--mode', 'worker', '--job', job['id'], '--split', 'test'], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn('explicit hash-bound permit', r.stderr)

    def permit(self, **changes):
        job = self.c['jobs'][0]
        value = {'schema': 'titantpp_components_evaluation_permit_v1', 'authorized': True,
            'new_training': False, 'frozen_selection': True, 'stage': 'validation',
            'contract_sha256': self.c['contract_sha256'], 'evaluator_sha256': eval.sha(SCRIPT),
            'host': job['evaluation_host'], 'job_ids': [job['id']], 'not_before_unix': 100., 'deadline_unix': 200.}
        value.update(changes)
        dump(self.root / 'control/permit.json', value)
        return 'control/permit.json', eval.sha(self.root / 'control/permit.json')

    def test_permit_hash_stage_host_scope_and_deadline(self):
        job = self.c['jobs'][0]
        path, digest = self.permit()
        eval.verify_permit(self.root, self.c, job, 'validation', path, digest, now=150.)
        other_host = '5080' if job['evaluation_host'] == '5090' else '5090'
        for changes in ({'stage': 'test'}, {'host': other_host}, {'authorized': False},
                        {'job_ids': []}, {'new_training': True}, {'frozen_selection': False},
                        {'deadline_unix': 50000.}, {'evaluator_sha256': 'a' * 64}):
            path, digest = self.permit(**changes)
            with self.assertRaises(ValueError):
                eval.verify_permit(self.root, self.c, job, 'validation', path, digest, now=150.)
        path, digest = self.permit()
        for now in (99., 200.):
            with self.assertRaises(ValueError):
                eval.verify_permit(self.root, self.c, job, 'validation', path, digest, now=now)
        with self.assertRaises(ValueError):
            eval.verify_permit(self.root, self.c, job, 'validation', path, 'a' * 64, now=150.)

    def test_incomplete_global_validation_gate_blocks_test_before_source(self):
        job = self.c['jobs'][0]
        path, digest = self.permit(stage='test', not_before_unix=0., deadline_unix=43200.)
        dump(self.root / 'validation_gate.json', {'status': 'PASS', 'contract_sha256': self.c['contract_sha256'],
            'job_count': 36, 'validation_receipts': {}, 'validation_worker_exits': {}, 'created_unix': 150.})
        with patch.object(eval, 'verify_permit', return_value={}), \
                patch.object(eval, 'verify_source', side_effect=AssertionError('source opened')):
            with self.assertRaisesRegex(ValueError, 'missing/foreign'):
                eval.worker(self.root, job['id'], 'test', 'cuda', permit_path=path, permit_sha=digest)

    def test_all_population_digest_fields_must_match(self):
        expected = self.c['datasets'][eval.DATASETS[0]]['validation_population']
        eval.verify_population(expected, expected)
        for key in expected:
            bad = {**expected, key: 3 if key == 'count' else 'a' * 64}
            with self.assertRaises(ValueError):
                eval.verify_population(bad, expected)

    def test_source_closure_and_bytes_before_science_import(self):
        source = self.c['sources']['components']
        with self.assertRaises(FileNotFoundError):
            eval.verify_source(self.root, source)
        with self.assertRaisesRegex(ValueError, 'source131'):
            eval.verify_source(self.root, {**source, 'revision': 'other'})

    def test_checkpoint_epoch_source_scope_selector_metadata_seals_reject(self):
        job = self.c['jobs'][0]
        source = self.c['sources']['components']
        definition = self.c['datasets'][job['dataset']]['definition']
        payload = {'best_epoch': job['selected_epoch'], 'backbone': job['arm'], 'seed': job['seed'],
            'variant': 'count_only_log_regression', 'evaluation_scope': 'validation_only',
            'held_out_test_evaluated': False, 'checkpoint_monitor': 'validation_raw_quantity_rmse',
            'checkpoint_monitor_history_key': 'val_qty_rmse', 'checkpoint_selection': 'best_validation_raw_quantity_rmse',
            'initial_state_sha256': job['initial_state_sha256'], 'source_revision': source['revision'],
            'source_revision_history': job['source_revision_history'], 'selected_metric_value': job['selected_metric_value'],
            'model_state_sha256': job['checkpoint_state_sha256'],
            'interface_meta': {'time_head': {'observation_likelihood': definition['model']['time_observation_contract'],
                'time_scale': definition['model']['time_scale'], 'mode': definition['model']['time_head_mode']}}}
        eval.validate_payload(payload, job, source, definition, deepcopy(payload))
        for key, value in (('best_epoch', 300), ('source_revision', 'current-checkout'),
                           ('evaluation_scope', 'test'), ('seed', 999), ('selected_metric_value', .1),
                           ('model_state_sha256', 'f' * 64), ('checkpoint_selection', 'test_best')):
            with self.assertRaises(ValueError):
                eval.validate_payload({**payload, key: value}, job, source, definition)
        with self.assertRaises(ValueError):
            eval.validate_payload({**payload, 'extra_metadata': 'changed'}, job, source, definition, payload)

    def test_actual_worker_exit_and_receipt_sha_are_required_for_global_gate(self):
        bindings, exits = {}, {}
        for job in self.c['jobs']:
            path = self.root / 'results' / job['id']
            dump(path / 'validation_replay.json', {'completed_unix': 100.})
            dump(path / 'validation_worker_exit.json', {'job_id': job['id'],
                'contract_sha256': self.c['contract_sha256'], 'exit_code': 0,
                'actual_child_exit_observed': True, 'completed_unix': 100.})
            bindings[job['id']] = eval.sha(path / 'validation_replay.json')
            exits[job['id']] = eval.sha(path / 'validation_worker_exit.json')
        gate = {'status': 'PASS', 'contract_sha256': self.c['contract_sha256'], 'job_count': 36,
            'validation_receipts': bindings, 'validation_worker_exits': exits, 'created_unix': 150.}
        dump(self.root / 'validation_gate.json', gate)
        with patch.object(eval, 'verify_validation_receipt', side_effect=lambda r, c, j: r):
            self.assertEqual(eval.verify_global_validation_gate(self.root, self.c)['status'], 'PASS')
            path = self.root / 'results' / self.c['jobs'][0]['id'] / 'validation_worker_exit.json'
            original = json.loads(path.read_text())
            dump(path, {**original, 'exit_code': 1})
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                eval.verify_global_validation_gate(self.root, self.c)
            gate['validation_worker_exits'][self.c['jobs'][0]['id']] = eval.sha(path)
            dump(self.root / 'validation_gate.json', gate)
            with self.assertRaisesRegex(ValueError, 'exit0'):
                eval.verify_global_validation_gate(self.root, self.c)

    def host_runtime(self, host):
        expected = {'host': host, 'python': f'/native/{host}/env/bin/python3.12', 'python_version': '3.12.13',
            'torch': '2.9.0+pinned', 'cuda': '13.0', 'cudnn': 91900 if host == '5080' else 92000,
            'numpy': '2.3.3', 'polars': '1.34.0', 'torch_num_threads': 1, 'interop_threads': 1,
            'gpu_uuid': eval.GPU_UUIDS[host], 'dtype': 'float32', 'deterministic_algorithms': True, 'TF32': False}
        recorded = {key: expected[key] for key in ('torch', 'cuda', 'cudnn', 'numpy', 'polars')}
        recorded.update(python=expected['python_version'], python_executable=expected['python'],
            resolved_python_executable=expected['python'], threads=1, interop_threads=1,
            gpu={'uuid': expected['gpu_uuid']}, default_dtype='torch.float32', deterministic_algorithms=True,
            cuda_matmul_tf32=False, cudnn_tf32=False)
        return expected, recorded

    def test_cross_host_28_and_8_receipts_form_global_gate_on_either_host(self):
        c = deepcopy(self.c)
        for host in ('5080', '5090'):
            c['runtimes'][host] = self.host_runtime(host)[0]
        c = seal(c)
        bindings, exits, host_counts = {}, {}, {'5080': 0, '5090': 0}
        for job in c['jobs']:
            host = job['evaluation_host']
            host_counts[host] += 1
            runtime = self.host_runtime(host)[1]
            receipt = {'schema': eval.KIND, 'status': 'complete', 'split': 'validation',
                'contract_sha256': c['contract_sha256'],
                **{key: job[key] for key in ('id', 'cohort', 'study', 'dataset', 'arm', 'seed',
                    'selected_epoch', 'initial_state_sha256')},
                'checkpoint_sha256': job['checkpoint']['sha256'], 'state_sha256': job['checkpoint_state_sha256'],
                'source_closure': eval.SOURCE_CLOSURE, 'retrained': False, 'reselected': False,
                'raw_predictions_written': False, 'full_population': True, 'validation_match': 'PASS',
                'native_runtime_checked': True, 'qualification': dict.fromkeys(eval.QUALIFICATION_KEYS, True),
                'metrics': job['validation_reference'],
                **{key: c['datasets'][job['dataset']]['validation_population'][key]
                   for key in ('target_identity_sha256', 'truth_sha256', 'time_sha256')},
                'runtime': runtime, 'runtime_after': deepcopy(runtime), 'completed_unix': 100.}
            path = self.root / 'results' / job['id']
            dump(path / 'validation_replay.json', receipt)
            dump(path / 'validation_worker_exit.json', {'job_id': job['id'], 'contract_sha256': c['contract_sha256'],
                'exit_code': 0, 'actual_child_exit_observed': True, 'completed_unix': 100.})
            bindings[job['id']] = eval.sha(path / 'validation_replay.json')
            exits[job['id']] = eval.sha(path / 'validation_worker_exit.json')
        self.assertEqual(host_counts, {'5080': 28, '5090': 8})
        dump(self.root / 'validation_gate.json', {'status': 'PASS', 'contract_sha256': c['contract_sha256'],
            'job_count': 36, 'validation_receipts': bindings, 'validation_worker_exits': exits, 'created_unix': 150.})
        for current_host in ('5080', '5090'):
            with patch.object(eval.sys, 'executable', c['runtimes'][current_host]['python']):
                # No runtime/receipt verifier is mocked: both physical hosts are
                # validated using their recorded identities from either caller.
                self.assertEqual(eval.verify_global_validation_gate(self.root, c)['status'], 'PASS')

    def test_live_runtime_checks_own_process_but_receipt_uses_recorded_host(self):
        expected, recorded = self.host_runtime('5090')
        with patch.object(eval.sys, 'executable', '/native/5080/env/bin/python3.12'):
            eval.check_runtime(recorded, expected, 'cuda', current_process=False)
            with self.assertRaisesRegex(ValueError, 'current Python'):
                eval.check_runtime(recorded, expected, 'cuda')
        with patch.object(eval.sys, 'executable', expected['python']):
            eval.check_runtime(recorded, expected, 'cuda')
        for corrupted in ({**recorded, 'python_executable': '/foreign/python'},
                          {**recorded, 'gpu': {'uuid': 'GPU-5080'}}):
            with self.assertRaises(ValueError):
                eval.check_runtime(corrupted, expected, 'cuda', current_process=False)

    def test_native_runtime_schema_missing_and_invalid_fields_fail_as_valueerror(self):
        expected, recorded = self.host_runtime('5090')
        eval.validate_runtime_spec(expected, '5090')
        for key in expected:
            missing = {k: v for k, v in expected.items() if k != key}
            with self.assertRaisesRegex(ValueError, 'Missing native Runtime'):
                eval.validate_runtime_spec(missing)
        for changes in ({'python': 'relative/python'}, {'python_version': None}, {'torch': ''},
                        {'cuda': None}, {'cudnn': True}, {'gpu_uuid': 'GPU-wrong'},
                        {'torch_num_threads': 4}, {'interop_threads': 2}, {'torch_num_threads': True},
                        {'TF32': True}, {'deterministic_algorithms': False}, {'dtype': 'float64'}):
            with self.assertRaises(ValueError):
                eval.validate_runtime_spec({**expected, **changes})
        for key in ('python', 'python_executable', 'threads', 'gpu', 'cuda_matmul_tf32'):
            missing = {k: v for k, v in recorded.items() if k != key}
            with self.assertRaisesRegex(ValueError, 'Missing observed Runtime'):
                eval.check_runtime(missing, expected, 'cuda', current_process=False)
        c = deepcopy(self.c)
        del c['runtimes']['5090']['python_version']
        dump(self.root / 'contract.json', seal(c))
        with self.assertRaisesRegex(ValueError, 'Missing native Runtime.*python_version'):
            eval.verify_contract(self.root, require_bound=False)

    def test_help_import_is_standard_library_only(self):
        code = f"import runpy,sys; runpy.run_path({str(SCRIPT)!r}); assert 'torch' not in sys.modules; assert 'polars' not in sys.modules"
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_synthetic_sealed_source_four_routes_causal_batch_strict_invariant(self):
        # Copy science files only to the temporary fixture; no input/artifact tree.
        for relative in self.c['sources']['components']['files']:
            dest = self.root / 'source' / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PARENT / 'source' / relative, dest)
        (self.root / 'source/sample_data').mkdir()
        r = subprocess.run([sys.executable, str(SCRIPT), '--root', str(self.root), '--mode', 'qualify-synthetic'],
            capture_output=True, text=True, timeout=120, env={**__import__('os').environ, 'PYTHONDONTWRITEBYTECODE': '1'})
        self.assertEqual(r.returncode, 0, r.stderr)
        value = json.loads(r.stdout)
        self.assertEqual(set(value['arms']), set(eval.ARMS))
        self.assertFalse(value['real_data_loaded'])
        self.assertFalse(value['held_out_test_evaluated'])
        self.assertEqual(value['optimizer_updates'], 0)
        self.assertEqual(value['checkpoint_reads'], 0)
        for arm in value['arms'].values():
            self.assertTrue(all(arm['qualification'].values()))
            self.assertTrue(arm['foreign_routes_rejected'])


if __name__ == '__main__':
    unittest.main()
