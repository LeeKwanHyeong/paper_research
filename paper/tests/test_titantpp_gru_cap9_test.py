"""Offline CAP9 orchestration contracts; no SSH and no real Test inference."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from paper.scripts import advance_titantpp_gru_cap9_test as a
from paper.scripts import evaluate_titantpp_gru_cap9_frozen as e


class CAP9EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp', prefix='cap9-eval-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_waiting_performs_no_remote_or_binding_and_does_not_claim(self):
        remote, bind = Mock(), Mock()
        with patch.object(a, 'BUNDLE', self.root), patch.object(a, 'verify_bootstrap', return_value={}), \
             patch.object(a, 'collection_gate', return_value=None), patch.object(a, 'transport', remote), \
             patch.object(a, 'bind', bind):
            result = a.advance('a' * 64)
        self.assertEqual(result['remote_calls'], 0)
        remote.assert_not_called(); bind.assert_not_called()
        self.assertFalse((self.root / 'advance.claim').exists())

    def test_integrity_failure_precedes_collection_and_transport(self):
        gate, remote = Mock(), Mock()
        with patch.object(a, 'verify_bootstrap', side_effect=ValueError('seal differs')), \
             patch.object(a, 'collection_gate', gate), patch.object(a, 'transport', remote):
            with self.assertRaisesRegex(ValueError, 'seal'):
                a.advance('a' * 64)
        gate.assert_not_called(); remote.assert_not_called()

    def test_failed_or_started_attempt_never_retries(self):
        for name in ('advance.claim', 'failure.json'):
            p = self.root / name; p.write_text('{}')
            with patch.object(a, 'BUNDLE', self.root), patch.object(a, 'verify_bootstrap', return_value={}), \
                 patch.object(a, 'collection_gate') as gate:
                with self.assertRaisesRegex(ValueError, 'no automatic retry'):
                    a.advance('a' * 64)
                gate.assert_not_called()
            p.unlink()

    def test_all_validation_precedes_test_and_failure_prevents_test(self):
        order = []
        with patch.object(a, 'BUNDLE', self.root), patch.object(a, 'verify_bootstrap', return_value={}), \
             patch.object(a, 'collection_gate', return_value={'status': 'complete'}), \
             patch.object(a, 'bind', return_value={'bound_files': {}}), \
             patch.object(a, 'read', return_value={'files': {}}), patch.object(a, 'deploy'), \
             patch.object(a, 'evaluate_stage', side_effect=lambda c, t, s: order.append(s)), \
             patch.object(a, 'make_gate', side_effect=ValueError('Validation mismatch')):
            with self.assertRaisesRegex(ValueError, 'Validation mismatch'):
                a.advance('a' * 64)
        self.assertEqual(order, ['validation'])
        self.assertTrue((self.root / 'failure.json').exists())

    def test_strict_earliest_finite_minimum(self):
        history = [{'epoch': 1, 'val_qty_rmse': None}, {'epoch': 2, 'val_qty_rmse': 2.},
                   {'epoch': 3, 'val_qty_rmse': 2.}, {'epoch': 4, 'val_qty_rmse': 3.}]
        self.assertEqual(e.strict_selected_epoch(history), (2, 2.))
        with self.assertRaises(ValueError):
            e.strict_selected_epoch(history[::-1])

    def test_unbound_checkpoint_rejected_before_data_access(self):
        with patch.object(e, 'read', return_value={'binding_complete': False}), \
             patch.object(e, 'canonical', return_value='x'), patch.object(e, 'verify_source') as source:
            with self.assertRaises(ValueError):
                e.worker(self.root, 'foreign', 'test', 'cuda')
            source.assert_not_called()

    def test_missing_global_gate_blocks_before_science(self):
        c = {'jobs': [{'id': 'job', 'evaluation_host': '5080'}], 'runtimes': {'5080': {}}}
        with patch.object(e, 'verify_contract', return_value=c), patch.object(e, 'verify_permit', return_value={}), \
             patch.object(e, 'verify_source') as source:
            with self.assertRaises(FileNotFoundError):
                e.worker(self.root, 'job', 'test', 'cuda', permit_path='x', permit_sha='a' * 64)
            source.assert_not_called()

    def test_gate_requires_all_nine_explicit_worker_exit0(self):
        jobs = [{'id': str(i)} for i in range(9)]
        c = {'jobs': jobs, 'contract_sha256': 'a' * 64}
        receipts, exits = {}, {}
        for j in jobs:
            stem = self.root / 'results' / j['id']
            a.write(stem / 'validation_replay.json', {'completed_unix': 10.})
            a.write(stem / 'validation_worker_exit.json', {'job_id': j['id'], 'contract_sha256': c['contract_sha256'],
                    'actual_child_exit_observed': True, 'exit_code': 0, 'completed_unix': 10.})
            receipts[j['id']] = a.sha(stem / 'validation_replay.json')
            exits[j['id']] = a.sha(stem / 'validation_worker_exit.json')
        gate = {'schema': 'titantpp_gru_cap9_validation_gate_v1', 'cohort': e.COHORT,
                'status': 'PASS', 'contract_sha256': c['contract_sha256'], 'job_count': 9,
                'validation_receipts': receipts, 'validation_worker_exits': exits, 'created_unix': 11.}
        path = self.root / 'validation_gate.json'; a.write(path, gate)
        with patch.object(e, 'verify_validation_receipt', side_effect=lambda r, c, j: r):
            e.verify_global_validation_gate(self.root, c)
            for change in ({'job_count': 8}, {'validation_worker_exits': {}}, {'created_unix': 9.}):
                path.write_text(json.dumps({**gate, **change}))
                with self.assertRaises(ValueError):
                    e.verify_global_validation_gate(self.root, c)
            bad = self.root / 'results/0/validation_worker_exit.json'
            record = a.read(bad); record['exit_code'] = 1; bad.write_text(json.dumps(record))
            gate['validation_worker_exits']['0'] = a.sha(bad); path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, 'actual worker exit0'):
                e.verify_global_validation_gate(self.root, c)

    def test_permit_rejects_wrong_host_stage_and_expired_or_extended_budget(self):
        p = {'schema': 'titantpp_gru_cap9_evaluation_permit_v1', 'authorized': True, 'new_training': False,
             'frozen_selection': True, 'stage': 'validation', 'host': '5080', 'job_ids': ['job'],
             'contract_sha256': 'a' * 64, 'evaluator_sha256': e.sha(e.__file__),
             'not_before_unix': 100., 'deadline_unix': 200.}
        job = {'id': 'job', 'evaluation_host': '5080'}
        c = {'contract_sha256': 'a' * 64, 'jobs': [job]}
        def check(value, stage='validation'):
            path = self.root / 'permit.json'; path.write_text(json.dumps(value))
            with patch.object(e, 'verify_partial_metadata', return_value={'remote_roots': {'5080': str(self.root)}}):
                return e.verify_permit(self.root, c, job, stage, 'permit.json', a.sha(path), now=150.)
        check(p)
        for change in ({'host': '5090'}, {'new_training': True}, {'job_ids': ['other']},
                       {'deadline_unix': 120.}, {'deadline_unix': 50000.}):
            with self.assertRaises(ValueError):
                check({**p, **change})
        with self.assertRaises(ValueError):
            check(p, 'test')

    def rows(self):
        rows = []
        for d in a.DATASETS:
            for arm, shift in ((a.CNN, 0.), (a.FULL, 1.), (a.CAP, 2.)):
                for seed, x in zip(a.SEEDS, (1., 2., 3.)):
                    metrics = {'qty_rmse': x + shift, 'qty_mae': (x + shift) / 2., 'time_nll': x + shift}
                    rows.append({'dataset': d, 'arm': arm, 'seed': seed, 'metrics': {**metrics, 'tail': metrics.copy()},
                                 'target_identity_sha256': 'a', 'truth_sha256': 'b', 'time_sha256': 'c'})
        return rows

    def test_aggregation_sample_sd_and_unfavorable_seed_differences(self):
        r = a.aggregate(self.rows())
        cnn = r['grouped'][0]['metrics']['whole_qty_rmse']
        self.assertEqual(cnn, {'mean': 2., 'sample_sd': 1.})
        self.assertEqual(len(r['seed_differences']), 18)
        self.assertEqual(r['seed_differences'][0]['metrics']['whole_qty_rmse']['cap_minus_baseline'], 2.)
        self.assertFalse(r['test_independent'])

    def test_aggregation_rejects_missing_duplicate_or_unmatched_population(self):
        rows = self.rows()
        for bad in (rows[:-1], rows[:-1] + [rows[0]]):
            with self.assertRaises(ValueError):
                a.aggregate(bad)
        bad = deepcopy(rows); bad[0]['truth_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'population'):
            a.aggregate(bad)

    def test_archive_rejects_unexpected_members_before_writing(self):
        archive = self.root / 'bad.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            raw = json.dumps({'../outside': 'a' * 64}).encode()
            info = tarfile.TarInfo('result_manifest.json'); info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        with patch.object(a, 'BUNDLE', self.root / 'target'):
            with self.assertRaisesRegex(ValueError, 'membership'):
                a.extract_results(archive, '5080', 'validation', {'jobs': []})
        self.assertFalse((self.root / 'target').exists())

    def test_binding_all_nine_uses_saved_history_and_selected_binary_only(self):
        import torch
        original_bundle = a.BUNDLE
        draft = a.read(original_bundle / 'draft_contract.json')
        training = a.training_contract()
        target = self.root / 'evaluation'
        train = self.root / 'training'
        approval = a.read(original_bundle / 'approval.json')
        approval['local_root'] = str(target)
        a.write(target / 'approval.json', approval)
        draft['partial_approval_sha256'] = a.sha(target / 'approval.json')
        for n in ('operation/evaluate.py', 'operation/pipeline.py', 'provenance/training_contract.json'):
            a.copy_checked(original_bundle / n, target / n, a.sha(original_bundle / n))
        a.write(target / 'draft_contract.json', draft)
        receipt = {'status': 'complete', 'candidate_fit_count': 9, 'seed_count': 3,
                   'baseline_reference_count': 18, 'held_out_test_evaluated': False,
                   'contract_sha256': a.TRAIN_CANONICAL}
        a.write(train / 'retrieval/final_receipt.json', receipt)
        index = {'jobs': {}}
        for j in draft['jobs']:
            base = train / 'retrieval/originals' / j['id']
            model = draft['datasets'][j['dataset']]['definition']['model']
            payload = {'best_epoch': 2, 'selected_metric_value': 1., 'seed': j['seed'], 'backbone': a.CAP,
                'variant': 'count_only_log_regression', 'evaluation_scope': 'validation_only',
                'held_out_test_evaluated': False, 'checkpoint_monitor': 'validation_raw_quantity_rmse',
                'checkpoint_monitor_history_key': 'val_qty_rmse', 'checkpoint_selection': 'best_validation_raw_quantity_rmse',
                'initial_state_sha256': 'a' * 64, 'source_revision': e.SOURCE_REVISION,
                'source_revision_history': [e.SOURCE_REVISION], 'model_state_sha256': 'b' * 64,
                'model_state_dict': {'synthetic': torch.tensor([1.])},
                'interface_meta': {'time_head': {'observation_likelihood': model['time_observation_contract'],
                                   'time_scale': model['time_scale'], 'mode': model['time_head_mode']}}}
            base.mkdir(parents=True)
            torch.save(payload, base / 'best_val_qty_rmse_model.pt')
            a.write(base / 'history.json', {'history': [{'epoch': 1, 'val_qty_rmse': 3.},
                                                       {'epoch': 2, 'val_qty_rmse': 1.},
                                                       {'epoch': 3, 'val_qty_rmse': 1.}]})
            metric = {'count': 1, 'qty_rmse': 1., 'qty_mae': 1., 'time_nll': 1.}
            endpoint = {**metric, 'tail': metric.copy(), 'state_sha256': 'b' * 64, 'split': 'validation'}
            a.write(base / 'endpoint_replays.json', {'selected': endpoint})
            for n in ('terminal_manifest.json', 'input_receipt.json', 'derived_validation_replay.json'):
                a.write(base / n, {'synthetic': True})
            row = {'state_tensor_sha256': 'b' * 64}
            for key, n in [('checkpoint', 'best_val_qty_rmse_model.pt'), ('history', 'history.json'),
                           ('endpoint_replays', 'endpoint_replays.json'), ('terminal_manifest', 'terminal_manifest.json'),
                           ('input_receipt', 'input_receipt.json'), ('derived_validation_replay', 'derived_validation_replay.json')]:
                row[key + '_path'] = j['id'] + '/' + n
                row[key + '_sha256'] = a.sha(base / n)
            index['jobs'][j['id']] = row
        a.write(train / 'retrieval/retrieval_index.json', index)
        with patch.object(a, 'BUNDLE', target), patch.object(a, 'TRAIN', train):
            bound = a.bind(training, receipt)
        self.assertEqual(len(bound['jobs']), 9)
        self.assertTrue(all(j['selected_epoch'] == 2 for j in bound['jobs']))
        self.assertTrue(bound['binding_complete'])
        self.assertFalse(any((target / 'results').glob('**/*')))


if __name__ == '__main__':
    unittest.main()
