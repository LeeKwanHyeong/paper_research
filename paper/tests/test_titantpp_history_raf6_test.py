"""Offline RAF6 orchestration contracts; no SSH and no real Test inference."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from paper.scripts import advance_titantpp_history_raf6_test as a
from paper.scripts import evaluate_titantpp_history_raf6_frozen as e


class RAF6EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/private/tmp', prefix='cap9-eval-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_native_reference_maps_only_verified_validation_and_preserves_original(self):
        endpoint = {'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
                    'qty_rmse': 1.25, 'tail': {'qty_rmse': 3.5}}
        before = deepcopy(endpoint)
        result = a.validation_reference(endpoint)
        self.assertEqual(result, {**before, 'split': 'validation'})
        result['tail']['qty_rmse'] = 99.
        self.assertEqual(endpoint, before)
        for change in ({'evaluation_scope': 'test'}, {'held_out_test_evaluated': True},
                       {'held_out_test_evaluated': None}, {'split': 'test'}):
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Validation-only'):
                a.validation_reference({**endpoint, **change})

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

    def test_gate_requires_all_six_explicit_worker_exit0(self):
        jobs = [{'id': str(i)} for i in range(6)]
        c = {'jobs': jobs, 'contract_sha256': 'a' * 64}
        receipts, exits = {}, {}
        for j in jobs:
            stem = self.root / 'results' / j['id']
            a.write(stem / 'validation_replay.json', {'completed_unix': 10.})
            a.write(stem / 'validation_worker_exit.json', {'job_id': j['id'], 'contract_sha256': c['contract_sha256'],
                    'actual_child_exit_observed': True, 'exit_code': 0, 'completed_unix': 10.})
            receipts[j['id']] = a.sha(stem / 'validation_replay.json')
            exits[j['id']] = a.sha(stem / 'validation_worker_exit.json')
        gate = {'schema': 'titantpp_history_raf6_validation_gate_v1', 'cohort': e.COHORT,
                'status': 'PASS', 'contract_sha256': c['contract_sha256'], 'job_count': 6,
                'validation_receipts': receipts, 'validation_worker_exits': exits, 'created_unix': 11.}
        path = self.root / 'validation_gate.json'; a.write(path, gate)
        with patch.object(e, 'verify_validation_receipt', side_effect=lambda r, c, j: r):
            e.verify_global_validation_gate(self.root, c)
            for change in ({'job_count': 5}, {'validation_worker_exits': {}}, {'created_unix': 9.}):
                path.write_text(json.dumps({**gate, **change}))
                with self.assertRaises(ValueError):
                    e.verify_global_validation_gate(self.root, c)
            bad = self.root / 'results/0/validation_worker_exit.json'
            record = a.read(bad); record['exit_code'] = 1; bad.write_text(json.dumps(record))
            gate['validation_worker_exits']['0'] = a.sha(bad); path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ValueError, 'actual worker exit0'):
                e.verify_global_validation_gate(self.root, c)

    def test_permit_rejects_wrong_host_stage_and_expired_or_extended_budget(self):
        p = {'schema': 'titantpp_history_raf6_evaluation_permit_v1', 'authorized': True, 'new_training': False,
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
            for arm, shift in ((a.CNN, 0.), (a.CAPACITY, 1.), (a.ADAPTIVE, 2.)):
                for seed, x in zip(a.SEEDS, (1., 2., 3.)):
                    metrics = {'qty_rmse': x + shift, 'qty_mae': (x + shift) / 2., 'time_nll': x + shift}
                    rows.append({'dataset': d, 'arm': arm, 'seed': seed, 'metrics': {**metrics, 'tail': metrics.copy()},
                                 'target_identity_sha256': 'a', 'truth_sha256': 'b', 'time_sha256': 'c'})
        return rows

    def test_aggregation_sample_sd_and_unfavorable_seed_differences(self):
        r = a.aggregate(self.rows())
        cnn = r['grouped'][0]['metrics']['whole_qty_rmse']
        self.assertEqual(cnn, {'mean': 2., 'sample_sd': 1.})
        self.assertEqual(len(r['seed_differences']), 9)
        self.assertEqual(r['seed_differences'][0]['metrics']['whole_qty_rmse']['candidate_minus_baseline'], 2.)
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


    def test_exact_raf_cohort_excludes_taxi_and_gru(self):
        self.assertEqual(e.expected_jobs(), {('raf_spare_parts', arm, seed) for arm in (a.ADAPTIVE,a.CAPACITY) for seed in (42,52,62)})

    def test_training_terminal_proof_is_complete_and_read_only(self):
        with patch.object(a,'transport') as remote:
            proof=a.collection_gate(a.training_contract())
        self.assertEqual(proof['new_fit_count'],6)
        self.assertFalse(proof['held_out_test_evaluated'])
        remote.assert_not_called()

    def test_bind_saved_checkpoints_in_isolated_copy_without_inference(self):
        original=a.BUNDLE;target=self.root/'evaluation'
        draft=a.read(original/'draft_contract.json');approval=a.read(original/'approval.json')
        approval['local_root']=str(target);a.write(target/'approval.json',approval)
        draft['partial_approval_sha256']=a.sha(target/'approval.json')
        for name in ('operation/evaluate.py','operation/pipeline.py','provenance/training_contract.json'):
            a.copy_checked(original/name,target/name,a.sha(original/name))
        a.write(target/'draft_contract.json',draft)
        with patch.object(a,'BUNDLE',target),patch.object(a,'transport') as remote:
            c=a.bind(a.training_contract(),a.collection_gate(a.training_contract()))
            self.assertEqual([j['selected_epoch'] for j in c['jobs']],[9,5,22,9,5,22])
            self.assertTrue(all(j['evaluation_host']=='5090' for j in c['jobs']))
            remote.assert_not_called()
        # Byte tampering is rejected locally, before any model or data access.
        path=target/c['jobs'][0]['selection_history_path'];path.write_text('[]')
        with self.assertRaisesRegex(ValueError,'artifact changed'):
            e.verify_contract(target)


if __name__ == '__main__':
    unittest.main()
