"""Fail-closed originals recovery and the fixed history Validation decision."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

from paper.scripts import finalize_titantpp_history_rtx_handoff as f


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.contract = f.read(f.BUNDLE / 'execution_contract.json')

    def tearDown(self):
        self.temp.cleanup()

    def snapshot(self):
        c = self.contract
        jobs = [j for j in c['jobs'] if j['host'] == '5080']
        raw = 'a' * 64
        snapshot = {'host': '5080', 'contract_sha256': f.CONTRACT, 'root': c['hosts']['5080']['root'], 'rawSHA_verified': True, 'observed_unix': 123, 'processes': [], 'gpu_processes': '', 'jobs': {}, 'failure.json': None, 'pipeline_failure.json': None, 'raw_files_sha256': {'supervisor_exit.json': raw}, 'supervisor_exit.json': {'contract_sha256': f.CONTRACT, 'returncode': 0, 'actual_child_exit_observed': True, 'phase': 'dispatch'}, 'status.json': {'status': 'complete', 'completed_conditions': 3}, 'pipeline_complete.json': {'status': 'complete', 'completed_conditions': 3, 'contract_sha256': f.CONTRACT, 'held_out_test_evaluated': False, 'supervisor_exit_sha256': raw, 'actual_exit_and_idle_observation': {'owned_processes': [], 'gpu_pids': []}}, 'pipeline_heartbeat.json': {'phase': 'complete', 'contract_sha256': f.CONTRACT}}
        for j in jobs:
            snapshot['jobs'][j['id']] = {'job': j, 'terminal': {'job': j, 'contract_sha256': f.CONTRACT, 'scientific_success': True, 'status': 'complete', 'held_out_test_evaluated': False}, 'terminal_manifest_sha256': 'b' * 64, 'worker_exit': {'job': j, 'contract_sha256': f.CONTRACT, 'returncode': 0, 'actual_child_exit_observed': True}, 'endpoint': {'completed_epochs': 40, 'best_epoch': 1}}
        snapshot['raw_files_sha256'].update({'pipeline_complete.json': 'c' * 64, 'pipeline_heartbeat.json': 'd' * 64})
        snapshot['pipeline_complete.json']['completed_unix'] = 100
        snapshot['pipeline_heartbeat.json']['updated_unix'] = 110
        return snapshot

    def cache(self, snapshot):
        monitor = self.root / 'hourly_monitor'
        raw_path = monitor / 'snapshot.json'
        f.write_new(raw_path, snapshot)
        out = f.observer.classify(self.contract, '5080', snapshot)
        if not snapshot.get('raw_files_sha256'):
            # Model an older observer's immutable cache, before strict raw gates.
            out['verified_terminal'] = True
        out.update(snapshot_path=str(raw_path), snapshot_raw_sha256=f.sha(raw_path))
        path = monitor / 'terminal_cache_5080.json'
        f.write_new(path, out)
        f.write_new(monitor / 'terminal_pointer_5080.json', {'sha256': f.sha(path)})
        f.write_new(self.root / 'deployment/launch_receipt_5080.json', {'actual_launch': True, 'contract_sha256': f.CONTRACT, 'root': self.contract['hosts']['5080']['root'], 'gpu_uuid': self.contract['hosts']['5080']['gpu_uuid']})
        return out

    def test_actual_workers_dispatch_completion_and_owner_absence_required(self):
        snapshot = self.snapshot()
        self.assertEqual(len(f.verify_terminal_snapshot(self.contract, '5080', snapshot)), 3)
        mutations = [lambda x: x['processes'].append({'pid': 7}), lambda x: x.update(gpu_processes='GPU,7'), lambda x: x['supervisor_exit.json'].update(returncode=1), lambda x: x['pipeline_complete.json'].update(completed_conditions=12), lambda x: x['pipeline_heartbeat.json'].update(phase='dispatch'), lambda x: next(iter(x['jobs'].values()))['worker_exit'].update(actual_child_exit_observed=False), lambda x: x['pipeline_complete.json'].update(supervisor_exit_sha256='c' * 64)]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                bad = copy.deepcopy(snapshot)
                mutate(bad)
                with self.assertRaises(ValueError):
                    f.verify_terminal_snapshot(self.contract, '5080', bad)

    def test_terminal_cache_keeps_actual_original_observation(self):
        self.cache(self.snapshot())
        evidence = f.terminal_cache(self.contract, '5080', self.root)
        self.assertEqual(evidence['observed_unix'], 123)
        self.assertFalse(evidence['pipeline_exit_code_directly_observed'])

    def test_terminal_cache_raw_sha_tamper_rejected(self):
        self.cache(self.snapshot())
        (self.root / 'hourly_monitor/snapshot.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'snapshot SHA'):
            f.terminal_cache(self.contract, '5080', self.root)

    def test_missing_pipeline_raw_receipt_remains_pending_without_remote(self):
        snapshot = self.snapshot()
        snapshot.pop('raw_files_sha256')
        self.cache(snapshot)
        self.assertIsNone(f.terminal_cache(self.contract, '5080', self.root))
        remote = mock.Mock(side_effect=AssertionError('Must not call SSH'))
        result = f.recover_host(self.contract, '5080', self.root, remote=remote)
        self.assertEqual((result['status'], result['remote_calls']), ('pending', 0))
        remote.assert_not_called()

    def test_unstarted_host_pending_remote0_and_no_claim(self):
        remote = mock.Mock(side_effect=AssertionError('Must not call SSH'))
        result = f.recover_host(self.contract, '5080', self.root, remote=remote)
        self.assertEqual(result['remote_calls'], 0)
        self.assertFalse((self.root / 'originals/5080/claim.json').exists())
        remote.assert_not_called()

    def test_preserved_failure_or_claim_never_retried(self):
        f.write_new(self.root / 'originals/5090/claim.json', {'no_retry': True})
        remote = mock.Mock(side_effect=AssertionError('Must not retry'))
        result = f.recover_host(self.contract, '5090', self.root, remote=remote)
        self.assertEqual(result['status'], 'claimed_or_failed_preserved')
        self.assertTrue(result['user_action_required'])
        remote.assert_not_called()

    def test_launch_object_binding_accepts_writer_format_difference_only(self):
        launch = {'host': '5090', 'root': '/owned/5090', 'actual_launch': True, 'contract_sha256': f.CONTRACT, 'launched_unix': 123.4}
        path = self.root / 'deployment/launch_receipt.json'
        path.parent.mkdir()
        path.write_text(json.dumps(launch))  # Native remote json.dump formatting.
        local_raw = json.dumps(launch, indent=2).encode()
        self.assertNotEqual(f.sha(path), digest(local_raw))
        f.write_new(self.root / 'start_permit.json', {'contract_sha256': f.CONTRACT})
        request = {'bindings': {'start_permit.json': f.sha(self.root / 'start_permit.json')}, 'object_bindings': {'deployment/launch_receipt.json': launch}}
        f.verify_root_bindings(self.root, request)
        path.write_text(json.dumps({**launch, 'launched_unix': 124}))
        with self.assertRaisesRegex(ValueError, 'object binding'):
            f.verify_root_bindings(self.root, request)

    def test_copied_start_permit_keeps_raw_binding(self):
        f.write_new(self.root / 'start_permit.json', {'contract_sha256': f.CONTRACT})
        request = {'bindings': {'start_permit.json': f.sha(self.root / 'start_permit.json')}, 'object_bindings': {}}
        f.verify_root_bindings(self.root, request)
        (self.root / 'start_permit.json').write_text(json.dumps({'contract_sha256': f.CONTRACT}))
        with self.assertRaisesRegex(ValueError, 'raw SHA'):
            f.verify_root_bindings(self.root, request)

    def test_failed_export_retains_exclusive_claim_and_no_second_call(self):
        self.cache(self.snapshot())
        remote = mock.Mock(side_effect=RuntimeError('Disconnected'))
        with mock.patch.object(f, 'export_code', return_value='stdlib source'):
            with self.assertRaisesRegex(RuntimeError, 'Disconnected'):
                f.recover_host(self.contract, '5080', self.root, remote=remote)
        failure = f.read(self.root / 'originals/5080/failure.json')
        self.assertEqual(failure['remote_calls'], 1)
        self.assertFalse(failure['automatic_retry'])
        self.assertEqual(f.recover_host(self.contract, '5080', self.root, remote=remote)['remote_calls'], 0)
        self.assertEqual(remote.call_count, 1)

    def archive(self, *, extra=None, duplicate=False, unsafe=False, wrong_terminal=False):
        job = {'id': 'd__42__full', 'host': '5080', 'arm': 'full', 'dataset': 'd', 'seed': 42}
        c = {'jobs': [job], 'hosts': {'5080': {'root': '/owned/5080', 'gpu_uuid': 'GPU1'}}, 'source': {'files': {}}, 'input_files': {}, 'operation': {'files': {}}}
        terminal = {'job': job, 'contract_sha256': f.canonical(c), 'scientific_success': True, 'status': 'complete', 'held_out_test_evaluated': False, 'files': {}}
        if wrong_terminal:
            terminal['job'] = {**job, 'seed': 52}
        files = {'execution_contract.json': encoded(c), 'run/' + job['id'] + '/terminal_manifest.json': encoded(terminal)}
        if extra:
            files[extra] = b'forbidden'
        filemap = {n: {'sha256': digest(raw), 'size': len(raw)} for n, raw in files.items()}
        terminals = {job['id']: digest(encoded(terminal))}
        manifest = {'schema': 'titantpp_rtx_history_originals_v1', 'contract_sha256': f.canonical(c), 'host': '5080', 'root': '/owned/5080', 'gpu_uuid': 'GPU1', 'files': filemap, 'terminal_manifest_sha256': terminals, 'scientific_success': True, 'held_out_test_evaluated': False, 'actual_idle_observation': {'owned_processes': [], 'gpu_pids': []}}
        raw_manifest = encoded(manifest)
        archive = self.root / 'archive.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            data = {'manifest.json': raw_manifest, **{'originals/' + n: raw for n, raw in files.items()}}
            for name, raw in data.items():
                info = tarfile.TarInfo(name)
                info.size = len(raw)
                tar.addfile(info, io.BytesIO(raw))
            if duplicate:
                info = tarfile.TarInfo('manifest.json')
                info.size = len(raw_manifest)
                tar.addfile(info, io.BytesIO(raw_manifest))
            if unsafe:
                info = tarfile.TarInfo('../escape')
                info.size = 1
                tar.addfile(info, io.BytesIO(b'x'))
        receipt = {'archive_sha256': f.sha(archive), 'archive_size': archive.stat().st_size, 'manifest_sha256': digest(raw_manifest), 'contract_sha256': f.canonical(c), 'host': '5080', 'root': '/owned/5080', 'held_out_test_evaluated': False, 'files': len(files), 'terminal_manifest_sha256': terminals}
        return archive, receipt, c

    def test_strict_archive_roundtrip_and_binary_raw_sha(self):
        archive, receipt, c = self.archive()
        dest = self.root / 'verified'
        m = f.verify_archive(archive, dest, receipt, c, '5080')
        f.verify_original_files(dest / 'originals', m)
        (dest / 'originals/execution_contract.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Local original'):
            f.verify_original_files(dest / 'originals', m)

    def test_archive_duplicate_traversal_and_heldout_rejected_before_extract(self):
        for options in ({'duplicate': True}, {'unsafe': True}, {'extra': 'run/heldout_predictions.json'}, {'wrong_terminal': True}):
            with self.subTest(options=options):
                archive, receipt, c = self.archive(**options)
                with self.assertRaises(ValueError):
                    f.verify_archive(archive, self.root / 'verified', receipt, c, '5080')
                self.assertFalse((self.root / 'verified').exists())

    def records(self):
        records = []
        for d in ('Taxi', 'RAF', 'Intermittent'):
            for seed in f.SEEDS:
                for index, arm in enumerate(f.ARMS):
                    rmse = 1.0 + index
                    endpoint = {'qty_rmse': rmse, 'qty_mae': 2.0, 'time_nll': 4.0 if index == 0 else 1.0, 'tail': {'qty_rmse': rmse * 3, 'qty_mae': 5.0, 'time_nll': 6.0 if index == 0 else 2.0}}
                    exposure = {'train': [{'epoch': 1, 'batch_order_sha256': d + str(seed)}], 'validation': [{'batch_order_sha256': 'validation'}]}
                    records.append({'id': d + str(seed) + arm, 'dataset': d, 'seed': seed, 'arm': arm, 'selected': endpoint, 'train_exposure': exposure, 'provenance': {'RAF42_physical_GPU_migration': d == 'RAF' and seed == 42, 'owner': 'GPU'}, 'best_epoch': 1, 'last_epoch': 41})
        return records

    def test_q1_requires_all3seed_both_controls_same2datasets(self):
        records = self.records()
        result = f.summarize(records)
        self.assertTrue(result['decision']['Q1_development_support'])
        # One losing seed makes that entire dataset ineligible, even if mean wins.
        for d in ('Taxi', 'RAF'):
            next(r for r in records if (r['dataset'], r['seed'], r['arm']) == (d, 62, f.ARMS[0]))['selected']['qty_rmse'] = 9
        result = f.summarize(records)
        self.assertFalse(result['decision']['Q1_development_support'])
        self.assertEqual(result['decision']['same_datasets_all3seed_strict_RMSE_wins'], ['Intermittent'])

    def test_ties_are_not_strict_wins_and_time_penalty_preserved(self):
        records = self.records()
        for d in ('Taxi', 'RAF'):
            next(r for r in records if (r['dataset'], r['seed'], r['arm']) == (d, 42, f.ARMS[0]))['selected']['qty_rmse'] = 2
        result = f.summarize(records)
        self.assertFalse(result['decision']['Q1_development_support'])
        differences = result['decision']['paired_results'][0]['comparisons'][f.ARMS[1]]
        self.assertEqual(differences['overall_time_nll_delta'], 3)
        self.assertEqual(differences['tail_time_nll_delta'], 4)
        self.assertIn('pipeline_exit_code_directly_observed=false', f.report_text(records, result))

    def test_duplicate_execution_and_paired_exposure_drift_rejected(self):
        records = self.records()
        records[1]['id'] = records[0]['id']
        with self.assertRaisesRegex(ValueError, 'distinct'):
            f.summarize(records)
        records = self.records()
        records[1]['train_exposure']['train'][0]['batch_order_sha256'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'Train order'):
            f.summarize(records)

    def test_mean_and_sample_sd_include_unfavorable_seed(self):
        records = self.records()
        group = [r for r in records if r['dataset'] == 'Taxi' and r['arm'] == f.ARMS[0]]
        for row, value in zip(group, (1., 2., 6.)):
            row['selected']['qty_rmse'] = value
        summary = next(r for r in f.summarize(records)['seed_summary'] if r['dataset'] == 'Taxi' and r['arm'] == f.ARMS[0])
        self.assertEqual(summary['metrics']['overall_qty_rmse']['mean'], 3)
        self.assertAlmostEqual(summary['metrics']['overall_qty_rmse']['sample_sd'], 7**0.5)

    def selected_fixture(self, *, best_epoch=1, changed_time=False):
        job = {'id': 'd__42__full', 'dataset': 'd', 'host': '5080', 'arm': f.ARMS[0], 'seed': 42}
        c = {'training': {'minimum_epochs': 2, 'maximum_epochs': 3}, 'datasets': [{'dataset_id': 'd', 'inherited_data_identity': {'populations': {'validation': {'target_count': 10}}}, 'quantity_boundaries_all_train_rows': [2]}]}
        folder = self.root / 'fit'
        run = folder / 'runs' / job['arm'] / f.VARIANT / 'seed_42'
        remote_folder = '/owned/run/' + job['id']
        remote_run = remote_folder + '/runs/' + job['arm'] + '/' + f.VARIANT + '/seed_42/'
        history = [{'epoch': i, 'val_qty_rmse': 1 if i < 3 else 2, 'val_qty_mae': 1, 'val_time_nll': 4} for i in (1, 2, 3)]
        endpoint = {'qty_rmse': 1, 'qty_mae': 1, 'time_nll': 4, 'count': 10, 'tail': {'qty_rmse': 3, 'qty_mae': 2, 'time_nll': 7, 'count': 2}, 'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False, 'quantity_boundaries': [2]}
        selected = {**endpoint, 'checkpoint_path': remote_run + 'best.pt'}
        last = {**endpoint, 'qty_rmse': 2, 'checkpoint_path': remote_run + 'last.pt'}
        if changed_time:
            selected['time_nll'] = 0
        endpoints = {'job': job, 'selected': selected, 'last': last, 'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False, 'completed_epochs': 3, 'best_epoch': best_epoch, 'initial_state_sha256': 'init', 'parameter_count': 32}
        f.write_new(folder / 'initialization.json', {job['arm']: 'init'})
        f.write_new(run / 'history.json', {'history': history})
        f.write_new(run / 'endpoint_replays.json', endpoints)
        f.write_new(run / 'exposure.json', {'train': [{'epoch': i} for i in (1, 2, 3)], 'validation': [{'epoch': i} for i in (0, 1, 2, 3)]})
        f.write_new(run / 'epoch_timing.json', {'epochs': [{'epoch': i, 'elapsed_seconds': 1} for i in (1, 2, 3)]})
        for name in ('best.pt', 'last.pt'):
            (run / name).write_bytes(b'opaque binary original, no model loaded')
        files = {p.relative_to(folder).as_posix(): f.sha(p) for p in folder.rglob('*') if p.is_file()}
        f.write_new(folder / 'terminal_manifest.json', {'job': job, 'contract_sha256': f.canonical(c), 'status': 'complete', 'scientific_success': True, 'held_out_test_evaluated': False, 'files': files})
        return (folder, job, c, f.canonical(c), remote_folder, f.sha(folder / 'terminal_manifest.json'), {'owner': 'synthetic'})

    def test_selected_epoch_is_earliest_finite_minimum(self):
        fixture = self.selected_fixture()
        record = f.selected_record(*fixture)
        self.assertEqual(record['best_epoch'], 1)
        self.assertEqual(record['binary_originals']['selected']['raw_sha256'], digest(b'opaque binary original, no model loaded'))

    def test_later_tie_or_mixed_epoch_time_metric_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Earliest strict'):
            f.selected_record(*self.selected_fixture(best_epoch=2))
        # A separate fixture changes timing while leaving quantity unchanged.
        self.root = self.root / 'second'
        self.root.mkdir()
        with self.assertRaisesRegex(ValueError, 'Same-epoch'):
            f.selected_record(*self.selected_fixture(changed_time=True))

    def test_all_pending_never_collects_parent_or_creates_analysis(self):
        with mock.patch.object(f, 'parent_records', side_effect=AssertionError('not ready')):
            result = f.finalize(bundle=self.root, verifier=lambda _: self.contract)
        self.assertEqual(result['status'], 'pending')
        self.assertEqual(result['remote_calls'], 0)
        self.assertFalse((self.root / 'analysis').exists())

    def test_local_parent15_and_resume_continuity_real_originals(self):
        # Read-only Validation originals; no GPU, SSH, Pod API, or model loading.
        records = f.parent_records(self.contract)
        self.assertEqual(len(records), 15)
        self.assertEqual(len({r['id'] for r in records}), 15)
        resumed = [r for r in records if 'resume_from_epoch' in r['provenance']]
        self.assertEqual(sorted(r['provenance']['resume_from_epoch'] for r in resumed), [23, 46])
        self.assertTrue(all(len(r['provenance']['timing_segments']) == 2 for r in resumed))


if __name__ == '__main__':
    unittest.main()
