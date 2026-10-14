"""Synthetic, offline tests for one-shot cap-fit original collection."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from paper.scripts import collect_titantpp_gru_cap_validation as collector


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='cap-collect-', dir='/private/tmp')
        self.root = Path(self.temp.name)
        self.archive = self.root / 'source.tar.gz'

    def tearDown(self):
        self.temp.cleanup()

    def archive_of(self, rows):
        with tarfile.open(self.archive, 'w:gz') as out:
            for name, raw, kind in rows:
                item = tarfile.TarInfo(name)
                item.type = kind
                if kind == tarfile.REGTYPE:
                    item.size = len(raw)
                    out.addfile(item, io.BytesIO(raw))
                else:
                    item.linkname = '/etc/passwd'
                    out.addfile(item)

    def test_exact_members_and_sha_verify_before_extraction(self):
        data = {'job/a.json': b'{}', 'job/run/model.pt': b'fixed binary'}
        self.archive_of([(name, raw, tarfile.REGTYPE) for name, raw in data.items()])
        expected = {name: digest(raw) for name, raw in data.items()}
        self.assertEqual(collector.verify_archive(self.archive, expected)['file_count'], 2)
        target = self.root / 'originals'
        collector.extract_verified(self.archive, expected, target)
        self.assertEqual((target / 'job/run/model.pt').read_bytes(), b'fixed binary')
        with self.assertRaises(FileExistsError):
            collector.extract_verified(self.archive, expected, target)

    def test_changed_member_leaves_no_extracted_tree(self):
        self.archive_of([('job/a.json', b'bad', tarfile.REGTYPE)])
        target = self.root / 'originals'
        with self.assertRaisesRegex(ValueError, 'SHA differs'):
            collector.extract_verified(self.archive, {'job/a.json': digest(b'good')}, target)
        self.assertFalse(target.exists())

    def test_traversal_symlink_hardlink_and_duplicate_rejected(self):
        examples = [('../outside', tarfile.REGTYPE), ('/outside', tarfile.REGTYPE),
                    ('job/link', tarfile.SYMTYPE), ('job/hard', tarfile.LNKTYPE)]
        for name, kind in examples:
            with self.subTest(name=name):
                self.archive_of([(name, b'a', kind)])
                with self.assertRaises(ValueError):
                    collector.verify_archive(self.archive, {name: digest(b'a')})
        self.archive_of([('job/a', b'a', tarfile.REGTYPE)] * 2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            collector.verify_archive(self.archive, {'job/a': digest(b'a')})

    def test_missing_and_unexpected_members_rejected(self):
        self.archive_of([('job/a', b'a', tarfile.REGTYPE)])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            collector.verify_archive(self.archive, {'job/a': digest(b'a'), 'job/b': digest(b'b')})
        with self.assertRaisesRegex(ValueError, 'Unexpected'):
            collector.verify_archive(self.archive, {'job/b': digest(b'b')})

    def test_existing_local_symlink_is_rejected(self):
        self.archive_of([('job/a', b'a', tarfile.REGTYPE)])
        target = self.root / 'originals'
        target.mkdir()
        (target / 'job').symlink_to(self.root)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            collector.extract_verified(self.archive, {'job/a': digest(b'a')}, target)

    def test_waiting_has_zero_remote_and_no_collection_writes(self):
        acquire = Mock()
        with patch.object(collector, 'completion_gate', return_value={'status': 'waiting_for_terminal_training',
                'remote_calls': 0, 'original_retrieval': 0, 'Test': 0}):
            result = collector.collect(self.root / 'round.json', 'a' * 64, 'b' * 64,
                bundle=self.root, monitor=self.root, report=self.root, acquire=acquire)
        self.assertEqual(result['remote_calls'], 0)
        acquire.assert_not_called()
        self.assertFalse((self.root / 'retrieval').exists())

    def test_incomplete_intent_blocks_retry_before_transport(self):
        (self.root / 'retrieval').mkdir()
        acquire = Mock()
        with patch.object(collector, 'completion_gate', return_value={'status': 'terminal_verified'}):
            with self.assertRaisesRegex(ValueError, 'no automatic retry'):
                collector.collect(self.root / 'round.json', 'a' * 64, 'b' * 64,
                    bundle=self.root, monitor=self.root, report=self.root, acquire=acquire)
        acquire.assert_not_called()

    def test_transport_failure_preserves_intent_failure_and_does_not_try_second_host(self):
        acquire = Mock(side_effect=RuntimeError('synthetic SSH error'))
        gate = {'status': 'terminal_verified', 'contract': {'hosts': {'5080': {}}}, 'expected': {'5080': {}}}
        with patch.object(collector, 'completion_gate', return_value=gate):
            with self.assertRaisesRegex(RuntimeError, 'synthetic SSH error'):
                collector.collect(self.root / 'round.json', 'a' * 64, 'b' * 64,
                    bundle=self.root, monitor=self.root, report=self.root, acquire=acquire)
        self.assertEqual(acquire.call_count, 1)
        self.assertTrue((self.root / 'retrieval/intent.json').exists())
        self.assertEqual(collector.read(self.root / 'retrieval/failure.json')['status'], 'failed_no_automatic_retry')
        self.assertFalse((self.root / 'retrieval/originals').exists())

    def test_terminal_manifest_binds_exact_job_scope_and_file_sha(self):
        job = {'id': 'cap', 'host': '5080'}
        c = {'jobs': [job]}
        item = {'status': 'complete', 'scientific_success': True, 'job': job,
                'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
                'contract_sha256': collector.canonical(c), 'files': {'runs/model.pt': digest(b'x')}}
        launcher = Mock()
        raw = json.dumps(item).encode()
        launcher.decode_snapshot.return_value = {'run/cap/terminal_manifest.json': raw}
        expected = collector.terminal_files(c, '5080', {}, launcher)
        self.assertEqual(expected, {'cap/terminal_manifest.json': digest(raw), 'cap/runs/model.pt': digest(b'x')})
        item['files'] = {'test_metrics.json': digest(b'x')}
        launcher.decode_snapshot.return_value = {'run/cap/terminal_manifest.json': json.dumps(item).encode()}
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            collector.terminal_files(c, '5080', {}, launcher)

    def test_pinned_helper_and_monitor_dependencies_must_all_match(self):
        bundle, monitor = self.root / 'bundle', self.root / 'monitor'
        bundle.mkdir(); monitor.mkdir()
        contract_path = bundle / 'execution_contract.json'
        contract_path.write_text('{}')
        files = {}
        names = ['paper/scripts/' + name + '.py' for name in (
            'collect_titantpp_gru_cap_validation', 'summarize_titantpp_gru_cap_validation',
            'observe_titantpp_gru_cap_validation', 'launch_titantpp_gru_cap_validation')]
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# fabricated helper')
            files[name] = collector.sha(path)
        collector.unique_write(monitor / 'monitor_integrity.json', {'files': files.copy()})
        monitor_sha = collector.sha(monitor / 'monitor_integrity.json')
        files['monitor/monitor_integrity.json'] = monitor_sha
        seal = {'schema': 'titantpp_gru_cap_collection_integrity_v1', 'files': files,
            'files_sha256': collector.canonical(files), 'monitor_integrity_raw_sha256': monitor_sha,
            'contract_raw_sha256': collector.sha(contract_path)}
        collector.unique_write(bundle / 'collection_integrity.json', seal)
        pin = collector.sha(bundle / 'collection_integrity.json')
        with patch.object(collector, 'PROJECT', self.root):
            self.assertEqual(collector.verify_seal(bundle, monitor, pin), seal)
            (self.root / names[1]).write_text('# changed helper')
            with self.assertRaisesRegex(ValueError, 'dependency SHA changed'):
                collector.verify_seal(bundle, monitor, pin)
            with self.assertRaisesRegex(ValueError, 'integrity SHA changed'):
                collector.verify_seal(bundle, monitor, 'f' * 64)

    def test_qualification_failure_round_waits_without_reading_remote_or_caches(self):
        from paper.scripts import observe_titantpp_gru_cap_validation as observer
        from paper.scripts import launch_titantpp_gru_cap_validation as launch
        from paper.tests.test_summarize_titantpp_gru_cap_validation import CapValidationSummaryTests
        fixture = CapValidationSummaryTests()
        fixture.setUp()
        try:
            c = fixture.contract
            c['evaluation_scope'] = 'validation_only'
            for job in c['jobs']:
                if job['dataset'] == 'intermittent_frozen_5000' and job['seed'] == 62:
                    job['host'] = '5080'
            collector.unique_write(self.root / 'execution_contract.json', c)
            reports = {host: {'status': 'query_or_evidence_unconfirmed', 'user_action_required': True,
                'jobs': []} for host in collector.HOSTS}
            counts = observer.unique_counts(c, reports)
            result = {'schema': 'titantpp_gru_cap_validation_monitor_v1', 'contract_sha256': collector.canonical(c),
                'monitor_integrity_raw_sha256': 'b' * 64, 'held_out_test_evaluated': False,
                'hosts': reports, 'counts': counts, 'all_training_terminal_verified': False}
            round_path = self.root / 'rounds/a.json'
            collector.unique_write(round_path, result)
            with patch.object(collector, 'verify_seal', return_value={'monitor_integrity_raw_sha256': 'b' * 64}), \
                 patch.object(observer, 'verify_monitor'), patch.object(launch, 'check_bundle', return_value=(c, {})), \
                 patch.object(observer, 'load_cache') as cache:
                gate = collector.completion_gate(self.root, self.root, round_path,
                    collector.sha(round_path), 'a' * 64)
            self.assertEqual(gate['status'], 'waiting_for_terminal_training')
            self.assertEqual(gate['remote_calls'], 0)
            cache.assert_not_called()
        finally:
            fixture.tearDown()

    def test_native_normalization_integrates_with_nine_job_summary(self):
        # Fabricated JSON/binary bytes only; the fixture never reads real predictions.
        from paper.tests.test_summarize_titantpp_gru_cap_validation import CapValidationSummaryTests
        fixture = CapValidationSummaryTests()
        fixture.setUp()
        try:
            for job in fixture.jobs:
                (fixture.retrieval / job['id'] / 'derived_validation_replay.json').unlink()
            index = fixture.root / 'collected_index.json'
            result = collector.normalize(fixture.contract_path, fixture.retrieval, index)
            self.assertEqual(len(result['jobs']), 9)
            from paper.scripts import summarize_titantpp_gru_cap_validation as summary
            verified = summary.summarize(fixture.contract_path, fixture.retrieval, index)
            self.assertEqual(verified['status'], 'complete_validation_only')
            self.assertEqual(verified['baseline_reference_count'], 18)
        finally:
            fixture.tearDown()


if __name__ == '__main__':
    unittest.main()
