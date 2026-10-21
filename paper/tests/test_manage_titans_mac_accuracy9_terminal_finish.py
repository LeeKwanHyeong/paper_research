"""The local seal must reject changed bytes before any remote action."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from paper.scripts import manage_titans_mac_accuracy9_terminal_finish as m


class LocalSealTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.root, self.report, self.bundle = root, root / 'report', root / 'bundle'
        self.report.mkdir()
        self.bundle.mkdir()
        self.diag = root / 'diagnosis'
        self.diag.mkdir()
        for name in m.SOURCE + m.TESTS:
            p = root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name)
        (self.bundle / 'deployment.tar.gz').write_bytes(b'archive')
        m.write(self.bundle / 'AUTHORIZATION.json', {'authorized': True})
        m.write(self.report / 'INDEPENDENT_QA.json', {'status': 'PASS'})
        self.c = {'root': 'upstream'}
        self.f = {'root': m.REMOTE, 'upstream_root': 'upstream',
                  'upstream_contract_sha256': m.canonical(self.c),
                  'operation_files': {'source/' + Path(n).name: m.sha(root / n) for n in m.SOURCE[-2:]},
                  'local_control_files': {n: m.sha(root / n) for n in m.SOURCE + m.TESTS},
                  'independent_qa_sha256': m.sha(self.report / 'INDEPENDENT_QA.json'),
                  'authorization': {'sha256': m.sha(self.bundle / 'AUTHORIZATION.json')},
                  'upstream_evidence_files': {}}
        self.expected = m.canonical(self.f)
        m.write(self.bundle / 'finish_contract.json', self.f)
        self.b = {'contract_sha256': self.expected,
                  'contract_raw_sha256': m.sha(self.bundle / 'finish_contract.json'),
                  'deployment_archive_sha256': m.sha(self.bundle / 'deployment.tar.gz'),
                  'authorization_sha256': self.f['authorization']['sha256'],
                  'source_files': self.f['local_control_files'],
                  'independent_qa_sha256': self.f['independent_qa_sha256']}
        m.write(self.report / 'CONTROL_BINDINGS.json', self.b)
        for name, value in (('ROOT', root), ('REPORT', self.report), ('BUNDLE', self.bundle), ('DIAGNOSIS', self.diag)):
            p = patch.object(m, name, value)
            p.start()
            self.addCleanup(p.stop)
        p = patch.object(m, 'original_local_checks', return_value=self.c)
        p.start()
        self.addCleanup(p.stop)

    def test_full_local_seal_passes_without_transport(self):
        with patch.object(m, 'transport', side_effect=AssertionError('remote forbidden')):
            f, report, bundle = m.verify_local(self.expected)
        self.assertEqual(f, self.f)
        self.assertEqual((report, bundle), (self.report, self.bundle))

    def test_changed_collector_and_modified_unpinned_binding_still_rejected(self):
        p = self.root / m.SOURCE[1]
        p.write_text('changed collector')
        self.b['source_files'] = {**self.b['source_files'], m.SOURCE[1]: m.sha(p)}
        m.write(self.report / 'CONTROL_BINDINGS.json', self.b, exclusive=False)
        with self.assertRaisesRegex(ValueError, 'control closure'):
            m.verify_local(self.expected)

    def test_changed_archive_rejected(self):
        (self.bundle / 'deployment.tar.gz').write_bytes(b'partial')
        with self.assertRaisesRegex(ValueError, 'seal changed'):
            m.verify_local(self.expected)

    def test_changed_authorization_rejected(self):
        (self.bundle / 'AUTHORIZATION.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'seal changed'):
            m.verify_local(self.expected)

    def test_changed_qa_rejected(self):
        (self.report / 'INDEPENDENT_QA.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'seal changed'):
            m.verify_local(self.expected)

    def test_wrong_contract_blocks_deploy_before_attempt_and_remote(self):
        with patch.object(m, 'transport', side_effect=AssertionError('remote forbidden')):
            with self.assertRaisesRegex(ValueError, 'scope changed'):
                m.deploy('0' * 64)
        self.assertFalse((self.report / 'DEPLOYMENT_ATTEMPT.json').exists())

    def test_existing_attempt_blocks_second_deploy_before_remote(self):
        m.write(self.report / 'DEPLOYMENT_ATTEMPT.json', {'prior': 'preserved'})
        with patch.object(m, 'transport', side_effect=AssertionError('remote forbidden')):
            with self.assertRaises(FileExistsError):
                m.deploy(self.expected)
        self.assertEqual(m.read(self.report / 'DEPLOYMENT_ATTEMPT.json'), {'prior': 'preserved'})


if __name__ == '__main__':
    unittest.main()
