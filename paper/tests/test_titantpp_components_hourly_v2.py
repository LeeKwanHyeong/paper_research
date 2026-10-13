"""CPU-only hourly observer regressions; remote.call is a local fake."""
import copy
from pathlib import Path
import unittest
from unittest.mock import patch
from paper.tests import test_titantpp_components_terminal_verifier as fixtures
from paper.scripts import launch_titantpp_cnn_gru_components_v5 as base
from paper.scripts import observe_titantpp_components_hourly_v2 as observer
from paper.scripts import titantpp_components_terminal_verifier as verifier


class FakeRemote:
    def __init__(self, snapshot=None, error=None):
        self.snapshot=snapshot;self.error=error;self.calls=[]
    def call(self, host, action, payload):
        self.calls.append((host,action))
        if self.error:raise self.error
        return copy.deepcopy(self.snapshot)


class ObserverTests(unittest.TestCase):
    setUpClass=classmethod(fixtures.CacheTests.setUpClass.__func__)
    setUp=fixtures.CacheTests.setUp

    def training_intent(self):
        path=self.bundle/'control/intents/train_5080.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text('{}')

    def test_seed_and_repeated_observation_cache_make_zero_remote_calls(self):
        verifier.seed_terminal_cache(self.bundle,self.output,'5080',self.path)
        remote=FakeRemote(error=AssertionError('No SSH permitted'))
        for _ in range(2):
            report=observer.observe_host(self.bundle,self.output,self.c,'5080',remote=remote)
            self.assertEqual(report['remote_calls'],0)
            self.assertEqual((report['completed'],report['total']),(24,24))
            self.assertEqual(report['observed_unix'],1791847640.420401)
            self.assertEqual(report['snapshot_raw_sha256'],base.sha(self.path))
            self.assertEqual(len(report['jobs']),24)
            self.assertTrue(all(j['eta']['status']=='complete' for j in report['jobs']))
        self.assertEqual(remote.calls,[])

    def test_local_snapshot_only_seeds_terminal_with_no_remote_calls(self):
        remote=FakeRemote(error=AssertionError('No SSH permitted'))
        report=observer.observe_host(self.bundle,self.output,self.c,'5080',local_only=True,remote=remote)
        self.assertEqual(report['status'],verifier.TERMINAL)
        self.assertEqual(report['remote_calls'],0)
        self.assertTrue((self.output/'terminal_cache/5080.json').exists())
        self.assertEqual(remote.calls,[])

    def test_exactly_one_status_call_then_cache(self):
        self.training_intent();remote=FakeRemote(self.snapshot)
        report=observer.observe_host(self.bundle,self.output,self.c,'5080',remote=remote)
        self.assertEqual((report['status'],report['remote_calls']),(verifier.TERMINAL,1))
        self.assertEqual(remote.calls,[('5080','status')])
        again=observer.observe_host(self.bundle,self.output,self.c,'5080',remote=remote)
        self.assertEqual(again['remote_calls'],0)
        self.assertEqual(remote.calls,[('5080','status')])

    def test_error_report_keeps_raw_snapshot_metadata_without_fit_failure_inference(self):
        self.training_intent();snapshot=copy.deepcopy(self.snapshot)
        snapshot['terminal_hashes'][self.job['id']]['all_files_verified']=False
        remote=FakeRemote(snapshot)
        report=observer.observe_host(self.bundle,self.output,self.c,'5080',remote=remote)
        self.assertEqual(report['status'],'training_identity_unconfirmed')
        self.assertEqual(report['remote_calls'],1)
        self.assertFalse(report['training_failure_inferred'])
        self.assertEqual(report['snapshot_raw_sha256'],base.sha(Path(report['snapshot_path'])))
        self.assertEqual(report['observed_unix'],snapshot['observed_unix'])
        self.assertTrue(report['observed_KST'])
        self.assertFalse((self.output/'terminal_cache/5080.json').exists())

    def test_query_failure_is_not_training_failure_or_retried(self):
        self.training_intent();remote=FakeRemote(error=OSError('read unavailable'))
        report=observer.observe_host(self.bundle,self.output,self.c,'5080',remote=remote)
        self.assertEqual(report['status'],'query_failed_or_identity_unconfirmed')
        self.assertFalse(report['training_failure_inferred'])
        self.assertEqual(remote.calls,[('5080','status')])

    def test_cache_write_error_preserves_one_call_and_snapshot_metadata(self):
        self.training_intent();remote=FakeRemote(self.snapshot)
        with patch.object(observer,'seed_terminal_cache',side_effect=OSError('cache unavailable')):
            report=observer.observe_host(self.bundle,self.output,self.c,'5080',remote=remote)
        self.assertEqual(report['status'],'training_identity_unconfirmed')
        self.assertEqual(report['remote_calls'],1)
        self.assertEqual(report['snapshot_raw_sha256'],base.sha(Path(report['snapshot_path'])))
        self.assertEqual(report['observed_unix'],1791847640.420401)
        self.assertFalse(report['training_failure_inferred'])
        self.assertEqual(remote.calls,[('5080','status')])

    def test_qualification_cache_before_train_is_not_counted_as_fit(self):
        remote=FakeRemote(error=AssertionError('No SSH permitted'))
        report=observer.observe_host(self.bundle,self.output,self.c,'5080',remote=remote)
        self.assertEqual((report['status'],report['completed'],report['remote_calls']),('qualification_passed_cached',0,0))
        self.assertEqual(remote.calls,[])

    def test_markdown_includes_final_saved_best_for_terminal_rows(self):
        report=observer.observe_host(self.bundle,self.output,self.c,'5080',local_only=True)
        result={'hosts':{'5080':report,'5090':{**report,'total':12}},'all_training_terminal_verified':False}
        text=observer.markdown(result)
        self.assertIn('87 / 47 (완료)',text)
        self.assertIn('과거 종료 관측',text)


if __name__=='__main__':unittest.main()
