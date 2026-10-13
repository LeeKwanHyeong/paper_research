"""Offline regressions against pinned actual Train/Validation status snapshots."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from paper.scripts import launch_titantpp_cnn_gru_components_v5 as base
from paper.scripts import titantpp_components_terminal_verifier as verifier

BUNDLE = base.PROJECT/'search_artifacts/titantpp_cnn_gru_components_20261012_v5'
PIN_5080 = '352b65c7f5c55a11fae7720b67c26c437b6999f662cfa0b8297fe0b2c33556f2'
PIN_5090 = '71dbc280f6581c8b193a6799cac04b8420905d34987312885cde677377ea828a'


def replace_file(snapshot, name, value):
    raw=(json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    snapshot['files'][name]={'base64':base64.b64encode(raw).decode(), 'raw_sha256':hashlib.sha256(raw).hexdigest()}


class ActualTerminalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=base.read(BUNDLE/'execution_contract.json')
        cls.paths={h:next((BUNDLE/'control/status'/h).glob(prefix)) for h,prefix in [('5080','1ecf*.json'),('5090','f3d*.json')]}
        for host,pin in [('5080',PIN_5080),('5090',PIN_5090)]:
            assert base.sha(cls.paths[host])==pin
        cls.snapshots={h:base.read(p) for h,p in cls.paths.items()}

    def setUp(self):
        self.snapshot=copy.deepcopy(self.snapshots['5080'])
        self.job=self.c['jobs'][0]

    def test_original_bug_is_reproduced(self):
        with self.assertRaisesRegex(ValueError, 'Terminal completion'):
            base.training_status(self.snapshot,self.c,'5080')

    def test_actual_final_manifest_overrides_stale_live_checkpoint(self):
        report=verifier.training_status(self.snapshot,self.c,'5080')
        self.assertEqual((report['status'],report['completed'],report['total']), (verifier.TERMINAL,24,24))
        self.assertEqual((report['jobs'][0]['saved_epoch'],report['jobs'][0]['best_epoch']), (87,47))
        self.assertFalse(self.snapshot['checkpoints'][self.job['id']]['snapshot_complete'])
        self.assertFalse(report['pipeline_os_exit_directly_observed'])

    def test_actual_live_host_keeps_completed_partial_and_owner_validation(self):
        report=verifier.training_status(self.snapshots['5090'],self.c,'5090')
        self.assertEqual((report['status'],report['completed'],report['total']),('training_running',3,12))
        self.assertTrue(report['identity_verified'])

    def test_remote_false_file_witness_rejected(self):
        witness=self.snapshot['terminal_hashes'][self.job['id']]
        witness['files'][next(iter(witness['files']))]=False
        with self.assertRaisesRegex(ValueError,'integrity'):
            verifier.verify_completed_job(self.snapshot,self.c,'5080',self.job)

    def test_incomplete_binary_manifest_even_consistent_witness_rejected(self):
        name='run/'+self.job['id']+'/terminal_manifest.json'
        terminal=json.loads(base.decode_snapshot(self.snapshot,self.c,'5080')[name])
        selected=next(key for key in terminal['files'] if key.endswith('/best_val_qty_rmse_model.pt'))
        del terminal['files'][selected]
        del self.snapshot['terminal_hashes'][self.job['id']]['files'][selected]
        replace_file(self.snapshot,name,terminal)
        with self.assertRaisesRegex(ValueError,'manifest incomplete'):
            verifier.verify_completed_job(self.snapshot,self.c,'5080',self.job)

    def test_captured_final_history_sha_rejected(self):
        name=next(name for name in self.snapshot['files'] if name.startswith('run/'+self.job['id']+'/') and name.endswith('/history.json'))
        history=json.loads(base.decode_snapshot(self.snapshot,self.c,'5080')[name])
        history['history'][0]['val_qty_rmse']+=1
        replace_file(self.snapshot,name,history)
        with self.assertRaisesRegex(ValueError,'captured file SHA'):
            verifier.verify_completed_job(self.snapshot,self.c,'5080',self.job)

    def test_actual_worker_nonzero_exit_rejected(self):
        name='worker_exits/'+self.job['id']+'.json'
        worker=json.loads(base.decode_snapshot(self.snapshot,self.c,'5080')[name]);worker['returncode']=1
        replace_file(self.snapshot,name,worker)
        with self.assertRaisesRegex(ValueError,'actual OS exit'):
            verifier.verify_completed_job(self.snapshot,self.c,'5080',self.job)

    def test_worker_parent_and_argv_identity_rejected(self):
        for field in ('supervisor_pid','argv'):
            snapshot=copy.deepcopy(self.snapshot);name='worker_exits/'+self.job['id']+'.json'
            worker=json.loads(base.decode_snapshot(snapshot,self.c,'5080')[name])
            if field=='argv':worker['argv'][5]='/foreign/campaign.py'
            else:worker[field]+=1
            replace_file(snapshot,name,worker)
            with self.assertRaises(ValueError):verifier.verify_completed_job(snapshot,self.c,'5080',self.job)

    def test_completed_worker_still_observed_rejected(self):
        files=base.decode_snapshot(self.snapshot,self.c,'5080')
        worker=json.loads(files['worker_exits/'+self.job['id']+'.json'])
        pid=worker['worker_group_pid'];self.snapshot['processes'][str(pid)]={'status':'live','pgrp':pid}
        with self.assertRaisesRegex(ValueError,'worker remains'):
            verifier.training_status(self.snapshot,self.c,'5080')

    def test_full_host_busy_or_dispatcher_present_rejected(self):
        for kind in ('gpu','dispatcher'):
            snapshot=copy.deepcopy(self.snapshot)
            if kind=='gpu':snapshot['gpu_pids']=[99999]
            else:
                parent=json.loads(base.decode_snapshot(snapshot,self.c,'5080')['supervisor.claim'])
                snapshot['processes'][str(parent['pid'])]={'status':'live','argv':parent['argv']}
            with self.assertRaises(ValueError):verifier.training_status(snapshot,self.c,'5080')

    def test_single_job_survives_unrelated_administrative_dispatch_failure(self):
        replace_file(self.snapshot,'phases/train_failure.json',{'reason':'FileExistsError','scientific_failure':False})
        exit_name='phases/train_exit.json';exit_record=json.loads(base.decode_snapshot(self.snapshot,self.c,'5080')[exit_name])
        exit_record['os_exit_code']=1;replace_file(self.snapshot,exit_name,exit_record)
        self.assertEqual(verifier.verify_completed_job(self.snapshot,self.c,'5080',self.job)['status'],'complete')
        report=verifier.training_status(self.snapshot,self.c,'5080')
        self.assertEqual((report['status'],report['completed']),('training_failed',24))
        self.assertFalse(report['scientific_failure_inferred'])

    def test_native_bridge_uses_original_idle_cache_for_busy_live_host(self):
        with self.assertRaisesRegex(ValueError,'GPU still occupied'):
            base.qualification_binding(self.snapshots['5090'],self.c,'5090')
        bridge=verifier.qualification_bridge(BUNDLE,self.c,'5090',self.snapshots['5090'])
        self.assertFalse(bridge['remote_training_permit_raw_directly_observed'])

    def test_native_raw_evidence_mutation_rejected(self):
        name='qualification/receipt.json'
        receipt=json.loads(base.decode_snapshot(self.snapshot,self.c,'5080')[name]);receipt['runtime']['cudnn']=92000
        replace_file(self.snapshot,name,receipt)
        with self.assertRaisesRegex(ValueError,'bridge rawSHA'):
            verifier.qualification_bridge(BUNDLE,self.c,'5080',self.snapshot)


class CacheTests(unittest.TestCase):
    setUpClass = classmethod(ActualTerminalTests.setUpClass.__func__)

    def setUp(self):
        ActualTerminalTests.setUp(self)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.bundle=Path(self.tmp.name)/'bundle';self.output=Path(self.tmp.name)/'monitor_v2'
        for name in ['launch_authorization_5080.json','launch_release_5080.json','launch_hold.json',
            'control/qualification_cache/5080/snapshot.json','control/training_permits/5080.json','control/deployment/5080.json']:
            dst=self.bundle/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(BUNDLE/name,dst)
        self.path=self.bundle/'control/status/5080/original.json';self.path.parent.mkdir(parents=True)
        shutil.copyfile(self.paths['5080'],self.path)
        self.mock=patch.object(base,'check_bundle',return_value=(self.c,{'contract_sha256':base.canonical(self.c)}))
        self.mock.start();self.addCleanup(self.mock.stop)

    def test_cache_twice_without_ssh_preserves_original_time_after_lease(self):
        with patch.object(base.SSHRemote,'call',side_effect=AssertionError('SSH forbidden')):
            cache=verifier.seed_terminal_cache(self.bundle,self.output,'5080',self.path)
            original=base.read(cache)
            with patch.object(verifier.time,'time',return_value=original['qualification_bridge']['original_host_lease']['deadline_unix']+1000):
                for _ in range(2):
                    path,snapshot,report=verifier.verified_terminal(self.bundle,self.output,self.c,'5080')
                    self.assertEqual(snapshot['observed_unix'],1791847640.420401)
                    self.assertEqual(report['status'],verifier.TERMINAL)
            self.assertEqual(base.read(cache),original)

    def test_cache_raw_snapshot_and_original_time_mutations_rejected(self):
        cache=verifier.seed_terminal_cache(self.bundle,self.output,'5080',self.path)
        item=base.read(cache);item['original_observed_unix']+=1;cache.write_text(json.dumps(item))
        with self.assertRaisesRegex(ValueError,'original observation time'):
            verifier.verified_terminal(self.bundle,self.output,self.c,'5080')
        item['original_observed_unix']-=1;cache.write_text(json.dumps(item));self.path.write_bytes(self.path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'raw snapshot SHA'):
            verifier.verified_terminal(self.bundle,self.output,self.c,'5080')

    def test_local_permit_raw_mutation_rejected_even_same_semantics(self):
        verifier.seed_terminal_cache(self.bundle,self.output,'5080',self.path)
        path=self.bundle/'control/training_permits/5080.json';path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'proof changed'):
            verifier.verified_terminal(self.bundle,self.output,self.c,'5080')

    def test_existing_cache_never_overwritten_on_conflict(self):
        cache=verifier.seed_terminal_cache(self.bundle,self.output,'5080',self.path)
        cache.write_text('{}')
        with self.assertRaisesRegex(ValueError,'Existing terminal cache differs'):
            verifier.seed_terminal_cache(self.bundle,self.output,'5080',self.path)
        self.assertEqual(cache.read_text(),'{}')

    def test_native_launch_exit_consistency_cannot_hide_foreign_argv(self):
        qpath=self.bundle/'control/qualification_cache/5080/snapshot.json'
        cached=base.read(qpath)
        for name in ('phases/qualify_launch.json','phases/qualify_exit.json'):
            value=json.loads(base.decode_snapshot(cached,self.c,'5080')[name]);value['argv'][1]='/foreign/campaign.py'
            replace_file(cached,name,value)
        qpath.write_text(json.dumps(cached))
        with self.assertRaisesRegex(ValueError,'Native qualification argv'):
            verifier.qualification_bridge(self.bundle,self.c,'5080',self.snapshot)


if __name__=='__main__':unittest.main()
