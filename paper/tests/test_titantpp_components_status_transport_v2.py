"""Fabricated status/proc/SSH responses only; no remote or scientific data."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from paper.scripts import launch_titantpp_cnn_gru_components_v5 as original
from paper.scripts import titantpp_components_status_transport_v2 as transport


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def raw_record(value):
    return json.dumps(value).encode()


class StatusTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='component-status-v2-', dir='/private/tmp')
        self.root = Path(self.temp.name)
        self.c = {'schema': 'fabricated', 'hosts': {'5090': {'root': '/owned/fabricated_5090',
            'python': '/native/python3.12', 'gpu_uuid': 'GPU-fixed',
            'environment': {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPATH': '/owned/source'}}}}
        spec = self.c['hosts']['5090']
        self.dispatcher = [spec['python'],spec['root']+'/operation/campaign.py','--contract',
            spec['root']+'/execution_contract.json','--host','5090','--mode','dispatch']
        self.wrapper = [spec['python'],spec['root']+'/operation/pipeline.py','--contract',
            spec['root']+'/execution_contract.json','--host','5090','--phase','train']
        self.worker = ['timeout','--signal=TERM','123s','/native/python3.12','/owned/worker.py']
        owner = {'pid': 100, 'ppid': 200, 'argv': self.dispatcher, 'contract_sha256': canonical(self.c)}
        self.files = {name: raw_record(owner) for name in
            ('supervisor.claim','phases/train_launch.json','phases/train_exit.json')}
        self.files['worker_exits/job.json'] = raw_record({'worker_group_pid': 300, 'supervisor_pid': 100,
            'argv': self.worker, 'contract_sha256': canonical(self.c), 'actual_child_exit_observed': True})
        self.state = {'status': 'failed', 'active_jobs': []}
        self.namespace = SimpleNamespace(REMOTE_SCRIPT=original.REMOTE_SCRIPT, canonical=canonical,
            intent=original.intent, write_exclusive=original.write_exclusive,
            authorization=lambda bundle,c,host: {'approved': True, 'contract_sha256': canonical(c)})

    def tearDown(self):
        self.temp.cleanup()

    def absent(self, pid, root):
        raise FileNotFoundError('fabricated absent')

    def collect(self, state=None, files=None, pids=None, reader=None):
        return transport.collect_process_observations(self.state if state is None else state,
            self.files if files is None else files, [] if pids is None else pids,
            self.c, '5090', self.root/'proc', self.absent if reader is None else reader)

    def remote(self):
        folder = self.root/'evidence'
        folder.mkdir()
        return transport.StatusOnlyRemote(self.namespace, folder)

    def payload(self):
        return {'host': '5090', 'contract': self.c, 'contract_sha256': canonical(self.c), 'approval': {}}

    def test_regression_failed_empty_list_and_running_dict(self):
        # Original statement reproduced locally, without any remote call.
        with self.assertRaises(AttributeError):
            list(self.state.get('active_jobs',{}).values())
        observations, processes, owners = self.collect()
        self.assertEqual(processes,{})
        self.assertEqual(owners,[100,200,300])
        self.assertTrue(all(o['state']=='absent' for o in observations.values()))
        running = {'status':'running', 'supervisor_pid':100, 'supervisor_argv':self.dispatcher,
            'active_jobs':{'job':{'worker_group_pid':300, 'worker_guard_argv':self.worker}}}
        self.assertEqual(self.collect(running)[2],[100,200,300])
        self.assertEqual(self.collect({'status':'complete'})[2],[100,200,300])

    def test_active_types_failed_nonempty_and_invalid_pids_reject(self):
        for active in (None, 1, 'bad', ['job']):
            with self.assertRaises(ValueError):
                self.collect({'status':'failed','active_jobs':active})
        with self.assertRaises(ValueError):
            self.collect({'status':'running','active_jobs':[]})
        for pid in (None, False, '300', -1, 0, 3.5, float('nan')):
            with self.assertRaises(ValueError):
                self.collect({'status':'running','active_jobs':{'job':{'worker_group_pid':pid,'worker_guard_argv':self.worker}}})
        with self.assertRaises(ValueError):
            self.collect(pids=[True])

    def test_exact_owner_sources_and_pid_fields_are_preserved(self):
        observations,_,_ = self.collect()
        for pid in ('100','200'):
            self.assertEqual(set(observations[pid]['evidence_sources']),
                {'supervisor.claim','phases/train_launch.json','phases/train_exit.json'} |
                ({'worker_exits/job.json'} if pid=='100' else set()))
        self.assertEqual(observations['200']['evidence_pid_fields']['supervisor.claim'], ['ppid'])
        self.assertEqual(observations['300']['evidence_pid_fields']['worker_exits/job.json'], ['worker_group_pid'])
        self.assertEqual(observations['100']['evidence_pid_fields']['worker_exits/job.json'], ['supervisor_pid'])

    def test_owner_contract_pid_ppid_argv_bindings_reject_malformed(self):
        for key, value in (('contract_sha256','foreign'),('pid',False),('ppid',0),('argv',['foreign'])):
            files = deepcopy(self.files)
            record=json.loads(files['supervisor.claim'])
            record[key]=value
            files['supervisor.claim']=raw_record(record)
            with self.assertRaises(ValueError): self.collect(files=files)
        files=deepcopy(self.files)
        record=json.loads(files['phases/train_exit.json']); record['ppid']=201
        files['phases/train_exit.json']=raw_record(record)
        with self.assertRaises(ValueError): self.collect(files=files)

    def test_live_reused_pid_and_query_error_are_not_absence(self):
        def live(pid, root):
            if pid!=100: raise FileNotFoundError()
            return {'pid':pid,'ppid':200,'pgrp':100,'argv':self.dispatcher,'start_ticks':77,
                'identity_stable_during_query':True}
        observations, processes,_=self.collect(reader=live)
        self.assertEqual(observations['100']['state'],'live')
        self.assertEqual(processes['100']['argv'],self.dispatcher)
        def reused(pid,root):
            actual=live(pid,root); actual['argv']=['unrelated']; return actual
        self.assertEqual(self.collect(reader=reused)[0]['100']['state'],'reused_pid')
        def raced(pid,root):
            actual=live(pid,root); actual['identity_stable_during_query']=False; return actual
        self.assertEqual(self.collect(reader=raced)[0]['100']['state'],'reused_pid')
        for error in (PermissionError, OSError, ValueError):
            def denied(pid, root): raise error('fabricated query error')
            observations, processes,_=self.collect(reader=denied)
            self.assertTrue(all(o['state']=='query_error' and not o['absence_observed'] for o in observations.values()))
            self.assertEqual(set(processes), {'100','200','300'})

    def test_proc_reader_stable_identity_and_pre_post_race(self):
        folder=self.root/'proc'/'100'; folder.mkdir(parents=True)
        fields=['S','200','100']+['0']*17; fields[19]='77'
        stat='100 (python with space) '+' '.join(fields)
        (folder/'stat').write_text(stat)
        (folder/'cmdline').write_bytes('\0'.join(self.dispatcher).encode()+b'\0')
        actual=transport.status_proc_read(100,self.root/'proc')
        self.assertTrue(actual['identity_stable_during_query'])
        self.assertEqual(actual['start_ticks'],77)
        original_read=Path.read_text
        count=0
        def fake_read(path,*args,**kwargs):
            nonlocal count
            if path==folder/'stat':
                count+=1
                return stat if count==1 else stat[:-2]+'88'
            return original_read(path,*args,**kwargs)
        with patch.object(Path,'read_text',fake_read):
            self.assertFalse(transport.status_proc_read(100,self.root/'proc')['identity_stable_during_query'])

    def test_proc_initial_absence_and_late_disappearance_are_distinct(self):
        self.assertEqual(self.collect(reader=transport.status_proc_read)[0]['100']['state'], 'absent')
        folder=self.root/'proc'/'100'; folder.mkdir(parents=True)
        fields=['S','200','100']+['0']*17; fields[19]='77'
        (folder/'stat').write_text('100 (python) '+' '.join(fields))
        # First stat exists but cmdline has disappeared: never explicit absent.
        observations, processes,_=self.collect(reader=transport.status_proc_read)
        self.assertEqual(observations['100']['state'], 'query_error')
        self.assertFalse(observations['100']['absence_observed'])
        self.assertTrue(processes['100']['query_error'])
        (folder/'cmdline').write_bytes('\0'.join(self.dispatcher).encode()+b'\0')
        original_read=Path.read_text
        count=0
        def lost_last_stat(path,*args,**kwargs):
            nonlocal count
            if path==folder/'stat':
                count+=1
                if count==2: raise FileNotFoundError('fabricated final stat disappeared')
            return original_read(path,*args,**kwargs)
        with patch.object(Path,'read_text',lost_last_stat):
            observations, processes,_=self.collect(reader=transport.status_proc_read)
        self.assertEqual(observations['100']['state'], 'query_error')
        self.assertFalse(observations['100']['absence_observed'])
        self.assertTrue(processes['100']['query_error'])

    def test_patched_script_keeps_original_and_only_status_guard(self):
        before=original.REMOTE_SCRIPT
        patched=transport.patched_remote_script(self.namespace)
        self.assertEqual(original.REMOTE_SCRIPT,before)
        self.assertIn("need(mode=='status'",patched)
        self.assertIn("--query-gpu=uuid",patched)
        self.assertIn('process_observations=process_observations',patched)
        with patch.object(sys,'argv',['python','health']), patch.object(sys,'stdin',io.StringIO(json.dumps(self.payload()))), \
                patch.object(subprocess,'run',side_effect=AssertionError('remote action attempted')):
            with self.assertRaisesRegex(ValueError,'read-only status'):
                exec(compile(patched,'status-guard-test','exec'),{})
        broken=SimpleNamespace(REMOTE_SCRIPT=before.replace("processes={}; process_ids=set(pids)", 'changed'))
        with self.assertRaisesRegex(ValueError,'anchors changed'):
            transport.patched_remote_script(broken)

    def test_actions_and_upload_are_rejected_before_any_subprocess(self):
        remote=self.remote()
        with patch.object(subprocess,'run',side_effect=AssertionError('unexpected remote')):
            for action in ('health','reserve','deploy','phase','upload'):
                with self.assertRaises(ValueError): remote.call('5090',action,self.payload())
            with self.assertRaises(ValueError): remote.upload('5090','a','b')

    def test_success_preserves_raw_stdout_stderr_and_pinned_command(self):
        remote=self.remote(); stdout=b'{"fabricated_snapshot":true}\n'; stderr=b'warning\n'
        completed=subprocess.CompletedProcess(['fake'],0,stdout,stderr)
        with patch.object(subprocess,'run',return_value=completed) as called:
            result=remote.call('5090','status',self.payload())
        self.assertEqual(called.call_count,1)
        argv=called.call_args.args[0]
        self.assertEqual(argv[:6],['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5090'])
        self.assertIn('/native/python3.12',argv[-1])
        self.assertTrue(called.call_args.kwargs['check'])
        self.assertIsInstance(called.call_args.kwargs['input'],bytes)
        receipt=json.loads(remote.evidence_receipt_path.read_text())
        self.assertEqual(Path(receipt['stdout']['path']).read_bytes(),stdout)
        self.assertEqual(Path(receipt['stderr']['path']).read_bytes(),stderr)
        self.assertEqual(receipt['stdout']['full_sha256'],hashlib.sha256(stdout).hexdigest())
        self.assertTrue(receipt['actual_transport_exit_observed'])
        self.assertEqual(receipt['transport_os_exit_code'],0)
        self.assertEqual(result['transport_evidence_sha256'],hashlib.sha256(remote.evidence_receipt_path.read_bytes()).hexdigest())

    def test_nonzero_stderr_timeout_decode_and_json_errors_preserved(self):
        cases = [(subprocess.CalledProcessError(1,['giant-secret-command'],output=b'partial',stderr=b'AttributeError: values\n'),
                  'nonzero_exit',True,1),
                 (subprocess.TimeoutExpired(['giant-secret-command'],120,output=b'partial',stderr=b'timeout stderr'),
                  'timeout_exit_unobserved',False,None),
                 (subprocess.CompletedProcess(['fake'],0,b'not-json',b'parse stderr'),
                  'invalid_or_oversized_json',True,0),
                 (subprocess.CompletedProcess(['fake'],0,b'\xff',b'decode stderr'), 'decode_error',True,0)]
        for index,(value,status,observed,code) in enumerate(cases):
            folder=self.root/str(index); folder.mkdir()
            remote=transport.StatusOnlyRemote(self.namespace,folder)
            kwargs={'side_effect':value} if isinstance(value,BaseException) else {'return_value':value}
            with patch.object(subprocess,'run',**kwargs):
                with self.assertRaises(transport.StatusEvidenceError) as caught:
                    remote.call('5090','status',self.payload())
            self.assertNotIn('giant-secret-command',str(caught.exception))
            receipt=json.loads(Path(caught.exception.evidence_receipt_path).read_text())
            self.assertEqual(receipt['status'],status)
            self.assertEqual(receipt['actual_transport_exit_observed'],observed)
            self.assertEqual(receipt['transport_os_exit_code'],code)
            self.assertTrue(Path(receipt['stderr']['path']).exists())

    def test_stream_bound_full_sha_and_exclusive_no_overwrite(self):
        folder=self.root/'streams'; folder.mkdir()
        with patch.object(transport,'MAX_BYTES',4):
            record=transport.preserve_stream(folder,'stdout',b'123456')
        self.assertTrue(record['truncated'])
        self.assertEqual(record['byte_count'],6)
        self.assertEqual(record['full_sha256'],hashlib.sha256(b'123456').hexdigest())
        self.assertEqual(record['preserved_sha256'],hashlib.sha256(b'1234').hexdigest())
        with self.assertRaises(FileExistsError): transport.preserve_stream(folder,'stdout',b'changed')
        self.assertEqual(Path(record['path']).read_bytes(),b'1234')

    def test_read_status_once_success_snapshot_intent_exit_and_authorization(self):
        completed=subprocess.CompletedProcess(['fake'],0,b'{"status":"fabricated"}',b'')
        with patch.object(subprocess,'run',return_value=completed) as called:
            path,snapshot=transport.read_status_once(self.root/'bundle',self.namespace,self.c,'5090',self.root/'diagnostics')
        self.assertEqual(called.call_count,1)
        self.assertEqual(json.loads(path.read_text()),snapshot)
        intents=list((self.root/'bundle/control/intents').glob('*.json'))
        self.assertEqual(len(intents),2)
        self.assertTrue(all(p.name.startswith('status_repair_v2_') for p in intents))
        exit_record=json.loads(next(p for p in intents if p.name.endswith('_exit.json')).read_text())
        self.assertEqual(exit_record['snapshot_raw_sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertFalse(exit_record['scientific_phase_exit_inferred'])
        denied=deepcopy(self.namespace)
        denied.authorization=lambda *args: (_ for _ in ()).throw(ValueError('approval denied'))
        with patch.object(subprocess,'run',side_effect=AssertionError('authorization failed yet remote called')):
            with self.assertRaises(ValueError):
                transport.read_status_once(self.root/'no-bundle',denied,self.c,'5090',self.root/'no-diagnostics')
        self.assertFalse((self.root/'no-diagnostics').exists())

    def test_read_status_once_failure_retains_original_stdout_stderr_exit(self):
        error=subprocess.CalledProcessError(1,['giant-command'],output=b'raw',stderr=b'original failure')
        with patch.object(subprocess,'run',side_effect=error):
            with self.assertRaises(transport.StatusEvidenceError) as caught:
                transport.read_status_once(self.root/'bundle',self.namespace,self.c,'5090',self.root/'diagnostics')
        exit_path=next((self.root/'bundle/control/intents').glob('*_exit.json'))
        record=json.loads(exit_path.read_text())
        self.assertEqual(record['evidence_receipt_path'],caught.exception.evidence_receipt_path)
        self.assertEqual(record['transport_os_exit_code'],1)
        self.assertTrue(record['actual_transport_exit_observed'])
        self.assertNotIn('giant-command',record['message'])
        self.assertFalse((self.root/'bundle/control/status').exists())


if __name__=='__main__':
    unittest.main()
