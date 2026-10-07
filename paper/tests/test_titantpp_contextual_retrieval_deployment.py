"""Archive admission and no-retry gates; fabricated local files, no network."""
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
from paper.scripts import deploy_titantpp_contextual_retrieval as d


class DeploymentTests(unittest.TestCase):
    def observation(self, b, *, offset=2, age=1):
        q=b/'qualification/5080';q.mkdir(parents=True)
        received_ns=1791349372178051000
        receipt={'status':'PASS'}
        raw=json.dumps(receipt)
        (q/'receipt.json').write_text(raw)
        observed={'host':'5080','observed_unix':received_ns/1e9+offset,
                  'local_received_unix_ns':received_ns,'receipt':receipt,
                  'receipt_json_text':raw,'receipt_sha256':d.sha(q/'receipt.json'),
                  'qualification_shell_exit':{'returncode':0},
                  'GPU_query_returncode':0,'GPU_process_query_returncode':0,
                  'ps_returncode':0,'owned_processes':[],'gpu_processes':''}
        path=q/(str(received_ns)+'.json');d.write(path,observed)
        latest={'path':str(path),'sha256':d.sha(path)}
        (q/'latest.json').write_text(json.dumps(latest))
        return q,path,latest,received_ns/1e9+age

    def test_small_native_clock_skew_uses_fresh_local_collection(self):
        for offset in (-2,0,1.979943,10):
            with self.subTest(offset=offset),tempfile.TemporaryDirectory() as tmp:
                b=Path(tmp);q,path,latest,now=self.observation(b,offset=offset)
                with patch.object(d,'B',b),patch.object(d.time,'time',return_value=now):
                    self.assertEqual(d.fresh_qualification_observation('5080')['host'],'5080')

    def test_old_readback_future_collection_and_large_server_skew_rejected(self):
        for offset,age in ((2,600),(2,-1),(11,1),(-11,1),(float('nan'),1)):
            with self.subTest(offset=offset,age=age),tempfile.TemporaryDirectory() as tmp:
                b=Path(tmp)
                # JSON writing rejects NaN too; emulate a malformed external input below.
                q,path,latest,now=self.observation(b,offset=0 if offset!=offset else offset,age=age)
                if offset!=offset:
                    observed=d.read(path);observed['observed_unix']=offset
                    path.write_text(json.dumps(observed));latest['sha256']=d.sha(path)
                    (q/'latest.json').write_text(json.dumps(latest))
                with patch.object(d,'B',b),patch.object(d.time,'time',return_value=now):
                    with self.assertRaises(ValueError):d.fresh_qualification_observation('5080')

    def test_readback_path_SHA_receipt_and_timestamp_bindings_rejected(self):
        for tamper in ('path','snapshot_sha','receipt_sha','timestamp','legacy','host'):
            with self.subTest(tamper=tamper),tempfile.TemporaryDirectory() as tmp:
                b=Path(tmp);q,path,latest,now=self.observation(b)
                if tamper=='path':
                    foreign=b/path.name;foreign.write_bytes(path.read_bytes())
                    latest['path']=str(foreign)
                elif tamper=='snapshot_sha':latest['sha256']='0'*64
                elif tamper=='receipt_sha':(q/'receipt.json').write_text('{"status":"different"}')
                else:
                    observed=d.read(path)
                    if tamper=='timestamp':observed['local_received_unix_ns']+=1
                    elif tamper=='legacy':del observed['local_received_unix_ns']
                    else:observed['host']='5090'
                    path.write_text(json.dumps(observed));latest['sha256']=d.sha(path)
                (q/'latest.json').write_text(json.dumps(latest))
                with patch.object(d,'B',b),patch.object(d.time,'time',return_value=now):
                    with self.assertRaises(ValueError):d.fresh_qualification_observation('5080')

    def test_stale_launch_does_not_write_permit_or_consume_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);q,path,latest,now=self.observation(b,age=601)
            with patch.object(d,'B',b),patch.object(d.time,'time',return_value=now), \
                    patch.object(d,'capacity_gate',return_value=True), \
                    patch.object(d,'contract',return_value={'hosts':{'5080':{}}}), \
                    patch.object(d,'qualified_receipts',return_value={'5080':{'status':'PASS'}}), \
                    patch.object(d,'ssh') as ssh:
                with self.assertRaisesRegex(ValueError,'stale'):d.launch('5080')
                ssh.assert_not_called()
                self.assertFalse((b/'training_permit_5080.json').exists())
                self.assertFalse((b/'deployment/launch_intent_5080.json').exists())

    def archive(self, root, name='x', payload=b'abc', digest=None, kind=None):
        archive = root/'package.tar.gz'
        raw = json.dumps({'files':{name:digest or d.hashlib.sha256(payload).hexdigest()}}).encode()
        with tarfile.open(archive,'w:gz') as t:
            m=tarfile.TarInfo('package_manifest.json');m.size=len(raw);t.addfile(m,io.BytesIO(raw))
            m=tarfile.TarInfo(name)
            if kind: m.type=kind;m.linkname='foreign'
            else: m.size=len(payload)
            t.addfile(m, None if kind else io.BytesIO(payload))
        return archive

    def test_safe_archive_and_member_SHA(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);_,m=d.verify_archive(self.archive(root))
            self.assertEqual(set(m['files']),{'x'})
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):d.verify_archive(self.archive(Path(tmp),digest='0'*64))

    def test_escaping_and_link_members_rejected(self):
        for name,kind in [('../foreign',None),('/foreign',None),('link',tarfile.SYMTYPE),('link',tarfile.LNKTYPE)]:
            with self.subTest(name=name,kind=kind),tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(ValueError):d.verify_archive(self.archive(Path(tmp),name=name,kind=kind))

    def test_incomplete_mutation_intents_never_replayed(self):
        for stage in ('transfer','qualification','launch'):
            with self.subTest(stage=stage),tempfile.TemporaryDirectory() as tmp:
                b=Path(tmp);folder=b/'deployment';folder.mkdir()
                if stage!='transfer':(folder/'deployment_receipt_5090.json').write_text('{}')
                if stage=='launch':(folder/'qualification_start_5090.json').write_text('{}')
                (folder/(stage+'_intent_5090.json')).write_text('{}')
                with patch.object(d,'B',b),patch.object(d,'capacity_gate',return_value=True),patch.object(d,'ssh') as ssh,patch.object(d,'deploy') as deploy,patch.object(d,'qualify') as qualify,patch.object(d,'collect') as collect,patch.object(d,'launch') as launch:
                    with self.assertRaisesRegex(ValueError,'manual action'):d.activate('5090')
                    for method in (ssh,deploy,qualify,collect,launch):method.assert_not_called()

    def test_capacity_pending_and_already_launched_have_no_remote_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);folder=b/'deployment';folder.mkdir()
            (folder/'launch_receipt_5080.json').write_text('{}')
            with patch.object(d,'B',b),patch.object(d,'capacity_gate',return_value=False),patch.object(d,'ssh') as ssh:
                d.activate('5080');ssh.assert_not_called()
            with patch.object(d,'B',b),patch.object(d,'capacity_gate',return_value=True),patch.object(d,'ssh') as ssh,patch.object(d,'collect') as collect:
                d.activate('5080');ssh.assert_not_called();collect.assert_not_called()

    def test_receipt_before_native_exit_cannot_consume_launch_intent(self):
        with tempfile.TemporaryDirectory() as tmp:
            b=Path(tmp);folder=b/'deployment';folder.mkdir()
            for name in ('deployment_receipt_5090.json','qualification_start_5090.json'):(folder/name).write_text('{}')
            q=b/'qualification/5090';q.mkdir(parents=True);(q/'receipt.json').write_text('{}')
            observed={'qualification_shell_exit':None,'qualification_failure':None,'failure':None,
                      'GPU_query_returncode':0,'GPU_process_query_returncode':0,'ps_returncode':0,
                      'owned_processes':['qualification still exits'],'gpu_processes':''}
            with patch.object(d,'B',b),patch.object(d,'capacity_gate',return_value=True),patch.object(d,'collect',return_value=observed),patch.object(d,'launch') as launch:
                d.activate('5090');launch.assert_not_called()
                self.assertFalse((folder/'launch_intent_5090.json').exists())
            observed['qualification_shell_exit']={'returncode':124}
            with self.assertRaisesRegex(ValueError,'shell failed'):d.qualification_terminated(observed)


if __name__=='__main__':unittest.main()
