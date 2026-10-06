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
