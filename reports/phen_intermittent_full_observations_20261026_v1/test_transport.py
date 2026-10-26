"""Mock SSH failure receipts and exclusive archive paths without any network calls."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import control_transport as transport


def record_bytes(data):
    return {"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}


def archive_bytes(extra=False, duplicate=False):
    payload=b'synthetic payload'
    manifest=json.dumps({'files':{'output_full/data':record_bytes(payload)}},sort_keys=True).encode()
    files=[('output_full/data',payload),('full_RESULT_MEMBERS.json',manifest)]
    if extra:files.append(('output_full/undeclared',b'extra'))
    if duplicate:files.append(('output_full/data',payload))
    buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tar:
        for name,data in files:
            info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
    return buf.getvalue(),manifest


class TransportFailureTests(unittest.TestCase):
    def test_captured_timeout_or_launch_error_preserves_raw_stream_and_prevents_duplicate_call(self):
        for failure in ('timeout','oserror'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as td:
                root=Path(td)
                error=(subprocess.TimeoutExpired(['mock-ssh'],45,output=b'partial stdout\n',stderr=b'partial stderr\n')
                       if failure=='timeout' else OSError('synthetic local launch error'))
                with patch.object(transport,'P',root),patch.object(transport.subprocess,'run',side_effect=error) as mocked:
                    with self.assertRaises(RuntimeError):transport.transport('5080','FAILURE_01',['mock-ssh'])
                    receipt=json.loads((root/'5080_FAILURE_01.receipt.json').read_text())
                    expected_out=b'partial stdout\n' if failure=='timeout' else b''
                    expected_err=b'partial stderr\n' if failure=='timeout' else b''
                    self.assertIsNone(receipt['actual_exit'])
                    self.assertIsNotNone(receipt['failure'])
                    self.assertEqual((root/'5080_FAILURE_01.stdout').read_bytes(),expected_out)
                    self.assertEqual((root/'5080_FAILURE_01.stderr').read_bytes(),expected_err)
                    for name,data in [('stdout',expected_out),('stderr',expected_err)]:
                        self.assertEqual({k:receipt[name][k] for k in ('bytes','sha256')},record_bytes(data))
                    preserved={p.name:p.read_bytes() for p in root.iterdir()}
                    with self.assertRaises(FileExistsError):transport.transport('5080','FAILURE_01',['mock-ssh'])
                    self.assertEqual(mocked.call_count,1)
                    self.assertEqual(preserved,{p.name:p.read_bytes() for p in root.iterdir()})

    def test_collection_rejects_partial_transfer_extra_or_duplicate_archive_members(self):
        for failure in ('timeout','oserror','extra_member','duplicate_member'):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as td:
                root=Path(td)
                data,manifest=archive_bytes(extra=failure=='extra_member',duplicate=failure=='duplicate_member')
                (root/'5080_GUARD.json').write_text(json.dumps({'remote_root':'/synthetic/remote','python':'python3'}))
                packed={'archive':{'path':'/synthetic/remote/full_results.tar.gz',**record_bytes(data)},
                        'manifest':{'path':'/synthetic/remote/full_RESULT_MEMBERS.json',**record_bytes(manifest)}}
                (root/'5080_FULL_PACK_01.stdout').write_text(json.dumps(packed))
                def fake_run(command,stdout,stderr,timeout):
                    if failure in ('timeout','oserror'):
                        stdout.write(b'partial archive');stdout.flush();stderr.write('raw ssh warning\n');stderr.flush()
                        if failure=='timeout':raise subprocess.TimeoutExpired(command,timeout)
                        raise OSError('synthetic transport launch error')
                    stdout.write(data);stdout.flush()
                    return subprocess.CompletedProcess(command,0)
                with (patch.object(transport,'P',root),
                      patch.object(transport.subprocess,'run',side_effect=fake_run) as mocked,
                      patch.object(transport.sys,'argv',['control_transport.py','collect','--host','5080','--mode','full'])):
                    with self.assertRaises(AssertionError):transport.main()
                target=root/'retrieval/5080/full'
                self.assertEqual(mocked.call_count,1)
                self.assertFalse((target/'COLLECTED.json').exists())
                receipt=json.loads((target/'TRANSFER.receipt.json').read_text())
                if failure in ('timeout','oserror'):
                    self.assertIsNone(receipt['actual_exit'])
                    self.assertIsNotNone(receipt['failure'])
                    self.assertEqual((target/'results.tar.gz').read_bytes(),b'partial archive')
                    self.assertEqual((target/'TRANSFER.stderr').read_bytes(),b'raw ssh warning\n')
                    self.assertEqual({k:receipt['archive'][k] for k in ('bytes','sha256')},record_bytes(b'partial archive'))
                else:self.assertEqual(receipt['actual_exit'],0)


if __name__=='__main__':unittest.main()
