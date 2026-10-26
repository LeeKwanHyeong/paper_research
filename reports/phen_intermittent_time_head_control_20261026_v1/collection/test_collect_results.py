import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

import collect_results as c

ROOT = Path(__file__).resolve().parent


def archive_file(path, entries):
    with tarfile.open(path, 'w:gz') as archive:
        for name, payload, kind in entries:
            info = tarfile.TarInfo(name)
            if kind == 'link':
                info.type = tarfile.SYMTYPE; info.linkname = '../../outside'
                archive.addfile(info)
            else:
                info.size = len(payload); archive.addfile(info, io.BytesIO(payload))


class CollectionTests(unittest.TestCase):
    def test_traversal_absolute_symlink_and_duplicate_rejected_before_writes(self):
        cases = [[('../outside', b'x', 'file')], [('/outside', b'x', 'file')],
                 [('inside', b'', 'link')], [('same', b'x', 'file'), ('same', b'y', 'file')]]
        for entries in cases:
            with self.subTest(entries=entries), tempfile.TemporaryDirectory(dir=ROOT) as tmp:
                root = Path(tmp); archive = root/'a.tar.gz'; output = root/'output'
                archive_file(archive, entries)
                with self.assertRaises(ValueError): c.safe_extract(archive, output)
                self.assertFalse(output.exists())

    def test_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp)/'file'; path.write_bytes(b'a'); ref = c.record(path)
            path.write_bytes(b'b')
            with self.assertRaises(ValueError): c.verify_file(path, ref)

    def test_actual_exit_and_postguard_are_mandatory(self):
        c.verify_exit({'actual_child_exit': 0})
        c.verify_exit({'actual_collector_exit': 0, 'post_guard_passed': True}, True)
        for receipt, controller in [({'actual_child_exit': 1}, False), ({'actual_child_exit': 0, 'timed_out': True}, False),
                                    ({'actual_collector_exit': 0, 'post_guard_passed': False}, True),
                                    ({'actual_collector_exit': None, 'post_guard_passed': True}, True)]:
            with self.assertRaises(ValueError): c.verify_exit(receipt, controller)

    def test_members_exact_inventory_and_remote_path(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp); remote = Path('/remote/attempt'); (root/'inside').write_bytes(b'a')
            manifest = {'files': {'inside': c.record(root/'inside', remote/'inside')}}
            p = root/'qualification_RESULT_MEMBERS.json'; p.write_text(json.dumps(manifest))
            payload = {'manifest': c.record(p, remote/p.name)}
            _, returned = c.verify_members(root, 'qualification', payload, remote)
            self.assertEqual(returned, manifest)
            (root/'extra').write_bytes(b'extra')
            with self.assertRaises(ValueError): c.verify_members(root, 'qualification', payload, remote)

    def test_pack_discovery_exact_match_and_ambiguity(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp); transports = root/'transports'; transports.mkdir()
            archive = root/'qualification_results.tar.gz'; archive.write_bytes(b'archive')
            remote = Path('/remote/attempt')
            payload = {'archive': c.record(archive, remote/archive.name), 'manifest': {'path': str(remote/'qualification_RESULT_MEMBERS.json'), 'bytes': 10, 'sha256': 'abc'}}
            path = transports/'01_5090_pack_qualification.stdout'; path.write_text(json.dumps(payload))
            receipt = {'actual_exit': 0, 'timed_out': False, 'stdout': c.record(path)}
            path.with_suffix('.receipt.json').write_text(json.dumps(receipt))
            selected, _ = c.find_pack(root, '5090', 'qualification', archive, remote)
            self.assertEqual(selected, path)
            path2 = transports/'02_5090_pack_qualification.stdout'; path2.write_text(json.dumps(payload))
            path2.with_suffix('.receipt.json').write_text(json.dumps({**receipt, 'stdout': c.record(path2)}))
            with self.assertRaises(ValueError): c.find_pack(root, '5090', 'qualification', archive, remote)
            self.assertEqual(c.find_pack(root, '5090', 'qualification', archive, remote, path)[0], path)

    def test_existing_destination_preserved(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp); archive = root/'a.tar.gz'; output = root/'output'
            archive_file(archive, [('inside', b'x', 'file')]); output.mkdir(); marker = output/'old'; marker.write_bytes(b'old')
            with self.assertRaises(ValueError): c.safe_extract(archive, output)
            self.assertEqual(marker.read_bytes(), b'old')

    def test_guard_protected_process_exact(self):
        frozen = {'boot_id': 'boot', 'gpu_uuid': 'uuid', 'protected': [{'pid': 3, 'state': 'T'}], 'minimum_free_mib': 6000}
        observed = {'boot_id': 'boot', 'gpu_uuid': 'uuid', 'protected': frozen['protected'], 'gpu_free_mib': 7000, 'compute_pids': [3]}
        c.verify_guard(observed, frozen)
        with self.assertRaises(ValueError): c.verify_guard({**observed, 'compute_pids': [3, 4]}, frozen)


if __name__ == '__main__': unittest.main()
