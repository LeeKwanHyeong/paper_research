import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper.scripts import launch_titantpp_components_5080_health_compat as bridge


class LockCompatibility(unittest.TestCase):
    def execute_lock_block(self, text, *, unreadable=False):
        package = '/var/lib/dpkg/lock'
        fake = SimpleNamespace(exists=lambda: True,
            stat=lambda: SimpleNamespace(st_dev=os.makedev(8, 2), st_ino=123456))
        def path(name):
            if name == '/proc/locks':
                def read_text():
                    if unreadable: raise PermissionError('proc locks unreadable')
                    return text
                return SimpleNamespace(read_text=read_text)
            if name == package: return fake
            return SimpleNamespace(exists=lambda: False)
        namespace = {'pathlib': SimpleNamespace(Path=path), 'os': os}
        with patch.object(os, 'open', side_effect=AssertionError('must not open root lock files')):
            exec('if True:\n' + bridge.NEW, namespace)
        return namespace['locks']

    def test_root_owned_lock_is_detected_without_opening_it(self):
        self.assertEqual(self.execute_lock_block('12: POSIX ADVISORY WRITE 123 08:02:123456 0 EOF'),
            ['/var/lib/dpkg/lock'])

    def test_inode_and_device_must_both_match(self):
        self.assertEqual(self.execute_lock_block('12: FLOCK ADVISORY WRITE 123 08:03:123456 0 EOF'), [])
        self.assertEqual(self.execute_lock_block('12: OFDLCK ADVISORY WRITE -1 08:02:123457 0 EOF'), [])

    def test_unreadable_inventory_fails_closed(self):
        with self.assertRaises(PermissionError): self.execute_lock_block('', unreadable=True)

    def test_only_single_health_block_changes(self):
        original = bridge.sealed.REMOTE_SCRIPT
        updated = bridge.health_script()
        self.assertEqual(updated.replace(bridge.NEW, bridge.OLD, 1), original)

    def test_other_host_rejected_before_transport(self):
        remote = bridge.CompatibleRemote()
        with patch.object(bridge.sealed.SSHRemote, 'call') as call:
            with self.assertRaises(ValueError): remote.call('5090', 'health', {'host': '5090'})
            call.assert_not_called()


if __name__ == '__main__': unittest.main()
