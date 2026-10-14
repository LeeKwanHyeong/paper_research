"""Offline routing and fail-closed seal checks for the manual2 follow-up."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from paper.scripts import advance_titantpp_gru_cap_manual2 as wrapper


class Manual2FollowupTests(unittest.TestCase):
    def test_isolated_helpers_route_only_to_manual_launcher_and_root(self):
        launcher = SimpleNamespace()
        observer, collector = wrapper.routed_helpers(launcher)
        self.assertIs(observer.observe.__globals__['_manual_launcher'], launcher)
        self.assertIs(collector.completion_gate.__globals__['_manual_launcher'], launcher)
        self.assertIs(collector.completion_gate.__globals__['_manual_observer'], observer)
        self.assertIn("+p['host']+'_manual2'", collector.REMOTE_SCRIPT)
        from paper.scripts import collect_titantpp_gru_cap_validation as original
        self.assertNotIn("+p['host']+'_manual2'", original.REMOTE_SCRIPT)
        self.assertIsNot(collector.__dict__, original.__dict__)

    def test_original_helper_pin_mismatch_blocks_namespace_execution(self):
        with patch.dict(wrapper.PINNED, {wrapper.OBSERVER: '0' * 64}):
            with self.assertRaisesRegex(ValueError, 'Original helper SHA changed'):
                wrapper.routed_helpers(SimpleNamespace())

    def test_missing_wrapper_or_launcher_seal_blocks_before_any_observation(self):
        with tempfile.TemporaryDirectory(prefix='cap-manual2-', dir='/private/tmp') as directory:
            root = Path(directory)
            path = root / 'monitor_integrity.json'
            path.write_text(json.dumps({'files': {}}))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch.object(wrapper, 'OUTPUT', root), patch.object(wrapper, 'routed_helpers') as routing:
                with self.assertRaisesRegex(ValueError, 'wrapper/launcher missing'):
                    wrapper.advance(digest, 'b' * 64)
            routing.assert_not_called()

    def test_one_round_then_one_waiting_collection_uses_explicit_manual2_paths(self):
        launcher = ModuleType('paper.scripts.launch_titantpp_gru_cap_manual2')
        observer, collector = Mock(), Mock()
        observed = {'round_path': str(wrapper.OUTPUT / 'rounds/fabricated.json'),
                    'round_raw_sha256': 'c' * 64}
        observer.observe.return_value = observed
        collector.collect.return_value = {'status': 'waiting_for_terminal_training', 'remote_calls': 0}
        with patch.dict(sys.modules, {launcher.__name__: launcher}), \
             patch.object(wrapper, 'verify_extra_seal') as seal, \
             patch.object(wrapper, 'routed_helpers', return_value=(observer, collector)), \
             patch.object(wrapper, 'checked_round') as checked:
            result = wrapper.advance('a' * 64, 'b' * 64)
        self.assertEqual(seal.call_count, 2)
        collector.verify_seal.assert_called_once_with(wrapper.BUNDLE, wrapper.OUTPUT, 'b' * 64)
        observer.observe.assert_called_once_with(wrapper.BUNDLE, wrapper.OUTPUT,
            monitor_integrity_sha='a' * 64)
        checked.assert_called_once_with(observed)
        collector.collect.assert_called_once_with(Path(observed['round_path']), 'c' * 64, 'b' * 64,
            bundle=wrapper.BUNDLE, monitor=wrapper.OUTPUT, report=wrapper.REPORT)
        self.assertEqual(result['collection']['remote_calls'], 0)

    def test_round_pointer_or_raw_sha_drift_blocks_collection(self):
        with tempfile.TemporaryDirectory(prefix='cap-round-', dir='/private/tmp') as directory:
            root = Path(directory)
            (root / 'rounds').mkdir()
            path = root / 'rounds/one.json'
            path.write_text('{}')
            markdown = path.with_suffix('.md')
            markdown.write_text('fabricated')
            latest = root / 'latest.json'
            result = {'round_path': str(path), 'round_raw_sha256': wrapper.sha(path),
                'markdown_path': str(markdown), 'markdown_raw_sha256': wrapper.sha(markdown),
                'contract_sha256': 'c' * 64, 'latest_path': str(latest)}
            latest.write_text(json.dumps({**result, 'round_raw_sha256': 'wrong'}))
            result['latest_raw_sha256'] = wrapper.sha(latest)
            with patch.object(wrapper, 'OUTPUT', root):
                with self.assertRaisesRegex(ValueError, 'latest pointer differs'):
                    wrapper.checked_round(result)
                path.write_text('{"tampered":true}')
                with self.assertRaisesRegex(ValueError, 'artifact SHA changed'):
                    wrapper.checked_round(result)


if __name__ == '__main__':
    unittest.main()
