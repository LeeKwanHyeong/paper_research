"""Offline mutations of audit metadata; no inference or source evidence edits."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class AuditGuards(unittest.TestCase):
    def reject_before_test(self, mutation):
        report = module(HERE / 'build_report.py', 'guarded_report')
        audit = module(report.BUNDLE / 'operation/evaluate_titantpp_cnn_gru_frozen_test.py', 'guarded_evidence')
        original_read, read_paths = audit.read, []

        def read(path):
            read_paths.append(str(path))
            value = deepcopy(original_read(path))
            mutation(Path(path), value)
            return value

        def load(path, name):
            if name != 'test3seed_evidence_audit':
                self.fail('Reference Test source opened after a rejected gate')
            return audit

        with patch.object(audit, 'read', read), patch.object(report, 'module', load):
            with self.assertRaises(ValueError):
                report.run()
        self.assertFalse(any('/runs/test/' in path or path.endswith('/inference_completion.json') for path in read_paths))

    def test_failed_validation_gate(self):
        self.reject_before_test(lambda path, value: value.update(passed=False) if path.name == 'validation_gate.json' else None)

    def test_foreign_gate_contract(self):
        self.reject_before_test(lambda path, value: value.update(contract_sha256='foreign') if path.name == 'validation_gate.json' else None)

    def test_duplicate_validation_condition(self):
        self.reject_before_test(lambda path, value: value['rows'].__setitem__(1, deepcopy(value['rows'][0])) if path.name == 'validation_gate.json' else None)

    def test_validation_gate_metric_tampering(self):
        def mutate(path, value):
            if path.name == 'validation_gate.json':
                value['rows'][0]['metrics']['qty_rmse'] += 1
        self.reject_before_test(mutate)

    def test_actual_exit_failure(self):
        self.reject_before_test(lambda path, value: value.update(returncode=1) if path.name == 'supervisor_process_exit.json' else None)

    def test_process_still_owned(self):
        self.reject_before_test(lambda path, value: value.update(owned_processes=[{'pid': 123}]) if path.name == 'native_terminal_observation.json' else None)

    def test_authorization_missing(self):
        def mutate(path, value):
            if path.name == 'execution_contract.json':
                value['approval']['test_inference_authorized'] = False
        self.reject_before_test(mutate)

    def test_archive_sha_tampering(self):
        self.reject_before_test(lambda path, value: value.update(archive_sha256='foreign') if path.name == 'retrieval_receipt.json' else None)


if __name__ == '__main__':
    unittest.main()
