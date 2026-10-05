"""Focused pure contract checks; no models, datasets, GPU, or network."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('local_prepare', Path(__file__).resolve().parents[1] / 'prepare.py')
P = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(P)

class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.old = P.read(P.OLD / 'execution_contract.json')
        self.parent = P.read(P.PROJECT / 'reports/titantpp_cnn_gru_a100_seed42_20261004_v1/execution_contract.json')
        self.c = copy.deepcopy(self.old)
        self.c.update(schema=P.NAME, arms=[P.ARM], jobs=P.expected_jobs(), canonical_conditions=9,
                      new_fits=6, already_terminal_reused=3)
        self.c['reuse'] = [x for x in self.c['reuse'] if x['seed'] in (52, 62)]
        files = {'operation/' + name: 'a' * 64 for name in ('campaign.py', 'diagnostic_adapter.py', 'monitor.py')}
        self.c['operation'] = {'files': files, 'files_sha256': P.digest(files)}
        self.c['anchors42'] = {}
        for d, epochs in {'yellow_trip_hourly': (126, 166), 'raf_spare_parts': (9, 49),
                          'intermittent_frozen_5000': (12, 52)}.items():
            selected, last = f'inputs/{d}/best.pt', f'inputs/{d}/last.pt'
            self.c['input_files'].update({selected: 'b' * 64, last: 'c' * 64})
            self.c['anchors42'][d] = dict(checkpoint=selected, checkpoint_sha256='b' * 64,
                last_checkpoint=last, last_checkpoint_sha256='c' * 64, selected_epoch=epochs[0], last_epoch=epochs[1])

    def check(self):
        return P.validate_contract(self.c, self.old, self.parent)

    def test_exact_scope_accepted(self):
        self.assertEqual(self.check()['status'], 'passed')

    def test_scope_selector_runtime_and_source_drift_rejected(self):
        mutations = [lambda c: c['jobs'][0].update(seed=42),
                     lambda c: c['training'].update(monitor='test_rmse'),
                     lambda c: c['hosts']['5080'].update(python='/other/python'),
                     lambda c: c['source']['files'].update({'unexpected.py': 'a' * 64}),
                     lambda c: c.update(held_out_test_evaluated=True),
                     lambda c: c['anchors42']['raf_spare_parts'].update(selected_epoch=10),
                     lambda c: c['limits'].update(automatic_retry=True)]
        valid = copy.deepcopy(self.c)
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.c = copy.deepcopy(valid)
                mutation(self.c)
                with self.assertRaises(ValueError): self.check()

    def test_canonical_json_is_key_order_independent(self):
        self.assertEqual(P.digest({'a': 1, 'b': 2}), P.digest({'b': 2, 'a': 1}))

    def test_freeze_can_repeat_but_never_silently_change(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / 'fixture.json'
            P.save(path, {'x': 1})
            P.save(path, {'x': 1})
            with self.assertRaises(ValueError): P.save(path, {'x': 2})

if __name__ == '__main__': unittest.main()
