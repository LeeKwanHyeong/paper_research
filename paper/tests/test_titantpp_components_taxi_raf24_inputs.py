"""Synthetic source/data/original boundaries for the native prediction0 audit."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import polars as pl
import torch
from paper.scripts import audit_titantpp_components_taxi_raf24_inputs as audit

PROJECT = Path(__file__).resolve().parents[2]


def frame():
    return pl.DataFrame({'oper_part_no': ['a', 'a', 'a'], 'seq': [0, 1, 2],
                         'delta_t': [1, 2, 3], 'demand_qty': [2., 3., 4.],
                         'chronological_split': ['train', 'validation', 'test'],
                         'demand_dt': ['2000-01-01', '2000-01-02', '2000-01-03']})


class NativeInputsBoundaryTests(unittest.TestCase):
    def jobs(self):
        return [{'id': f'{d}__{s}__{a}', 'dataset': d, 'seed': s, 'arm': a, 'host': '5080'}
                for d in audit.DATASETS for s in audit.SEEDS for a in audit.ARMS]

    def test_exact24_scope(self):
        jobs = self.jobs()
        approved = sorted(j['id'] for j in jobs)
        self.assertEqual(audit.validate_job_scope({'jobs': jobs}, {'job_ids': approved}), approved)

    def test_extra_wrong_seed_is_rejected(self):
        jobs = self.jobs()
        jobs[0]['id'] = jobs[0]['id'].replace('__42__', '__72__')
        with self.assertRaisesRegex(ValueError, 'Exact Taxi/RAF24'):
            audit.validate_job_scope({'jobs': jobs}, {'job_ids': sorted(j['id'] for j in jobs)})

    def test_wrong_host_is_rejected(self):
        jobs = self.jobs()
        approval = {'job_ids': sorted(j['id'] for j in jobs)}
        jobs[0]['host'] = '5090'
        with self.assertRaises(ValueError):
            audit.validate_job_scope({'jobs': jobs}, approval)

    def test_missing_or_duplicate_job_is_rejected(self):
        for jobs in (self.jobs()[:-1], self.jobs() + [self.jobs()[0]]):
            with self.assertRaises(ValueError):
                audit.validate_job_scope({'jobs': jobs}, {'job_ids': sorted(j['id'] for j in jobs)})

    def test_raf_exact_train_validation_slice(self):
        full = frame()
        training = full.filter(pl.col('chronological_split') != 'test')
        audit.require_raf_training_slice(full, training, pl)
        # Additional full-parent columns do not alter the frozen export.
        audit.require_raf_training_slice(full.with_columns(pl.lit('x').alias('extra')), training, pl)

    def test_raf_changed_quantity_rejected(self):
        full = frame()
        training = full.filter(pl.col('chronological_split') != 'test').with_columns((pl.col('demand_qty') + 1).alias('demand_qty'))
        with self.assertRaisesRegex(ValueError, 'differs from frozen export'):
            audit.require_raf_training_slice(full, training, pl)

    def test_raf_missing_row_rejected(self):
        with self.assertRaises(ValueError):
            audit.require_raf_training_slice(frame(), frame().head(1), pl)

    def test_raf_missing_parent_column_rejected(self):
        with self.assertRaisesRegex(ValueError, 'misses frozen export columns'):
            audit.require_raf_training_slice(frame().drop('demand_dt'), frame().head(2), pl)

    def test_validation_cannot_materialize_test(self):
        definition = {'model': {'time_observation_contract': {'top_code': None}}}
        with self.assertRaisesRegex(ValueError, 'Unapproved split'):
            audit.validate_frame(frame(), definition, 'validation', np, pl)

    def test_chronological_backwards_split_rejected(self):
        full = frame().with_columns(pl.Series('chronological_split', ['train', 'test', 'validation']))
        definition = {'model': {'time_observation_contract': {'top_code': None}}}
        with self.assertRaisesRegex(ValueError, 'out of order'):
            audit.validate_frame(full, definition, 'test', np, pl)

    def test_science_boundary_functions_are_identical_to_base(self):
        base = ast.parse((PROJECT / 'paper/scripts/evaluate_titantpp_components_frozen.py').read_text())
        current = ast.parse(Path(audit.__file__).read_text())
        for name in ('validate_frame', 'require_raf_training_slice', 'population_identity'):
            left = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == name)
            right = next(n for n in current.body if isinstance(n, ast.FunctionDef) and n.name == name)
            self.assertEqual(ast.dump(left, include_attributes=False), ast.dump(right, include_attributes=False))

    def test_synthetic_fingerprints_preserve_original_duration_scheme(self):
        metadata = frame().head(2)
        dataset = SimpleNamespace(seq_lists=[[0, 1]], index=[(0, 0)], parts=['a'])
        quantities = torch.tensor([[2., 3.]])
        sample = (torch.tensor([0]), torch.tensor([[1., 2.]]), torch.tensor([[True, True]]),
                  torch.tensor([0]), quantities)
        class Loader:
            _audit_split = 'validation'
            def __init__(self):
                self.dataset = dataset
            def __iter__(self):
                return iter([sample])
        # len is provided by a minimal mutable dataset class.
        class Dataset:
            seq_lists, index, parts = [[0, 1]], [(0, 0)], ['a']
            def __len__(self):
                return 1
        dataset = Dataset()
        result = audit.fingerprints(Loader(), 'synthetic', 'f' * 64, metadata, np)
        expected = hashlib.sha256(b'frozen_lognormal_target_dt_v1\0' + np.array([2.], dtype='<f8').tobytes()).hexdigest()
        self.assertEqual(result['original_recorded_duration_sha256'], expected)
        self.assertEqual(result['count'], 1)

    def test_sealed_real_packet_reads_no_checkpoint_or_data(self):
        path = PROJECT / 'reports/titantpp_components_taxi_raf24_test_20261014_v1/prepare_original_retrieval.py'
        spec = importlib.util.spec_from_file_location('taxi_raf_original_packet', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        packet = module.create_packet()
        self.assertEqual(len(packet['expected_files']), 360)
        self.assertEqual(len(packet['job_ids']), 24)
        self.assertEqual(packet['model_predictions'], 0)
        self.assertEqual(packet['original_root_writes'], 0)


if __name__ == '__main__':
    unittest.main()
