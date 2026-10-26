#!/usr/bin/env python3
"""Synthetic collector tests. No research checkpoint/data or encoder is loaded."""
from pathlib import Path
import importlib
import json
import os
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import collector_v2 as c


class FakePrototype(nn.Module):
    def __init__(self):
        super().__init__()
        self.mem = nn.Parameter(torch.arange(64 * 64, dtype=torch.float32).reshape(1, 64, 64) / 100000)
        self.topk = 4

    def retrieve(self, value):
        ids = torch.tensor([0, 1, 2, 3]).view(1, 1, 4).expand(value.shape[0], value.shape[1], -1)
        residual = self.mem[0][ids].mean(2)
        cosine = torch.tensor([.9, .8, .7, .6]).view(1, 1, 4).expand_as(ids)
        return residual, {'prototype_indices': ids, 'topk_similarity': cosine}

    def forward(self, value):
        residual, _ = self.retrieve(value)
        return value + residual


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lmm = FakePrototype()
        self.quantity_head = nn.Linear(64, 1)
        nn.init.constant_(self.quantity_head.weight, .01)
        nn.init.zeros_(self.quantity_head.bias)
        self.calls = 0

    def encode_task_states(self, dts, q, mask, *, memory_write_mask=None):
        self.calls += 1
        base = torch.arange(64, dtype=torch.float32).view(1, 1, 64).expand(*dts.shape, 64) / 100
        result = self.lmm(base)
        return result, result

    def _lognormal_time_terms(self, hidden, target):
        mu = hidden.double().mean(1)
        sigma = .001 + F.softplus(hidden.double().sum(1) / 64)
        return mu, sigma, torch.log(target.double())

    def quantity_outputs(self, hidden, target):
        loc = F.softplus(self.quantity_head(hidden).squeeze(-1))
        loss = (loc - torch.log1p(target.clamp_min(0))).square()
        return {'location': loc, 'point_prediction': torch.expm1(loc), 'train_loss': loss}


def fake_native(model, leak=False):
    mask = torch.zeros(3, 256, dtype=torch.bool)
    lengths = torch.tensor([3, 5, 7])
    for i, length in enumerate(lengths): mask[i, :length] = True
    dts = torch.ones(3, 256); q = torch.ones(3, 256)
    ids = torch.arange(3); target_positions = lengths - 1
    q[ids, target_positions] = 1. if leak else 0.
    observed = mask.clone(); observed[ids, target_positions] = False
    t, qstate = model.encode_task_states(dts, q, mask, memory_write_mask=observed)
    h = qstate[ids, lengths - 2]; gap = dts[ids, target_positions]
    mu, sigma, _ = model._lognormal_time_terms(t[ids, lengths - 2], gap)
    target_q = torch.tensor([1., 2., 3.]); quantity = model.quantity_outputs(h, target_q)
    return {'true_qty': target_q, 'pred_qty': quantity['point_prediction'],
            'log_qty_loss': quantity['train_loss'], 'quantity_train_loss': quantity['train_loss'],
            'time_loss': mu + sigma, 'history_length': lengths - 1}


class CollectorTests(unittest.TestCase):
    def test_source_inventory_rejects_extra_missing_changed_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp); p = source / 'module.py'; p.write_text('value = 1\n')
            manifest = {'module.py': c.sha(p)}
            c.verify_source(source, manifest)
            (source / 'extra.py').write_text('')
            with self.assertRaisesRegex(ValueError, 'inventory'): c.verify_source(source, manifest)
            (source / 'extra.py').unlink(); p.write_text('value = 2\n')
            with self.assertRaisesRegex(ValueError, 'SHA'): c.verify_source(source, manifest)
            p.unlink()
            with self.assertRaisesRegex(ValueError, 'inventory'): c.verify_source(source, manifest)
            p.write_text('value = 1\n'); (source / 'alias.py').symlink_to(p)
            with self.assertRaisesRegex(ValueError, 'Symlink'): c.verify_source(source, manifest)

    def test_namespace_import_ignores_foreign_path_added_by_legacy_bootstrap(self):
        before_modules = set(sys.modules); before_paths = sys.path[:]; before_meta = sys.meta_path[:]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp).resolve(); source = base / 'source'; foreign = base / 'foreign'
                for root in (source, foreign): (root / 'paper/scripts').mkdir(parents=True)
                (source / 'paper/scripts/good.py').write_text('VALUE = "frozen"\n')
                (foreign / 'paper/scripts/good.py').write_text('VALUE = "foreign"\n')
                marker = base / 'FOREIGN_EXECUTED'
                (foreign / 'paper/scripts/evil.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).touch()\n')
                manifest = {str(p.relative_to(source)): c.sha(p) for p in source.rglob('*') if p.is_file()}
                c.install_frozen_imports(source, manifest)
                importlib.import_module('paper.scripts')
                sys.path.insert(0, str(foreign))
                good = importlib.import_module('paper.scripts.good')
                self.assertEqual(good.VALUE, 'frozen')
                self.assertEqual(list(sys.modules['paper'].__path__), [str(source / 'paper')])
                self.assertEqual(c.verify_imports(source, manifest)['paper.scripts.good'], 'paper/scripts/good.py')
                with self.assertRaisesRegex(ModuleNotFoundError, 'absent'): importlib.import_module('paper.scripts.evil')
                self.assertFalse(marker.exists())
        finally:
            sys.path[:] = before_paths; sys.meta_path[:] = before_meta
            for name in set(sys.modules) - before_modules:
                if name.split('.')[0] in c.FROZEN_PACKAGES: sys.modules.pop(name)

    def test_preloaded_foreign_module_is_rejected(self):
        with mock.patch.dict(sys.modules, {'paper': object()}):
            with self.assertRaisesRegex(ValueError, 'not source-isolated'):
                c.install_frozen_imports(Path('/unused'), {})

    def test_input_layout_uses_original_relative_paths_and_checks_sidecar(self):
        old = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp).resolve(); source = base / 'source'; source.mkdir()
                inputs = base / 'inputs/data/intermittent_frozen_5000'; inputs.mkdir(parents=True)
                parquet = inputs / 'train_validation.parquet'; parquet.write_bytes(b'synthetic parquet fixture')
                sidecar = inputs / 'split_manifest.json'; sidecar.write_text('{}')
                os.chdir(base)
                spec = {'working_dir': str(base), 'split_manifest_record': c.file_record(sidecar)}
                data = {'prepared_train_validation_path': 'inputs/data/intermittent_frozen_5000/train_validation.parquet',
                        'inherited_data_identity': {'data': {'path': str(parquet.relative_to(base)), 'sha256': c.sha(parquet)},
                         'split_manifest': {'path': str(sidecar.relative_to(base)), 'sha256': c.sha(sidecar)}}}
                job = {'data_override': {'prepared_train_validation_path': str(parquet)}}
                self.assertEqual(c.bind_input_layout(spec, job, data, source), (parquet, base / 'inputs/data', sidecar))
                sidecar.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'Bound file changed'):
                    c.bind_input_layout(spec, job, data, source)
        finally: os.chdir(old)

    def test_order_contains_every_target_once_and_original_sample_first(self):
        sample = np.random.default_rng(3).choice(5000, 2048, replace=False)
        order = c.fixed_order(5000, sample, 'full', 1, 128)
        np.testing.assert_array_equal(order[:2048], sample)
        np.testing.assert_array_equal(np.sort(order), np.arange(5000))
        self.assertTrue(np.all(np.diff(order[2048:]) > 0))

    def test_qualification_is_sample_prefix(self):
        sample = np.arange(2048)[::-1]
        np.testing.assert_array_equal(c.fixed_order(3000, sample, 'qualification', 1, 128), sample[:128])

    def test_repeated_reference_target_is_rejected(self):
        with self.assertRaises(ValueError): c.fixed_order(3000, [1] * 2048, 'full', 1, 128)

    def test_observer_keeps_original_outputs_and_single_forward(self):
        model = FakeModel().eval().requires_grad_(False)
        before = c.state_sha(model)
        with torch.no_grad():
            expected = fake_native(model)
            with c.NativeObserver(model) as observer: actual = fake_native(model)
            observed = observer.validate(actual)
        self.assertTrue(all(torch.equal(expected[k], actual[k]) for k in actual))
        self.assertEqual(model.calls, 2)  # One separate uninstrumented synthetic control, one observed call.
        self.assertEqual(observer.calls, {'encode': 1, 'time': 1, 'quantity': 1, 'retrieve': 1, 'head': 1})
        self.assertEqual(observed['time_location_float64'].dtype, torch.float64)
        self.assertEqual(observed['head_hidden'].dtype, torch.float32)
        self.assertEqual(c.state_sha(model), before)
        for name in ('encode_task_states', '_lognormal_time_terms', 'quantity_outputs'):
            self.assertNotIn(name, model.__dict__)
        self.assertNotIn('retrieve', model.lmm.__dict__)

    def test_observer_rejects_target_quantity_leak_and_restores_wrappers(self):
        model = FakeModel().eval()
        with self.assertRaises(ValueError):
            with torch.no_grad(), c.NativeObserver(model): fake_native(model, leak=True)
        self.assertNotIn('encode_task_states', model.__dict__)
        self.assertNotIn('retrieve', model.lmm.__dict__)

    def example_arrays(self):
        model = FakeModel().eval().requires_grad_(False)
        with torch.no_grad(), c.NativeObserver(model) as observer: outputs = fake_native(model)
        observed = observer.validate(outputs)
        identities = [{'target_index': i, 'entity_id': f's{i}', 'target_seq': i + 7, 'raw_quantity': float(i + 1)} for i in range(3)]
        arrays, hidden = c.export_batch(outputs, observed, identities)
        return arrays, hidden

    def test_chunk_preserves_dtypes_keys_and_hidden_alignment(self):
        import polars as pl
        arrays, hidden = self.example_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            writer = c.ChunkWriter(Path(tmp), 'train', 4, save_hidden=True)
            writer.append(arrays, hidden); writer.flush()
            self.assertEqual(writer.total, 3)
            record = writer.records[0]
            self.assertEqual(c.sha(record['observations']['path']), record['observations']['sha256'])
            frame = pl.read_parquet(record['observations']['path'])
            self.assertEqual(frame['time_location_float64'].dtype, pl.Float64)
            self.assertEqual(frame['pred_quantity'].dtype, pl.Float32)
            np.testing.assert_array_equal(frame['target_index'].to_numpy(), arrays['target_index'])
            with np.load(record['head_hidden']['path'], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['head_hidden'], hidden)
                np.testing.assert_array_equal(saved['target_index'], arrays['target_index'])
            second = c.ChunkWriter(Path(tmp), 'train', 4, save_hidden=True)
            second.append(arrays, hidden)
            with self.assertRaises(FileExistsError): second.flush()

    def test_validation_does_not_write_hidden(self):
        arrays, hidden = self.example_arrays()
        with tempfile.TemporaryDirectory() as tmp:
            writer = c.ChunkWriter(Path(tmp), 'validation', 4096, save_hidden=False)
            writer.append(arrays, hidden); writer.flush()
            self.assertNotIn('head_hidden', writer.records[0])
            self.assertEqual(list(Path(tmp).glob('*.npz')), [])

    def test_parity_failure_is_preserved_without_tolerance_changes(self):
        arrays, hidden = self.example_arrays()
        references = [{k: arrays[k][i].item() if isinstance(arrays[k][i], np.generic) else arrays[k][i]
                       for k in (*c.IDENTITY, 'history_length', 'true_quantity', *c.REFERENCE_SCALARS)} for i in range(3)]
        for i, ref in enumerate(references):
            ref['prototype_indices'] = arrays['prototype_indices'][i].tolist()
            ref['topk_similarity'] = arrays['topk_similarity'][i].tolist()
        tolerances = {k: {'atol': 1e-7, 'rtol': 1e-5} for k in (*c.REFERENCE_SCALARS, 'topk_similarity')}
        with tempfile.TemporaryDirectory() as tmp:
            c.compare_reference(arrays, hidden, references, tolerances, Path(tmp), 'ok')
            references[0]['pred_quantity'] += 1.
            with self.assertRaises(ValueError): c.compare_reference(arrays, hidden, references, tolerances, Path(tmp), 'bad')
            self.assertTrue((Path(tmp) / 'PARITY_FAILURE_bad.json').is_file())

    def test_top4_order_has_no_silent_tie_exception(self):
        arrays, hidden = self.example_arrays()
        references = []
        for i in range(3):
            ref = {k: arrays[k][i].item() if isinstance(arrays[k][i], np.generic) else arrays[k][i]
                   for k in (*c.IDENTITY, 'history_length', 'true_quantity', *c.REFERENCE_SCALARS)}
            ref.update(prototype_indices=arrays['prototype_indices'][i].tolist(), topk_similarity=arrays['topk_similarity'][i].tolist())
            references.append(ref)
        references[0]['prototype_indices'] = [1, 0, 2, 3]
        tolerances = {k: {'atol': 1e-7, 'rtol': 1e-5} for k in (*c.REFERENCE_SCALARS, 'topk_similarity')}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError): c.compare_reference(arrays, hidden, references, tolerances, Path(tmp), 'order')


if __name__ == '__main__': unittest.main(verbosity=2)
