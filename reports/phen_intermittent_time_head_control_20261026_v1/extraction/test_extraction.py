import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import collector_hidden as c
import prepare_spec


def fixture(n=4):
    arrays = {'target_index': np.arange(n, dtype=np.int64), 'entity_id': np.array([str(i) for i in range(n)]),
              'target_seq': np.arange(n, dtype=np.int64) + 3, 'raw_quantity': np.ones(n),
              'history_length': np.ones(n, dtype=np.int64) * 2, 'target_gap': np.ones(n, dtype=np.int64),
              'pred_quantity': np.ones(n), 'time_nll': np.ones(n) * 2,
              'time_location_float64': np.ones(n) * 3, 'time_sigma_float64': np.ones(n) * 4}
    return arrays


class ExtractionTests(unittest.TestCase):
    def test_parity_and_failure_preservation(self):
        a = fixture()
        h = np.zeros((4, 64), dtype=np.float32)
        r = c.reference_rows(a, [0, 1, 2, 3])
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            self.assertEqual(c.compare_reference(a, h, r, c.TOLERANCES, out, 'ok')['time_nll'], 0)
            changed = {**a, 'time_nll': a['time_nll'].copy()}
            changed['time_nll'][1] += .1
            with self.assertRaises(ValueError): c.compare_reference(changed, h, r, c.TOLERANCES, out, 'bad')
            self.assertTrue((out / 'PARITY_FAILURE_bad.json').is_file())
            with self.assertRaises(ValueError): c.compare_reference(a, h, r, {**c.TOLERANCES, 'time_nll': {'atol': 1., 'rtol': 1.}}, out, 'loose')

    def test_parameter_precision_is_diagnostic_not_unapproved_gate(self):
        a = fixture()
        r = c.reference_rows(a, [0, 1, 2, 3])
        a['time_location_float64'] += .2
        with tempfile.TemporaryDirectory() as tmp:
            result = c.compare_reference(a, np.zeros((4,64)), r, c.TOLERANCES, Path(tmp), 'diagnostic')
        self.assertAlmostEqual(result['time_location_float64'], .2)

    def test_identity_mismatch_never_float_tolerated(self):
        a = fixture()
        r = c.reference_rows(a, [0, 1, 2, 3])
        a['target_seq'][0] += 1
        with self.assertRaises(ValueError): c.compare_reference(a, np.zeros((4,64)), r, c.TOLERANCES, Path('.'), 'identity')

    def test_scalar_npz_schema_and_full_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ref.npz'
            np.savez_compressed(path, **fixture())
            a = c.read_reference(c.file_record(path))
            self.assertEqual(set(a), set(c.REFERENCE_FIELDS))
            bad = fixture(); bad['target_index'][0] = 1
            np.savez_compressed(path, **bad)
            with self.assertRaises(ValueError): c.read_reference(c.file_record(path))

    def test_chunk_hidden_float32_and_identity_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = c.ChunkWriter(Path(tmp), 'validation', 4096, True)
            a = fixture(); a['true_quantity'] = a['raw_quantity']
            h = np.arange(256, dtype=np.float32).reshape(4,64)
            w.append(a, h); w.flush()
            self.assertEqual(w.total, 4)
            with np.load(w.records[0]['head_hidden']['path'], allow_pickle=False) as n:
                self.assertEqual(n['head_hidden'].dtype, np.float32)
                np.testing.assert_array_equal(n['head_hidden'], h)
                np.testing.assert_array_equal(n['target_index'], a['target_index'])

    def test_fixed_order_smoke_full_exhaustive(self):
        r = np.arange(2048)[::-1]
        smoke = c.fixed_order(3000, r, 'qualification', 16, 128)
        full = c.fixed_order(3000, r, 'full', 16, 128)
        np.testing.assert_array_equal(smoke, r)
        np.testing.assert_array_equal(full[:2048], r)
        np.testing.assert_array_equal(np.sort(full), np.arange(3000))

    def test_smoke_gate_requires_all_rows_and_same_job(self):
        spec = {'mode': 'full', 'runner_sha256': 'runner'}
        job = {'id': 'job', 'splits': ['train', 'validation']}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            done = {'status': 'qualification_complete', 'qualification_binding': c.qualification_binding(spec, job),
                    'input_and_model_immutability_verified': True, 'context_state_unchanged': True,
                    'splits': {s: {'rows': 2048, 'native_reference_rows_checked': 2048} for s in job['splits']}}
            exit_record = {'job': 'job', 'actual_child_exit': 0, 'collector_sha256': 'runner'}
            (p/'done.json').write_text(json.dumps(done)); (p/'exit.json').write_text(json.dumps(exit_record))
            spec['qualification_gate'] = {'job': {'job_complete': c.file_record(p/'done.json'), 'actual_exit': c.file_record(p/'exit.json')}}
            self.assertIsNotNone(c.verify_qualification(spec, job))
            job2 = {**job, 'splits': ['validation']}
            with self.assertRaises(ValueError): c.verify_qualification(spec, job2)
            done['splits']['train']['rows'] = 128
            (p/'done.json').write_text(json.dumps(done))
            spec['qualification_gate']['job']['job_complete'] = c.file_record(p/'done.json')
            with self.assertRaises(ValueError): c.verify_qualification(spec, job)

    def test_observer_final_common_input_without_retrieval_capture(self):
        import torch

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.quantity_head = torch.nn.Linear(64, 1)
            def encode_task_states(self, dts, quantities, mask, *, memory_write_mask=None): return None
            def _lognormal_time_terms(self, hidden, target):
                return torch.zeros(len(hidden), dtype=torch.float64), torch.ones(len(hidden), dtype=torch.float64), None
            def quantity_outputs(self, hidden, target): return self.quantity_head(hidden)

        model = Toy().eval().requires_grad_(False)
        before = c.state_sha(model)
        mask = torch.zeros((2,256), dtype=torch.bool); mask[:,:3] = True
        quantities = torch.ones((2,256)); quantities[:,2] = 0
        dts = torch.ones((2,256)); write = mask.clone(); write[:,2] = False
        hidden = torch.arange(128, dtype=torch.float32).reshape(2,64)
        with torch.no_grad(), c.NativeObserver(model) as observer:
            model.encode_task_states(dts, quantities, mask, memory_write_mask=write)
            model.quantity_outputs(hidden, torch.ones(2))
            model._lognormal_time_terms(hidden, torch.ones(2))
        v = observer.validate({'history_length': torch.tensor([2,2])})
        self.assertTrue(torch.equal(v['head_hidden'], hidden))
        self.assertNotIn('prototype_indices', v)
        self.assertEqual(c.state_sha(model), before)
        self.assertNotIn('_lognormal_time_terms', model.__dict__)

    def test_prepared_specs_select_only_six_missing_cohorts(self):
        root = Path(__file__).resolve().parent
        science = root.parent / 'SCIENCE_SPEC.json'
        references = root / 'references_v1/REFERENCES.json'
        self.assertTrue(science.is_file() and references.is_file())
        cohorts = []
        for seed in (52, 42):
            remote = f'/tmp/phen_intermittent_time_head_control_test_seed{seed}'
            spec, context = prepare_spec.prepare(seed, remote, science, str(science), references, 'qualification')
            c.validate_spec(spec)
            self.assertEqual(spec['allowed_output_root'], remote)
            self.assertEqual(context['host'], '5090' if seed == 52 else '5080')
            for job in spec['jobs']:
                self.assertIn(f'source{seed}', job['source_root'])
                for split in job['splits']: cohorts.append((seed, job['checkpoint_kind'], split))
            with tempfile.TemporaryDirectory() as tmp:
                gate = Path(tmp) / 'gate.json'
                gate.write_text(json.dumps({j['id']: {'job_complete': {}, 'actual_exit': {}} for j in spec['jobs']}))
                full, _ = prepare_spec.prepare(seed, remote, science, str(science), references, 'full', gate)
                c.validate_spec(full)
                for job in spec['jobs']:
                    self.assertEqual(c.qualification_binding(spec, job), c.qualification_binding(full, job))
        self.assertEqual(cohorts, [(s,k,p) for s in (52,42) for k,p in [('best','validation'),('last','train'),('last','validation')]])


if __name__ == '__main__': unittest.main()
