"""Offline contract/ownership gates for the twelve-fit RTX handoff."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy

PROJECT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('rtx_bundle_prepare',
    PROJECT / 'paper/scripts/prepare_titantpp_history_rtx_handoff.py')
g = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(g)


class RTXBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = g.read(g.RTX / 'execution_contract.json')
        cls.original = (g.RTX / 'operation/campaign.py').read_text()
        cls.parallel = (g.PARALLEL / 'operation/campaign.py').read_text()
        cls.hosts = g.host_specs(cls.parent)
        cls.runtime = g.build_runtime(cls.original, cls.parallel, cls.parent, cls.hosts)
        cls.contract = g.make_contract(cls.parent, cls.hosts, {
            'operation/campaign.py': hashlib.sha256(cls.runtime.encode()).hexdigest(),
            'operation/rtx_gpu_operations.py': '0' * 64})
        cls.namespace = g.validate_local_contract(cls.runtime, cls.contract)

    def test_exact_original_twelve_and_same_host_triplets(self):
        original_ids = {j['id'] for j in self.parent['jobs']}
        jobs = self.contract['jobs']
        self.assertEqual(len(jobs), 12)
        self.assertEqual(len({j['id'] for j in jobs}), 12)
        self.assertTrue(all(j['id'] in original_ids for j in jobs))
        self.assertEqual([len(self.namespace['batch_groups'](self.contract, h))
            for h in ('5080', '5090')], [1, 3])

    def test_scientific_fit_and_native_qualification_unchanged(self):
        marker = "    budget();data=next(d for d in c['datasets'] if d['dataset_id']==job['dataset'])"
        self.assertEqual(g.function(self.original, 'run_fit').split(marker, 1)[1],
            g.function(self.runtime, 'run_fit').split(marker, 1)[1])
        for name in ('qualify', 'initial_states', 'training_args', 'replay_checkpoint',
                     'verify_full_validation', 'strict_training_seed'):
            self.assertEqual(g.function(self.original, name), g.function(self.runtime, name))
        self.assertEqual(g.scientific_fingerprint(self.parent), g.scientific_fingerprint(self.contract))

    def test_unstarted_queue_does_not_add_raf52_or_taxi(self):
        self.assertEqual({(j['dataset'], j['seed']) for j in self.contract['jobs']},
            {('raf_spare_parts', 62), ('intermittent_frozen_5000', 42),
             ('intermittent_frozen_5000', 52), ('intermittent_frozen_5000', 62)})

    def test_host_move_or_extra_fit_is_rejected(self):
        for mutation in ('host', 'extra'):
            c = deepcopy(self.contract)
            if mutation == 'host':
                c['jobs'][0]['host'] = '5090'
            else:
                c['jobs'].append(deepcopy(c['jobs'][0]))
            with self.assertRaises(ValueError):
                self.namespace['validate'](c, verify_files=False)

    def test_gpu_runtime_loss_or_optimizer_mutation_rejected(self):
        for mutation in ('gpu', 'runtime', 'optimizer'):
            c = deepcopy(self.contract)
            if mutation == 'gpu':
                c['hosts']['5090']['gpu_uuid'] = 'GPU-foreign'
            elif mutation == 'runtime':
                c['hosts']['5090']['runtime_expected']['cudnn'] = 91900
            else:
                c['datasets'][0]['optimizer']['lr'] = .002
            with self.assertRaises(ValueError):
                self.namespace['validate'](c, verify_files=False)

    def test_missing_ownership_release_blocks_any_gpu_work(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                self.namespace['verify_ownership_release'](self.contract, Path(folder))

    def release_fixture(self, folder):
        root = Path(folder)
        proof = {'A100_reserved_execution_ids': [j['id'] for j in self.contract['jobs']],
                 'original_receipt': 'Native A100 exclusion evidence retained unchanged'}
        g.write_json(root / 'handoff/ownership_release_source.json', proof)
        release = {'contract_sha256': g.canonical(self.contract),
            'parent_contract_sha256': g.PARENT_SHA,
            'execution_ids': [j['id'] for j in self.contract['jobs']],
            'A100_queue_exclusion_verified': True, 'unstarted_verified': True,
            'released_to_rtx': True,
            'source_receipt_raw_sha256': g.sha(root / 'handoff/ownership_release_source.json')}
        g.write_json(root / 'ownership_release.json', release)
        return root, release

    def test_bound_release_accepts_and_raw_change_rejects(self):
        with tempfile.TemporaryDirectory() as folder:
            root, release = self.release_fixture(folder)
            self.assertEqual(self.namespace['verify_ownership_release'](self.contract, root), release)
            (root / 'handoff/ownership_release_source.json').write_text('{}\n')
            with self.assertRaises(ValueError):
                self.namespace['verify_ownership_release'](self.contract, root)

    def test_unconfirmed_or_partial_release_rejected(self):
        for field in ('unstarted_verified', 'A100_queue_exclusion_verified', 'execution_ids'):
            with tempfile.TemporaryDirectory() as folder:
                root, release = self.release_fixture(folder)
                release[field] = release[field][:-1] if field == 'execution_ids' else False
                (root / 'ownership_release.json').write_text(json.dumps(release))
                with self.assertRaises(ValueError):
                    self.namespace['verify_ownership_release'](self.contract, root)

    def test_new_paths_preserve_runtime_and_gpu_identity(self):
        for host, current in self.hosts.items():
            previous = self.parent['hosts'][host]
            for field in ('python', 'runtime_expected', 'gpu_uuid', 'tmux_binary'):
                self.assertEqual(current[field], previous[field])
            self.assertEqual(Path(current['root']).name, g.SCHEMA + '_' + host)
            self.assertEqual(current['environment']['SOURCE_REVISION'], g.SCIENCE_REVISION)


if __name__ == '__main__':
    unittest.main()
