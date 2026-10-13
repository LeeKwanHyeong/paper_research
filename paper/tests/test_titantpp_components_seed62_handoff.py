"""Focused stdlib tests: queue exclusion, race preservation and adapter scope."""
import ast
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from paper.scripts import titantpp_components_seed62_handoff as h
from paper.scripts import prepare_titantpp_components_seed62_handoff as p

PARENT = h.PROJECT / 'search_artifacts' / h.PARENT_SCHEMA


class Seed62HandoffTests(unittest.TestCase):
    def setUp(self):
        self.parent = h.read(PARENT / 'execution_contract.json')
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'parent'; self.root.mkdir(); (self.root / 'claims').mkdir()
        self.patch = patch.object(h, 'PARENT_ROOT', self.root); self.patch.start(); self.addCleanup(self.patch.stop)
        self.c = p.make_contract(self.parent, deepcopy(self.parent['operation']['files']), 'a' * 64)
        h.write_exclusive(self.root / 'execution_contract.json', self.parent)
        self.observation = self.make_observation()

    def make_observation(self):
        spec = self.parent['hosts']['5090']
        job = next(j for j in self.parent['jobs'] if j['host'] == '5090' and j['seed'] == 42 and j['arm'] == h.ARMS[3])
        phase = [spec['python'], spec['operation_root'] + '/campaign.py', '--contract',
                 spec['root'] + '/execution_contract.json', '--host', '5090', '--mode', 'dispatch']
        guard = ['timeout', '--signal=TERM', '--kill-after=15s', '3600', spec['python'],
                 spec['operation_root'] + '/campaign.py', '--contract', spec['root'] + '/execution_contract.json',
                 '--host', '5090', '--mode', 'fit', '--job', job['id'], '--deadline', '3000.0']
        worker = {'job': job, 'worker_group_pid': 11, 'worker_guard_argv': guard, 'condition_deadline_unix': 3000.}
        state = {'status': 'running', 'active_jobs': {job['id']: worker}, 'supervisor_pid': 10,
                 'supervisor_argv': phase, 'updated_unix': 1199., 'deadline_unix': 4000.}
        return {'observed_unix': 1200., 'host': '5090', 'root': str(self.root), 'contract_sha256': h.PARENT_SHA,
            'gpu_uuid': spec['gpu_uuid'], 'gpu_pids': [12], 'status': state,
            'processes': {'10': {'pid': 10, 'argv': phase}, '11': {'pid': 11, 'ppid': 10, 'pgrp': 11, 'argv': guard},
                          '12': {'pid': 12, 'ppid': 11, 'pgrp': 11, 'argv': guard[4:]}},
            'leases': {job['id']: {'job': job['id'], 'contract_sha256': h.PARENT_SHA,
                       'worker_group_pid': 11, 'supervisor_pid': 10, 'expires_unix': 1280.}}}

    def reserve(self, **extra):
        return h.reserve_boundary(self.root, self.c, '사용자 승인: 미시작62 네 arm5080 이관',
            verifier=lambda root, parent: None, probe=lambda root, parent: self.observation,
            now=lambda: 1201., **extra)

    def test_contract_only_four_seed62_same_gpu_and_original_lease(self):
        h.validate_contract(self.c, self.parent)
        self.assertEqual(self.c['jobs'], h.transferred_jobs('5080'))
        self.assertEqual(self.c['operational_replacement']['host_leases']['5080']['deadline_unix'], h.DEADLINE)
        self.assertEqual(self.c['source'], self.parent['source'])

    def test_science_or_mapping_or_lease_drift_rejected(self):
        for mutate in (lambda c: c['jobs'][0].update(seed=52),
                       lambda c: c['jobs'][0].update(host='5090'),
                       lambda c: c['training'].update(patience=41),
                       lambda c: c['operational_replacement']['host_leases']['5080'].update(deadline_unix=h.DEADLINE + 1)):
            c = deepcopy(self.c); mutate(c)
            with self.assertRaises(ValueError): h.validate_contract(c, self.parent)

    def test_reservation_partial_exclusion_not_owner_release(self):
        proof = self.reserve(); h.validate_queue_proof(proof, self.c, self.parent)
        self.assertTrue(proof['queue_exclusion_verified']); self.assertFalse(proof['parent_full_owner_released'])
        self.assertEqual(proof['transfer_jobs'], h.transferred_jobs())
        self.assertEqual(len(list((self.root / 'claims').iterdir())), 1)
        self.assertEqual(h.sha(Path(proof['barrier_path'])), proof['barrier_raw_sha256'])

    def test_repeated_reservation_cannot_retry(self):
        self.reserve()
        with self.assertRaises(FileExistsError): self.reserve()

    def test_existing_scientific_claim_preserved(self):
        target = self.root / 'claims' / (h.transferred_jobs()[0]['id'] + '.json')
        target.write_bytes(b'original scientific claim')
        with self.assertRaisesRegex(ValueError, 'already attempted'): self.reserve()
        self.assertEqual(target.read_bytes(), b'original scientific claim')

    def test_atomic_race_loser_keeps_existing_claim(self):
        target = self.root / 'claims' / 'race.json'
        original_link = h.os.link
        def raced(source, destination):
            Path(destination).write_bytes(b'real dispatcher won')
            return original_link(source, destination)
        with patch.object(h.os, 'link', side_effect=raced):
            with self.assertRaises(FileExistsError): h.atomic_exclusive(target, {'administrative': True})
        self.assertEqual(target.read_bytes(), b'real dispatcher won')
        self.assertFalse(list(target.parent.glob('.seed62-exclusion-*')))

    def test_attempted_other_arm_blocks_exclusion(self):
        target = self.root / 'run' / h.transferred_jobs()[2]['id']; target.mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'already attempted'): self.reserve()
        self.assertEqual(list((self.root / 'claims').iterdir()), [])

    def test_foreign_gpu_pid_or_stale_or_active_seed62_rejected(self):
        for mutate in (lambda o: o['processes']['12'].update(ppid=99),
                       lambda o: o.update(observed_unix=1000.),
                       lambda o: next(iter(o['status']['active_jobs'].values()))['job'].update(seed=62)):
            o = deepcopy(self.observation); mutate(o)
            with self.assertRaises(ValueError): h.verify_running_parent(o, self.parent, 1201.)

    def test_proof_raw_scope_and_owner_semantics_revalidated(self):
        proof = self.reserve()
        for mutate in (lambda v: v.update(barrier_raw_sha256='b' * 64),
                       lambda v: v.update(parent_full_owner_released=True),
                       lambda v: v.update(transfer_jobs=h.transferred_jobs('5080')),
                       lambda v: v['reservation'].update(destination_contract_sha256='c' * 64),
                       lambda v: v.update(after_never_started=[])):
            v = deepcopy(proof); mutate(v)
            with self.assertRaises(ValueError): h.validate_queue_proof(v, self.c, self.parent)

    def test_post_publication_attempt_fails_and_keeps_barrier(self):
        original = h.atomic_exclusive
        def race(path, value):
            raw = original(path, value)
            target = self.root / 'worker_exits' / (h.transferred_jobs()[1]['id'] + '.json')
            target.parent.mkdir(); target.write_text('{}')
            return raw
        with patch.object(h, 'atomic_exclusive', side_effect=race):
            with self.assertRaisesRegex(ValueError, 'already attempted'): self.reserve()
        self.assertTrue((self.root / 'claims' / (h.transferred_jobs()[0]['id'] + '.json')).exists())
        self.assertTrue((self.root / 'handoff_seed62/failure.json').exists())

    def test_adapter_preserves_every_function_except_validation_authorization(self):
        original = (PARENT / 'operation/campaign.py').read_text()
        adapter = p.adapter_source(original); before = p.functions(original); after = p.functions(adapter)
        for name, fingerprint in before.items():
            if name not in ('validate', 'authorization'): self.assertEqual(after[name], fingerprint, name)
        self.assertNotEqual(after['validate'], before['validate'])
        self.assertNotEqual(after['authorization'], before['authorization'])
        self.assertEqual(set(after) - set(before), {'_seed62_validate_queue_exclusion'})

    def test_original_claim_before_Popen_and_no_claim_deletion(self):
        node = next(n for n in ast.parse((PARENT / 'operation/campaign.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'dispatch')
        text = ast.unparse(node)
        self.assertLess(text.index("root / 'claims'"), text.index('subprocess.Popen('))
        self.assertIn('exclusive=True', text); self.assertNotIn('.unlink(', text)

    def test_remote_program_compiles_without_project_imports_at_start(self):
        c = deepcopy(self.c)
        c['seed62_handoff']['administrative_helper_sha256'] = h.sha(h.PROJECT / 'paper/scripts/titantpp_components_seed62_handoff.py')
        source = h.remote_reservation_source(c, '명시 승인')
        compile(source, '<seed62 remote reservation>', 'exec')
        tree = ast.parse(source)
        imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        self.assertFalse(any(isinstance(n, ast.ImportFrom) and (n.module or '').startswith('paper') for n in imports))

    def test_main_bootstrap_finds_queue_globals_before_dispatch(self):
        original = (PARENT / 'operation/campaign.py').read_text()
        adapter = p.adapter_source(original); tree = ast.parse(adapter)
        self.assertIsInstance(tree.body[-1], ast.If)
        self.assertLess(adapter.index('_HANDOFF_QUEUE_CHECK_SOURCE ='), adapter.index('if __name__ =='))
        namespace = {'__name__': 'local_adapter_unit_test', '__file__': str(PARENT / 'operation/campaign.py')}
        exec(compile(adapter, '<adapter unit>', 'exec'), namespace)
        with patch.object(h, 'PARENT_ROOT', h.ROOT_PARENT / (h.PARENT_SCHEMA + '_5090')):
            c = p.make_contract(self.parent, deepcopy(self.parent['operation']['files']), 'a' * 64)
        seen = []
        namespace['read'] = lambda path: self.parent if str(path).endswith('parent_execution_contract.json') else c
        namespace['bootstrap'] = lambda contract, host: (namespace['validate'](contract, verify_files=False), seen.append(('bootstrap', host)))
        namespace['dispatch'] = lambda contract, host, root: seen.append(('dispatch', host))
        namespace['owned_signal_guard'] = nullcontext
        namespace['__name__'] = '__main__'
        spec = c['hosts']['5080']
        with patch.object(namespace['sys'], 'argv', ['campaign.py', '--contract', spec['root'] + '/execution_contract.json', '--host', '5080', '--mode', 'dispatch']), \
                patch.object(Path, 'cwd', return_value=Path(spec['source_root'])), \
                patch.object(Path, 'resolve', lambda self, **kwargs: self), \
                patch.dict(namespace['os'].environ, spec['environment'], clear=False):
            exec(compile(ast.Module(body=[tree.body[-1]], type_ignores=[]), '<actual main guard unit>', 'exec'), namespace)
        self.assertEqual(seen, [('bootstrap', '5080'), ('dispatch', '5080')])


if __name__ == '__main__': unittest.main()
