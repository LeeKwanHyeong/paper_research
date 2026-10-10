import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from paper.scripts import titantpp_history_recovery_operations as repair

ROOT = Path(__file__).resolve().parents[2]
SEALED = ROOT / 'search_artifacts/titantpp_contribution_history_a100_parallel3_20261008_v1/operation/campaign.py'


def original_verifier():
    tree = ast.parse(SEALED.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('require', 'verify_gpu_membership')]
    ns = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SEALED), 'exec'), ns)
    return ns['verify_gpu_membership']


class OwnershipRegression(unittest.TestCase):
    def setUp(self):
        self.root = Path('/approved')
        self.argv = ['python', '/approved/operation/campaign.py', '--contract', '/approved/execution_contract.json', '--host', 'a100', '--mode', 'fit', '--job', 'full', '--deadline', '123']
        self.active = {'full': {'child': SimpleNamespace(pid=10), 'job': {'id': 'full'}, 'command': ['timeout', '--signal=TERM', '--kill-after=15s', '100', *self.argv]}}
        self.proc = {'ppid': 10, 'pgrp': 10, 'state': 'R', 'start_ticks': 22, 'argv': self.argv}

    def test_old_verifier_reproduces_exit_cmdline_false_positive(self):
        with self.assertRaisesRegex(ValueError, 'Foreign GPU process group'):
            original_verifier()({11}, {11: {**self.proc, 'argv': ['']}}, self.active, root=self.root, used_bytes=1, total_bytes=10)

    def test_exiting_worker_removed_only_after_fresh_inventory(self):
        inventory = iter([{11}, set()])
        def exiting(pid):
            raise repair.ProcessTransition('zombie')
        processes, observations = repair.collect_gpu_processes(lambda: next(inventory), exiting)
        self.assertEqual(processes, {})
        self.assertEqual(observations[-1]['fresh_gpu_pids'], [])

    def test_live_foreign_group_still_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Foreign GPU process group'):
            original_verifier()({11}, {11: {**self.proc, 'pgrp': 999}}, self.active, root=self.root, used_bytes=1, total_bytes=10)

    def test_persistent_empty_cmdline_is_not_exempted(self):
        def empty(pid):
            raise repair.ProcessTransition('empty argv')
        with self.assertRaisesRegex(repair.GPUInspectionError, 'remained unavailable'):
            repair.collect_gpu_processes(lambda: {11}, empty, pause=lambda _: None)

    def test_new_gpu_pid_is_read_after_transition(self):
        inventory = iter([{11}, {12}, {12}])
        seen = []
        def read(pid):
            seen.append(pid)
            if pid == 11:
                raise FileNotFoundError()
            return self.proc
        processes, _ = repair.collect_gpu_processes(lambda: next(inventory), read, pause=lambda _: None)
        self.assertEqual(seen, [11, 12])
        self.assertEqual(set(processes), {12})

    def test_stat_start_ticks_and_zombie_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / '11'; folder.mkdir()
            fields = ['R', '10', '10'] + ['0'] * 16 + ['22']
            (folder / 'stat').write_text('11 (a worker) ' + ' '.join(fields))
            (folder / 'cmdline').write_bytes(b'python\0--job\0full\0')
            self.assertEqual(repair.read_process(11, Path(tmp))['start_ticks'], 22)
            fields[0] = 'Z'
            (folder / 'stat').write_text('11 (a worker) ' + ' '.join(fields))
            with self.assertRaises(repair.ProcessTransition):
                repair.read_process(11, Path(tmp))

    def test_full_argv_mismatch_has_forensic_evidence(self):
        campaign = SimpleNamespace(common=SimpleNamespace(gpu_pids=lambda _: {11}), verify_gpu_membership=original_verifier())
        with patch.object(repair.subprocess, 'run', return_value=SimpleNamespace(stdout='UUID, 10, 1\n')), patch.object(repair, 'collect_gpu_processes', return_value=({11: {**self.proc, 'argv': self.argv + ['extra']}}, [{'gpu_pids': [11]}])):
            with self.assertRaisesRegex(repair.GPUInspectionError, 'exact command mismatch') as caught:
                repair.check_active_gpu(campaign, {'hosts': {'a100': {'gpu_uuid': 'UUID'}}}, 'a100', self.root, self.active)
            self.assertEqual(caught.exception.observations, [{'gpu_pids': [11]}])

class RecoveryPreparationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from paper.scripts import prepare_titantpp_history_recovery as prepare
        cls.prepare = prepare
        text = prepare.build_runtime(SEALED.read_text(), (prepare.PARENT / 'source/paper/scripts/run_observed_slot_partition.py').read_text())
        cls.runtime = {'__name__': 'recovery_test_module'}
        exec(compile(text, 'generated_recovery_campaign', 'exec'), cls.runtime)

    def test_completed_jobs_removed_from_queue_but_ids_preserved(self):
        c = self.prepare.read(self.prepare.PARENT / 'execution_contract.json')
        c['recovery'] = {'reused_jobs': [j['id'] for j in c['jobs'][:9]] + [c['jobs'][10]['id']]}
        groups = self.runtime['batch_groups'](c, 'a100')
        self.assertEqual([len(g) for g in groups], [2, 3, 3, 3, 3, 3])
        self.assertEqual(len(c['jobs']), 27)
        self.assertEqual(len({j['id'] for group in groups for j in group}), 17)

    def test_reused_manifest_has_parent_contract_and_separate_folder(self):
        c = {'recovery': {'reused_jobs': ['already_done'], 'parent_contract_sha256': 'parent'}}
        job = {'id': 'already_done'}
        self.assertEqual(self.runtime['result_folder'](c, Path('/new'), job), Path('/new/recovery_parent/run/already_done'))
        self.assertEqual(self.runtime['result_contract'](c, job), 'parent')

    def test_monetary_cap_includes_prior_storage_separately(self):
        c = self.prepare.read(self.prepare.PARENT / 'execution_contract.json')
        c['hosts']['a100']['root'] = self.runtime['OWNED_ROOT']
        c['limits']['total_wall_seconds'] = 3600
        c['budget'] = {'maximum_total_usd': 50, 'independent_controller_required': True,
                      'confirmed_gpu_hourly_usd': 1.0, 'confirmed_running_hourly_usd': 1.0,
                      'reserved_non_compute_usd': 5, 'recovery_reserve_usd': 5,
                      'parent_billed_usd': 43.5, 'prior_network_storage_usd': .5,
                      'billing_started_at_unix': 1000, 'hard_deadline_unix': 4600}
        self.runtime['validate_a100_scope'](c)
        c['budget']['prior_network_storage_usd'] = .501
        with self.assertRaisesRegex(ValueError, 'Monetary cap exceeded'):
            self.runtime['validate_a100_scope'](c)

    def test_input_root_bridge_preserves_actual_receipt_and_all_nonpaths(self):
        from copy import deepcopy
        parent_contract = self.prepare.read(self.prepare.PARENT / 'execution_contract.json')
        old_root = parent_contract['hosts']['a100']['source_root']
        path = self.prepare.PARENT / 'originals/partial/verified/originals/run/raf_spare_parts__42__titantpp_cnn_gru54/input_receipt.json'
        parent = self.prepare.read(path); snapshot = deepcopy(parent)
        current = deepcopy(parent)
        for key, value in parent['resolved_inputs'].items():
            current['resolved_inputs'][key] = value.replace(old_root, '/recovery/source', 1)
        self.assertNotEqual(parent, current)  # Former full-equality guard rejects valid migration.
        mapping = self.runtime['verify_input_root_bridge'](parent, current, old_root, '/recovery/source')
        self.assertEqual(len(mapping), 2)
        self.assertEqual(parent, snapshot)
        for field in ('raw_scale', 'input_identity', 'populations'):
            changed = deepcopy(current); changed[field] = 'tampered'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'statistics/identity'):
                self.runtime['verify_input_root_bridge'](parent, changed, old_root, '/recovery/source')
        changed = deepcopy(current); changed['resolved_inputs']['local_data_path'] += '.other'
        with self.assertRaisesRegex(ValueError, 'Unauthorized input path'):
            self.runtime['verify_input_root_bridge'](parent, changed, old_root, '/recovery/source')

    def test_interface_bridge_changes_only_authorized_contract_digest(self):
        from copy import deepcopy
        parent = {'execution_contract_sha256': 'parent', 'statistics': {'mean': 1.2},
                  'source_files_sha256': 'source', 'backbone_design': {'training': {'epochs': 300}}}
        current = deepcopy(parent); current['execution_contract_sha256'] = 'recovery'
        bridged = self.runtime['bridge_resume_interface'](current, parent, 'parent', 'recovery')
        self.assertEqual(bridged, parent)
        self.assertEqual(current['execution_contract_sha256'], 'recovery')
        for field in ('statistics', 'source_files_sha256', 'backbone_design'):
            wrong = deepcopy(current); wrong[field] = 'changed'
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'scientific interface'):
                self.runtime['bridge_resume_interface'](wrong, parent, 'parent', 'recovery')
        with self.assertRaisesRegex(ValueError, 'not bound'):
            self.runtime['bridge_resume_interface'](current, parent, 'other-parent', 'recovery')

    def test_fresh_interface_has_no_bridge_or_checkpoint_access(self):
        interface = {'execution_contract_sha256': 'new'}
        self.assertIs(self.runtime['migrate_resume_interface'](None, {}, {}, None, None, interface, None), interface)

    def test_resume_interface_bridge_receipt_preserves_checkpoint_bytes(self):
        from copy import deepcopy
        import json
        c = {'recovery': {'parent_contract_sha256': 'parent'}}
        digest = self.runtime['sha_json'](c)
        parent = {'execution_contract_sha256': 'parent', 'statistics': {'mean': 1.2}}
        current = deepcopy(parent); current['execution_contract_sha256'] = digest
        payload = {'interface_meta': parent, 'resume_identity': {'interface_meta': deepcopy(parent)}}
        self.runtime['common'] = SimpleNamespace(write_json=lambda path, value, exclusive: path.write_text(json.dumps(value)))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); checkpoint = folder / 'last_epoch_state.pt'; checkpoint.write_bytes(b'immutable parent binary')
            before = checkpoint.read_bytes()
            entry = {'last_checkpoint_sha256': self.runtime['sha_file'](checkpoint)}
            training = SimpleNamespace(torch_load_checkpoint=lambda *a, **k: payload)
            value = self.runtime['migrate_resume_interface'](training, c, {'id': 'resumed'}, folder, folder, current, entry)
            self.assertEqual(value, parent)
            self.assertEqual(checkpoint.read_bytes(), before)
            receipt = json.loads((folder / 'interface_contract_bridge.json').read_text())
            self.assertEqual(receipt['recovery_contract_sha256'], digest)
            self.assertEqual(receipt['parent_contract_sha256'], 'parent')
            self.assertTrue(receipt['all_other_interface_fields_equal'])
            entry['last_checkpoint_sha256'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'binary bridge binding'):
                self.runtime['migrate_resume_interface'](training, c, {}, folder, folder, current, entry)

    def test_resume_audit_keeps_prefix_and_optimizer_step_check(self):
        import json
        saved = []
        writes = {}
        training = SimpleNamespace(make_loader=lambda *a, **k: [], atomic_torch_save=lambda payload, path: saved.append(payload['epoch']))
        original_loader, original_save = training.make_loader, training.atomic_torch_save
        def exposure_loader(loader, split, records, budget, on_start=None):
            if on_start: on_start()
            return loader
        self.runtime['shared'] = SimpleNamespace(ExposureLoader=exposure_loader)
        self.runtime['common'] = SimpleNamespace(write_json=lambda path, value: writes.update({Path(path).name: value}))
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            prior = {'train': [{'batches': 1, 'count': 2}] * 23, 'validation': [{'batches': 1, 'count': 2}] * 23}
            (run / 'exposure.json').write_text(json.dumps(prior))
            (run / 'epoch_timing.json').write_text(json.dumps({'epochs': [{'epoch': i, 'elapsed_seconds': 1} for i in range(1, 24)]}))
            with self.runtime['recovery_audited_training'](training, {}, lambda: None, lambda e, r: None, run, {'epoch': 23}) as records:
                training.make_loader(target_split='train')
                records['train'].append({'batches': 1, 'count': 2}); records['validation'].append({'batches': 1, 'count': 2})
                payload = {'epoch': 24, 'optimizer_state_dict': {'state': {0: {'step': 24}}}, 'history': [{'train_event_count': 2, 'train_batch_count': 1}]}
                training.atomic_torch_save(payload, run / 'last_epoch_state.pt')
                self.assertEqual(records['train'][:23], prior['train'])
            self.assertEqual(len(writes['epoch_timing.json']['epochs']), 24)
            self.assertEqual(saved, [24])
            self.assertIs(training.make_loader, original_loader)
            self.assertIs(training.atomic_torch_save, original_save)


if __name__ == '__main__':
    unittest.main()
