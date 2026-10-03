"""CPU-only recovery authority/audit checks and real synthetic trainer integration."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import pytest

from paper.scripts import run_state_transport_intermittent_recovery as runner

PARENT_RELATIVE = "search_artifacts/state_transport_launch_20260919_v1/recovery_preparation/execution_contract.json"


def parent_contract():
    # A standalone extracted bundle carries its immutable parent inside the
    # sibling execution contract; a source checkout can read original evidence.
    sibling = runner.ROOT.parent / 'frozen_execution/execution_contract.json'
    if sibling.is_file():
        value = runner.read(sibling)
        if 'parent_contract' in value:
            return value['parent_contract']
    for path in (runner.ROOT, *runner.ROOT.parents):
        if (path / PARENT_RELATIVE).is_file():
            return runner.read(path / PARENT_RELATIVE)
    raise RuntimeError('Frozen parent contract fixture unavailable')


def fixture():
    parent = parent_contract()
    files = deepcopy(parent['source']['files'])
    files[runner.ENTRYPOINT] = 'a' * 64
    files[runner.TEST_FILE] = 'b' * 64
    old = parent['hosts']['5090']
    spec = {**old, 'root': runner.NEW_ROOT, 'source_root': runner.NEW_ROOT + '/source',
            'output_dir': runner.NEW_ROOT + '/run', 'tmux': Path(runner.NEW_ROOT).name + '_5090'}
    contract = {'schema': 'state_transport_intermittent_completion_v1', 'parent_contract': parent,
                'parent_contract_sha256': runner.PARENT_SHA, 'host': '5090', 'host_spec': spec,
                'old_root': old['root'], 'scope': deepcopy(runner.SCOPE), 'limits': deepcopy(runner.LIMITS),
                'source': {**parent['source'], 'files': files, 'files_sha256': runner.common.sha_json(files)},
                'environment_overrides': {**parent['limited_recovery']['environment_overrides']['5090'],
                                          'XDG_CACHE_HOME': runner.NEW_ROOT + '/cache'},
                'preserved_files': {name: 'd' * 64 for name in runner.required_preserved_paths(old['root'],parent)}}
    contract["metadata_recovery"] = deepcopy(runner.RECOVERY_CLOCK)
    contract["required_source_directories"] = ["sample_data"]
    contract["preserved_files"].update({str(Path(old['root']) / 'source' / name): digest
                                        for name, digest in parent['source']['files'].items()})
    approval = {'explicit_metadata_recovery': True, 'approved': True, 'hosts': ['5090'], 'contract_sha256': runner.common.sha_json(contract), 'user_instruction': '진행하자'}
    return contract, approval, runner.make_permit(contract, approval, now=runner.RECOVERY_CLOCK["started_at_unix"])



class AuthorityTests(unittest.TestCase):
    def test_exact_authority_accepts_before_deadline(self):
        c, a, p = fixture()
        self.assertEqual(runner.verify_authority(c, a, p, now=runner.RECOVERY_CLOCK["started_at_unix"] + 1.)['root'], runner.NEW_ROOT)

    def test_rejects_missing_approval_or_wrong_host(self):
        c, a, p = fixture()
        for altered in ({**a, 'approved': False}, {**a, 'hosts': ['5080']}, {**a, 'user_instruction': ''}):
            with self.subTest(approval=altered), self.assertRaises(ValueError):
                runner.verify_authority(c, altered, p, now=runner.RECOVERY_CLOCK["started_at_unix"] + 1.)

    def test_rejects_future_expired_and_extended_permit(self):
        c, a, p = fixture()
        for now, value in ((runner.RECOVERY_CLOCK["started_at_unix"] - 1., p), (p['deadline_unix'], p), (runner.RECOVERY_CLOCK["started_at_unix"] + 1., {**p, 'deadline_unix': p['deadline_unix'] + 1.})):
            with self.subTest(now=now), self.assertRaises(ValueError):
                runner.verify_authority(c, a, value, now=now)

    def test_scope_cannot_train_completed_arm(self):
        c, _, _ = fixture(); c['scope']['train_arm'] = runner.ARMS[0]
        with self.assertRaisesRegex(ValueError, 'scope'):
            runner.validate_contract(c)

    def test_parent_or_training_code_mutation_rejected(self):
        c, _, _ = fixture(); c['parent_contract']['seed'] = 43
        with self.assertRaisesRegex(ValueError, 'parent'):
            runner.validate_contract(c)
        c, _, _ = fixture()
        c['source']['files']['paper/scripts/count_aware_tpp_backbone/training.py'] = '0' * 64
        c['source']['files_sha256'] = runner.common.sha_json(c['source']['files'])
        with self.assertRaisesRegex(ValueError, 'computation'):
            runner.validate_contract(c)

    def test_only_metadata_validator_change_allowed(self):
        c, _, _ = fixture()
        c['source']['files']['models/TPPs/CountAwareTitanStateTransport.py'] = '0' * 64
        c['source']['files_sha256'] = runner.common.sha_json(c['source']['files'])
        runner.validate_contract(c)

    def test_missing_preserved_checkpoint_rejected(self):
        c, _, _ = fixture()
        del c['preserved_files'][str(runner.arm_dir(c['old_root'], runner.ARMS[1]) / 'last_epoch_state.pt')]
        with self.assertRaisesRegex(ValueError, 'Missing preserved'):
            runner.validate_contract(c)

    def test_preserved_byte_mutation_rejected_without_checkpoint_load(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / 'metadata.json'; p.write_text('{"value":1}')
            c = {'old_root': directory, 'preserved_files': {str(p): runner.common.sha_file(p)}}
            p.write_text('{"value":2}')
            with self.assertRaisesRegex(ValueError, 'Preserved artifact changed'):
                runner.verify_preserved(c)

    def test_fresh_output_rejects_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'run'
            self.assertEqual(runner.verify_fresh_output({'output_dir': str(output)}), output)
            output.mkdir()
            with self.assertRaisesRegex(ValueError, 'no retry/resume'):
                runner.verify_fresh_output({'output_dir': str(output)})

    def test_only_new_candidate_reaches_training_callback(self):
        calls = []
        def record(action):
            def operation(arm):
                calls.append((action, arm)); return arm
            return operation
        result = runner.ordered_recovery(record('reuse'), record('replay'), record('train'))
        self.assertEqual(calls, [('reuse', runner.ARMS[0]), ('replay', runner.ARMS[1]), ('train', runner.ARMS[2])])
        self.assertEqual(set(result), set(runner.ARMS))

    def test_replay_failure_prevents_new_training(self):
        train = mock.Mock()
        with self.assertRaisesRegex(ValueError, 'bad replay'):
            runner.ordered_recovery(lambda arm: arm, mock.Mock(side_effect=ValueError('bad replay')), train)
        train.assert_not_called()


def test_real_cpu_preserved_replay_and_single_candidate_training(monkeypatch, tmp_path):
    import numpy as np
    import polars as pl
    import torch
    from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
    from paper.scripts import quantity_comparison_data as qdata
    from paper.scripts import run_count_aware_tpp_backbone_control as cli
    from paper.scripts.count_aware_tpp_backbone import training
    from paper.scripts.run_quantity_comparison import bind_frozen_statistics
    from paper.scripts.run_hard_lmm_frozen_lognormal_duration import target_dt_sha256
    c, _, permit = fixture()
    parent = c['parent_contract']
    data = next(d for d in parent['datasets'] if d['dataset_id'] == runner.DATASET)
    rows = [{'oper_part_no': str(part), 'seq': i, 'delta_t': [0, 1, 2, 3, 4, 7][i % 6],
             'demand_qty': float(1 + i + part * 20), 'chronological_split': 'train' if i < 8 else 'validation'}
            for part in range(3) for i in range(12)]
    raw = pl.DataFrame(rows)
    path, manifest = tmp_path / 'synthetic.parquet', tmp_path / 'synthetic_split.json'
    raw.write_parquet(path); manifest.write_text('{"synthetic":true}')
    identity = data['inherited_data_identity']
    identity['data'] = {'path': str(path), 'sha256': cli.sha256_file(path)}
    identity['split_manifest'] = {'path': str(manifest), 'sha256': cli.sha256_file(manifest)}
    frame = cli.prepare_count_frame(raw)
    loader = {k: data['loader'][k] for k in ('lookback_weeks', 'max_seq_len')}
    identity['populations'] = {split: qdata._population_view(qdata.exact_target_population(frame, target_split=split, **loader)[1])
                               for split in ('train', 'validation')}
    quantities = raw.filter(pl.col('chronological_split') == 'train')['demand_qty'].to_numpy()
    data['mu_all_train_rows'] = float(np.log1p(quantities).mean())
    stats = qdata.prepare_dataset_statistics(data)
    data['statistics'] = {'train_log_mean': stats['all_train_rows']['log1p_mean'],
                         'train_log_std': stats['all_train_rows']['log1p_std'],
                         'raw_scale': stats['canonical_train_targets']['raw_scale'],
                         'all_train_quantity_sha256': stats['all_train_rows']['quantity_sha256']}
    data['quantity_boundaries_all_train_rows'] = np.quantile(quantities, [.5, .9, .95, .99]).tolist()
    data['expected_global_steps'] = 120
    c['scope']['new_optimizer_steps'] = 120
    time_stats = cli.derive_train_time_contract(frame, **loader)
    frozen = data['time_statistics']
    for name, key in (('time_scale', 'train_time_scale'), ('target_log_scaled_mean', 'train_log_scaled_mean'),
                      ('target_log_scaled_std', 'train_log_scaled_std')):
        frozen[key] = time_stats[name]
    for split in ('train', 'validation'):
        ds = RMTPPWeekLookbackDataset(frame, **loader, val_ratio=.2, mode='all', split_col='chronological_split', target_splits={split})
        values = torch.tensor([max(1., float(ds.dt_lists[p][i+1])) for p, i in ds.index], dtype=torch.float64)
        frozen[f'expected_{split}_targets'] = values.numel()
        frozen[f'expected_{split}_target_dt_sha256'] = target_dt_sha256(values)
    data['model'].update(time_scale=time_stats['time_scale'], time_initial_location=time_stats['target_log_scaled_mean'],
                         time_initial_scale=time_stats['target_log_scaled_std'])
    old, new = tmp_path / 'old', tmp_path / 'new'
    c['old_root'] = str(old); c['host_spec']['root'] = str(new); c['host_spec']['output_dir'] = str(new / 'run')
    (new / 'run').mkdir(parents=True)
    old_dataset = old / 'run' / runner.DATASET; old_dataset.mkdir(parents=True)
    runtime = {'synthetic_cpu': True}
    # Only deployment checks, pinned production contract and device are substituted.
    # Loader, train statistics, model, optimizer, selector, checkpoint/replay are real.
    monkeypatch.setattr(runner.common, 'runtime_check', lambda *a: runtime)
    monkeypatch.setattr(runner, 'validate_contract', lambda *a, **kw: None)
    monkeypatch.setattr(runner, 'verify_preserved', lambda *a: None)
    monkeypatch.setattr(runner.parent_runner, 'verify_recovery_environment', lambda *a, **kw: None, raising=False)
    real_args, real_replay = runner.parent_runner.training_args, runner.parent_runner.replay_checkpoint
    def cpu_args(*args):
        parsed = real_args(*args); parsed.device = 'cpu'; return parsed
    monkeypatch.setattr(runner.parent_runner, 'training_args', cpu_args)
    monkeypatch.setattr(runner.parent_runner, 'replay_checkpoint', lambda *a, **kw: real_replay(*a, **{**kw, 'device': 'cpu'}))
    frame, metadata = qdata.prepare_quantity_comparison_data(data)
    metadata = bind_frozen_statistics(data, metadata)
    interface = runner.parent_runner.time_interface(data, frame, parent)
    initial = runner.parent_runner.initialization(data)
    runner.common.write_json(old_dataset / 'input_receipt.json', metadata)
    runner.common.write_json(old_dataset / 'initialization.json', initial)
    q = {'boundaries': data['quantity_boundaries_all_train_rows'], 'strata': [{'label': f'frozen_quantity_bin_{i}'} for i in range(5)]}
    threads = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        for arm in runner.ARMS[:2]:
            args = cpu_args(parent, data, old_dataset)
            run = runner.arm_dir(old, arm)
            with runner.shared.audited_training(training, data, lambda: None, lambda *a: None) as exposure:
                summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=q, interface_meta=interface,
                                                    backbone=arm, quantity_variant=runner.VARIANT, seed=42)
            runner.common.write_json(run / 'exposure.json', exposure)
            if arm == runner.ARMS[0]:
                history = runner.read(run / 'history.json')['history']
                identity = training._resume_identity(args=args, backbone=arm, quantity_variant=runner.VARIANT, seed=42,
                    monitor='validation_raw_quantity_rmse', quantity_contract=q, interface_meta=interface)
                endpoints = {}
                for label, epoch in [('selected', summary['best_epoch']), ('last', 120)]:
                    checkpoint = run / ('best_val_qty_rmse_model.pt' if label == 'selected' else 'last_epoch_state.pt')
                    endpoints[label] = real_replay(checkpoint, data, frame, lambda: None, device='cpu', expected_arm=arm,
                        expected_identity=identity, expected_initial=initial[arm], expected_epoch=epoch)
                endpoints.update(best_epoch=summary['best_epoch'], global_steps=120, initial_state_sha256=initial[arm],
                                 last30=runner.shared.last30_summary(history))
                runner.common.write_json(run / 'endpoint_replays.json', endpoints)
        before = {str(p): runner.common.sha_file(p) for p in old.rglob('*') if p.is_file()}
        captured = []; real_train = training.train_one
        def capture_train(**kwargs):
            captured.append(kwargs); return real_train(**kwargs)
        monkeypatch.setattr(training, 'train_one', capture_train)
        result = runner.production(c, permit, {'runtime': runtime}, lambda: None)
        assert result['status'] == 'complete'
        assert result['new_optimizer_steps'] == 120 and result['preserved_optimizer_steps'] == 240
        assert result['total_optimizer_steps'] == 360
        assert len(captured) == 1 and captured[0]['backbone'] == runner.ARMS[2]
        assert result['datasets'][runner.DATASET]['exposure_equal'] is True
        assert all(value['global_steps'] == 120 for value in result['datasets'][runner.DATASET]['arms'].values())
        actual = captured[0]['interface_meta']
        assert actual['source_files_sha256'] == c['source']['files_sha256']
        assert actual['execution_contract_sha256'] == runner.common.sha_json(c)
        assert actual['parent_scientific_contract_sha256'] == runner.PARENT_SHA
        assert actual['time_head'] == interface['time_head']
        assert {str(p): runner.common.sha_file(p) for p in old.rglob('*') if p.is_file()} == before
        assert not runner.arm_dir(new, runner.ARMS[0]).exists()
        assert not runner.arm_dir(new, runner.ARMS[1]).exists()
        assert result['new_endpoint_replays'] == 4 and result['reused_endpoint_replays'] == 2
    finally:
        torch.set_num_threads(threads)


def test_recovery_cannot_reset_the_original_clock():
    c,a,p=fixture()
    moved={**p,'started_at_unix':p['started_at_unix']+60,'deadline_unix':p['deadline_unix']+60}
    with pytest.raises(ValueError,match='reset'):
        runner.verify_authority(c,a,moved,now=moved['started_at_unix']+1)


def test_parent_source_binding_cannot_be_replaced_with_modified_bytes():
    c,_,_=fixture()
    key=str(Path(c['old_root'])/'source/models/TPPs/CountAwareFactory.py')
    c['preserved_files'][key]='0'*64
    with pytest.raises(ValueError,match='parent source'):
        runner.validate_contract(c)


def test_same_epoch_checkpoint_reuse_does_not_grant_recovery_approval():
    c,a,p=fixture(); a.pop('explicit_metadata_recovery')
    with pytest.raises(ValueError,match='metadata recovery approval'):
        runner.verify_authority(c,a,p,now=p['started_at_unix']+1)


def test_parent_candidate_directory_prevents_duplication(tmp_path):
    parent_file=tmp_path/'frozen_execution/execution_contract.json'
    parent_file.parent.mkdir(); parent_file.write_text('{}')
    runner.arm_dir(tmp_path,runner.ARMS[2]).mkdir(parents=True)
    c={'old_root':str(tmp_path),'parent_contract':{},
       'preserved_files':{str(parent_file):runner.common.sha_file(parent_file)}}
    with pytest.raises(ValueError,match='duplication'):
        runner.verify_preserved(c)


def test_busy_gpu_refuses_before_spawn_and_keeps_existing_job(monkeypatch):
    c,a,p=fixture()
    monkeypatch.setattr(runner,'read',lambda path:{'c':c,'a':a,'p':p}[path])
    monkeypatch.setattr(runner,'verify_authority',lambda *a,**kw:c['host_spec'])
    monkeypatch.setattr(runner,'validate_contract',lambda *a,**kw:None)
    monkeypatch.setattr(runner,'verify_preserved',lambda *a:None)
    monkeypatch.setattr(runner.shared,'verify_host_paths',lambda *a:None)
    monkeypatch.setattr(runner.common,'gpu_pids',lambda *a:{987654})
    spawn=mock.Mock(); stop=mock.Mock()
    monkeypatch.setattr(runner.subprocess,'Popen',spawn)
    monkeypatch.setattr(runner.shared,'kill_owned_process_group',stop)
    with pytest.raises(ValueError,match='GPU busy'):
        runner.supervisor('c','a','p')
    spawn.assert_not_called(); stop.assert_not_called()


def test_budget_preserves_both_deadlines_and_foreign_gpu_job(monkeypatch,tmp_path):
    c,a,p=fixture(); c['host_spec']['output_dir']=str(tmp_path)
    monkeypatch.setattr(runner.os,'getppid',lambda:321)
    monkeypatch.setattr(runner.time,'time',lambda:p['started_at_unix']+100)
    monkeypatch.setattr(runner.time,'monotonic',lambda:500.)
    monkeypatch.setattr(runner.shared,'check_storage',lambda *a:None)
    monkeypatch.setattr(runner.common,'gpu_pids',lambda *a:{runner.os.getpid()})
    budget=runner.Budget(c,p,321); budget()
    monkeypatch.setattr(runner.time,'monotonic',lambda:511.)
    monkeypatch.setattr(runner.common,'gpu_pids',lambda *a:{runner.os.getpid(),999})
    with pytest.raises(ValueError,match='Other GPU'):
        budget()
    monkeypatch.setattr(runner.time,'time',lambda:p['deadline_unix'])
    with pytest.raises(ValueError,match='deadline'):
        budget()
    monkeypatch.setattr(runner.time,'time',lambda:p['started_at_unix']+100)
    monkeypatch.setattr(runner.time,'monotonic',lambda:budget.monotonic_deadline)
    with pytest.raises(ValueError,match='deadline'):
        budget()


def test_child_cannot_continue_after_its_supervisor_exits(monkeypatch):
    c,a,p=fixture()
    monkeypatch.setattr(runner.time,'time',lambda:p['started_at_unix']+100)
    monkeypatch.setattr(runner.time,'monotonic',lambda:500.)
    budget=runner.Budget(c,p,321)
    monkeypatch.setattr(runner.os,'getppid',lambda:1)
    with pytest.raises(ValueError,match='supervisor exited'):
        budget()
