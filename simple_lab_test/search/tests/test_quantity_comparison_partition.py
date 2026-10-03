import copy
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from paper.scripts import run_quantity_comparison as parent_cli
from paper.scripts import run_quantity_comparison_partition as partition


def parent_fixture(tmp_path):
    value = {"schema": parent_cli.SCHEMA, "status": "frozen_pending_explicit_approval",
             "source": {"files": {}, "files_sha256": parent_cli.sha_json({})},
             "datasets": [{"dataset_id": dataset} for dataset in partition.DATASETS],
             "runtime_expected": {"device": "cuda:0"},
             "limits": {"max_wall_seconds": 172800}, "execution": {"output_dir": "/original"}}
    return value


def fixture(tmp_path):
    parent = parent_fixture(tmp_path)
    arms = []
    for dataset in partition.DATASETS:
        for case in partition.CASES:
            host = "taxi_5080" if dataset == "yellow_trip_hourly" else "continuation_5090"
            mode = "fresh"
            item = {"dataset_id": dataset, "case": case, "host": host, "mode": mode}
            if dataset == "intermittent_frozen_5000" and case != "raw_softplus":
                item.update(mode="reuse", reuse_summary={"path": "/prior/summary.json", "sha256": "a" * 64},
                            reuse_contract={"path": "/prior/contract.json", "sha256": "b" * 64})
            elif dataset == "intermittent_frozen_5000":
                item.update(mode="resume", resume_from="/copied/raw_softplus",
                            resume_identity={"execution_contract_sha256": partition.sha_json(parent)})
            arms.append(item)
    runtime = {"device": "cuda:0"}
    value = {"schema": partition.SCHEMA, "status": "frozen_pending_explicit_approval",
             "parent": {"contract_sha256": partition.sha_json(parent), "source_files_sha256": parent["source"]["files_sha256"],
                        "deadline_unix": 200.0, "started_at_unix": 200.0 - 172800,
                        "execution_receipt": {"contract_sha256": partition.sha_json(parent),
                            "started_at_unix": 200.0 - 172800, "deadline_unix": 200.0}}, "wrapper": {"sha256": partition.sha_file(partition.__file__)},
             "deadline_unix": 200.0,
             "limits": {"qualification_seconds": 900, "output_bytes_per_host": 900 * 1024 ** 2,
                        "per_file_bytes": 64 * 1024 ** 2, "minimum_free_bytes": 5 * 1024 ** 3,
                        "max_concurrent_gpu_jobs_per_host": 1},
             "policy": {"automatic_retry": False, "automatic_resume": False, "held_out": False,
                        "budget_extension": False, "cpu_fallback": False},
             "hosts": {role: {"role": role, "qualification": qualification, "runtime_expected": runtime,
                               "source_root": "/src", "output_dir": "/out",
                               "server_alias": "5080" if role == "taxi_5080" else "5090",
                               "wrapper_path": partition.__file__, "python": str(__import__("sys").executable)}
                       for role, qualification in (("taxi_5080", "new_cuda_required"),
                                                  ("continuation_5090", "reused_parent_receipt"))}, "arms": arms}
    return value, parent


def test_contract_enforces_exact_split_and_parent_identity(tmp_path):
    contract, parent = fixture(tmp_path)
    assert partition.validate_contract(contract, parent, wrapper_path=partition.__file__, now=100)["arms"] == 12
    altered = copy.deepcopy(contract)
    altered["arms"][0]["host"] = "taxi_5080"
    with pytest.raises(ValueError, match="Host arm split"):
        partition.validate_contract(altered, parent, wrapper_path=partition.__file__, now=100)
    altered = copy.deepcopy(contract)
    altered["parent"]["contract_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Parent contract SHA"):
        partition.validate_contract(altered, parent, wrapper_path=partition.__file__, now=100)


def test_contract_rejects_duplicate_resume_cpu_fallback_and_deadline_extension(tmp_path):
    contract, parent = fixture(tmp_path)
    altered = copy.deepcopy(contract)
    altered["arms"][1]["dataset_id"] = altered["arms"][0]["dataset_id"]
    altered["arms"][1]["case"] = altered["arms"][0]["case"]
    with pytest.raises(ValueError, match="Exactly one assignment"):
        partition.validate_contract(altered, parent, wrapper_path=partition.__file__, now=100)
    altered = copy.deepcopy(contract)
    altered["policy"]["cpu_fallback"] = True
    with pytest.raises(ValueError, match="policy"):
        partition.validate_contract(altered, parent, wrapper_path=partition.__file__, now=100)
    altered = copy.deepcopy(contract)
    altered["deadline_unix"] = 201
    with pytest.raises(ValueError, match="deadline"):
        partition.validate_contract(altered, parent, wrapper_path=partition.__file__, now=100)


def test_approval_and_launch_gates_are_sha_bound_and_gpu_safe(tmp_path):
    contract, parent = fixture(tmp_path)
    approval = {"status": "approved_by_user", "partition_contract_sha256": partition.sha_json(contract),
                "scope": "partitioned_continuation_taxi_5080_and_remaining_5090_arms", "user_instruction": "split it"}
    partition.validate_approval(contract, approval)
    with pytest.raises(ValueError, match="GPU already"):
        partition.assert_launch_gates(contract, "taxi_5080", disk_free=10 * 1024 ** 3, gpu_pids=[1], executable=__import__("sys").executable)
    with pytest.raises(ValueError, match="Pinned host Python"):
        partition.assert_launch_gates(contract, "taxi_5080", disk_free=10 * 1024 ** 3, gpu_pids=[], executable="/wrong/python")


def test_inherited_deadline_and_training_identity_are_not_rewritten(tmp_path):
    contract, parent = fixture(tmp_path)
    with pytest.raises(TimeoutError):
        partition.Budget(contract, tmp_path, clock=lambda: 201, monotonic=lambda: 0)()
    identity = partition.partition_identity(parent, contract, "continuation_5090", {"device": "cuda:0"})
    assert identity["execution_contract_sha256"] == partition.sha_json(parent)
    assert set(identity) == {"execution_contract_sha256", "runtime", "source"}
    assert partition.effective_parent_contract(parent, contract, "continuation_5090") == parent
    with pytest.raises(ValueError, match="runtime"):
        partition.partition_identity(parent, contract, "taxi_5080", {"device": "cpu"})


def runtime_fixture(tmp_path):
    contract, parent = fixture(tmp_path)
    deadline = time.time() + 7200
    contract['deadline_unix'] = contract['parent']['deadline_unix'] = deadline
    contract['parent']['started_at_unix'] = deadline - 172800
    for role, host in contract['hosts'].items():
        host.update(source_root=str(tmp_path / 'source'), output_dir=str(tmp_path / role))
        host['runtime_expected'] = {'device': 'cuda:0', 'environment': {'PYTHONHASHSEED': '42'},
                                    'gpu': {'uuid': role}}
    parent['runtime_expected'] = contract['hosts']['continuation_5090']['runtime_expected']
    contract['parent']['contract_sha256'] = partition.sha_json(parent)
    contract['parent']['execution_receipt'] = {'contract_sha256': partition.sha_json(parent),
        'started_at_unix': deadline - 172800, 'deadline_unix': deadline}
    for arm in contract['arms']:
        if arm['mode'] == 'resume':
            arm['resume_identity']['execution_contract_sha256'] = partition.sha_json(parent)
    approval = {'status': 'approved_by_user', 'partition_contract_sha256': partition.sha_json(contract),
                'scope': 'partitioned_continuation_taxi_5080_and_remaining_5090_arms', 'user_instruction': 'split'}
    handoff = {'partition_contract_sha256': partition.sha_json(contract), 'approved_transfer': True,
               'observed_old_process_dead': True}
    return contract, parent, approval, handoff


def authority_fixture(tmp_path, role='taxi_5080'):
    contract, parent, approval, handoff = runtime_fixture(tmp_path)
    host = contract['hosts'][role]
    authority = {'contract': contract, 'parent': parent, 'role': role, 'approval': approval,
                 'handoff': handoff, 'source_root': host['source_root'], 'output': host['output_dir'],
                 'deadline_unix': contract['deadline_unix'], 'supervisor_pid': 1234}
    path = Path(host['output_dir']) / 'partition_authority.json'
    partition.write_json(path, authority)
    return authority, path


def test_training_identity_exact_resume_and_native_5080_binding(tmp_path):
    contract, parent, _, _ = runtime_fixture(tmp_path)
    data = {'dataset_id': 'intermittent_frozen_5000', 'inherited_data_identity': {'split': 'synthetic'}}
    runtime = contract['hosts']['continuation_5090']['runtime_expected']
    expected = {'execution_contract_sha256': partition.sha_json(parent), 'runtime': runtime,
                'source': parent['source'], 'data': data['inherited_data_identity'], 'dataset_id': data['dataset_id']}
    assert partition.training_identity(parent, contract, 'continuation_5090', data, runtime) == expected
    effective = partition.effective_parent_contract(parent, contract, 'taxi_5080')
    assert effective['source'] == parent['source'] and effective['datasets'] == parent['datasets']
    native = partition.training_identity(parent, contract, 'taxi_5080', data,
                                        contract['hosts']['taxi_5080']['runtime_expected'])
    assert native['execution_contract_sha256'] == partition.sha_json(effective) != partition.sha_json(parent)
    assert 'partition_contract_sha256' not in native


@pytest.mark.parametrize('mutation, message', [('approval', 'approval SHA'), ('handoff', 'ownership'),
                                              ('source', 'source root'), ('deadline', 'Child deadline')])
def test_each_child_rechecks_authority_before_loading_runtime(tmp_path, monkeypatch, mutation, message):
    authority, path = authority_fixture(tmp_path)
    monkeypatch.setattr(partition, 'verify_source', Mock())
    if mutation == 'approval':
        authority['approval']['partition_contract_sha256'] = 'wrong'
    elif mutation == 'handoff':
        authority['handoff']['observed_old_process_dead'] = False
    elif mutation == 'source':
        authority['source_root'] = '/wrong'
    else:
        authority['deadline_unix'] += 1
    partition.write_json(path, authority)
    runtime = Mock()
    monkeypatch.setattr(partition, '_loaded_frozen', runtime)
    with pytest.raises(ValueError, match=message):
        partition.child_main('_probe', path, case=partition.CASES[0], stage='full')
    runtime.assert_not_called()


@pytest.mark.parametrize('mode', ['_probe', '_shapes'])
def test_child_routes_use_frozen_functions_without_legacy_cli_approval(tmp_path, monkeypatch, mode):
    authority, path = authority_fixture(tmp_path)
    monkeypatch.setattr(partition, 'verify_source', Mock())
    monkeypatch.setattr(partition.os, 'kill', Mock())
    monkeypatch.setattr(partition.os, 'getppid', lambda: 555)
    monkeypatch.setattr(partition.os, 'getpgid', lambda _: 555)
    import resource
    monkeypatch.setattr(resource, 'setrlimit', Mock())
    partition.write_json(path.parent / 'worker_owner.json',
                         {'pid': 555, 'partition_contract_sha256': partition.sha_json(authority['contract'])})
    frozen = SimpleNamespace(probe_worker=Mock(return_value={}), configured_runtime=Mock(),
                             production_shape_checks=Mock())
    monkeypatch.setattr(partition, '_loaded_frozen', lambda _: frozen)
    partition.child_main(mode, path, case=partition.CASES[0], stage='full')
    effective = partition.effective_parent_contract(authority['parent'], authority['contract'], 'taxi_5080')
    if mode == '_probe':
        frozen.probe_worker.assert_called_once_with(effective, partition.CASES[0], 'full',
            authority['contract']['deadline_unix'] - 172800, 'cuda:0')
    else:
        args = frozen.production_shape_checks.call_args.args
        assert args[1] == 'cuda:0'
        assert [d['dataset_id'] for d in args[0]['datasets']] == ['yellow_trip_hourly']
    assert not (path.parent / 'execution_receipt.json').exists()


def test_5080_qualification_launches_thirteen_wrapper_children_and_checks_replay(tmp_path, monkeypatch):
    authority, path = authority_fixture(tmp_path)
    contract, parent = authority['contract'], authority['parent']
    runtime = contract['hosts']['taxi_5080']['runtime_expected']
    effective = partition.effective_parent_contract(parent, contract, 'taxi_5080')
    calls = []
    def run(command, *, check, timeout):
        assert check and 0 < timeout <= 900
        assert command[1] == str(Path(partition.__file__).resolve())
        assert '--authority' in command and '--contract' not in command
        calls.append(command)
        if command[2] == '_probe':
            case = command[command.index('--case') + 1]
            stage = command[command.index('--stage') + 1]
            folder = path.parent / 'qualification' / case / ('full' if stage == 'full' else 'split')
            partition.write_json(folder / 'summary.json', {'case': case})
    from simple_lab_test.search.common import runner
    monkeypatch.setattr(partition.subprocess, 'run', run)
    monkeypatch.setattr(runner, 'torch_load_checkpoint', lambda *a, **k: {
        'optimizer_state_sha256': 'a', 'rng_state_sha256': 'b', 'model_state_sha256': 'c'})
    frozen = SimpleNamespace(configured_runtime=lambda *args: runtime, compare_replay=Mock(),
                             audit_pairs=Mock(), validate_contract=Mock())
    receipt = partition._qualification(contract, parent, 'taxi_5080', frozen, effective, path)
    assert len(calls) == 13 and calls[-1][2] == '_shapes'
    assert frozen.compare_replay.call_count == 4
    assert receipt['execution_contract_sha256'] == partition.sha_json(effective)
    assert receipt['passed'] and receipt['qualifies_cuda']


def test_budget_ignores_disappearing_atomic_temporary_file(tmp_path, monkeypatch):
    contract, _ = fixture(tmp_path)
    missing = SimpleNamespace(stat=Mock(side_effect=FileNotFoundError))
    monkeypatch.setattr(Path, 'rglob', lambda *args: iter([missing]))
    partition.Budget(contract, tmp_path, clock=lambda: 100, monotonic=lambda: 0)('before_epoch_commit')
    missing.stat.assert_called_once()


def test_monotonic_deadline_does_not_extend_if_wall_clock_moves_backwards(tmp_path):
    contract, _ = fixture(tmp_path)
    tick = [0]
    clock = [100]
    budget = partition.Budget(contract, tmp_path, clock=lambda: clock[0], monotonic=lambda: tick[0])
    clock[0], tick[0] = 0, 101
    with pytest.raises(TimeoutError):
        budget()


def fake_completion(data, case, identity):
    arm_contract = {'identity': identity, 'condition': {'case': case}}
    summary = {'status': 'complete', 'epochs_completed': 120, 'epochs_budget': 120,
               'global_step': data['expected_global_steps'], 'contract_sha256': partition.sha_json(arm_contract),
               'condition': {'case': case}, 'evaluation_scope': 'validation_only', 'held_out_test_evaluated': False,
               'initial_state_sha256': 'same', 'history': []}
    for epoch in range(1, 121):
        summary['history'].append({'epoch': epoch, 'global_step': epoch,
            'train_count': 128, 'validation_count': 64, 'train_batches': 1, 'validation_batches': 1,
            'train_batch_order_sha256': str(epoch)})
    return arm_contract, summary


def test_selected_5090_execution_preserves_resume_identity_and_audits_two_datasets(tmp_path, monkeypatch):
    contract, parent, _, handoff = runtime_fixture(tmp_path)
    role = 'continuation_5090'
    runtime = contract['hosts'][role]['runtime_expected']
    output = Path(contract['hosts'][role]['output_dir'])
    for data in parent['datasets']:
        data.update(expected_global_steps=120, inherited_data_identity={'populations': {
            'train': {'target_count': 128}, 'validation': {'target_count': 64}}})
    for arm in partition.assigned_arms(contract, role):
        data = partition._parent_map(parent)[arm['dataset_id']]
        identity = partition.training_identity(parent, contract, role, data, runtime)
        saved_contract, summary = fake_completion(data, arm['case'], identity)
        inherited = tmp_path / 'inherited' / arm['case']
        if arm['mode'] in {'resume', 'reuse'}:
            partition.write_json(inherited / 'contract.json', saved_contract)
            partition.write_json(inherited / 'summary.json', summary)
        if arm['mode'] == 'reuse':
            arm['reuse_summary'] = {'path': str(inherited / 'summary.json'), 'sha256': partition.sha_file(inherited / 'summary.json')}
            arm['reuse_contract'] = {'path': str(inherited / 'contract.json'), 'sha256': partition.sha_file(inherited / 'contract.json')}
        elif arm['mode'] == 'resume':
            (inherited / 'last_epoch_state.pt').write_bytes(b'synthetic-copy-only-never-loaded')
            arm.update(resume_from=str(inherited), resume_identity=identity,
                       resume_contract_sha256=partition.sha_json(saved_contract))
            handoff['inherited_files'] = {str(p.resolve()): partition.sha_file(p) for p in inherited.iterdir()}
    calls = []
    def run_case(**kwargs):
        calls.append(kwargs)
        data = partition._parent_map(parent)[kwargs['identity']['dataset_id']]
        saved_contract, summary = fake_completion(data, kwargs['case'], kwargs['identity'])
        partition.write_json(Path(kwargs['output_dir']) / 'contract.json', saved_contract)
        return summary
    frozen = SimpleNamespace(configured_runtime=lambda *a: runtime, bind_frozen_statistics=lambda d, m: m,
                             audit_pairs=parent_cli.audit_pairs)
    modules = {
        'paper.scripts.quantity_comparison_data': SimpleNamespace(prepare_quantity_comparison_data=lambda d: (None, {'train_log_mean': 1, 'raw_scale': 2})),
        'paper.scripts.quantity_comparison_engine': SimpleNamespace(run_case=run_case),
        'paper.scripts.quantity_objective_comparison': SimpleNamespace(QuantityCase=lambda c: c, QuantityStatistics=lambda *a: a),
        'paper.scripts.run_time_quantity_diagnostic': SimpleNamespace(build_arm_inputs=lambda *a: (None, None, None, None))}
    original_import = partition.importlib.import_module
    monkeypatch.setattr(partition.importlib, 'import_module', lambda name: modules[name] if name in modules else original_import(name))
    monkeypatch.setattr(partition, '_frozen_runner', lambda root: frozen)
    results = partition._run_selected_arms(contract, parent, role, '/fake/source', output,
                                           {'runtime': runtime}, handoff)
    assert len(results) == 8 and len(calls) == 5
    assert calls[0]['resume'] is True and calls[0]['identity'] == contract['arms'][3]['resume_identity']
    assert all('partition_contract_sha256' not in call['identity'] and call['device'] == 'cuda:0' for call in calls)
    for dataset in ('intermittent_frozen_5000', 'insta_market_basket'):
        assert partition.read_json(output / dataset / 'paired_comparison.json')['status'] == 'passed'


@pytest.mark.parametrize('tamper', ['identity', 'steps', 'population', 'scope'])
def test_completed_reuse_audit_rejects_wrong_run_or_incomplete_exposure(tamper):
    data = {'expected_global_steps': 120, 'inherited_data_identity': {'populations': {
        'train': {'target_count': 128}, 'validation': {'target_count': 64}}}}
    saved_contract, summary = fake_completion(data, 'raw_original', {'run': 'original'})
    if tamper == 'identity':
        saved_contract['identity'] = {'run': 'different'}
    elif tamper == 'steps':
        summary['global_step'] = 119
    elif tamper == 'population':
        summary['history'][-1]['train_count'] = 127
    else:
        summary['held_out_test_evaluated'] = True
    with pytest.raises(ValueError):
        partition.audit_arm(summary, saved_contract, data, 'raw_original', {'run': 'original'})


def supervisor_fixture(tmp_path, monkeypatch):
    contract, parent, approval, handoff = runtime_fixture(tmp_path)
    monkeypatch.setattr(partition, 'verify_source', Mock())
    monkeypatch.setattr(partition, '_loaded_frozen', Mock())
    monkeypatch.setattr(partition, '_gpu_pids', Mock(return_value=[]))
    monkeypatch.setattr(partition.shutil, 'disk_usage', lambda _: SimpleNamespace(free=10 * 1024**3))
    return contract, parent, approval, handoff


def test_execute_requires_actual_gpu_idle_query_and_held_flock(tmp_path, monkeypatch):
    contract, parent, approval, handoff = supervisor_fixture(tmp_path, monkeypatch)
    role = 'taxi_5080'
    host = contract['hosts'][role]
    partition._gpu_pids.return_value = ['123']
    popen = Mock()
    monkeypatch.setattr(partition.subprocess, 'Popen', popen)
    import fcntl
    lock = Mock()
    monkeypatch.setattr(fcntl, 'flock', lock)
    with pytest.raises(ValueError, match='GPU already'):
        partition.execute(contract, approval, parent=parent, role=role, handoff=handoff,
                          source_root=host['source_root'], output=host['output_dir'])
    partition._gpu_pids.assert_called_once_with(host)
    lock.assert_called_once()
    popen.assert_not_called()


@pytest.mark.parametrize('exit_code', [0, 1])
def test_execute_supervises_worker_and_persists_failure_without_retry(tmp_path, monkeypatch, exit_code):
    contract, parent, approval, handoff = supervisor_fixture(tmp_path, monkeypatch)
    role = 'continuation_5090'
    host = contract['hosts'][role]
    output = Path(host['output_dir'])
    def launch(command, *, start_new_session):
        assert command[2] == '_worker' and start_new_session
        authority = partition.read_json(output / 'partition_authority.json')
        assert authority['approval'] == approval and authority['handoff'] == handoff
        if exit_code == 0:
            partition.write_json(output / 'partition_receipt.json',
                {'status': 'complete', 'partition_contract_sha256': partition.sha_json(contract)})
        return SimpleNamespace(wait=lambda timeout: exit_code)
    popen = Mock(side_effect=launch)
    cleanup = Mock()
    monkeypatch.setattr(partition.subprocess, 'Popen', popen)
    monkeypatch.setattr(partition, 'stop_owned_process_group', cleanup)
    kwargs = dict(parent=parent, role=role, handoff=handoff, source_root=host['source_root'], output=output)
    if exit_code:
        with pytest.raises(ValueError, match='no retry'):
            partition.execute(contract, approval, **kwargs)
        assert partition.read_json(output / 'status.json')['status'] == 'failed'
        cleanup.assert_called_once()
    else:
        assert partition.execute(contract, approval, **kwargs)['status'] == 'complete'
    popen.assert_called_once()


def test_supervisor_enforces_qualification_timeout_without_worker_callbacks(tmp_path, monkeypatch):
    contract, parent, approval, handoff = supervisor_fixture(tmp_path, monkeypatch)
    host = contract['hosts']['taxi_5080']
    budget = Mock()
    budget.deadline_mono = 10000
    monkeypatch.setattr(partition, 'Budget', lambda *args: budget)
    ticks = iter([0, 901])
    monkeypatch.setattr(partition.time, 'monotonic', lambda: next(ticks))
    process = SimpleNamespace(wait=Mock())
    monkeypatch.setattr(partition.subprocess, 'Popen', lambda *a, **k: process)
    cleanup = Mock()
    monkeypatch.setattr(partition, 'stop_owned_process_group', cleanup)
    with pytest.raises(TimeoutError, match='qualification'):
        partition.execute(contract, approval, parent=parent, role='taxi_5080', handoff=handoff,
                          source_root=host['source_root'], output=host['output_dir'])
    process.wait.assert_not_called()
    cleanup.assert_called_once_with(process)
    assert partition.read_json(Path(host['output_dir']) / 'status.json')['status'] == 'timeout'


def test_worker_dispatch_uses_partition_receipt_and_retains_inherited_deadline(tmp_path, monkeypatch):
    authority, path = authority_fixture(tmp_path, 'continuation_5090')
    monkeypatch.setattr(partition, 'verify_source', Mock())
    monkeypatch.setattr(partition.os, 'kill', Mock())
    monkeypatch.setattr(partition.os, 'getppid', lambda: authority['supervisor_pid'])
    monkeypatch.setattr(partition.os, 'getpid', lambda: 555)
    monkeypatch.setattr(partition.os, 'getpgid', lambda _: 555)
    import resource
    limits = Mock()
    monkeypatch.setattr(resource, 'setrlimit', limits)
    frozen = Mock()
    monkeypatch.setattr(partition, '_loaded_frozen', lambda _: frozen)
    receipt = {'passed': True, 'qualifies_cuda': True}
    qualify = Mock(return_value=receipt)
    run = Mock(return_value={('insta_market_basket', 'raw_original'): {}})
    monkeypatch.setattr(partition, '_qualification', qualify)
    monkeypatch.setattr(partition, '_run_selected_arms', run)
    result = partition.child_main('_worker', path)
    assert qualify.call_args.args[4] == authority['parent']
    assert run.call_args.args[-1] == authority['handoff']
    assert result['deadline_unix'] == authority['deadline_unix']
    assert partition.read_json(path.parent / 'qualification/receipt.json') == receipt
    assert not (path.parent / 'execution_receipt.json').exists()
    limits.assert_called_once_with(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    with pytest.raises(ValueError, match='already claimed'):
        partition.child_main('_worker', path)
    run.assert_called_once()


@pytest.mark.parametrize('field', ['deadline_unix', 'started_at_unix'])
def test_original_execution_receipt_prevents_joint_deadline_extension(tmp_path, field):
    contract, parent = fixture(tmp_path)
    contract['parent']['execution_receipt'][field] += 1
    with pytest.raises(ValueError, match='Original execution deadline'):
        partition.validate_contract(contract, parent, now=100)


def test_runtime_environment_sets_and_removes_pinned_values(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'wrong')
    monkeypatch.setenv('CUDA_DEVICE_ORDER', 'wrong')
    partition._apply_environment({'runtime_expected': {'environment': {
        'CUDA_VISIBLE_DEVICES': '0', 'CUDA_DEVICE_ORDER': None}}})
    assert partition.os.environ['CUDA_VISIBLE_DEVICES'] == '0'
    assert 'CUDA_DEVICE_ORDER' not in partition.os.environ
