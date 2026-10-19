"""Fabricated-only raw-time queue and parent-completion tests; no inference/SSH."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import shlex
import sys
import tarfile
from types import ModuleType, SimpleNamespace

import pytest
from paper.scripts import manage_titantpp_raw_time_mlp27 as m
from paper.scripts import titantpp_raw_time_mlp27_runtime as r
from paper.scripts import manage_titantpp_quantity_loss_mlp18_v2 as pm
from paper.scripts import titantpp_quantity_loss_mlp18_runtime_v2 as pr
from paper.tests.test_titantpp_quantity_loss_mlp18_operations import contract as old_fixture, proof, stopped


@pytest.fixture
def contract(tmp_path):
    c = old_fixture.__wrapped__(tmp_path)
    c.update(schema=r.SCHEMA, jobs=m.build_jobs(), parent_quantity_loss27={'contract_raw_sha256': 'parent-raw', 'contract_canonical_sha256': 'parent-canonical', 'source_files_sha256': 'parent-source'})
    for host in c['hosts']:
        c['hosts'][host].update(tmux_binary='tmux', tmux='raw_time27_' + host)
    c['architecture'].update(arms=list(r.ARMS), quantity_loss_modes=list(r.LOSS_MODES), time_input_transform='raw_nonnegative_float32', quantity_input_transform='log1p_nonnegative_float32')
    c['baseline_validation_references'] = [{'dataset': d, 'host': '5080' if d == 'yellow_trip_hourly' or (d == 'intermittent_frozen_5000' and s == 42) else '5090', 'seed': s, 'loss_mode': loss, 'time_alpha': 0., 'time_head_trained': False, 'reevaluated': False, 'retrained': False, 'sha256': 'bound'} for d in r.DATASETS for s in (42, 52, 62) for loss in r.LOSS_MODES]
    return c


def test_exact27_12_15_same_assignment_no_test(contract):
    assert len(r.validate_contract(contract, '5080')) == 12
    assert len(r.validate_contract(contract, '5090')) == 15
    assert r.STAGES == ('train', 'validation') and r.LOSS_MODES == ('log_mse', 'raw_mse', 'raw_mae')
    assert r.MODULE == 'paper.scripts.titantpp_raw_time_mlp27_runtime'
    assert r.WORKER_MODULE == 'paper.scripts.run_titantpp_raw_time_mlp27'
    assert m._impl.runtime is r


@pytest.mark.parametrize('field,value', [('time_alpha', 1.), ('time_alpha', False), ('loss_mode', 'normalized_mse'), ('arm', 'cnn'), ('seed', 32), ('host', '5090'), ('id', '../bad')])
def test_job_mutation_rejected(contract, field, value):
    c = deepcopy(contract); c['jobs'][0][field] = value
    with pytest.raises((ValueError, KeyError)): r.validate_contract(c, '5080')


@pytest.mark.parametrize('mutation', ['test_auth', 'test_input', 'budget', 'baseline', 'transform', 'quantity', 'normalization', 'deadline', 'paused', 'parent'])
def test_scope_rejected(contract, mutation):
    c = deepcopy(contract)
    if mutation == 'test_auth': c['held_out_test_authorized'] = True
    if mutation == 'test_input': c['input_files']['inputs/test.parquet'] = 'test'
    if mutation == 'budget': c['training']['minimum_epochs'] = 1
    if mutation == 'baseline': c['baseline_validation_references'].pop()
    if mutation == 'transform': c['architecture']['time_input_transform'] = 'log1p_nonnegative_float32'
    if mutation == 'quantity': c['architecture']['quantity_input_transform'] = 'raw'
    if mutation == 'normalization': c['architecture']['quantity_loss_normalization'] = 'cap'
    if mutation == 'deadline': c['hosts']['5080']['absolute_host_deadline_unix'] = None
    if mutation == 'paused': c['hosts']['5080']['paused_processes'][0]['state'] = 'S'
    if mutation == 'parent': c['hosts']['5080']['completed_predecessor']['final_receipt_sha256'] = ''
    with pytest.raises((ValueError, KeyError)): r.validate_contract(c, '5080')


def make_receipt(c, host, job, stage):
    root = Path(c['hosts'][host]['root']); folder = root / 'run' / job['id']; folder.mkdir(parents=True, exist_ok=True)
    binary = folder / 'selected.pt'
    if not binary.exists(): binary.write_bytes(b'fixed selection')
    value = {'schema': 'titantpp_raw_time_mlp27_stage_v1', 'status': 'complete', 'job_id': job['id'], 'stage': stage, 'contract_sha256': r.sha_json(c), **{k: job[k] for k in ('dataset', 'arm', 'seed', 'time_alpha', 'loss_mode')}, 'held_out_test_evaluated': False, 'quantity_only': True, 'time_head_trained': False, 'quantity_loss_normalization': 'none', 'time_head_no_training_verification': proof(), 'time_input_transform': 'raw_nonnegative_float32', 'quantity_input_transform': 'log1p_nonnegative_float32', 'selected_checkpoint': str(binary), 'checkpoint_sha256': r.sha_file(binary), 'selected_epoch': 3, 'state_sha256': 's' * 64, 'target_identity': {'target_count': 4}, 'files': {str(binary.relative_to(root)): r.sha_file(binary)}}
    if stage == 'validation': value['validation_replay_passed'] = True
    r.write(folder / (stage + '_receipt.json'), value, exclusive=True)
    return value


def test_scientific_receipt_selected_binary_and_input_proof(contract):
    j = r.validate_contract(contract, '5080')[0]
    make_receipt(contract, '5080', j, 'train'); make_receipt(contract, '5080', j, 'validation')
    assert r.verify_receipt(contract, '5080', j, 'validation')['sha256']
    p = Path(contract['hosts']['5080']['root']) / 'run' / j['id'] / 'validation_receipt.json'
    v = r.read(p); v['time_input_transform'] = 'log1p_nonnegative_float32'; r.write(p, v)
    with pytest.raises(ValueError, match='transform proof'): r.verify_receipt(contract, '5080', j, 'validation')


def test_validation_requires_all_exact_jobs(contract):
    with pytest.raises(ValueError): r.validation_gate(contract, '5080', {})
    with pytest.raises(ValueError): r.verify_receipt(contract, '5080', contract['jobs'][0], 'test')


def test_modules_remain_private_and_parent_sources_unchanged():
    assert r._impl is not pr._impl and m._impl is not pm._impl
    assert r._impl.run_stage.__globals__ is r._impl.__dict__
    assert m._impl.collect.__globals__ is m._impl.__dict__
    assert pr.WORKER_MODULE == 'paper.scripts.run_titantpp_quantity_loss_mlp18'
    assert pm.NAME == 'titantpp_quantity_loss_mlp18_20261019_v2'
    assert r.sha_file(Path(r.__file__).with_name('titantpp_quantity_loss_mlp18_runtime.py')) == r._BASE_SHA
    assert m.sha(Path(m.__file__).with_name('manage_titantpp_quantity_loss_mlp18.py')) == m._BASE_SHA


def test_prepare_does_not_create_bundle_before_parent_complete(tmp_path, monkeypatch):
    bundle = tmp_path / 'new'; monkeypatch.setattr(m._impl, 'BUNDLE', bundle)
    def missing(): raise ValueError('parent incomplete')
    monkeypatch.setattr(m, 'verify_parent_completion', missing)
    with pytest.raises(ValueError, match='parent incomplete'): m.prepare()
    assert not bundle.exists()


def test_qualification_self_context_scope_and_finally_restore(contract, monkeypatch):
    h = contract['hosts']['5080']; monkeypatch.chdir(h['source_root'])
    ident = {'pid': os.getpid(), 'pgid': os.getpid(), 'start_ticks': 8, 'cwd': h['source_root'], 'argv': [h['python'], '-u', '-m', r.MODULE, '--contract', str(Path(h['root']) / 'execution_contract.json'), '--host', '5080', '--mode', 'qualify']}
    r.write(Path(h['root']) / 'qualification_attempt.json', {'host': '5080', 'contract_sha256': r.sha_json(contract)}, exclusive=True)
    monkeypatch.setattr(r._impl, 'proc_identity', lambda p: deepcopy(ident) if p == os.getpid() else None)
    calls = []; original = r._impl.gpu_state
    def gpu(c, host, active_pids=()): calls.append(tuple(active_pids)); return {'compute_pids': list(active_pids)}, stopped()
    monkeypatch.setattr(r._impl, 'gpu_state', gpu)
    def checks(c, host):
        snapshot, paused = r._impl.gpu_state(c, host)
        return {'status': 'passed', 'gpu_after': snapshot}
    monkeypatch.setattr(r, '_original_three', checks)
    result = r.qualify_three_concurrent(contract, '5080')
    assert result['qualification_self_context_allowed'] is True and calls == [(os.getpid(),)]
    assert r._impl.gpu_state is gpu and original is not gpu


def test_generated_supervisor_json_real_readback(contract, tmp_path, monkeypatch):
    host = '5080'; captured = []
    monkeypatch.setattr(m._impl, 'verify_local', lambda **kw: contract)
    monkeypatch.setattr(m._impl, 'BUNDLE', tmp_path / 'bundle'); (tmp_path / 'bundle').mkdir(); (tmp_path / 'bundle/deployment.tar.gz').write_bytes(b'pack')
    r.write(tmp_path / 'bundle/execution_contract.json', contract, exclusive=True)
    monkeypatch.setattr(m._impl, 'REPORT', tmp_path / 'report')
    def ssh(host, command, **kwargs): captured.append(command); return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')
    monkeypatch.setattr(m._impl, 'ssh', ssh); monkeypatch.setattr(m._impl, 'save_transport', lambda *a: b'')
    m.deploy(host)
    shell = shlex.split(captured[-1])[-1]; words = shlex.split(shell); program = words[words.index('-c') + 1]
    fake = SimpleNamespace(pid=456, wait=lambda: 0)
    monkeypatch.setattr(m._impl.subprocess, 'Popen', lambda *a, **kw: fake)
    with pytest.raises(SystemExit) as exc: exec(program, {})
    assert exc.value.code == 0
    record = json.loads((Path(contract['hosts'][host]['root']) / 'dispatcher_exit.json').read_text())
    assert record['actual_os_returncode'] == 0 and record['pid'] == 456


def test_generated_collection_json_real_readback(contract, monkeypatch):
    host = '5080'; root = Path(contract['hosts'][host]['root']); jobs = r.validate_contract(contract, host); ids = {j['id'] for j in jobs}
    final = {'status': 'complete', 'host': host, 'contract_sha256': r.sha_json(contract), 'completed_fit_count': len(jobs), 'validation_replay_count': len(jobs), 'test_count': 0, 'stages': {stage: {'status': 'complete', 'exits': {j: {'actual_os_returncode': 0, 'timeout': False, 'pid': i + 100} for i, j in enumerate(ids)}, 'receipts': {j: {} for j in ids}} for stage in r.STAGES}}
    r.write(root / 'execution_contract.json', contract, exclusive=True); r.write(root / 'final_receipt.json', final, exclusive=True); r.write(root / 'dispatcher_exit.json', {'actual_os_returncode': 0, 'pid': 99}, exclusive=True)
    monkeypatch.setattr(r, 'verify_receipt', lambda *a: None); monkeypatch.setattr(r, 'gpu_state', lambda *a: ({'compute_pids': []}, stopped())); monkeypatch.setattr(r, 'proc_identity', lambda *a: None)
    buffer = io.BytesIO(); monkeypatch.setattr(sys, 'stdout', SimpleNamespace(buffer=buffer))
    exec(m.collection_program(contract, host), {})
    buffer.seek(0)
    with tarfile.open(fileobj=buffer, mode='r:gz') as a:
        observation = json.loads(a.extractfile('collection_observation.json').read())
    assert observation['owned_pids_absent'] is True and observation['prior_campaign_still_stopped'] is True


def test_source_snapshot_parent952_plus_only_new3_without_completion(tmp_path, monkeypatch):
    files = {}
    for i in range(952):
        name = f'old/{i}.py'; p = tmp_path / name; p.parent.mkdir(exist_ok=True); p.write_text(str(i)); files[name] = m.sha(p)
    for name in m.REQUIRED_NEW_SOURCE:
        p = tmp_path / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('new adapter')
    c = {'source': {'files': files, 'files_sha256': m.canonical(files), 'model_science_source_revision': 'science', 'model_science_source_files_sha256': 'science941'}}
    monkeypatch.setattr(m, 'parent_contract', lambda: (c, tmp_path / 'parent.json'))
    monkeypatch.setattr(m._impl, 'ROOT', tmp_path)
    def git(cmd, **kwargs):
        return '\n'.join(m.REQUIRED_NEW_SOURCE) if cmd[1] == 'ls-files' else ('newcommit\n' if cmd[1] == 'rev-parse' else '')
    monkeypatch.setattr(m._impl.subprocess, 'check_output', git)
    result = m.source_snapshot()
    assert len(result['files']) == 955 and result['parent_source_file_count'] == 952
    (tmp_path / 'old/1.py').write_text('changed')
    with pytest.raises(ValueError, match='Parent source changed'): m.source_snapshot()


def test_prepare_exact_inputs_no_test_and_nooverwrite(contract, tmp_path, monkeypatch):
    parent = deepcopy(contract); parent['stored_cap_mlp_reference'] = {'unchanged': True}; parent['data_receipt'] = {'unchanged': True}
    binding = {'contract_raw_sha256': 'parent-raw', 'contract_canonical_sha256': 'parent-canonical', 'source_files_sha256': 'parent-source', 'completion_sha256': 'complete-sha'}
    verified = (parent, {}, {}, binding)
    project = tmp_path / 'project'; project.mkdir(); bundle = project / 'new'; report = project / 'reports/new'; report.mkdir(parents=True)
    (report / 'SCIENCE_SPEC.json').write_text('{}')
    source = {'base_git_revision': 'newcommit', 'files': {'science.py': 'digest'}, 'files_sha256': r.sha_json({'science.py': 'digest'}), 'parent_source_file_count': 952}
    origin_root = project / 'search_artifacts' / m.PARENT_NAME
    for name, digest in parent['input_files'].items():
        origin = Path(contract['hosts']['5080']['source_root']) / name
        out = origin_root / 'input_snapshot' / name; out.parent.mkdir(parents=True, exist_ok=True); out.write_bytes(origin.read_bytes())
    for host in ('5080', '5090'):
        root = origin_root / 'retrieval' / host / 'originals'; root.mkdir(parents=True)
        r.write(root / 'final_receipt.json', {'status': 'complete'}, exclusive=True)
        r.write(root / 'dispatcher_exit.json', {'pid': 800, 'actual_os_returncode': 0}, exclusive=True)
    monkeypatch.setattr(m._impl, 'ROOT', project); monkeypatch.setattr(m._impl, 'BUNDLE', bundle); monkeypatch.setattr(m._impl, 'REPORT', report)
    monkeypatch.setattr(m, 'verify_parent_completion', lambda: verified)
    monkeypatch.setattr(m, 'baseline_validation_references', lambda **kw: deepcopy(contract['baseline_validation_references']))
    monkeypatch.setattr(m, 'source_snapshot', lambda: source)
    def package(c, s):
        archive = bundle / 'deployment.tar.gz'; archive.write_bytes(b'fake archive'); return archive
    monkeypatch.setattr(m._impl, 'package_archive', package)
    result = m.prepare(); made = r.read(bundle / 'execution_contract.json')
    assert result['planned_fit_count'] == result['baseline_reference_count'] == 27
    assert result['host_fit_counts'] == {'5080': 12, '5090': 15}
    assert len(made['input_files']) == 6 and made['packaging']['test_inputs_packaged'] is False
    assert made['limits'] == parent['limits'] and made['training'] == parent['training']
    for host in ('5080', '5090'):
        assert made['hosts'][host]['absolute_host_deadline_unix'] == parent['hosts'][host]['absolute_host_deadline_unix']
        assert made['hosts'][host]['paused_processes'] == parent['hosts'][host]['paused_processes']
        assert made['hosts'][host]['environment']['MPLCONFIGDIR'].startswith(made['hosts'][host]['root'])
    assert all(r.sha_file(bundle / 'input_snapshot' / name) == digest for name, digest in made['input_files'].items())
    with pytest.raises(ValueError, match='no overwrite'): m.prepare()


def fabricated_parent_completion(tmp_path, monkeypatch):
    parent_c = old_fixture.__wrapped__(tmp_path / 'parentdata')
    bundle = tmp_path / 'parentbundle'; report = tmp_path / 'parentreport'; report.mkdir()
    cp = bundle / 'execution_contract.json'; r.write(cp, parent_c, exclusive=True)
    monkeypatch.setattr(m, 'parent_contract', lambda: (parent_c, cp))
    monkeypatch.setattr(pm, 'BUNDLE', bundle); monkeypatch.setattr(pm, 'REPORT', report)
    monkeypatch.setattr(pm, 'verify_local', lambda **kw: parent_c)
    monkeypatch.setattr(pm, 'verify_collected', lambda *a: {})
    records = [{'dataset': d, 'host': '5080' if d == 'yellow_trip_hourly' or (d == 'intermittent_frozen_5000' and s == 42) else '5090', 'seed': s, 'loss_mode': mode, 'time_alpha': 0., 'selected_epoch': 3, 'target_identity': {'count': 4}, 'metrics': {'overall': {'qty_rmse': 2.}}} for d in r.DATASETS for s in (42, 52, 62) for mode in r.LOSS_MODES]
    comparison = {'status': 'complete', 'scope': 'validation_only', 'comparison_model_count': 27, 'held_out_test_evaluated': False, 'records': records}
    comparison_path = report / 'comparison_v1/THREE_LOSS_VALIDATION.json'; r.write(comparison_path, comparison, exclusive=True)
    archives = {}; endpoints = {}; logrefs = []
    for rec in records:
        if rec['loss_mode'] == 'log_mse': logrefs.append({**rec, 'time_head_trained': False, 'retrained': False, 'reevaluated': False, 'sha256': 'log-bound'})
    parent_c['baseline_validation_references'] = deepcopy(logrefs)
    for host in ('5080', '5090'):
        folder = bundle / 'retrieval' / host; originals = folder / 'originals'; originals.mkdir(parents=True)
        (originals / 'source-evidence').write_text('bound')
        for job in parent_c['jobs']:
            if job['host'] != host: continue
            directory = originals / 'run' / job['id']; directory.mkdir(parents=True)
            binary = directory / 'selected.pt'; binary.write_bytes(b'fixed')
            receipt = next(x for x in records if (x['dataset'], x['seed'], x['loss_mode']) == (job['dataset'], job['seed'], job['loss_mode']))
            receipt = {**receipt, 'time_head_no_training_verification': proof(), 'selected_checkpoint': str(Path(parent_c['hosts'][host]['root']) / 'run' / job['id'] / 'selected.pt'), 'checkpoint_sha256': r.sha_file(binary), 'state_sha256': 's' * 64}
            vp = directory / 'validation_receipt.json'; r.write(vp, receipt, exclusive=True); endpoints[str(vp)] = r.sha_file(vp)
            r.write(directory / 'runs/arm/data/seed_1/history.json', {'history': [{'epoch': 1, 'val_qty_rmse': 3.}, {'epoch': 3, 'val_qty_rmse': 2.}, {'epoch': 4, 'val_qty_rmse': 2.}]}, exclusive=True)
        inventory = {str(p.relative_to(originals)): r.sha_file(p) for p in originals.rglob('*') if p.is_file()}
        r.write(folder / 'manifest.json', {'files': inventory}, exclusive=True); (folder / 'original.tar.gz').write_bytes(b'archive')
        r.write(folder / 'receipt.json', {'status': 'complete', 'host': host, 'held_out_test_evaluated': False, 'archive_sha256': r.sha_file(folder / 'original.tar.gz'), 'manifest_sha256': r.sha_file(folder / 'manifest.json'), 'file_count': len(inventory)}, exclusive=True)
        for name in ('manifest.json', 'receipt.json', 'original.tar.gz'): archives[str(folder / name)] = r.sha_file(folder / name)
    r.write(cp, parent_c)
    final = {'status': 'complete', 'new_fit_count': 18, 'validation_replay_count': 18, 'baseline_reference_count': 9, 'comparison_model_count': 27, 'seed_count': 3, 'time_alpha': 0., 'time_head_no_training_verified_count': 18, 'actual_scientific_child_exit0_count': 36, 'actual_dispatcher_exit0_count': 2, 'held_out_test_evaluated': False, 'baseline_reevaluated': False, 'source_bindings': endpoints, 'archive_bindings': archives, 'report_files': {str(comparison_path): r.sha_file(comparison_path)}}
    fp = bundle / 'retrieval/final_receipt.json'; r.write(fp, final, exclusive=True)
    r.write(report / 'completion_v1/complete.json', {'status': 'complete', 'final_receipt_path': str(fp), 'final_receipt_sha256': r.sha_file(fp)}, exclusive=True)
    monkeypatch.setattr(pm, 'baseline_validation_references', lambda: deepcopy(logrefs))
    return bundle, report, final


def test_parent_complete_local_inventory_and_exact27_baselines(tmp_path, monkeypatch):
    bundle, report, final = fabricated_parent_completion(tmp_path, monkeypatch)
    verified = m.verify_parent_completion(); refs = m.baseline_validation_references(verified=verified)
    assert len(refs) == 27 and all(r['time_input_transform'] == 'log1p_nonnegative_float32' for r in refs)
    assert {x['loss_mode'] for x in refs} == set(r.LOSS_MODES)
    assert all(x['reevaluated'] is False and x['retrained'] is False for x in refs)
    original = bundle / 'retrieval/5080/originals/source-evidence'; original.write_text('tampered')
    with pytest.raises(ValueError, match='Parent original changed'): m.verify_parent_completion()


def test_parent_completion_missing_or_wrong_final_sha_blocks(tmp_path, monkeypatch):
    bundle, report, final = fabricated_parent_completion(tmp_path, monkeypatch)
    path = report / 'completion_v1/complete.json'; path.unlink()
    with pytest.raises(FileNotFoundError): m.verify_parent_completion()
    r.write(path, {'status': 'complete', 'final_receipt_path': str(bundle / 'retrieval/final_receipt.json'), 'final_receipt_sha256': 'wrong'}, exclusive=True)
    with pytest.raises(ValueError, match='finite completion'): m.verify_parent_completion()
