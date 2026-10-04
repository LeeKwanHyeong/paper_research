"""Synthetic offline finalizer proofs, immutable transfers and no-retry gates."""
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('capacity_test_finalizer', Path(__file__).resolve().parents[1] / 'scripts/finalize_titantpp_capacity_test.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)


@pytest.fixture
def local(tmp_path, monkeypatch):
    for key, relative in [('ROOT', ''), ('B', 'handoff'), ('PARENT', 'parent'), ('HERE', 'reports/evaluation'),
                          ('AUTHORITY', 'reports/approval.json'), ('CAMPAIGN', 'reports/evaluation/intermittent'), ('STATE', 'state')]:
        monkeypatch.setattr(f, key, tmp_path / relative)
    f.STATE.mkdir()
    return tmp_path


@pytest.fixture
def terminal_gate(local):
    jobs = [{'id': f'condition{i}', 'host': '5080' if i < 12 else '5090', 'seed': 42} for i in range(18)]
    parent, c = {'jobs': jobs}, {'jobs': deepcopy(jobs)}
    refs, analyses = {}, {}
    for role, base, chosen in [('historical12', f.PARENT, jobs[:12]), ('source4', f.B, jobs[12:16]), ('target2', f.B, jobs[16:])]:
        path = base / 'hourly_monitor' / role / 'host_snapshot.json'
        f.write(path, {'role': role})
        refs[role] = {'snapshot': str(path), 'snapshot_sha256': f.sha(path)}
        analyses[role] = {'server_terminal_verified': True, 'status_counts': {'completed': len(chosen), 'failed': 0, 'in_progress': 0, 'pending': 0, 'unknown': 0},
                          'rows': [{'job': j, 'terminal_verified': True} for j in chosen], 'approved_handoff_boundary_verified': True}
    f.write(f.B / 'source_reservation.json', {})
    f.write(f.AUTHORITY, {'approved': True, 'user_instruction': 'Evaluate after all six finish', 'parent_contract_sha256': 'parent-seal',
                         'evaluation': {'deferred_until_training_complete': ['intermittent_frozen_5000'], 'widths': [8, 12],
                                        'seeds': [42, 52, 62], 'splits': ['validation', 'test'], 'new_training': False, 'raw_predictions_written': False}})
    analyse = lambda raw, *a, **kw: deepcopy(analyses[raw['role']])
    obs = SimpleNamespace(old=SimpleNamespace(load_contract=lambda path: parent, analyse=analyse),
                          handoff=SimpleNamespace(PARENT_SHA='parent-seal', load=lambda path: (parent, c, {'contract_sha256': 'op-seal'}),
                                                  verify_native=lambda *a: {}, validate_reservation=lambda *a: True),
                          analyse_target=analyse, analyse_source=analyse)
    return refs, analyses, obs


def test_pending_invokes_neither_network_nor_scientific_import(local, monkeypatch):
    monkeypatch.setattr(f, 'observer', lambda: pytest.fail('Pending imported downstream science'))
    result = f.run_once(lambda *a, **k: pytest.fail('Pending attempted network/GPU'))
    assert result['status'] == 'pending' and result['remote_calls'] == 0
    assert not (f.STATE / 'gate_receipt.json').exists()


def test_only_one_latest_terminal_still_pending(local):
    f.write(f.B / 'hourly_monitor/latest_5080.json', {'server_terminal_verified': True, 'snapshot': 'irrelevant', 'snapshot_sha256': 'irrelevant'})
    assert f.run_once(lambda *a, **k: pytest.fail('Incomplete18 network'))['status'] == 'pending'


def test_crypto_reanalysis_accepts_exact18_and_all6(terminal_gate):
    refs, analyses, obs = terminal_gate
    result = f.gate(refs, obs)
    assert result['receipt']['conditions'] == 18 and result['receipt']['intermittent_conditions'] == 6


@pytest.mark.parametrize('role', ['target2', 'source4', 'historical12'])
def test_false_or_unknown_terminal_cannot_open_gate(terminal_gate, role):
    refs, analyses, obs = terminal_gate
    analyses[role]['server_terminal_verified'] = False
    with pytest.raises(ValueError, match='unknown/incomplete'): f.gate(refs, obs)


def test_snapshot_digest_and_duplicate_condition_rejected(terminal_gate):
    refs, analyses, obs = terminal_gate
    raw = Path(refs['source4']['snapshot']); original = raw.read_bytes(); raw.write_text('{}')
    with pytest.raises(ValueError, match='digest differs'): f.gate(refs, obs)
    raw.write_bytes(original)
    analyses['target2']['rows'][0]['job'] = analyses['source4']['rows'][0]['job']
    with pytest.raises(ValueError, match='Duplicate/missing'): f.gate(refs, obs)


def test_source_intentional_boundary_and_authority_required(terminal_gate):
    refs, analyses, obs = terminal_gate
    analyses['source4']['approved_handoff_boundary_verified'] = False
    with pytest.raises(ValueError, match='queue boundary'): f.gate(refs, obs)
    analyses['source4']['approved_handoff_boundary_verified'] = True
    approval = f.read(f.AUTHORITY); approval['evaluation']['new_training'] = True; f.write(f.AUTHORITY, approval)
    with pytest.raises(ValueError, match='authority changed'): f.gate(refs, obs)


def original_fixture(local):
    job = {'id': 'Inter__62__width8', 'dataset': 'intermittent_frozen_5000', 'host': '5080', 'seed': 62, 'arm': 'width8'}
    source = local / 'remote_original'; source.mkdir()
    (source / 'checkpoint.pt').write_bytes(b'original synthetic binary')
    (source / 'history.json').write_text('{"history":[]}')
    manifest = {'status': 'complete', 'scientific_success': True, 'job': job, 'contract_sha256': 'sealed',
                'files': {p.name: f.sha(p) for p in source.iterdir()}}
    f.write(source / 'terminal_manifest.json', manifest)
    proof = {'terminal_manifest_sha256': f.sha(source / 'terminal_manifest.json')}
    return job, source, proof


def test_retrieve_all_bound_bytes_once_then_verify_reuse(local):
    job, source, proof = original_fixture(local); calls = []
    def transfer(argv, **kwargs):
        calls.append(argv); compile(kwargs['input'].decode(), '<remote retrieval>', 'exec')
        with tarfile.open(fileobj=kwargs['stdout'], mode='w|gz') as archive:
            for path in source.iterdir(): archive.add(path, arcname=job['id'] + '/' + path.name, recursive=False)
        return SimpleNamespace(returncode=0)
    destination = local / 'retrieved'
    assert f.retrieve_originals('5080', '/root', [job], 'sealed', {job['id']: proof}, destination, transfer)['remote_calls'] == 1
    assert f.retrieve_originals('5080', '/root', [job], 'sealed', {job['id']: proof}, destination, transfer)['remote_calls'] == 0
    assert len(calls) == 1 and (destination / job['id'] / 'checkpoint.pt').read_bytes() == b'original synthetic binary'
    (destination / job['id'] / 'checkpoint.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='raw SHA differs'): f.retrieve_originals('5080', '/root', [job], 'sealed', {job['id']: proof}, destination, transfer)
    assert len(calls) == 1


def test_unsafe_tar_never_extracts_outside_staging(local):
    job, source, proof = original_fixture(local)
    def unsafe(argv, **kwargs):
        with tarfile.open(fileobj=kwargs['stdout'], mode='w|gz') as archive:
            info = tarfile.TarInfo('../escaped'); info.size = 4; archive.addfile(info, io.BytesIO(b'bad!'))
        return SimpleNamespace(returncode=0)
    with pytest.raises(ValueError, match='Unsafe original/archive'): f.retrieve_originals('5080', '/root', [job], 'sealed', {job['id']: proof}, local / 'retrieved', unsafe)
    assert not (f.STATE / 'escaped').exists()


def deployment_fixture(local):
    marker = f.PARENT / 'frozen_source/sample_data/.keep'; marker.parent.mkdir(parents=True); marker.write_text('')
    f.CAMPAIGN.mkdir(parents=True)
    c = {'resources': {'root': f.REMOTE, 'python': '/pinned/python', 'gpu_uuid': 'GPU-native', 'campaign_timeout_seconds': 7200,
                       'condition_timeout_seconds': 900, 'concurrent_workers': 1, 'output_limit_gib': 1},
         'approval': {'retraining_authorized': False}, 'raw_predictions_written': False,
         'validation_gate': {'conditions': 6, 'all_before_test': True}, 'environment': {'CUDA_VISIBLE_DEVICES': '0', 'PYTHONDONTWRITEBYTECODE': '1'}}
    f.write(f.B / 'execution_contract.json', {'hosts': {'5080': {'python': '/pinned/python', 'gpu_uuid': 'GPU-native',
             'environment': {'CUDA_VISIBLE_DEVICES': '0', 'SOURCE_REVISION': 'sealed-revision'}}}})
    f.write(f.CAMPAIGN / 'execution_contract.json', c)
    request = {'root': f.REMOTE, 'conditions': 6, 'dataset_transfer_required': False, 'environment': c['environment'],
               'files': {str(marker.relative_to(f.ROOT)): f.sha(marker)},
               'command': ['/pinned/python', str(Path(f.REMOTE) / f.HERE.relative_to(f.ROOT) / 'pipeline.py'), '--root', f.REMOTE,
                           '--campaign', str(Path(f.REMOTE) / f.CAMPAIGN.relative_to(f.ROOT)), '--device', 'cuda'],
               'preflight_command': ['/pinned/python', str(Path(f.REMOTE) / f.HERE.relative_to(f.ROOT) / 'preflight.py'), '--root', f.REMOTE,
                                     '--campaign', str(Path(f.REMOTE) / f.CAMPAIGN.relative_to(f.ROOT))]}
    return request


def test_deploy_preflight_launch_are_exclusive_and_idempotent(local, monkeypatch):
    request = deployment_fixture(local); phases = []
    def ssh(host, script, *a, **kw):
        compile(script, '<remote phase>', 'exec'); phases.append(script)
        if 'capture_output=True,text=True,timeout=180' in script: return {'status': 'complete', 'inference_calls': 0, 'gpu_calls': 0}
        return {'status': 'accepted'}
    monkeypatch.setattr(f, 'ssh', ssh)
    assert f.deploy_launch(request)['status'] == 'running'
    assert f.deploy_launch(request)['status'] == 'running' and len(phases) == 3
    assert 'timeout' in phases[-1] and '7200' in phases[-1] and '--device' in phases[-1]


def test_ambiguous_launch_never_retries(local, monkeypatch):
    request = deployment_fixture(local)
    for phase in ('deployment', 'preflight'): f.write(f.STATE / (phase + '_receipt.json'), {'request_sha256': f.canon(request)})
    f.write(f.STATE / 'launch_intent.json', {'requested': True})
    monkeypatch.setattr(f, 'ssh', lambda *a, **kw: pytest.fail('Ambiguous launch retried'))
    with pytest.raises(ValueError, match='retry forbidden'): f.deploy_launch(request)


def test_missing_marker_and_training_mutation_block_deployment(local):
    request = deployment_fixture(local)
    bad = deepcopy(request); bad['files'] = {}
    with pytest.raises(ValueError, match='marker missing'): f.validate_request(bad)
    c = f.read(f.CAMPAIGN / 'execution_contract.json'); c['approval']['retraining_authorized'] = True; f.write(f.CAMPAIGN / 'execution_contract.json', c)
    with pytest.raises(ValueError, match='resources/training'): f.validate_request(request)


def test_aggregate_collection_reads_named_files_only_and_detects_dead_session(local, monkeypatch):
    f.CAMPAIGN.mkdir(parents=True); f.write(f.CAMPAIGN / 'evaluation_registry.json', {'rows': []})
    scripts = []
    def collect(host, script, *a, **kw):
        scripts.append(script); compile(script, '<remote aggregate>', 'exec')
        return {'files': {}, 'session_alive': False}
    monkeypatch.setattr(f, 'ssh', collect)
    assert f.collect_aggregates()['status'] == 'failed'
    assert 'rglob' not in scripts[0] and 'predictions' not in scripts[0]
    monkeypatch.setattr(f, 'ssh', lambda *a, **kw: {'files': {'predictions/raw.json': {'text': '{}', 'sha256': 'wrong'}}, 'session_alive': True})
    with pytest.raises(ValueError, match='transport digest'): f.collect_aggregates()


def test_incomplete_population_ledger_cannot_analyse(local):
    f.CAMPAIGN.mkdir(parents=True)
    rows = [{'dataset': 'intermittent_frozen_5000', 'model': model, 'seed': seed} for model in ('titantpp_history_mlp_width8', 'titantpp_history_mlp_width12') for seed in (42, 52, 62)]
    f.write(f.CAMPAIGN / 'evaluation_registry.json', {'rows': rows}); f.write(f.CAMPAIGN / 'execution_contract.json', {})
    f.write(f.CAMPAIGN / 'validation_gate.json', {'passed': True})
    f.write(f.CAMPAIGN / 'inference_completion.json', {'status': 'complete', 'conditions': 6, 'full_population_splits': 12,
                'contract_sha256': f.sha(f.CAMPAIGN / 'execution_contract.json'), 'new_training': False, 'selection_changed': False,
                'raw_predictions_written': False, 'completed': [{}] * 11})
    with pytest.raises(ValueError, match='Incomplete/foreign'): f.verify_completion()


def test_unknown_gate_failure_returns_no_remote(local, monkeypatch):
    monkeypatch.setattr(f, 'gate', lambda *a: (_ for _ in ()).throw(ValueError('Completion snapshot digest differs')))
    result = f.run_once(lambda *a, **kw: pytest.fail('Unknown gate attempted network'))
    assert result['status'] == 'unknown' and result['automatic_retry'] is False


@pytest.mark.parametrize('kind', ['python', 'gpu', 'environment'])
def test_mutable_test_contract_cannot_change_native_owner(local, kind):
    request = deployment_fixture(local)
    c = f.read(f.CAMPAIGN / 'execution_contract.json')
    if kind == 'python': c['resources']['python'] = '/foreign/python'
    if kind == 'gpu': c['resources']['gpu_uuid'] = 'GPU-foreign'
    if kind == 'environment': c['environment']['CUDA_VISIBLE_DEVICES'] = '1'
    f.write(f.CAMPAIGN / 'execution_contract.json', c)
    with pytest.raises(ValueError, match='native Python/GPU/environment'): f.validate_request(request)


def test_completed_local_cache_revalidates_without_remote(local, monkeypatch):
    request = deployment_fixture(local)
    f.write(f.CAMPAIGN / 'deployment_request.json', request)
    f.write(f.STATE / 'launch_receipt.json', {'request_sha256': f.canon(request)})
    f.write(f.STATE / 'completion.json', {'status': 'completed', 'report': '/verified/results'})
    monkeypatch.setattr(f, 'gate', lambda *a: {'parent': {}, 'contract': f.read(f.B / 'execution_contract.json'), 'receipt': {'authority_sha256': 'authority'}})
    f.write(f.STATE / 'gate_receipt.json', {'refs': {}, 'authority_sha256': 'authority'})
    checks = []
    monkeypatch.setattr(f, 'verify_completion', lambda: checks.append('all12 receipts'))
    monkeypatch.setattr(f, 'analyse', lambda: checks.append('analysis SHA') or '/verified/results')
    result = f.run_once(lambda *a, **kw: pytest.fail('Completed cache attempted SSH'))
    assert result['status'] == 'completed' and result['remote_calls'] == 0 and result['reused_completed_evidence']
    assert checks == ['all12 receipts', 'analysis SHA']
