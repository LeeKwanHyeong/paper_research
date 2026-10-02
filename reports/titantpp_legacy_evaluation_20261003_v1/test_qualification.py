"""Synthetic regressions for validation evidence and campaign release guards."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


q = load_module('legacy_qualification', 'qualify.py')
d = load_module('legacy_dispatch', 'dispatch.py')

QUALIFICATION_CHECKS = (
    'target_quantity_perturbation_preserves_prediction',
    'target_quantity_perturbation_preserves_time_loss',
    'target_gap_perturbation_preserves_quantity_prediction',
    'single_vs_batch_prediction_matches',
    'single_vs_batch_time_loss_matches',
    'parameters_unchanged',
    'finite_outputs',
)


@pytest.fixture
def evidence():
    key = ('synthetic', 'rmtpp', 42)
    row = {
        'dataset': key[0], 'model': key[1], 'seed': key[2],
        'checkpoint_file_sha256': 'checkpoint-sha',
        'state_tensor_sha256': 'state-sha', 'selected_epoch': 7,
        'evaluator_source_bundle': 'frozen-bundle', 'validation_count': 1025,
    }
    contract = {
        '_sha256': 'contract-sha', 'registry_sha256': 'registry-sha',
        'dataset_manifest_sha256': 'manifest-sha',
        'dataset_sha256': {'synthetic': 'data-sha'},
        'resources': {'rss_limit_gib': 32, 'gpu_memory_limit_gib': 12},
    }
    dataset = {'populations': {'validation': {'target_count': 1025}}}
    receipt = {
        'schema_version': 1, 'status': 'complete',
        'dataset': key[0], 'model': key[1], 'seed': key[2], 'split': 'validation',
        'evaluation_scope': 'validation_qualification', 'full_population': False,
        'heldout_read': False, 'retrained': False, 'max_batches': 1, 'batch_size': 128,
        'expected_target_count': 1025, 'prediction_rows': 128,
        'registry_sha256': 'registry-sha', 'dataset_manifest_sha256': 'manifest-sha',
        'contract_sha256': 'contract-sha', 'runner_sha256': 'evaluator-sha',
        'data_file_sha256': 'data-sha',
        'checkpoint_file_sha256': row['checkpoint_file_sha256'],
        'state_tensor_sha256': row['state_tensor_sha256'],
        'selected_epoch': row['selected_epoch'],
        'evaluator_source_bundle': row['evaluator_source_bundle'],
        'qualification': {name: True for name in QUALIFICATION_CHECKS},
        'runtime': {'device': 'cpu', 'cpu_peak_rss_bytes': 1024**3,
                    'cuda_peak_allocated_bytes': 0, 'cuda_peak_reserved_bytes': 0},
    }
    return receipt, key, row, contract, dataset


def check(evidence, phase='qualification_cpu'):
    receipt, key, row, contract, dataset = evidence
    return q.check_receipt(receipt, key, phase, row, contract, dataset, 'evaluator-sha')


@pytest.mark.parametrize('phase,device,count,batches,full,scope', [
    ('qualification_cpu', 'cpu', 128, 1, False, 'validation_qualification'),
    ('qualification_cuda', 'cuda', 512, 4, False, 'validation_qualification'),
    ('validation_full', 'cuda', 1025, None, True, 'full_population'),
])
def test_receipt_accepts_only_the_declared_phase_shape(
        evidence, phase, device, count, batches, full, scope):
    receipt = evidence[0]
    receipt.update(prediction_rows=count, max_batches=batches,
                   full_population=full, evaluation_scope=scope)
    receipt['runtime']['device'] = device
    assert check(evidence, phase) is None


@pytest.mark.parametrize('field,value', [
    ('dataset', 'different-dataset'), ('model', 'different-model'), ('seed', 52),
    ('split', 'test'), ('status', 'failed'), ('heldout_read', True), ('retrained', True),
    ('registry_sha256', 'stale-registry'), ('dataset_manifest_sha256', 'stale-manifest'),
    ('contract_sha256', 'stale-contract'), ('runner_sha256', 'stale-evaluator'),
    ('data_file_sha256', 'changed-data'), ('checkpoint_file_sha256', 'changed-checkpoint'),
    ('state_tensor_sha256', 'changed-state'), ('selected_epoch', 8),
    ('evaluator_source_bundle', 'other-source'), ('expected_target_count', 1024),
    ('prediction_rows', 127), ('max_batches', 4), ('batch_size', 32),
    ('full_population', True), ('evaluation_scope', 'full_population'),
])
def test_receipt_cannot_substitute_other_run_or_population(evidence, field, value):
    evidence[0][field] = value
    with pytest.raises(ValueError):
        check(evidence)


@pytest.mark.parametrize('qualification', [
    {},
    {name: True for name in QUALIFICATION_CHECKS if name != 'finite_outputs'},
    {name: (False if name == 'parameters_unchanged' else True) for name in QUALIFICATION_CHECKS},
    {name: ('true' if name == 'finite_outputs' else True) for name in QUALIFICATION_CHECKS},
    {**{name: True for name in QUALIFICATION_CHECKS}, 'unrecognized_check': True},
])
def test_qualification_requires_every_check_to_be_literal_true(evidence, qualification):
    evidence[0]['qualification'] = qualification
    with pytest.raises(ValueError):
        check(evidence)


@pytest.mark.parametrize('phase,device,count,batches', [
    ('qualification_cpu', 'cuda', 128, 1),
    ('qualification_cuda', 'cpu', 512, 4),
    ('qualification_cuda', 'cuda', 128, 4),
    ('qualification_cuda', 'cuda', 512, 1),
])
def test_cpu_and_cuda_qualification_are_not_interchangeable(evidence, phase, device, count, batches):
    receipt = evidence[0]
    receipt.update(prediction_rows=count, max_batches=batches)
    receipt['runtime']['device'] = device
    with pytest.raises(ValueError):
        check(evidence, phase)


@pytest.mark.parametrize('field,value', [
    ('cpu_peak_rss_bytes', 32 * 1024**3 + 1),
    ('cuda_peak_reserved_bytes', 12 * 1024**3 + 1),
])
def test_receipt_must_satisfy_resource_limits(evidence, field, value):
    receipt = evidence[0]
    receipt.update(prediction_rows=512, max_batches=4)
    receipt['runtime'].update(device='cuda', **{field: value})
    with pytest.raises(ValueError):
        check(evidence, 'qualification_cuda')


def test_full_validation_rejects_a_partial_population(evidence):
    receipt = evidence[0]
    receipt.update(prediction_rows=1024, max_batches=None,
                   full_population=True, evaluation_scope='full_population')
    receipt['runtime']['device'] = 'cuda'
    with pytest.raises(ValueError):
        check(evidence, 'validation_full')


@pytest.fixture
def gate(tmp_path):
    proof = tmp_path / 'runs' / 'qualification_cpu' / 'receipt.json'
    proof.parent.mkdir(parents=True)
    proof.write_text('{"status": "complete"}\n')
    value = {
        'status': 'passed', 'code_seal_sha256': 'seal-sha',
        'contract_sha256': 'contract-sha',
        'evidence_hashes': {
            str(proof.relative_to(tmp_path)): hashlib.sha256(proof.read_bytes()).hexdigest(),
        },
    }
    return value, proof


def test_release_rehashes_evidence_instead_of_trusting_passed_status(tmp_path, gate):
    value, proof = gate
    d.verify_gate(value, tmp_path, 'seal-sha', 'contract-sha')
    proof.write_text('{"status": "failed"}\n')
    with pytest.raises(ValueError):
        d.verify_gate(value, tmp_path, 'seal-sha', 'contract-sha')


@pytest.mark.parametrize('field,value', [
    ('status', 'failed'), ('code_seal_sha256', 'previous-seal'),
    ('contract_sha256', 'previous-contract'), ('evidence_hashes', {}),
])
def test_release_rejects_stale_or_unbound_gate(tmp_path, gate, field, value):
    current = copy.deepcopy(gate[0])
    current[field] = value
    with pytest.raises(ValueError):
        d.verify_gate(current, tmp_path, 'seal-sha', 'contract-sha')


def test_release_evidence_cannot_escape_the_sealed_root(tmp_path, gate):
    root = tmp_path / 'sealed-root'
    root.mkdir()
    value, proof = gate
    value['evidence_hashes'] = {
        '../' + str(proof.relative_to(tmp_path)): hashlib.sha256(proof.read_bytes()).hexdigest(),
    }
    with pytest.raises(ValueError, match='escapes root'):
        d.verify_gate(value, root, 'seal-sha', 'contract-sha')


@pytest.mark.parametrize('later_epoch', [70000, 90000])
def test_campaign_reuses_original_24_hour_deadline_across_phases(tmp_path, later_epoch):
    path = tmp_path / 'campaign.json'
    first = d.load_campaign(path, 'contract-sha', 86400, now_epoch=1000)
    assert first['contract_sha256'] == 'contract-sha'
    assert first['started_epoch'] == 1000
    assert first['deadline_epoch'] == 87400
    original = path.read_bytes()
    later = d.load_campaign(path, 'contract-sha', 86400, now_epoch=later_epoch)
    assert later == first
    assert path.read_bytes() == original


def test_campaign_rejects_new_contract_without_overwriting_original(tmp_path):
    path = tmp_path / 'campaign.json'
    d.load_campaign(path, 'original-contract', 86400, now_epoch=1000)
    original = path.read_bytes()
    with pytest.raises(ValueError):
        d.load_campaign(path, 'changed-contract', 86400, now_epoch=2000)
    assert path.read_bytes() == original


def test_campaign_does_not_extend_existing_deadline_when_limit_changes(tmp_path):
    path = tmp_path / 'campaign.json'
    original = {'contract_sha256': 'contract-sha', 'started_epoch': 1000, 'deadline_epoch': 87400}
    path.write_text(json.dumps(original))
    saved = path.read_bytes()
    with pytest.raises(ValueError, match='deadline differs'):
        d.load_campaign(path, 'contract-sha', 172800, now_epoch=70000)
    assert path.read_bytes() == saved


def test_dispatch_manifest_paths_and_contract_alias_remain_analyzer_compatible(tmp_path, monkeypatch):
    here = tmp_path / 'reports' / 'synthetic_evaluation'
    here.mkdir(parents=True)
    registry_path = tmp_path / 'reports' / 'titantpp_final_eval_checkpoint_binding_20261003_v1' / 'evaluation_registry.json'
    registry_path.parent.mkdir()
    registry_path.write_text(json.dumps({'rows': []}))
    dataset_path = here / 'dataset_manifest.json'
    dataset_path.write_text(json.dumps({'datasets': []}))
    contract = {
        'approval': {'models': []}, 'resources': {},
        'registry_path': str(registry_path.relative_to(tmp_path)),
        'registry_sha256': d.sha(registry_path),
        'dataset_manifest_sha256': d.sha(dataset_path),
    }
    contract_path = here / 'execution_contract.json'
    contract_path.write_text(json.dumps(contract))
    contract_sha = d.sha(contract_path)
    monkeypatch.setattr(d, 'ROOT', tmp_path)
    monkeypatch.setattr(d, 'HERE', here)
    monkeypatch.setattr(d.time, 'time', lambda: 1000)

    def no_processes(*args, **kwargs):
        pytest.fail('An empty CPU manifest test must not start a worker or GPU query')

    monkeypatch.setattr(d.subprocess, 'Popen', no_processes)
    monkeypatch.setattr(d, 'bounded_output', no_processes)
    args = SimpleNamespace(phase='qualification_cpu', attempt='attempt1')
    assert d.run_phase(args, contract, {'rows': []}, 'seal-sha', contract_sha,
                       {'deadline_epoch': 87400}) == 0
    manifest_path = here / 'runs' / args.phase / args.attempt / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    for field, expected in (
        ('execution_contract_path', contract_path),
        ('dataset_manifest_path', dataset_path),
        ('registry_path', registry_path),
    ):
        assert (manifest_path.parent / manifest[field]).resolve() == expected
    assert manifest['execution_contract_sha256'] == manifest['contract_sha256'] == contract_sha
    assert manifest['conditions'] == []
