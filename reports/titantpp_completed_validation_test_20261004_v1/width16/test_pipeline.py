"""Synthetic orchestration tests; no remote, dataset, or CUDA access."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('width16_pipeline', Path(__file__).with_name('pipeline.py'))
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


def setup_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(p, 'HERE', tmp_path)
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    monkeypatch.setattr(p, 'output', lambda command: 'gpu')
    registry = {'rows': [{'dataset': 'synthetic', 'model': 'model', 'seed': seed} for seed in (42, 52)]}
    manifest = {'datasets': []}
    for name, value in [('evaluation_registry.json', registry), ('dataset_manifest.json', manifest), ('code_seal.json', {'files': {}})]:
        (tmp_path / name).write_text(json.dumps(value))
    contract = {'registry_sha256': p.sha(tmp_path / 'evaluation_registry.json'),
        'dataset_manifest_sha256': p.sha(tmp_path / 'dataset_manifest.json'),
        'resources': {'gpu_uuid': 'gpu', 'campaign_timeout_seconds': 60},
        'environment': {}, 'approval': {'datasets': ['synthetic']}, 'reuse_conditions': [],
        'validation_gate': {'new_conditions': 2}, 'reuse_split_count': 0}
    (tmp_path / 'execution_contract.json').write_text(json.dumps(contract))
    return contract


def test_validation_failure_stops_before_any_test(tmp_path, monkeypatch):
    setup_queue(tmp_path, monkeypatch)
    calls = []

    def fail(row, split, contract, deadline):
        calls.append(split)
        raise ValueError('frozen validation mismatch')

    monkeypatch.setattr(p, 'run_condition', fail)
    with pytest.raises(ValueError, match='frozen validation mismatch'):
        p.main()
    assert calls == ['validation']
    assert not (tmp_path / 'validation_gate.json').exists()
    assert p.read(tmp_path / 'pipeline_status.json')['status'] == 'failed'


def test_all_validation_gates_precede_test_and_existing_attempt_is_preserved(tmp_path, monkeypatch):
    setup_queue(tmp_path, monkeypatch)
    calls = []

    def complete(row, split, contract, deadline):
        if split == 'test':
            gate = p.read(tmp_path / 'validation_gate.json')
            assert gate['passed'] and gate['conditions'] == 2
        calls.append((split, row['seed']))
        return {'receipt_sha256': 'synthetic'}

    monkeypatch.setattr(p, 'run_condition', complete)
    p.main()
    assert calls == [('validation', 42), ('validation', 52), ('test', 42), ('test', 52)]
    assert p.read(tmp_path / 'inference_completion.json')['new_split_count'] == 4
    before = (tmp_path / 'pipeline_status.json').read_bytes()
    with pytest.raises(AssertionError, match='Never overwrite'):
        p.main()
    assert (tmp_path / 'pipeline_status.json').read_bytes() == before


def test_changed_registry_stops_before_worker(tmp_path, monkeypatch):
    setup_queue(tmp_path, monkeypatch)
    with (tmp_path / 'evaluation_registry.json').open('a') as stream:
        stream.write(' ')
    monkeypatch.setattr(p, 'run_condition', lambda *args: pytest.fail('Worker must not start'))
    with pytest.raises(AssertionError):
        p.main()
