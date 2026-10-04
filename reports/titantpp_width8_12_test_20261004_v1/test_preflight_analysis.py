"""Synthetic deployment-path and seed-summary checks without scientific inference."""
import importlib.util
import json
from pathlib import Path

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preflight = load('width8_12_preflight', 'preflight.py')
analysis = load('width8_12_analysis', 'analyze.py')


def test_missing_dataset_is_detected_before_any_inference(tmp_path):
    campaign = tmp_path / 'campaign'
    campaign.mkdir()
    checkpoint = tmp_path / 'checkpoint.pt'
    checkpoint.write_text('synthetic bytes only')
    data = tmp_path / 'data.parquet'
    data.write_text('synthetic bytes only')
    split = tmp_path / 'split.json'
    split.write_text('{}')
    files = {'evaluation_registry.json': {'rows': [{'checkpoint_path': 'checkpoint.pt', 'checkpoint_file_sha256': preflight.sha(checkpoint)}], 'bundles': {}},
             'dataset_manifest.json': {'datasets': [{'path': str(data), 'sha256': preflight.sha(data), 'split_manifest': {'path': 'split.json', 'sha256': preflight.sha(split)}}]},
             'selected_binding.json': {}, 'code_seal.json': {'files': {}}}
    for name, value in files.items():
        (campaign / name).write_text(json.dumps(value))
    contract = {'resources': {'root': str(tmp_path)}, **{key: preflight.sha(campaign / name) for name, key in
                [('evaluation_registry.json', 'registry_sha256'), ('dataset_manifest.json', 'dataset_manifest_sha256'), ('selected_binding.json', 'selected_binding_sha256')]}}
    (campaign / 'execution_contract.json').write_text(json.dumps(contract))
    result = preflight.main(tmp_path, campaign)
    assert result['status'] == 'complete' and result['inference_calls'] == result['gpu_calls'] == 0
    data.unlink()
    with pytest.raises(ValueError, match='does not exist'):
        preflight.main(tmp_path, campaign)


def test_sample_standard_deviation_and_incomplete_seed_group():
    rows = [{'dataset': 'synthetic', 'model': 'model', 'view': 'overall', 'seed': seed, 'count': 10,
             'threshold': None, 'qty_rmse': value, 'qty_mae': value, 'time_nll': value}
            for seed, value in zip((42, 52, 62), (1., 2., 3.))]
    result = analysis.aggregate(rows)[0]
    assert result['qty_rmse_mean'] == 2. and result['qty_rmse_sample_sd'] == 1.
    with pytest.raises(ValueError, match='Partial 3seed'):
        analysis.aggregate(rows[:2])
    with pytest.raises(ValueError, match='Duplicate'):
        analysis.aggregate(rows + [rows[0]])
