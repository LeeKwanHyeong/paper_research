"""Synthetic safety checks; no production data, checkpoint, GPU, or network."""
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import polars as pl
import pytest


def load(name):
    path = Path(__file__).with_name(name + '.py')
    spec = importlib.util.spec_from_file_location('a100_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load('verify')
prepare = load('prepare')


def fixture(tmp_path):
    rows = []
    for i, (q, pred) in enumerate([(1., 2.), (2., 2.), (3., 1.)]):
        row = {'target_index': i, 'entity_id': 'item', 'seq': i + 1, 'split': 'test',
            'recorded_date': '2026-01-01', 'time_bucket': None, 'site_id': None,
            'raw_quantity': q, 'recorded_gap': 1., 'history_length': 1,
            'last_quantity': 1., 'mean_quantity': 1., 'predicted_raw_quantity': pred, 'time_nll': .1 * (i + 1)}
        row['target_id'] = hashlib.sha256(verify.compact(['synthetic', 'dataSHA',
            row['entity_id'], row['seq'], row['split'], row['recorded_date']])).hexdigest()
        rows.append(row)
    ids = hashlib.sha256(''.join(r['target_id'] + '\n' for r in rows).encode()).hexdigest()
    truth = hashlib.sha256(b''.join(verify.compact([r[k] for k in verify.TRUTH_FIELDS]) + b'\n' for r in rows)).hexdigest()
    path = tmp_path / 'part.parquet'
    pl.DataFrame(rows).write_parquet(path)
    receipt = {'status': 'complete', 'full_population': True, 'retrained': False, 'max_batches': None,
        'dataset': 'synthetic', 'split': 'test', 'heldout_read': True, 'data_file_sha256': 'dataSHA',
        'truth_digest_fields': list(verify.TRUTH_FIELDS), 'qualification': {'finite': True},
        'prediction_rows': 3, 'expected_target_count': 3, 'target_identity_sha256': ids,
        'truth_sha256': truth, 'loader': {'lookback_weeks': 10},
        'parts': [{'path': path.name, 'rows': 3, 'sha256': verify.sha(path),
                   'target_identity_sha256': ids, 'truth_sha256': truth}]}
    (tmp_path / 'receipt.json').write_text(json.dumps(receipt))
    return receipt


def test_streaming_metrics_and_strict_tail(tmp_path):
    fixture(tmp_path)
    _, result = verify.inspect_predictions(tmp_path, 2.)
    assert result['overall']['qty_mae'] == pytest.approx(1.)
    assert result['overall']['qty_rmse'] == pytest.approx(math.sqrt(5 / 3))
    assert result['overall']['time_nll'] == pytest.approx(.2)
    assert result['tail']['count'] == 1
    assert result['tail']['qty_rmse'] == 2.


@pytest.mark.parametrize('field,value', [('status', 'failed'), ('full_population', False),
    ('retrained', True), ('max_batches', 1), ('heldout_read', False),
    ('prediction_rows', 4), ('truth_sha256', 'wrong'), ('target_identity_sha256', 'wrong')])
def test_rejects_incomplete_or_tampered_receipt(tmp_path, field, value):
    receipt = fixture(tmp_path)
    receipt[field] = value
    (tmp_path / 'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        verify.inspect_predictions(tmp_path, 2.)


def test_rejects_changed_prediction_file(tmp_path):
    fixture(tmp_path)
    with (tmp_path / 'part.parquet').open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        verify.inspect_predictions(tmp_path, 2.)


@pytest.mark.parametrize('field', ['dataset', 'split', 'data_file_sha256', 'expected_target_count',
    'prediction_rows', 'target_identity_sha256', 'truth_sha256', 'loader', 'truth_digest_fields'])
def test_no_population_intersection_or_history_relaxation(tmp_path, field):
    baseline = fixture(tmp_path)
    candidate = copy.deepcopy(baseline)
    candidate[field] = 'changed'
    with pytest.raises(ValueError, match='identity differs'):
        verify.compare_identity(candidate, baseline)


def test_selection_preserves_first_tie_and_resume_original_epochs():
    history = [{'epoch': 1, 'val_qty_rmse': 2.}, {'epoch': 2, 'val_qty_rmse': 3.},
               {'epoch': 3, 'val_qty_rmse': 2.}]
    assert prepare.select_first_minimum(history)['epoch'] == 1
    with pytest.raises(ValueError, match='chronological'):
        prepare.select_first_minimum(history[2:])


def test_validation_gate_checks_time_as_well_as_quantity():
    metrics = {'count': 3, 'qty_mae': 1., 'qty_rmse': 2., 'time_nll': .2}
    assert all(verify.validation_gate(metrics, metrics, 1e-5, 1e-5).values())
    with pytest.raises(ValueError, match='replay mismatch'):
        verify.validation_gate({**metrics, 'time_nll': .3}, metrics, 1e-5, 1e-5)


def test_empty_tail_is_reported_not_dropped(tmp_path):
    fixture(tmp_path)
    _, result = verify.inspect_predictions(tmp_path, 99.)
    assert result['tail']['count'] == 0
    assert all(result['tail'][key] is None for key in verify.METRICS)
