"""Synthetic tests of deferred readiness; no training or evaluation occurs."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('width8_12_followup', Path(__file__).with_name('followup.py'))
follow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(follow)


def fixture(root, keys):
    for dataset, model, seed in keys:
        folder = root / f'{dataset}__{seed}__{model}'
        folder.mkdir()
        files = {}
        for name in follow.REQUIRED:
            path = folder / name
            path.write_text('{}')
            files[name] = follow.prepare.sha(path)
        terminal = {'job': {'dataset': dataset, 'arm': model, 'seed': seed},
                    'status': 'complete', 'scientific_success': True,
                    'contract_sha256': 'bound_contract', 'files': files}
        (folder / 'terminal_manifest.json').write_text(json.dumps(terminal))


def test_five_complete_never_admits_six_condition_test(tmp_path):
    fixture(tmp_path, sorted(follow.EXPECTED)[:5])
    result = follow.readiness([tmp_path], {'bound_contract'})
    assert result['status'] == 'pending' and result['verified_terminal_conditions'] == 5
    assert result['evaluation_started'] is False and result['remote_calls'] == 0


def test_all_six_complete_ready_and_changed_binary_denied(tmp_path):
    fixture(tmp_path, sorted(follow.EXPECTED))
    result = follow.readiness([tmp_path], {'bound_contract'})
    assert result['status'] == 'ready' and result['verified_terminal_conditions'] == 6
    next(tmp_path.rglob('best_val_qty_rmse_model.pt')).write_text('changed')
    with pytest.raises(ValueError, match='SHA mismatch'):
        follow.readiness([tmp_path], {'bound_contract'})


def test_unbound_parent_or_derived_contract_denied(tmp_path):
    fixture(tmp_path, sorted(follow.EXPECTED)[:1])
    with pytest.raises(ValueError, match='Unbound'):
        follow.readiness([tmp_path], {'other_contract'})
