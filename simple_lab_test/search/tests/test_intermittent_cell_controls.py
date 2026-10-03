"""Stdlib synthetic checks; no model, real checkpoint, data, GPU, or remote access."""
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from paper.scripts import intermittent_cell_controls as controls

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def measurement():
    return json.loads((ROOT / 'paper/contracts/intermittent_mixed_cell_diagnosis_v1.json').read_text())


@pytest.mark.parametrize('kind', ['validation_forward_batches', 'train_forward_batches', 'autograd_calls'])
def test_work_counter_never_increments_past_limit(measurement, kind):
    budget = controls.Budget(measurement['limits'], clock=lambda: 0.)
    maximum = measurement['limits']['max_' + kind]
    budget.consume(kind, maximum)
    with pytest.raises(ValueError, match='budget exceeded'):
        budget.consume(kind)
    assert budget.counts[kind] == maximum


def test_cuda_memory_byte_limit_without_initializing_cuda(measurement):
    used = [measurement['limits']['max_cuda_allocated_bytes']]
    budget = controls.Budget(measurement['limits'], clock=lambda: 0., memory_bytes=lambda: used[0])
    budget.check()
    used[0] += 1
    with pytest.raises(ValueError, match='CUDA allocation limit'):
        budget.check()


@pytest.fixture
def overlay(tmp_path, measurement):
    root = tmp_path / 'overlay'
    root.mkdir()
    files = {}
    for name in sorted(controls.OVERLAY_NAMES):
        path = root / name
        path.write_text('# Synthetic checksum fixture; never executed: ' + name + '\n')
        files[name] = controls.sha_file(path)
    binding = {
        'schema': 'intermittent_cell_execution_binding_v1',
        'measurement_canonical_sha256': controls.MEASUREMENT_SHA,
        'status': 'implemented_cpu_verified_pending_gpu_approval',
        'gpu_execution_authorized': False, 'limits': deepcopy(measurement['limits']),
        'server': '5090', 'overlay_files': files,
    }
    return root, binding


def approval(measurement, binding):
    return {
        'schema': 'intermittent_cell_execution_approval_v1', 'status': 'approved',
        'binding_canonical_sha256': controls.sha_json(binding),
        'measurement_canonical_sha256': controls.MEASUREMENT_SHA,
        'server': '5090', 'limits': deepcopy(measurement['limits']),
        'checkpoint_model_replay_authorized': True, 'optimizer_updates': 0,
        'held_out_access_allowed': False, 'user_message': 'Synthetic fixture approval only.',
        'recorded_at': '2026-09-13T00:00:00+09:00',
    }


def test_binding_verifies_six_exact_overlay_files_and_separate_execution_approval(measurement, overlay):
    path, binding = overlay
    result = controls.verify_binding(measurement, binding, path)
    assert result['execution_approved'] is False and result['overlay_files_verified'] == 6
    assert result['measurement_canonical_sha256'] == controls.MEASUREMENT_SHA
    result = controls.verify_binding(measurement, binding, path, approval=approval(measurement, binding), execute=True)
    assert result['execution_approved'] is True
    assert result['binding_canonical_sha256'] == controls.sha_json(binding)


@pytest.mark.parametrize('receipt', [None, {}, {'status': 'approved'}])
def test_unapproved_execution_fails_closed(measurement, overlay, receipt):
    path, binding = overlay
    with pytest.raises(ValueError, match='approval|Unapproved'):
        controls.verify_binding(measurement, binding, path, approval=receipt, execute=True)


@pytest.mark.parametrize('change', ['measurement', 'schema', 'measurement_binding', 'readiness', 'embedded_approval',
                                  'limits', 'server', 'missing_file', 'extra_file', 'file_hash', 'file_bytes'])
def test_tampered_measurement_binding_or_overlay_rejected(measurement, overlay, change):
    path, binding = overlay
    if change == 'measurement': measurement['limits']['optimizer_updates'] = 1
    elif change == 'schema': binding['schema'] = 'other'
    elif change == 'measurement_binding': binding['measurement_canonical_sha256'] = '0' * 64
    elif change == 'readiness': binding['status'] = 'ready'
    elif change == 'embedded_approval': binding['gpu_execution_authorized'] = True
    elif change == 'limits': binding['limits']['max_wall_seconds'] += 1
    elif change == 'server': binding['server'] = '5080'
    elif change == 'missing_file': binding['overlay_files'].pop(next(iter(binding['overlay_files'])))
    elif change == 'extra_file': binding['overlay_files']['unexpected.py'] = '0' * 64
    elif change == 'file_hash': binding['overlay_files'][next(iter(binding['overlay_files']))] = '0' * 64
    else: (path / next(iter(binding['overlay_files']))).write_text('# Changed after freeze\n')
    with pytest.raises(ValueError):
        controls.verify_binding(measurement, binding, path)


def test_overlay_symlink_cannot_escape_even_if_target_bytes_match(measurement, overlay, tmp_path):
    path, binding = overlay
    name = next(iter(binding['overlay_files']))
    file = path / name
    outside = tmp_path / 'outside.py'
    outside.write_bytes(file.read_bytes())
    file.unlink()
    file.symlink_to(outside)
    assert controls.sha_file(file) == binding['overlay_files'][name]
    with pytest.raises(ValueError, match='Overlay hash'):
        controls.verify_binding(measurement, binding, path)


@pytest.mark.parametrize('key,value', [
    ('schema', 'training_approval'), ('status', 'pending'), ('binding_canonical_sha256', '0' * 64),
    ('measurement_canonical_sha256', '0' * 64), ('server', '5080'),
    ('checkpoint_model_replay_authorized', False), ('optimizer_updates', 1),
    ('held_out_access_allowed', True), ('user_message', ''), ('recorded_at', ' '),
])
def test_approval_scope_and_provenance_are_exact(measurement, overlay, key, value):
    path, binding = overlay
    receipt = approval(measurement, binding)
    receipt[key] = value
    with pytest.raises(ValueError):
        controls.verify_binding(measurement, binding, path, approval=receipt, execute=True)


def test_approval_cannot_extend_time_or_work_limits(measurement, overlay):
    path, binding = overlay
    receipt = approval(measurement, binding)
    receipt['limits']['max_wall_seconds'] += 1
    with pytest.raises(ValueError, match='resource scope'):
        controls.verify_binding(measurement, binding, path, approval=receipt, execute=True)


def synthetic_payload(scope):
    weights = {'synthetic.weight': [1., 2., 3.], 'synthetic.bias': [0.]}
    arm = {'schema_version': 'quantity_comparison_engine_v1', 'epochs': 120, 'seed': 42,
           'condition': {'case': 'B_log_original', 'mixed_objective': {'name': 'B_log_original', 'alpha': 0.}},
           'identity': {'kind': 'synthetic_dictionary_only'}}
    epoch = 77 if scope == 'primary' else 120
    state = {'scope': scope, 'epoch': epoch, 'global_step': epoch * 2,
             'tensor_state_sha256': controls.sha_json(weights),
             'expected_validation': {'metrics': {'raw_quantity_rmse': 1.25}},
             'expected_history_metrics': {'epoch': 120, 'global_step': 240, 'raw_quantity_rmse': 1.5,
                                          'quantity_mae': .5, 'validation_count': 13}}
    payload = {'schema_version': 'quantity_comparison_engine_v1', 'contract_sha256': controls.sha_json(arm),
               'global_step': state['global_step'], 'model_state_dict': weights}
    if scope == 'primary':
        payload.update(condition=deepcopy(arm['condition']), selector='raw_quantity_rmse', applicable=True,
                       best_epoch=epoch, best_value=1.25, state_sha256=controls.sha_json(weights))
    else:
        history = [{'epoch': i, 'global_step': i * 2, 'raw_quantity_rmse': 1.5, 'quantity_mae': .5,
                    'validation_count': 13} for i in range(1, 121)]
        payload.update(contract=deepcopy(arm), epoch=120, history=history, model_state_sha256=controls.sha_json(weights))
    return payload, state, arm


@pytest.mark.parametrize('scope', ['primary', 'last120'])
def test_primary_and_final_synthetic_payloads_return_exact_verified_state(scope):
    payload, state, arm = synthetic_payload(scope)
    before = deepcopy(payload)
    actual = controls.verify_payload(payload, state, arm, controls.sha_json)
    assert actual is payload['model_state_dict'] and payload == before


@pytest.mark.parametrize('scope', ['primary', 'last120'])
@pytest.mark.parametrize('change', ['schema', 'contract_hash', 'global_step', 'tensor_content', 'tensor_hash', 'frozen_tensor_hash'])
def test_payload_schema_contract_step_and_tensor_identity_reject_tampering(scope, change):
    payload, state, arm = synthetic_payload(scope)
    if change == 'schema': payload['schema_version'] = 'other'
    elif change == 'contract_hash': payload['contract_sha256'] = '0' * 64
    elif change == 'global_step': payload['global_step'] += 1
    elif change == 'tensor_content': payload['model_state_dict']['synthetic.weight'][0] += 1
    elif change == 'tensor_hash': payload['state_sha256' if scope == 'primary' else 'model_state_sha256'] = '0' * 64
    else: state['tensor_state_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        controls.verify_payload(payload, state, arm, controls.sha_json)


@pytest.mark.parametrize('key,value', [('condition', {}), ('selector', 'legacy_time_loss'), ('applicable', False),
                                      ('best_epoch', 78), ('best_value', 1.250001)])
def test_primary_selector_is_frozen_and_cannot_be_reselected(key, value):
    payload, state, arm = synthetic_payload('primary')
    payload[key] = value
    with pytest.raises(ValueError, match='selector/condition'):
        controls.verify_payload(payload, state, arm, controls.sha_json)


@pytest.mark.parametrize('change', ['contract', 'epoch', 'missing_history', 'history_order', 'last_metric', 'last_count'])
def test_final_payload_requires_complete_ordered_history_and_exact_frozen_last_row(change):
    payload, state, arm = synthetic_payload('last120')
    if change == 'contract': payload['contract']['seed'] += 1
    elif change == 'epoch': payload['epoch'] = 119
    elif change == 'missing_history': payload['history'].pop(0)
    elif change == 'history_order': payload['history'][0], payload['history'][1] = payload['history'][1], payload['history'][0]
    elif change == 'last_metric': payload['history'][-1]['quantity_mae'] = .6
    else: payload['history'][-1]['validation_count'] = 12
    with pytest.raises(ValueError):
        controls.verify_payload(payload, state, arm, controls.sha_json)


def test_unknown_payload_scope_is_rejected_instead_of_falling_through_to_final():
    payload, state, arm = synthetic_payload('last120')
    state['scope'] = 'legacy_time_selector'
    with pytest.raises(ValueError, match='scope'):
        controls.verify_payload(payload, state, arm, controls.sha_json)


def test_finite_replay_uses_frozen_absolute_or_relative_tolerance(measurement):
    policy = measurement['validation_gates']
    expected = {'zero': 0., 'large': 1000., 'negative_legacy_time_loss': -3.}
    actual = {'zero': 1e-5, 'large': 1000.0099, 'negative_legacy_time_loss': -3.00002}
    result = controls.check_replay(actual, expected, policy)
    assert result == {'passed': True, 'metric_count': 3}
    actual['zero'] = math.nextafter(1e-5, math.inf)
    with pytest.raises(ValueError, match='replay mismatch'):
        controls.check_replay(actual, expected, policy)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), 1.001])
def test_nonfinite_or_out_of_tolerance_replay_rejected(measurement, bad):
    with pytest.raises(ValueError, match='replay mismatch'):
        controls.check_replay({'metric': bad}, {'metric': 1.}, measurement['validation_gates'])


def test_replay_missing_metric_or_nonfinite_expected_value_rejected(measurement):
    policy = measurement['validation_gates']
    with pytest.raises(ValueError): controls.check_replay({}, {'metric': 1.}, policy)
    with pytest.raises(ValueError): controls.check_replay({'metric': 1.}, {'metric': float('nan')}, policy)


def test_evidence_root_and_json_files_are_exclusive_and_preserve_existing_bytes(tmp_path):
    root = tmp_path / 'evidence'
    writer = controls.EvidenceWriter(root, 4096)
    writer.write('summary.json', {'n': 13})
    original = (root / 'summary.json').read_bytes()
    used = writer.used
    with pytest.raises(FileExistsError): writer.write('summary.json', {'n': 1})
    assert writer.used == used and (root / 'summary.json').read_bytes() == original
    with pytest.raises(FileExistsError): controls.EvidenceWriter(root, 4096)


def test_evidence_budget_reserves_terminal_space_and_caps_every_byte(tmp_path):
    writer = controls.EvidenceWriter(tmp_path / 'evidence', 128, reserve=16)
    assert writer.reserve == 16
    with pytest.raises(ValueError, match='byte budget'):
        writer.write('too_big.json', 'x' * 110)
    assert writer.used == 0 and not (writer.root / 'too_big.json').exists()
    writer.write('data.json', 'x' * 109)
    assert writer.used == 112
    with pytest.raises(ValueError, match='byte budget'): writer.append('batches.jsonl', {})
    assert writer.used == 112 and not (writer.root / 'batches.jsonl').exists()
    writer.write('terminal_status.json', 'T' * 13, terminal=True)
    assert writer.used == 128 == sum(p.stat().st_size for p in writer.root.iterdir())
    with pytest.raises(ValueError, match='byte budget'): writer.write('overflow.json', {}, terminal=True)
    assert not (writer.root / 'overflow.json').exists()


def test_evidence_jsonl_counts_cumulative_bytes_and_preserves_all_lines(tmp_path):
    writer = controls.EvidenceWriter(tmp_path / 'evidence', 4096)
    values = [{'batch': 0, 'count': 128}, {'batch': 1, 'count': 13}]
    for value in values: writer.append('batches.jsonl', value)
    assert [json.loads(line) for line in (writer.root / 'batches.jsonl').read_text().splitlines()] == values
    assert writer.used == (writer.root / 'batches.jsonl').stat().st_size


@pytest.mark.parametrize('name', ['../outside.json', '/tmp/outside.json', '.hidden.json', 'subdirectory/file.json'])
def test_evidence_path_escape_rejected(tmp_path, name):
    writer = controls.EvidenceWriter(tmp_path / 'evidence', 4096)
    with pytest.raises(ValueError, match='Unsafe'): writer.write(name, {'x': 1})
    with pytest.raises(ValueError, match='Unsafe'): writer.append(name, {'x': 1})
    assert writer.used == 0 and not list(writer.root.iterdir())


def test_evidence_stream_symlink_and_nonfinite_output_rejected(tmp_path):
    writer = controls.EvidenceWriter(tmp_path / 'evidence', 4096)
    target = tmp_path / 'outside.jsonl'; target.write_text('original\n')
    (writer.root / 'stream.jsonl').symlink_to(target)
    with pytest.raises(ValueError, match='symlink'): writer.append('stream.jsonl', {'x': 1})
    assert target.read_text() == 'original\n' and writer.used == 0
    with pytest.raises(ValueError): writer.write('nan.json', {'x': float('nan')})
    with pytest.raises(ValueError): writer.append('nan.jsonl', {'x': float('inf')})
    assert writer.used == 0 and not (writer.root / 'nan.json').exists() and not (writer.root / 'nan.jsonl').exists()


def test_controls_import_is_stdlib_only_in_isolated_process():
    path = str(Path(controls.__file__).resolve())
    script = ('import importlib.util,sys; '
              'spec=importlib.util.spec_from_file_location("isolated_controls",sys.argv[1]); '
              'module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); '
              'assert "numpy" not in sys.modules and "torch" not in sys.modules and "polars" not in sys.modules')
    result = subprocess.run([sys.executable, '-I', '-c', script, path], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
