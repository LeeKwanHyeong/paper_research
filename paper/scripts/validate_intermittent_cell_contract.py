"""Offline validator for a measurement contract, never a model execution entrypoint."""
import argparse
import hashlib
import json
import math
from pathlib import Path

CASES = ('B_log_original', 'mixed_original', 'mixed_matched_original')
DATASET = 'intermittent_frozen_5000'
PRIMARY_EPOCHS = dict(zip(CASES, (77, 32, 56)))
EXPECTED_GATES = {
    'replay_absolute_tolerance': 1e-5, 'replay_relative_tolerance': 1e-5,
    'paired_delta_absolute_tolerance': 0.001, 'paired_delta_relative_tolerance': 0.01,
    'partition_sum_absolute_tolerance': 1e-8, 'partition_sum_relative_tolerance': 1e-10,
    'gradient_reconstruction_absolute_L2_tolerance': 1e-6, 'gradient_reconstruction_relative_L2_tolerance': 1e-4,
    'gradient_gate_rule': 'Pass if absolute L2 residual <=1e-6 OR relative residual <=1e-4; zero reference requires absolute bound.',
    'row_id_counts_state_source_runtime_hashes': 'Exact match required.',
    'nonfinite_values': 'Stop on any NaN/Inf in predictions, losses, gradients or nonempty aggregates.',
    'failure_action': 'Stop the owned diagnostic process, preserve partial evidence and mark affected comparisons unverified; no adaptive tolerances or automatic rerun.'}
EXPECTED_CLIPPING = {
    'norm_type': 2, 'max_norm': 1.0, 'epsilon': 1e-6,
    'coefficient': 'k_b = min(1, 1/(norm(g_joint_all_batch)+1e-6))',
    'application': 'Multiply every task/cell contribution by the SAME full-batch k_b; never clip cells independently.',
    'postclip_sample_mean': 'sum_b(n_b/N_train_probe * k_b * g_cell_b), not clip(sum_b n_b/N * g_cell_b).',
    'meaning': 'Checkpoint probe gradient entering an optimizer hypothetically; no AdamW step or effective learning-rate claim.',
    'parameter_update_count': 0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_contract(c, *, verify_local_evidence=False):
    require(c['schema'] == 'intermittent_mixed_cell_measurement_v1', 'Wrong measurement schema')
    require(c['status'] == 'measurement_contract_frozen_execution_pending_implementation_and_approval', 'This contract does not authorize execution')
    for flag in ('gpu_execution_authorized', 'checkpoint_model_replay_authorized', 'inherited_training_approval_applies'):
        require(c['authorization'][flag] is False, f'Approval boundary changed: {flag}')
    require(c['readiness']['gpu_execution_ready'] is False and c['readiness']['execution_overlay_sha256'] is None, 'Execution binding must be a separate reviewed artifact')
    require(c['server']['ssh_alias'] == '5090' and c['server']['device'] == 'cuda:0' and c['server']['threads'] == 4, 'Server/runtime scope changed')
    require(c['source']['files_sha256'] == canonical(c['source']['files']) == '3ac65c42b35a329ede32ad95862c594579dfca9c3ae4ebf6458560cec1d0cbd8', 'Frozen source changed')
    require(c['source']['frozen_source_modification_allowed'] is False, 'Frozen source must be immutable')
    data = c['dataset']
    require(data['dataset_id'] == DATASET and data['model']['time_intercept_limit'] == 300, 'Dataset/time-head scope changed')
    require(c['training_identity']['dataset_id'] == DATASET and c['training_identity']['data'] == data['inherited_data_identity'], 'Training data identity changed')
    require(c['training_identity']['source']['files'] == c['source']['files'], 'Model/source identity mismatch')
    require(c['server']['runtime_expected'] == c['training_identity']['runtime'], 'Historical runtime binding changed')
    cells = c['cells']
    require(cells['quantity_boundaries'] == data['quantity_boundaries_all_train_rows'] == [2, 31, 46, 187], 'Quantity boundaries changed')
    require(cells['history_boundaries'] == data['history_boundaries'] == [64, 128], 'History boundaries changed')
    require(cells['boundary_equality'] == 'lower_bin' and cells['cross_cell_count'] == 15, 'Cell partition changed')
    require(cells['history_range_inclusive'] == [0, 255], 'History target exclusion changed')
    require(c['comparisons']['time_selector_included'] is False and c['comparisons']['reselection_allowed'] is False, 'Selector scope expanded')
    states = c['states']
    require(len(states) == 6 and {(s['case'], s['scope']) for s in states} == {(case, scope) for case in CASES for scope in ('primary', 'last120')}, 'Exactly six unique states required')
    require(len({s['id'] for s in states}) == 6 and len({s['checkpoint']['path'] for s in states}) == 6, 'Duplicate state/path')
    require(set(c['arms']) == set(CASES), 'Arm set changed')
    for s in states:
        require(s['id'] == f"{s['case']}__{s['scope']}", 'Unsafe state ID')
        epoch = PRIMARY_EPOCHS[s['case']] if s['scope'] == 'primary' else 120
        require(s['epoch'] == epoch and s['global_step'] == epoch * 3077, 'Wrong checkpoint epoch/step')
        require(s['selector'] == ('raw_quantity_rmse' if s['scope'] == 'primary' else 'last'), 'Wrong selector')
        filename = 'best_raw_quantity_rmse_model.pt' if s['scope'] == 'primary' else 'last_epoch_state.pt'
        expected_path = f"/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/mixed_quantity_partition_v1_5090/run/{DATASET}/{s['case']}/{filename}"
        require(s['checkpoint']['path'] == s['checkpoint']['resolved_path'] == expected_path, 'Checkpoint path escaped original arm')
        require(0 < s['checkpoint']['bytes'] <= c['limits']['max_checkpoint_bytes_each'], 'Invalid checkpoint size')
        for digest in (s['checkpoint']['sha256'], s['tensor_state_sha256']):
            require(len(digest) == 64 and all(ch in '0123456789abcdef' for ch in digest), 'Malformed checkpoint digest')
        ref = s['expected_validation']
        require(ref['count'] == 86285 and ref['epoch'] == epoch and ref['global_step'] == s['global_step'] and ref['state_sha256'] == s['tensor_state_sha256'], 'Validation state binding changed')
        require(ref['cell_counts'] == {'body': 81739, 'lowest': 45036, 'tail': 1141}, 'Original validation population changed')
        require(all(math.isfinite(value) for value in ref['metrics'].values()), 'Nonfinite replay target')
        history = s['expected_history_metrics']
        require(history['epoch'] == epoch and history['global_step'] == s['global_step'] and history['validation_count'] == 86285, 'History reference changed')
        for key in ('raw_quantity_rmse', 'quantity_mae', 'legacy_time_loss'):
            require(history[key] == ref['metrics'][key], 'Replay metric sources disagree')
    val, train = c['validation'], c['train_probe']
    require(val['split'] == 'validation' and val['count_per_state'] == 86285 and val['model_mode'] == 'eval' and val['gradient_mode'] == 'no_grad', 'Validation scope changed')
    require(val['batch_size'] == 128 and val['batches_per_state'] == 675 and val['last_batch_size'] == 13 and val['total_state_target_evaluations'] == 517710, 'Validation budget changed')
    require(train['split'] == 'train' and train['selected_sample_count'] == 8192 and train['batch_size'] == 128 and train['batches_per_state'] == 64, 'Train probe budget changed')
    require(train['selection_seed'] == train['loader_generator_seed'] == 1042, 'Calibration sample selection changed')
    require(train['expected_selected_indices_sha256'] == '5367caf16fa7222205fd9e58f97dfca60d3fc0f1def01ae55ef6d27601b0c47e', 'Training indices changed')
    require(train['expected_batch_sha256'] == 'b268b426e292a0f02d02aa46c191ac8a188e2926d3910e73d2beb25fcb7ae067', 'Training input batches changed')
    require(train['model_mode'] == 'train' and train['dropout_seed_base'] == 9042 and train['forward_graphs_per_batch'] == 1, 'Single train-mode graph rule changed')
    require(train['cell_normalization'] == 'sum(loss_vector[cell_mask]) / original_batch_target_count', 'Cell means cannot be summed as contributions')
    require(train['coefficient_refit_allowed'] is False, 'Coefficient refit prohibited')
    for case, arm in c['arms'].items():
        objective = arm['condition']['mixed_objective']
        require(objective['name'] == case and objective['calibration_sha256'] == train['calibration_sha256'], 'Mixed condition/calibration changed')
        require(objective['alpha'] == (0 if case == CASES[0] else train['coefficients']['alpha']), 'Objective alpha changed')
        require(objective['quantity_scale'] == (train['coefficients']['matched_quantity_scale'] if case == CASES[2] else 1), 'Quantity scale changed')
        require(arm['condition']['raw_scale'] == train['coefficients']['raw_scale'] == data['statistics']['raw_scale'], 'Raw unit scale changed')
    require(c['clipping'] == EXPECTED_CLIPPING, 'Clipping policy changed')
    limits = c['limits']
    for key, value in {'max_wall_seconds': 7200, 'max_concurrent_gpu_jobs': 1, 'max_train_selected_samples': 8192, 'max_validation_forward_batches': 4050, 'max_train_forward_batches': 384, 'max_autograd_calls': 18816, 'optimizer_updates': 0}.items():
        require(type(limits[key]) is int and limits[key] == value, f'Execution limit changed: {key}')
    require(limits['stage_seconds'] == {'preflight_and_finalize': 900, 'validation': 1800, 'train_gradient': 4500}, 'Stage budget changed')
    require(sum(limits['stage_seconds'].values()) == limits['max_wall_seconds'], 'Stage budgets do not reconcile')
    require(train['max_autograd_calls_per_batch'] == 49 and train['max_autograd_calls_total'] == limits['max_autograd_calls'] == 49 * 64 * 6, 'Backward budget changed')
    for key in ('held_out_access_allowed', 'automatic_retry_or_resume', 'automatic_budget_extension', 'scheduler_create_or_resume'):
        require(limits[key] is False, f'Execution boundary changed: {key}')
    require(c['validation_gates'] == EXPECTED_GATES, 'Replay/conservation tolerance or failure policy changed')
    if verify_local_evidence:
        evidence = c['evidence']
        for name in ('snapshot', 'remote_inventory'):
            require(file_sha(evidence[name]['path']) == evidence[name]['sha256'], f'Local evidence changed: {name}')
        snapshot = json.loads(Path(evidence['snapshot']['path']).read_text())
        inventory = json.loads(Path(evidence['remote_inventory']['path']).read_text())
        parent_ref = evidence['parent_execution_contract']
        require(file_sha(parent_ref['path']) == parent_ref['file_sha256'], 'Parent file hash changed')
        parent = json.loads(Path(parent_ref['path']).read_text())
        require(canonical(parent) == parent_ref['canonical_sha256'] == 'fe5f421cfc81486140119d2ed7e0eef67502c22b27c2e59bbdc40addd6c00e87', 'Parent execution identity changed')
        expected_data = {k: v for k, v in next(d for d in parent['datasets'] if d['dataset_id'] == DATASET).items() if k != 'validation_diagnosis'}
        require(data == expected_data, 'Dataset/model differs from original parent')
        original_base = snapshot['files'][f'{DATASET}/B_log_original/contract.json']
        require(c['training_identity'] == original_base['identity'] and c['initial_state_sha256'] == original_base['initial_state_sha256'], 'Original training/runtime/initialization identity changed')
        freeze_path = Path(evidence['remote_inventory']['path']).parent / 'freeze_receipt.json'
        freeze = json.loads(freeze_path.read_text())
        require(canonical(c) == freeze['canonical_sha256'], 'Whole measurement contract differs from the external freeze receipt')
        for rel, expected in c['source']['files'].items():
            require(file_sha(Path(c['source']['local_root']) / rel) == inventory['source_files'][rel]['sha256'] == expected, f'Source evidence changed: {rel}')
        for s in states:
            require(snapshot['files'][s['evidence_key']][s['scope']] == s['expected_validation'], 'State reference changed versus snapshot')
            key = f"{s['case']}/{Path(s['checkpoint']['path']).name}"
            require(s['checkpoint'] == inventory['checkpoints'][key], 'Transport hash differs from inventory')
        for case, arm in c['arms'].items():
            original = snapshot['files'][arm['contract_snapshot_key']]
            require(original == inventory['metadata'][f'{case}/contract.json']['json'], 'Remote metadata changed versus snapshot')
            require(canonical(original) == arm['contract_canonical_sha256'] and original['condition'] == arm['condition'], 'Arm contract binding changed')
        calibration = snapshot['files'][f'{DATASET}/calibration.json']
        require(train['coefficients'] == {'alpha': calibration['alpha'], 'matched_quantity_scale': calibration['quantity_scale'], 'raw_scale': calibration['statistics']['raw_scale']}, 'Calibration values changed versus source')
    return {'valid_measurement_contract': True, 'states': 6, 'cross_cells': 15, 'local_evidence_verified': verify_local_evidence, 'gpu_execution_ready': False, 'models_or_datasets_loaded': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('contract', type=Path)
    parser.add_argument('--verify-local-evidence', action='store_true')
    args = parser.parse_args()
    c = json.loads(args.contract.read_text())
    result = validate_contract(c, verify_local_evidence=args.verify_local_evidence)
    print(json.dumps(result | {'contract_canonical_sha256': canonical(c)}, indent=2))


if __name__ == '__main__':
    main()
