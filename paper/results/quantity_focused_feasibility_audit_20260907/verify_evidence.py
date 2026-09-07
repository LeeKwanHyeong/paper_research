"""Recheck the frozen quantity audit using stdlib; never runs training or inference.

Run from any directory: python3 /absolute/path/to/verify_evidence.py
Remote snapshots are historical evidence; this script does not refresh remote state.
"""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)


def main():
    contract_path = ROOT / 'paper/contracts/raw_rmse_baseline_alignment_seed42_v1.json'
    b_path = ROOT / 'paper/results/hard_lmm_raw_rmse_checkpoint_alignment_seed42_20260906/a_vs_b_metrics.json'
    contract, b_report = read(contract_path), read(b_path)
    datasets = {row['dataset']: row for row in contract['datasets']}
    records = read(HERE / 'remote_evidence.json')['records']
    by_path = {row['path']: row for row in records}
    receipts = [row for row in records if row['path'].endswith('/audit_receipt.json')]
    require(len(receipts) == 8, 'Expected six e1 and two full-fit receipts')
    b_checks = []
    for row in b_report['datasets']:
        dataset = row['dataset']
        evidence = row['evidence']['B_t0_raw_rmse']
        summary_path = Path(evidence['summary_path'])
        summary = read(summary_path)
        history_path = summary_path.parent / 'history.json'
        history = read(history_path)['history']
        require(sha(summary_path) == evidence['summary_sha256'], 'B summary hash drift')
        require(sha(history_path) == evidence['history_sha256'], 'B history hash drift')
        require(sha(Path(evidence['checkpoint_path'])) == evidence['checkpoint_file_sha256'], 'B checkpoint hash drift')
        require(summary['status'] == 'success', 'B not complete')
        require(summary['checkpoint_monitor'] == 'validation_raw_quantity_rmse', 'B selector drift')
        require(all(math.isfinite(x['val_qty_rmse']) for x in history), 'B nonfinite selector')
        selected = min(history, key=lambda x: (x['val_qty_rmse'], x['epoch']))
        require(selected['epoch'] == evidence['best_epoch'] == summary['best_epoch'], 'B selection drift')
        require(len(history) == evidence['completed_epochs'] == summary['completed_epochs'], 'B history length drift')
        require(all(x['train_event_count'] == datasets[dataset]['expected_train_targets'] for x in history), 'B population drift')
        metrics = row['metrics']['B_t0_raw_rmse']
        require(close(summary['best_val_qty_rmse'], metrics['raw_rmse']), 'B RMSE drift')
        require(close(summary['best_val_qty_mae'], metrics['overall_mae']), 'B MAE drift')
        require(metrics['validation_target_count'] == datasets[dataset]['expected_validation_targets'], 'B count drift')
        require(metrics['raw_rmse'] == datasets[dataset]['B_t0_raw_rmse_validation_metrics']['raw_rmse'], 'B contract drift')
        b_checks.append({'dataset': dataset, 'best_epoch': selected['epoch'], 'completed_epochs': len(history), 'metrics': metrics, 'checks_passed': True})

    receipt_by_job = {}
    for entry in receipts:
        receipt = entry['data']
        parent = entry['path'].rsplit('/', 1)[0]
        status = by_path[parent + '/job_status.json']['data']
        identity = by_path[parent + '/job_identity.json']['data']
        require(status['status'] == 'complete' and receipt['status'] == 'passed', 'Incomplete job')
        require(status['receipt_sha256'] == entry['sha256'], 'Receipt binding drift')
        require(receipt['contract_sha256'] == identity['contract_sha256'] == sha(contract_path), 'Contract hash drift')
        require(receipt['source_revision'] == identity['source_revision'] == '9440609673cfdf7bcfec5c77d7f677affd04ae49', 'Source drift')
        require(receipt['checkpoint_monitor'] == 'validation_raw_quantity_rmse', 'Selector drift')
        require(receipt['held_out_test_evaluated'] is False, 'Test unlocked')
        ds = datasets[receipt['dataset']]
        require(identity['dataset_input']['data_sha256'] == ds['data_sha256'], 'Data hash binding drift')
        require(identity['dataset_input']['split_manifest_sha256'] == ds['split_manifest_sha256'], 'Split binding drift')
        require(receipt['validation_target_count'] == ds['expected_validation_targets'], 'Population drift')
        receipt_by_job[receipt['job_id']] = receipt

    comparisons = []
    for check in read(HERE / 'remote_independent_checks.json')['checks']:
        receipt = receipt_by_job[check['job_id']]
        require(all(check['hashes_match_receipt'].values()), 'Remote artifact hash drift')
        require(check['selection_match'] and check['all_selector_values_finite'] and check['full_train_count_every_epoch'], 'Remote selector or count drift')
        require(close(check['selected_raw_rmse_recomputed'], receipt['metrics']['raw_rmse']), 'Recomputed raw RMSE drift')
        ds = datasets[receipt['dataset']]
        population = check['validation_target_population']
        require(population['target_identity_sha256'] == ds['expected_validation_target_identity_sha256'], 'Target identity drift')
        require(population['target_quantity_sha256'] == ds['expected_validation_target_quantity_sha256'], 'Target quantity drift')
        require(set(check['split_rows']) == {'train', 'validation'}, 'Unexpected split')
        rows = check['quantity_rows']
        count = sum(row['count'] for row in rows)
        mae = sum(row['count'] * row['qty_mae'] for row in rows) / count
        rmse = math.sqrt(sum(row['count'] * row['qty_rmse'] ** 2 for row in rows) / count)
        require(count == ds['expected_validation_targets'], 'Strata count drift')
        require(close(mae, receipt['metrics']['overall_mae']) and close(rmse, receipt['metrics']['raw_rmse']), 'Strata recomputation drift')
        # Aggregate replay is a verification; decisions retain the frozen receipt values.
        mae, rmse = receipt['metrics']['overall_mae'], receipt['metrics']['raw_rmse']
        b = ds['B_t0_raw_rmse_validation_metrics']
        comparisons.append({'baseline': receipt['backbone'], 'baseline_mae': mae, 'baseline_rmse': rmse,
                            'B_mae': b['overall_mae'], 'B_rmse': b['raw_rmse'],
                            'B_mae_relative_change_percent': 100 * (b['overall_mae'] / mae - 1),
                            'B_rmse_relative_change_percent': 100 * (b['raw_rmse'] / rmse - 1),
                            'B_wins_both': b['overall_mae'] < mae and b['raw_rmse'] < rmse})
    require(len(comparisons) == 2, 'Expected two full comparisons')
    controller = next(x['data'] for x in records if x['path'].endswith('/controller_5080_status.json'))
    require(controller['status'] == 'stopped_gate_failed', 'Controller state changed')
    require(not any(x['B_wins_both'] for x in comparisons), 'Expected gate outcome changed')
    decision = {
        'audit_status': 'passed_with_scope_limitations',
        'evidence_scope': 'local B artifacts plus historical 5080-retained 5080/5090 receipts; no dataset or held-out inference',
        'selector_replay_tolerance': {'relative': 1e-10, 'absolute': 1e-8},
        'floating_point_note': 'Remote strict summary equality is false; history RMSE differs by about 1e-14. Stratum-weighted MAE/RMSE replay within the existing audit tolerance. Rank decisions use full-precision receipt metrics.',
        'B_checks': b_checks,
        'instacart_seed42_comparisons': comparisons,
        'completed': {'B_seed42_full_fits': 3, 'baseline_e1': 6, 'baseline_seed42_full_fits': 2},
        'not_completed_in_frozen_campaign': {'baseline_seed42_full_fits': 4, 'aligned_seeds_52_62_full_fits': 18, 'final_held_out_checkpoint_evaluations': 27},
        'remaining_counts_are_inventory_not_execution_authorization': True,
        'quantity_focused_scope_can_be_defined': True,
        'all_dataset_quantity_superiority_supported': False,
        'final_paper_evidence_complete': False,
        'K2_local_implementation_gate': 'deferred_quantity_superiority_or_final_paper_evidence_not_established',
        'training_executed_this_audit': False,
        'held_out_test_evaluated': False,
        'sources': {str(p.relative_to(ROOT)): sha(p) for p in [contract_path, b_path, HERE/'remote_evidence.json', HERE/'remote_independent_checks.json']},
    }
    (HERE / 'decision.json').write_text(json.dumps(decision, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'audit_status': decision['audit_status'], 'comparisons': comparisons, 'K2': decision['K2_local_implementation_gate']}, indent=2))


if __name__ == '__main__':
    main()
