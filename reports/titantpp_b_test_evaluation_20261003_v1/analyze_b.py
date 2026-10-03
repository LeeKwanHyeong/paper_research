"""Exploratory B versus frozen MLP/external models on the same legacy test targets.

No inference, model selection, population intersection, or changes to frozen
artifacts. Reuse the sealed legacy streaming and paired-bootstrap machinery in
a private module instance, with B as the difference reference only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEGACY = HERE.parent / 'titantpp_legacy_evaluation_20261003_v1'
DATASETS = ['yellow_trip_hourly', 'intermittent_frozen_5000', 'insta_market_basket']
B = 'titantpp'
REPRESENTATIVE = 'titantpp_history_mlp'
EXTERNAL = ['rmtpp', 'thp', 'nhp', 'sahp', 's2p2_matched_head', 'attnhp_matched_head']
MODELS = [B, REPRESENTATIVE] + EXTERNAL
SEEDS = [42, 52, 62]
SCOPE = 'exploratory_b_after_legacy_test_review'
INTERPRETATION = ('Exploratory comparison added after viewing the existing legacy test results; '
                  'conditional on the fixed selected checkpoints; MLP remains the frozen representative.')


def load_legacy(name):
    spec = importlib.util.spec_from_file_location(name, LEGACY / 'analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_legacy('legacy_b_binding_helpers')
paired = load_legacy('legacy_b_paired_helpers')
paired.MODELS = list(MODELS)
paired.INTERPRETATION = INTERPRETATION
sha256_file = legacy.sha256_file


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def resolve_from(base, value):
    return (Path(base) / value).resolve()


def inspect_b_manifest(path):
    """Refuse partial/extra/duplicate conditions before any predictions are read."""
    path = Path(path).resolve()
    manifest = read_json(path)
    expected = {(dataset, B, seed) for dataset in DATASETS for seed in SEEDS}
    entries = manifest.get('conditions', [])
    keys = [(row['dataset'], row['model'], row['seed']) for row in entries]
    require(len(keys) == 9 and set(keys) == expected, 'B manifest must declare exactly nine unique conditions')
    conditions, receipts, missing, failures = {}, {}, [], []
    for key, row in zip(keys, entries):
        output = resolve_from(path.parent, row['output_dir'])
        conditions[key] = {**row, 'output_dir': str(output)}
        receipt_path = output / 'receipt.json'
        failure_path = output / 'failure.json'
        if failure_path.exists():
            failures.append({'condition': list(key), 'failure': read_json(failure_path)})
        if not receipt_path.exists():
            missing.append(list(key))
            continue
        receipt = read_json(receipt_path)
        require(receipt.get('status') == 'complete' and receipt.get('split') == 'test' and
                tuple(receipt.get(field) for field in ['dataset', 'model', 'seed']) == key,
                'B receipt status/condition/test split mismatch')
        receipts[key] = receipt
    pending = None
    if missing or failures:
        pending = {'status': 'blocked_failed_conditions' if failures else 'pending_conditions',
                   'expected_new_b_conditions': 9, 'completed_new_b_conditions': len(receipts),
                   'missing_conditions': missing, 'failed_conditions': failures,
                   'performance_aggregation_started': False, 'results': None,
                   'exploratory': True, 'interpretation': INTERPRETATION}
    return manifest, conditions, receipts, pending


def load_b_bindings(manifest, manifest_path, contract_path, registry_path, dataset_manifest_path):
    """Use the actual B campaign identities; never relabel them as legacy IDs."""
    base = Path(manifest_path).resolve().parent
    artifacts = {'execution_contract_path': Path(contract_path).resolve(),
                 'registry_path': Path(registry_path).resolve(),
                 'dataset_manifest_path': Path(dataset_manifest_path).resolve()}
    for field, path in artifacts.items():
        require(field in manifest and resolve_from(base, manifest[field]) == path,
                'B manifest artifact path differs from CLI: ' + field)
    contract = read_json(contract_path)
    registry = read_json(registry_path)
    data = read_json(dataset_manifest_path)
    contract_sha = sha256_file(contract_path)
    require(manifest.get('execution_contract_sha256') == contract_sha and
            manifest.get('contract_sha256') == contract_sha, 'B contract SHA mismatch')
    for field, path in [('registry_sha256', registry_path), ('dataset_manifest_sha256', dataset_manifest_path)]:
        require(contract.get(field) == manifest.get(field) == sha256_file(path),
                'B frozen artifact SHA mismatch: ' + field)
    approval = contract.get('approval', {})
    require(approval.get('test_inference_authorized') is True and
            approval.get('retraining_authorized') is False and
            set(approval.get('datasets', [])) == set(DATASETS) and
            approval.get('models') == [B] and set(approval.get('seeds', [])) == set(SEEDS) and
            'test' in approval.get('allowed_splits', []), 'B approval scope mismatch')
    rows = {(row['dataset'], row['model'], row['seed']): row for row in registry['rows']}
    expected = {(dataset, B, seed) for dataset in DATASETS for seed in SEEDS}
    require(len(registry['rows']) == 9 and set(rows) == expected, 'B registry coverage mismatch')
    datasets = {row['dataset']: row for row in data['datasets']}
    require(len(datasets) == len(data['datasets']) and set(datasets) == set(DATASETS),
            'B dataset manifest coverage mismatch')
    for dataset, row in datasets.items():
        require(contract['dataset_sha256'].get(dataset) == row['sha256'], 'B source data SHA mismatch')
    seal_path = Path(contract_path).resolve().parent / 'code_seal.json'
    seal = read_json(seal_path)
    require(manifest.get('code_seal_sha256') == sha256_file(seal_path), 'B code seal SHA mismatch')
    project_root = HERE.parents[1]
    runner = Path(contract_path).resolve().parent / 'evaluate.py'
    for path in [runner, Path(__file__).resolve()]:
        require(seal['files'].get(str(path.relative_to(project_root))) == sha256_file(path),
                'B code differs from seal: ' + path.name)
    require(sha256_file(runner) == sha256_file(LEGACY / 'evaluate.py'),
            'B evaluator must be byte-identical to frozen legacy evaluator')
    return {'contract': contract, 'contract_sha256': contract_sha, 'registry': registry,
            'rows': rows, 'datasets': datasets, 'runner_sha256': sha256_file(runner)}


def stream_paired_dataset(conditions, receipts, dataset, expected_population):
    # Each campaign binding is checked by analyze_manifest before this helper.
    # These two provenance keys necessarily differ between campaigns; remove
    # them only from transient views used by the common-population comparator.
    # All data/target/truth/history fields stay intact and are independently read.
    views = [{key: value for key, value in receipt.items()
              if key not in ('registry_sha256', 'dataset_manifest_sha256')}
             for receipt in receipts]
    units, evidence = paired.stream_dataset(conditions, views, dataset, expected_population)
    evidence['separately_bound_campaign_provenance'] = True
    evidence['paired_history_check_scope'] = (
        'identical target identities, recorded quantity/gap, history length, last/mean observed '
        'quantity and frozen loader contracts; not independent reconstruction of all history arrays')
    for row, receipt in zip(evidence['conditions'], receipts):
        row.update(campaign='b' if row['model'] == B else 'legacy',
                   registry_sha256=receipt.get('registry_sha256'),
                   dataset_manifest_sha256=receipt.get('dataset_manifest_sha256'))
    return units, evidence


def bootstrap_b(counts, sums, *, dataset, kind, cluster_key, block_hours=None, draws=10000):
    result = paired.bootstrap_sums(counts, sums, dataset=dataset, kind=kind,
                                   cluster_key=cluster_key, block_hours=block_hours,
                                   draws=draws, scope=SCOPE)
    # The inherited four-anchor correction belongs only to the earlier frozen
    # MLP family. B is an added exploratory comparison, not a new primary family.
    result.update(exploratory=True, difference_reference=B,
                  multiplicity='unadjusted_exploratory_pointwise_95_percentile',
                  representative_model=REPRESENTATIVE, representative_replaced=False)
    for field in ['means', 'sample_sd', 'seed_metrics']:
        result[field] = {model: result[field][model] for model in MODELS}
    result.pop('deterministic_models', None)
    result.pop('deterministic_prediction_vectors_per_model', None)
    if result['intervals'] is not None:
        result['intervals'] = {model: result['intervals'][model] for model in MODELS[1:]}
        for entry in result['intervals'].values():
            entry['qty_rmse'].pop('four_anchor_bonferroni_98_75_percentile', None)
    result['point_differences'] = {
        model: {metric: result['means'][B][metric] - result['means'][model][metric]
                for metric in paired.METRICS} for model in MODELS[1:]}
    return result


def verify_legacy_results(result, prior, dataset, panel_name):
    """Verify fixed comparator values and identical draw weights, not just policy labels."""
    reference = prior['results'][dataset][panel_name]
    for field in ['count', 'units', 'kind', 'cluster_key', 'block_hours', 'rng_seed',
                  'draws_requested', 'draws_completed', 'interval_status']:
        require(result[field] == reference[field], 'Prior bootstrap specification mismatch: ' + field)
    for field in ['resampling_weights_sha256', 'invalid_empty_draws']:
        require(result.get(field) == reference.get(field), 'Prior paired resampling differs: ' + field)
    for model in MODELS[1:]:
        for seed in SEEDS:
            for metric in paired.METRICS:
                actual = result['seed_metrics'][model][str(seed)][metric]
                expected = reference['seed_metrics'][model][str(seed)][metric]
                require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12),
                        'Legacy comparator metric changed: ' + model + '/' + metric)


def analyze_manifest(run_manifest, legacy_run_manifest, contract, registry, dataset_manifest, legacy_analysis):
    manifest, b_conditions, b_receipts, pending = inspect_b_manifest(run_manifest)
    if pending:
        return pending
    binding = load_b_bindings(manifest, run_manifest, contract, registry, dataset_manifest)
    for key, receipt in b_receipts.items():
        legacy.verify_receipt_binding(receipt, key, binding)
    old_manifest, old_conditions, old_receipts, old_pending = legacy.inspect_manifest(legacy_run_manifest)
    require(old_pending is None, 'The original complete 108-condition campaign is required')
    require(all(row['split'] == 'test' for row in old_receipts.values()), 'Legacy comparison must use test only')
    prior = read_json(legacy_analysis)
    require(prior.get('status') == 'complete' and prior.get('split') == 'test' and
            prior.get('learned_conditions') == 108 and
            prior.get('run_manifest_sha256') == sha256_file(legacy_run_manifest) and
            prior.get('analysis_script_sha256') == sha256_file(LEGACY / 'analyze.py'),
            'Original analysis is incomplete or belongs to different frozen inputs')
    results, validation = {}, {}
    for dataset in DATASETS:
        population = binding['datasets'][dataset]['populations']['test']
        old_dataset = old_manifest['_verified_frozen_datasets'][dataset]
        require(old_dataset['sha256'] == binding['datasets'][dataset]['sha256'] and
                old_dataset['populations']['test'] == population,
                'B and legacy frozen test populations differ')
        conditions, receipts = [], []
        for model in MODELS:
            for seed in SEEDS:
                key = (dataset, model, seed)
                conditions.append((b_conditions if model == B else old_conditions)[key])
                receipts.append((b_receipts if model == B else old_receipts)[key])
        # Same loader contract is required even though source/checkpoints differ.
        require(all(row['loader'] == receipts[0]['loader'] for row in receipts),
                'B and legacy frozen loader contracts differ')
        units, evidence = stream_paired_dataset(conditions, receipts, dataset, population)
        previous = prior['validation'][dataset]
        for field in ['target_count', 'data_file_sha256', 'target_identity_sha256', 'truth_sha256',
                      'ordered_entity_seq_recorded_date_sha256', 'independent_column_fingerprints']:
            require(evidence[field] == previous[field], 'Prior population evidence mismatch: ' + field)
        prior_receipts = {(row['model'], row['seed']): row['receipt_sha256']
                          for row in previous['conditions']}
        for row in evidence['conditions']:
            if row['model'] != B:
                require(prior_receipts[(row['model'], row['seed'])] == row['receipt_sha256'],
                        'Previously analyzed comparator receipt changed')
        validation[dataset] = evidence
        results[dataset] = {}
        for name, (kind, key, block) in paired.analysis_specs(dataset).items():
            panel = bootstrap_b(units[key].counts, units[key].sums, dataset=dataset,
                                kind=kind, cluster_key=key, block_hours=block)
            verify_legacy_results(panel, prior, dataset, name)
            results[dataset][name] = panel
    return {'status': 'complete', 'split': 'test', 'scope': SCOPE, 'exploratory': True,
            'interpretation': INTERPRETATION, 'new_b_conditions': 9,
            'legacy_comparison_conditions': 63, 'learned_conditions': 72,
            'representative_model': REPRESENTATIVE, 'representative_replaced': False,
            'prediction_ensemble': False, 'retrained': False,
            'run_manifest_sha256': sha256_file(run_manifest),
            'legacy_run_manifest_sha256': sha256_file(legacy_run_manifest),
            'legacy_analysis_sha256': sha256_file(legacy_analysis),
            'analysis_script_sha256': sha256_file(__file__),
            'legacy_helper_sha256': sha256_file(LEGACY / 'analyze.py'),
            'contract_sha256': sha256_file(contract), 'registry_sha256': sha256_file(registry),
            'dataset_manifest_sha256': sha256_file(dataset_manifest),
            'validation': validation, 'results': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ['run-manifest', 'legacy-run-manifest', 'contract', 'registry',
                  'dataset-manifest', 'legacy-analysis', 'output']:
        parser.add_argument('--' + field, type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('Refusing to overwrite existing output: ' + str(args.output))
    try:
        result = analyze_manifest(args.run_manifest, args.legacy_run_manifest, args.contract,
                                  args.registry, args.dataset_manifest, args.legacy_analysis)
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        result = {'status': 'failed_contract_or_artifact', 'error_type': type(exc).__name__,
                  'error': str(exc), 'results': None, 'exploratory': True,
                  'interpretation': INTERPRETATION}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return 0 if result['status'] == 'complete' else (3 if result['status'] == 'pending_conditions' else 2)


if __name__ == '__main__':
    raise SystemExit(main())
