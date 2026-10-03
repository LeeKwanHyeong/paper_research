"""Shared target identity and deterministic-reference contracts; no data I/O."""
import hashlib
import json
import math

SIMPLE = ('last_observed_quantity', 'mean_quantity_in_same_observed_window')
SEEDS = [42, 52, 62]
MODELS = ['titantpp_history_mlp', 'rmtpp', 'thp', 'nhp', 'sahp',
          's2p2_matched_head', 'attnhp_matched_head',
          'titantpp_current_only_param_matched', 'titantpp_all_available_history_mlp']
DATASETS = ['yellow_trip_hourly', 'intermittent_frozen_5000', 'insta_market_basket', 'raf_spare_parts']


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def target_metadata(dataset, entity, seq, row, data_sha):
    if row.get('chronological_split') != 'validation':
        raise ValueError('Only validation target metadata is permitted')
    if row.get('demand_dt') is None or not data_sha or len(data_sha) != 64:
        raise ValueError('Missing recorded date or data identity')
    event_key = [entity, seq, str(row['demand_dt'])]
    return {'dataset': dataset, 'entity_id': entity, 'seq': seq, 'split': 'validation',
            'legacy_target_id': digest([dataset, entity, seq, 'validation']),
            'target_id': digest([dataset, data_sha, entity, seq, 'validation', event_key]),
            'recorded_event_key': event_key, 'event_key_scope': 'prepared_dataset_row_not_upstream_raw_lineage',
            'data_file_sha256': data_sha,
            'time_bucket': str(row['time_bucket']) if row.get('time_bucket') is not None else None,
            'site_id': entity.split('::', 1)[0] if dataset == 'intermittent_frozen_5000' else None}


def nonlearned_predictions(quantities, masks):
    if len(quantities) != len(masks):
        raise ValueError('Mismatched batch sizes')
    results = []
    for row, mask in zip(quantities, masks):
        if len(row) != len(mask) or any(type(x) is not bool for x in mask):
            raise ValueError('Invalid history mask')
        valid = [float(q) for q, observed in zip(row, mask) if observed]
        if len(valid) < 2 or not all(math.isfinite(q) and q >= 0 for q in valid):
            raise ValueError('History and target must be finite and present')
        history = valid[:-1]
        results.append({SIMPLE[0]: history[-1], SIMPLE[1]: sum(history) / len(history)})
    return results


def merge_prior_and_controls(registry, prior, outputs, prior_registry):
    """Reuse proven old predictions only after identity/source compatibility checks."""
    if prior['status'] != 'passed' or prior['conditions'] != 84 or prior['heldout_read']:
        raise ValueError('Unverified prior smoke')
    new_rows = {(r['dataset'], r['model'], r['seed']): r for r in registry['rows']}
    old_rows = {(r['dataset'], r['model'], r['seed']): r for r in prior_registry['rows']}
    assert len(new_rows) == 108 and len(old_rows) == 84
    for key, row in old_rows.items():
        bound = new_rows[key]
        for field in ('selected_epoch', 'checkpoint_file_sha256', 'state_tensor_sha256', 'contract_canonical_sha256'):
            assert row[field] == bound[field], (key, field)
        old_bundle = prior_registry['bundles'][row['evaluator_source_bundle']]
        new_bundle = registry['bundles'][bound['evaluator_source_bundle']]
        assert old_bundle['source_files'] == new_bundle['source_files']
    control_checks = [c for out in outputs for c in out['checks']]
    assert len(control_checks) == 24 and all(out['status'] == 'passed' for out in outputs)
    controls = [r for out in outputs for r in out['records']]
    metadata = {}
    simples = {}
    truth_fields = ['dataset', 'entity_id', 'seq', 'split', 'raw_quantity', 'recorded_gap', 'site_id', 'time_bucket', 'history_length']
    for row in controls:
        key = row['legacy_target_id']
        if key in metadata:
            assert all(row[field] == metadata[key][field] for field in truth_fields)
            assert row['target_id'] == metadata[key]['target_id']
        else:
            metadata[key] = row
    for out in outputs:
        for row in out['simple_records']:
            key = (row['target_id'], row['model'])
            assert key not in simples or simples[key] == row
            simples[key] = row
    enriched = []
    for row in prior['records']:
        meta = metadata[row['target_id']]
        assert all(row[field] == meta[field] for field in truth_fields)
        binding = new_rows[(row['dataset'], row['model'], row['seed'])]
        for field in ('checkpoint_file_sha256', 'state_tensor_sha256'):
            assert row[field] == binding[field]
        extra = {field: meta[field] for field in ('target_id', 'legacy_target_id', 'recorded_event_key', 'event_key_scope', 'data_file_sha256')}
        enriched.append({**row, **extra, 'inference_evidence': 'prior_20261001_smoke_reused'})
    records = enriched + [{**r, 'inference_evidence': 'new_structural_control_smoke'} for r in controls] + list(simples.values())
    from paired_statistics import align_panel
    counts = {}
    for dataset in DATASETS:
        subset = [r for r in records if r['dataset'] == dataset]
        meta, _ = align_panel(subset, MODELS + list(SIMPLE), SEEDS, SIMPLE)
        counts[dataset] = len(meta)
    assert len(simples) == 2 * sum(counts.values())
    return {'status': 'passed', 'learned_conditions': 108, 'reused_learned_conditions': 84,
            'new_learned_conditions': 24, 'simple_model_count': 2,
            'prediction_rows': len(records), 'learned_prediction_rows': len(enriched) + len(controls),
            'simple_prediction_rows': len(simples), 'targets_by_dataset': counts,
            'all_models_and_seeds_pair_on_identity_truth_gap_and_history': True,
            'checks': control_checks, 'records': records, 'heldout_read': False,
            'full_endpoint_replay': False, 'native_gpu_tested': False,
            'performance_conclusion': 'none; adapter correctness sample only'}


def finish_statistics(records):
    from paired_statistics import paired_bootstrap
    result = {}
    for idx, dataset in enumerate(DATASETS):
        subset = [r for r in records if r['dataset'] == dataset]
        common = {'draws': 10000, 'rng_seed': 20261001 + idx, 'deterministic_models': SIMPLE,
                  'anchor': 'rmtpp' if idx < 2 else 's2p2_matched_head'}
        if dataset == DATASETS[0]:
            analyses = {'primary_168_hour': dict(kind='calendar', block_hours=168),
                        'sensitivity_24_hour': dict(kind='calendar', block_hours=24),
                        'sensitivity_336_hour': dict(kind='calendar', block_hours=336),
                        'sensitivity_whole_cell': dict(kind='cluster')}
        elif dataset == DATASETS[1]:
            analyses = {'primary_site': dict(kind='cluster', cluster_key='site_id'),
                        'sensitivity_series': dict(kind='cluster', cluster_key='entity_id')}
        else:
            analyses = {'primary_entity': dict(kind='cluster', cluster_key='entity_id')}
        result[dataset] = {name: paired_bootstrap(subset, MODELS + list(SIMPLE), SEEDS, **common, **options)
                           for name, options in analyses.items()}
    return {'scope': 'validation_sample_correctness_only', 'results': result,
            'paper_confidence_intervals': False, 'heldout_read': False}
