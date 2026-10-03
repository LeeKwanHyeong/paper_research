"""Metadata/byte-identity inventory only; never deserialize a dataset or checkpoint."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
MODELS = ['titantpp_history_mlp', 'rmtpp', 'thp', 'nhp', 'sahp',
          's2p2_matched_head', 'attnhp_matched_head']
DATASETS = ['yellow_trip_hourly', 'intermittent_frozen_5000',
            'insta_market_basket', 'raf_spare_parts']
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    p = ROOT / path
    SOURCES[str(p.relative_to(ROOT))] = sha(p)
    return json.loads(p.read_text())


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    now = datetime.now(timezone.utc).isoformat()
    core_contract_path = 'search_artifacts/titantpp_core_ablation_20260928_v1/frozen_execution/execution_contract.json'
    core = read(core_contract_path)
    raf_path = 'search_artifacts/titantpp_raf_5080_20260930_v1/execution_contract.json'
    raf = read(raf_path)
    additional_path = 'search_artifacts/titantpp_additional_tpp_20260930_v1/launch_layout_fix_v3/execution_contract.json'
    read(additional_path)
    old_audit_path = 'reports/local_detail_3seed_final_20260927_v1/terminal_audit.json'
    old_audit = read(old_audit_path)
    specs = core['datasets'] + raf['datasets']
    datasets = []
    for spec in specs:
        ds = spec['dataset_id']
        ident = spec.get('parent_data_identity', spec['inherited_data_identity'])
        identities = {}
        for k, v in ident.items():
            if k not in ['data', 'split_manifest']:
                continue
            raw = v['path']
            rel = raw.split('paper_research/', 1)[1]
            actual = sha(ROOT / rel)  # bytes only; no row decoding
            assert actual == v['sha256'], (ds, k)
            identities[k] = {'path': rel, 'sha256': actual, 'operation': 'byte_hash_only'}
        if ds == 'raf_spare_parts':
            for k, v in spec['inherited_data_identity'].items():
                if k in ['data', 'split_manifest']:
                    rel = 'search_artifacts/titantpp_raf_5080_20260930_v1/' + v['path'].removeprefix('../')
                    assert sha(ROOT / rel) == v['sha256']
                    identities['deployed_' + k] = {'path': rel, 'sha256': v['sha256'], 'scope': 'train_validation_only'}
        datasets.append({'dataset': ds, 'identities': identities,
                         'frozen_loader': spec['loader'], 'frozen_model_head': spec['model'],
                         'quantity_boundaries_train_only': spec['quantity_boundaries_all_train_rows'],
                         'train_validation_population': spec['inherited_data_identity']['populations'],
                         'source_contract': raf_path if ds == 'raf_spare_parts' else core_contract_path,
                         'heldout_row_or_performance_access_this_work': False})

    audits = {}
    for host, p in [('5080', 'monitor/20260928T195732Z/terminal_5080'),
                    ('5090', 'final_audit_20260930_v1/terminal_5090')]:
        path = f'search_artifacts/titantpp_core_ablation_20260928_v1/{p}/terminal_audit.json'
        audits[host] = (path, read(path))
    rows = []
    cp_name = 'best_val_qty_rmse_model.pt'

    def add(ds, arm, seed, epoch, source, tensor_sha, audit_path, expected_file_sha,
            contract_sha, closure_sha, closure_label='source_closure_sha256'):
        folder = ROOT / source
        replay_path = folder / 'endpoint_replays.json'
        replay = read(replay_path)
        selected = replay['selected']
        assert selected['evaluation_scope'] == 'validation_only'
        assert selected['held_out_test_evaluated'] is False
        assert selected['state_sha256'] == tensor_sha
        assert int(replay['best_epoch']) == int(epoch)
        summary = read(folder / 'summary.json')
        assert summary['checkpoint_state_sha256'] == tensor_sha
        cp = folder / cp_name
        file_sha = sha(cp) if cp.is_file() else None
        if expected_file_sha is not None:
            assert file_sha == expected_file_sha, cp
        rows.append({'dataset': ds, 'model': arm, 'seed': int(seed),
                     'endpoint': 'selected', 'selected_epoch': int(epoch),
                     'selection': 'strict_first_min_validation_raw_quantity_rmse',
                     'state_tensor_sha256': tensor_sha,
                     'checkpoint_path': str(cp.relative_to(ROOT)), 'checkpoint_file_sha256': file_sha,
                     'local_binary_present': cp.is_file(),
                     'binary_status': 'prior_cpu_audit_reused_and_file_sha_rechecked' if expected_file_sha else 'not_present_at_recorded_path_retrieval_and_cpu_audit_required',
                     'recorded_remote_checkpoint_path': selected['checkpoint_path'],
                     'original_source_revision': summary.get('source_revision'),
                     'contract_canonical_sha256': contract_sha,
                     closure_label: closure_sha,
                     'audit_path': audit_path,
                     'validation_replay_path': str(replay_path.relative_to(ROOT)),
                     'validation_count': selected['count']})

    table_path = ROOT / 'reports/titantpp_core_ablation_execution_20260928_v1/final_conditions.csv'
    SOURCES[str(table_path.relative_to(ROOT))] = sha(table_path)
    for r in csv.DictReader(table_path.open()):
        if r['endpoint'] != 'selected' or r['arm'] not in MODELS[:5]:
            continue
        ds, arm, seed = r['dataset'], r['arm'], int(r['seed'])
        if arm == MODELS[0]:
            ap, audit = audits['5090' if ds == 'insta_market_basket' else '5080']
            check = next(x for x in audit['checkpoint_checks'] if x['job'] == f'{ds}__{seed}__{arm}' and x['endpoint'] == 'selected')
            assert check['state_sha256'] == r['state_sha256']
            add(ds, arm, seed, r['epoch'], r['source'], r['state_sha256'], ap,
                check['file_sha256'], audit['contract_sha256'],
                'e21c45b5f095d258e62998ce818b07bef38ff2cfff6bc408b3594bd1adba9265')
        else:
            receipt = next(v for v in old_audit['source_receipts'].values() if r['source'].startswith(v['root'] + '/'))
            add(ds, arm, seed, r['epoch'], r['source'], r['state_sha256'], old_audit_path,
                None, receipt['contract_sha256'], receipt['source_files_sha256'],
                'legacy_source_manifest_sha256')

    for label, folder in [
        ('additional_5080', 'search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5080_20261001_v1'),
        ('additional_5090', 'search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5090_20261001_v1'),
        ('raf', 'search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2')]:
        ap = folder + '/terminal_audit.json'
        audit = read(ap)
        for c in audit['conditions']:
            job = c['job']
            if job['arm'] not in MODELS:
                continue
            source = f"{folder}/original/run/{job['id']}/runs/{job['arm']}/count_only_log_regression/seed_{job['seed']}"
            add(job['dataset'], job['arm'], job['seed'], c['selected_epoch'], source,
                c['model_tensor_sha256']['selected'], ap, c['checkpoint_files'][cp_name],
                audit['contract_sha256'], audit['source_closure_sha256'])
    rows.sort(key=lambda r: (DATASETS.index(r['dataset']), MODELS.index(r['model']), r['seed']))
    assert len(rows) == 84
    assert len({(r['dataset'], r['model'], r['seed']) for r in rows}) == 84
    save('checkpoint_manifest.json', {'created_utc': now, 'rows': rows,
        'summary': {'selected_conditions': len(rows), 'local_selected_binaries': sum(r['local_binary_present'] for r in rows),
                    'local_missing_at_recorded_path': sum(not r['local_binary_present'] for r in rows),
                    'tensor_identities_bound': len(rows)},
        'scope': 'metadata identity freeze and byte hash; no model loading or forward call in this work'})
    keys = sorted({k for r in rows for k in r})
    with (OUT / 'checkpoint_manifest.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    save('dataset_manifest.json', {'datasets': datasets, 'new_heldout_content_read': False})

    old_path = 'reports/titantpp_followup_contract_5080_20260927_v1/heldout_provenance_audit.json'
    old = read(old_path)
    hist = old['matching_campaign_records']
    linked = Counter(d['name'] for r in hist for d in r['dataset_metadata'])
    raf_hist = []
    for p in [
        'paper/results/hard_lmm_local_time_screening_20260904/source_5080/raf_spare_parts/launch_contract.json',
        'search_artifacts/count_aware_b012_seed42_screening_e300_20260828_recovery1/raf_spare_parts/titan_b012_screening/launch_contract.json']:
        j = read(p)
        allow = ['dataset', 'data_path', 'data_sha256', 'split_manifest_path', 'split_manifest_sha256',
                 'evaluation_scope', 'held_out_test_evaluated', 'source_revision']
        raf_hist.append({'path': p, 'metadata': {k:j[k] for k in allow if k in j}})
    save('split_access_ledger.json', {
        'created_utc': now, 'scope': 'recorded evidence, not exhaustive human access logs',
        'historical_audit': {'path': old_path, 'checked_at_utc': old['checked_at_utc'],
            'manifest_count_checked': old['historical_manifest_count_checked'],
            'test_metric_named_file_count': old['test_metric_named_file_count'],
            'dataset_record_occurrences_not_unique_experiments': dict(linked),
            'limits': old['limits'], 'future_test_policy': old['future_test_policy']},
        'datasets': [
            {'dataset': DATASETS[0], 'status': 'independence_not_established',
             'evidence': 'Historical test-named result files and current split-manifest path references; historical bytes and human use unresolved.'},
            {'dataset': DATASETS[1], 'status': 'independence_not_established',
             'evidence': 'Historical intermittent/head_office evaluation exists; current frozen_5000 cohort and historical target overlap unresolved. No exact-path match does not prove non-exposure.'},
            {'dataset': DATASETS[2], 'status': 'independence_not_established',
             'evidence': 'Historical test-named result files and current split-manifest path references; historical bytes and human use unresolved.'},
            {'dataset': DATASETS[3], 'status': 'current_campaign_validation_only_prior_coverage_incomplete',
             'evidence': 'Current deployment contains train/validation only. Two sampled earlier contracts declare validation_only and held_out_test_evaluated=false; this is not an exhaustive historical access audit.'}],
        'raf_historical_allowlisted_contracts': raf_hist,
        'researcher_attestation': {'status': 'requested_pending_response', 'scope': 'prior human viewing or selection use; does not replace historical data identity audit'},
        'current_work': {'data_bytes_hashed': True, 'heldout_rows_or_predictions_or_performance_opened': False,
                         'new_inference_or_training': False, 'new_remote_access': False},
        'interpretation': 'Do not relabel existing test as untouched. Test-file creation is not proof of human viewing. Existing validation is development evidence.'})
    save('sources.json', {'created_utc': now, 'local_files_sha256': SOURCES,
         'source_semantics': 'All endpoint metric records read here are explicitly validation_only. Historical mixed results are not opened. Split data files are hashed as opaque bytes.'})
    print(json.dumps({'conditions': len(rows), 'local_binaries': sum(r['local_binary_present'] for r in rows), 'sources': len(SOURCES)}))


if __name__ == '__main__':
    main()
