"""Bind the nine already trained B checkpoints; no training or inference."""
import copy
import datetime
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'reports/titantpp_legacy_evaluation_20261003_v1'
DATASETS = ['yellow_trip_hourly', 'intermittent_frozen_5000', 'insta_market_basket']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(name, data):
    with (HERE / name).open('x') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def main():
    inventory = read(HERE / 'binding_inventory.json')
    rows = copy.deepcopy(inventory['rows'])
    keys = {(r['dataset'], r['model'], r['seed']) for r in rows}
    assert len(rows) == 9 and keys == {(d, 'titantpp', s) for d in DATASETS for s in (42, 52, 62)}
    for row in rows:
        assert row['endpoint'] == 'selected'
        assert sha(ROOT / row['checkpoint_path']) == row['checkpoint_file_sha256']
    registry = {
        'status': 'bound_original_selected_B_checkpoints', 'rows': rows,
        'bundles': inventory['bundles'], 'binding_inventory_sha256': sha(HERE / 'binding_inventory.json'),
        'scope': 'B9 same scientific conditions; RAF older head/selector excluded, no new training',
    }
    write('evaluation_registry.json', registry)
    manifest = copy.deepcopy(read(OLD / 'dataset_manifest.json'))
    manifest['datasets'] = [d for d in manifest['datasets'] if d['dataset'] in DATASETS]
    write('dataset_manifest.json', manifest)
    refs = {
        'relative_tolerance': 1e-5, 'absolute_tolerance': 1e-5,
        'selection': 'original first strict minimum validation raw quantity RMSE',
        'references': {f"{r['dataset']}__{r['model']}__seed{r['seed']}": r['validation_reference'] for r in rows},
    }
    write('validation_references.json', refs)
    contract = copy.deepcopy(read(OLD / 'execution_contract.json'))
    contract.update({
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'route': 'exploratory_B_test_comparison_after_MLP_test_results',
        'registry_path': str((HERE / 'evaluation_registry.json').relative_to(ROOT)),
        'registry_sha256': sha(HERE / 'evaluation_registry.json'),
        'dataset_manifest_sha256': sha(HERE / 'dataset_manifest.json'),
        'dataset_sha256': {d['dataset']: d['sha256'] for d in manifest['datasets']},
        'supersedes': 'Separate user-approved B evaluation; preserves the completed 108-condition campaign.',
        'qualification': {'cpu': 'all nine conditions, first validation batch',
                          'cuda': 'all nine conditions, first four validation batches',
                          'full_validation': 'all nine original selected validation metrics must reproduce'},
        'comparison_scope': {
            'claim': 'Exploratory B versus MLP and the six previously evaluated external encoders',
            'comparison_models': ['titantpp_history_mlp', 'rmtpp', 'thp', 'nhp', 'sahp', 's2p2_matched_head', 'attnhp_matched_head'],
            'legacy_analysis_path': str((OLD / 'analysis.json').relative_to(ROOT)),
            'legacy_analysis_sha256': sha(OLD / 'analysis.json'),
            'legacy_test_run_manifest_path': str((OLD / 'runs/test/attempt1/run_manifest.json').relative_to(ROOT)),
            'legacy_test_run_manifest_sha256': sha(OLD / 'runs/test/attempt1/run_manifest.json'),
            'post_test_addition': True, 'representative_reselection': False,
            'RAF': 'No comparable observed-time B fit; older RAF joint-objective/legacy-head fit not pooled.',
            'Deep_Renewal': 'Existing research records preserved; outside this already defined common-head panel.',
        },
    })
    contract['approval'].update({
        'datasets': DATASETS, 'models': ['titantpp'],
        'user_text': '오케이 B 관련해 Test 진행해보자',
        'scope': 'Existing B selected checkpoints on the same Test populations; original nine conditions, no retraining or paid rental.',
    })
    contract['resources']['campaign_timeout_seconds'] = 86400
    contract['resources']['output_limit_gib'] = 15
    contract['aggregation']['difference'] = 'B_minus_comparator'
    contract['aggregation']['relative_improvement'] = '100*(comparator_mean_RMSE-B_mean_RMSE)/comparator_mean_RMSE'
    contract['resampling'].pop('fixed_four_RMSE_anchor_interval_quantiles', None)
    contract.pop('fixed_validation_selected_anchors', None)
    contract['resampling']['units'] = {d: contract['resampling']['units'][d] for d in DATASETS}
    contract['resampling']['multiplicity_scope'] = 'exploratory pointwise intervals only; no confirmatory family-wise coverage claim'
    contract['resampling']['implementation_and_real_data_CIs_status'] = 'Reuse the validated existing-test bootstrap algorithm; new B code tests and real B comparisons recorded separately.'
    contract['resampling']['inference_scope'] = 'exploratory pointwise 95% intervals conditional on fixed trained seeds; no confirmatory anchor correction'
    contract['release']['reporting'] = 'All nine B conditions, including failures/unstarted, alongside retained legacy MLP and six external comparators. Previous 108-condition evidence preserved; no representative reselection.'
    write('execution_contract.json', contract)
    print(json.dumps({'conditions': len(rows), 'registry_sha256': contract['registry_sha256'],
                      'contract_sha256': sha(HERE / 'execution_contract.json')}))


if __name__ == '__main__':
    main()
