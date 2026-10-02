"""Freeze existing split identities and explicit legacy-evaluation authorization."""
import datetime
import hashlib
import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
REGISTRY = ROOT / 'reports/titantpp_final_eval_checkpoint_binding_20261003_v1/evaluation_registry.json'
PRIOR = ROOT / 'reports/titantpp_final_eval_runner_validation_20261003_v1/effective_protocol.json'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def main():
    registry = json.loads(REGISTRY.read_text())
    prior = json.loads(PRIOR.read_text())
    old_data = json.loads((ROOT / 'reports/titantpp_independent_final_evaluation_design_20261003_v1/dataset_freeze.json').read_text())
    datasets = []
    for d in old_data['datasets']:
        ident = d['identities']['data']
        path = ROOT / ident['path']
        assert sha(path) == ident['sha256']
        split_identity = d['identities']['split_manifest']
        assert sha(ROOT / split_identity['path']) == split_identity['sha256']
        frame = pl.read_parquet(path).sort(['oper_part_no', 'seq'])
        assert not frame.select(pl.struct(['oper_part_no', 'seq']).is_duplicated().any()).item()
        assert set(frame['chronological_split'].unique()) == {'train', 'validation', 'test'}
        assert frame.select((pl.col('seq').diff().over('oper_part_no').drop_nulls() > 0).all()).item()
        pop = {}
        for split in ['train', 'validation', 'test']:
            selected = frame.filter((pl.col('chronological_split') == split) & (pl.col('seq') != pl.col('seq').min().over('oper_part_no')))
            h = hashlib.sha256()
            for row in selected.select(['oper_part_no', 'seq', 'demand_dt']).iter_rows():
                h.update((json.dumps([str(row[0]), int(row[1]), str(row[2])], separators=(',', ':'))+'\n').encode())
            pop[split] = {'target_count': selected.height, 'entity_count': selected['oper_part_no'].n_unique(),
                          'ordered_entity_seq_recorded_date_sha256': h.hexdigest(),
                          'recorded_date_min': selected['demand_dt'].min(), 'recorded_date_max': selected['demand_dt'].max()}
        entry = {'dataset': d['dataset'], 'path': ident['path'], 'sha256': ident['sha256'],
                 'split_manifest': d['identities']['split_manifest'], 'populations': pop,
                 'loader': d['frozen_loader'], 'selection': 'all existing noninitial targets of each frozen split; no new entity/date filtering'}
        if 'deployed_data' in d['identities']:
            deployed = d['identities']['deployed_data']
            assert sha(ROOT / deployed['path']) == deployed['sha256']
            assert frame.filter(pl.col('chronological_split').is_in(['train', 'validation'])).equals(pl.read_parquet(ROOT / deployed['path']).sort(['oper_part_no', 'seq']))
            entry['prior_train_validation_file'] = deployed
            entry['full_file_train_validation_rows_equal_deployed'] = True
        for split in ['train', 'validation']:
            assert pop[split]['target_count'] == d['train_validation_population'][split]['target_count']
        datasets.append(entry)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    manifest = {'schema_version': 1, 'created_utc': now, 'route': 'retrospective_locked_test_reevaluation', 'datasets': datasets,
                'target_definition': 'canonical frozen loader next observed event after at least one history event',
                'date_shift': {'dataset': 'intermittent_frozen_5000', 'status': 'user_confirmed_constant_offset_for_security_and_deidentification',
                               'evidence': 'reports/titantpp_intermittent_date_shift_note_20261003_v1/confirmation.json'},
                'not_independent_untouched_data': True}
    dump(OUT / 'dataset_manifest.json', manifest)
    models = list(dict.fromkeys(r['model'] for r in registry['rows']))
    contract = {'schema_version': 1, 'created_utc': now, 'route': manifest['route'],
                'approval': {'test_inference_authorized': True, 'retraining_authorized': False,
                             'allowed_splits': ['validation', 'test'], 'datasets': prior['datasets'],
                             'models': models, 'seeds': [42, 52, 62],
                             'user_text': '진행하자. 승인할게',
                             'scope': 'Three-step legacy split freeze, full execution qualification, frozen evaluation and manuscript update explicitly approved after plan was quoted.',
                             'additional_paid_resources_authorized': False},
                'registry_path': str(REGISTRY.relative_to(ROOT)), 'registry_sha256': sha(REGISTRY),
                'dataset_manifest_sha256': sha(OUT / 'dataset_manifest.json'),
                'dataset_sha256': {d['dataset']: d['sha256'] for d in datasets},
                'parent_protocol': str(PRIOR.relative_to(ROOT)), 'parent_protocol_sha256': sha(PRIOR),
                'supersedes': 'Earlier preparation-only prohibition on legacy test inference; no change to frozen scientific models or selection.',
                'comparison_scope': prior['comparison_scope'], 'prediction_protocol': prior['prediction_protocol'],
                'aggregation': prior['aggregation'], 'resampling': prior['resampling'],
                'fixed_validation_selected_anchors': prior['fixed_validation_selected_anchors'],
                'resources': {'host': '5080', 'root': '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_legacy_evaluation_20261003_v1',
                              'python': '/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12',
                              'gpu_uuid': 'GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6', 'concurrent_workers': 1,
                              'gpu_memory_limit_gib': 12, 'rss_limit_gib': 32, 'output_limit_gib': 15,
                              'condition_timeout_seconds': 10800, 'campaign_timeout_seconds': 86400,
                              'new_rental_cost_usd': 0, 'shared_runtime_changes': False},
                'release': {'before_test': 'Local unit tests, full data-loader enumeration, CUDA validation qualification and source/contract seals must pass.',
                            'reporting': 'All 108 conditions and two deterministic references per dataset, including failures. No model selection using these results.',
                            'failure_policy': 'Preserve attempt and diagnose; no automatic scientific changes or favorable target intersections.'}}
    dump(OUT / 'execution_contract.json', contract)
    print(json.dumps({d['dataset']: d['populations']['test']['target_count'] for d in datasets}))

if __name__ == '__main__':
    main()
