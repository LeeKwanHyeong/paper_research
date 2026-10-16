"""Freeze the approved 36 Validation-selected conditions for Test inference."""
from pathlib import Path
import hashlib
import json
import shutil
import time

ROOT = Path('/Users/igwanhyeong/PycharmProjects/paper_research')
B = ROOT / 'search_artifacts/titantpp_history_instacart_test_20261012_v1'
R = ROOT / 'reports/titantpp_history_instacart_test_20261012_v1'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda: f.read(2**20), b''):
            h.update(part)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write('\n')


def copy(src, dst, expected):
    assert sha(src) == expected, str(src)
    assert not dst.exists(), str(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    assert sha(dst) == expected, str(dst)


def main():
    import torch
    inventory_path = Path('/private/tmp/titantpp_test_selected_inventory_20261012.json')
    assert sha(inventory_path) == '4d9e2dfffafba0e67fb12a40bbcd26d1998a8843f2de54cbe969549f88e6fd88'
    inventory = json.loads(inventory_path.read_text())
    assert inventory['condition_count'] == 36 and not (B / 'contract.json').exists()
    copy(inventory_path, B / 'provenance/selected_inventory.json', sha(inventory_path))
    sources, datasets, original_contracts = {}, {}, {}
    full_paths = {
        'yellow_trip_hourly': ROOT/'sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet',
        'intermittent_frozen_5000': ROOT/'sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet',
        'raf_spare_parts': ROOT/'benchmark_data/data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet',
        'insta_market_basket': ROOT/'sample_data/insta_market_basket/instacart_marked_target_with_split.parquet',
    }
    full_shas = {
        'yellow_trip_hourly': 'b47e98e9fdb75d4274a18e3f8a5d8f463418a1d56a6db4db7d9b834c9d89ca46',
        'intermittent_frozen_5000': '85d1fe3ade3ae5a90241018e99a3e9463828d5ba35bc374b56def0168ffffc3f',
        'raf_spare_parts': '439cc3bdd2f969f1121b149de5aa9ece4a7e1ce0c6da9e734a7687c54989780d',
        'insta_market_basket': '06296e48f5ca6c7e0c849f4b4a3c6d54a968ef892754f59369caf1d378424ef2',
    }
    test_counts = dict(zip(full_paths, [8327, 88019, 5226, 578387]))
    thresholds = dict(zip(full_paths, [3449, 187, 200, 35]))
    for group, study in [('history27', 'history'), ('instacart9', 'instacart')]:
        src = inventory['source'][group]
        original_path = Path(src['contract']['path'])
        copy(original_path, B / f'provenance/{study}_execution_contract.json', src['contract']['sha256'])
        original = json.loads(original_path.read_text())
        original_contracts[study] = original
        files = {rel: v['sha256'] for rel, v in src['source_files'].items()}
        assert canonical(files) == src['source_closure']
        for rel, value in src['source_files'].items():
            assert '..' not in Path(rel).parts and not Path(rel).is_absolute()
            copy(value['path'], B / 'sources' / study / rel, value['sha256'])
        sources[study] = {'root': f'sources/{study}', 'files': files, 'closure': src['source_closure'], 'revision': src['scientific_revision']}
        for rel, digest in original['input_files'].items():
            copy(Path(src['source_root']) / rel, B / 'sources' / study / rel, digest)
        for definition in original['datasets']:
            ds = definition['dataset_id']
            assert ds not in datasets
            path = f'data/full/{ds}.parquet'
            copy(full_paths[ds], B / path, full_shas[ds])
            train_rel = definition['prepared_train_validation_path']
            train_sha = original['input_files'][train_rel]
            train_path = f'sources/{study}/{train_rel}'
            manifest = definition['inherited_data_identity']['split_manifest']
            datasets[ds] = {
                'data_path': path, 'data_sha256': full_shas[ds],
                'training_data_path': train_path, 'training_data_sha256': train_sha,
                'definition': definition, 'tail_threshold': thresholds[ds],
                'expected_test_count': test_counts[ds],
                'split_manifest_path': f'sources/{study}/' + manifest['path'],
                'split_manifest_sha256': manifest['sha256'],
                'new_partition': False, 'training_statistics_refit': False,
            }
    selected = {}
    for study, bundle in [
        ('history', ROOT/'search_artifacts/titantpp_contribution_history_rtx_handoff_20261010_v1'),
        ('instacart', ROOT/'search_artifacts/titantpp_instacart_completion_recovery_20261010_v1'),
    ]:
        p = bundle / 'analysis/selected_conditions.json'
        copy(p, B/f'provenance/{study}_selected_conditions.json', sha(p))
        selected[study] = {c['id']: c for c in json.loads(p.read_text())['conditions']}
        p = bundle / 'analysis/completion_receipt.json'
        copy(p, B/f'provenance/{study}_completion_receipt.json', sha(p))
    jobs = []
    for item in inventory['conditions']:
        study = 'history' if item['group'] == 'history27' else 'instacart'
        ident = item['scientific_id']
        p = Path(item['selected_checkpoint']['path'])
        digest = item['selected_checkpoint']['sha256']
        copy(p, B / f'checkpoints/{ident}.pt', digest)
        payload = torch.load(p, map_location='cpu', weights_only=True)
        assert payload['best_epoch'] == item['selected_epoch']
        assert payload['seed'] == item['seed'] and payload['backbone'] == item['arm']
        assert payload['initial_state_sha256'] == item['initial_state_sha256']
        assert payload['model_state_sha256'] == item['checkpoint_state_sha256']
        reference = dict(selected[study][ident]['selected'])
        reference['split'] = 'validation'
        assert payload['selected_metric_value'] == reference['qty_rmse']
        metadata = {k: v for k, v in payload.items() if k != 'model_state_dict'}
        write(B / f'provenance/checkpoint_metadata/{ident}.json', metadata)
        item_provenance = {}
        for key in ['summary', 'input_receipt', 'initialization', 'terminal_manifest']:
            value = item[key]
            rel = f'provenance/jobs/{ident}/{key}.json'
            copy(value['path'], B/rel, value['sha256'])
            item_provenance[key] = {'relative_path': rel, 'sha256': value['sha256']}
        jobs.append({
            'id': ident, 'study': study, 'dataset': item['dataset'], 'arm': item['arm'],
            'seed': item['seed'], 'selected_epoch': item['selected_epoch'],
            'checkpoint': {'relative_path': f'checkpoints/{ident}.pt', 'sha256': digest},
            'checkpoint_state_sha256': item['checkpoint_state_sha256'],
            'checkpoint_metadata_path': f'provenance/checkpoint_metadata/{ident}.json',
            'checkpoint_metadata_sha256': sha(B/f'provenance/checkpoint_metadata/{ident}.json'),
            'initial_state_sha256': item['initial_state_sha256'],
            'selected_metric_value': payload['selected_metric_value'],
            'source_revision_history': payload['source_revision_history'],
            'frozen_source_closure': sources[study]['closure'],
            'source_revision': sources[study]['revision'],
            'training_host': item['training_host'], 'validation_reference': reference,
            'original_provenance': item_provenance,
        })
    assert len(jobs) == 36 and len({j['id'] for j in jobs}) == 36
    host = original_contracts['history']['hosts']['5080']
    contract = {
        'schema': 'titantpp_history_instacart_frozen_test_20261012_v1',
        'created_unix': time.time(), 'evaluation_scope': 'test_final_comparison',
        'metrics_schema': 'quantity_time_tail_v1', 'expected_seeds': [42, 52, 62],
        'authorization': {'human_request': 'ㅇㅇ 그거 진행하자.',
            'approved_task': '현재 이력 보정27 및 Instacart9의 Validation-selected checkpoint를 고정해 기존 Test 최종 비교 수행',
            'existing_Test_access_and_inference_authorized': True,
            'training': False, 'checkpoint_or_hyperparameter_reselection': False,
            'new_dataset': False, 'new_retrieval_training': False,
            'shared_Runtime_or_auth_changes': False, 'manuscript_edit': False},
        'test_is_independent': False, 'heldout_exposure_history_retained': True,
        'original_data_and_checkpoint_modification': False,
        'sources': sources, 'datasets': datasets, 'jobs': jobs,
        'validation_replay_tolerance': {'absolute': 1e-5, 'relative': 1e-5},
        'execution': {'host': '5080', 'remote_root': '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_instacart_test_20261012_v1_5080',
            'maximum_workers': 3, 'per_job_timeout_seconds': 10800,
            'total_timeout_seconds': 43200, 'automatic_retry': False,
            'checkpoint_reselection': False, 'gate': 'all36_validation_replay_before_any_Test'},
        'runtime': {'host': '5080', 'python': host['python'], 'gpu_uuid': host['gpu_uuid'],
            **host['runtime_expected'], 'environment': host['environment'],
            'torch_num_threads': 4, 'interop_threads': 1, 'TF32': False,
            'dtype': 'float32', 'deterministic_algorithms': True},
    }
    contract['contract_sha256'] = canonical(contract)
    write(B/'contract.json', contract)
    file_map = {str(p.relative_to(B)): sha(p) for p in B.rglob('*') if p.is_file()}
    write(B/'input_integrity.json', {'schema': 'selected_inputs_immutable_v1', 'contract_sha256': contract['contract_sha256'], 'files': file_map})
    copy(inventory_path, R/'selected_inventory.json', sha(inventory_path))
    print(json.dumps({'status': 'prepared', 'jobs': len(jobs), 'contract_sha256': contract['contract_sha256'], 'selected_checkpoint_bytes': sum((B/j['checkpoint']['relative_path']).stat().st_size for j in jobs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
