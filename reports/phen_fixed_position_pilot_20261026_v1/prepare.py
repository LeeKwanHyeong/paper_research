"""Prepare an isolated additive pilot bundle from the frozen A seed42 source."""
from pathlib import Path
import hashlib
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
import argparse

ROOT = Path(__file__).resolve().parents[2]
CONTROL = Path(__file__).resolve().parent
HERE = CONTROL
BASE = ROOT / 'search_artifacts/shared_component_ablation_20261022_v1'
FROZEN = ROOT / 'reports/phen_intermittent_best_last_20261026_v1/attempt3/bundle/source42'
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/phen_fixed_position_pilot_20261026_v1_5080'
DATASETS = ('raf_spare_parts', 'yellow_trip_hourly')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def main():
    original = json.loads((BASE / 'execution_contract.json').read_text())
    dest = HERE / 'bundle'
    dest.mkdir(exist_ok=False)
    source = dest / 'source'
    source.mkdir()
    for name, digest in original['source']['files'].items():
        src = FROZEN / name
        assert sha(src) == digest, name
        dst = source / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    for name in ('models', 'utils', 'sample_data'):
        (source / name).mkdir(exist_ok=True)
    overlay = ['paper/scripts/run_phen_fixed_position_pilot.py',
               'paper/tests/test_phen_fixed_position_pilot.py']
    for name in overlay:
        dst = source / name
        assert not dst.exists(), 'Overlay must only add new files'
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, dst)
    (source / 'pilot_assets').mkdir()
    shutil.copyfile(CONTROL / 'plot_curves.py', source / 'pilot_assets/plot_curves.py')
    shutil.copyfile(CONTROL / 'remote_parent.py', source / 'pilot_assets/remote_parent.py')
    files = {**original['source']['files'], **{n: sha(source / n) for n in overlay},
             'pilot_assets/plot_curves.py': sha(source / 'pilot_assets/plot_curves.py'),
             'pilot_assets/remote_parent.py': sha(source / 'pilot_assets/remote_parent.py')}
    data = [deepcopy(d) for d in original['datasets'] if d['dataset_id'] in DATASETS]
    inputs = {}
    for item in data:
        assert set(item['inherited_data_identity']['populations']) == {'train', 'validation'}
        # This pilot needs only the already admitted Train/Validation bytes.
        item.pop('full_data_identity', None)
        for key in ('data', 'split_manifest'):
            identity = item['inherited_data_identity'][key]
            name, digest = identity['path'], identity['sha256']
            src = BASE / 'input_snapshot' / name
            assert sha(src) == digest
            dst = source / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            inputs[name] = digest
    host = deepcopy(original['hosts']['5080'])
    host['root'] = REMOTE
    host['source_root'] = REMOTE + '/source'
    host['environment'].update(PYTHONPATH=REMOTE+'/source',MPLCONFIGDIR=REMOTE+'/cache/matplotlib',
                               XDG_CACHE_HOME=REMOTE+'/cache',TORCHINDUCTOR_CACHE_DIR=REMOTE+'/cache/torchinductor')
    host.pop('output_dir', None)
    policy = {'kind': 'fixed_sinusoidal', 'base': 10000.0, 'index_origin': 0,
              'order': 'oldest_to_newest_in_observed_window', 'trainable': False, 'amplitude': 1.0}
    contract = {
        'schema': 'phen_fixed_position_pilot_v1', 'campaign': HERE.name,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'RAF and Taxi full PHEN fixed positional encoding pilot; two fresh fits only',
        'model_role': 'observed_time_joint_v1', 'evaluation_scope': 'validation_only',
        'held_out_test_authorized': False, 'MAC_deferred': True,
        'seeds': [42], 'maximum_parallel_workers': 1,
        'datasets': data, 'training': deepcopy(original['training']),
        'common_condition': deepcopy(original['common_condition']),
        'architecture': {**deepcopy(original['architecture']), 'position_encoding': policy},
        'design_sha256': canonical({'base_design_sha256': original['design_sha256'], 'position_encoding': policy}),
        'position_encoding': policy,
        'source': {'base_git_revision': original['source']['base_git_revision'],
                   'files': files, 'files_sha256': canonical(files),
                   'base_files_sha256': original['source']['files_sha256'],
                   'base_files_count': len(original['source']['files']), 'added_files': overlay + ['pilot_assets/plot_curves.py', 'pilot_assets/remote_parent.py']},
        'input_files': inputs, 'hosts': {'5080': host},
        'jobs': [{'id': d + '__fixed_sinusoidal__seed42', 'dataset': d, 'seed': 42, 'host': '5080'} for d in DATASETS],
        'monitoring': {'epoch_train_eval': 'full_population_eval_mode', 'validation_eval': 'original_full_population',
                       'online_train_loss_separate': True, 'additional_evaluation_preserves_rng': True,
                       'historical_validation_rng_consumption_preserved': True,
                       'wall_time_includes_train_eval': True},
        'authorization': {'user_request': '5080으로 RAF, Taxi 먼저 seed 42 테스트 한번 해보자. 그리고 Train/Validation에 대해서 제대로 되는지 learning curve도 산출하자',
                          'new_fits': 2, 'forbidden': ['held_out_test', '5090_changes', 'retraining_Instacart', 'MAC', 'original_overwrite', 'automatic_scientific_retry']},
        'baseline_contract': {'path': str(BASE / 'execution_contract.json'), 'sha256': sha(BASE / 'execution_contract.json')},
    }
    text = json.dumps(contract, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    (dest / 'execution_contract.json').write_text(text)
    (HERE / 'EXECUTION_CONTRACT.json').write_text(text)
    proof = {'base_files_verified': len(original['source']['files']), 'base_files_changed': 0,
             'source_files': len(files), 'source_files_sha256': canonical(files), 'input_files': inputs,
             'contract_raw_sha256': sha(dest / 'execution_contract.json'),
             'contract_canonical_sha256': canonical(contract), 'remote_root': REMOTE}
    (HERE / 'PREPARED.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',type=int,default=1);args=parser.parse_args()
    if args.attempt != 1:
        assert args.attempt == 2, 'Only the documented pre-training environment repair is admitted'
        HERE=CONTROL/'attempt2';HERE.mkdir(exist_ok=False)
        REMOTE=REMOTE.replace('_v1_5080','_v1_attempt2_5080')
    main()
