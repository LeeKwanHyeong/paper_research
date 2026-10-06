"""Freeze the approved five-way Taxi seed42 retrieval pilot; local files only."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
NAME = 'titantpp_contextual_retrieval_pilot_dual_20261007_v1'
BUNDLE = PROJECT / 'search_artifacts' / NAME
PARENT = PROJECT / 'search_artifacts/titantpp_cnn_gru54_3seed_dual_20261005_v1'
PARENT_SHA = '845a891db284f2cf88aa2ab17bee47716ea8dc11e6268646a71c772073ab7130'
ARMS = ['titantpp_retrieval_' + s for s in ('mean', 'weighted', 'fixed', 'adaptive', 'generic')]
ADDITIONS = ['models/TPPs/CountAwareTitanContextualRetrieval.py',
             'paper/scripts/run_titantpp_contextual_retrieval.py',
             'paper/tests/test_titantpp_contextual_retrieval_model.py']
USER_INSTRUCTION = '이거 실험 진행하자. 5080도 얼마 안남았으니 병행해서 진행하는 방향으로.'


def read(p): return json.loads(Path(p).read_text())
def canonical(v): return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x') as f: json.dump(v, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')


def prepare():
    assert not (BUNDLE / 'execution_contract.json').exists(), 'Existing frozen pilot preserved; no retry/replacement'
    parent = read(PARENT / 'execution_contract.json')
    assert canonical(parent) == PARENT_SHA
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    for name in ADDITIONS:
        content = subprocess.check_output(['git', 'show', revision + ':' + name], cwd=PROJECT)
        assert hashlib.sha256(content).hexdigest() == sha(PROJECT / name), 'Uncommitted scientific addition: ' + name
    source = BUNDLE / 'source'; source.mkdir(parents=True, exist_ok=False)
    files = deepcopy(parent['source']['files'])
    for name, digest in files.items():
        original = PARENT / 'source' / name
        assert sha(original) == digest, 'Frozen parent changed: ' + name
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, target)
    for name in ADDITIONS:
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / name, target); files[name] = sha(target)
    data = deepcopy(next(d for d in parent['datasets'] if d['dataset_id'] == 'yellow_trip_hourly'))
    inputs = {}
    for record in (data['inherited_data_identity']['data'], data['inherited_data_identity']['split_manifest']):
        name = record['path']; assert not Path(name).is_absolute() and '..' not in Path(name).parts
        original = PARENT / 'source' / name; assert sha(original) == record['sha256']
        target = source / name; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(original, target)
        inputs[name] = sha(target)
    sentinel = source / 'sample_data/.frozen_root_sentinel'
    sentinel.parent.mkdir(exist_ok=True); sentinel.write_bytes(b''); inputs[str(sentinel.relative_to(source))] = sha(sentinel)
    bootstrap = "from pathlib import Path; from paper.scripts import run_titantpp_contextual_retrieval as e; e.install_hooks(); assert e.ROOT == Path.cwd(); print('isolated frozen bootstrap passed')"
    subprocess.run([sys.executable, '-c', bootstrap], cwd=source,
        env={**os.environ, 'PYTHONPATH':str(source), 'PYTHONDONTWRITEBYTECODE':'1'}, check=True)
    hosts = {}
    for host in ('5090', '5080'):
        spec = deepcopy(parent['hosts'][host])
        root = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/' + NAME + '_' + host
        spec.update(root=root, source_root=root + '/source', operation_root=root + '/operation',
                    output_dir=root + '/run', tmux=NAME + '_' + host, assigned_datasets=['yellow_trip_hourly'])
        spec['environment'].update(SOURCE_REVISION=revision, PYTHONPATH=root + '/source',
                                  MPLCONFIGDIR=root + '/cache/matplotlib', XDG_CACHE_HOME=root + '/cache')
        spec['old_owner_roots'] = [
            '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_architecture_' +
            ('capacity_handoff_20261006_v1_5080' if host == '5080' else 'contribution_dual_20261006_v1_5090_attempt2')]
        hosts[host] = spec
    jobs = [{'id': f'{host}__yellow_trip_hourly__42__{arm}', 'configuration_id': f'yellow_trip_hourly__42__{arm}',
             'host': host, 'dataset': 'yellow_trip_hourly', 'seed': 42, 'arm': arm}
            for host in hosts for arm in ARMS]
    operation_files = {str(p.relative_to(BUNDLE)): sha(p) for p in sorted((BUNDLE / 'operation').glob('*.py'))}
    assert {'operation/campaign.py', 'operation/qualify.py', 'operation/monitor.py'} <= set(operation_files)
    design = {
        'schema': 'titantpp_contextual_retrieval_design_v1', 'backbone': 'titantpp_cnn_gru54',
        'memory_slots': 64, 'memory_dim': 64, 'topk': 4, 'context_window': 32,
        'point_normalization': 'Frozen Train log1p(quantity) and log(dt/time_scale) mean/std, then causal summaries; no Validation fit',
        'features_per_channel': ['last', 'mean', 'population_std', 'last_difference'],
        'missing': 'Only observed prefix; dt<=0 excluded; quantity zero valid; n>=1 last/mean, n>=2 std/difference; unavailable features zero; reset at withheld valid event',
        'weighted_temperature': 1.0,
        'fixed': 'cos(h+phi_t Wt,mem)/2 + cos(h+phi_q Wq,mem)/2; shared 4x64 zero-initialized projections',
        'adaptive': 'alpha=sigmoid(phi_all8 g8+b); alpha time score+(1-alpha) quantity score',
        'generic': 'query=h+0.5 phi_all8 W8x64; tau=exp(log(2)*tanh(phi_all8 g8+b)), tau in [0.5,2]',
        'additional_parameters': {ARMS[0]: 0, ARMS[1]: 0, ARMS[2]: 512, ARMS[3]: 521, ARMS[4]: 521},
        'selection_and_value_gradient': 'Inherited native deterministic torch.topk policy; index choice non-differentiable, selected cosine softmax and values differentiable',
        'mean_identity': 'Exact frozen CNNGRU54 mean retrieval, all original parameters/buffers/RNG preserved',
        'weighted_initialization': 'Four weighted arms share zero-shift E0 query, retrieval, common tensors, post-initialization RNG',
        'interpretation': 'Past time/quantity-conditioned latent compatibility with learned static prototypes; keys are not stored events',
        'not_included': ['correction gain training', 'time head refitting', 'new loss', 'new checkpoint selector'],
        'comparison_limits': ['fixed512 vs adaptive521 includes nine gate parameters',
                              'adaptive vs generic is semantic query-mixture vs generic query/temperature package contrast',
                              'two GPU blocks are same-seed repetitions, not two seeds or independent evaluation']}
    c = {'schema': NAME, 'runner_module': 'paper.scripts.run_titantpp_contextual_retrieval',
         'objective': 'Isolate retrieval aggregation, time/quantity query decomposition and history-conditioned mixture for quantity prediction; time harm mandatory',
         'arms': ARMS, 'seeds': [42], 'hosts': hosts, 'datasets': [data],
         'dataset_sha256': {'yellow_trip_hourly': canonical(data)},
         'inherited_dataset_sha256': {'yellow_trip_hourly': parent['dataset_sha256']['yellow_trip_hourly']},
         'parent_dataset_contract': data, 'jobs': jobs,
         'canonical_configs': 5, 'canonical_conditions': 5, 'execution_conditions': 10,
         'new_fits': 10, 'already_terminal_reused': 0,
         'source': {'git_revision': revision, 'base_git_revision': revision,
                    'parent_base_git_revision': parent['source']['git_revision'],
                    'files': files, 'files_sha256': canonical(files),
                    'parent_files': parent['source']['files'], 'parent_files_sha256': parent['source']['files_sha256'],
                    'parent_science117_closure': '40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd'},
         'operation': {'files': operation_files, 'files_sha256': canonical(operation_files)},
         'input_files': inputs, 'training': deepcopy(parent['training']), 'architecture': design,
         'design_sha256': canonical(design),
         'limits': {'total_wall_seconds': 28800, 'per_condition_seconds': 7200, 'qualification_seconds': 1800,
                    'workers_per_host': 1, 'automatic_retry': False, 'peak_device_fraction_max': .8,
                    'cpu_rss_max_bytes': 16*1024**3, 'owned_root_max_bytes': 8*1024**3,
                    'minimum_free_disk_bytes': 5*1024**3},
         'cost_limits': {'inference_median_ratio_max': 1.20, 'training_step_median_ratio_max': 1.30,
                         'peak_allocated_ratio_max': 1.20, 'warmup_steps_min': 5, 'measured_steps_min': 20,
                         'technical_qualification_and_candidate_cost_eligibility_separate': True},
         'pilot_success': {'primary_host': '5090', 'candidate': ARMS[3],
                           'comparators': [ARMS[0], ARMS[2], ARMS[4]],
                           'strict_lower': ['selected_fullValidation_quantity_RMSE', 'selected_tailValidation_quantity_RMSE'],
                           'cost_limits_required': True, 'MAE_and_TimeNLL': 'mandatory same-epoch overall/tail reporting, not posthoc selection',
                           'replication_host': '5080', 'automatic_multidataset_expansion': False},
         'diagnostic': deepcopy(parent.get('diagnostic', {})), 'evaluation_scope': 'validation_only',
         'held_out_test_evaluated': False, 'reporting': {'mandatory_time_harm': True,
                    'same_selected_epoch_all_metrics': True, 'historical_Test_exposure': True,
                    'independent_evaluation': False, 'hardware_timings_not_pooled': True},
         'authorization': {'user_instruction': USER_INSTRUCTION, 'five_retrieval_pilot': True,
                           'two_paired_GPU_blocks': True, 'Test_access': False, 'shared_Runtime_or_auth_change': False},
         'parent_contract_sha256': PARENT_SHA,
         'capacity_gate': {'5080': 'Existing54 scientific terminal, actual shutdown, all original SHA recovery and final Validation receipt before GPU qualification/launch',
                           '5090': 'Preserved original12 terminal boundary plus live current GPU idle and no old owner at each launch gate'},
         'policy': {'automatic_retry': False, 'single_worker_per_host': True,
                    'originals_claims_unfavorable_results_preserved': True}}
    c['limits'].update(root_bytes=8*1024**3, min_free_bytes=5*1024**3,
                       cpu_rss_bytes=16*1024**3, gpu_fraction_max=.8)
    spec = importlib.util.spec_from_file_location('_pilot_operation_validation', BUNDLE / 'operation/campaign.py')
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); m.validate(c, verify_files=False)
    digest = canonical(c)
    control_files = {str(p.relative_to(BUNDLE)):sha(p) for p in sorted((BUNDLE/'control').glob('*.py'))}
    assert {'control/finalize_once.py','control/test_finalize_offline.py'} <= set(control_files)
    integrity = {'schema':'titantpp_contextual_retrieval_control_integrity_v1',
                 'contract_sha256':digest,'files':control_files,'files_sha256':canonical(control_files),
                 'automatic_retry':False}
    write(BUNDLE/'control/integrity_receipt.json',integrity)
    approval = {'schema': 'titantpp_contextual_retrieval_approval_v1', 'approved': True,
                'contract_sha256': digest, 'user_instruction': USER_INSTRUCTION, 'recorded_unix': time.time(),
                'hosts': ['5090','5080'], 'fresh_fits': 10, 'configurations': 5,
                'Test_access': False, 'shared_Runtime_or_auth_change': False, 'automatic_retry': False}
    for name, value in [('execution_contract.json', c), ('approval.json', approval), ('design_contract.json', design),
                        ('source_registry.json', c['source']), ('operations_manifest.json', c['operation'])]: write(BUNDLE / name, value)
    write(BUNDLE / 'current.json', {'schema': NAME, 'canonical_sha256': digest,
           'contract_path': 'execution_contract.json', 'source_closure_sha256': c['source']['files_sha256'],
           'operation_closure_sha256': c['operation']['files_sha256'], 'hosts': {h:s['root'] for h,s in hosts.items()},
           'control_integrity_receipt_sha256':sha(BUNDLE/'control/integrity_receipt.json'),
           'control_closure_sha256':canonical(control_files),
           'status': 'frozen_awaiting_native_qualification', 'training_started': False,
           'canonical_configs': 5, 'execution_conditions': 10, 'automatic_retry': False})
    print(json.dumps({'status':'frozen','contract_sha256':digest,'source_files':len(files),
                      'configurations':5,'fresh_fits':10,'remote_calls':0}, ensure_ascii=False))


if __name__ == '__main__': prepare()
