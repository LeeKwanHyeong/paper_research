"""Prepare the approved RAF6 extension without changing Taxi or its science."""
from pathlib import Path
import base64
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[2]
PARENT = ROOT / 'search_artifacts/titantpp_cnn_history_adaptation_matched_20261015_v1/execution_v1'
OLD = ROOT / 'search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution_manual2'
MEMORY = ROOT / 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3'
BUNDLE = ROOT / 'search_artifacts/titantpp_cnn_history_raf6_20261015_v1'
REPORT = ROOT / 'reports/titantpp_cnn_history_raf6_20261015_v1'
SCHEMA = 'titantpp_cnn_history_raf6_20261015_v1'
ARMS = ('titantpp_cnn_history_adaptive_v1', 'titantpp_cnn_history_capacity_v1')
SEEDS = (42, 52, 62)
PARENT_INTEGRITY = 'c34a26cce9b8656917e0c1dc10d67b53104b0fbd23ae02f96caed2d4b444661a'
OLD_CONTRACT = '274ff80ef771788dc4652749bd2875716c7ddeb59ba0a1a0e497358545fe1fdc'
MEMORY_CONTRACT = 'd51ab06365a7165e00cc18849b84caba4ee2dd8af6a36aee84a5c1d62e01fab3'
PYTHON_5090 = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1_5090/runtime_env/bin/python3.12'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')


def main():
    assert sha(PARENT/'followup_integrity.json') == PARENT_INTEGRITY
    pins = read(PARENT/'followup_integrity.json')['files']
    for name, digest in pins.items():
        assert sha(ROOT/name) == digest, name
    assert sha(OLD/'execution_contract.json') == OLD_CONTRACT
    assert sha(MEMORY/'contract.json') == MEMORY_CONTRACT
    pc, old, memory = read(PARENT/'execution_contract.json'), read(OLD/'execution_contract.json'), read(MEMORY/'contract.json')
    data = next(d for d in old['datasets'] if d['dataset_id'] == 'raf_spare_parts')
    jobs = [j for j in memory['jobs'] if j['dataset'] == 'raf_spare_parts']
    assert len(jobs) == 3 and {j['seed'] for j in jobs} == set(SEEDS)
    baseline_pins = {}
    for job in jobs:
        orig = job['original']
        ref = next(r for r in old['baseline_references'] if r['id'] == job['id'])
        assert orig['selected_epoch'] == ref['selected_epoch']
        assert orig['checkpoint']['sha256'] == ref['checkpoint_sha256']
        assert orig['checkpoint_state_sha256'] == ref['state_tensor_sha256']
        assert orig['validation_reference']['count'] == 6690
        assert orig['validation_reference']['qty_rmse'] == ref['metrics']['whole_qty_rmse']
        for path_key, hash_key in [('checkpoint_metadata_path','checkpoint_metadata_sha256'),
                ('selection_history_path','selection_history_sha256'),('worker_exit_path','worker_exit_sha256'),
                ('terminal_manifest_path','terminal_manifest_sha256')]:
            p = MEMORY/orig[path_key]; assert sha(p) == orig[hash_key]
            baseline_pins[str(p.relative_to(ROOT))] = orig[hash_key]
        p = MEMORY/orig['checkpoint']['relative_path']; assert sha(p) == orig['checkpoint']['sha256']
        baseline_pins[str(p.relative_to(ROOT))] = orig['checkpoint']['sha256']
        meta = read(MEMORY/orig['checkpoint_metadata_path'])
        # All metadata and selected bytes remain pinned; no Test file is read.
        assert isinstance(meta, dict)
    BUNDLE.mkdir(exist_ok=False); REPORT.mkdir(exist_ok=False)
    shutil.copytree(PARENT/'source', BUNDLE/'source')
    # Parent source includes Taxi input files; remove only these newly copied
    # redundant bytes from this new bundle, never from the parent bundle.
    shutil.rmtree(BUNDLE/'source/inputs')
    adapter = 'paper/scripts/run_titantpp_adaptive_history_cnn.py'
    text = (PARENT/'source'/adapter).read_text()
    old_check = '== 90678, "Active count changed"'
    assert text.count(old_check) == 1
    text = text.replace(old_check, '== sum(p.numel() for p in reference_model.parameters() if p.requires_grad) + 243, "Active count changed"')
    (BUNDLE/'source'/adapter).write_text(text)
    source = {n: sha(BUNDLE/'source'/n) for n in pc['source']['files']}
    assert [n for n in source if source[n] != pc['source']['files'][n]] == [adapter]
    inputs = {}
    for key in ('data', 'split_manifest'):
        entry = data['inherited_data_identity'][key]; name, digest = entry['path'], entry['sha256']
        assert sha(OLD/'source'/name) == digest
        target = BUNDLE/'source'/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OLD/'source'/name, target); inputs[name] = digest
    sentinel = BUNDLE/'source/sample_data/.frozen_root_sentinel'
    assert sentinel.is_file(); inputs[str(sentinel.relative_to(BUNDLE/'source'))] = sha(sentinel)
    prep = {'schema':'raf_history_cnn_preparation_v1','dataset':data,'baseline_jobs':jobs,'source_files':source,
        'parent_contract_raw_sha256':sha(PARENT/'execution_contract.json'),'parent_design_sha256':pc['design_sha256'],
        'architecture_identical_to_taxi':True,'adapter_change':'dataset-independent baseline active parameter count +243',
        'primary_contrast':'Adaptive-Capacity on same 5090 runtime','secondary_contrast':'each candidate versus historical CNN on 5080',
        'runtime_confound':'historical CNN=5080/cuDNN91900; new arms=5090/cuDNN92000',
        'held_out_test_evaluated':False}
    write(BUNDLE/'preparation/contract.json', prep)
    operation = (PARENT/'operation/runtime.py').read_text()
    operation = operation.replace('Taxi','RAF').replace('taxi6','raf6').replace('yellow_trip_hourly','raf_spare_parts')
    operation = operation.replace('5080','5090').replace('GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6','GPU-c9adc246-ef66-7906-bc07-6e152fa9952f')
    operation = operation.replace('/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12', PYTHON_5090)
    operation = operation.replace('c5ae7ffed7e89c798dcb7bf7264e15a781adf10c6162a632501c3b8019bcdb3b',sha(BUNDLE/'preparation/contract.json'))
    target = ROOT/'paper/scripts/titantpp_history_raf6_runtime.py'
    with target.open('x') as f: f.write(operation)
    (BUNDLE/'operation').mkdir(); shutil.copy2(target, BUNDLE/'operation/runtime.py')
    shutil.copy2(PARENT/'operation/frozen_runtime.py', BUNDLE/'operation/frozen_runtime.py')
    snapshot = read(ROOT/'search_artifacts/titantpp_cnn_gru_components_20261012_v5/control/qualification_cache/5090/snapshot.json')
    entry = snapshot['files']['qualification/receipt.json']; raw = base64.b64decode(entry['base64'])
    assert hashlib.sha256(raw).hexdigest() == entry['raw_sha256'] == '4e0c035f12f8dd3cc75d8b8ddde349f46375bd8e5709a275a4fdc38986914fdb'
    (BUNDLE/'preparation/original_5090_qualification.json').write_bytes(raw)
    qr = json.loads(raw)['runtime']; host = old['hosts']['5090']; remote = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+SCHEMA+'_5090'
    host.update(root=remote,source_root=remote+'/source',operation_root=remote+'/operation',output_dir=remote+'/run',tmux=SCHEMA+'_5090',assigned_datasets=['raf_spare_parts'])
    host['environment'].update(PYTHONPATH=remote+'/source',MPLCONFIGDIR=remote+'/cache/matplotlib',XDG_CACHE_HOME=remote+'/cache')
    keys=('python','torch','numpy','polars','cuda','cudnn','threads','interop_threads','default_dtype','deterministic_algorithms','deterministic_warn_only','cuda_matmul_tf32','cudnn_tf32','gpu')
    host['runtime_expected']={k:qr[k] for k in keys}
    c = {**pc,'schema':SCHEMA,'datasets':[data],'baseline_jobs':jobs,'hosts':{'5090':host},'input_files':inputs,
        'jobs':[{'id':f'raf_spare_parts__{s}__{a}','host':'5090','dataset':'raf_spare_parts','seed':s,'arm':a} for a in ARMS for s in SEEDS],
        'source':{**pc['source'],'files':source,'files_sha256':canonical(source)},
        'preparation_contract_sha256':sha(BUNDLE/'preparation/contract.json'),
        'architecture':{'prepared_contract_sha256':sha(BUNDLE/'preparation/contract.json'),'added_parameters_each':243},
        'original_training_runtime_receipt_sha256':sha(BUNDLE/'preparation/original_5090_qualification.json'),
        'comparison_runtime_limitation':prep['runtime_confound'],'primary_contrast':prep['primary_contrast']}
    c['operation']={'files':{str(p.relative_to(BUNDLE)):sha(p) for p in (BUNDLE/'operation').glob('*.py')}}
    write(BUNDLE/'execution_contract.json',c)
    write(BUNDLE/'approval.json',{'approved':True,'hosts':['5090'],'contract_sha256':canonical(c),
        'user_instruction':'오케이 그러면 맞춰서 RAF 를 5090에서 바로 시험할 수 있도록 하자',
        'clarification_question':'RAF도 Taxi와 같은 두 후보 × 3seed를 준비하고, 5090 검증을 통과하면 학습까지 바로 시작할까요?',
        'clarification_answer':'검증 통과 후 6개 학습 시작','fresh_fit_count':6,'new_training_authorized':True,'Test':False})
    files={str(p.relative_to(BUNDLE)):sha(p) for p in sorted(BUNDLE.rglob('*')) if p.is_file()}
    write(BUNDLE/'deployment_manifest.json',{'files':files,'contract_sha256':canonical(c)})
    write(REPORT/'LOCAL_READY.json',{'status':'prepared_native_checks_pending','contract_raw_sha256':sha(BUNDLE/'execution_contract.json'),
        'contract_canonical_sha256':canonical(c),'deployment_manifest_sha256':sha(BUNDLE/'deployment_manifest.json'),
        'baseline_pins':baseline_pins,'parent_pins':pins,'source_files':len(source),'identical_parent_source_files':len(source)-1,
        'scientific_model_unchanged':True,'frozen_scientific_fit_unchanged':True,'scientific_fits_started':0,'held_out_test_evaluated':False})
    print(json.dumps({'bundle':str(BUNDLE),'contract_sha256':canonical(c),'files':len(files)}))


if __name__ == '__main__':
    main()
