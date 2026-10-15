"""Prepare an isolated current-candidate Train/Validation row diagnostic; never remote."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

PROJECT = Path(__file__).resolve().parents[2]
NAME = 'titantpp_backbone_value_difference_rows_20261016_v1'
BUNDLE = PROJECT / 'search_artifacts' / NAME
TRAIN = PROJECT / 'search_artifacts/titantpp_backbone_value_difference_raf3_20261016_v1'
PARENT = TRAIN / 'evaluation_v1'
REPORT = PROJECT / 'reports/titantpp_backbone_value_difference_raf3_20261016_v1'
SPEC = REPORT / 'CURRENT_CANDIDATE_ROWS_SPEC.json'
SPEC_SHA = 'c6ee63c3dfb6ee750ce1ff3b27ae40257a62fcacd1b2e7044853e15c04c8ed0c'
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/' + NAME + '_5090'
ARM = 'titantpp_cnn_value_difference_v1'
DEADLINE = 1792679647.915156
SOURCE = '4fce860e41ee716592abf168af8c3985859b74ea6d080357690124b19a88d361'
TRAIN_RECEIPT = 'c90ea7adfbc7d687650f98b0b2905818b2c7b469dbbf02d6f37ed24badc650fb'
EVALUATION_RECEIPT = '4eb93c8ea1ca8ed589b9ce526caa1e85066fbd138bb92e4a5e8b457776cbc670'
TRAIN_RAW = 'c40a2921595bf60ac13d34d915f5a5b3fabe57db00b0eabf394d085c3c00ad99'
BASELINE_DIAG = REPORT / 'diagnostics_paired_regret_v1'
BASELINE_RECEIPT = '0c96e3be22deeae3f8d8ab8dde87640ab546acfd7be98f11b553c68b187cc7b9'
HELPER_SHA = '6f3d333694bb23109ccc5d1ed6d6257f6eb34be7af41e96bae14aab12d66cbc5'
ROW_UTILS_SHA = '0ef413db7f6f41d4fe274ea60b8bea7372a5851cc688247406169f1d494b4225'
LINEAGE_SHA = '1d1d680ffbfaaf34f75c2aed884c37b1c3071a6a6fe081129204b359a75ae30b'
LOCAL_OPERATIONS = ('prepare_titantpp_value_difference_rows.py', 'advance_titantpp_value_difference_rows.py',
                    'titantpp_value_difference_rows_runtime.py', 'export_titantpp_value_difference_rows.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def safe(root, relative):
    root = Path(root).resolve(); rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe relative path')
    path = root / rel
    require(path.resolve().is_relative_to(root), 'Path escapes root')
    require(not any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root)), 'Symlink forbidden')
    return path


def check_file(path, expected, dependencies):
    require(sha(path) == expected, 'Frozen file SHA differs: ' + str(path))
    dependencies[str(Path(path).resolve())] = expected


def copy_file(source, destination, expected):
    require(sha(source) == expected, 'Copy source SHA differs')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst)
    require(sha(destination) == expected, 'Copied SHA differs')


def verify_parent():
    dependencies = {}
    check_file(SPEC, SPEC_SHA, dependencies)
    spec = read(SPEC)
    require(spec['candidate_only'] == ARM and spec['allowed_splits'] == ['train', 'validation']
            and spec['seeds'] == [42, 52, 62] and spec['remote_root'] == REMOTE
            and spec['inherited_absolute_host_deadline'] == DEADLINE, 'Diagnostic spec changed')
    require(time.time() < DEADLINE, 'Inherited absolute deadline expired')
    check_file(TRAIN / 'execution_contract.json', TRAIN_RAW, dependencies)
    training = read(TRAIN / 'execution_contract.json')
    check_file(TRAIN / 'retrieval/final_receipt.json', TRAIN_RECEIPT, dependencies)
    tr = read(TRAIN / 'retrieval/final_receipt.json')
    require(tr['status'] == 'complete' and tr['new_fit_count'] == tr['seed_count'] == 3
            and tr['all_training_terminal_verified'] is True and tr['held_out_test_evaluated'] is False,
            'Original training incomplete')
    for rel, digest in tr['files'].items():
        check_file(safe(PROJECT, rel), digest, dependencies)
    # Metadata receipt SHA only: do not open any Test predictions or Test metric artifact.
    check_file(PARENT / 'final_receipt.json', EVALUATION_RECEIPT, dependencies)
    parent = read(PARENT / 'contract.json')
    require(parent['contract_sha256'] == '837f28048ab35b39ab74fa094a67c6ffe167eb1ec95aac1d802bc052e204c4f9' == canonical({k:v for k,v in parent.items() if k != 'contract_sha256'}),
            'Parent contract canonical SHA differs')
    dependencies[str(PARENT / 'contract.json')] = sha(PARENT / 'contract.json')
    source = parent['sources']['components']
    require(source['closure'] == SOURCE == canonical(source['files']) and len(source['files']) == 141,
            'Source141 closure differs')
    for rel, digest in source['files'].items():
        check_file(safe(PARENT / source['root'], rel), digest, dependencies)
    check_file(BASELINE_DIAG / 'RECEIPT.json', BASELINE_RECEIPT, dependencies)
    br = read(BASELINE_DIAG / 'RECEIPT.json')
    require(br['status'] == 'complete' and br['allowed_splits'] == ['train', 'validation']
            and br['inference_count'] == br['remote_count'] == 0 and br['held_out_test_read'] is False,
            'Saved CNN reference scope differs')
    for rel, digest in br['inputs'].items():
        check_file(safe(PROJECT, rel), digest, dependencies)
    for rel, digest in br['output_files'].items():
        check_file(BASELINE_DIAG / rel, digest, dependencies)
    require({j['seed'] for j in parent['jobs']} == {42,52,62} and len(parent['jobs']) == 3
            and all(j['arm'] == ARM and j['dataset'] == 'raf_spare_parts' for j in parent['jobs']),
            'Current candidate cohort differs')
    for job in parent['jobs']:
        require(job['selected_epoch'] == spec['selected_epochs'][str(job['seed'])], 'Selected epoch differs')
        check_file(safe(PARENT, job['checkpoint']['relative_path']), job['checkpoint']['sha256'], dependencies)
        for name in ('checkpoint_metadata', 'selection_history'):
            check_file(safe(PARENT, job[name+'_path']), job[name+'_sha256'], dependencies)
    for filename in (*LOCAL_OPERATIONS, 'titantpp_inter62_transport_evidence.py'):
        path = PROJECT / 'paper/scripts' / filename
        require(path.is_file(), 'New operation missing: ' + filename)
        dependencies[str(path)] = sha(path)
    for filename in ('test_titantpp_value_difference_rows_operations.py', 'test_titantpp_value_difference_rows.py'):
        path = PROJECT / 'paper/tests' / filename
        if path.exists(): dependencies[str(path)] = sha(path)
    return spec, training, parent, br, dependencies


def prepare():
    require(not BUNDLE.exists(), 'Existing diagnostic bundle: no overwrite or automatic retry')
    spec, training, parent, baseline, dependencies = verify_parent()
    # All prerequisites validated before the first new bundle write.
    dataset = parent['datasets']['raf_spare_parts']
    input_sha = dataset['training_data_sha256']
    require(input_sha == 'd805cad0242078fedfbb87eed18bde9b73d37bfc9b93c6b75aac49087244efa2', 'Pure input differs')
    require(dataset['definition']['inherited_data_identity']['populations']['train']['target_count'] == 25779
            and dataset['definition']['inherited_data_identity']['populations']['validation']['target_count'] == 6690,
            'Population differs')
    check_file(PARENT / dataset['training_data_path'], input_sha, dependencies)
    frozen = PROJECT / 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/operation'
    helpers = [(PARENT/'operation/evaluate.py','helper.py',HELPER_SHA),
               (PARENT/'operation/lineage.json','lineage.json',LINEAGE_SHA),
               (frozen/'row_utils.py','row_utils.py',ROW_UTILS_SHA)]
    for path, _, expected in helpers: check_file(path, expected, dependencies)
    operation_commit=subprocess.run(['git','rev-parse','HEAD'],cwd=PROJECT,capture_output=True,text=True,check=True).stdout.strip()
    require(len(operation_commit)==40,'Operation commit missing')
    for filename in LOCAL_OPERATIONS:
        tracked=subprocess.run(['git','show',operation_commit+':paper/scripts/'+filename],cwd=PROJECT,capture_output=True,check=True).stdout
        require(hashlib.sha256(tracked).hexdigest()==sha(PROJECT/'paper/scripts'/filename),'Operation code must match its committed revision')
    BUNDLE.mkdir(parents=True, exist_ok=False)
    files, inputs, operations = {}, {}, {}
    def add(source, name, expected, category):
        copy_file(source, safe(BUNDLE, name), expected)
        files[name] = expected; category[name] = expected
    for rel, digest in parent['sources']['components']['files'].items():
        add(PARENT/'source'/rel, 'source/'+rel, digest, inputs)
    sentinel=PARENT/'source/sample_data/.frozen_root_sentinel'
    require(sentinel.is_file() and sentinel.stat().st_size == 0, 'Frozen sample root sentinel missing')
    add(sentinel,'source/sample_data/.frozen_root_sentinel',sha(sentinel),inputs)
    add(PARENT/dataset['training_data_path'],'inputs/raf_train_validation.parquet',input_sha,inputs)
    add(SPEC,'provenance/spec.json',SPEC_SHA,inputs)
    add(TRAIN/'retrieval/final_receipt.json','provenance/training_receipt.json',TRAIN_RECEIPT,inputs)
    add(BASELINE_DIAG/'PLAN.json','provenance/paired_regret_plan.json',spec['analysis_plan_sha256'],inputs)
    jobs=[]
    for original in parent['jobs']:
        job=copy.deepcopy(original)
        prefix='inputs/jobs/'+job['id']+'/'
        for field in ('checkpoint_metadata','selection_history'):
            name=prefix+field+'.json'; add(PARENT/original[field+'_path'],name,original[field+'_sha256'],inputs); job[field+'_path']=name
        name=prefix+'selected.pt';add(PARENT/original['checkpoint']['relative_path'],name,original['checkpoint']['sha256'],inputs)
        job['checkpoint']['relative_path']=name
        jobs.append(job)
    for path, name, expected in helpers: add(path,'operation/'+name,expected,operations)
    for filename, dest in [('export_titantpp_value_difference_rows.py','evaluate.py'),
                           ('titantpp_value_difference_rows_runtime.py','runtime.py')]:
        path=PROJECT/'paper/scripts'/filename;add(path,'operation/'+dest,sha(path),operations)
    approval={'authorized':True,'instruction':spec['authorization'],'scope':'current ValueDifference3 Train/Validation row export only',
              'new_training':False,'frozen_selection':True,'allowed_splits':['train','validation'],'historical_CNN_inference':False,'held_out_test_evaluated':False,'checkpoint_reselection':False,
              'remote_root':REMOTE,'absolute_host_deadline':DEADLINE,'automatic_retry':False,'spec_sha256':SPEC_SHA}
    write(BUNDLE/'approval.json',approval);files['approval.json']=inputs['approval.json']=sha(BUNDLE/'approval.json')
    runtime=copy.deepcopy(parent['runtimes']['5090'])
    require(runtime['cudnn']==92000 and runtime['torch_num_threads']==4
            and runtime['gpu_uuid']=='GPU-c9adc246-ef66-7906-bc07-6e152fa9952f','Native runtime differs')
    environment=copy.deepcopy(training['hosts']['5090']['environment'])
    environment.update(PYTHONPATH=REMOTE+'/source',MPLCONFIGDIR=REMOTE+'/runtime_cache/matplotlib',XDG_CACHE_HOME=REMOTE+'/runtime_cache')
    c={'operation_code_commit':operation_commit,'training_source_revision':parent['sources']['components']['revision'],'schema':'titantpp_value_difference_rows_v1','created_unix':time.time(),'allowed_splits':['train','validation'],
       'new_training':False,'frozen_selection':True,'checkpoint_reselection':False,'historical_CNN_inference':False,
       'held_out_test_evaluated':False,'automatic_retry':False,'max_parallel_per_host':3,'host':'5090',
       'remote_root':REMOTE,'deadline_unix':DEADLINE,'worker_timeout_seconds':10800,'pipeline_timeout_seconds':43200,
       'source':copy.deepcopy(parent['sources']['components']),'jobs':jobs,
       'datasets':{'raf_spare_parts':{'path':'inputs/raf_train_validation.parquet','sha256':input_sha,
            'definition':copy.deepcopy(dataset['definition']),'populations':copy.deepcopy(dataset['definition']['inherited_data_identity']['populations'])}},
       'runtimes':{'5090':runtime},'environment':environment,'operation_files':operations,'input_files':inputs,
       'approval_path':'approval.json','approval_sha256':files['approval.json'],'spec_path':'provenance/spec.json','spec_sha256':SPEC_SHA,
       'stages':['qualification','validation','train'],'validation_tolerance':{'rtol':1e-5,'atol':1e-5},
       'baseline_mode':'local saved CNN rows only; no baseline model copied or instantiated',
       'local_bundle':str(BUNDLE)}
    c['contract_sha256']=canonical(c);write(BUNDLE/'contract.json',c);files['contract.json']=sha(BUNDLE/'contract.json')
    # References remain local and are not part of the native execution packet.
    refs={'schema':'stored_cnn_train_validation_reference_pins_v1','source_receipt_sha256':BASELINE_RECEIPT,
          'input_files':baseline['inputs'],'artifact_files':baseline['output_files'],'source_directory':str(BASELINE_DIAG)}
    write(BUNDLE/'baseline_local_references.json',refs);files['baseline_local_references.json']=sha(BUNDLE/'baseline_local_references.json')
    boot={'schema':'titantpp_value_difference_rows_bootstrap_v1','files':files,'external_dependencies':dependencies,
          'contract_raw_sha256':files['contract.json'],'contract_sha256':c['contract_sha256'],
          'spec_sha256':SPEC_SHA,'native_packet_files':{k:v for k,v in files.items() if k!='baseline_local_references.json'},
          'remote_calls':0,'held_out_test_evaluated':False}
    write(BUNDLE/'bootstrap_integrity.json',boot)
    return {'status':'prepared','bundle':str(BUNDLE),'candidate_count':3,'native_stage_count':3,'data_worker_count':6,
            'bootstrap_integrity_sha256':sha(BUNDLE/'bootstrap_integrity.json'),'contract_raw_sha256':files['contract.json'],'remote_calls':0}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare'],default='prepare');args=p.parse_args(argv)
    print(json.dumps(prepare(),ensure_ascii=False))


if __name__=='__main__':main()
