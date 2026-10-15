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
NAME = 'titantpp_backbone_qk_branch_probe_20261016_v1'
BUNDLE = PROJECT / 'search_artifacts' / NAME
TRAIN = PROJECT / 'search_artifacts/titantpp_backbone_qk_only_raf3_20261016_v1'
PARENT = TRAIN / 'evaluation_v1'
REPORT = PROJECT / 'reports' / NAME
SPEC = REPORT / 'SCIENCE_SPEC.json'
SPEC_SHA = '10371800c7e0f4095defad1380344a5bef19a9b31578dd599bc163779b1af7ee'
ACCEPTANCE = REPORT / 'ACCEPTANCE.json'
ACCEPTANCE_SHA = '5bc011e4473f71db5fd520af8ff3763753c767ceeb98d12cfdc1a6f2c4063193'
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/' + NAME + '_5090'
ARM = 'titantpp_cnn_qk_only_v1'
DEADLINE = 1792679647.915156
SOURCE = '74e79d12538ac005665043fd20f875a9883a436c6a5d02f2768cfd538a8566f7'
TRAIN_RECEIPT = 'e16e983b571a271a8bc86abe82136023e863e0d8cfdc7caa15fe9d8a1bbf059e'
EVALUATION_RECEIPT = 'f3c6823b69a5ad8f0e3faea727828f291563d88dba0082a4724a2fd263e83f1a'
TRAIN_RAW = '46a14ffcd674b6c37f5b7d103832e0bdcca232e332faa67da5b47a020e1f1a9b'
HELPER_SHA = '82a9ae8dfc95aaaaf2a773bb9a0cd80f9f212ef04ba9e51217954b0f9f51b2f6'
ROW_UTILS_SHA = '0ef413db7f6f41d4fe274ea60b8bea7372a5851cc688247406169f1d494b4225'
LINEAGE_SHA = '09cc750ab59db1dda82c3f9409240a2e49e3d953327bbaeabb0df461d271d8bf'
LOCAL_OPERATIONS = ('prepare_titantpp_qk_branch_probe.py', 'advance_titantpp_qk_branch_probe.py',
                    'titantpp_qk_branch_probe_runtime.py', 'export_titantpp_qk_branch_probe.py')


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
    check_file(ACCEPTANCE, ACCEPTANCE_SHA, dependencies)
    require(read(ACCEPTANCE)['spec_sha256']==SPEC_SHA,'Acceptance specification differs')
    spec = read(SPEC)
    require(spec['candidate_only'] == ARM and spec['allowed_splits'] == ['train', 'validation']
            and spec['seeds'] == [42, 52, 62] and spec['new_remote_root'] == REMOTE
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
    require(parent['contract_sha256'] == '47cf557d6f0d526088ac4af11bd2d43fb188f3789dac308a71ed22a4fae526ef' == canonical({k:v for k,v in parent.items() if k != 'contract_sha256'}),
            'Parent contract canonical SHA differs')
    dependencies[str(PARENT / 'contract.json')] = sha(PARENT / 'contract.json')
    source = parent['sources']['components']
    require(source['closure'] == SOURCE == canonical(source['files']) and len(source['files']) == 143,
            'Source143 closure differs')
    for rel, digest in source['files'].items():
        check_file(safe(PARENT / source['root'], rel), digest, dependencies)
    require({j['seed'] for j in parent['jobs']} == {42,52,62} and len(parent['jobs']) == 3
            and all(j['arm'] == ARM and j['dataset'] == 'raf_spare_parts' for j in parent['jobs']),
            'Current candidate cohort differs')
    for job in parent['jobs']:
        require(job['selected_epoch'] == spec['selected_epochs'][str(job['seed'])], 'Selected epoch differs')
        check_file(safe(PARENT, job['checkpoint']['relative_path']), job['checkpoint']['sha256'], dependencies)
        for name in ('checkpoint_metadata', 'selection_history'):
            check_file(safe(PARENT, job[name+'_path']), job[name+'_sha256'], dependencies)
    for filename in (*LOCAL_OPERATIONS, 'compare_titantpp_qk_branch_probe.py', 'titantpp_inter62_transport_evidence.py'):
        path = PROJECT / 'paper/scripts' / filename
        require(path.is_file(), 'New operation missing: ' + filename)
        dependencies[str(path)] = sha(path)
    for filename in ('test_titantpp_qk_branch_probe_ops.py', 'test_export_titantpp_qk_branch_probe.py', 'test_compare_titantpp_qk_branch_probe.py'):
        path = PROJECT / 'paper/tests' / filename
        if path.exists(): dependencies[str(path)] = sha(path)
    return spec, training, parent, dependencies


def prepare():
    require(not BUNDLE.exists(), 'Existing diagnostic bundle: no overwrite or automatic retry')
    spec, training, parent, dependencies = verify_parent()
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
    add(ACCEPTANCE,'provenance/acceptance.json',ACCEPTANCE_SHA,inputs)
    add(PARENT/'final_receipt.json','provenance/evaluation_receipt.json',EVALUATION_RECEIPT,inputs)
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
    for filename, dest in [('export_titantpp_qk_branch_probe.py','evaluate.py'),
                           ('titantpp_qk_branch_probe_runtime.py','runtime.py')]:
        path=PROJECT/'paper/scripts'/filename;add(path,'operation/'+dest,sha(path),operations)
    approval={'authorized':True,'instruction':spec['authorization'],'scope':'current QKOnly3 simultaneous Q/K residual on/off Train/Validation probe only',
              'conditions':['on','off'],'new_training':False,'frozen_selection':True,'allowed_splits':['train','validation'],'historical_CNN_inference':False,'held_out_test_evaluated':False,'checkpoint_reselection':False,
              'remote_root':REMOTE,'absolute_host_deadline':DEADLINE,'automatic_retry':False,'spec_sha256':SPEC_SHA,'acceptance_sha256':ACCEPTANCE_SHA}
    write(BUNDLE/'approval.json',approval);files['approval.json']=inputs['approval.json']=sha(BUNDLE/'approval.json')
    runtime=copy.deepcopy(parent['runtimes']['5090'])
    require(runtime['cudnn']==92000 and runtime['torch_num_threads']==4
            and runtime['gpu_uuid']=='GPU-c9adc246-ef66-7906-bc07-6e152fa9952f','Native runtime differs')
    environment=copy.deepcopy(training['hosts']['5090']['environment'])
    environment.update(PYTHONPATH=REMOTE+'/source',MPLCONFIGDIR=REMOTE+'/runtime_cache/matplotlib',XDG_CACHE_HOME=REMOTE+'/runtime_cache')
    c={'operation_code_commit':operation_commit,'training_source_revision':parent['sources']['components']['revision'],'schema':'titantpp_qk_branch_probe_v1','created_unix':time.time(),'allowed_splits':['train','validation'],
       'new_training':False,'frozen_selection':True,'checkpoint_reselection':False,'historical_CNN_inference':False,
       'held_out_test_evaluated':False,'automatic_retry':False,'max_parallel_per_host':3,'host':'5090',
       'remote_root':REMOTE,'deadline_unix':DEADLINE,'worker_timeout_seconds':10800,'pipeline_timeout_seconds':43200,
       'source':copy.deepcopy(parent['sources']['components']),'jobs':jobs,
       'datasets':{'raf_spare_parts':{'path':'inputs/raf_train_validation.parquet','sha256':input_sha,
            'definition':copy.deepcopy(dataset['definition']),'populations':copy.deepcopy(dataset['definition']['inherited_data_identity']['populations'])}},
       'runtimes':{'5090':runtime},'environment':environment,'operation_files':operations,'input_files':inputs,
       'approval_path':'approval.json','approval_sha256':files['approval.json'],'spec_path':'provenance/spec.json','spec_sha256':SPEC_SHA,
       'conditions':['on','off'],'stages':['qualification','on_validation','on_train','off_validation','off_train'],
       'acceptance_path':'provenance/acceptance.json','acceptance_sha256':ACCEPTANCE_SHA,
       'parent_evaluation_receipt_sha256':EVALUATION_RECEIPT,'validation_tolerance':{'rtol':1e-5,'atol':1e-5},
       'baseline_mode':'Same current QKOnly candidate on/off only; no historical CNN input or inference',
       'local_bundle':str(BUNDLE)}
    c['contract_sha256']=canonical(c);write(BUNDLE/'contract.json',c);files['contract.json']=sha(BUNDLE/'contract.json')
    boot={'schema':'titantpp_qk_branch_probe_bootstrap_v1','files':files,'external_dependencies':dependencies,
          'contract_raw_sha256':files['contract.json'],'contract_sha256':c['contract_sha256'],
          'spec_sha256':SPEC_SHA,'acceptance_sha256':ACCEPTANCE_SHA,'native_packet_files':files.copy(),
          'remote_calls':0,'held_out_test_evaluated':False}
    write(BUNDLE/'bootstrap_integrity.json',boot)
    return {'status':'prepared','bundle':str(BUNDLE),'candidate_count':3,'native_stage_count':5,'data_worker_count':12,
            'bootstrap_integrity_sha256':sha(BUNDLE/'bootstrap_integrity.json'),'contract_raw_sha256':files['contract.json'],'remote_calls':0}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare'],default='prepare');args=p.parse_args(argv)
    print(json.dumps(prepare(),ensure_ascii=False))


if __name__=='__main__':main()
