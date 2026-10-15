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
import tarfile

PROJECT = Path(__file__).resolve().parents[2]
NAME = 'titantpp_backbone_key_content_probe_20261016_v1'
BUNDLE = PROJECT / 'search_artifacts' / NAME
TRAIN = PROJECT / 'search_artifacts/titantpp_backbone_qk_location_raf6_20261016_v1'
PARENT = PROJECT / 'search_artifacts/titantpp_backbone_key_history_probe_20261016_v1'
PARENT_RECEIPT = '42951c5ffb69639a438a002f0dd9e70eee1f3d8c234ad87b2ad42861c5041ff7'
PARENT_BOOTSTRAP = '16d88442ddd8fdb381c934dd65ba503f56fdad627cb648232be7c26f810eb86c'
REPORT = PROJECT / 'reports' / NAME
SPEC = REPORT / 'SCIENCE_SPEC.json'
SPEC_SHA = 'f8ce822618bc0dcbacda3512215a9e368afb054a26466616cca1089b12d18678'
ACCEPTANCE = REPORT / 'NEXT_MECHANISM_ACCEPTANCE.json'
ACCEPTANCE_SHA = '6520d428c1cd3c4ac4f211c2c4e89a6b5b570d4efe22cf4c64ff73a948771d2e'
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/' + NAME + '_5090'
ARM = 'titantpp_cnn_key_only_v1'
DEADLINE = 1792679647.915156
SOURCE = 'fc0668016b8eb89752e03d0751e317e77b06d8135c0b82f8b43625e345abf9a3'
TRAIN_RECEIPT = 'ac680e7665de247681a9e223b0f5e635217f342461104c31c10221e2e6c41342'
EVALUATION_RECEIPT = '7325da1da450bbee0dc1c4bb5c50df1130a07d9988bbdae9392993e071ba992c'
TRAIN_RAW = '255fb0b322a71d3bc86568fd99fa10606dd6a8f095e0fa50ca20622be632bed8'
HELPER_SHA = '310363afe86f5ef474cb2cfd555d29d425ce261e783fa28c0aeeaf67c3108445'
ROW_UTILS_SHA = '0ef413db7f6f41d4fe274ea60b8bea7372a5851cc688247406169f1d494b4225'
LINEAGE_SHA = '8e25bc21cc05d97bef302df95ea0101e7e9c64894f1970806b6d9e473898c40a'
LOCAL_OPERATIONS = ('prepare_titantpp_key_content_probe.py', 'advance_titantpp_key_content_probe.py',
                    'titantpp_key_content_probe_runtime.py', 'export_titantpp_key_content_probe.py')


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
    """Read only pinned completed KeyHistory evidence; no prior helper/remote execution."""
    dependencies = {}
    check_file(SPEC, SPEC_SHA, dependencies)
    check_file(ACCEPTANCE, ACCEPTANCE_SHA, dependencies)
    spec = read(SPEC)
    require(read(ACCEPTANCE)['spec_sha256']==SPEC_SHA, 'Acceptance differs')
    require(spec['candidate_only']==ARM and spec['allowed_splits']==['train','validation']
            and spec['seeds']==[42,52,62] and spec['new_remote_root']==REMOTE
            and spec['inherited_absolute_host_deadline']==DEADLINE, 'Diagnostic scope differs')
    require(time.time()<DEADLINE, 'Inherited deadline expired')
    for rel,digest in spec['evidence_pins'].items(): check_file(safe(PROJECT,rel),digest,dependencies)
    check_file(PARENT/'retrieval/final_receipt.json',PARENT_RECEIPT,dependencies)
    check_file(PARENT/'bootstrap_integrity.json',PARENT_BOOTSTRAP,dependencies)
    r=read(PARENT/'retrieval/final_receipt.json'); boot=read(PARENT/'bootstrap_integrity.json')
    require(r['status']=='complete' and r['candidate_count']==3 and r['seed_count']==3
            and r['actual_native_worker_exit0_count']==15 and r['actual_pipeline_exit0_count']==1
            and r['actual_supervisor_child_exit0_count']==1 and r['all_owned_absent'] is True
            and r['held_out_test_evaluated'] is False and r['historical_CNN_inference'] is False,
            'Parent completion scope differs')
    require(len(r['files'])==167 and len(boot['files'])>145,'Parent inventory incomplete')
    for rel,digest in r['files'].items():check_file(safe(PROJECT,rel),digest,dependencies)
    for rel,digest in boot['files'].items():check_file(safe(PARENT,rel),digest,dependencies)
    parent=read(PARENT/'contract.json')
    require(parent['contract_sha256']==r['contract_sha256']==canonical({k:v for k,v in parent.items() if k!='contract_sha256'}), 'Parent canonical differs')
    require(sha(PARENT/'contract.json')==r['contract_raw_sha256']==boot['contract_raw_sha256'], 'Parent raw binding differs')
    require(parent['source']['closure']==SOURCE==canonical(parent['source']['files']) and len(parent['source']['files'])==145, 'Source closure differs')
    originals=PARENT/'retrieval/originals'; manifest=read(originals/'ARCHIVE_MANIFEST.json'); archive=PARENT/'retrieval/originals.tar.gz'
    require(sha(archive)==r['archive_sha256'] and sha(originals/'ARCHIVE_MANIFEST.json')==r['manifest_sha256'], 'Parent archive SHA differs')
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers(); names=[m.name for m in members]
        require(all(m.isfile() for m in members) and len(names)==len(set(names)) and set(names)==set(manifest['files'])|{'ARCHIVE_MANIFEST.json'},'Parent archive inventory differs')
        require(tar.extractfile('ARCHIVE_MANIFEST.json').read()==(originals/'ARCHIVE_MANIFEST.json').read_bytes(),'Parent archive manifest differs')
        for rel,item in manifest['files'].items():
            p=safe(originals,rel); raw=tar.extractfile(rel).read()
            require(hashlib.sha256(raw).hexdigest()==sha(p)==item['sha256'] and len(raw)==p.stat().st_size==item['bytes'], 'Parent archived original differs')
    terminal=read(originals/'runtime/terminal.json')
    require(terminal['status']=='complete' and terminal['all_owned_absent'] is True
            and len(terminal['owner_probes'])==16 and all(x['state']=='absent' and x['explicit_absence'] for x in terminal['owner_probes'])
            and terminal['gpu']['compute_pids']==[] and not terminal['root_owners']['owners'] and not terminal['root_owners']['query_errors'], 'Parent terminal proof differs')
    require(len(parent['jobs'])==3 and {j['seed'] for j in parent['jobs']}=={42,52,62},'Parent cohort differs')
    for job in parent['jobs']:
        require(job['arm']==ARM and job['selected_epoch']==spec['selected_epochs'][str(job['seed'])], 'Selection differs')
        for split in ('train','validation'):
            ref=spec['parent_prediction_references'][job['id']+'/on_'+split]
            for field in ('receipt','manifest'):check_file(safe(PROJECT,ref[field+'_path']),ref[field+'_sha256'],dependencies)
            for part in ref['parts']:check_file(safe(PROJECT,part['path']),part['sha256'],dependencies)
    for filename in (*LOCAL_OPERATIONS, 'compare_titantpp_key_content_probe.py','titantpp_inter62_transport_evidence.py'):
        p=PROJECT/'paper/scripts'/filename;require(p.is_file(),'Operation missing: '+filename);dependencies[str(p)]=sha(p)
    for filename in ('test_titantpp_key_content_probe_ops.py','test_export_titantpp_key_content_probe.py','test_compare_titantpp_key_content_probe.py'):
        p=PROJECT/'paper/tests'/filename;require(p.is_file(),'Test missing: '+filename);dependencies[str(p)]=sha(p)
    return spec, parent, dependencies


def prepare():
    require(not BUNDLE.exists(), 'Existing diagnostic bundle: no overwrite or automatic retry')
    spec, parent, dependencies = verify_parent()
    ready_path=REPORT/'LOCAL_READY.json';ready=read(ready_path)
    require(ready.get('status')=='ready_for_native_diagnostic' and ready.get('native_started') is False
            and ready.get('spec_sha256')==SPEC_SHA and ready.get('acceptance_sha256')==ACCEPTANCE_SHA
            and isinstance(ready.get('files'),dict) and len(ready['files'])>=12,
            'Verified local and independent readiness evidence required')
    for rel,digest in ready['files'].items():check_file(safe(PROJECT,rel),digest,dependencies)
    require(all(str((PROJECT/'paper/scripts'/n).relative_to(PROJECT)) in ready['files']
                for n in (*LOCAL_OPERATIONS,'compare_titantpp_key_content_probe.py')), 'Readiness operation coverage missing')
    check_file(ready_path,sha(ready_path),dependencies)
    # All prerequisites validated before the first new bundle write.
    dataset = parent['datasets']['raf_spare_parts']
    input_sha = dataset['sha256']
    require(input_sha == 'd805cad0242078fedfbb87eed18bde9b73d37bfc9b93c6b75aac49087244efa2', 'Pure input differs')
    require(dataset['definition']['inherited_data_identity']['populations']['train']['target_count'] == 25779
            and dataset['definition']['inherited_data_identity']['populations']['validation']['target_count'] == 6690,
            'Population differs')
    check_file(PARENT / dataset['path'], input_sha, dependencies)
    frozen = PROJECT / 'search_artifacts/titantpp_cnn_memory_sensitivity_20261015_v3/operation'
    helpers = [(PARENT/'operation/helper.py','helper.py',HELPER_SHA),
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
    for rel, digest in parent['source']['files'].items():
        add(PARENT/'source'/rel, 'source/'+rel, digest, inputs)
    sentinel=PARENT/'source/sample_data/.frozen_root_sentinel'
    require(sentinel.is_file() and sentinel.stat().st_size == 0, 'Frozen sample root sentinel missing')
    add(sentinel,'source/sample_data/.frozen_root_sentinel',sha(sentinel),inputs)
    add(PARENT/dataset['path'],'inputs/raf_train_validation.parquet',input_sha,inputs)
    add(SPEC,'provenance/spec.json',SPEC_SHA,inputs)
    add(PARENT/'retrieval/final_receipt.json','provenance/parent_native_receipt.json',PARENT_RECEIPT,inputs)
    add(PARENT/'contract.json','provenance/parent_contract.json',sha(PARENT/'contract.json'),inputs)
    add(ACCEPTANCE,'provenance/acceptance.json',ACCEPTANCE_SHA,inputs)
    jobs=[]
    for original in (j for j in parent['jobs'] if j['arm'] == ARM):
        job=copy.deepcopy(original)
        prefix='inputs/jobs/'+job['id']+'/'
        for field in ('checkpoint_metadata','selection_history'):
            name=prefix+field+'.json'; add(PARENT/original[field+'_path'],name,original[field+'_sha256'],inputs); job[field+'_path']=name
        name=prefix+'selected.pt';add(PARENT/original['checkpoint']['relative_path'],name,original['checkpoint']['sha256'],inputs)
        job['checkpoint']['relative_path']=name
        jobs.append(job)
    references={}
    for job in jobs:
        references[job['id']]={}
        for split in ('train','validation'):
            ref=spec['parent_prediction_references'][job['id']+'/on_'+split]
            prefix='inputs/references/'+job['id']+'/on_'+split+'/'
            add(PROJECT/ref['receipt_path'],prefix+'receipt.json',ref['receipt_sha256'],inputs)
            add(PROJECT/ref['manifest_path'],prefix+'manifest.json',ref['manifest_sha256'],inputs)
            original_manifest=read(PROJECT/ref['manifest_path'])
            for part in original_manifest['parts']:
                src=safe((PROJECT/ref['manifest_path']).parent,part['path'])
                add(src,prefix+part['path'],part['sha256'],inputs)
            references[job['id']][split]={'receipt_path':prefix+'receipt.json','receipt_sha256':ref['receipt_sha256'],
                 'manifest_path':prefix+'manifest.json','manifest_sha256':ref['manifest_sha256']}
    for path, name, expected in helpers: add(path,'operation/'+name,expected,operations)
    for filename, dest in [('export_titantpp_key_content_probe.py','evaluate.py'),
                           ('titantpp_key_content_probe_runtime.py','runtime.py')]:
        path=PROJECT/'paper/scripts'/filename;add(path,'operation/'+dest,sha(path),operations)
    approval={'authorized':True,'instruction':spec['authorization'],'scope':'current KeyOnly3 available past content replacement with current K, same gain, Train/Validation only',
              'conditions':['on','off'],'new_training':False,'frozen_selection':True,'allowed_splits':['train','validation'],'historical_CNN_inference':False,'held_out_test_evaluated':False,'checkpoint_reselection':False,
              'remote_root':REMOTE,'absolute_host_deadline':DEADLINE,'automatic_retry':False,'spec_sha256':SPEC_SHA,'acceptance_sha256':ACCEPTANCE_SHA}
    write(BUNDLE/'approval.json',approval);files['approval.json']=inputs['approval.json']=sha(BUNDLE/'approval.json')
    runtime=copy.deepcopy(parent['runtimes']['5090'])
    require(runtime['cudnn']==92000 and runtime['torch_num_threads']==4
            and runtime['gpu_uuid']=='GPU-c9adc246-ef66-7906-bc07-6e152fa9952f','Native runtime differs')
    environment=copy.deepcopy(parent['environment'])
    environment.update(PYTHONPATH=REMOTE+'/source',MPLCONFIGDIR=REMOTE+'/runtime_cache/matplotlib',XDG_CACHE_HOME=REMOTE+'/runtime_cache')
    c={'operation_code_commit':operation_commit,'training_source_revision':parent['source']['revision'],'schema':'titantpp_key_content_probe_v1','created_unix':time.time(),'allowed_splits':['train','validation'],
       'new_training':False,'frozen_selection':True,'checkpoint_reselection':False,'historical_CNN_inference':False,
       'held_out_test_evaluated':False,'automatic_retry':False,'max_parallel_per_host':3,'host':'5090',
       'remote_root':REMOTE,'deadline_unix':DEADLINE,'worker_timeout_seconds':10800,'pipeline_timeout_seconds':43200,
       'source':copy.deepcopy(parent['source']),'jobs':jobs,
       'datasets':{'raf_spare_parts':{'path':'inputs/raf_train_validation.parquet','sha256':input_sha,
            'definition':copy.deepcopy(dataset['definition']),'populations':copy.deepcopy(dataset['definition']['inherited_data_identity']['populations'])}},
       'runtimes':{'5090':runtime},'environment':environment,'operation_files':operations,'input_files':inputs,
       'approval_path':'approval.json','approval_sha256':files['approval.json'],'spec_path':'provenance/spec.json','spec_sha256':SPEC_SHA,
       'intervention':'replace_available_past_key_content_with_current_preserve_gain','content_multipliers':[1,0],
       'reference_rows':references,'new_on_train_count':0,'reused_on_train_count':3,
       'parent_native_receipt_path':'provenance/parent_native_receipt.json','parent_native_receipt_sha256':PARENT_RECEIPT,
       'parent_contract_path':'provenance/parent_contract.json','parent_contract_raw_sha256':sha(PARENT/'contract.json'),'parent_contract_sha256':parent['contract_sha256'],
       'conditions':['on','off'],'stages':['qualification','on_validation','off_train','off_validation'],
       'acceptance_path':'provenance/acceptance.json','acceptance_sha256':ACCEPTANCE_SHA,
       'parent_evaluation_receipt_sha256':EVALUATION_RECEIPT,'validation_tolerance':{'rtol':1e-5,'atol':1e-5},
       'baseline_mode':'Same current KeyOnly candidate content on/off only; stored onTrain3 reused, no historical CNN input or inference',
       'local_bundle':str(BUNDLE)}
    c['contract_sha256']=canonical(c);write(BUNDLE/'contract.json',c);files['contract.json']=sha(BUNDLE/'contract.json')
    boot={'schema':'titantpp_key_content_probe_bootstrap_v1','files':files,'external_dependencies':dependencies,
          'contract_raw_sha256':files['contract.json'],'contract_sha256':c['contract_sha256'],
          'spec_sha256':SPEC_SHA,'acceptance_sha256':ACCEPTANCE_SHA,'native_packet_files':files.copy(),
          'remote_calls':0,'held_out_test_evaluated':False}
    write(BUNDLE/'bootstrap_integrity.json',boot)
    return {'status':'prepared','bundle':str(BUNDLE),'candidate_count':3,'native_stage_count':4,'data_worker_count':9,'new_on_train_count':0,'reused_on_train_count':3,
            'bootstrap_integrity_sha256':sha(BUNDLE/'bootstrap_integrity.json'),'contract_raw_sha256':files['contract.json'],'remote_calls':0}


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['prepare'],default='prepare');args=p.parse_args(argv)
    print(json.dumps(prepare(),ensure_ascii=False))


if __name__=='__main__':main()
