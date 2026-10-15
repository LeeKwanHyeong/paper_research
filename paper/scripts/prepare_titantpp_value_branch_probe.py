"""Prepare one current-candidate V-branch sensitivity diagnostic, locally only."""
from __future__ import annotations
import argparse,copy,hashlib,json,shutil,subprocess,time
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[2]
NAME='titantpp_backbone_value_difference_branch_probe_20261016_v1'
BUNDLE=PROJECT/'search_artifacts'/NAME
PARENT=PROJECT/'search_artifacts/titantpp_backbone_value_difference_rows_20261016_v1'
REPORT=PROJECT/'reports/titantpp_backbone_value_difference_raf3_20261016_v1'
SPEC=REPORT/'NEXT_MECHANISM_SPEC.json';ACCEPTANCE=REPORT/'NEXT_MECHANISM_ACCEPTANCE.json'
SPEC_SHA='d509654b763c50017ef0e0100974b86e839af12db6103c95aac979f27ce66415'
ACCEPTANCE_SHA='ae9aac806efd68069daf63b5cc725c6512c46e9812198966b2445ddca44d63e8'
PARENT_RECEIPT_SHA='326d793b61806a32435bea1536746e2b18ec2f756a818dc89bafdb6fd631445c'
PARENT_CONTRACT_SHA='8f5a4ed206ff36297b29f7fdc1d3191ba536301213d72764bd8777e26b8793dc'
PARENT_BOOTSTRAP_SHA='e009b432b29e07649289e558feff7cc806c00b0d1f2cd0f9a6cb1fbeb21851a5'
REMOTE='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/'+NAME+'_5090'
SOURCE='4fce860e41ee716592abf168af8c3985859b74ea6d080357690124b19a88d361'
DEADLINE=1792679647.915156
ARM='titantpp_cnn_value_difference_v1'
LOCAL_OPERATIONS=('prepare_titantpp_value_branch_probe.py','advance_titantpp_value_branch_probe.py',
                  'titantpp_value_branch_probe_runtime.py','export_titantpp_value_branch_probe.py')


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def safe(root,name):
    root=Path(root).resolve();rel=Path(name);p=root/rel
    require(not rel.is_absolute() and rel.parts and '..' not in rel.parts and p.resolve().is_relative_to(root),'Unsafe relative artifact path')
    require(not any(x.is_symlink() for x in (p,*p.parents) if x.is_relative_to(root)),'Symlink forbidden')
    return p


def check_file(path,expected,deps):
    require(sha(path)==expected,'Frozen file SHA differs: '+str(path));deps[str(Path(path).resolve())]=expected


def copy_file(src,dst,expected):
    require(sha(src)==expected,'Copy source SHA differs');dst.parent.mkdir(parents=True,exist_ok=True)
    with Path(src).open('rb') as source,dst.open('xb') as destination:shutil.copyfileobj(source,destination)
    require(sha(dst)==expected,'Copied bytes differ')


def verify_parent():
    deps={};check_file(SPEC,SPEC_SHA,deps);check_file(ACCEPTANCE,ACCEPTANCE_SHA,deps)
    spec=read(SPEC);acceptance=read(ACCEPTANCE)
    require(spec['candidate_only']==ARM and spec['allowed_splits']==['train','validation'] and spec['seeds']==[42,52,62]
            and spec['new_remote_root']==REMOTE and spec['inherited_absolute_host_deadline']==DEADLINE
            and spec['reference_rows_receipt_sha256']==PARENT_RECEIPT_SHA and acceptance['spec_sha256']==SPEC_SHA,'New specification differs')
    require(time.time()<DEADLINE,'Inherited deadline expired')
    check_file(PARENT/'retrieval/final_receipt.json',PARENT_RECEIPT_SHA,deps);receipt=read(PARENT/'retrieval/final_receipt.json')
    require(receipt['status']=='complete' and receipt['candidate_count']==receipt['seed_count']==3
            and receipt['qualification_count']==3 and receipt['data_worker_count']==6 and receipt['actual_native_worker_exit0_count']==9
            and receipt['actual_pipeline_exit0_count']==1 and receipt['all_owned_absent'] is True
            and receipt['new_training'] is False and receipt['held_out_test_evaluated'] is False and receipt['historical_CNN_inference'] is False,'Parent row export incomplete')
    for rel,digest in receipt['files'].items():check_file(safe(PROJECT,rel),digest,deps)
    check_file(PARENT/'contract.json',PARENT_CONTRACT_SHA,deps);parent=read(PARENT/'contract.json')
    require(parent['contract_sha256']==receipt['contract_sha256']==canonical({k:v for k,v in parent.items() if k!='contract_sha256'}),'Parent canonical differs')
    check_file(PARENT/'bootstrap_integrity.json',PARENT_BOOTSTRAP_SHA,deps)
    require(parent['source']['closure']==SOURCE==canonical(parent['source']['files']) and len(parent['source']['files'])==141,'Source141 differs')
    # Current candidate source/input/selection and operational receipts only. No historical CNN checkpoint or rows are read.
    for rel,digest in parent['input_files'].items():check_file(safe(PARENT,rel),digest,deps)
    for rel,digest in parent['operation_files'].items():check_file(safe(PARENT,rel),digest,deps)
    require(len(parent['jobs'])==3 and {j['seed'] for j in parent['jobs']}=={42,52,62}
            and all(j['arm']==ARM and j['selected_epoch']==spec['selected_epochs'][str(j['seed'])] for j in parent['jobs']),'Selected current cohort differs')
    originals=safe(PROJECT,receipt['originals_path'])
    manifest=read(originals/'ARCHIVE_MANIFEST.json')
    require(sha(PARENT/'retrieval/originals.tar.gz')==receipt['archive_sha256'] and sha(originals/'ARCHIVE_MANIFEST.json')==receipt['manifest_sha256'],'Parent archive binding differs')
    for rel,item in manifest['files'].items():check_file(safe(originals,rel),item['sha256'],deps)
    references={}
    for j in parent['jobs']:
        references[j['id']]={}
        for split in ('train','validation'):
            item=receipt['data_receipts'][j['id']+'/'+split];rpath=safe(originals,item['path']);check_file(rpath,item['sha256'],deps);r=read(rpath)
            require(r['status']=='complete' and r['job_id']==j['id'] and r['split']==split and r['checkpoint_sha256']==j['checkpoint']['sha256']
                    and r['state_sha256_before']==r['state_sha256_after']==j['checkpoint_state_sha256']
                    and r['population']==parent['datasets']['raf_spare_parts']['populations'][split],'Reference current rows differ')
            mp=safe(originals,r['manifest_path']);check_file(mp,r['manifest_sha256'],deps);m=read(mp)
            require(m['prediction_rows']==r['population']['target_count'] and m['job_id']==j['id'] and m['split']==split,'Reference manifest identity differs')
            for part in m['parts']:check_file(safe(mp.parent,part['path']),part['sha256'],deps)
            references[j['id']][split]={'receipt_path':item['path'],'receipt_sha256':item['sha256'],'manifest_path':r['manifest_path'],'manifest_sha256':r['manifest_sha256']}
    for filename in (*LOCAL_OPERATIONS,'titantpp_inter62_transport_evidence.py'):
        path=PROJECT/'paper/scripts'/filename;require(path.is_file(),'Operation missing: '+filename);deps[str(path)]=sha(path)
    for filename in ('test_titantpp_value_branch_probe_ops.py','test_titantpp_value_branch_probe.py'):
        path=PROJECT/'paper/tests'/filename
        if path.is_file():deps[str(path)]=sha(path)
    return spec,parent,receipt,originals,references,deps


def prepare():
    require(not BUNDLE.exists(),'Existing diagnostic bundle; no overwrite/retry')
    spec,parent,receipt,originals,references,deps=verify_parent()
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=PROJECT,capture_output=True,text=True,check=True).stdout.strip()
    require(len(commit)==40,'Operation commit missing')
    for filename in LOCAL_OPERATIONS:
        content=subprocess.run(['git','show',commit+':paper/scripts/'+filename],cwd=PROJECT,capture_output=True,check=True).stdout
        require(hashlib.sha256(content).hexdigest()==sha(PROJECT/'paper/scripts'/filename),'Operation must match committed revision')
    BUNDLE.mkdir(parents=True,exist_ok=False);files={};inputs={};operations={}
    def add(src,rel,digest,category):
        copy_file(src,safe(BUNDLE,rel),digest);files[rel]=digest;category[rel]=digest
    for rel,digest in parent['source']['files'].items():add(PARENT/parent['source']['root']/rel,'source/'+rel,digest,inputs)
    sentinel=PARENT/'source/sample_data/.frozen_root_sentinel';require(sentinel.stat().st_size==0,'Source sentinel differs')
    add(sentinel,'source/sample_data/.frozen_root_sentinel',sha(sentinel),inputs)
    dataset=copy.deepcopy(parent['datasets']['raf_spare_parts']);add(PARENT/dataset['path'],dataset['path'],dataset['sha256'],inputs)
    jobs=copy.deepcopy(parent['jobs'])
    for job in jobs:
        for key,hashkey in (('checkpoint_metadata_path','checkpoint_metadata_sha256'),('selection_history_path','selection_history_sha256')):
            add(PARENT/job[key],job[key],job[hashkey],inputs)
        add(PARENT/job['checkpoint']['relative_path'],job['checkpoint']['relative_path'],job['checkpoint']['sha256'],inputs)
    add(SPEC,'provenance/spec.json',SPEC_SHA,inputs);add(ACCEPTANCE,'provenance/acceptance.json',ACCEPTANCE_SHA,inputs)
    add(PARENT/'retrieval/final_receipt.json','provenance/current_rows_receipt.json',PARENT_RECEIPT_SHA,inputs)
    remapped=copy.deepcopy(references)
    for ident,splits in references.items():
        for split,ref in splits.items():
            for key in ('receipt','manifest'):
                old=ref[key+'_path'];new='references/current_rows/'+old
                add(originals/old,new,ref[key+'_sha256'],inputs);remapped[ident][split][key+'_path']=new
            manifest=read(originals/ref['manifest_path'])
            for part in manifest['parts']:
                old=str(Path(ref['manifest_path']).parent/part['path']);add(originals/old,'references/current_rows/'+old,part['sha256'],inputs)
    for filename in ('helper.py','row_utils.py','lineage.json'):
        rel='operation/'+filename;add(PARENT/rel,rel,parent['operation_files'][rel],operations)
    for src,dst in (('export_titantpp_value_branch_probe.py','evaluate.py'),('titantpp_value_branch_probe_runtime.py','runtime.py')):
        path=PROJECT/'paper/scripts'/src;add(path,'operation/'+dst,sha(path),operations)
    approval={'authorized':True,'instruction':spec['authorization'],'scope':'current selected candidate3 V residual on/off Train/Validation probe only',
              'allowed_splits':['train','validation'],'conditions':['on','off'],'new_training':False,'frozen_selection':True,
              'historical_CNN_inference':False,'held_out_test_evaluated':False,'checkpoint_reselection':False,'automatic_retry':False,
              'remote_root':REMOTE,'absolute_host_deadline':DEADLINE,'spec_sha256':SPEC_SHA,'acceptance_sha256':ACCEPTANCE_SHA}
    write(BUNDLE/'approval.json',approval);files['approval.json']=inputs['approval.json']=sha(BUNDLE/'approval.json')
    environment=copy.deepcopy(parent['environment']);environment.update(PYTHONPATH=REMOTE+'/source',MPLCONFIGDIR=REMOTE+'/runtime_cache/matplotlib',XDG_CACHE_HOME=REMOTE+'/runtime_cache')
    c={'schema':'titantpp_value_branch_probe_v1','created_unix':time.time(),'operation_code_commit':commit,'training_source_revision':parent['source']['revision'],
       'allowed_splits':['train','validation'],'conditions':['on','off'],'stages':['qualification','on_validation','off_validation','off_train'],
       'new_training':False,'frozen_selection':True,'checkpoint_reselection':False,'historical_CNN_inference':False,'held_out_test_evaluated':False,
       'automatic_retry':False,'max_parallel_per_host':3,'host':'5090','remote_root':REMOTE,'deadline_unix':DEADLINE,'worker_timeout_seconds':10800,'pipeline_timeout_seconds':43200,
       'source':copy.deepcopy(parent['source']),'jobs':jobs,'datasets':{'raf_spare_parts':dataset},'runtimes':copy.deepcopy(parent['runtimes']),
       'environment':environment,'operation_files':operations,'input_files':inputs,'reference_rows':remapped,
       'spec_path':'provenance/spec.json','spec_sha256':SPEC_SHA,'acceptance_path':'provenance/acceptance.json','acceptance_sha256':ACCEPTANCE_SHA,
       'approval_path':'approval.json','approval_sha256':files['approval.json'],'parent_rows_receipt_path':'provenance/current_rows_receipt.json',
       'parent_rows_receipt_sha256':PARENT_RECEIPT_SHA,'row_replay_tolerance':{'relative':1e-5,'absolute':1e-5},
       'validation_tolerance':{'rtol':1e-5,'atol':1e-5},'local_bundle':str(BUNDLE),
       'baseline_mode':'Stored same current-candidate on rows only; no old CNN input to this diagnostic.'}
    c['contract_sha256']=canonical(c);write(BUNDLE/'contract.json',c);files['contract.json']=sha(BUNDLE/'contract.json')
    boot={'schema':'titantpp_value_branch_probe_bootstrap_v1','files':files,'external_dependencies':deps,'contract_raw_sha256':files['contract.json'],
          'contract_sha256':c['contract_sha256'],'spec_sha256':SPEC_SHA,'acceptance_sha256':ACCEPTANCE_SHA,'native_packet_files':files.copy(),
          'remote_calls':0,'held_out_test_evaluated':False}
    write(BUNDLE/'bootstrap_integrity.json',boot)
    return {'status':'prepared','bundle':str(BUNDLE),'candidate_count':3,'native_stage_count':4,'data_worker_count':9,
            'bootstrap_integrity_sha256':sha(BUNDLE/'bootstrap_integrity.json'),'contract_raw_sha256':files['contract.json'],'remote_calls':0}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare'],default='prepare');parser.parse_args()
    print(json.dumps(prepare(),ensure_ascii=False))


if __name__=='__main__':main()
