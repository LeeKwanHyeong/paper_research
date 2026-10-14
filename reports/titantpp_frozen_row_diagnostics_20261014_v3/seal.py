"""Seal approved diagnostic operation and construct isolated host packets."""
import hashlib
import io
import json
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = Path(__file__).resolve().parent
B = ROOT / "search_artifacts/titantpp_frozen_row_diagnostics_20261014_v3"
PREFIX = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_frozen_row_diagnostics_20261014_v3_"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, value):
    with p.open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def main():
    bindings = json.loads((B / 'bindings.json').read_text())
    native_training_path=ROOT/'search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1/provenance/training_contract.json'
    native_training=json.loads(native_training_path.read_text())
    approval = json.loads((B / 'approval.json').read_text())
    approval.update(authorized=True, frozen_selection=True, allowed_splits=['train','validation','test'],
        original_approval_sha256=sha(B / 'approval.json'))
    write(B / 'execution_approval.json', approval)
    evaluator = ROOT / 'paper/scripts/evaluate_titantpp_frozen_rows.py'
    previous=ROOT/'search_artifacts/titantpp_frozen_row_diagnostics_20261014_v2/native/5080'
    reused={}
    for p in previous.glob('results/*/validation/receipt.json'):
        rr=json.loads(p.read_text());ex=p.parent.parent/'validation_worker_exit.json'
        assert rr['status']=='complete' and rr['family']=='component' and json.loads(ex.read_text())['exit_code']==0
        reused[rr['job_id']]={'receipt_sha256':sha(p),'worker_exit_sha256':sha(ex),'contract_sha256':rr['contract_raw_sha256']}
    assert len(reused)==14
    markers=sorted({'external_parent/'+j['source_bundle']['source_root']+'/sample_data' for j in bindings['jobs'] if j['family']=='external'})
    operations = {'operation/evaluate.py':evaluator, 'operation/pipeline.py':R/'pipeline.py',
        'operation/supervise.py':R/'supervise.py'}
    for host in ['5080']:
        jobs = [dict(j) for j in bindings['jobs'] if j['host']==host]
        remote_root = PREFIX+host
        for j in jobs:
            j['parent_root'] = j['parent_root'].replace('{REMOTE_ROOT}',remote_root)
        c={'schema':'titantpp_frozen_row_diagnostics_v1','host':host,'jobs':jobs,
            'new_training':False,'frozen_selection':True,'checkpoint_reselection':False,
            'allowed_splits':['train','validation','test'],'automatic_retry':False,
            'authorized_remote_root':remote_root,'runtime':bindings['runtimes'][host],
            'approval':{'path':'execution_approval.json','sha256':sha(B/'execution_approval.json')},
            'evaluator_sha256':sha(evaluator),'pipeline_sha256':sha(R/'pipeline.py'),
            'operation_files':{n:sha(p) for n,p in operations.items()},
            'deadline_unix':time.time()+24*3600,'worker_deadline_seconds':3600,'parallel_workers':2,
            'metric_tolerance':bindings['metric_tolerance'],'local_bindings_sha256':sha(B/'bindings.json'),
            'scope':'fixed selected models diagnostic inference only; no model/source/input/Runtime changes',
            'process_environment':{k:v for k,v in native_training['hosts'][host]['environment'].items() if k in {'LD_LIBRARY_PATH','CUBLAS_WORKSPACE_CONFIG','CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','MKL_NUM_THREADS','NVIDIA_TF32_OVERRIDE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONHASHSEED','PYTHONDONTWRITEBYTECODE'}},
            'native_environment_source_sha256':sha(native_training_path),
            'nvrtc_builtins_sha256':'91dcd944d01da9c0f08fff5d779db136a47f6d62bdb63bae900d2f481e92c3a2',
            'external_Test_reinference':False,'test_is_independent':False, 'reused_validation':reused, 'external_source_layout_marker_dirs':markers}
        cp=B/f'contract_{host}.json';write(cp,c)
        files={**operations,'contract.json':cp,'execution_approval.json':B/'execution_approval.json'}
        for jid in reused:
            folder=previous/'results'/jid/'validation'
            for path in folder.rglob('*'):
                if path.is_file() and 'runtime_cache' not in path.parts:files[path.relative_to(previous).as_posix()]=path
            ep=previous/'results'/jid/'validation_worker_exit.json';files[ep.relative_to(previous).as_posix()]=ep
        if host=='5080':
            manifest=json.loads((B/'external_parent_manifest.json').read_text())
            for rel, expected in manifest.items():
                path=B/'external_parent'/rel
                assert sha(path)==expected
                files['external_parent/'+rel]=path
        raw_manifest={n:sha(p) for n,p in files.items()}
        with tarfile.open(B/f'packet_{host}.tar.gz','x:gz') as t:
            for rel,p in files.items():t.add(p,arcname=rel,recursive=False)
            raw=json.dumps(raw_manifest,sort_keys=True).encode();info=tarfile.TarInfo('transport_manifest.json');info.size=len(raw);t.addfile(info,io.BytesIO(raw))
        print(json.dumps({'host':host,'fits':len(jobs),'inference_jobs':sum(len(j['allowed_splits']) for j in jobs),
            'contract_sha256':sha(cp),'packet_sha256':sha(B/f'packet_{host}.tar.gz'),
            'packet_bytes':(B/f'packet_{host}.tar.gz').stat().st_size}))


if __name__=='__main__':main()
