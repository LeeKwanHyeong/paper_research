"""Seal and package the Train-only operation; staged frozen inputs are copied."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import time

PROJECT=Path('/Users/igwanhyeong/PycharmProjects/paper_research')
ROOT=PROJECT/'search_artifacts/titantpp_intermittent_time_cross_20261012_v1'
REPORT=PROJECT/'reports/titantpp_intermittent_time_cross_20261012_v1'
PARENT=PROJECT/'search_artifacts/titantpp_history_train_fit_20261012_v1'

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda:stream.read(1048576),b''):
            h.update(data)
    return h.hexdigest()

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n')

def main():
    require_files={
        'control/evaluate.py':PROJECT/'paper/scripts/evaluate_titantpp_intermittent_time_cross.py',
        'control/pipeline.py':PROJECT/'paper/scripts/run_titantpp_intermittent_time_cross_pipeline.py',
        'control/original_evaluate.py':PARENT/'control/original_evaluate.py',
        'control/finalize.py':REPORT/'remote_finalize.py',
        'control/launch.sh':REPORT/'launch.sh',
        'control/remote_setup.py':REPORT/'remote_setup.py',
        'control/prerequisite.py':REPORT/'verify_prerequisite.py',
        'control/approval.json':REPORT/'APPROVAL.json',
        'control/initial_health.json':REPORT/'INITIAL_5080_HEALTH.json',
        'control/prelaunch_qa.md':REPORT/'PRELAUNCH_QA.md',
    }
    assert sha(require_files['control/evaluate.py'])=='e749eedfa19703476bbd9caab6ee31253691f450b1a78f7b9eee7588caf3ec1e'
    assert sha(require_files['control/original_evaluate.py'])=='71aa18a719bdd1c1f735b6f78a5f7f50d93e696002d3561c35df22d1f7b0338c'
    c=json.loads((ROOT/'contract.json').read_bytes())
    health=json.loads((REPORT/'INITIAL_5080_HEALTH.json').read_bytes())
    assert health['gpu_processes']['exit_code']==0 and not health['gpu_processes']['stdout']
    assert health['package_manager_processes']['exit_code']==1 and not health['package_manager_processes']['stdout']
    assert health['parent_contract_raw_sha256']==c['parent_contract_raw_sha256']
    assert not health['reboot_required'] and not health['new_root_exists']
    files={}
    for rel,src in require_files.items():
        dest=ROOT/rel
        dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(src.read_bytes())
        assert sha(dest)==sha(src)
        files[rel]=sha(dest)
    inputs=json.loads((ROOT/'input_integrity.json').read_bytes())['files']
    reused={rel:digest for rel,digest in inputs.items() if rel!='contract.json' and not rel.startswith(('reference/','prerequisite/'))}
    write(ROOT/'deployment/reuse.json',{'contract_sha256':c['contract_sha256'],
        'parent_root':'/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_history_train_fit_20261012_v1_5080',
        'files':reused,'originals_modified':False})
    for rel in ('input_integrity.json','deployment/reuse.json'):
        files[rel]=sha(ROOT/rel)
    write(ROOT/'operation_integrity.json',{'schema':'sealed_intermittent_time_cross_operation_v1',
        'contract_sha256':c['contract_sha256'],'files':files,'new_fits':0,
        'new_train_evaluations':9,'new_validation_evaluations':9,'new_test_evaluations':0})
    members=set(files)|{'contract.json','operation_integrity.json'}|{r for r in inputs if r.startswith(('reference/','prerequisite/'))}
    archive=REPORT/'transport.tar.gz'
    with tarfile.open(archive,'x:gz') as stream:
        for rel in sorted(members):
            stream.add(ROOT/rel,arcname=rel,recursive=False)
    record={'status':'sealed','contract_sha256':c['contract_sha256'],
        'operation_integrity_sha256':sha(ROOT/'operation_integrity.json'),'transport_archive_sha256':sha(archive),
        'transport_archive_bytes':archive.stat().st_size,'transport_members':len(members),
        'copied_immutable_inputs':len(reused),'operation_files':len(files),'observed_unix':time.time()}
    write(REPORT/'DEPLOYMENT_PACKAGE.json',record)
    print(json.dumps(record))

if __name__=='__main__':main()
