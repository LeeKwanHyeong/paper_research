"""Retrieve complete owned evaluation outputs and verify their original SHA manifest."""
from pathlib import Path
import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(1024*1024),b''):h.update(value)
    return h.hexdigest()

def main():
    contract=json.loads((HERE/'execution_contract.json').read_text())
    resources=contract['resources']
    assert not (HERE/'retrieval_receipt.json').exists(),'Preserve previous retrieval'
    code=r'''
from pathlib import Path
import json,hashlib,tarfile,datetime
root=Path(REMOTE_ROOT);here=root/'reports/titantpp_completed_validation_test_20261004_v1/deep_renewal'
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(p.read_text())
complete=read(here/'inference_completion.json');gate=read(here/'validation_gate.json')
assert complete['status']=='complete' and complete['new_split_count']==24
assert gate['passed'] and gate['conditions']==12
registry=read(here/'evaluation_registry.json');seal=read(here/'code_seal.json')
for relative,expected in seal['files'].items():assert sha(root/relative)==expected,relative
for bundle in registry['bundles'].values():
 for relative,expected in bundle['source_files'].items():assert sha(root/bundle['source_root']/relative)==expected,relative
for row in registry['rows']:assert sha(root/row['checkpoint_path'])==row['checkpoint_file_sha256'],row['checkpoint_path']
for data in read(here/'dataset_manifest.json')['datasets']:assert sha(root/data['path'])==data['sha256'],data['dataset']
selected=[]
for top in ['runs','logs']:
 selected.extend(p for p in (here/top).rglob('*') if p.is_file() and 'runtime_cache' not in p.parts)
for name in ['pipeline_status.json','inference_completion.json','validation_gate.json','launch_receipt.json','controller.log']:
 selected.append(here/name)
assert len(list((here/'runs').glob('*/*/receipt.json')))==24
assert not list((here/'runs').glob('*/*/failure.json'))
files={str(p.relative_to(here)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(selected)}
manifest={'status':'complete','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,
 'selected_checkpoints_rehashed':12,'source_bundles_rehashed':3,'datasets_rehashed':4,'sealed_files_rehashed':len(seal['files']),
 'contract_sha256':sha(here/'execution_contract.json'),'split_receipts':24,'excluded':'Only non-scientific runtime_cache excluded; original datasets/checkpoints are referenced and remain unchanged.'}
manifest_path=here/'result_manifest.json'
with manifest_path.open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
archive_path=here/'results.tar.gz'
with tarfile.open(archive_path,'x:gz') as archive:
 for p in selected:archive.add(p,arcname=str(p.relative_to(here)),recursive=False)
 archive.add(manifest_path,arcname='result_manifest.json',recursive=False)
print(json.dumps({'status':'archived','archive_path':str(archive_path),'archive_sha256':sha(archive_path),
 'archive_bytes':archive_path.stat().st_size,'manifest_sha256':sha(manifest_path),'files':len(files)}))
'''.replace('REMOTE_ROOT',repr(resources['root']))
    text=subprocess.check_output(['ssh','5080',shlex.join([resources['python'],'-B','-c',code])],text=True,timeout=240)
    remote=json.loads(text.strip().splitlines()[-1])
    retrieval=HERE/'retrieved';retrieval.mkdir(exist_ok=False)
    archive=retrieval/'results.tar.gz'
    subprocess.run(['scp','5080:'+remote['archive_path'],str(archive)],check=True,timeout=240)
    assert sha(archive)==remote['archive_sha256']
    original=retrieval/'original';original.mkdir()
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            assert member.isfile() and (original/member.name).resolve().is_relative_to(original)
        tar.extractall(original,filter='data')
    assert sha(original/'result_manifest.json')==remote['manifest_sha256']
    manifest=json.loads((original/'result_manifest.json').read_text())
    for name,record in manifest['files'].items():
        path=original/name
        assert path.stat().st_size==record['bytes'] and sha(path)==record['sha256'],name
    # Materialize the analysis layout while preserving both the original archive and its manifest.
    for name,record in manifest['files'].items():
        source=original/name;target=HERE/name
        if target.exists():
            # The local launch receipt includes the first observation in addition to the remote launch record.
            if name=='launch_receipt.json':continue
            assert sha(target)==record['sha256'],('Prior local evidence differs',name)
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        assert sha(target)==record['sha256'],name
    result={'status':'verified','created_utc':datetime.now(timezone.utc).isoformat(),'host':'5080',
            **remote,'status':'verified','archive_path':str(archive),'original_root':str(original),
            'result_manifest_path':str(original/'result_manifest.json'),'files_rehashed':len(manifest['files']),
            'split_receipts':24,'scientific_conditions':12,'immutable_inputs_reverified_remotely':True,
            'runtime_cache_excluded':True,'original_last_binary_CPU_audit_performed':False}
    with (HERE/'retrieval_receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:result[k] for k in ('status','files_rehashed','archive_bytes','archive_sha256','split_receipts')}))

if __name__=='__main__':main()
