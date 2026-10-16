"""Verify every archive member and materialize immutable terminal outputs."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import tarfile
import time

ROOT=Path('/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_history_diagnostics_20261012_v2')
ORIGINALS=ROOT/'originals'
CONTRACT='70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d'
SEAL='236ad7a57096bd91811bf8ad0cc1a6b389c43b3272ac8520687683b7b8458682'

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def safe(relative):
    p=PurePosixPath(relative)
    require(not p.is_absolute() and '..' not in p.parts,'Unsafe manifest relative path')
    return p

def main():
    verification=ORIGINALS/'LOCAL_VERIFICATION.json'
    require(not verification.exists(),'Local verification is exclusive; no overwrite')
    receipt=json.loads((ORIGINALS/'receipt.json').read_bytes())
    manifest=json.loads((ORIGINALS/'manifest.json').read_bytes())
    archive=ORIGINALS/'originals.tar.gz'
    require(receipt['status']=='verified_terminal_archived' and receipt['contract_sha256']==CONTRACT
        and receipt['operation_integrity_sha256']==SEAL and receipt['actual_pipeline_exit0'] is True
        and receipt['actual_worker_exit0_count']==54 and receipt['all27_validation_gate_pass'] is True
        and receipt['own_processes_absent'] is True and receipt['own_gpu_pids_absent'] is True,
        'Terminal retrieval proof differs')
    require(sha(archive)==receipt['archive_sha256'] and sha(ORIGINALS/'manifest.json')==receipt['manifest_sha256'],
        'Archive or manifest transport SHA differs')
    require(manifest['contract_sha256']==CONTRACT and len(manifest['files'])==receipt['file_count'],
        'Manifest contract/count differs')
    verified=ORIGINALS/'verified'
    require(not verified.exists(),'Verified extraction already exists; no retry')
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers()
        require(len(members)==len(manifest['files']) and {m.name for m in members}==set(manifest['files']),
            'Archive members differ from manifest')
        for member in members:
            safe(member.name)
            require(member.isfile(),'Nonregular archive member')
            h=hashlib.sha256()
            with t.extractfile(member) as f:
                for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
            require(h.hexdigest()==manifest['files'][member.name],'Archive member SHA mismatch: '+member.name)
        verified.mkdir(exist_ok=False)
        t.extractall(verified,filter='data')
    for relative,expected in manifest['files'].items():
        safe(relative)
        extracted=(verified/relative).resolve()
        require(extracted.is_relative_to(verified) and not extracted.is_symlink() and sha(extracted)==expected,
            'Extracted file differs: '+relative)
        target=(ROOT/relative).resolve()
        require(target.is_relative_to(ROOT),'Materialized path escapes root')
        if target.exists():
            require(not target.is_symlink() and sha(target)==expected,'Existing immutable input differs: '+relative)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(extracted,target)
            require(sha(target)==expected,'Materialized output copy differs: '+relative)
    result={'status':'verified','contract_sha256':CONTRACT,'operation_integrity_sha256':SEAL,
        'manifest_sha256':sha(ORIGINALS/'manifest.json'),'archive_sha256':sha(archive),
        'file_count':len(manifest['files']),'all_archive_members_and_local_files_sha_verified':True,
        'verified_root':str(verified),'checked_unix':time.time()}
    with verification.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result))

if __name__=='__main__':main()
