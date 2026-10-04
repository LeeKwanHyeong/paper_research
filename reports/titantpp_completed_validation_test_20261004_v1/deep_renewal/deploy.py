"""Checksum transfer to approved existing 5080 dedicated inference workspace."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    contract = json.loads((HERE / 'execution_contract.json').read_text())
    registry = json.loads((HERE / 'evaluation_registry.json').read_text())
    seal = json.loads((HERE / 'code_seal.json').read_text())
    manifest = json.loads((HERE / 'dataset_manifest.json').read_text())
    resource = contract['resources']
    remote_root, python = resource['root'], resource['python']
    files = dict(seal['files'])
    files[str(HERE.relative_to(ROOT) / 'code_seal.json')] = sha(HERE / 'code_seal.json')
    for row in registry['rows']:
        files[row['checkpoint_path']] = row['checkpoint_file_sha256']
    for bundle in registry['bundles'].values():
        for path, digest in bundle['source_files'].items():
            files[str(Path(bundle['source_root']) / path)] = digest
    for path, digest in files.items():
        assert not Path(path).is_absolute() and '..' not in Path(path).parts
        assert sha(ROOT / path) == digest, path
    verification = {'root': remote_root, 'files': files, 'datasets': manifest['datasets']}
    with (HERE / 'deployment_manifest.json').open('x') as stream:
        json.dump(verification, stream, indent=2)
    artifact = HERE / 'payload.tar.gz'
    with tarfile.open(artifact, 'x:gz') as archive:
        for path in files:
            archive.add(ROOT / path, arcname=path, recursive=False)
        archive.add(HERE / 'deployment_manifest.json', arcname='deployment_manifest.json', recursive=False)
    # The host/runtime are existing approved research resources. No dependency changes.
    probe = f"from pathlib import Path; p=Path({remote_root!r}); assert not p.exists(), 'Prior workspace preserved'; p.mkdir(parents=True)"
    subprocess.run(['ssh', '5080', shlex.join([python, '-c', probe])], check=True, timeout=30)
    subprocess.run(['scp', str(artifact), f'5080:{remote_root}/payload.tar.gz'], check=True, timeout=120)
    code = '''
from pathlib import Path
import hashlib,json,tarfile,os
root=Path(REMOTE_ROOT)
with tarfile.open(root/'payload.tar.gz') as archive:
    for member in archive.getmembers():
        assert (root/member.name).resolve().is_relative_to(root)
    archive.extractall(root,filter='data')
manifest=json.loads((root/'deployment_manifest.json').read_text())
for rel,expected in manifest['files'].items():
    assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==expected,rel
donors=[Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_legacy_evaluation_20261003_v1'),Path('/home/leekwanhyeong/workspace/paper_research')]
data=[]
for spec in manifest['datasets']:
    target=root/spec['path']
    assert not target.exists()
    candidates=[p/spec['path'] for p in donors]
    found=None
    for source in candidates:
        if source.is_file() and hashlib.sha256(source.read_bytes()).hexdigest()==spec['sha256']:
            found=source
            break
    assert found is not None,('Existing dataset bytes unavailable',spec['dataset'])
    target.parent.mkdir(parents=True,exist_ok=True)
    target.symlink_to(found)
    data.append({'dataset':spec['dataset'],'path':str(target),'source':str(found),'sha256':spec['sha256']})
result={'status':'verified','files':len(manifest['files']),'datasets':data,'new_training':False,'runtime_modified':False}
(root/'deployment_verification.json').write_text(json.dumps(result,indent=2)+'\\n')
print(json.dumps(result))
'''.replace('REMOTE_ROOT', repr(remote_root))
    result = subprocess.check_output(['ssh', '5080', shlex.join([python, '-c', code])], text=True, timeout=120)
    (HERE / 'deployment_verification.json').write_text(result)
    print(result)


if __name__ == '__main__':
    main()
