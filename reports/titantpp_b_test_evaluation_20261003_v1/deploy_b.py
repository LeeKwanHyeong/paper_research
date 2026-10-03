"""Deploy sealed B-only additions to the existing authorized 5080 evaluation root."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE_ROOT = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_legacy_evaluation_20261003_v1'
REMOTE_PYTHON = '/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '5080']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    seal = json.loads((HERE / 'code_seal.json').read_text())
    registry = json.loads((HERE / 'evaluation_registry.json').read_text())
    files = dict(seal['files'])
    files[str((HERE / 'code_seal.json').relative_to(ROOT))] = sha(HERE / 'code_seal.json')
    for row in registry['rows']:
        files[row['checkpoint_path']] = row['checkpoint_file_sha256']
    for relative, expected in files.items():
        assert sha(ROOT / relative) == expected, relative
    archive = HERE / 'deployment.tar.gz'
    with tarfile.open(archive, 'x:gz') as stream:
        for relative in sorted(files):
            stream.add(ROOT / relative, arcname=relative, recursive=False)
    archive_sha = sha(archive)
    remote_archive = REMOTE_ROOT + '/B_deployment_' + archive_sha + '.tar.gz'
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    str(archive), '5080:' + remote_archive], check=True, timeout=600)
    config = {'root': REMOTE_ROOT, 'python': REMOTE_PYTHON, 'archive': remote_archive,
              'archive_sha': archive_sha, 'files': files,
              'campaign': str(HERE.relative_to(ROOT)),
              'tmux': 'titantpp_B_test_20261003_v1',
              'source_files': {str(Path(b['source_root']) / relative): expected
                               for b in registry['bundles'].values()
                               for relative, expected in b['source_files'].items()}}
    script = "CONFIG=" + repr(config) + '\n' + REMOTE_SCRIPT
    result = subprocess.run(SSH + [REMOTE_PYTHON, '-'], input=script, text=True,
                            capture_output=True, timeout=300, check=True)
    receipt = json.loads(result.stdout)
    (HERE / 'deployment_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


REMOTE_SCRIPT = r'''
import datetime,hashlib,json,shlex,subprocess,tarfile,sys
from pathlib import Path
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
root=Path(CONFIG['root']); folder=root/CONFIG['campaign']
assert not folder.exists(), 'Preserve existing B attempt'
assert sha(CONFIG['archive'])==CONFIG['archive_sha']
for relative,expected in CONFIG['source_files'].items():
 assert sha(root/relative)==expected, relative
with tarfile.open(CONFIG['archive']) as archive:
 members=archive.getmembers()
 assert len(members)==len(CONFIG['files'])
 assert {m.name for m in members}==set(CONFIG['files'])
 for m in members:
  assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
  destination=(root/m.name).resolve()
  assert destination.is_relative_to(root)
  if destination.exists():assert sha(destination)==CONFIG['files'][m.name],m.name
 for m in members:
  destination=root/m.name
  if not destination.exists():
   destination.parent.mkdir(parents=True,exist_ok=True)
   with destination.open('xb') as output:output.write(archive.extractfile(m).read())
for relative,expected in CONFIG['files'].items():assert sha(root/relative)==expected,relative
contract=json.loads((folder/'execution_contract.json').read_text())
for ds in json.loads((folder/'dataset_manifest.json').read_text())['datasets']:
 assert sha(root/ds['path'])==ds['sha256'],ds['dataset']
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert gpu==contract['resources']['gpu_uuid']
pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not pids,'GPU occupied; leave other work untouched'
command=shlex.join([CONFIG['python'],'-u',str(folder/'pipeline.py')])+' > '+shlex.quote(str(folder/'pipeline.log'))+' 2>&1'
tmux='/usr/bin/tmux'
subprocess.run([tmux,'new-session','-d','-s',CONFIG['tmux'],'-c',str(root),command],check=True)
receipt={'status':'launched','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'root':str(root),'campaign':str(folder),'tmux':CONFIG['tmux'],'python':sys.executable,
 'gpu_uuid':gpu,'uploaded_files_verified':len(CONFIG['files']),'source_files_verified':len(CONFIG['source_files']),
 'archive_sha256':CONFIG['archive_sha'],'contract_sha256':sha(folder/'execution_contract.json'),
 'code_seal_sha256':sha(folder/'code_seal.json'),'test_requires_all_nine_validation_replays':True}
(folder/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
'''


if __name__ == '__main__':
    main()
