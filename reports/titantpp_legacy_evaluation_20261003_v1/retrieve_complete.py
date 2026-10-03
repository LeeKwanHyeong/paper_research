"""Retrieve finished, hash-checked evaluation evidence from its owned 5080 root."""
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


HERE = Path(__file__).resolve().parent
REMOTE = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_legacy_evaluation_20261003_v1/reports/titantpp_legacy_evaluation_20261003_v1'
SCRIPT = r'''
from pathlib import Path
import json, hashlib, tarfile, datetime
def sha(path):
 value=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
 return value.hexdigest()
p=Path(REMOTE)
state=json.loads((p/'pipeline_status.json').read_text())
assert state['status']=='complete', state
terminal=json.loads((p/'runs/test/attempt1/terminal_manifest.json').read_text())
assert terminal['status']=='complete' and len(terminal['completed'])==108 and not terminal['failures']
files=[]
for name in ['runs','failed_deployment_1']:
 for f in sorted((p/name).rglob('*')):
  if f.is_file() and not any(x in f.parts for x in ['runtime_cache','__pycache__']):
   files.append(f)
for name in ['qualification_gate.json','pipeline_status.json','pipeline.log','launch_receipt.json','campaign_state.json','deployment_layout_correction.json']:
 if (p/name).is_file():files.append(p/name)
manifest={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':[{'path':str(f.relative_to(p)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]}
out=p/'retrieval_package'
out.mkdir(exist_ok=False)
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=out/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
 for f in files:tf.add(f,arcname=str(f.relative_to(p)),recursive=False)
 tf.add(out/'manifest.json',arcname='retrieval_manifest.json')
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'files':len(files)}))
'''


def main():
    folder = HERE / 'retrieval'
    folder.mkdir(exist_ok=False)
    result = subprocess.run(
        ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '5080',
         '/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12', '-'],
        input='REMOTE='+repr(REMOTE)+'\n'+SCRIPT, text=True,
        capture_output=True, check=True, timeout=300)
    receipt = json.loads(result.stdout)
    (folder / 'remote_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    archive = folder / 'evidence.tar.gz'
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    '5080:'+receipt['archive'], str(archive)], check=True, timeout=900)
    assert sha(archive) == receipt['sha256']
    extracted = folder / 'original'
    extracted.mkdir()
    with tarfile.open(archive) as tf:
        for item in tf.getmembers():
            path = Path(item.name)
            assert not path.is_absolute() and '..' not in path.parts and item.isfile()
        tf.extractall(extracted, filter='data')
    manifest = json.loads((extracted / 'retrieval_manifest.json').read_text())
    for item in manifest['files']:
        path = extracted / item['path']
        assert path.stat().st_size == item['bytes']
        assert sha(path) == item['sha256'], item['path']
    # Preserve the full original package, then expose only needed evaluation trees
    # at the frozen manifest's original relative locations.
    import shutil
    for name in ['runs', 'failed_deployment_1']:
        assert not (HERE / name).exists(), name
        shutil.copytree(extracted / name, HERE / name)
    for name in ['qualification_gate.json']:
        assert not (HERE / name).exists(), name
        shutil.copy2(extracted / name, HERE / name)
    receipt.update(status='retrieved_and_sha_verified', original_manifest=str(extracted / 'retrieval_manifest.json'))
    (folder / 'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
