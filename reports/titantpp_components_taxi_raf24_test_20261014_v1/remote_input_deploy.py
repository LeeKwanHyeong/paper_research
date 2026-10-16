"""Extract a verified input packet exclusively into the new native5080 root."""
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path

p = json.load(sys.stdin)
r = Path(p['root'])
assert str(r) == '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_taxi_raf24_test_20261014_v1_5080'
archive = r / 'incoming/input_deployment.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == p['archive_sha256']
with tarfile.open(archive) as tar:
    manifest = json.load(tar.extractfile('deployment_integrity.json'))
    files = manifest['files']
    assert manifest['approval_sha256'] == p['approval_sha256'] and len(files) == p['count']
    members = tar.getmembers()
    assert len(members) == len(files) + 1 and len({item.name for item in members}) == len(members)
    for item in members:
        name, target = Path(item.name), r / item.name
        assert item.isfile() and not name.is_absolute() and '..' not in name.parts
        assert target.resolve().is_relative_to(r.resolve()) and not target.exists() and not target.is_symlink()
        raw = tar.extractfile(item).read()
        assert item.name == 'deployment_integrity.json' or hashlib.sha256(raw).hexdigest() == files[item.name]
    for item in members:
        target = r / item.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as handle:
            handle.write(tar.extractfile(item).read())
with (r / 'deployment_receipt.json').open('x') as handle:
    json.dump({'status': 'deployed', 'files': files, 'archive_sha256': p['archive_sha256'],
               'original_root_writes': 0, 'completed_unix': time.time()}, handle)
print(json.dumps({'status': 'deployed', 'files': len(files)}))
