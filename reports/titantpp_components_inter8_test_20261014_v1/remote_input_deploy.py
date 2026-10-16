"""Verify/extract only the explicitly authorized new Inter8 input root."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

payload = json.load(sys.stdin)
root = Path(payload['root'])
assert root == Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_inter8_test_20261014_v1_5090')
intent = json.loads((root / 'deployment_intent.json').read_text())
archive_path = root / 'incoming/input_deployment.tar.gz'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


assert digest(archive_path) == payload['archive_sha256'] == intent['archive_sha256']
assert not (root / 'deployment_receipt.json').exists(), 'Existing deployment preserved; no automatic repeat'
with tarfile.open(archive_path, 'r:gz') as archive:
    integrity = json.load(archive.extractfile('deployment_integrity.json'))
    expected = integrity['files']
    assert len(expected) == 261 and integrity['approval_sha256'] == intent['approval_sha256']
    members = archive.getmembers()
    assert len(members) == 262
    seen = set()
    for member in members:
        assert member.isfile() and member.name not in seen
        assert member.name in expected or member.name == 'deployment_integrity.json'
        relative = Path(member.name)
        assert not relative.is_absolute() and '..' not in relative.parts
        target = root / relative
        assert target.resolve().is_relative_to(root.resolve()) and not target.exists()
        raw = archive.extractfile(member).read()
        if member.name in expected:
            assert hashlib.sha256(raw).hexdigest() == expected[member.name], member.name
        seen.add(member.name)
    assert seen == set(expected) | {'deployment_integrity.json'}
    for member in members:
        target = root / member.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(archive.extractfile(member).read())
for name, value in expected.items():
    assert digest(root / name) == value
with (root / 'deployment_receipt.json').open('x') as f:
    json.dump({'status': 'deployed', 'archive_sha256': payload['archive_sha256'],
               'files': expected, 'original_root_writes': 0, 'new_training': False,
               'completed_unix': time.time(), 'automatic_retry': False}, f)
print(json.dumps({'status': 'deployed', 'files': len(expected), 'original_root_writes': 0}))
