"""Read exact360 Taxi/RAF original files to stdout; original root writes0."""
import hashlib
import json
import sys
import tarfile
from pathlib import Path

p = json.load(sys.stdin)
r = Path(p['remote_original_root'])
assert str(r) == '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_components_20261012_v5_5080'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


assert sha(r / 'execution_contract.json') == p['training_contract_raw_sha256']
expected = p['expected_files']
assert len(expected) == 360 and len(p['job_ids']) == 24
for name, digest in expected.items():
    path, relative = r / name, Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    assert relative.parts[0] in ('run', 'worker_exits')
    assert not path.is_symlink() and path.resolve().is_relative_to(r.resolve())
    assert sha(path) == digest
with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for name, digest in sorted(expected.items()):
        path = r / name
        assert sha(path) == digest
        info = tarfile.TarInfo('originals/' + name)
        info.size, info.mode = path.stat().st_size, 0o600
        with path.open('rb') as handle:
            archive.addfile(info, handle)
