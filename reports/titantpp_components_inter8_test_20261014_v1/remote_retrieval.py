"""Read verified original files to stdout tar; remote filesystem writes0."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

request = json.load(sys.stdin)
root = Path(request['remote_original_root'])
assert root == Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_gru_components_20261012_v5_5090')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


assert digest(root / 'execution_contract.json') == request['training_contract_raw_sha256']
expected = request['expected_files']
assert len(expected) == 120
paths = {}
for name, value in expected.items():
    relative = Path(name)
    assert not relative.is_absolute() and '..' not in relative.parts
    assert relative.parts[0] in ('run', 'worker_exits')
    path = root / relative
    assert not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
    assert path.is_file() and digest(path) == value, name
    paths[name] = path

with tarfile.open(fileobj=sys.stdout.buffer, mode='w|gz') as archive:
    for name in sorted(paths):
        path = paths[name]
        assert digest(path) == expected[name], name
        info = tarfile.TarInfo('originals/' + name)
        info.size = path.stat().st_size
        info.mode = 0o600
        with path.open('rb') as stream:
            archive.addfile(info, stream)
