"""Return exact24 verified native checkpoint metadata blobs to stdout."""
import base64
import hashlib
import json
import sys
from pathlib import Path

p = json.load(sys.stdin)
root = Path(p['root'])
assert str(root) == '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_taxi_raf24_test_20261014_v1_5080'
assert len(p['refs']) == 24
out = {}
for name, digest in p['refs'].items():
    relative, path = Path(name), root / name
    assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'metadata'
    assert not path.is_symlink() and path.resolve().is_relative_to(root.resolve())
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == digest
    out[name] = base64.b64encode(raw).decode()
print(json.dumps(out))
