"""Read the exact60 seed62 originals; original training root writes0."""
import hashlib,json,sys,tarfile
from pathlib import Path
p=json.load(sys.stdin);r=Path(p['remote_original_root'])
assert str(r)=='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_components_seed62_handoff_20261013_v1_5080'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
assert sha(r/'execution_contract.json')==p['training_contract_raw_sha256']
expected=p['expected_files'];assert len(expected)==60
for n,d in expected.items():
 t=r/n;relative=Path(n);assert not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] in ('run','worker_exits');assert not t.is_symlink() and t.resolve().is_relative_to(r.resolve());assert sha(t)==d
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for n,d in sorted(expected.items()):
  t=r/n;assert sha(t)==d;info=tarfile.TarInfo('originals/'+n);info.size=t.stat().st_size;info.mode=0o600
  with t.open('rb') as f:archive.addfile(info,f)
