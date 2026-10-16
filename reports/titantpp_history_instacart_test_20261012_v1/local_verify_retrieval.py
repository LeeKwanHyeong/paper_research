from pathlib import Path
import json,hashlib,tarfile,shutil,time
b=Path('search_artifacts/titantpp_history_instacart_test_20261012_v1').resolve()
r=b/'originals';receipt=json.loads((r/'receipt.json').read_bytes()); manifest=json.loads((r/'manifest.json').read_bytes())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
assert sha(r/'manifest.json')==receipt['manifest_sha256']
assert sha(r/'originals.tar.gz')==receipt['archive_sha256']
assert len(manifest['files'])==receipt['file_count']
verified=r/'verified';assert not verified.exists()
with tarfile.open(r/'originals.tar.gz','r:gz') as t:
 members=t.getmembers();assert len(members)==len(manifest['files']) and {m.name for m in members}==set(manifest['files'])
 for m in members:
  p=Path(m.name);assert m.isfile() and not p.is_absolute() and '..' not in p.parts
 verified.mkdir();t.extractall(verified,filter='data')
for rel,expected in manifest['files'].items():
 p=verified/rel;assert p.resolve().is_relative_to(verified) and sha(p)==expected,rel
for rel,expected in manifest['files'].items():
 target=b/rel;assert target.resolve().is_relative_to(b)
 if target.exists():assert sha(target)==expected,rel
 else:
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as dst,(verified/rel).open('rb') as src:shutil.copyfileobj(src,dst)
 assert sha(target)==expected,rel
result={'status':'verified','contract_sha256':receipt['contract_sha256'],'file_count':len(manifest['files']),'archive_sha256':receipt['archive_sha256'],'manifest_sha256':receipt['manifest_sha256'],'all_local_originals_match':True,'existing_files_overwritten':0,'new_binary_evaluation':False,'observed_unix':time.time()}
with (r/'LOCAL_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
