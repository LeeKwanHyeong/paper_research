"""Retrieve completed B9 evidence without modifying any scientific source.

Run once after pipeline completion. Existing local/remote retrieval directories
are deliberately preserved and cause a refusal rather than an implicit retry.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REMOTE_ROOT = '/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_legacy_evaluation_20261003_v1'
CAMPAIGN = 'reports/titantpp_b_test_evaluation_20261003_v1'
REMOTE = REMOTE_ROOT + '/' + CAMPAIGN
REMOTE_PYTHON = '/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python3.12'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '5080']
MAX_MANIFEST_BYTES = 64 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def relative_name(name):
    require(isinstance(name, str) and name and '\\' not in name, 'Invalid evidence path')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts
            and str(path) == name and name not in ('.', ''), 'Unsafe evidence path')
    return path


def condition_ids(registry):
    rows = registry['rows']
    ids = {f"{r['dataset']}__{r['model']}__seed{r['seed']}" for r in rows}
    expected = {f'{dataset}__titantpp__seed{seed}'
                for dataset in ('yellow_trip_hourly', 'intermittent_frozen_5000', 'insta_market_basket')
                for seed in (42, 52, 62)}
    require(len(rows) == 9 and ids == expected, 'Registry must be exactly the approved B9 conditions')
    return sorted(ids)


REMOTE_SCRIPT = r'''
from pathlib import Path
import datetime, hashlib, json, os, stat, tarfile
def require(condition,message):
 if not condition:raise ValueError(message)
def read(path):return json.loads(path.read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
 return h.hexdigest()
p=Path(CONFIG['remote']);root=Path(CONFIG['root'])
require(p.resolve()==p and p.is_relative_to(root),'Unexpected campaign path')
for relative,expected in CONFIG['sealed_files'].items():
 candidate=root/relative
 require(candidate.is_file() and not candidate.is_symlink(),'Missing or linked sealed file: '+relative)
 require(sha(candidate)==expected,'Changed sealed source: '+relative)
require(sha(p/'code_seal.json')==CONFIG['seal_sha256'],'Code seal changed')
require(sha(p/'execution_contract.json')==CONFIG['contract_sha256'],'Contract changed')
require(sha(p/'evaluation_registry.json')==CONFIG['registry_sha256'],'Registry changed')
state=read(p/'pipeline_status.json')
require(state.get('status')=='complete' and state.get('phase')=='test','Pipeline is not complete')
expected=set(CONFIG['condition_ids'])
phase_checks={}
for phase in ('qualification_cpu','qualification_cuda','validation_full','test'):
 folder=p/'runs'/phase/'attempt1';terminal=read(folder/'terminal_manifest.json');manifest=read(folder/'run_manifest.json')
 require(terminal.get('status')=='complete' and terminal.get('total')==9,'Incomplete phase: '+phase)
 require(len(terminal.get('completed',[]))==9 and set(terminal['completed'])==expected,'Missing or duplicate completed conditions: '+phase)
 require(terminal.get('failures')==[] and terminal.get('unstarted')==[],'Failed or unstarted conditions: '+phase)
 require(terminal.get('run_manifest_sha256')==sha(folder/'run_manifest.json'),'Phase manifest mismatch: '+phase)
 require(manifest.get('contract_sha256')==CONFIG['contract_sha256'],'Wrong phase contract: '+phase)
 require(manifest.get('code_seal_sha256')==CONFIG['seal_sha256'],'Wrong phase code seal: '+phase)
 for ident in expected:
  receipt=read(folder/ident/'receipt.json')
  require(receipt.get('status')=='complete','Incomplete receipt: '+phase+'/'+ident)
 phase_checks[phase]={'completed':9,'failures':0,'unstarted':0,'terminal_sha256':sha(folder/'terminal_manifest.json')}
gate=read(p/'qualification_gate.json')
require(gate.get('status')=='passed','Qualification gate not passed')
require(gate.get('contract_sha256')==CONFIG['contract_sha256'] and gate.get('code_seal_sha256')==CONFIG['seal_sha256'],'Gate binding changed')
require(bool(gate.get('evidence_hashes')),'Missing gate evidence')
for relative,expected_sha in gate['evidence_hashes'].items():
 f=root/relative
 require(f.resolve().is_relative_to(root) and sha(f)==expected_sha,'Changed qualification evidence: '+relative)
campaign=read(p/'campaign_state.json');launch=read(p/'launch_receipt.json')
require(campaign.get('contract_sha256')==CONFIG['contract_sha256'],'Campaign clock binding changed')
require(launch.get('contract_sha256')==CONFIG['contract_sha256'] and launch.get('code_seal_sha256')==CONFIG['seal_sha256'],'Launch binding changed')
mandatory={'qualification_gate.json','pipeline_status.json','pipeline.log','launch_receipt.json','campaign_state.json',
           'execution_contract.json','code_seal.json','evaluation_registry.json','dataset_manifest.json','validation_references.json'}
for name in mandatory:require((p/name).is_file() and not (p/name).is_symlink(),'Missing required evidence: '+name)
sealed_local={str((root/name).relative_to(p)) for name in CONFIG['sealed_files'] if (root/name).is_relative_to(p)}
files=[];excluded=[]
# Preserve every generated file, including any earlier failure records and logs.
# Only sealed source/preparation inputs, caches, locks, and this packaging tree
# are omitted; metadata that binds the evidence is explicitly included above.
for base,dirs,names in os.walk(p,followlinks=False):
 base=Path(base)
 for dirname in list(dirs):
  d=base/dirname
  require(not d.is_symlink(),'Evidence directory symlink: '+str(d))
  if dirname in {'runtime_cache','__pycache__','retrieval_package'}:
   dirs.remove(dirname);excluded.append({'path':str(d.relative_to(p)),'reason':'cache_or_retrieval_package'})
 for name in sorted(names):
  f=base/name;relative=str(f.relative_to(p));mode=f.lstat().st_mode
  require(stat.S_ISREG(mode),'Non-regular evidence file: '+relative)
  if relative in sealed_local and relative not in mandatory:
   excluded.append({'path':relative,'reason':'unchanged_sealed_source_or_preparation_input'});continue
  if name.endswith('.lock'):
   excluded.append({'path':relative,'reason':'advisory_lock'});continue
  files.append(f)
files.sort(key=lambda f:str(f.relative_to(p)))
records=[{'path':str(f.relative_to(p)),'bytes':f.stat().st_size,'sha256':sha(f)} for f in files]
require(sum(r['bytes'] for r in records)<=CONFIG['max_evidence_bytes'],'Evidence exceeds approved output limit plus metadata allowance')
manifest={'schema_version':1,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'remote_campaign':str(p),'scope':'B9_original_evaluation_outputs_no_retraining',
          'contract_sha256':CONFIG['contract_sha256'],'code_seal_sha256':CONFIG['seal_sha256'],
          'registry_sha256':CONFIG['registry_sha256'],'phase_checks':phase_checks,'files':records,
          'excluded':excluded,'remote_originals_preserved':True}
out=p/'retrieval_package';out.mkdir(exist_ok=False)
with (out/'manifest.json').open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
archive=out/'evidence.tar.gz'
with tarfile.open(archive,'x:gz') as tf:
 for f in files:tf.add(f,arcname=str(f.relative_to(p)),recursive=False)
 tf.add(out/'manifest.json',arcname='retrieval_manifest.json',recursive=False)
# Refuse a package made while any recorded evidence was still changing.
for f,record in zip(files,records):
 require(f.stat().st_size==record['bytes'] and sha(f)==record['sha256'],'Evidence changed during packaging: '+record['path'])
print(json.dumps({'archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,
                  'manifest_sha256':sha(out/'manifest.json'),'files':len(records),
                  'uncompressed_evidence_bytes':sum(r['bytes'] for r in records),
                  'phase_checks':phase_checks,'contract_sha256':CONFIG['contract_sha256'],
                  'code_seal_sha256':CONFIG['seal_sha256'],'registry_sha256':CONFIG['registry_sha256']}))
'''


def extract_verified(archive, extracted, receipt, max_bytes):
    """Allow only unique regular files declared by the hashed evidence manifest."""
    extracted.mkdir(exist_ok=False)
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        names = [item.name for item in members]
        require(len(names) == len(set(names)), 'Duplicate archive entries')
        for item in members:
            relative_name(item.name)
            require(item.isfile(), 'Archive contains a link or non-regular file')
        require(names.count('retrieval_manifest.json') == 1, 'Missing archive manifest')
        manifest_member = stream.getmember('retrieval_manifest.json')
        require(manifest_member.size <= MAX_MANIFEST_BYTES, 'Manifest is unexpectedly large')
        raw = stream.extractfile(manifest_member).read()
        require(hashlib.sha256(raw).hexdigest() == receipt['manifest_sha256'], 'Manifest SHA mismatch')
        manifest = json.loads(raw)
        rows = manifest['files']
        expected = {}
        for row in rows:
            relative_name(row['path'])
            require(row['path'] not in expected and row['path'] != 'retrieval_manifest.json', 'Duplicate manifest path')
            require(type(row['bytes']) is int and row['bytes'] >= 0, 'Invalid evidence size')
            expected[row['path']] = row
        require(set(names) == set(expected) | {'retrieval_manifest.json'}, 'Archive/manifest inventory mismatch')
        require(len(expected) == receipt['files'], 'Evidence file count mismatch')
        total = sum(row['bytes'] for row in rows)
        require(total == receipt['uncompressed_evidence_bytes'] and total <= max_bytes, 'Evidence size limit or total mismatch')
        for name in ('contract_sha256', 'code_seal_sha256', 'registry_sha256'):
            require(manifest[name] == receipt[name], 'Evidence identity mismatch: ' + name)
        for item in members:
            destination = extracted / item.name
            require(destination.resolve().is_relative_to(extracted.resolve()), 'Archive path escapes destination')
            if item.name != 'retrieval_manifest.json':
                require(item.size == expected[item.name]['bytes'], 'Archive size mismatch: ' + item.name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as output:
                shutil.copyfileobj(stream.extractfile(item), output, 1 << 20)
            if item.name != 'retrieval_manifest.json':
                require(sha(destination) == expected[item.name]['sha256'], 'Evidence SHA mismatch: ' + item.name)
    return manifest


def main():
    require(str(HERE.relative_to(ROOT)) == CAMPAIGN, 'Unexpected local campaign directory')
    # Preflight before network calls; never replace prior outputs or a partial retrieval.
    for name in ('retrieval', 'runs', 'qualification_gate.json'):
        require(not os.path.lexists(HERE / name), 'Preserve existing local path: ' + name)
    contract = read(HERE / 'execution_contract.json')
    seal = read(HERE / 'code_seal.json')
    registry = read(HERE / 'evaluation_registry.json')
    for name, expected in seal['files'].items():
        relative_name(name)
        require(sha(ROOT / name) == expected, 'Sealed local file changed: ' + name)
    require(contract['resources']['root'] == REMOTE_ROOT, 'Unexpected approved remote root')
    require(contract['resources']['host'] == '5080', 'Unexpected approved host')
    require(sha(HERE / 'evaluation_registry.json') == contract['registry_sha256'], 'Local registry changed')
    config = {'remote': REMOTE, 'root': REMOTE_ROOT, 'condition_ids': condition_ids(registry),
              'sealed_files': seal['files'], 'seal_sha256': sha(HERE / 'code_seal.json'),
              'contract_sha256': sha(HERE / 'execution_contract.json'),
              'registry_sha256': sha(HERE / 'evaluation_registry.json'),
              'max_evidence_bytes': (contract['resources']['output_limit_gib'] + 1) * 1024**3}
    folder = HERE / 'retrieval'
    folder.mkdir(exist_ok=False)
    result = subprocess.run(SSH + [REMOTE_PYTHON, '-'],
                            input='CONFIG=' + repr(config) + '\n' + REMOTE_SCRIPT,
                            text=True, capture_output=True, timeout=600, check=False)
    with (folder / 'packaging_stdout.txt').open('x') as output:
        output.write(result.stdout)
    with (folder / 'packaging_stderr.txt').open('x') as output:
        output.write(result.stderr)
    result.check_returncode()
    receipt = json.loads(result.stdout)
    require(receipt['archive'] == REMOTE + '/retrieval_package/evidence.tar.gz', 'Unexpected archive source')
    for field, expected in (('contract_sha256', config['contract_sha256']),
                            ('code_seal_sha256', config['seal_sha256']),
                            ('registry_sha256', config['registry_sha256'])):
        require(receipt[field] == expected, 'Remote binding mismatch: ' + field)
    write_new(folder / 'remote_receipt.json', receipt)
    archive = folder / 'evidence.tar.gz'
    require(not os.path.lexists(archive), 'Archive destination already exists')
    subprocess.run(['scp', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                    '5080:' + receipt['archive'], str(archive)], check=True, timeout=900)
    require(archive.stat().st_size == receipt['bytes'] and sha(archive) == receipt['sha256'], 'Archive identity mismatch')
    extracted = folder / 'original'
    manifest = extract_verified(archive, extracted, receipt, config['max_evidence_bytes'])
    # Only the paths consumed by the existing frozen analyzer are exposed here.
    # All status, logs, failure histories, and binding metadata remain in original/.
    require(not os.path.lexists(HERE / 'runs'), 'Local runs appeared during retrieval')
    shutil.copytree(extracted / 'runs', HERE / 'runs', dirs_exist_ok=False)
    with (HERE / 'qualification_gate.json').open('xb') as output:
        with (extracted / 'qualification_gate.json').open('rb') as source:
            shutil.copyfileobj(source, output)
    exposed = 0
    for row in manifest['files']:
        if row['path'].startswith('runs/') or row['path'] == 'qualification_gate.json':
            require(sha(HERE / row['path']) == row['sha256'], 'Exposed evidence SHA mismatch')
            exposed += 1
    for name, expected in seal['files'].items():
        require(sha(ROOT / name) == expected, 'Sealed local file changed during retrieval: ' + name)
    receipt.update(status='retrieved_and_sha_verified',
                   completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   original_manifest=str(extracted / 'retrieval_manifest.json'),
                   exposed_files_verified=exposed, remote_originals_preserved=True,
                   scientific_sources_modified=False, new_inference=False)
    write_new(folder / 'receipt.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
