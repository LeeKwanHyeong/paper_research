from pathlib import Path
import base64, hashlib, json, shlex, subprocess

base = Path('search_artifacts/local_detail_instacart_remaining_5090_20260926_v1')
snapshot = base/'monitor/20260927T130927Z/5090.json'
observed = json.loads(snapshot.read_text())
c = json.loads((base/'frozen_execution/execution_contract.json').read_text())
spec = c['hosts']['5090']
assert observed['files']['run/status.json']['status'] == 'complete'
assert observed['tmux_exit'] == 1 and len(observed['owned_processes'].strip().splitlines()) == 1
names = ['dispatch_status.json','dispatch_claim.json','run/status.json','run/wrapper_manifest.json',
         'run/partition_summary.json','run/active_arm.json','frozen_execution/execution_contract.json',
         'approval.json','start_permit.json','training_permit.json',
         'qualification/status.json','qualification/receipt.json','qualification/wrapper_manifest.json']
stem = 'run/insta_market_basket/seed_62/'
names += [stem+n for n in ['paired_comparison.json','initialization.json','input_receipt.json']]
logs = ['dispatch.log','qualification/worker.log']
checkpoints = []
for arm in c['execution_arms']:
    run = stem+f'runs/{arm}/count_only_log_regression/seed_62/'
    names += [run+n for n in ['summary.json','history.json','exposure.json','endpoint_replays.json','epoch_timing.json']]
    names += [f'run/{kind}/{arm}.json' for kind in ['workers','progress','arm_receipts']]
    logs += [run+'train.log', f'run/workers/{arm}.log']
    checkpoints += [run+n for n in ['best_val_qty_rmse_model.pt','last_epoch_state.pt']]
code = '''from pathlib import Path
import base64,hashlib,json
root=Path(ROOT);names=NAMES;logs=LOGS;checkpoints=CHECKPOINTS
c=json.loads((root/'frozen_execution/execution_contract.json').read_text())
assert json.loads((root/'run/status.json').read_text())['status']=='complete'
assert json.loads((root/'dispatch_status.json').read_text())['status']=='complete'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
 return h.hexdigest()
lineage={}
for name,expected in c['remaining_recovery']['completed_files_sha256'].items():
 actual=digest(Path(c['remaining_recovery']['old_root'])/name)
 assert actual==expected, name
 lineage[name]=actual
result={}
for name in names+logs:
 p=root/name
 if name in names:
  assert p.stat().st_size<8*1024**2
  raw=p.read_bytes();tail=False
 else:
  if not p.exists():continue
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-12000));raw=f.read()
  tail=True
 result[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'tail_only':tail,'base64':base64.b64encode(raw).decode()}
print(json.dumps({'files':result,'checkpoint_file_hashes':{name:digest(root/name) for name in checkpoints},'predecessor_lineage_sha256':lineage}))
'''.replace('ROOT',repr(spec['root'])).replace('NAMES',repr(names)).replace('LOGS',repr(logs)).replace('CHECKPOINTS',repr(checkpoints))
p = subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','5090',shlex.join([spec['python'],'-'])],input=code,text=True,capture_output=True,timeout=55)
assert p.returncode == 0,p.stderr
result=json.loads(p.stdout)
dest=snapshot.parent/'terminal_5090_recovery';dest.mkdir(exist_ok=False)
for name,info in result['files'].items():
 raw=base64.b64decode(info.pop('base64'));assert hashlib.sha256(raw).hexdigest()==info['sha256']
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(raw)
manifest={**result,'mode':'read_only_copy_and_byte_hashing_no_checkpoint_deserialization_or_data_load','remote_root':spec['root'],
          'checked_at_utc':observed['checked_at_utc'],'terminal_process_snapshot':str(snapshot)}
(dest/'collection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'status':'collected','files':len(result['files']),'bytes':sum(x['bytes'] for x in result['files'].values()),
 'new_checkpoint_hashes':len(result['checkpoint_file_hashes']),'preserved_predecessor_files_verified':len(result['predecessor_lineage_sha256']),'destination':str(dest)}))
