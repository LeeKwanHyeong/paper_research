import os,sys,json,time,hashlib,pathlib,subprocess,signal,shutil,datetime,tarfile
ROOT=pathlib.Path(sys.argv[1]); HOST=sys.argv[2]; EXPECTED=sys.argv[3]
def read(p): return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
c=read(ROOT/'execution_contract.json');assert canon(c)==EXPECTED;assert c['hosts'][HOST]['root']==str(ROOT)
assert all(sha(ROOT/'source'/p)==v for p,v in c['source']['files'].items())
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True);assert c['hosts'][HOST]['gpu_uuid'] in gpu
SCRIPT=str(ROOT/'source/paper/scripts/run_titantpp_gru_control_campaign.py')
def proc(pid):
 try:
  p=pathlib.Path('/proc')/str(pid);argv=(p/'cmdline').read_bytes().decode().split('\0')[:-1];stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  return {'pid':pid,'argv':argv,'start_ticks':stat[19],'state':stat[0],'ppid':int(stat[1])}
 except (FileNotFoundError,ProcessLookupError):return None
def own():
 rows=[]
 for p in pathlib.Path('/proc').iterdir():
  if not p.name.isdigit():continue
  x=proc(int(p.name))
  if x and SCRIPT in x['argv'] and '--contract' in x['argv'] and str(ROOT/'execution_contract.json') in x['argv'] and '--mode' in x['argv'] and any(m in x['argv'] for m in ('fit','dispatch')):rows.append(x)
 return rows
def match(x):
 y=proc(x['pid']);return bool(y and y['start_ticks']==x['start_ticks'] and y['argv']==x['argv'] and y['state']!='Z')
def sig(x,s):
 if match(x):os.kill(x['pid'],s)
before=own(); dispatch=[x for x in before if 'dispatch' in x['argv'] and pathlib.Path(x['argv'][0]).name!='timeout'];assert len(dispatch)<=1
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');out=ROOT/'user_stop'/stamp;out.mkdir(parents=True,exist_ok=False)
write(out/'authorization.json',{'authorized_by':'user','request':'지금 진행중인 실험 중지하고 cnn+gru54로 5080/5090 실험진행하자 승인할게','host':HOST,'contract_sha256':EXPECTED,'reason':'research direction change; not numerical training failure','observed_before':before,'gpu':gpu,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
frozen=[]
try:
 for x in dispatch:sig(x,signal.SIGSTOP);frozen.append(x)
 rows=own();fits=[x for x in rows if 'fit' in x['argv'] and pathlib.Path(x['argv'][0]).name!='timeout'];assert len(fits)<=1
 pre=[]
 for x in fits:
  job=x['argv'][x['argv'].index('--job')+1];j=next(j for j in c['jobs'] if j['id']==job and j['host']==HOST)
  run=ROOT/'run'/job/'runs'/j['arm']/'count_only_log_regression'/('seed_'+str(j['seed']))
  for attempt in range(12):
   sig(x,signal.SIGSTOP)
   receipt=read(run/'server_checkpoint_receipt.json') if (run/'server_checkpoint_receipt.json').exists() else None
   keys=['last_epoch_state.pt','history.json','exposure.json','epoch_timing.json']
   valid=receipt and all(k in receipt['files'] and (run/k).is_file() and sha(run/k)==receipt['files'][k] for k in keys)
   if valid:break
   sig(x,signal.SIGCONT);time.sleep(0.5)
  assert valid,'No consistent saved checkpoint/receipt; stop aborted with owner restored'
  frozen.append(x);saved=out/'pre_stop'/job;saved.mkdir(parents=True)
  for k in keys+['server_checkpoint_receipt.json']:shutil.copy2(run/k,saved/k)
  import torch
  state=torch.load(saved/'last_epoch_state.pt',map_location='cpu',weights_only=False);hist=read(saved/'history.json')['history'];assert state['epoch']==len(hist) and state['history']==hist
  item={'job':j,'epoch':state['epoch'],'best_epoch':state['best_epoch'],'files':{k:sha(saved/k) for k in keys+['server_checkpoint_receipt.json']},'owned_process':x};pre.append(item);del state
 write(out/'pre_stop_receipt.json',{'host':HOST,'contract_sha256':EXPECTED,'consistent_checkpoints':pre,'queue_dispatch_frozen':dispatch})
 # Terminate only processes whose command and Linux start-time still match this owned campaign.
 for x in fits:sig(x,signal.SIGTERM);sig(x,signal.SIGCONT)
 deadline=time.monotonic()+10
 while any(match(x) for x in fits) and time.monotonic()<deadline:time.sleep(0.2)
 for x in fits:
  if match(x):sig(x,signal.SIGKILL)
 for x in dispatch:sig(x,signal.SIGTERM);sig(x,signal.SIGCONT)
 for x in own():
  if pathlib.Path(x['argv'][0]).name=='timeout':sig(x,signal.SIGTERM);sig(x,signal.SIGCONT)
 deadline=time.monotonic()+10
 while any(match(x) for x in own()) and time.monotonic()<deadline:time.sleep(0.2)
 remain=[x for x in own() if x['state']!='Z'];assert not remain,'Owned processes remain'
 gpu_apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
 oldpids={x['pid'] for x in before+rows};assert not any(int(line.split(',')[1]) in oldpids for line in gpu_apps.splitlines() if line.startswith(c['hosts'][HOST]['gpu_uuid']))
 inventory=[]
 for j in [j for j in c['jobs'] if j['host']==HOST]:
  folder=ROOT/'run'/j['id'];run=folder/'runs'/j['arm']/'count_only_log_regression'/('seed_'+str(j['seed']));terminal=read(folder/'terminal_manifest.json') if (folder/'terminal_manifest.json').exists() else None
  item={'job':j,'state':'not_started' if not folder.exists() else 'user_stopped_partial','scientific_success':False}
  if terminal:
   assert terminal['scientific_success'] and terminal['contract_sha256']==EXPECTED
   assert all(sha(folder/p)==v for p,v in terminal['files'].items());item.update(state='complete',scientific_success=True,terminal_manifest_sha256=sha(folder/'terminal_manifest.json'))
  if (run/'last_epoch_state.pt').exists():
   import torch
   state=torch.load(run/'last_epoch_state.pt',map_location='cpu',weights_only=False);item.update(epoch=state['epoch'],best_epoch=state['best_epoch'],last_checkpoint_sha256=sha(run/'last_epoch_state.pt'));del state
  inventory.append(item)
 archive=out/'sealed_originals.tar.gz'
 with tarfile.open(archive,'w:gz') as tf:
  for name in ['run','logs','status.json','failure.json','supervisor_process_exit.json','execution_contract.json','approval.json','start_permit.json','training_permit.json','claims','server_lease.json','supervisor.claim']:
   p=ROOT/name
   if p.exists():tf.add(p,arcname=name)
 write(out/'stop_receipt.json',{'host':HOST,'contract_sha256':EXPECTED,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'user_stopped','reason':'explicit user strategy change to CNN+GRU54 three-seed','training_failure':False,'processes_before':before,'processes_after':remain,'gpu_apps_after':gpu_apps,'jobs':inventory,'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'original_status_preserved':True,'claims_preserved':True})
 print(json.dumps({'host':HOST,'stop_receipt':str(out/'stop_receipt.json'),'archive':str(archive),'archive_sha256':sha(archive),'inventory':inventory},ensure_ascii=False))
except BaseException:
 for x in frozen:sig(x,signal.SIGCONT)
 raise
