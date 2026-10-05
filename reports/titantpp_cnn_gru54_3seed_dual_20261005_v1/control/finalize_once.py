"""Retrieve verified terminal binaries once and summarize nine Validation conditions. No Test."""
from pathlib import Path
import json,hashlib,subprocess,shlex,tarfile,fcntl,time,statistics,math,csv
BUNDLE=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
REMOTE=r'''from pathlib import Path
import json,hashlib,subprocess,tarfile,time
root=Path(ROOT);c=json.loads((root/'execution_contract.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
canon=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert canon(c)==EXPECTED and c['hosts'][HOST]['root']==str(root)
ps=subprocess.check_output(['ps','-eo','args'],text=True)
assert not any(str(root/'operation/campaign.py') in x and ('--mode fit' in x or '--mode dispatch' in x) for x in ps.splitlines())
assert read(root/'status.json')['status']=='complete' and read(root/'supervisor_process_exit.json')['returncode']==0
for j in [j for j in c['jobs'] if j['host']==HOST]:
 folder=root/'run'/j['id'];m=read(folder/'terminal_manifest.json');assert m['scientific_success'] and m['contract_sha256']==EXPECTED and m['job']==j
 for rel,s in m['files'].items():
  p=Path(rel);assert not p.is_absolute() and '..' not in p.parts;assert sha(folder/p)==s
transfer=root/'transfer';transfer.mkdir(exist_ok=True);archive=transfer/'training_originals.tar.gz';receipt=transfer/'receipt.json'
if receipt.exists():
 r=read(receipt);assert sha(archive)==r['archive_sha256']
else:
 assert not archive.exists(),'Incomplete archive requires manual action; no retry'
 with tarfile.open(archive,'w:gz') as tf:
  for name in ['execution_contract.json','approval.json','start_permit.json','training_permit.json','status.json','supervisor_process_exit.json','qualification','run','logs','operation']:
   tf.add(root/name,arcname=name)
  for rel in c['source']['files']:tf.add(root/'source'/rel,arcname='source/'+rel)
 r={'status':'sealed','host':HOST,'contract_sha256':EXPECTED,'archive_path':str(archive),'archive_sha256':sha(archive),'created_unix':time.time(),'held_out_test_evaluated':False}
 receipt.write_text(json.dumps(r,indent=2))
print(json.dumps(r))
'''
def retrieve(c,h):
 dest=BUNDLE/'retrieved'/h;receipt=dest/'local_original_SHA_receipt.json';cache=read(BUNDLE/'hourly_monitor'/('terminal_cache_'+h+'.json'))
 if receipt.exists():
  r=read(receipt);assert r['contract_sha256']==digest(c) and sha(dest/'training_originals.tar.gz')==r['archive_sha256'];return r
 dest.mkdir(parents=True,exist_ok=True)
 with (dest/'retrieval.claim').open('x') as f:json.dump({'host':h,'started_unix':time.time()},f)
 v=c['hosts'][h];program='\n'.join(k+'='+repr(x) for k,x in {'ROOT':v['root'],'HOST':h,'EXPECTED':digest(c)}.items())+'\n'+REMOTE
 r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',h,'python3 -'],input=program,text=True,capture_output=True,check=True,timeout=180)
 remote=json.loads(r.stdout);write(dest/'remote_archive_receipt.json',remote)
 subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=15',h+':'+remote['archive_path'],str(dest/'training_originals.tar.gz')],check=True,timeout=600)
 assert sha(dest/'training_originals.tar.gz')==remote['archive_sha256'];original=dest/'original';original.mkdir(exist_ok=False)
 with tarfile.open(dest/'training_originals.tar.gz') as tf:
  for m in tf.getmembers():
   p=Path(m.name);assert not p.is_absolute() and '..' not in p.parts and not m.issym() and not m.islnk()
  tf.extractall(original,filter='data')
 manifests=[]
 for j in [j for j in c['jobs'] if j['host']==h]:
  folder=original/'run'/j['id'];m=read(folder/'terminal_manifest.json');observed=next(r for r in cache['analysis']['rows'] if r['id']==j['id'])
  assert sha(folder/'terminal_manifest.json')==observed['terminal_manifest_sha256'] and m['scientific_success'] and m['contract_sha256']==digest(c)
  for rel,s in m['files'].items():assert sha(folder/rel)==s
  manifests.append({'id':j['id'],'terminal_manifest_sha256':sha(folder/'terminal_manifest.json'),'all_binary_small_SHA_verified':True})
 for rel,s in c['source']['files'].items():assert sha(original/'source'/rel)==s
 for rel,s in c['operation']['files'].items():assert sha(original/rel)==s
 value={**remote,'status':'passed','manifests':manifests,'CPU_binary_reinference_audit':'not_performed'};write(receipt,value);return value
FIELDS=['qty_rmse','qty_mae','time_nll','tail_qty_rmse','tail_qty_mae','tail_time_nll']
def metric(ep):
 m={k:ep[k] for k in FIELDS[:3]};m.update({'tail_'+k:ep['tail'][k] for k in FIELDS[:3]});assert all(math.isfinite(x) for x in m.values());return m

def run():
 c=read(BUNDLE/'execution_contract.json');current=read(BUNDLE/'current.json');assert current['canonical_sha256']==digest(c)
 out=BUNDLE/'analysis';out.mkdir(exist_ok=True);final=out/'completion_receipt.json'
 if final.exists():
  r=read(final);assert r['contract_sha256']==digest(c)
  for rel,s in r['outputs'].items():assert sha(out/rel)==s
  print('already_finalized');return
 if (out/'failure.json').exists():print('prior_failure_preserved_manual_action_required');return
 for h in ['5080','5090']:
  p=BUNDLE/'hourly_monitor'/('terminal_cache_'+h+'.json')
  if not p.exists():write(out/'pending.json',{'status':'pending_verified_terminals','held_out_test_evaluated':False,'remote_retrieval_started':False,'time':time.time()});print('pending_verified_terminals');return
  cache=read(p);expected=[j for j in c['jobs'] if j['host']==h];assert cache['contract_sha256']==digest(c) and cache['analysis']['counts']['complete']==len(expected) and not cache['analysis']['owned_fit_pids']
 with (out/'finalize.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  originals={h:retrieve(c,h) for h in ['5080','5090']};rows=[]
  for j in c['jobs']:
   folder=BUNDLE/'retrieved'/j['host']/'original/run'/j['id'];run=folder/'runs'/j['arm']/'count_only_log_regression'/('seed_'+str(j['seed']));ep=read(run/'endpoint_replays.json');history=read(run/'history.json')['history'];best=min((x for x in history if math.isfinite(x['val_qty_rmse'])),key=lambda x:x['val_qty_rmse'])
   assert ep['evaluation_scope']=='validation_only' and ep['held_out_test_evaluated'] is False and ep['best_epoch']==best['epoch'] and ep['completed_epochs']==len(history)
   assert all(math.isclose(ep['selected'][k],best[v],rel_tol=1e-10,abs_tol=1e-8) for k,v in [('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll')])
   rows.append({'dataset':j['dataset'],'arm':j['arm'],'seed':j['seed'],'best_epoch':best['epoch'],'new_fit':True,'metrics':metric(ep['selected']),'endpoint_sha256':sha(run/'endpoint_replays.json')})
  for d,ref in c['anchors42'].items():
   path=BUNDLE/'source'/ref['endpoint'];assert sha(path)==ref['endpoint_sha256'];ep=read(path);assert ep['evaluation_scope']=='validation_only' and ep['held_out_test_evaluated'] is False
   rows.append({'dataset':d,'arm':'titantpp_cnn_gru54','seed':42,'best_epoch':ref['selected_epoch'],'new_fit':False,'metrics':metric(ep['selected']),'endpoint_sha256':sha(path),'heterogeneous_gpu_reuse':True})
  assert len(rows)==9 and len({(r['dataset'],r['seed']) for r in rows})==9
  grouped=[]
  for d in sorted(c['anchors42']):
   data=[r for r in rows if r['dataset']==d];grouped.append({'dataset':d,'seeds':[42,52,62],'metrics':{k:{'mean':statistics.mean(r['metrics'][k] for r in data),'sample_SD':statistics.stdev(r['metrics'][k] for r in data)} for k in FIELDS}})
  write(out/'Validation_comparison.json',{'scope':'Validation_only','rows':rows,'grouped_3seed':grouped,'held_out_test_evaluated':False,'independent_evaluation':False,'external_superiority':'must compare frozen S2P2 and other external references; not inferred from this completion receipt'})
  with (out/'Validation_per_seed.csv').open('x',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['dataset','arm','seed','best_epoch',*FIELDS]);w.writeheader();w.writerows({**{k:r[k] for k in ['dataset','arm','seed','best_epoch']},**r['metrics']} for r in rows)
  text=['# CNN+GRU54 3seed Validation 완료','', '신규6 + 기존seed42재사용3. selected/last full Validation 및 원본 binary SHA 검증. Test3seed와 독립 평가는 별도이며 CPU 재추론 감사도 미완료입니다.','', '|데이터|전체 RMSE 평균±표본SD|MAE|TimeNLL|큰 수량 RMSE|큰 수량 MAE|큰 수량 TimeNLL|','|---|---:|---:|---:|---:|---:|---:|']
  for g in grouped:
   m=g['metrics'];text.append('|'+g['dataset']+'|'+f"{m['qty_rmse']['mean']:.6g} ± {m['qty_rmse']['sample_SD']:.4g}"+'|'+'|'.join(f"{m[k]['mean']:.6g}" for k in FIELDS[1:])+'|')
  (out/'README.md').write_text('\n'.join(text)+'\n');write(final,{'status':'nine_conditions_Validation_originals_complete','contract_sha256':digest(c),'canonical_conditions':9,'new_fits':6,'reused':3,'originals':originals,'outputs':{n:sha(out/n) for n in ['Validation_comparison.json','Validation_per_seed.csv','README.md']},'held_out_test_evaluated':False,'CPU_binary_reinference_audit':'not_performed','independent_evaluation':False,'time':time.time()});print('validation_complete_originals_SHA_retrieved')
def main():
 try:run()
 except BlockingIOError:print('another_finalizer_owns_lock')
 except Exception as e:
  write(BUNDLE/'analysis/failure.json',{'status':'manual_action_required','error_type':type(e).__name__,'error':str(e),'automatic_retry':False,'held_out_test_evaluated':False,'time':time.time()});raise
if __name__=='__main__':main()
