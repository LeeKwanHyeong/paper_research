"""Read-only hourly observation and one terminal collection for approved RAF6 nonlinear-CNN."""
from __future__ import annotations
import argparse, base64, hashlib, io, json, math, shlex, statistics, subprocess, tarfile, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'search_artifacts/titantpp_backbone_nonlinear_raf6_20261015_v1'
REPORT=ROOT/'reports/titantpp_backbone_nonlinear_raf6_20261015_v1'
REMOTE='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_nonlinear_raf6_20261015_v1_5090'
KST=timezone(timedelta(hours=9))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False);f.write('\n')
def require(ok,msg):
    if not ok:raise ValueError(msg)

def parse_control(raw):
    # Preserve the first supervisor launcher bytes. Its final newline was
    # serialized as literal backslash-n; allow exactly that known suffix only.
    text=raw.decode();obj,end=json.JSONDecoder().raw_decode(text)
    require(text[end:] in ('','\n','\\n'),'Unknown trailing control bytes')
    return obj

REMOTE_CODE=r'''
import base64,hashlib,io,json,math,os,subprocess,tarfile,time
from pathlib import Path
root=Path(ROOT)
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):d.update(b)
 return d.hexdigest()
def canon(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def rd(p):
 s=p.read_text();v,e=json.JSONDecoder().raw_decode(s);assert s[e:] in ('',chr(10),chr(92)+'n');return v
def safe(n):
 p=Path(n);assert not p.is_absolute() and '..' not in p.parts
 f=(root/p).resolve();assert f.is_relative_to(root);return f
assert h(root/'deployment_manifest.json')==MANIFEST_SHA
manifest=rd(root/'deployment_manifest.json')
for n,d in manifest['files'].items():assert h(safe(n))==d,'Deployment SHA mismatch '+n
c=rd(root/'execution_contract.json');digest=manifest['contract_sha256'];assert canon(c)==digest
assert c['hosts']['5090']['root']==str(root)
arms=('titantpp_cnn_channel_nonlinear_v1','titantpp_cnn_joint_nonlinear_v1')
expected=[{'id':f'raf_spare_parts__{s}__{a}','host':'5090','dataset':'raf_spare_parts','seed':s,'arm':a} for a in arms for s in (42,52,62)]
assert c['jobs']==expected and c['arms']==list(arms)
base=[c['hosts']['5090']['python'],str(root/'operation/runtime.py'),'--contract',str(root/'execution_contract.json'),'--host','5090']
files={};values={}
def get(n):
 p=safe(n)
 if not p.exists():return None
 b=p.read_bytes();entry={'raw_sha256':hashlib.sha256(b).hexdigest(),'byte_count':len(b),'base64':base64.b64encode(b).decode()};files[n]=entry
 try:v=rd(p)
 except (ValueError,AssertionError) as e:entry['parse_error']=str(e);return None
 values[n]=v;return v
def probe(pid):
 p=Path('/proc')/str(pid)
 try:before=(p/'stat').read_text()
 except FileNotFoundError:return {'pid':pid,'state':'absent','probe':'initial_stat_enoent'}
 try:
  argv=(p/'cmdline').read_bytes().decode().rstrip('\x00').split('\x00');after=(p/'stat').read_text()
 except FileNotFoundError:return {'pid':pid,'state':'unconfirmed','reason':'vanished_mid_probe'}
 a=before.rsplit(')',1)[1].split();b=after.rsplit(')',1)[1].split()
 return {'pid':pid,'state':'live' if a[19]==b[19] and a[1:3]==b[1:3] and b[0] not in ('Z','X') else 'unconfirmed',
   'ppid':int(b[1]),'pgrp':int(b[2]),'start_ticks':int(b[19]),'argv':argv}
def identity(v):
 return isinstance(v,dict) and type(v.get('pid')) is int and v['pid']>0 and type(v.get('start_ticks')) is int and v['start_ticks']>0 and v.get('contract_sha256')==digest
def finite(v):return type(v) in (float,int) and math.isfinite(v)
def guard(argv,tail):
 return isinstance(argv,list) and len(argv)==len(base)+len(tail)+4 and argv[:3]==['timeout','--signal=TERM','--kill-after=15s'] and argv[3].isdigit() and int(argv[3])>0 and argv[4:]==base+tail
def child_binding(child,launch,tail):
 return bool(identity(child) and identity(launch) and child.get('ppid')==launch['pid'] and child.get('guard_start_ticks')==launch['start_ticks'] and child.get('argv')==base+tail and guard(launch.get('argv'),tail))
def exited(name,launch_name,**fields_required):
 e=values.get(name);l=values.get(launch_name)
 return bool(identity(l) and isinstance(e,dict) and type(e.get('returncode')) is int and e['returncode']==0 and e.get('actual_child_exit_observed') is True and e.get('contract_sha256')==digest and e.get('argv')==l.get('argv') and e.get('worker_group_pid',e.get('pid'))==l['pid'] and e.get('supervisor_pid',e.get('observer_pid'))==l.get('observer_pid') and e.get('launch_sha256')==files.get(launch_name,{}).get('raw_sha256') and all(e.get(k)==v for k,v in fields_required.items()))
def live_matches(observed,saved):
 return bool(identity(saved) and observed and observed['state']=='live' and observed.get('pid')==saved['pid'] and observed.get('start_ticks')==saved['start_ticks'] and observed.get('argv')==saved.get('argv') and observed.get('ppid')==saved.get('ppid'))
for n in ('approval.json','status.json','qualification/receipt.json','qualification/actual_exit.json','qualification/launch.json','training_permit.json',
 'start_permit.json','supervisor.claim','pipeline.claim','pipeline_launch.json','dispatcher_launch.json','failure.json','pipeline_failure.json',
 'train_complete.json','dispatcher_exit.json','pipeline_exit.json','deployment_receipt.json'):
 get(n)
status=values.get('status.json',{});owned=set();jobs=[]
pipeline=values.get('pipeline.claim');pl=values.get('pipeline_launch.json');supervisor=values.get('supervisor.claim');dl=values.get('dispatcher_launch.json')
q=values.get('qualification/receipt.json');ql=values.get('qualification/launch.json');tp=values.get('training_permit.json');start=values.get('start_permit.json');approval=values.get('approval.json')
pipeline_bound=bool(identity(pipeline) and identity(pl) and pipeline['pid']==pl['pid'] and pipeline['start_ticks']==pl['start_ticks'] and pipeline.get('ppid')==pl.get('observer_pid') and pipeline.get('wrapper_start_ticks')==pl.get('observer_start_ticks') and pipeline.get('argv')==pl.get('argv')==base+['--mode','pipeline'])
start_bound=bool(pipeline_bound and isinstance(start,dict) and isinstance(approval,dict) and approval.get('approved') is True and approval.get('contract_sha256')==digest and approval.get('hosts')==['5090'] and approval.get('fresh_fit_count')==6 and approval.get('maximum_concurrent_workers')==3 and start.get('host')=='5090' and start.get('contract_sha256')==digest and start.get('approval_sha256')==canon(approval) and finite(start.get('started_at_unix')) and finite(start.get('deadline_unix')) and start['deadline_unix']-start['started_at_unix']==c['limits']['total_wall_seconds'])
qualification_bound=bool(start_bound and child_binding(q,ql,['--mode','native']) and ql.get('observer_pid')==pipeline['pid'] and ql.get('observer_start_ticks')==pipeline['start_ticks'] and exited('qualification/actual_exit.json','qualification/launch.json',phase='native') and q.get('status')=='passed' and q.get('host')=='5090' and q.get('source_closure')==c['source']['files_sha256'] and q.get('held_out_test_evaluated') is False and q.get('scientific_optimizer_updates')==0 and set(q.get('synthetic',{}))==set(arms) and all(q['synthetic'][a].get('status')=='passed' and q['synthetic'][a].get('arm')==a and q['synthetic'][a].get('contract_sha256')==digest for a in arms))
if qualification_bound:
 cap=q.get('capacity_check',{});qualification_bound=(cap.get('passed') is True and cap.get('maximum_workers')==3 and cap.get('simultaneous_synthetic_workers_executed') is True and cap.get('worker_actual_exit0_count')==3 and bool(cap.get('evidence_files')))
 if qualification_bound:
  for n,d in cap['evidence_files'].items():
   assert h(safe(n))==d,'Qualification admission evidence drift '+n
  for phase in ('ready','trained'):
   v=get('qualification/admission/'+phase+'_concurrency.json');m=(v or {}).get('memory',{})
   qualification_bound=qualification_bound and bool(v and v.get('contract_sha256')==digest and len(v.get('workers',[]))==3 and m.get('gpu_uuid')==c['hosts']['5090']['gpu_uuid'] and finite(m.get('total_bytes')) and m['total_bytes']>0 and finite(m.get('used_bytes')) and 0<=m['used_bytes']<=.8*m['total_bytes'] and m.get('limit_fraction')==.8)
  for admission_id,arm,seed in (('channel42',arms[0],42),('channel52',arms[0],52),('joint62',arms[1],62)):
   prefix='qualification/admission/'+admission_id
   e=get(prefix+'.actual_exit.json');r=get(prefix+'.ready.json');l=get(prefix+'.launch.json')
   qualification_bound=qualification_bound and bool(identity(l) and r and e and exited(prefix+'.actual_exit.json',prefix+'.launch.json',admission_id=admission_id,arm=arm,seed=seed) and r.get('contract_sha256')==digest and r.get('pid')==l['pid'] and r.get('start_ticks')==l['start_ticks'] and l.get('observer_pid')==q['pid'] and l.get('observer_start_ticks')==q['start_ticks'] and l.get('argv')==base+['--mode','admission','--admission-id',admission_id])
   if r and type(r.get('pid')) is int:owned.add(r['pid'])
training_bound=bool(qualification_bound and isinstance(tp,dict) and tp.get('contract_sha256')==digest and tp.get('qualification')==q and tp.get('qualification_sha256')==files['qualification/receipt.json']['raw_sha256'] and tp.get('start_permit_sha256')==canon(start))
supervisor_bound=bool(training_bound and child_binding(supervisor,dl,['--mode','dispatch']) and dl.get('observer_pid')==pipeline['pid'] and dl.get('observer_start_ticks')==pipeline['start_ticks'])
for value in (pipeline,pl,supervisor,dl,q,ql):
 if isinstance(value,dict):
  owned.update(value[k] for k in ('pid','ppid','observer_pid') if type(value.get(k)) is int and value[k]>0)
pipeline_probe=probe(pipeline['pid']) if identity(pipeline) else None
supervisor_probe=probe(supervisor['pid']) if identity(supervisor) else None
pipeline_live=bool(pipeline_bound and live_matches(pipeline_probe,pipeline))
supervisor_live=bool(supervisor_bound and live_matches(supervisor_probe,supervisor) and status.get('supervisor_pid')==supervisor['pid'] and status.get('supervisor_start_ticks')==supervisor['start_ticks'] and status.get('supervisor_argv')==supervisor['argv'] and status.get('contract_sha256')==digest)
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
gpupids={int(row.split(',')[1]) for row in gpu.splitlines() if row.split(',')[0].strip()==c['hosts']['5090']['gpu_uuid']}
for j in c['jobs']:
 name=j['id'];folder='run/'+name
 wi=get('worker_identities/'+name+'.json');ex=get('worker_exits/'+name+'.json');launch=get('worker_launches/'+name+'.json');claim=get('claims/'+name+'.json')
 get(folder+'/status.json');lease=get('leases/'+name+'.json')
 prefix=folder+'/runs/'+j['arm']+'/count_only_log_regression/seed_'+str(j['seed'])+'/'
 history=get(prefix+'history.json');timing=get(prefix+'epoch_timing.json');terminal=get(folder+'/terminal_manifest.json')
 row={'job':j,'state':'waiting','history':history,'timing':timing,'identity':wi,'process':None,'selected_binary_present':False,'terminal_binary_verified':False}
 bound=bool(supervisor_bound and isinstance(claim,dict) and claim.get('job')==j and claim.get('contract_sha256')==digest and finite(claim.get('deadline_unix')) and claim['deadline_unix']<=start['deadline_unix'] and child_binding(wi,launch,['--mode','fit','--job',name,'--deadline',str(claim['deadline_unix'])]) and wi.get('job')==j and wi.get('deadline_unix')==claim['deadline_unix'] and wi.get('supervisor_pid')==supervisor['pid'] and wi.get('supervisor_start_ticks')==supervisor['start_ticks'] and wi.get('pgrp')==launch['pid'] and wi.get('launch_sha256')==files.get('worker_launches/'+name+'.json',{}).get('raw_sha256') and launch.get('job')==j and launch.get('observer_pid')==supervisor['pid'] and launch.get('observer_start_ticks')==supervisor['start_ticks'])
 for value in (wi,launch):
  if isinstance(value,dict):owned.update(value[k] for k in ('pid','ppid','observer_pid') if type(value.get(k)) is int and value[k]>0)
 if terminal:
  assert terminal['contract_sha256']==digest and terminal['job']==j and terminal['status']=='complete' and terminal['scientific_success'] is True and terminal['held_out_test_evaluated'] is False and terminal['evaluation_scope']=='validation_only'
  for n,d in terminal['files'].items():assert h(safe(folder+'/'+n))==d,'Terminal SHA mismatch '+name+'/'+n
  selected=prefix[len(folder)+1:]+'best_val_qty_rmse_model.pt';last=prefix[len(folder)+1:]+'last_epoch_state.pt'
  row['selected_binary_present']=selected in terminal['files'];row['terminal_binary_verified']=selected in terminal['files'] and last in terminal['files']
  exit_bound=bool(bound and exited('worker_exits/'+name+'.json','worker_launches/'+name+'.json',job=j) and ex.get('worker_identity_sha256')==files.get('worker_identities/'+name+'.json',{}).get('raw_sha256') and ex.get('worker_claim_sha256')==files.get('claims/'+name+'.json',{}).get('raw_sha256'))
  row['state']='complete' if exit_bound and row['terminal_binary_verified'] else 'unconfirmed'
 elif wi and identity(wi) and type(wi.get('ppid')) is int:
  pr=probe(wi['pid']);row['process']=pr;gr=probe(wi['ppid'])
  live=bool(bound and pipeline_live and supervisor_live and isinstance(lease,dict) and lease.get('contract_sha256')==digest and lease.get('job')==name and lease.get('supervisor_pid')==supervisor['pid'] and lease.get('supervisor_start_ticks')==supervisor['start_ticks'] and lease.get('worker_group_pid')==launch['pid'] and lease.get('worker_group_start_ticks')==launch['start_ticks'] and finite(lease.get('expires_unix')) and time.time()<lease['expires_unix']<=claim['deadline_unix'] and live_matches(pr,wi) and pr.get('pgrp')==wi['pgrp'] and gr['state']=='live' and gr.get('start_ticks')==launch['start_ticks'] and gr.get('argv')==launch['argv'] and gr.get('ppid')==supervisor['pid'] and wi['pid'] in gpupids)
  row['state']='running' if live and name in status.get('active_jobs',{}) else 'unconfirmed'
 elif wi or ex or launch or claim or name in status.get('active_jobs',{}):row['state']='unconfirmed'
 jobs.append(row)
probes={str(p):probe(p) for p in sorted(owned)}
failed=bool(values.get('failure.json') or values.get('pipeline_failure.json') or any(values.get(n,{}).get('returncode',0)!=0 for n in ('qualification/actual_exit.json','dispatcher_exit.json','pipeline_exit.json')))
complete=values.get('train_complete.json',{})
terminal=bool(not failed and supervisor_bound and all(j['state']=='complete' and j['terminal_binary_verified'] for j in jobs)
 and status.get('status')=='complete' and status.get('contract_sha256')==digest and set(status.get('completed',{}))=={j['id'] for j in c['jobs']}
 and complete.get('status')=='complete' and complete.get('contract_sha256')==digest and complete.get('scientific_worker_count')==6 and complete.get('dispatcher_actual_exit')==0
 and exited('dispatcher_exit.json','dispatcher_launch.json',phase='dispatch') and exited('pipeline_exit.json','pipeline_launch.json')
 and bool(probes) and all(x['state']=='absent' for x in probes.values()) and not (owned&gpupids))
out={'observed_unix':time.time(),'contract_sha256':digest,'jobs':jobs,'files':files,'owner_probes':probes,
 'gpu_compute_pids':sorted(gpupids),'all_training_terminal_verified':terminal,'failure':failed,
 'proof_bindings':{'pipeline':pipeline_bound,'start':start_bound,'qualification':qualification_bound,'training_permit':training_bound,'dispatcher':supervisor_bound},
 'supervisor_wrapper_own_os_exit_directly_observed':False}
archive=b''
if terminal:
 names=[]
 for folder in ('run','worker_exits','worker_identities','worker_launches','claims','leases','progress','qualification','logs'):
  names.extend(str(p.relative_to(root)) for p in (root/folder).rglob('*') if p.is_file())
 names.extend(n for n in ('execution_contract.json','approval.json','start_permit.json','training_permit.json',
  'status.json','supervisor.claim','pipeline.claim','pipeline_launch.json','dispatcher_launch.json','train_complete.json','dispatcher_exit.json','pipeline_exit.json',
  'deployment_receipt.json','pipeline.log') if (root/n).is_file())
 rawmanifest={n:h(safe(n)) for n in sorted(set(names))};buffer=io.BytesIO()
 with tarfile.open(fileobj=buffer,mode='w:gz') as tar:
  for n in rawmanifest:tar.add(safe(n),arcname=n,recursive=False)
 archive=buffer.getvalue();out['retrieval_manifest']=rawmanifest;out['archive_sha256']=hashlib.sha256(archive).hexdigest()
header=json.dumps(out,sort_keys=True,allow_nan=False).encode()
import sys
sys.stdout.buffer.write(str(len(header)).encode()+b'\n'+header+archive)

'''

def metric_rows(endpoint):
    return {'RMSE':endpoint['qty_rmse'],'MAE':endpoint['qty_mae'],
        'Large_RMSE':endpoint['tail']['qty_rmse'],'Large_MAE':endpoint['tail']['qty_mae'],
        'TimeNLL':endpoint['time_nll'],'Large_TimeNLL':endpoint['tail']['time_nll']}

def analyze(c,local):
    rows=[{'arm':'CNN','seed':j['seed'],'best_epoch':j['original']['selected_epoch'],
           **metric_rows(j['original']['validation_reference'])} for j in c['baseline_jobs']]
    for j in c['jobs']:
        folder=local/'run'/j['id'];manifest=read(folder/'terminal_manifest.json')
        require(manifest['job']==j and manifest['held_out_test_evaluated'] is False,'Foreign result')
        for n,d in manifest['files'].items():require(sha(folder/n)==d,'Terminal file mismatch')
        endpoint=read(folder/'runs'/j['arm']/'count_only_log_regression'/f"seed_{j['seed']}"/'endpoint_replays.json')
        history=read(folder/'runs'/j['arm']/'count_only_log_regression'/f"seed_{j['seed']}"/'history.json')['history']
        finite=[r for r in history if type(r.get('val_qty_rmse')) in (int,float) and math.isfinite(r['val_qty_rmse'])]
        require(finite and all(type(r.get('epoch')) is int for r in history)
            and all(a['epoch']<b['epoch'] for a,b in zip(history,history[1:])),'Invalid saved selection history')
        best=min(finite,key=lambda r:r['val_qty_rmse'])
        require(best['epoch']==endpoint['best_epoch'],'Selector not earliest minimum')
        require(endpoint['selected']['count']==6690,'Not full Validation')
        rows.append({'arm':{'titantpp_cnn':'CNN','titantpp_cnn_channel_nonlinear_v1':'Channel','titantpp_cnn_joint_nonlinear_v1':'Joint'}[j['arm']],'seed':j['seed'],
            'best_epoch':endpoint['best_epoch'],**metric_rows(endpoint['selected'])})
    metrics=['RMSE','MAE','Large_RMSE','Large_MAE','TimeNLL','Large_TimeNLL'];aggregates={};differences=[]
    for arm in ('CNN','Channel','Joint'):
        subset=[r for r in rows if r['arm']==arm];require({r['seed'] for r in subset}=={42,52,62},'Missing/duplicate seed')
        aggregates[arm]={k:{'mean':statistics.mean(r[k] for r in subset),'sample_sd':statistics.stdev(r[k] for r in subset)} for k in metrics}
    for seed in (42,52,62):
        group={r['arm']:r for r in rows if r['seed']==seed}
        for a,b in (('Channel','CNN'),('Joint','CNN'),('Joint','Channel')):
            differences.append({'seed':seed,'contrast':a+'−'+b,**{k:{'absolute':group[a][k]-group[b][k],
                'relative_percent':100*(group[a][k]-group[b][k])/group[b][k] if group[b][k]!=0 else None} for k in metrics}})
    result={'split':'validation','dataset':'raf_spare_parts','seed_count':3,'new_fit_count':6,
        'baseline_reference_count':3,'held_out_test_evaluated':False,'rows':rows,'aggregate':aggregates,'seed_differences':differences,
        'comparison_runtime_limitation':c['comparison_runtime_limitation'],
        'primary_contrast':c['primary_contrast'],
        'interpretation':'Post-Test-informed exploratory follow-up; not independent generalization or causal proof.'}
    write(REPORT/'THREE_SEED_VALIDATION.json',result)
    lines=['# RAF 비선형 CNN: 고정 선택 checkpoint의 Validation 비교','',
      '평균±표본SD, seed42·52·62. 모든 지표는 낮을수록 좋음. 큰 수량은 Train 고정 q>200. 기존 CNN3는 저장된 Validation 결과를 재사용.','',
      '| 구조 | RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL | 큰 수량 TimeNLL |','|---|---:|---:|---:|---:|---:|---:|']
    lines[4:4]=[c['comparison_runtime_limitation']+'; Channel·Joint는 같은 5090 환경이며 기존 CNN 비교는 환경 차이를 포함한다.', '']
    for a in aggregates:lines.append('| '+a+' | '+' | '.join(f"{aggregates[a][k]['mean']:.6g} ± {aggregates[a][k]['sample_sd']:.6g}" for k in metrics)+' |')
    lines+=['','| seed | 구조 | 선택 epoch | RMSE | MAE |','|---|---|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:(r['seed'],r['arm'])):lines.append(f"| {r['seed']} | {r['arm']} | {r['best_epoch']} | {r['RMSE']:.6g} | {r['MAE']:.6g} |")
    lines+=['','차이는 앞 구조−뒤 구조. 음수는 개선, 양수는 악화.','',
        '| seed | 비교 | RMSE 차이 | RMSE 상대차이 | MAE 차이 | MAE 상대차이 |','|---|---|---:|---:|---:|---:|']
    for r in differences:lines.append(f"| {r['seed']} | {r['contrast']} | {r['RMSE']['absolute']:+.6g} | {r['RMSE']['relative_percent']:+.3f}% | {r['MAE']['absolute']:+.6g} | {r['MAE']['relative_percent']:+.3f}% |")
    lines+=['','전체 지표의 seed별 차이는 JSON에도 모두 보존한다. 기존 Test를 본 뒤 설계한 탐색적 실험이며, 이번 결과는 Test 개선이나 오류 원인 확정의 근거가 아니다.','']
    with (REPORT/'THREE_SEED_VALIDATION.md').open('x') as f:f.write('\n'.join(lines))
    return result


def brief_round(snapshot):
    rows=[]
    for item in snapshot['jobs']:
        j=item['job'];history=(item.get('history') or {}).get('history',[])
        finite=[x for x in history if isinstance(x.get('val_qty_rmse'),(float,int)) and math.isfinite(x['val_qty_rmse'])]
        best=min(finite,key=lambda x:x['val_qty_rmse']) if finite else None
        timing=(item.get('timing') or {}).get('epochs',[])
        times=[x['elapsed_seconds'] for x in timing if isinstance(x.get('elapsed_seconds'),(int,float)) and math.isfinite(x['elapsed_seconds']) and x['elapsed_seconds']>0][-10:]
        epoch=max([x['epoch'] for x in history],default=0)
        row={'arm':j['arm'],'seed':j['seed'],'state':item['state'],'saved_epoch':epoch or None,'best_epoch':best['epoch'] if best else None,'timing_samples':len(times)}
        if item['state']=='running' and times and best:
            target=min(300,max(40,best['epoch']+40));speed=statistics.median(times)
            row.update(conditional_stop_epoch=target,conditional_stop_seconds=max(0,target-epoch)*speed,epoch300_seconds=max(0,300-epoch)*speed)
        rows.append(row)
    return {'observed_kst':datetime.fromtimestamp(snapshot['observed_unix'],KST).isoformat(),
        'rows':rows,'all_training_terminal_verified':snapshot['all_training_terminal_verified'],'failure':snapshot['failure']}


def advance(integrity_sha):
    integrity=BUNDLE/'followup_integrity.json';require(sha(integrity)==integrity_sha,'Follow-up integrity mismatch')
    for n,d in read(integrity)['files'].items():require(sha(ROOT/n)==d,'Local evidence drift '+n)
    final=BUNDLE/'retrieval/final_receipt.json'
    if final.exists():
        receipt=read(final)
        for n,d in receipt['files'].items():require(sha(ROOT/n)==d,'Final artifact drift')
        print(json.dumps({'status':'complete','remote_calls':0,'receipt':str(final)}));return
    blocked=REPORT/'ACTION_REQUIRED.json'
    if blocked.exists():print(json.dumps({'status':'action_required','remote_calls':0,'receipt':str(blocked)}));return
    round_dir=REPORT/'hourly'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ');round_dir.mkdir(parents=True)
    try:
        require(not (BUNDLE/'retrieval').exists(),'Incomplete retrieval exists; no automatic retry')
        remote='ROOT='+repr(REMOTE)+'\nMANIFEST_SHA='+repr(sha(BUNDLE/'deployment_manifest.json'))+'\n'+REMOTE_CODE
        try:
            proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','5090',shlex.join(['/usr/bin/python3','-c',remote])],capture_output=True,timeout=180)
            stdout,stderr,rc=proc.stdout,proc.stderr,proc.returncode
        except subprocess.TimeoutExpired as exc:
            stdout,stderr,rc=exc.stdout or b'',exc.stderr or b'',None
        for name,value in [('transport.stdout',stdout),('transport.stderr',stderr)]:
            with (round_dir/name).open('xb') as stream:stream.write(value)
        write(round_dir/'transport.json',{'actual_transport_exit':rc,'timeout':rc is None,
            'stdout_sha256':sha(round_dir/'transport.stdout'),'stderr_sha256':sha(round_dir/'transport.stderr'),
            'stdout_bytes':len(stdout),'stderr_bytes':len(stderr),'truncated':False})
        require(rc==0,'Observation transport/query failed; not a scientific failure')
        count,body=stdout.split(b'\n',1);size=int(count);snapshot=json.loads(body[:size]);archive=body[size:]
        write(round_dir/'round.json',snapshot);brief=brief_round(snapshot);write(round_dir/'summary.json',brief)
        pointer={'round':str(round_dir/'round.json'),'round_sha256':sha(round_dir/'round.json'),'summary':str(round_dir/'summary.json'),'summary_sha256':sha(round_dir/'summary.json')}
        tmp=REPORT/'latest.pending.json';write(tmp,pointer);tmp.replace(REPORT/'latest.json')
        require(not snapshot['failure'],'Owned campaign failed; preserve records, no retry')
        if not snapshot['all_training_terminal_verified']:
            print(json.dumps({'status':'running_or_unconfirmed',**brief},ensure_ascii=False));return
        require(archive and hashlib.sha256(archive).hexdigest()==snapshot['archive_sha256'],'Archive mismatch')
        dest=BUNDLE/'retrieval';dest.mkdir(exist_ok=False);write(dest/'intent.json',pointer)
        (dest/'originals.tar.gz').write_bytes(archive);local=dest/'originals';local.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as t:
            names=set()
            for member in t.getmembers():
                p=Path(member.name);require(member.isfile() and not p.is_absolute() and '..' not in p.parts and member.name not in names,'Unsafe/duplicate archive member')
                names.add(member.name);target=local/p;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(t.extractfile(member).read())
            require(names==set(snapshot['retrieval_manifest']),'Archive inventory differs')
        for n,d in snapshot['retrieval_manifest'].items():require(sha(local/n)==d,'Retrieved file SHA mismatch')
        write(dest/'manifest.json',snapshot['retrieval_manifest'])
        c=read(BUNDLE/'execution_contract.json');analyze(c,local)
        files={str(p.relative_to(ROOT)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
        for n in ('THREE_SEED_VALIDATION.json','THREE_SEED_VALIDATION.md'):files[str((REPORT/n).relative_to(ROOT))]=sha(REPORT/n)
        write(final,{'status':'complete','new_fit_count':6,'baseline_reference_count':3,'seed_count':3,
            'held_out_test_evaluated':False,'all_training_terminal_verified':True,'files':files})
        print(json.dumps({'status':'complete','receipt':str(final),'report':str(REPORT/'THREE_SEED_VALIDATION.md')},ensure_ascii=False))
    except BaseException as exc:
        if not blocked.exists():write(blocked,{'status':'action_required','type':type(exc).__name__,'message':str(exc),'round':str(round_dir),'automatic_retry':False})
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--integrity-sha',required=True);a=p.parse_args();advance(a.integrity_sha)
