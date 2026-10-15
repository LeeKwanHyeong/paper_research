"""Read-only hourly observation and one terminal collection for approved Taxi6."""
from __future__ import annotations
import argparse, base64, hashlib, io, json, math, shlex, statistics, subprocess, tarfile, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
ROOT=Path(__file__).resolve().parents[2]
BUNDLE=ROOT/'search_artifacts/titantpp_cnn_history_adaptation_matched_20261015_v1/execution_v1'
REPORT=ROOT/'reports/titantpp_cnn_history_adaptation_matched_20261015_v1/execution_v1'
REMOTE='/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_history_taxi6_20261015_v1_5080'
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
import base64,hashlib,io,json,os,subprocess,tarfile,time
from pathlib import Path
root=Path(ROOT)
def h(p):
 d=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):d.update(b)
 return d.hexdigest()
def rd(p):
 s=p.read_text();v,e=json.JSONDecoder().raw_decode(s);assert s[e:] in ('',chr(10),chr(92)+'n');return v
def safe(n):
 p=Path(n);assert not p.is_absolute() and '..' not in p.parts
 f=(root/p).resolve();assert f.is_relative_to(root);return f
assert h(root/'deployment_manifest.json')==MANIFEST_SHA
manifest=rd(root/'deployment_manifest.json')
for n,d in manifest['files'].items():assert h(safe(n))==d,'Deployment SHA mismatch '+n
c=rd(root/'execution_contract.json');digest=manifest['contract_sha256']
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
for n in ('status.json','qualification/receipt.json','qualification/actual_exit.json','training_permit.json',
 'start_permit.json','supervisor.claim','pipeline.claim','failure.json','pipeline_failure.json',
 'train_complete.json','dispatcher_exit.json','pipeline_exit.json','deployment_receipt.json'):
 get(n)
status=values.get('status.json',{});owned=set();jobs=[]
dispatcher=probe(status['supervisor_pid']) if status.get('supervisor_pid') else None
pipeline=probe(values['pipeline.claim']['pid']) if values.get('pipeline.claim') else None
dispatch_live=(dispatcher and dispatcher['state']=='live' and dispatcher['argv']==status.get('supervisor_argv'))
pipeline_live=(pipeline and pipeline['state']=='live' and str(root/'operation/runtime.py') in pipeline['argv'] and pipeline['argv'][-2:]==['--mode','pipeline'])
for n in ('supervisor.claim','pipeline.claim'):
 if values.get(n):owned.add(values[n]['pid'])
for n in ('dispatcher_exit.json','pipeline_exit.json'):
 if values.get(n):
  owned.update(values[n].get(k) for k in ('pid','observer_pid','worker_group_pid') if values[n].get(k))
gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
gpupids={int(row.split(',')[1]) for row in gpu.splitlines() if row.split(',')[0].strip()==c['hosts']['5080']['gpu_uuid']}
for j in c['jobs']:
 name=j['id'];folder='run/'+name;identity=get('worker_identities/'+name+'.json');exit=get('worker_exits/'+name+'.json')
 get(folder+'/status.json');lease=get('leases/'+name+'.json')
 if identity:owned.update((identity['pid'],identity['ppid']))
 prefix=folder+'/runs/'+j['arm']+'/count_only_log_regression/seed_'+str(j['seed'])+'/'
 history=get(prefix+'history.json');timing=get(prefix+'epoch_timing.json');terminal=get(folder+'/terminal_manifest.json')
 row={'job':j,'state':'waiting','history':history,'timing':timing,'identity':identity,'process':None,'selected_binary_present':False}
 if terminal:
  assert terminal['contract_sha256']==digest and terminal['job']==j and terminal['scientific_success'] and not terminal['held_out_test_evaluated']
  for n,d in terminal['files'].items():assert h(safe(folder+'/'+n))==d,'Terminal SHA mismatch '+name+'/'+n
  row['selected_binary_present']=any(n.endswith('best_val_qty_rmse_model.pt') for n in terminal['files'])
  row['state']='complete' if exit and exit['returncode']==0 and exit['actual_child_exit_observed'] else 'unconfirmed'
 elif identity:
  pr=probe(identity['pid']);row['process']=pr
  guard=probe(identity['ppid']);expected_guard=status.get('active_jobs',{}).get(name,{}).get('worker_guard_argv')
  live=(dispatch_live and pipeline_live and lease and time.time()<lease['expires_unix']
   and guard['state']=='live' and guard['argv']==expected_guard and guard['ppid']==status.get('supervisor_pid')
   and pr['state']=='live' and pr['argv']==identity['argv'] and pr['ppid']==identity['ppid'] and pr['pgrp']==identity['pgrp'] and identity['pid'] in gpupids)
  row['state']='running' if live and name in status.get('active_jobs',{}) else 'unconfirmed'
 elif name in status.get('active_jobs',{}):row['state']='unconfirmed'
 jobs.append(row)
probes={str(p):probe(p) for p in sorted(owned)}
failed=bool(values.get('failure.json') or values.get('pipeline_failure.json') or any(values.get(n,{}).get('returncode',0)!=0 for n in ('qualification/actual_exit.json','dispatcher_exit.json','pipeline_exit.json')))
terminal=(not failed and all(j['state']=='complete' and j['selected_binary_present'] for j in jobs)
 and status.get('status')=='complete' and values.get('train_complete.json',{}).get('scientific_worker_count')==6
 and all(values.get(n,{}).get('returncode')==0 and values[n].get('actual_child_exit_observed') for n in ('dispatcher_exit.json','pipeline_exit.json'))
 and bool(probes) and all(x['state']=='absent' for x in probes.values()) and not (owned&gpupids))
out={'observed_unix':time.time(),'contract_sha256':digest,'jobs':jobs,'files':files,'owner_probes':probes,
 'gpu_compute_pids':sorted(gpupids),'all_training_terminal_verified':terminal,'failure':failed,
 'supervisor_wrapper_own_os_exit_directly_observed':False}
archive=b''
if terminal:
 names=[]
 for folder in ('run','worker_exits','worker_identities','claims','leases','progress','qualification','logs'):
  names.extend(str(p.relative_to(root)) for p in (root/folder).rglob('*') if p.is_file())
 names.extend(n for n in ('execution_contract.json','approval.json','start_permit.json','training_permit.json',
  'status.json','supervisor.claim','pipeline.claim','train_complete.json','dispatcher_exit.json','pipeline_exit.json',
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
    rows=[]
    for j in c['baseline_jobs']:
        rows.append({'arm':'CNN','seed':j['seed'],'best_epoch':j['original']['selected_epoch'],
            **metric_rows(j['original']['validation_reference'])})
    for j in c['jobs']:
        folder=local/'run'/j['id'];manifest=read(folder/'terminal_manifest.json')
        require(manifest['job']==j and manifest['held_out_test_evaluated'] is False,'Foreign result')
        for n,d in manifest['files'].items():require(sha(folder/n)==d,'Terminal file mismatch')
        endpoint=read(folder/'runs'/j['arm']/'count_only_log_regression'/f"seed_{j['seed']}"/'endpoint_replays.json')
        history=read(folder/'runs'/j['arm']/'count_only_log_regression'/f"seed_{j['seed']}"/'history.json')['history']
        best=min(history,key=lambda r:r['val_qty_rmse'])
        require(best['epoch']==endpoint['best_epoch'],'Selector not earliest minimum')
        require(endpoint['selected']['count']==8268,'Not full Validation')
        rows.append({'arm':'Adaptive' if 'adaptive' in j['arm'] else 'Capacity','seed':j['seed'],
            'best_epoch':endpoint['best_epoch'],**metric_rows(endpoint['selected'])})
    metrics=['RMSE','MAE','Large_RMSE','Large_MAE','TimeNLL','Large_TimeNLL'];aggregates={};differences=[]
    for arm in ('CNN','Adaptive','Capacity'):
        subset=[r for r in rows if r['arm']==arm];require({r['seed'] for r in subset}=={42,52,62},'Missing/duplicate seed')
        aggregates[arm]={k:{'mean':statistics.mean(r[k] for r in subset),'sample_sd':statistics.stdev(r[k] for r in subset)} for k in metrics}
    for seed in (42,52,62):
        group={r['arm']:r for r in rows if r['seed']==seed}
        for a,b in (('Adaptive','CNN'),('Capacity','CNN'),('Adaptive','Capacity')):
            differences.append({'seed':seed,'contrast':a+'−'+b,**{k:{'absolute':group[a][k]-group[b][k],
                'relative_percent':100*(group[a][k]-group[b][k])/group[b][k] if group[b][k]!=0 else None} for k in metrics}})
    result={'split':'validation','dataset':'yellow_trip_hourly','seed_count':3,'new_fit_count':6,
        'baseline_reference_count':3,'held_out_test_evaluated':False,'rows':rows,'aggregate':aggregates,'seed_differences':differences,
        'interpretation':'Post-Test-informed exploratory follow-up; not independent generalization or causal proof.'}
    write(REPORT/'THREE_SEED_VALIDATION.json',result)
    lines=['# Taxi 이력 CNN: 고정 선택 checkpoint의 Validation 비교','',
      '평균±표본SD, seed42·52·62. 모든 지표는 낮을수록 좋음. 큰 수량은 Train 고정 q>3449. 기존 CNN은 그대로 재사용.','',
      '| 구조 | RMSE | MAE | 큰 수량 RMSE | 큰 수량 MAE | TimeNLL | 큰 수량 TimeNLL |','|---|---:|---:|---:|---:|---:|---:|']
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
        proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','5080',shlex.join(['/usr/bin/python3','-c',remote])],capture_output=True,timeout=180)
        (round_dir/'transport.stdout').write_bytes(proc.stdout);(round_dir/'transport.stderr').write_bytes(proc.stderr)
        write(round_dir/'transport.json',{'actual_transport_exit':proc.returncode,'stdout_sha256':sha(round_dir/'transport.stdout'),'stderr_sha256':sha(round_dir/'transport.stderr')})
        require(proc.returncode==0,'Observation transport/query failed; not a scientific failure')
        count,body=proc.stdout.split(b'\n',1);size=int(count);snapshot=json.loads(body[:size]);archive=body[size:]
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
