"""One read-only SSH per launched live owner; terminal originals keep timestamp."""
from __future__ import annotations
import argparse
import concurrent.futures
from datetime import datetime,timezone,timedelta
import json
import math
from pathlib import Path
import statistics
import time
import sys
_PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:sys.path.insert(0,str(_PROJECT_ROOT))
from paper.scripts import titantpp_contribution_runtime as op
from paper.scripts.deploy_titantpp_contribution_campaigns import bundle,local_contract,remote,prefix
from paper.scripts.prepare_titantpp_contribution_campaigns import REPORT,write

def kst(t):return datetime.fromtimestamp(t,timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')

def eta(timing,epoch,best):
    rows=[r for r in timing.get('epochs',[]) if isinstance(r.get('elapsed_seconds'),(int,float)) and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>0 and 0<r.get('epoch',0)<=epoch][-10:]
    if not rows:return {'status':'unconfirmed_no_own_completed_epoch_timing'}
    median=statistics.median(r['elapsed_seconds'] for r in rows);end=min(300,max(40,best+40))
    return {'median_epoch_seconds':median,'sample_epochs':len(rows),'no_new_best_end_epoch':end,
        'no_new_best_remaining_seconds':max(0,end-epoch)*median,'max300_remaining_seconds':max(0,300-epoch)*median,
        'assumptions':'own GPU/fit recent speed retained; new best changes ETA; endpoint/retrieval/CPU audit excluded'}

def owned_processes_match(c,host,job,claim,out,status,timeout,supervisor,worker):
    if not all((claim,timeout,supervisor,worker,out.get('lease'))):return False
    root=c['hosts'][host]['root'];python=c['hosts'][host]['python'];digest=op.sha_json(c)
    deadline=claim.get('deadline_unix');lease=out['lease'];guard=status.get('worker_guard_argv')
    if not isinstance(deadline,(float,int)) or not isinstance(guard,list):return False
    sup=[python,root+'/operation/campaign.py','--contract',root+'/execution_contract.json','--host',host,'--mode','dispatch']
    fit=[python,root+'/operation/campaign.py','--contract',root+'/execution_contract.json','--host',host,'--mode','fit','--job',job['id'],'--deadline',str(deadline)]
    if len(guard)!=len(fit)+4 or guard[:3]!=['timeout','--signal=TERM','--kill-after=15s'] or guard[4:]!=fit:return False
    try:
        if not 0<int(guard[3])<=c['limits']['per_condition_seconds']:return False
    except (ValueError,TypeError):return False
    return bool(claim.get('job')==job and claim.get('contract_sha256')==digest
        and status.get('condition_deadline_unix')==deadline and status.get('supervisor_argv')==sup
        and supervisor['argv']==sup and timeout['argv']==guard and worker['argv']==fit
        and timeout['ppid']==supervisor['pid'] and worker['ppid']==timeout['pid']
        and lease.get('contract_sha256')==digest and lease.get('job')==job['id']
        and lease.get('supervisor_pid')==supervisor['pid'] and lease.get('worker_group_pid')==timeout['pid']
        and lease.get('issued_unix',math.inf)<=out['observed_unix']<lease.get('expires_unix',0)<=deadline)

def observe_host(stage,host):
    b,c=local_contract(stage);jobs=[j for j in c['jobs'] if j['host']==host];monitor=b/'hourly_monitor'
    cache=monitor/f'terminal_cache_{host}.json'
    if cache.exists():
        pointer=op.read(monitor/f'terminal_pointer_{host}.json');assert op.sha_file(cache)==pointer['sha256']
        out=op.read(cache);assert out['contract_sha256']==op.sha_json(c) and out['verified_terminal'] is True
        return {**out,'observation_source':'verified_terminal_cache','remote_calls':0}
    if not (b/'deployment'/f'launch_receipt_{host}.json').exists():
        return {'stage':stage,'host':host,'counts':{'complete':0,'failed':0,'running':0,'waiting':len(jobs),'unconfirmed':0},
            'status':'not_launched','remote_calls':0,'observed_kst':None}
    code=prefix(c,host,deployed=False)+'''import shlex
def readif(path):return json.loads(path.read_text()) if path.exists() else None
assert json.loads((b/'execution_contract.json').read_text())==c,'Remote contract drift'
for p,v in {**c['source']['files'],**c['input_files']}.items():assert sha(b/'source'/p)==v,'Source/input drift '+p
for p,v in c['operation']['files'].items():assert sha(b/p)==v,'Operation drift '+p
ps=subprocess.run(['ps','-eo','pid=,ppid=,args='],capture_output=True,text=True,check=True).stdout
processes=[]
for line in ps.splitlines():
 fields=line.strip().split(None,2)
 if len(fields)==3 and str(b) in fields[2]:processes.append({'pid':int(fields[0]),'ppid':int(fields[1]),'argv':shlex.split(fields[2])})
out={'stage':c['stage'],'host':%r,'contract_sha256':canonical(c),'observed_unix':time.time(),'GPU':g,'gpu_processes':compute,
 'processes':processes,'status':readif(b/'status.json'),'failure':readif(b/'failure.json'),'supervisor_exit':readif(b/'supervisor_exit.json'),
 'lease':readif(b/'server_lease.json'),'jobs':{},'rawSHA_verified':True}
for job in c['jobs']:
 if job['host']!=%r:continue
 folder=b/'run'/job['id'];run=folder/'runs'/job['arm']/'count_only_log_regression'/('seed_'+str(job['seed']))
 rec={'job':job,'status':readif(folder/'status.json'),'claim':readif(b/'claims'/(job['id']+'.json')),'terminal':readif(folder/'terminal_manifest.json')}
 if rec['terminal']:
  m=rec['terminal'];assert m['job']==job and m['contract_sha256']==canonical(c) and m['scientific_success'] is True
  for p,v in m['files'].items():assert sha(folder/p)==v,'Terminal original drift '+p
  rec.update(terminal_manifest_sha256=sha(folder/'terminal_manifest.json'),endpoint=readif(run/'endpoint_replays.json'),history=readif(run/'history.json'),timing=readif(run/'epoch_timing.json'))
 else:
  receipt=readif(run/'server_checkpoint_receipt.json');rec['checkpoint_receipt']=receipt
  if receipt:
   snapshots={p:(run/p).read_bytes() for p in ('history.json','epoch_timing.json') if (run/p).exists()}
   matches=all((hashlib.sha256(snapshots[p]).hexdigest() if p in snapshots else sha(run/p) if (run/p).exists() else None)==v for p,v in receipt['files'].items())
   same=receipt==readif(run/'server_checkpoint_receipt.json');rec['checkpoint_stable']=matches and same
   if rec['checkpoint_stable']:rec.update(history=json.loads(snapshots['history.json']),timing=json.loads(snapshots['epoch_timing.json']))
 out['jobs'][job['id']]=rec
print(json.dumps(out))
'''%(host,host)
    out=remote(c,host,code,'observe',timeout=240);out['observed_kst']=kst(out['observed_unix']);out['remote_calls']=1
    counts={'complete':0,'failed':0,'running':0,'waiting':0,'unconfirmed':0};active=None
    status=out['status'] or {};pids={p['pid']:p for p in out['processes']}
    for j in jobs:
        r=out['jobs'][j['id']]
        if r.get('terminal'):counts['complete']+=1;continue
        if out['failure'] and (out['failure'].get('active_job') or {}).get('id')==j['id']:counts['failed']+=1;continue
        if (status.get('active_job') or {}).get('id')==j['id']:
            timeout=pids.get(status.get('worker_group_pid'));supervisor=pids.get(status.get('supervisor_pid'))
            worker=next((p for p in out['processes'] if timeout and p['ppid']==timeout['pid'] and '--mode' in p['argv'] and p['argv'][p['argv'].index('--mode')+1]=='fit'),None)
            valid=owned_processes_match(c,host,j,r.get('claim'),out,status,timeout,supervisor,worker)
            gpu_pids={int(x.split(',')[1].strip()) for x in out['gpu_processes'].splitlines() if c['hosts'][host]['gpu_uuid'] in x}
            valid=valid and (not gpu_pids or gpu_pids=={worker['pid']})
            counts['running' if valid else 'unconfirmed']+=1
            receipt=r.get('checkpoint_receipt');history=(r.get('history') or {}).get('history',[])
            best=min((x for x in history if isinstance(x.get('val_qty_rmse'),(int,float)) and math.isfinite(x['val_qty_rmse'])),key=lambda x:x['val_qty_rmse'],default=None)
            epoch=receipt['epoch'] if receipt and r.get('checkpoint_stable') else None
            active={'job':j,'owned_PID_verified':bool(valid),'worker_PID':worker['pid'] if worker else None,'timeout_PID':timeout['pid'] if timeout else None,'supervisor_PID':supervisor['pid'] if supervisor else None,'stored_epoch':epoch,'best_epoch':best.get('epoch') if best else None,
                'first_checkpoint_not_saved':receipt is None,'checkpoint_stable':r.get('checkpoint_stable'),
                'ETA':eta(r.get('timing') or {},epoch,best['epoch']) if epoch and best else {'status':'unconfirmed'}}
        elif r['claim'] or r['status']:counts['unconfirmed']+=1
        else:counts['waiting']+=1
    verified=counts['complete']==len(jobs) and not out['processes'] and not out['gpu_processes'].strip() and out['supervisor_exit'] and out['supervisor_exit']['returncode']==0 and status.get('status')=='complete'
    out.update(counts=counts,active=active,verified_terminal=bool(verified),observation_source='live_read_only_SSH')
    path=monitor/f'observation_{host}_{time.time_ns()}.json';write(path,out)
    latest=monitor/f'latest_{host}.json';latest.parent.mkdir(parents=True,exist_ok=True);latest.write_text(json.dumps({'path':str(path),'sha256':op.sha_file(path),'observed_kst':out['observed_kst']},indent=2)+'\n')
    if verified:write(cache,out);write(monitor/f'terminal_pointer_{host}.json',{'sha256':op.sha_file(cache)})
    return out

def observe(stage):
    result={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures={h:pool.submit(observe_host,stage,h) for h in ('5080','5090')}
        for h,f in futures.items():
            try:result[h]=f.result()
            except Exception as exc:result[h]={'host':h,'stage':stage,'status':'retrieval_failed_not_training_failure','type':type(exc).__name__,'message':str(exc)}
    REPORT.mkdir(parents=True,exist_ok=True);name=f'STATUS_{stage}_{time.time_ns()}'
    write(REPORT/(name+'.json'),result)
    lines=['# '+stage+' GPU 관측','', '|서버|실제 관측시각|완료/실패/진행/대기/미확정|진행 조건|저장/best epoch|조건부 ETA|','|---|---|---|---|---|---|']
    for h,r in result.items():
        count=r.get('counts');a=r.get('active') or {};j=a.get('job') or {};e=a.get('ETA') or {}
        counts='/'.join(str(count[k]) for k in ('complete','failed','running','waiting','unconfirmed')) if count else '조회 실패'
        desc=f"{j.get('dataset','—')} / {j.get('arm','—')} / seed{j.get('seed','—')}" if j else r.get('status','종료')
        epochs='첫 저장 전' if a.get('first_checkpoint_not_saved') else f"{a.get('stored_epoch','—')}/{a.get('best_epoch','—')}"
        et=f"best 미갱신 {e['no_new_best_remaining_seconds']/3600:.2f}h / E300 {e['max300_remaining_seconds']/3600:.2f}h" if 'median_epoch_seconds' in e else '미확정'
        lines.append(f"|{h}|{r.get('observed_kst') or '원격 미관측'}|{counts}|{desc}|{epochs}|{et}|")
    lines+=['','ETA는 해당 fit의 최근 최대10 완료 epoch 중앙값과 best 미갱신·속도 유지 가정이다. 전체 큐 ETA와 원본 회수·CPU 감사는 별도다.','', 'Test·새 독립 평가는 수행하지 않았다.']
    md=REPORT/(name+'.md');md.write_text('\n'.join(lines)+'\n')
    (REPORT/'latest.json').write_text(json.dumps({'json':str(REPORT/(name+'.json')),'json_sha256':op.sha_file(REPORT/(name+'.json')),'markdown':str(md),'markdown_sha256':op.sha_file(md)},indent=2)+'\n')
    return {h:{k:v for k,v in r.items() if k in ('status','counts','active','observed_kst','verified_terminal','remote_calls','message')} for h,r in result.items()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=tuple(op.STAGES),required=True);a=p.parse_args();print(json.dumps(observe(a.stage),ensure_ascii=False))
