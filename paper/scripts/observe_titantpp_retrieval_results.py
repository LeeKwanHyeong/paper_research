"""Read-only result recovery for terminal exposure append-one; no training/evaluation."""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

NAME = 'titantpp_contextual_retrieval_pilot_repair_20261007_v1'
PARENT_SHA = '5372f2ec93c98e6b45ecf37d2b19ae1375a838fdfd9eb8a9c4f3a908456e7953'
PROJECT = Path(__file__).resolve().parents[2] if __file__ != '<stdin>' else Path.cwd()
BUNDLE = PROJECT / 'search_artifacts' / NAME
HOSTS = ('5080', '5090')
STATES = ('complete', 'failed', 'running', 'waiting', 'uncertain')
KST = dt.timezone(dt.timedelta(hours=9))

def read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()

def safe(root, name):
    root = Path(root).resolve(); p = Path(name)
    if p.is_absolute() or '..' in p.parts or '\\' in str(p): raise ValueError('Unsafe evidence path')
    out = root / p
    if not out.resolve().is_relative_to(root) or any(x.is_symlink() for x in (out, *out.parents) if x.is_relative_to(root)):
        raise ValueError('Evidence path escapes owned root')
    return out

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


RECOVERY_PROFILE = 'control/result_recovery_20261008_v1/profile.json'

def raw_record_sha(value):
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()).hexdigest()

def terminal_exposure_bridge(c, row):
    """Prove final Validation append-one without changing either historical record."""
    try:
        job = row['job']; terminal = row['terminal']; receipt = row['checkpoint_receipt']
        if (job not in c['jobs'] or terminal.get('status') != 'complete' or terminal.get('scientific_success') is not True
                or terminal.get('job') != job or terminal.get('contract_sha256') != canonical(c)
                or terminal.get('parent_contract_sha256') != PARENT_SHA or terminal.get('evaluation_scope') != 'validation_only'
                or terminal.get('held_out_test_evaluated') is not False or row.get('SHA_verified') is not True
                or receipt.get('contract_sha256') != canonical(c) or receipt.get('parent_contract_sha256') != PARENT_SHA):
            return False
        n = receipt['epoch']; exp = row['terminal_exposure']; files = receipt['files']
        if (type(n) is not int or not 40 <= n <= 300 or set(exp) != {'train', 'validation'}
                or len(exp['train']) != n or len(exp['validation']) != n + 1
                or set(files) != {'last_epoch_state.pt', 'history.json', 'exposure.json', 'epoch_timing.json', 'train.log'}):
            return False
        final_files = terminal['files']; mismatches = []
        for name, digest in files.items():
            candidates = [value for key, value in final_files.items() if key.endswith('/' + name)]
            if len(candidates) != 1:
                return False
            if candidates[0] != digest:
                mismatches.append(name)
        if mismatches != ['exposure.json']:
            return False
        final_sha = next(v for k, v in final_files.items() if k.endswith('/exposure.json'))
        if raw_record_sha(exp) != final_sha:
            return False
        prefix = {**exp, 'validation': exp['validation'][:-1]}
        if raw_record_sha(prefix) != files['exposure.json']:
            return False
        data = next(d for d in c['parent_contract']['datasets'] if d['dataset_id'] == job['dataset'])
        for split in ('train', 'validation'):
            population = data['inherited_data_identity']['populations'][split]['target_count']
            batches = math.ceil(population / data['loader']['batch_size'])
            for i, record in enumerate(exp[split], 1):
                digest = record.get('batch_order_sha256', '')
                if (record.get('epoch') != i or record.get('split') != split or record.get('count') != population
                        or record.get('batches') != batches or not isinstance(digest, str) or len(digest) != 64
                        or any(x not in '0123456789abcdef' for x in digest)):
                    return False
        return len({r['batch_order_sha256'] for r in exp['validation']}) == 1
    except (KeyError, ValueError, TypeError, StopIteration, OverflowError):
        return False

def validate_recovery_profile(bundle):
    bundle = Path(bundle).resolve(); profile = read(bundle / RECOVERY_PROFILE)
    if not profile or profile.get('schema') != 'titantpp_retrieval_result_recovery_20261008_v1':
        raise ValueError('Missing result recovery profile')
    current = read(bundle / 'current.json')
    for key, path in [('original_current_sha256', bundle / 'current.json'),
                      ('original_control_integrity_sha256', bundle / 'control/integrity_receipt.json')]:
        if profile.get(key) != sha(path):
            raise ValueError('Original recovery binding changed: ' + key)
    if profile.get('contract_sha256') != current.get('canonical_sha256'):
        raise ValueError('Recovery scientific contract changed')
    integrity = read(bundle / 'control/integrity_receipt.json')
    for section, root in [('files', bundle), ('project_files', PROJECT)]:
        for name, digest in integrity[section].items():
            if sha(safe(root, name)) != digest:
                raise ValueError('Original frozen control changed: ' + name)
    for section, root in [('files', bundle), ('project_files', PROJECT)]:
        for name, digest in profile[section].items():
            if sha(safe(root, name)) != digest:
                raise ValueError('Result recovery source changed: ' + name)
    if (profile.get('selection_changed') is not False or profile.get('new_training') is not False
            or profile.get('new_evaluation') is not False or profile.get('held_out_test_evaluated') is not False):
        raise ValueError('Recovery scope changed')
    return profile


def preflight(bundle, *, native=False):
    bundle = Path(bundle); c = read(bundle / 'execution_contract.json')
    if not c or c['schema'] != NAME or canonical(c['parent_contract']) != PARENT_SHA or c['parent_contract_sha256'] != PARENT_SHA:
        raise ValueError('Repair/parent contract binding mismatch')
    if not native:
        current = read(bundle / 'current.json')
        if not current or current.get('canonical_sha256') != canonical(c) or current.get('hosts') != {h:s['root'] for h,s in c['hosts'].items()}:
            raise ValueError('Repair current pointer mismatch')
    parent = c['parent_contract']
    for closure, root in ((parent['source'], bundle/'source'), (c['operation'], bundle)):
        if canonical(closure['files']) != closure['files_sha256']: raise ValueError('File closure mismatch')
        for name, digest in closure['files'].items():
            if sha(safe(root, name)) != digest: raise ValueError('Sealed file SHA changed: '+name)
    if canonical(parent['operation']['files']) != parent['operation']['files_sha256']: raise ValueError('Parent operation closure mismatch')
    for name, digest in parent['operation']['files'].items():
        if sha(safe(bundle/'parent_operation', name)) != digest: raise ValueError('Parent operation SHA changed')
    for name, digest in parent['input_files'].items():
        if sha(safe(bundle/'source', name)) != digest: raise ValueError('Input SHA changed')
    runtime_path = bundle/'operation/repair_runtime.py'
    module_spec = importlib.util.spec_from_file_location('_repair_observer_validation', runtime_path)
    module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
    module.validate(c, verify_files=False)
    return c

def collect(c, host):
    """Native read-only collection. CPU checkpoint metadata, never predictions/Test."""
    spec = c['hosts'][host]; root = Path(spec['root'])
    preflight(root, native=True)
    if canonical(read(root/'execution_contract.json')) != canonical(c): raise ValueError('Native contract changed')
    binding = c['qualification_bindings'][host]
    if sha(safe(root,binding['receipt_file'])) != binding['receipt_sha256']: raise ValueError('Qualification original SHA changed')
    start=read(root/'start_permit.json');permit=read(root/'training_permit.json')
    if (not start or not permit or start.get('host')!=host or start.get('contract_sha256')!=canonical(c)
            or permit.get('host')!=host or permit.get('contract_sha256')!=canonical(c)
            or permit.get('start_permit_sha256')!=canonical(start) or permit.get('qualification_sha256')!=binding['receipt_sha256']):
        raise ValueError('Native authorization binding changed')
    entry = str(root/'operation/repair_runtime.py')
    ps = subprocess.run(['ps','-eo','pid,ppid,args'], capture_output=True, text=True, check=True).stdout
    processes = [s.strip() for s in ps.splitlines() if entry in s and ('--mode worker' in s or '--mode dispatch' in s)]
    gpu = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used','--format=csv,noheader,nounits'], capture_output=True,text=True,check=True).stdout
    gp = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader,nounits'], capture_output=True,text=True,check=True).stdout
    inventory = [s.split(',')[0].strip() for s in gpu.splitlines() if s.strip()]
    if inventory != [spec['gpu_uuid']]: raise ValueError('GPU UUID mismatch')
    rows = []
    for job in (j for j in c['jobs'] if j['host'] == host):
        folder = root/'run'/job['id']; run = folder/'runs'/job['arm']/'count_only_log_regression'/f"seed_{job['seed']}"
        row = {'job':job, 'started':folder.exists(), 'claim':read(root/'claims'/(job['id']+'.json')),
            'history':read(run/'history.json'), 'timing':read(run/'epoch_timing.json'),
            'checkpoint_receipt':read(run/'server_checkpoint_receipt.json'), 'terminal':read(folder/'terminal_manifest.json'),
            'resume_startup_gate':read(folder/'resume_startup_gate.json'), 'resume_copy_receipt':read(run/'resume_copy_receipt.json')}
        if row['terminal']:
            row['SHA_verified'] = all(sha(safe(folder,n)) == s for n,s in row['terminal']['files'].items())
            row['endpoint'] = read(run/'endpoint_replays.json'); row['manifest_sha256'] = sha(folder/'terminal_manifest.json')
            row['terminal_exposure'] = read(run/'exposure.json')
        receipt = row['checkpoint_receipt']; checkpoint = run/'last_epoch_state.pt'
        if receipt and checkpoint.exists():
            actual = sha(checkpoint)
            receipt_verified = all(sha(safe(run,n)) == v for n,v in receipt['files'].items())
            row['checkpoint_receipt_SHA_verified'] = receipt_verified
            bridge = (not receipt_verified and not processes and not gp.strip()
                      and (read(root/'supervisor_process_exit.json') or {}).get('returncode') == 0
                      and terminal_exposure_bridge(c, row))
            row['terminal_exposure_bridge_verified'] = bridge
            if actual == receipt['files']['last_epoch_state.pt'] and (receipt_verified or bridge):
                import torch
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if payload.get('evaluation_scope') != 'validation_only' or payload.get('held_out_test_evaluated') is not False:
                    raise ValueError('Checkpoint outside Validation scope')
                if payload.get('backbone') != job['arm'] or payload.get('seed') != job['seed'] or payload.get('checkpoint_monitor') != 'validation_raw_quantity_rmse':
                    raise ValueError('Checkpoint job/selector mismatch')
                if sha(checkpoint) == actual:
                    row['actual_saved_epoch'] = payload.get('epoch', payload.get('best_epoch'))
                    row['actual_best_epoch'] = payload.get('best_epoch'); row['last_checkpoint_file_sha256'] = actual
                else: row['checkpoint_race'] = True
                del payload
            else: row['checkpoint_race'] = True
        rows.append(row)
    return {'host':host, 'contract_sha256':canonical(c), 'actual_observed_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
        'observed_unix':time.time(), 'processes':processes, 'gpu':gpu, 'gpu_pids':gp,
        'gpu_uuid_verified':True, 'source_SHA_verified':True, 'operation_SHA_verified':True, 'parent_operation_SHA_verified':True,
        'server_status':read(root/'status.json'), 'failure':read(root/'failure.json'), 'server_lease':read(root/'server_lease.json'),
        'qualification':read(safe(root,binding['receipt_file'])), 'start_permit':read(root/'start_permit.json'), 'supervisor_exit':read(root/'supervisor_process_exit.json'), 'rows':rows}

def finite(value): return type(value) in (int,float) and math.isfinite(value)

def endpoint_ok(endpoint, history, data, epoch):
    if not epoch or epoch > len(history): return False
    if endpoint.get('evaluation_scope') != 'validation_only' or endpoint.get('held_out_test_evaluated') is not False: return False
    if endpoint.get('count', endpoint.get('target_count')) != data['inherited_data_identity']['populations']['validation']['target_count']: return False
    if endpoint.get('quantity_boundaries') != data['quantity_boundaries_all_train_rows'] or len(endpoint.get('quantity_cells',[])) != 5: return False
    if endpoint.get('tail',{}).get('count',0) <= 0: return False
    for metric, key in (('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll')):
        if not finite(endpoint.get(metric)) or not finite(endpoint.get('tail',{}).get(metric)) or not finite(history[epoch-1].get(key)): return False
        if not math.isclose(endpoint[metric], history[epoch-1][key], rel_tol=1e-10, abs_tol=1e-8): return False
    return True

def analyze_epoch_receipts(c, snapshot):
    host = snapshot['host']; expected = [j for j in c['jobs'] if j['host']==host]
    if snapshot['contract_sha256'] != canonical(c) or [r['job'] for r in snapshot['rows']] != expected: raise ValueError('Snapshot identity/coverage mismatch')
    if not all(snapshot.get(k) is True for k in ('gpu_uuid_verified','source_SHA_verified','operation_SHA_verified','parent_operation_SHA_verified')): raise ValueError('Unverified snapshot')
    actual = {}
    for line in snapshot['processes']:
        pid, ppid, args = line.split(None,2); actual[int(pid)] = {'ppid':int(ppid), 'argv':shlex.split(args)}
    gpu_pids = {int(s.split(',')[1]) for s in snapshot['gpu_pids'].splitlines() if s.split(',')[0].strip()==c['hosts'][host]['gpu_uuid']}
    workers = {pid for pid,p in actual.items() if '--mode' in p['argv'] and p['argv'][p['argv'].index('--mode')+1]=='worker' and p['argv'][0]!='timeout'}
    def flag(argv,key): return argv[argv.index(key)+1] if key in argv and argv.index(key)+1<len(argv) else None
    status = snapshot.get('server_status') or {}; active = status.get('active_job') or {}; root = c['hosts'][host]['root']; spec=c['hosts'][host]
    rows=[]
    for row in snapshot['rows']:
        job=row['job']; history=(row.get('history') or {}).get('history',[])
        best=min((v for v in history if finite(v.get('val_qty_rmse'))), key=lambda v:v['val_qty_rmse'], default=None)
        saved=row.get('actual_saved_epoch'); state='waiting';reason=None; terminal=row.get('terminal');endpoint=row.get('endpoint') or {};claim=row.get('claim') or {}
        if terminal:
            data=next(d for d in c['parent_contract']['datasets'] if d['dataset_id']==job['dataset'])
            ok=(terminal.get('status')=='complete' and terminal.get('scientific_success') is True and terminal.get('job')==job
                and terminal.get('contract_sha256')==canonical(c) and terminal.get('parent_contract_sha256')==PARENT_SHA
                and terminal.get('evaluation_scope')=='validation_only' and terminal.get('held_out_test_evaluated') is False and row.get('SHA_verified') is True
                and claim.get('job')==job and claim.get('contract_sha256')==canonical(c)
                and finite(claim.get('deadline_unix')) and claim['deadline_unix']<=(snapshot.get('start_permit') or {}).get('deadline_unix',-math.inf)
                and endpoint.get('job')==job and endpoint.get('evaluation_scope')=='validation_only' and endpoint.get('held_out_test_evaluated') is False
                and [v.get('epoch') for v in history]==list(range(1,len(history)+1)) and 40<=len(history)<=300
                and all(v.get('train_all_finite') is True and all(finite(v.get(k)) for k in ('val_qty_rmse','val_qty_mae','val_time_nll')) for v in history)
                and endpoint.get('completed_epochs')==len(history)==saved and best is not None and endpoint.get('best_epoch')==best['epoch']
                and endpoint_ok(endpoint.get('selected',{}),history,data,best['epoch'] if best else 0) and endpoint_ok(endpoint.get('last',{}),history,data,len(history)))
            state='complete' if ok else 'uncertain';reason=None if ok else 'Terminal/selector/full Validation/SHA verification failed'
        elif active.get('id')==job['id']:
            if snapshot.get('failure') or status.get('status')=='failed': state='failed';reason=(snapshot.get('failure') or status).get('message')
            else:
                sup=status.get('supervisor_pid');group=status.get('worker_group_pid');lease=snapshot.get('server_lease') or {}
                owner_ok=False
                if len(workers)==1 and workers<=gpu_pids and sup in actual and group in actual:
                    pid=next(iter(workers)); worker=actual[pid];argv=worker['argv'];timeout=actual[group]['argv'];supervisor=actual[sup]['argv']
                    entry=str(Path(root)/'operation/repair_runtime.py')
                    expected_worker=[spec['python'],entry,'--bundle',root,'--host',host,'--mode','worker','--job',job['id'],'--claim',str(Path(root)/'claims'/(job['id']+'.json'))]
                    owner_ok=(argv==expected_worker and supervisor==[spec['python'],entry,'--bundle',root,'--host',host,'--mode','dispatch']
                        and worker['ppid']==group and actual[group]['ppid']==sup and timeout[:3]==['timeout','--signal=TERM','--kill-after=15s'] and timeout[4:]==argv
                        and len(timeout)>4 and timeout[3].isdigit() and 0<int(timeout[3])<=c['limits']['per_condition_seconds']
                        and lease.get('job')==job['id'] and lease.get('contract_sha256')==canonical(c) and lease.get('worker_group_pid')==group and lease.get('supervisor_pid')==sup
                        and claim.get('job')==job and claim.get('contract_sha256')==canonical(c) and snapshot['observed_unix']<lease.get('expires_unix',0)<=claim.get('deadline_unix',0))
                state='running' if owner_ok else 'uncertain';reason=None if owner_ok else 'Exact supervisor/timeout/worker/GPU ownership incomplete'
        elif row.get('started'): state='uncertain';reason='Started run has no verified active owner'
        if state=='running' and (row.get('checkpoint_race') or (saved is not None and (saved!=len(history) or row.get('actual_best_epoch')!=(best['epoch'] if best else None)))):
            state='uncertain';reason='Checkpoint/history publication race; no extra query attempted'
        prefix=c.get('resume',{}).get(job['id'],{}).get('last_epoch',0)
        times=[t['elapsed_seconds'] for t in (row.get('timing') or {}).get('epochs',[]) if t.get('epoch',0)>prefix and finite(t.get('elapsed_seconds')) and t['elapsed_seconds']>0][-10:]
        eta=None
        if state=='running' and saved and best and times:
            median=statistics.median(times);stop=min(300,max(40,best['epoch']+40))
            eta={'median_completed_epoch_seconds':median,'completed_epochs_used':len(times),'conditional_stop_epoch':stop,
                'conditional_remaining_seconds':max(0,stop-saved)*median,'max300_remaining_seconds':max(0,300-saved)*median}
        rows.append({'job':job,'id':job['id'],'dataset':job['dataset'],'arm':job['arm'],'seed':job['seed'],'state':state,'reason':reason,
            'actual_saved_epoch':saved,'best_epoch':best['epoch'] if best else None,'resume_parent_epoch':prefix or None,'new_saved_epochs':max(0,saved-prefix) if saved else 0,'eta':eta,
            'best_so_far':None if best is None else {k:best.get(v) for k,v in (('RMSE','val_qty_rmse'),('MAE','val_qty_mae'),('TimeNLL','val_time_nll'))},
            'selected_endpoint':endpoint.get('selected'),'terminal_manifest_sha256':row.get('manifest_sha256')})
    counts={s:sum(r['state']==s for r in rows) for s in STATES}
    return {'host':host,'actual_observed_utc':snapshot['actual_observed_utc'],'counts':counts,'rows':rows,'query_failed':False,
        'terminal_verified':counts['complete']==len(expected) and not actual and not snapshot.get('failure') and status.get('status')=='complete' and (snapshot.get('supervisor_exit') or {}).get('returncode')==0,
        'automatic_retry':False,'held_out_test_evaluated':False}


def analyze(c, snapshot):
    checked = copy.deepcopy(snapshot)
    no_owned_process = not checked.get('processes') and not checked.get('gpu_pids', '').strip()
    for row in checked['rows']:
        if row.get('terminal'):
            bridge_required = row.get('checkpoint_receipt_SHA_verified') is not True
            bridge_ok = (no_owned_process and (checked.get('supervisor_exit') or {}).get('returncode') == 0
                         and row.get('terminal_exposure_bridge_verified') is True and terminal_exposure_bridge(c, row))
            if bridge_required and not bridge_ok:
                row['actual_saved_epoch'] = None
            receipt = row.get('checkpoint_receipt') or {}
            if (row.get('actual_best_epoch') != (row.get('endpoint') or {}).get('best_epoch')
                    or row.get('actual_saved_epoch') != receipt.get('epoch')
                    or row.get('last_checkpoint_file_sha256') != receipt.get('files', {}).get('last_epoch_state.pt')):
                row['actual_saved_epoch'] = None
    return analyze_epoch_receipts(c, checked)

def empty(c,host,state,reason,query_failed=False):
    jobs=[j for j in c['jobs'] if j['host']==host]
    return {'host':host,'actual_observed_utc':None,'counts':{s:len(jobs) if s==state else 0 for s in STATES},
        'rows':[dict(job=j,id=j['id'],dataset=j['dataset'],arm=j['arm'],seed=j['seed'],state=state,reason=reason,actual_saved_epoch=None,best_epoch=None,eta=None) for j in jobs],
        'terminal_verified':False,'query_failed':query_failed,'reason':reason,'automatic_retry':False}

def verify_cache(c,path):
    cache=read(path)
    if not cache:return None
    if sha(cache['snapshot'])!=cache['snapshot_sha256'] or sha(cache['analysis_path'])!=cache['analysis_sha256']:raise ValueError('Terminal cache bytes changed')
    snapshot=read(cache['snapshot']);result=analyze(c,snapshot)
    if not result['terminal_verified'] or result!=read(cache['analysis_path']):raise ValueError('Terminal cache revalidation failed')
    return result

def host_once(c,bundle,host,*,runner=subprocess.run):
    if not (bundle/'deployment'/('launch_receipt_'+host+'.json')).exists():return empty(c,host,'waiting','승인된 새 실행의 launch receipt 대기')
    launch=read(bundle/'deployment'/('launch_receipt_'+host+'.json'));spec=c['hosts'][host];root=spec['root']
    expected=[spec['python'],str(Path(root)/'operation/repair_runtime.py'),'--bundle',root,'--host',host,'--mode','dispatch']
    if launch.get('host')!=host or launch.get('contract_sha256')!=canonical(c) or launch.get('command')!=expected or launch.get('root')!=root:
        raise ValueError('Launch receipt identity/root/command mismatch')
    cache=verify_cache(c,bundle/'hourly_monitor'/('terminal_cache_'+host+'.json'))
    if cache is not None:
        if cache['host']!=host:raise ValueError('Foreign host terminal cache')
        cache['terminal_cache_reused']=True;return cache
    spec=c['hosts'][host];root=spec['root']
    environment=['env',*[str(k)+'='+str(v) for k,v in spec.get('environment',{}).items()]]
    command=shlex.join([*environment,'PYTHONDONTWRITEBYTECODE=1',spec['python'],'-','--native','--bundle',root,'--host',host])
    response=runner(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',host,command],input=Path(__file__).read_text(),capture_output=True,text=True,timeout=240)
    if response.returncode:raise RuntimeError(response.stderr[-3000:] or 'Readonly SSH failed')
    snapshot=json.loads(response.stdout);result=analyze(c,snapshot)
    if snapshot['host']!=host:raise ValueError('Wrong observed host')
    folder=bundle/'hourly_monitor'/str(time.time_ns())/host
    sp=folder/'snapshot.json';ap=folder/'analysis.json';write(sp,snapshot);write(ap,result)
    pointer={'snapshot':str(sp.resolve()),'snapshot_sha256':sha(sp),'analysis_path':str(ap.resolve()),'analysis':str(ap.resolve()),'analysis_sha256':sha(ap),'actual_observed_utc':snapshot['actual_observed_utc']}
    dest=bundle/'hourly_monitor'/('latest_'+host+'.json');tmp=dest.with_suffix('.'+str(time.time_ns())+'.tmp');write(tmp,pointer);tmp.replace(dest)
    if result['terminal_verified']:write(bundle/'hourly_monitor'/('terminal_cache_'+host+'.json'),pointer)
    return result

def run(bundle=BUNDLE,hosts=HOSTS,*,runner=subprocess.run):
    bundle=Path(bundle).resolve();validate_recovery_profile(bundle);c=preflight(bundle)
    now=dt.datetime.now(dt.timezone.utc);report={'schema':NAME+'_observation','contract_sha256':canonical(c),'reported_utc':now.isoformat(),'reported_kst':now.astimezone(KST).strftime('%Y-%m-%d %H:%M:%S KST'),'hosts':{},'reused_parent_execution_count':len(c['reused_jobs']),'new_execution_count':len(c['jobs']),'automatic_retry':False,'queue_eta':None}
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        pending={h:pool.submit(host_once,c,bundle,h,runner=runner) for h in hosts}
        for h,future in pending.items():
            try:report['hosts'][h]=future.result()
            except Exception as exc:report['hosts'][h]=empty(c,h,'uncertain',str(exc),query_failed=True)
    lines=['# 검색부 재개 관측 — '+report['reported_kst'],'','기존 5090 mean·weighted 2개는 보존 원본 재사용 대상이며 아래 신규 8개와 중복 집계하지 않습니다.','', '| GPU | 실제 관측(KST) | 완료/실패/진행/대기/미확정 | 방식 | 저장/best | 조건부 잔여 / 300epoch |','|---|---|---|---|---|---|']
    for host,result in report['hosts'].items():
        stamp=result.get('actual_observed_utc');kst=dt.datetime.fromisoformat(stamp).astimezone(KST).strftime('%m-%d %H:%M:%S') if stamp else '미관측'
        counts='/'.join(str(result['counts'][s]) for s in STATES)
        for row in result['rows']:
            eta=row.get('eta') or {};duration=lambda n: '미확정' if n is None else f'{n/3600:.2f}h'
            lines.append(f"| {host} | {kst} | {counts} | {row['arm'].removeprefix('titantpp_retrieval_')}·seed42 | {row.get('actual_saved_epoch')}/{row.get('best_epoch')} | {duration(eta.get('conditional_remaining_seconds'))} / {duration(eta.get('max300_remaining_seconds'))} |")
        if result.get('reason'):lines.append('\n'+result['reason'])
    lines+=['','ETA는 해당 GPU·실행의 새 완료 epoch 최대10개 중앙값이며 best 미갱신을 가정합니다. endpoint·회수 시간은 제외합니다. Test·독립 평가·CPU 재추론 감사는 별도 미완료입니다.']
    directory=bundle/'hourly_monitor';stamp=str(time.time_ns());jp=directory/(stamp+'.json');mp=directory/(stamp+'.md');write(jp,report);mp.write_text('\n'.join(lines)+'\n')
    pointer={'report':str(jp),'report_sha256':sha(jp),'markdown':str(mp),'markdown_sha256':sha(mp)}
    tmp=directory/('latest.'+stamp+'.tmp');write(tmp,pointer);tmp.replace(directory/'latest.json')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=BUNDLE);p.add_argument('--host',choices=HOSTS);p.add_argument('--native',action='store_true');a=p.parse_args()
    if a.native:
        if not a.host:raise ValueError('Native host required')
        c=preflight(a.bundle,native=True)
        if a.bundle.resolve()!=Path(c['hosts'][a.host]['root']).resolve():raise ValueError('Foreign native bundle')
        result=collect(c,a.host)
    else:result=run(a.bundle,(a.host,) if a.host else HOSTS)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))

if __name__=='__main__':main()
