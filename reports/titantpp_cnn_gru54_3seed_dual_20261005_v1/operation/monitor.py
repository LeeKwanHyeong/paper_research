"""Read-only native observation or offline snapshot validation; no network/retry."""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import shlex


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def safe(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts: raise ValueError('Unsafe terminal file')
    result = (root/p).resolve()
    if not result.is_relative_to(root.resolve()): raise ValueError('Terminal path escapes root')
    return result


def analyze(c, snapshot):
    if snapshot['contract_sha256'] != digest(c): raise ValueError('Foreign snapshot contract')
    host = snapshot['host']; expected = [j for j in c['jobs'] if j['host'] == host]
    if [r['job'] for r in snapshot['rows']] != expected: raise ValueError('Snapshot job coverage/order changed')
    processes = snapshot.get('processes', [])
    fit_pids = {int(x.split()[0]) for x in processes if '--mode fit' in x
                and not x.split(None, 2)[2].startswith('timeout ')}
    gpu_pids = {int(x.split(',')[1].strip()) for x in snapshot.get('gpu_pids', '').splitlines()
                if x.split(',')[0].strip() == c['hosts'][host]['gpu_uuid']}
    actual = {}
    for line in processes:
        pid,ppid,args=line.split(None,2)
        actual[int(pid)]={'ppid':int(ppid),'argv':shlex.split(args)}
    def flag(argv,name):
        return argv[argv.index(name)+1] if name in argv and argv.index(name)+1<len(argv) else None
    counts = dict(complete=0, failed=0, running=0, waiting=0, uncertain=0)
    rows = []; active = (snapshot.get('server_status') or {}).get('active_job')
    for row in snapshot['rows']:
        job = row['job']; h = (row.get('history') or {}).get('history', [])
        best = min((r for r in h if math.isfinite(r.get('val_qty_rmse', math.nan))),
                   key=lambda r: r['val_qty_rmse'], default=None)
        state, reason = 'waiting', None
        terminal = row.get('terminal'); endpoint = row.get('endpoint') or {}
        if terminal:
            data = next(d for d in c['datasets'] if d['dataset_id'] == job['dataset'])
            population = data['inherited_data_identity']['populations']['validation']['target_count']
            ok = terminal.get('status') == 'complete' and terminal.get('scientific_success') is True
            ok = ok and terminal.get('job') == job and terminal.get('contract_sha256') == digest(c)
            ok = ok and terminal.get('held_out_test_evaluated') is False and row.get('SHA_verified') is True
            ok = ok and endpoint.get('job') == job and endpoint.get('evaluation_scope') == 'validation_only'
            ok = ok and endpoint.get('held_out_test_evaluated') is False
            ok = ok and [r.get('epoch') for r in h] == list(range(1, len(h)+1)) and 40 <= len(h) <= 300
            ok = ok and endpoint.get('completed_epochs') == len(h) and row.get('actual_saved_epoch') == len(h)
            ok = ok and best is not None and endpoint.get('best_epoch') == best['epoch']
            for label, epoch in [('selected', best['epoch'] if best else 0), ('last', len(h))]:
                cell = endpoint.get(label, {})
                ok = ok and cell.get('evaluation_scope') == 'validation_only' and cell.get('held_out_test_evaluated') is False
                ok = ok and cell.get('count', cell.get('target_count')) == population
                if epoch:
                    for m, key in [('qty_rmse','val_qty_rmse'), ('qty_mae','val_qty_mae'), ('time_nll','val_time_nll')]:
                        a, b = cell.get(m, math.nan), h[epoch-1].get(key, math.nan)
                        ok = ok and math.isfinite(a) and math.isfinite(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-8)
            state = 'complete' if ok else 'uncertain'
            if not ok: reason = 'Terminal/full-Validation/history/epoch/SHA evidence rejected'
        elif active and job['id'] == active['id']:
            if snapshot.get('failure') or (snapshot.get('server_status') or {}).get('status') == 'failed':
                state, reason = 'failed', (snapshot.get('failure') or {}).get('message')
            elif len(fit_pids)==1 and fit_pids <= gpu_pids and snapshot.get('gpu_uuid_verified') is True:
                fit=actual[next(iter(fit_pids))];ss=snapshot['server_status']
                group=ss.get('worker_group_pid');supervisor=ss.get('supervisor_pid')
                owner_ok=(flag(fit['argv'],'--job')==job['id'] and flag(fit['argv'],'--host')==host
                    and fit['ppid']==group and group in actual and actual[group]['ppid']==supervisor
                    and supervisor in actual and flag(actual[supervisor]['argv'],'--mode')=='dispatch'
                    and flag(actual[supervisor]['argv'],'--host')==host)
                state='running' if owner_ok else 'uncertain'
                if not owner_ok:reason='Active job/native timeout/supervisor PID linkage incomplete'
            else: state, reason = 'uncertain', 'Owned fit/supervisor/GPU PID evidence incomplete'
        elif row.get('started'):
            state, reason = 'uncertain', 'Nonterminal run lacks active ownership'
        counts[state] += 1
        times = [r['elapsed_seconds'] for r in (row.get('timing') or {}).get('epochs', [])][-10:]
        eta = None; saved = row.get('actual_saved_epoch')
        if state == 'running' and saved is not None and best and times and all(math.isfinite(t) and t > 0 for t in times):
            med = statistics.median(times); stop = min(300, max(40, best['epoch']+40))
            eta = {'median_completed_epoch_seconds': med, 'completed_epochs_used': len(times),
                   'conditional_stop_epoch': stop, 'conditional_remaining_seconds': max(0,stop-saved)*med,
                   'max300_remaining_seconds': max(0,300-saved)*med,
                   'assumption': 'No later improvement; measured GPU speed unchanged; replay/retrieval excluded'}
        rows.append({'job': job, 'id':job['id'], 'dataset':job['dataset'], 'arm':job['arm'], 'seed':job['seed'],
                     'state': state, 'reason': reason, 'actual_saved_epoch': saved,
                     'history_epochs': len(h), 'best_epoch': best['epoch'] if best else None, 'eta': eta,
                     'best_so_far': None if not best else {'RMSE':best['val_qty_rmse'],
                         'MAE':best.get('val_qty_mae'),'TimeNLL':best.get('val_time_nll')},
                     'terminal_manifest_sha256': row.get('manifest_sha256')})
    return {'host': host, 'actual_observed_utc': snapshot['actual_observed_utc'], 'counts': counts,
            'rows': rows, 'owned_fit_pids': sorted(fit_pids), 'gpu_pids': sorted(gpu_pids),
            'anchors42_reused': len(c['hosts'][host]['assigned_datasets']),
            'held_out_test_evaluated': False, 'automatic_retry': False}


def verify_native_files(c,host):
    spec=c['hosts'][host];root=Path(spec['root']).resolve();source=Path(spec['source_root']).resolve()
    operation=Path(spec['operation_root']).resolve()
    if source!=root/'source' or operation!=root/'operation':raise ValueError('Foreign source/operation root')
    if Path(__file__).resolve().parent!=operation:raise ValueError('Monitor outside owned operation root')
    native=read(root/'execution_contract.json')
    if native is None or digest(native)!=digest(c):raise ValueError('Native contract changed')
    if digest(c['source']['files'])!=c['source']['files_sha256'] or len(c['source']['files'])!=123:
        raise ValueError('Invalid source closure')
    if c['source']['files_sha256']!='4e94229fae002dc678025ecf056862f832c64459969f403b3cb5ac8cea8ba7b0':
        raise ValueError('Frozen source123 changed')
    if digest(c['operation']['files'])!=c['operation']['files_sha256']:raise ValueError('Invalid operation closure')
    for name,value in c['source']['files'].items():
        if sha(safe(source,name))!=value:raise ValueError('Frozen scientific source changed '+name)
    for name,value in c['operation']['files'].items():
        if sha(safe(root,name))!=value:raise ValueError('Frozen operation changed '+name)


def collect(c, host):
    verify_native_files(c,host)
    root = Path(c['hosts'][host]['root']); entry = str(Path(c['hosts'][host]['operation_root'])/'campaign.py')
    ps = subprocess.run(['ps','-eo','pid,ppid,args'],capture_output=True,text=True,check=True).stdout
    processes = [s.strip() for s in ps.splitlines() if entry in s and ('--mode fit' in s or '--mode dispatch' in s)]
    gpu = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    gpu_pids = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout
    if c['hosts'][host]['gpu_uuid'] not in gpu: raise ValueError('Wrong GPU inventory')
    rows = []
    for job in (j for j in c['jobs'] if j['host'] == host):
        folder = root/'run'/job['id']; run = folder/'runs'/job['arm']/'count_only_log_regression'/f"seed_{job['seed']}"
        row = {'job': job, 'started': folder.exists(), 'status': read(folder/'status.json'),
               'history': read(run/'history.json'), 'timing': read(run/'epoch_timing.json'),
               'checkpoint_receipt': read(run/'server_checkpoint_receipt.json'),
               'terminal': read(folder/'terminal_manifest.json')}
        if row['terminal']:
            row['SHA_verified'] = all(sha(safe(folder,p)) == digest for p,digest in row['terminal']['files'].items())
            row['endpoint'] = read(run/'endpoint_replays.json')
            row['manifest_sha256'] = sha(folder/'terminal_manifest.json')
        receipt = row['checkpoint_receipt']; checkpoint = run/'last_epoch_state.pt'
        if receipt and checkpoint.exists():
            actual = sha(checkpoint); row['last_checkpoint_file_sha256'] = actual
            if actual == receipt['files']['last_epoch_state.pt']:
                import torch
                payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if payload.get('evaluation_scope') != 'validation_only' or payload.get('held_out_test_evaluated') is not False:
                    raise ValueError('Saved checkpoint is outside Validation scope')
                if sha(checkpoint)!=actual:
                    row['checkpoint_race']='Checkpoint changed during CPU metadata read'
                elif payload.get('backbone')!=job['arm'] or payload.get('seed')!=job['seed'] or payload.get('checkpoint_monitor')!='validation_raw_quantity_rmse':
                    raise ValueError('Saved checkpoint job/selector mismatch')
                else:
                    row['actual_saved_epoch'] = payload.get('epoch',payload.get('best_epoch'))
                    row['actual_best_epoch'] = payload.get('best_epoch')
                del payload
            else: row['checkpoint_race'] = 'New checkpoint differs from previous receipt'
        rows.append(row)
    return {'host': host, 'contract_sha256': digest(c),
            'actual_observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'processes': processes, 'gpu': gpu, 'gpu_pids': gpu_pids,
            'gpu_uuid_verified':True,'source_SHA_verified':True,'operation_SHA_verified':True,
            'server_status': read(root/'status.json'), 'failure': read(root/'failure.json'),
            'supervisor_exit': read(root/'supervisor_process_exit.json'), 'rows': rows,
            'scope': 'Owned Train/Validation files and CPU checkpoint metadata only'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    paths=p.add_mutually_exclusive_group(required=True)
    paths.add_argument('--contract'); paths.add_argument('--bundle')
    p.add_argument('--host',choices=('5080','5090'),required=True)
    p.add_argument('--snapshot'); p.add_argument('--raw',action='store_true')
    a = p.parse_args(); bundle=Path(a.bundle).resolve() if a.bundle else None
    c = read(bundle/'execution_contract.json' if bundle else Path(a.contract))
    if c['schema'] != 'titantpp_cnn_gru54_3seed_dual_20261005_v1': raise ValueError('Foreign campaign')
    if bundle and not a.snapshot:
        current=read(bundle/'current.json')
        if current is None or current['canonical_sha256']!=digest(c) or current['source_closure_sha256']!=c['source']['files_sha256'] or current['operation_closure_sha256']!=c['operation']['files_sha256'] or current['hosts']!={h:v['root'] for h,v in c['hosts'].items()}:
            raise ValueError('Current contract/source/operation/hosts changed; no old-root query')
        cache=bundle/'hourly_monitor'/('terminal_cache_'+a.host+'.json')
        if cache.exists():
            saved=read(cache)
            if saved['contract_sha256']!=digest(c): raise ValueError('Foreign terminal cache')
            print(json.dumps({**saved['analysis'],'terminal_cache_reused':True,
                              'observation_time_preserved':True},ensure_ascii=False)); return
        spec=c['hosts'][a.host]
        command=shlex.join([spec['python'],str(Path(spec['operation_root'])/'monitor.py'),
                           '--contract',str(Path(spec['root'])/'execution_contract.json'),
                           '--host',a.host,'--raw'])
        result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',a.host,command],
                              capture_output=True,text=True,timeout=180)
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        folder=bundle/'hourly_monitor'/stamp/a.host;folder.mkdir(parents=True,exist_ok=False)
        if result.returncode:
            failure={'host':a.host,'query_failed':True,'queried_utc':stamp,'error':result.stderr[-4000:],
                     'training_failure':False,'automatic_retry':False}
            (folder/'failure.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2)+'\n')
            print(json.dumps(failure,ensure_ascii=False));return
        remote=json.loads(result.stdout);snapshot=remote['snapshot']
        analysis=analyze(c,snapshot)
        (folder/'snapshot.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        (folder/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        latest={'analysis':str(folder/'analysis.json'),'snapshot':str(folder/'snapshot.json'),
                'actual_observed_utc':snapshot['actual_observed_utc']}
        (bundle/'hourly_monitor'/('latest_'+a.host+'.json')).write_text(json.dumps(latest,indent=2)+'\n')
        if analysis['counts']['complete']==len(analysis['rows']) and not snapshot['processes'] and (snapshot.get('supervisor_exit') or {}).get('returncode')==0:
            cache.write_text(json.dumps({'contract_sha256':digest(c),'analysis':analysis,
                                        'snapshot':str(folder/'snapshot.json')},ensure_ascii=False,indent=2)+'\n')
    else:
        snapshot = read(Path(a.snapshot)) if a.snapshot else collect(c,a.host)
    if snapshot['host'] != a.host: raise ValueError('Foreign host snapshot')
    print(json.dumps({'snapshot': snapshot, 'analysis': analyze(c,snapshot)} if a.raw else analyze(c,snapshot),
                     ensure_ascii=False,allow_nan=False))


if __name__ == '__main__': main()
