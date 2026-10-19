"""Isolated matched late-path queue after raw-time completion; no prior process is changed."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import traceback

_BASE_PATH = Path(__file__).with_name('titantpp_quantity_loss_mlp18_runtime.py')
_BASE_SHA = '03a4d8c07e7ee1af2164e2a7b1cfe088a86bc189dc1e35ba014a4e6b2f28e9ec'
from paper.scripts.titantpp_cross_fusion_precheck_runtime import require as _require, sha_file as _sha_file
_require(_sha_file(_BASE_PATH) == _BASE_SHA, 'Sealed predecessor runtime changed')
_spec = importlib.util.spec_from_file_location('paper.scripts._late_path18_runtime_private', _BASE_PATH)
_impl = importlib.util.module_from_spec(_spec)
# The private copy contains the existing bounded queue and exit/SHA collection API.
_source = _BASE_PATH.read_text().replace('titantpp_quantity_loss_mlp18', 'titantpp_late_path18')
_source = _source.replace('LOSS_MODES = ("raw_mse", "raw_mae")', 'LOSS_MODES = ("log_mse", "raw_mse", "raw_mae")')
_source = _source.replace('ARMS = ("titantpp_history_mlp_width16",)', 'ARMS = ("titantpp_mlp16_late_observation_v1", "titantpp_mlp16_late_hidden_v1")')
exec(compile(_source, str(_BASE_PATH), 'exec'), _impl.__dict__)
_original_three = None
_original_receipt = _impl.verify_receipt


def validate_contract(contract,host):
    r=_impl
    r.require(contract.get('schema')=='titantpp_late_path18_v1','Wrong late-path contract')
    r.require(set(contract['hosts'])=={'5080','5090'} and host in contract['hosts'],'Host mismatch')
    r.require(len(contract['datasets'])==3 and {d['dataset_id'] for d in contract['datasets']}==set(r.DATASETS),'Dataset matrix changed')
    common=contract['common_condition'];mode=common['loss_mode'];time_input=common['time_input']
    transforms={'log1p':'log1p_nonnegative_float32','raw':'raw_nonnegative_float32'}
    r.require(mode in r.LOSS_MODES and time_input in transforms,'Unknown common condition')
    r.alpha_label(common['time_alpha'])
    jobs=contract['jobs'];expected={(d,a,s,mode) for d in r.DATASETS for a in r.ARMS for s in (42,52,62)}
    r.require(len(jobs)==len({j['id'] for j in jobs})==18 and {(j['dataset'],j['arm'],j['seed'],j['loss_mode']) for j in jobs}==expected,'Expected18 unique common-condition candidate fits')
    for j in jobs:
        r.alpha_label(j['time_alpha'])
        assigned='5080' if j['dataset']=='yellow_trip_hourly' or (j['dataset']=='intermittent_frozen_5000' and j['seed']==42) else '5090'
        r.require(type(j['seed']) is int and j['host']==assigned and j.get('time_input')==time_input
            and j['id']==f"{j['dataset']}__{j['arm']}__seed{j['seed']}__loss{j['loss_mode']}",'Job identity/host/common-condition mismatch')
    r.require(all(contract['training'].get(k)==v for k,v in {'maximum_epochs':300,'minimum_epochs':40,'patience':40,'batch_size':128}.items()),'Training budget changed')
    r.require(contract['training'].get('monitor','validation_raw_quantity_rmse')=='validation_raw_quantity_rmse','Selector changed')
    r.require(contract['limits']=={'fit_wall_seconds':129600,'queue_wall_seconds':604800,'evaluation_wall_seconds':7200},'Relative limits changed')
    r.require(contract['evaluation_scope']=='validation_only' and contract['held_out_test_authorized'] is False,'Test forbidden')
    r.require(contract['source']['files'] and r.sha_json(contract['source']['files'])==contract['source']['files_sha256'],'Source inventory mismatch')
    r.require(set(contract['input_files'])=={f'inputs/data/{d}/{n}' for d in r.DATASETS for n in ('train_validation.parquet','split_manifest.json')},'Only TrainVal/manifest6 may be packaged')
    a=contract['architecture']
    r.require(a['arms']==list(r.ARMS) and a['time_alpha']==0. and a['quantity_loss_mode']==mode
        and a['quantity_loss_modes']==[mode] and a['quantity_loss_normalization']=='none'
        and a['checkpoint_warmstart'] is False and a['model_architecture_changed'] is True
        and a['late_path_parameter_count']==1024,'Late-path scientific scope changed')
    r.require(a['time_input_transform']==transforms[time_input] and a['quantity_input_transform']=='log1p_nonnegative_float32','Input transform mismatch')
    refs=contract['baseline_validation_references']
    r.require(len(refs)==9 and {(x['dataset'],x['seed'],x['loss_mode']) for x in refs}=={(d,s,mode) for d in r.DATASETS for s in (42,52,62)},'Stored common-condition MLP16 reference9 changed')
    r.require(all(x['time_input']==time_input and x['time_input_transform']==transforms[time_input]
        and x['time_alpha']==0. and x['time_head_trained'] is False and x['reevaluated'] is False
        and x['retrained'] is False and x['sha256'] for x in refs),'Stored references changed')
    h=contract['hosts'][host]
    r.require(all(Path(h[k]).is_absolute() for k in ('root','source_root','python')) and Path(h['source_root'])==Path(h['root'])/'source','Host path mismatch')
    r.require(type(h['absolute_host_deadline_unix']) in (int,float) and h['absolute_host_deadline_unix']>0,'Missing absolute deadline')
    paused=h['paused_processes']
    r.require(len(paused)==len({p['pid'] for p in paused})==4 and all(p['state'] in ('T','t') for p in paused),'Paused campaign identity missing')
    parent,binding=h['completed_predecessor'],contract['parent_raw_time27']
    r.require(Path(parent['root']).is_absolute() and parent['root']!=h['root']
        and all(parent[k]==binding[k] for k in ('contract_raw_sha256','contract_canonical_sha256','source_files_sha256'))
        and parent['dispatcher_identity']['pid']>0 and parent['final_receipt_sha256'] and parent['dispatcher_exit_sha256'],'Completed predecessor binding mismatch')
    return [j for j in jobs if j['host']==host]


def verify_receipt(contract,host,job,stage):
    _impl.require(stage in _impl.STAGES,'Test stage forbidden')
    result=_original_receipt(contract,host,job,stage)
    receipt=_impl.read(Path(contract['hosts'][host]['root'])/result['path'])
    transform={'log1p':'log1p_nonnegative_float32','raw':'raw_nonnegative_float32'}[contract['common_condition']['time_input']]
    mode={'titantpp_mlp16_late_observation_v1':'observation','titantpp_mlp16_late_hidden_v1':'hidden'}[job['arm']]
    _impl.require(receipt.get('time_input')==job['time_input'] and receipt.get('time_input_transform')==transform
        and receipt.get('quantity_input_transform')=='log1p_nonnegative_float32'
        and receipt.get('candidate_name')==job['arm'] and receipt.get('late_path_mode')==mode
        and receipt.get('late_path_parameter_count')==1024,'Scientific candidate/input proof missing')
    return result


def predecessor_gate(contract, host, *, require_complete=False):
    """Recheck the completed v2 predecessor without changing or signalling it."""
    from paper.scripts import titantpp_raw_time_mlp27_runtime as parent
    r = _impl
    h = contract['hosts'][host]['completed_predecessor']; root = Path(h['root'])
    cp = root / 'execution_contract.json'
    r.require(r.sha_file(cp) == h['contract_raw_sha256'], 'Predecessor contract bytes changed')
    c = r.read(cp)
    r.require(parent.sha_json(c) == h['contract_canonical_sha256'] and c['source']['files_sha256'] == h['source_files_sha256'], 'Predecessor contract/source binding changed')
    r.require(c['hosts'][host]['absolute_host_deadline_unix'] == contract['hosts'][host]['absolute_host_deadline_unix'], 'Predecessor deadline changed')
    jobs = parent.validate_contract(c, host); ids = {j['id'] for j in jobs}
    r.paused_processes(contract, host)
    r.require(not (root / 'dispatch_failure.json').exists(), 'Failed predecessor; no automatic continuation')
    fp, xp = root / 'final_receipt.json', root / 'dispatcher_exit.json'
    r.require(r.sha_file(fp) == h['final_receipt_sha256'] and r.sha_file(xp) == h['dispatcher_exit_sha256'], 'Predecessor terminal SHA changed')
    final, exit_record = r.read(fp), r.read(xp)
    r.require(final['status'] == 'complete' and final['host'] == host and final['contract_sha256'] == parent.sha_json(c) and final['completed_fit_count'] == final['validation_replay_count'] == len(jobs) and final['test_count'] == 0 and final['held_out_test_evaluated'] is False and final['owned_worker_pids_absent'] is True, 'Predecessor incomplete')
    r.require(exit_record['actual_os_returncode'] == 0 and exit_record['pid'] == h['dispatcher_identity']['pid'], 'Missing predecessor actual exit0')
    pids = [exit_record['pid']]
    r.require(set(final['stages']) == set(parent.STAGES), 'Predecessor stages changed')
    for stage in parent.STAGES:
        result = final['stages'][stage]
        r.require(result['status'] == 'complete' and set(result['exits']) == set(result['receipts']) == ids, 'Predecessor stage incomplete')
        for job in jobs:
            e, binding = result['exits'][job['id']], result['receipts'][job['id']]
            r.require(e['actual_os_returncode'] == 0 and e['timeout'] is False, 'Predecessor child failed')
            pids.append(e['pid']); receipt_path = r.local_path(root, binding['path'])
            r.require(r.sha_file(receipt_path) == binding['sha256'], 'Predecessor stage SHA changed')
            parent.verify_receipt(c, host, job, stage)
    m = final['artifact_manifest']; mp = r.local_path(root, m['path'])
    r.require(r.sha_file(mp) == m['sha256'], 'Predecessor manifest changed')
    inventory = r.read(mp)['files']
    r.require(inventory and len(inventory) == m['file_count'], 'Predecessor inventory incomplete')
    for name, digest in inventory.items(): r.require(r.sha_file(r.local_path(root, name)) == digest, 'Predecessor artifact changed')
    r.require(all(r.proc_identity(pid) is None for pid in pids), 'Predecessor process remains')
    gpu, paused = r.gpu_state(contract, host)
    r.require(not set(pids).intersection(gpu['compute_pids']), 'Predecessor GPU owner remains')
    return {'status': 'complete', 'parent_root': str(root), 'parent_final_receipt_sha256': r.sha_file(fp), 'parent_dispatcher_exit_sha256': r.sha_file(xp), 'parent_artifact_manifest_sha256': r.sha_file(mp), 'parent_owned_pids_absent': True, 'parent_gpu_pids_absent': True, 'paused_processes': paused, 'gpu': gpu}


def _candidate_three_concurrent(contract,host):
    """Six children: three simultaneous workers per arm, under the one common condition."""
    r=_impl;target=contract['hosts'][host];root=Path(target['root']);source=Path(target['source_root'])
    mode=contract['common_condition']['loss_mode'];results={};all_pids=[]
    for arm in r.ARMS:
        children=[]
        try:
            for index in range(3):
                prefix=root/f'admission_{arm}_{index}'
                command=[target['python'],'-u','-c',
                    'import json,sys; from paper.scripts.run_titantpp_late_path18 import synthetic_condition; c=json.load(open(sys.argv[1])); r=synthetic_condition(c,sys.argv[2],sys.argv[3],arm=sys.argv[4]); print(json.dumps(r,allow_nan=False))',
                    str(root/'execution_contract.json'),host,mode,arm]
                with prefix.with_suffix('.stdout').open('xb') as out,prefix.with_suffix('.stderr').open('xb') as err:
                    process=r.subprocess.Popen(command,cwd=source,env={**os.environ,**target['environment'],'PYTHONDONTWRITEBYTECODE':'1'},stdout=out,stderr=err,start_new_session=True)
                children.append((process,prefix,command));all_pids.append(process.pid)
            rows=[];started=r.time.monotonic()
            for process,prefix,command in children:
                limit=min(contract['limits']['evaluation_wall_seconds']-(r.time.monotonic()-started),target['absolute_host_deadline_unix']-r.time.time());timed_out=False
                try:code=process.wait(timeout=max(.001,limit))
                except r.subprocess.TimeoutExpired:timed_out=True;r.terminate_owned_child(process);code=process.returncode
                record={'pid':process.pid,'command':command,'actual_os_returncode':code,'timeout':timed_out,
                    'stdout_sha256':r.sha_file(prefix.with_suffix('.stdout')),'stderr_sha256':r.sha_file(prefix.with_suffix('.stderr')),
                    'stdout_bytes':prefix.with_suffix('.stdout').stat().st_size,'stderr_bytes':prefix.with_suffix('.stderr').stat().st_size,'observed_exit_at':r.now()}
                r.write(prefix.with_suffix('.json'),record,exclusive=True)
                r.require(code==0 and not timed_out,'Three-worker native synthetic failed; original output preserved')
                result=r.json.loads(prefix.with_suffix('.stdout').read_text())
                r.require(result.get('status')=='passed' and result.get('real_data_loaded') is False
                    and result.get('loss_mode')==mode and result.get('candidate_name')==arm
                    and result.get('time_input')==contract['common_condition']['time_input'],'Native synthetic candidate condition failed')
                rows.append({**record,'result':result})
            results[arm]=rows
        finally:
            for process,prefix,command in children:
                cleaned=process.poll() is None
                if cleaned:r.terminate_owned_child(process)
                if not prefix.with_suffix('.json').exists():
                    r.write(prefix.with_suffix('.json'),{'pid':process.pid,'command':command,'actual_os_returncode':process.returncode,
                        'timeout':False,'owned_synthetic_cleanup_after_failure':cleaned,
                        'stdout_bytes':prefix.with_suffix('.stdout').stat().st_size,'stderr_bytes':prefix.with_suffix('.stderr').stat().st_size,
                        'stdout_sha256':r.sha_file(prefix.with_suffix('.stdout')),'stderr_sha256':r.sha_file(prefix.with_suffix('.stderr')),'observed_exit_at':r.now()},exclusive=True)
    r.require(all(r.proc_identity(pid) is None for pid in all_pids),'Native synthetic process remains')
    gpu,paused=r.gpu_state(contract,host)
    r.require(not set(all_pids).intersection(gpu['compute_pids']),'Native synthetic GPU owner remains')
    return {'status':'passed','maximum_simultaneous_workers':3,'common_condition':contract['common_condition'],
        'per_arm':results,'native_worker_exit0_count':6,'owned_worker_pids_absent':True,'gpu_after':gpu,'paused_after':paused}


_original_three=_candidate_three_concurrent

def qualification_identity(contract, host):
    h = contract['hosts'][host]
    expected = [h['python'], '-u', '-m', _impl.MODULE, '--contract', str(Path(h['root']) / 'execution_contract.json'), '--host', host, '--mode', 'qualify']
    live = _impl.proc_identity(os.getpid())
    _impl.require(live is not None and live['pid'] == os.getpid() and live['pgid'] == os.getpid() and live['cwd'] == str(Path(h['source_root']).resolve()) and live['argv'] == expected, 'Qualification own process identity mismatch')
    attempt = _impl.read(Path(h['root']) / 'qualification_attempt.json')
    _impl.require(attempt['host'] == host and attempt['contract_sha256'] == _impl.sha_json(contract), 'Qualification attempt changed')
    return live


def qualify_three_concurrent(contract, host):
    """Keep the v2 self-context allowance scoped to the exact qualification child."""
    r = _impl; identity = qualification_identity(contract, host); root = Path(contract['hosts'][host]['root'])
    r.write(root / 'qualification_owner_identity.json', {'host': host, 'registered_at': r.now(), 'contract_sha256': r.sha_json(contract), 'qualification_owner_identity': identity}, exclusive=True)
    original = r.gpu_state
    def current_gpu(c, h, active_pids=()):
        r.require(c is contract and h == host, 'Qualification allowance escaped scope')
        current = qualification_identity(contract, host)
        r.require(all(current[k] == identity[k] for k in ('pid', 'start_ticks', 'pgid', 'argv', 'cwd')), 'Qualification identity changed')
        return original(c, h, (*active_pids, identity['pid']))
    r.gpu_state = current_gpu
    try:
        result = _original_three(contract, host)
        result.update(qualification_owner_identity=identity, qualification_self_context_allowed=True, qualification_self_context_scope='only current verified qualification child')
        return result
    finally:
        r.gpu_state = original


_impl.validate_contract = validate_contract
_impl.verify_receipt = verify_receipt
_impl.predecessor_gate = predecessor_gate
_impl.qualify_three_concurrent = qualify_three_concurrent
for _name, _value in tuple(_impl.__dict__.items()):
    if not _name.startswith('_'): globals()[_name] = _value

if __name__ == '__main__':
    try: raise SystemExit(_impl.main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
