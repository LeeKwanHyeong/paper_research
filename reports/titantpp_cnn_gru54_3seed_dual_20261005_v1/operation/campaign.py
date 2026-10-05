"""Owned six-fit operation adapter; frozen scientific source is never edited.

The timeout, lease, claims, audited train_one and endpoint architecture is
retained from the frozen dual-host campaign. Import starts no GPU/remote work.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

SCHEMA = 'titantpp_cnn_gru54_3seed_dual_20261005_v1'
TRAINING_PERMIT_SCHEMA = 'titantpp_cnn_gru54_3seed_training_permit_v1'
ARM = 'titantpp_cnn_gru54'
ARMS = (ARM,)
SEEDS = (52, 62)
ASSIGNMENTS = {'5080': ['yellow_trip_hourly', 'raf_spare_parts'],
               '5090': ['intermittent_frozen_5000']}
VARIANT = 'count_only_log_regression'
FROZEN_SOURCE_SHA = '4e94229fae002dc678025ecf056862f832c64459969f403b3cb5ac8cea8ba7b0'
FROZEN_DATA_SHA = {
    'intermittent_frozen_5000': '85a4a394dd04e64075f645fa1fb83f2b8bb069ca3c2c2fcf9a617475412d91dd',
    'yellow_trip_hourly': 'ab3b436e802e5c4d4667a4bff8e7d2cf56b0db77e8b3514da3936a0d28773639',
    'raf_spare_parts': 'b902581e51575508e874923774159dbd469826f29333147030acb0b353704f12'}
TRAINING = {'batch_size': 128, 'maximum_epochs': 300, 'minimum_epochs': 40,
            'patience': 40, 'monitor': 'validation_raw_quantity_rmse',
            'tie': 'strict_earliest_finite_minimum', 'warm_start': False}
ROOT = None
common = core = prior = base = shared = engine = width = checks = diagnostic = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def expected_jobs():
    return [{'id': f'{d}__{s}__{ARM}', 'host': h, 'dataset': d, 'seed': s, 'arm': ARM}
            for h, datasets in ASSIGNMENTS.items() for d in datasets for s in SEEDS]


def safe_relative(root, name):
    p = Path(name)
    require(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative path')
    target = (Path(root) / p).resolve()
    require(target.is_relative_to(Path(root).resolve()), 'Path escapes owned root')
    return target


def validate(c, *, source_root=None, operation_root=None, verify_files=True):
    require(c['schema'] == SCHEMA and c['arms'] == list(ARMS), 'Foreign comparison')
    require(c['training'] == TRAINING, 'Training drift')
    require(c['limits']['total_wall_seconds'] == 168*3600
            and c['limits']['per_condition_seconds'] == 36*3600, 'Budget drift')
    require(c['limits']['workers_per_host'] == 1 and c['limits']['automatic_retry'] is False,
            'Execution drift')
    require(c['jobs'] == expected_jobs(), 'Scope/order changed')
    require(c['canonical_conditions'] == 9 and c['new_fits'] == 6
            and c['already_terminal_reused'] == 3, 'Fit/reuse count changed')
    require(c['evaluation_scope'] == 'validation_only' and c['held_out_test_evaluated'] is False,
            'Split drift')
    require(set(c['hosts']) == set(ASSIGNMENTS), 'Foreign hosts')
    for h, spec in c['hosts'].items():
        require(spec['assigned_datasets'] == ASSIGNMENTS[h], 'Assignment changed')
        require(Path(spec['source_root']).resolve() == Path(spec['root']).resolve()/'source'
                and Path(spec['operation_root']).resolve() == Path(spec['root']).resolve()/'operation',
                'Source/operation roots changed')
    expected_reuse = {(j['host'], j['dataset'], j['seed']) for j in expected_jobs()}
    observed_reuse = [(r['host'], r['dataset'], r['seed']) for r in c['reuse']]
    require(len(observed_reuse) == len(expected_reuse) and set(observed_reuse) == expected_reuse
            and all(r['arm'] == 'titantpp_history_mlp' for r in c['reuse']),
            'Per-seed baseline exposure coverage changed')
    require(set(c['baseline_replays']) == set(FROZEN_DATA_SHA)
            and all(set(map(str, SEEDS)) <= set(refs) <= {'42', '52', '62'}
                    for refs in c['baseline_replays'].values()), 'Baseline coverage changed')
    require(c['reference_baseline'] == 'titantpp_history_mlp_width16', 'Primary baseline drift')
    require(set(c['anchors42']) == set(FROZEN_DATA_SHA), 'Three CNNGRU42 anchors required')
    for ref in c['anchors42'].values():
        require(1 <= ref['selected_epoch'] <= ref['last_epoch'] <= 300
                and ref['last_epoch'] >= 40, 'Anchor epoch changed')
        for k in ('checkpoint', 'checkpoint_sha256', 'last_checkpoint', 'last_checkpoint_sha256',
                  'initial_state_sha256', 'source_revision'):
            require(bool(ref[k]), 'Missing anchor '+k)
        for label in ('metrics', 'last_metrics', 'history_metrics', 'last_history_metrics'):
            require(all(type(ref[label][k]) in (int, float) and math.isfinite(ref[label][k])
                        for k in ('qty_rmse', 'qty_mae', 'time_nll')), 'Nonfinite anchor metrics')
    source = c['source']
    require(len(source['files']) == 123 and source['files_sha256'] == FROZEN_SOURCE_SHA
            and sha_json(source['files']) == FROZEN_SOURCE_SHA, 'Frozen source123 changed')
    require(c['dataset_sha256'] == FROZEN_DATA_SHA and len(c['datasets']) == 3
            and {d['dataset_id'] for d in c['datasets']} == set(FROZEN_DATA_SHA), 'Dataset scope drift')
    for data in c['datasets']:
        require(sha_json(data) == FROZEN_DATA_SHA[data['dataset_id']], 'Frozen dataset contract drift')
    if 'architecture' in c:
        require(c['architecture']['arms'] == list(ARMS)
                and c['design_sha256'] == sha_json(c['architecture']), 'Architecture drift')
    operations = c['operation']
    require(sha_json(operations['files']) == operations['files_sha256'], 'Operation closure invalid')
    require({'operation/campaign.py', 'operation/diagnostic_adapter.py', 'operation/monitor.py'}
            <= set(operations['files']), 'Required operation files missing')
    if verify_files:
        sr = Path(source_root or ROOT)
        op = Path(operation_root or Path(__file__).resolve().parent)
        require(os.environ.get('SOURCE_REVISION') == source['git_revision'], 'Source revision missing')
        for name, digest in source['files'].items():
            require(sha_file(safe_relative(sr, name)) == digest, 'Source drift '+name)
        for name, digest in c['input_files'].items():
            require(sha_file(safe_relative(sr, name)) == digest, 'Immutable input changed '+name)
        for name, digest in operations['files'].items():
            require(sha_file(safe_relative(op.parent, name)) == digest, 'Operation drift '+name)
    return c


def bootstrap(c, host):
    global ROOT, common, core, prior, base, shared, engine, width, checks, diagnostic
    spec = c['hosts'][host]
    ROOT = Path(spec['source_root']).resolve()
    validate(c, source_root=ROOT, operation_root=spec['operation_root'])
    require(Path(__file__).resolve().parent == Path(spec['operation_root']).resolve(),
            'Operation adapter outside pinned root')
    sys.path.insert(0, str(ROOT))
    common = importlib.import_module('paper.scripts.observed_slot_parallel_common')
    core = importlib.import_module('paper.scripts.run_titantpp_core_ablation')
    prior = importlib.import_module('paper.scripts.run_local_detail_benchmark')
    base = importlib.import_module('paper.scripts.run_multilag_detail_execution')
    shared = importlib.import_module('paper.scripts.run_observed_slot_partition')
    engine = importlib.import_module('paper.scripts.run_titantpp_cnn_gru')
    width = importlib.import_module('paper.scripts.run_titantpp_history_width')
    checks = importlib.import_module('paper.scripts.titantpp_gru_control_checks')
    diagnostic = importlib.import_module('diagnostic_adapter')
    for module in (common, core, prior, base, shared, engine, width, checks):
        require(Path(module.__file__).resolve().is_relative_to(ROOT), 'Import outside frozen source')
    require(Path(diagnostic.__file__).resolve().parent == Path(__file__).resolve().parent,
            'Foreign diagnostic adapter')
    engine.install_hooks()
    return engine


def authorization(c, root):
    validate(c)
    require(not (root/'failure.json').exists() and not (root/'qualification/failure.json').exists(),
            'Recorded failure; manual action required, no automatic retry')
    approval, start = read(root/'approval.json'), read(root/'start_permit.json')
    require(approval['approved'] is True and approval['contract_sha256'] == sha_json(c)
            and approval['hosts'] == ['5080', '5090'] and approval['user_instruction'], 'Missing approval')
    require(start['contract_sha256'] == sha_json(c)
            and start['approval_sha256'] == sha_json(approval), 'Foreign start')
    require(start['deadline_unix']-start['started_at_unix'] == c['limits']['total_wall_seconds']
            and start['started_at_unix'] <= time.time() < start['deadline_unix'], 'Expired/enlarged budget')
    return start


def variant_for(arm):
    require(arm == ARM, 'Foreign fit arm')
    return VARIANT


def initial_states(data, seed):
    return engine.initial_states(data, seed, arms=(engine.BASELINE, ARM))


def training_args(c, data, output, arm):
    require(arm == ARM, 'Foreign fit arm')
    return engine.training_args(c, data, output, arm)


def time_interface(data, frame, c, arm=ARM):
    return engine.time_interface(data, frame, c, arm=arm)


def replay_checkpoint(path, data, frame, budget, **kwargs):
    return engine.replay_checkpoint(path, data, frame, budget, **kwargs)


def verify_full_validation(endpoint, data):
    require(endpoint.get('evaluation_scope') == 'validation_only'
            and endpoint.get('held_out_test_evaluated') is False,
            'Endpoint is outside Validation scope')
    require(endpoint.get('count', endpoint.get('target_count'))
            == data['inherited_data_identity']['populations']['validation']['target_count'],
            'Endpoint is not the full original Validation population')
    return True


@contextmanager
def strict_training_seed(training):
    """Preserve frozen seed/RNG behavior and retain the qualified strict flag."""
    import torch
    original = training.set_seed
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    def strict_seed(*args, **kwargs):
        result = original(*args, **kwargs)
        torch.use_deterministic_algorithms(True, warn_only=False)
        return result
    training.set_seed = strict_seed
    try:
        yield
    finally:
        training.set_seed = original
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def reset_check(engine_module, budget, *, device='cuda:0'):
    import torch
    torch.manual_seed(42)
    model, _ = engine_module.build_model(checks.synthetic_data(256), ARM)
    correction = model.multilag_detail.to(device).eval()
    torch.nn.init.normal_(correction.output_projection.weight, std=.02)
    hidden = torch.randn(2, 8, 64, device=device)
    valid = torch.ones(2, 8, device=device, dtype=torch.bool)
    observed = valid.clone(); observed[:, 3] = False
    with torch.no_grad():
        budget(); original = correction(hidden, valid, memory_write_mask=observed)
        changed = hidden.clone(); changed[:, :3] += 1000
        after = correction(changed, valid, memory_write_mask=observed)
        torch.testing.assert_close(original[:, 4:], after[:, 4:], rtol=0, atol=0)
        require(bool((original[:, 3:5] == 0).all()) and bool(original[:, 5:].abs().sum() > 0),
                'Withheld reset/eligibility mismatch')
        padding = hidden.new_full((2, 2, 64), float('nan'))
        padded_valid = torch.cat((valid[:, :2], torch.zeros(2,2,device=device,dtype=torch.bool), valid[:, 2:]),1)
        padded_observed = torch.cat((observed[:, :2], torch.zeros(2,2,device=device,dtype=torch.bool), observed[:, 2:]),1)
        padded = correction(torch.cat((hidden[:, :2], padding, hidden[:, 2:]),1), padded_valid,
                            memory_write_mask=padded_observed)
        torch.testing.assert_close(original, torch.cat((padded[:, :2],padded[:, 4:]),1), rtol=1e-5, atol=1e-6)
        correction(changed, valid, memory_write_mask=observed)
        torch.testing.assert_close(original, correction(hidden, valid, memory_write_mask=observed), rtol=0, atol=0)
    del model, correction
    gc.collect()
    if str(device).startswith('cuda'): torch.cuda.empty_cache()
    return {'withheld_valid_resets': True, 'two_observed_eligibility': True,
            'padding_skips_state': True, 'no_state_carry_between_calls': True,
            'real_data_loaded': False, 'held_out_test_evaluated': False}


class PulseBudget:
    def __init__(self,root,digest,job,deadline,lease=False):
        self.root,self.digest,self.job,self.deadline,self.lease=root,digest,job,deadline,lease
        self.last=0.;self.last_storage_check=0.;self.monotonic_deadline=time.monotonic()+max(0,deadline-time.time())
    def __call__(self):
        now=time.time()
        require(now<self.deadline and time.monotonic()<self.monotonic_deadline,'Absolute time budget exhausted')
        if now-self.last<2:return
        if self.lease:
            r=read(self.root/'server_lease.json')
            require(r['contract_sha256']==self.digest and r['job']==self.job and now<r['expires_unix']<=self.deadline,
                'Server-owned supervisor lease expired')
        require(shutil.disk_usage(self.root).free>5*1024**3,'Server free disk below5GiB')
        common.write_json(self.root/'progress.json',{'job':self.job,'pid':os.getpid(),'updated_unix':now,
            'deadline_unix':self.deadline,'source':'genuine_phase_or_batch_progress','mac_required':False})
        rss=int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmRSS:')))*1024
        require(rss<16*1024**3,'Owned process CPU RSS exceeds16GiB')
        if now-self.last_storage_check>30:
            require(sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())<16*1024**3,'Owned-root bytes exceed16GiB')
            self.last_storage_check=now
        self.last=now


def verify_fit_runtime(c,host,q,job,actual):
    expected=deepcopy(q['runtime'])
    require(expected['environment']['PYTHONHASHSEED']==c['hosts'][host]['environment']['PYTHONHASHSEED']=='42',
        'Qualification hash seed changed')
    require(job in c['jobs'] and job['host']==host and job['seed'] in SEEDS,'Unapproved runtime seed')
    expected['environment']['PYTHONHASHSEED']=str(job['seed'])
    require(q['status']=='passed' and expected==actual,'Qualified runtime drift')
    return True


def run_fit(c,host,root,job,deadline):
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    start=authorization(c,root);q=read(root/'qualification/receipt.json');verify_training_permit(c,root)
    verify_fit_runtime(c,host,q,job,common.runtime_check(c,host))
    require(job in c['jobs'] and job['host']==host and job['arm']==ARM,'Unapproved fit')
    claim=read(root/'claims'/f"{job['id']}.json")
    require(claim['job']==job and claim['contract_sha256']==common.sha_json(c) and claim['deadline_unix']==deadline and deadline<=start['deadline_unix'],'Foreign fit claim/deadline')
    budget=PulseBudget(root,common.sha_json(c),job['id'],min(deadline,start['deadline_unix']),lease=True)
    lease=read(root/'server_lease.json')
    require(os.getpgrp()==lease['worker_group_pid'] and os.getppid()!=lease['supervisor_pid'],'Fit requires native timeout process group')
    parent=subprocess.run(['ps','-p',str(os.getppid()),'-o','args='],capture_output=True,text=True,check=True).stdout
    require(parent.strip().startswith('timeout --signal=TERM --kill-after=15s ') and str(deadline) in parent,'Native timeout parent required')
    budget();data=next(d for d in c['datasets'] if d['dataset_id']==job['dataset'])
    arm,seed=job['arm'],job['seed']
    frame,receipt=base.prepare_admitted_data(data,data_root=ROOT);interface=time_interface(data,frame,c,arm)
    require(common.sha_json(interface)==q['inputs'][data['dataset_id']]['interface_sha256'][arm],'Qualified interface changed')
    initial=initial_states(data,seed)
    require(initial==q['initialization'][data['dataset_id']][str(seed)],'Native initialization changed')
    folder=root/'run'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    args=training_args(c,data,folder,arm)
    args.epochs=300;args.min_epochs=40;args.early_stopping_patience=40;args.seeds=str(seed)
    args.execution_role='fresh_cnn_gru_validation'
    common.write_json(folder/'input_receipt.json',receipt,exclusive=True)
    common.write_json(folder/'initialization.json',initial,exclusive=True)
    run=folder/'runs'/arm/variant_for(arm)/f'seed_{seed}'
    quantity={'boundaries':data['quantity_boundaries_all_train_rows'],'strata':[{'label':f'frozen_quantity_bin_{i}'} for i in range(5)]}
    def status(epoch,records):
        budget();common.write_json(run/'exposure.json',records)
        names=('last_epoch_state.pt','history.json','exposure.json','epoch_timing.json','train.log')
        common.write_json(run/'server_checkpoint_receipt.json',{'epoch':epoch,'contract_sha256':common.sha_json(c),
            'files':{n:common.sha_file(run/n) for n in names},'mac_ack_required':False,'time':time.time()})
        common.write_json(folder/'status.json',{'status':'training','epoch':epoch,'global_steps':sum(x['batches'] for x in records['train']),
            'job':job,'updated_unix':time.time(),'deadline_unix':deadline})
        if epoch==2:
            times=[x['elapsed_seconds'] for x in read(run/'epoch_timing.json')['epochs']]
            remaining=max(times)*298*1.2+600
            common.write_json(folder/'two_epoch_cost_receipt.json',{'status':'warning_only','termination_gate':False,'same_scientific_fit':True,
                'measured_epoch_seconds':times,'projected_remaining_seconds':remaining,'not_a_completion_guarantee':True})
    with strict_training_seed(training), shared.audited_training(training,data,budget,status) as exposure:
        summary,_,_=training.train_one(args=args,frame=frame,quantity_contract=quantity,interface_meta=interface,
            backbone=arm,quantity_variant=variant_for(arm),seed=seed)
    history=read(run/'history.json')['history'];steps=prior.audit_arm(history,summary,exposure,data,c['training'])
    require(summary['initial_state_sha256']==initial[arm],'Initial tensor identity mismatch')
    identity=training._resume_identity(args=args,backbone=arm,quantity_variant=variant_for(arm),seed=seed,
        monitor='validation_raw_quantity_rmse',interface_meta=interface,quantity_contract=quantity)
    endpoints={}
    for label,path in (('selected',Path(summary['checkpoint_path'])),('last',run/'last_epoch_state.pt')):
        epoch=summary['best_epoch'] if label=='selected' else len(history)
        endpoints[label]=replay_checkpoint(path,data,frame,budget,expected_arm=arm,expected_identity=identity,
            expected_initial=initial[arm],expected_epoch=epoch,expected_seed=seed)
        verify_full_validation(endpoints[label],data)
        prior.audit_replay_accounting(endpoints[label])
        require(all(math.isclose(endpoints[label][k],history[epoch-1][v],rel_tol=1e-10,abs_tol=1e-8)
            for k,v in (('qty_mae','val_qty_mae'),('qty_rmse','val_qty_rmse'),('time_nll','val_time_nll'))),'Endpoint replay mismatch')
    # Compare exposure with completed reference B/Full, preserving the exact batch prefix.
    for ref in c['reuse']:
        if ref['host']==host and ref['dataset']==data['dataset_id'] and ref['seed']==seed:
            require(common.sha_file(Path(ref['remote_run'])/'exposure.json')==ref['file_sha256']['exposure.json'],
                'Reference exposure changed')
    core.check_baseline_prefix(c,host,data['dataset_id'],seed,exposure)
    selected_diag=diagnostic.evaluate_checkpoint(data,Path(summary['checkpoint_path']),frame,device='cuda:0',
        engine=engine,budget_check=budget,validation_replay=endpoints['selected'],train_sample_contract=c['diagnostic']['train_sampling'])
    baseline_path=root/'qualification'/f"{data['dataset_id']}__{seed}_baseline_diagnostic.json"
    require(common.sha_file(baseline_path)==q['baselines'][data['dataset_id']][str(seed)]['diagnostic_sha256'],'Baseline diagnostic changed')
    baseline_diag=read(baseline_path)
    comparison=diagnostic.compare_evaluations(baseline_diag,selected_diag)
    common.write_json(run/'selected_train_validation_diagnostic.json',selected_diag,exclusive=True)
    common.write_json(run/'width4_reference_comparison.json',comparison,exclusive=True)
    result={**endpoints,'status':'complete','job':job,'best_epoch':summary['best_epoch'],'completed_epochs':len(history),
        'global_steps':steps,'initial_state_sha256':initial[arm],'parameter_count':summary['parameter_count'],
        'elapsed_seconds':summary['elapsed_seconds'],'last30':shared.last30_summary(history),
        'first40':{'mean':statistics.mean(x['val_qty_rmse'] for x in history[:40]),
            'sd':statistics.stdev(x['val_qty_rmse'] for x in history[:40]),'count':40},
        'evaluation_scope':'validation_only','held_out_test_evaluated':False}
    common.write_json(run/'endpoint_replays.json',result,exclusive=True);common.write_json(run/'exposure.json',exposure)
    common.write_json(folder/'status.json',{'status':'complete','job':job,'updated_unix':time.time()})
    files={str(p.relative_to(folder)):common.sha_file(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    common.write_json(folder/'terminal_manifest.json',{'status':'complete','scientific_success':True,'files':files,
        'contract_sha256':common.sha_json(c),'job':job,'held_out_test_evaluated':False,'evaluation_scope':'validation_only','completed_unix':time.time()},exclusive=True)


def verify_training_permit(c,root):
    owners=[h for h,v in c['hosts'].items() if Path(v['root']).resolve()==Path(root).resolve()]
    require(len(owners)==1,'Unknown or ambiguous host root')
    host=owners[0]
    p=read(root/'training_permit.json');start=read(root/'start_permit.json')
    require(p['schema']==TRAINING_PERMIT_SCHEMA and p['host']==host,'Foreign host training permit')
    require(p['contract_sha256']==common.sha_json(c) and p['start_permit_sha256']==common.sha_json(start),'Foreign training permit')
    require(set(p['qualifications'])=={'5080','5090'},'Both native qualifications required')
    for h,qr in p['qualifications'].items():
        require(qr['status']=='passed' and qr['host']==h and qr['contract_sha256']==common.sha_json(c)
            and qr['source_files_sha256']==c['source']['files_sha256'] and qr['operation_files_sha256']==c['operation']['files_sha256'],'Foreign qualification')
    r=p['qualifications'][host]
    require(r['status']=='passed' and r['host']==host and r['contract_sha256']==common.sha_json(c)
        and r['source_files_sha256']==c['source']['files_sha256'],'Native qualification not passed')
    require(read(root/'qualification/receipt.json')==r,'Local qualification changed')
    require(r.get('held_out_test_evaluated') is False,'Held-out qualification rejected')
    return p


def stop_owned(child):
    if child.poll() is not None:return
    require(os.getpgid(child.pid)==child.pid,'Owned child process-group identity changed')
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)


def dispatch(c,host,root):
    start=authorization(c,root);permit=verify_training_permit(c,root)
    require(read(root/'qualification/receipt.json')==permit['qualifications'][host],'Local qualification changed')
    require(not common.gpu_pids(c['hosts'][host]),'GPU occupied before launch')
    require(shutil.which('timeout') is not None,'Independent native timeout guard unavailable')
    with (root/'supervisor.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        common.write_json(root/'supervisor.claim',{'pid':os.getpid(),'started_unix':time.time(),'contract_sha256':common.sha_json(c)},exclusive=True)
        results={};child=None;job=None
        try:
            # Historical nonlearned baselines are reused; no redundant evaluation.
            for job in (j for j in c['jobs'] if j['host']==host):
                validate(c);require(time.time()<start['deadline_unix'],'Campaign budget expired')
                require(not common.gpu_pids(c['hosts'][host]),'Other GPU owner detected')
                require(not (root/'run'/job['id']).exists(),'Run exists; no implicit restart')
                deadline=min(start['deadline_unix'],time.time()+36*3600,
                    c.get('retained_condition_deadlines',{}).get(job['id'],float('inf')))
                common.write_json(root/'claims'/f"{job['id']}.json",{'job':job,'contract_sha256':common.sha_json(c),'deadline_unix':deadline},exclusive=True)
                def pulse():
                    now=time.time();require(now<deadline,'Absolute deadline reached')
                    common.write_json(root/'server_lease.json',{'job':job['id'],'contract_sha256':common.sha_json(c),
                        'issued_unix':now,'expires_unix':min(now+90,deadline),'supervisor_pid':os.getpid(),'worker_group_pid':None if child is None else child.pid})
                    common.write_json(root/'status.json',{'status':'running','active_job':job,'supervisor_pid':os.getpid(),
                        'worker_group_pid':None if child is None else child.pid,'completed':results,'updated_unix':now,
                        'deadline_unix':start['deadline_unix'],'condition_deadline_unix':deadline,'mac_required':False})
                pulse();log=root/'logs'/f"{job['id']}.log";log.parent.mkdir(exist_ok=True)
                command=['timeout','--signal=TERM','--kill-after=15s',str(max(1,int(deadline-time.time()))),
                    sys.executable,str(Path(__file__).resolve()),'--contract',str(root/'execution_contract.json'),
                    '--host',host,'--mode','fit','--job',job['id'],'--deadline',str(deadline)]
                with log.open('xb') as stream:
                    child=subprocess.Popen(command,cwd=ROOT,env={**os.environ, 'PYTHONHASHSEED':str(job['seed'])},stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                    pulse()
                    while child.poll() is None:pulse();time.sleep(5)
                    require(child.returncode==0,'Worker exited '+str(child.returncode)+'; no automatic retry')
                manifest=read(root/'run'/job['id']/'terminal_manifest.json')
                require(manifest['scientific_success'] is True and manifest['status']=='complete' and manifest['job']==job and manifest['contract_sha256']==common.sha_json(c) and manifest['held_out_test_evaluated'] is False,'Missing/foreign successful terminal receipt')
                for p,s in manifest['files'].items():require(common.sha_file(safe_relative(root/'run'/job['id'],p))==s,'Terminal integrity mismatch')
                results[job['id']]={'terminal_manifest_sha256':common.sha_file(root/'run'/job['id']/'terminal_manifest.json')}
                child=None
            common.write_json(root/'status.json',{'status':'complete','completed':results,'completed_conditions':len(results),
                'endpoint_roles':len(results)*2,'updated_unix':time.time(),'mac_required':False})
        except BaseException as exc:
            if child is not None:stop_owned(child)
            failure={'status':'failed','type':type(exc).__name__,'message':str(exc),'active_job':job,
                'traceback':traceback.format_exc(),'completed':results,'updated_unix':time.time(),'automatic_retry':False}
            common.write_json(root/'failure.json',failure,exclusive=True);common.write_json(root/'status.json',failure)
            raise


def verify_anchor_payload(payload, ref, label):
    epoch = ref['selected_epoch'] if label == 'selected' else ref['last_epoch']
    require(payload.get('backbone') == ARM and payload.get('seed') == 42
            and payload.get('variant') == VARIANT, 'Foreign CNNGRU42 anchor')
    require(payload.get('epoch', payload.get('best_epoch')) == epoch
            and payload.get('initial_state_sha256') == ref['initial_state_sha256'], 'Anchor epoch/initialization changed')
    require(payload.get('evaluation_scope') == 'validation_only'
            and payload.get('held_out_test_evaluated') is False, 'Held-out anchor rejected')
    require(payload.get('source_revision') == ref['source_revision']
            and payload.get('source_revision_history') == [ref['source_revision']], 'Anchor source revision changed')
    require(payload.get('checkpoint_monitor') == 'validation_raw_quantity_rmse'
            and payload.get('resume_identity'), 'Anchor selector/identity missing')
    return True


def qualify(c, host, root):
    import torch
    from simple_lab_test.search.common.runner import torch_load_checkpoint
    start = authorization(c, root)
    parent = subprocess.run(['ps', '-p', str(os.getppid()), '-o', 'args='],
                            capture_output=True, text=True, check=True).stdout
    require(parent.strip().startswith('timeout --signal=TERM --kill-after=15s '),
            'Qualification requires native timeout parent')
    require(not common.gpu_pids(c['hosts'][host]), 'GPU is occupied')
    common.write_json(root/'qualification.claim', {'pid': os.getpid(), 'time': time.time(),
                      'contract_sha256': sha_json(c)}, exclusive=True)
    runtime = common.runtime_check(c, host)
    budget = PulseBudget(root, sha_json(c), 'qualification',
                         min(start['deadline_unix'], time.time()+c['limits']['qualification_seconds']))
    result = {'status': 'checking', 'host': host, 'runtime': runtime,
        'contract_sha256': sha_json(c), 'source_files_sha256': c['source']['files_sha256'],
        'operation_files_sha256': c['operation']['files_sha256'], 'initialization': {},
        'inputs': {}, 'baselines': {}, 'anchors42': {}, 'correctness': {ARM: {'lengths': {}}}}
    updates = 0
    for length in (84, 256):
        data = checks.synthetic_data(length)
        first = checks.synthetic_repeat(engine, data, ARM, budget)
        second = checks.synthetic_repeat(engine, data, ARM, budget)
        require(first['fingerprint'] == second['fingerprint'], 'Native determinism mismatch')
        for row in (first, second):
            require(row['parameter_count'] == (104159 if length == 84 else 115167)
                    and row['peak_allocated_bytes'] <= row['peak_reserved_bytes']
                    < c['limits']['peak_device_fraction_max']*row['total_memory_bytes'],
                    'Native parameter/memory gate failed')
        result['correctness'][ARM]['lengths'][str(length)] = {
            'repeat': first, 'identical_repeat': True}
        updates += 6
    result['correctness'][ARM]['causality'] = checks.causality_check(engine, ARM, budget)
    result['correctness'][ARM]['reset'] = reset_check(engine, budget)
    for data in c['datasets']:
        d = data['dataset_id']
        if d not in ASSIGNMENTS[host]: continue
        budget(); frame, receipt = base.prepare_admitted_data(data, data_root=ROOT)
        result['inputs'][d] = {'receipt': receipt, 'interface_sha256': {
            ARM: sha_json(time_interface(data, frame, c, ARM))}}
        result['initialization'][d], result['baselines'][d] = {}, {}
        for seed in SEEDS:
            initial = initial_states(data, seed)
            require(initial == initial_states(data, seed), 'Repeated native initialization changed')
            result['initialization'][d][str(seed)] = initial
            ref = c['baseline_replays'][d][str(seed)]
            path = safe_relative(ROOT, ref['checkpoint'])
            require(sha_file(path) == ref['checkpoint_sha256'], 'Width4 checkpoint changed')
            baseline_payload = torch_load_checkpoint(path, map_location='cpu')
            require(baseline_payload.get('backbone') == 'titantpp_history_mlp'
                    and baseline_payload.get('seed') == seed
                    and baseline_payload.get('evaluation_scope') == 'validation_only'
                    and baseline_payload.get('held_out_test_evaluated') is False,
                    'Foreign width4 reference seed/scope')
            del baseline_payload
            endpoint = core.replay_checkpoint(path, data, frame, budget)
            verify_full_validation(endpoint, data)
            require(all(math.isclose(endpoint[k], ref['metrics'][k], rel_tol=1e-5, abs_tol=1e-5)
                        for k in ('qty_rmse', 'qty_mae', 'time_nll')), 'Frozen width4 replay mismatch')
            diag = diagnostic.evaluate_checkpoint(data, path, frame, engine=engine, device='cuda:0',
                budget_check=budget, validation_replay=endpoint,
                train_sample_contract=c['diagnostic']['train_sampling'])
            dp = root/'qualification'/f'{d}__{seed}_baseline_diagnostic.json'
            common.write_json(dp, diag, exclusive=True)
            result['baselines'][d][str(seed)] = {'status': 'passed',
                'checkpoint_sha256': ref['checkpoint_sha256'], 'diagnostic_sha256': sha_file(dp),
                'metrics': {k: endpoint[k] for k in ('qty_rmse', 'qty_mae', 'time_nll')},
                'reference_arm': 'titantpp_history_mlp', 'purpose': 'exposure and auxiliary diagnostic'}
        ref = c['anchors42'][d]
        history_path = safe_relative(ROOT, ref['history'])
        require(sha_file(history_path) == ref['history_sha256'], 'Anchor history changed')
        historical = read(history_path)['history']
        require([row['epoch'] for row in historical] == list(range(1,len(historical)+1))
                and len(historical) == ref['last_epoch'], 'Anchor history is not sequential/full')
        best = min((row for row in historical if math.isfinite(row['val_qty_rmse'])),
                   key=lambda row: row['val_qty_rmse'])
        require(best['epoch'] == ref['selected_epoch'], 'Anchor strict first-minimum epoch changed')
        for label, epoch in (('history_metrics',ref['selected_epoch']),('last_history_metrics',ref['last_epoch'])):
            require(all(math.isclose(ref[label][m],historical[epoch-1][k],rel_tol=1e-10,abs_tol=1e-8)
                        for m,k in [('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll')]),
                    'Anchor metrics are not from the bound epoch')
        require(initial_states(data, 42)[ARM] == ref['initial_state_sha256'],
                'CNNGRU42 scientific initialization differs')
        result['anchors42'][d] = {}
        for label, path_key, sha_key, metrics_key in (
                ('selected', 'checkpoint', 'checkpoint_sha256', 'metrics'),
                ('last', 'last_checkpoint', 'last_checkpoint_sha256', 'last_metrics')):
            path = safe_relative(ROOT, ref[path_key])
            require(sha_file(path) == ref[sha_key], 'CNNGRU42 anchor binary changed')
            payload = torch_load_checkpoint(path, map_location='cpu')
            verify_anchor_payload(payload, ref, label)
            epoch = ref['selected_epoch'] if label == 'selected' else ref['last_epoch']
            endpoint = engine.replay_checkpoint(path, data, frame, budget, expected_arm=ARM,
                expected_identity=payload['resume_identity'], expected_initial=ref['initial_state_sha256'],
                expected_epoch=epoch, expected_seed=42)
            verify_full_validation(endpoint, data)
            prior.audit_replay_accounting(endpoint)
            require(all(math.isclose(endpoint[k], ref[metrics_key][k], rel_tol=1e-5, abs_tol=1e-5)
                        for k in ('qty_rmse', 'qty_mae', 'time_nll')), 'CNNGRU42 full-Val replay mismatch')
            result['anchors42'][d][label] = {'status': 'passed', 'metrics': endpoint,
                'checkpoint_sha256': ref[sha_key], 'epoch': epoch, 'full_validation_replay': True}
            common.write_json(root/'qualification'/f'{d}__42_{label}_validation_replay.json',
                              endpoint, exclusive=True)
            del payload
        del frame, diag
        gc.collect(); torch.cuda.empty_cache()
    result.update(status='passed', completed_unix=time.time(), synthetic_optimizer_updates=updates,
                  held_out_test_evaluated=False, scientific_training_launched=False)
    common.write_json(root/'qualification/receipt.json', result, exclusive=True)
    print(json.dumps({'status': 'passed', 'host': host,
                      'qualification': str(root/'qualification/receipt.json')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', required=True)
    p.add_argument('--host', choices=('5080', '5090'), required=True)
    p.add_argument('--mode', choices=('qualify', 'fit', 'dispatch', 'verify'), required=True)
    p.add_argument('--job'); p.add_argument('--deadline', type=float)
    a = p.parse_args(); c = read(a.contract); spec = c['hosts'][a.host]; root = Path(spec['root'])
    require(Path(a.contract).resolve() == root.resolve()/'execution_contract.json', 'Wrong contract root')
    require(Path.cwd().resolve() == Path(spec['source_root']).resolve(), 'Wrong pinned source cwd')
    expected_env = dict(spec['environment'])
    if a.mode == 'fit':
        job = next(j for j in c['jobs'] if j['id'] == a.job)
        require(job['host'] == a.host and a.deadline is not None, 'Wrong fit host/deadline')
        expected_env['PYTHONHASHSEED'] = str(job['seed'])
    for k, v in expected_env.items(): require(os.environ.get(k) == v, 'Process environment mismatch '+k)
    bootstrap(c, a.host)
    common.ENVIRONMENT = {**common.ENVIRONMENT, 'PYTHONHASHSEED': expected_env['PYTHONHASHSEED']}
    if a.mode == 'verify':
        print(json.dumps({'status': 'verified', 'contract_sha256': sha_json(c),
                          'scientific_source_unchanged': True})); return
    try:
        if a.mode == 'qualify': qualify(c, a.host, root)
        elif a.mode == 'dispatch': dispatch(c, a.host, root)
        else: run_fit(c, a.host, root, job, a.deadline)
    except BaseException as exc:
        if a.mode == 'qualify':
            common.write_json(root/'qualification/failure.json', {'status': 'failed',
                'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
                'automatic_retry': False, 'held_out_test_evaluated': False}, exclusive=True)
        raise


if __name__ == '__main__': main()
