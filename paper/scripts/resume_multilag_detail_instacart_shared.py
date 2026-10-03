#!/usr/bin/env python3
"""Resume the stopped control and fresh candidate; GPU neighbours are telemetry, not a stop trigger."""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

SOURCE_ROOT = Path('/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/multilag_detail_observed_time_seed42_20260921_data_v3')
OLD_ROOT = SOURCE_ROOT.with_name('multilag_detail_instacart_resume_20260921_v1')
NEW_ROOT = SOURCE_ROOT.with_name('multilag_detail_instacart_shared_resume_20260921_v2')
PREVIOUS_SHA = '36da88553089c161e5d12850e824c9f8209c7abe8ad733871e278210121ab933'
PARENT_SHA = 'b69ddb960ccacb65d987428e1b40cdbfb8f3a7493dd18138f1da2836c0588800'
DATASET = 'insta_market_basket'
SAVED_EPOCH = 17
TOTAL_DEADLINE = 1790119857.99564
CONTROL_DEADLINE = 1790061159.7332053
ENTRYPOINT = 'paper/scripts/resume_multilag_detail_instacart_shared.py'
IMPORT_ROOT = SOURCE_ROOT/'source' if (SOURCE_ROOT/'source').is_dir() else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(IMPORT_ROOT))
from paper.scripts import run_multilag_detail_execution as r

common, shared, require = r.common, r.shared, r.require
SCOPE = {'host': '5090', 'dataset': DATASET, 'arms': list(r.ARMS), 'seed': 42,
         'epochs': 120, 'batch_size': 128, 'resume_arm': r.ARMS[1], 'reuse_arm': r.ARMS[0], 'train_arms': list(r.ARMS[1:]), 'saved_epoch': 17,
         'new_optimizer_steps': 3469211, 'checkpoint_replays': 4, 'reused_checkpoint_replays': 2,
         'maximum_partial_epoch_recomputation': 1, 'automatic_retry_resume': False,
         'maximum_explicit_dispatches': 1}
GPU_POLICY = {'other_processes': 'observe_without_stopping', 'startup_min_free_bytes': 2*1024**3,
              'rolling_free_memory_abort': False, 'terminate_other_processes': False,
              'automatic_oom_retry': False, 'max_own_training_workers': 1}


def read(path):
    return json.loads(Path(path).read_text())


def arm_dir(root, arm):
    return Path(root)/'run'/DATASET/'runs'/arm/r.VARIANT/'seed_42'


def validate(contract, approval, *, now=None, verify_source=False):
    parent = contract['parent_contract']
    require(contract.get('schema') == 'multilag_detail_instacart_shared_resume_v2'
            and common.sha_json(parent) == PARENT_SHA
            and contract.get('parent_contract_sha256') == PARENT_SHA, 'Wrong parent contract')
    require(contract.get('scope') == SCOPE and contract.get('old_root') == str(OLD_ROOT)
            and contract.get('new_root') == str(NEW_ROOT), 'Recovery scope or root changed')
    require(contract.get('deadline_unix') == TOTAL_DEADLINE == parent['launch']['fixed_deadline_unix']
            and contract.get('control_deadline_unix') == CONTROL_DEADLINE
            and contract.get('original_started_at_unix') == parent['launch']['fixed_started_at_unix'],
            'Original deadline changed')
    current = time.time() if now is None else now
    require(contract['original_started_at_unix'] <= current < min(TOTAL_DEADLINE, CONTROL_DEADLINE),
            'Recovery is outside the original control/common deadline')
    require(approval.get('approved') is True and approval.get('explicit_resume') is True
            and approval.get('hosts') == ['5090']
            and approval.get('contract_sha256') == common.sha_json(contract)
            and approval.get('user_instruction') == '5090 다시 진행하자', 'Explicit resume approval required')
    require(contract.get('previous_recovery_contract_sha256') == PREVIOUS_SHA
            and contract.get('gpu_policy') == GPU_POLICY, 'Recovery lineage or GPU policy changed')
    saved = contract['saved_files']
    required = {'recovery_contract.json','approval.json','run/status.json','run/active_arm.json','run/wrapper_manifest.json',
        f'run/{DATASET}/input_receipt.json',f'run/{DATASET}/initialization.json',
        'source/paper/scripts/resume_multilag_detail_instacart.py'}
    for arm in r.ARMS[:2]:
        prefix = f'run/{DATASET}/runs/{arm}/{r.VARIANT}/seed_42/'
        names = ['last_epoch_state.pt','history.json','exposure.json','epoch_timing.json','train.log']
        if arm == r.ARMS[0]: names += ['summary.json','best_val_qty_rmse_model.pt','endpoint_replays.json','combined_epoch_timing.json']
        required.update(prefix+n for n in names)
    require(required <= set(saved) and all(not Path(n).is_absolute() and '..' not in Path(n).parts for n in saved),
            'Preserved evidence is incomplete or escapes its root')
    if verify_source:
        verify_preserved(contract)
        require(read(OLD_ROOT/'run/status.json').get('status') == 'stopped', 'Parent is not stopped')
        require(not arm_dir(OLD_ROOT,r.ARMS[2]).exists(), 'Unstarted candidate already exists')
    return parent


def verify_preserved(contract):
    require(common.sha_file(__file__) == contract['wrapper_sha256'], 'Recovery wrapper changed')
    r.proposal.validate_contract(contract['parent_contract'])
    require(read(SOURCE_ROOT/'frozen_execution/execution_contract.json') == contract['parent_contract'], 'Parent identity changed')
    require(common.sha_json(read(OLD_ROOT/'recovery_contract.json')) == PREVIOUS_SHA, 'Previous recovery changed')
    previous_approval = read(OLD_ROOT/'approval.json')
    require(previous_approval['approved'] is True and previous_approval['contract_sha256'] == PREVIOUS_SHA, 'Previous authority changed')
    for name, digest in contract['saved_files'].items():
        path = OLD_ROOT/name
        require(path.is_file() and not path.is_symlink() and common.sha_file(path) == digest,
                'Preserved artifact changed: '+name)


def seed_exposure(records, saved, history, data):
    """Populate the real audit wrapper before the restored loader is iterated."""
    require(all(not records[k] for k in ('train','validation')), 'Exposure must start empty')
    epochs = len(history)
    require([x['epoch'] for x in history] == list(range(1, epochs+1)), 'Saved history is not contiguous')
    for split in ('train','validation'):
        expected = data['inherited_data_identity']['populations'][split]['target_count']
        rows = saved[split]
        require(len(rows) == epochs and all(x['epoch'] == i and x['split'] == split
                and x['count'] == expected and x['batches'] == math.ceil(expected/data['loader']['batch_size'])
                and len(x['batch_order_sha256']) == 64 for i,x in enumerate(rows,1)), 'Saved exposure mismatch')
    require(all(x['train_batch_count'] == y['batches'] and x['train_event_count'] == y['count']
                for x,y in zip(history,saved['train'])), 'Saved telemetry/exposure mismatch')
    for split in records: records[split].extend(deepcopy(saved[split]))


def verify_free_memory(rows, gpu_uuid):
    matches = [x.split(',') for x in rows.splitlines() if x.split(',')[0].strip() == gpu_uuid]
    require(len(matches) == 1 and len(matches[0]) == 2, 'Pinned GPU is unavailable')
    free = int(matches[0][1].strip())*1024**2
    require(free >= GPU_POLICY['startup_min_free_bytes'], 'Insufficient free memory to start this training')
    return free


def verify_start_resources(spec):
    rows = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'],text=True,timeout=15)
    return verify_free_memory(rows,spec['gpu_uuid'])


class SharedBudget:
    """Keep ownership/time/storage limits; other GPU jobs cannot trigger termination."""
    def __init__(self,parent,permit,owner_pid,output):
        self.parent,self.permit,self.owner_pid,self.output = parent,permit,owner_pid,Path(output)
        self.deadline = shared.fixed_start(permit,wall=time.time,monotonic=time.monotonic,total_seconds=172800)
        self.last_scan,self.last_observation = -math.inf,None

    def __call__(self):
        require(os.getppid() == self.owner_pid,'Owned supervisor exited')
        require(time.time() < self.permit['deadline_unix'] and time.monotonic() < self.deadline,
                'Original common deadline reached')
        if time.monotonic()-self.last_scan >= 10:
            check_storage(self.parent)
            try:
                observation={'other_gpu_pids':sorted(common.gpu_pids(self.parent['hosts']['5090'])-{os.getpid()}),
                             'telemetry_status':'available'}
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                observation={'telemetry_status':'unavailable','error':str(error)}
            if observation != self.last_observation:
                with (self.output/'gpu_sharing_events.jsonl').open('a') as f:
                    f.write(json.dumps({'observed_at_unix':time.time(),**observation,
                        'action':'continue_training','pure_throughput_comparison_valid':False})+'\n')
                self.last_observation=observation
            self.last_scan=time.monotonic()


def load_completed_baseline(contract,data,interface,initial):
    """Reuse completed validation evidence after checking identity, selection and accounting."""
    from paper.scripts.count_aware_tpp_backbone import training
    path=arm_dir(OLD_ROOT,r.ARMS[0]); summary=read(path/'summary.json')
    history=read(path/'history.json')['history']; exposure=read(path/'exposure.json')
    endpoints=read(path/'endpoint_replays.json')
    require(summary.get('status')=='success' and summary.get('completed_epochs')==120
            and summary.get('stopped_early') is False and summary.get('evaluation_scope')=='validation_only'
            and summary.get('held_out_test_evaluated') is False,'Baseline is not complete')
    args=r.training_args(contract['parent_contract'],data,OLD_ROOT/'run'/DATASET)
    q={'boundaries':data['quantity_boundaries_all_train_rows'],
       'strata':[{'label':f'frozen_quantity_bin_{i}'} for i in range(5)]}
    identity=training._resume_identity(args=args,backbone=r.ARMS[0],quantity_variant=r.VARIANT,seed=42,
        monitor='validation_raw_quantity_rmse',quantity_contract=q,interface_meta=interface)
    require(summary['resume_identity']==identity and summary['initial_state_sha256']==initial[r.ARMS[0]],
            'Baseline scientific identity differs')
    last=training.torch_load_checkpoint(path/'last_epoch_state.pt',map_location='cpu')
    saved,best=training._validate_resume_payload(last,expected_identity=identity,
        expected_initial_state_sha256=initial[r.ARMS[0]],monitor='validation_raw_quantity_rmse')
    require(saved==history and last['epoch']==120 and summary['best_epoch']==best['epoch'],'Baseline selector/history mismatch')
    selected=training.torch_load_checkpoint(path/'best_val_qty_rmse_model.pt',map_location='cpu')
    training._validate_cached_checkpoint_digest(summary,selected)
    training._validate_artifact_identity(selected,expected_identity=identity,allow_legacy_identity=False,artifact='Reused baseline')
    require(selected['best_epoch']==best['epoch'],'Baseline selected epoch differs')
    steps=shared.audit_exposure(exposure,data)
    require(steps==endpoints['global_steps'] and all(float(x['step'])==steps for x in last['optimizer_state_dict']['state'].values()),
            'Baseline optimizer/exposure mismatch')
    require(endpoints['best_epoch']==best['epoch'] and endpoints['initial_state_sha256']==initial[r.ARMS[0]]
            and endpoints['last30']==shared.last30_summary(history),'Baseline endpoint identity differs')
    for label,epoch,payload in (('selected',best['epoch'],selected),('last',120,last)):
        item=endpoints[label];r.audit_replay_accounting(item)
        require(item['state_sha256']==payload['model_state_sha256']
                and item['count']==data['inherited_data_identity']['populations']['validation']['target_count']
                and item['evaluation_scope']=='validation_only' and item['held_out_test_evaluated'] is False,
                'Baseline replay population/state differs')
        require(all(math.isclose(item[m],history[epoch-1][k],rel_tol=1e-10,abs_tol=1e-8)
            for m,k in (('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll'))),
            'Baseline replay metrics differ from history')
    return endpoints,exposure


class ArmClock:
    def __init__(self, contract, arm, outer):
        self.outer = outer
        started = contract['old_arm_clock']['started_at_unix'] if arm == r.ARMS[1] else time.time()
        self.deadline = min(started + 86400, TOTAL_DEADLINE)
        if arm == r.ARMS[1]: require(self.deadline == CONTROL_DEADLINE, 'Control clock reset')
        self.monotonic_deadline = time.monotonic() + self.deadline - time.time()
        self.path = NEW_ROOT/'run/active_arm.json'
        self.record = {'status':'active','host':'5090','dataset':DATASET,'backbone':arm,
                      'recovery_contract_sha256':common.sha_json(contract),'parent_contract_sha256':PARENT_SHA,
                      'started_at_unix':started,'deadline_unix':self.deadline,
                      'deadline_monotonic':self.monotonic_deadline}
        common.write_json(self.path,self.record)
        self()

    def __call__(self):
        self.outer()
        require(time.time() < self.deadline and time.monotonic() < self.monotonic_deadline,
                'Original per-arm deadline reached')

    def finish(self):
        self(); common.write_json(self.path,{**self.record,'status':'complete'})


def verify_parent_authority(parent):
    approval, permit = read(SOURCE_ROOT/'approval.json'), read(SOURCE_ROOT/'training_permit.json')
    r.verify_authorization(parent,approval,permit,'5090',training=True)
    qualification = r.verify_local_qualification(parent,permit,'5090')
    return permit, qualification


def production(contract, parent, permit, qualification, budget):
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    require(common.runtime_check(parent,'5090') == qualification['runtime'], 'Qualified runtime changed')
    data = next(d for d in parent['datasets'] if d['dataset_id'] == DATASET)
    frame, metadata = r.prepare_admitted_data(data)
    interface, initial = r.time_interface(data,frame,parent), r.initialization(data)
    q = {'boundaries':data['quantity_boundaries_all_train_rows'],
         'strata':[{'label':f'frozen_quantity_bin_{i}'} for i in range(5)]}
    dest = NEW_ROOT/'run'/DATASET; dest.mkdir(exist_ok=False)
    common.write_json(dest/'input_receipt.json',metadata,exclusive=True)
    common.write_json(dest/'initialization.json',initial,exclusive=True)
    baseline,baseline_exposure = load_completed_baseline(contract,data,interface,initial)
    old = arm_dir(OLD_ROOT,r.ARMS[1]); new = arm_dir(NEW_ROOT,r.ARMS[1])
    new.mkdir(parents=True,exist_ok=False)
    for name in ('last_epoch_state.pt','history.json','exposure.json','train.log'):
        shutil.copy2(old/name,new/name)
        require(common.sha_file(new/name) == contract['saved_files'][str((old/name).relative_to(OLD_ROOT))],
                'Checkpoint copy mismatch')
    payload = training.torch_load_checkpoint(new/'last_epoch_state.pt',map_location='cpu')
    args = r.training_args(parent,data,dest)
    identity = training._resume_identity(args=args,backbone=r.ARMS[1],quantity_variant=r.VARIANT,seed=42,
        monitor='validation_raw_quantity_rmse',quantity_contract=q,interface_meta=interface)
    history, _ = training._validate_resume_payload(payload,expected_identity=identity,
        expected_initial_state_sha256=initial[r.ARMS[1]],monitor='validation_raw_quantity_rmse')
    require(payload['epoch'] == SAVED_EPOCH and read(old/'history.json')['history'] == history, 'Saved epoch changed')
    saved_exposure = read(old/'exposure.json')
    prefix = {'train':[],'validation':[]}; seed_exposure(prefix,saved_exposure,history,data)
    old_steps = sum(x['batches'] for x in prefix['train'])
    require(old_steps == 264469 and all(float(x['step']) == old_steps
        for x in payload['optimizer_state_dict']['state'].values()), 'Saved optimizer step mismatch')
    common.write_json(NEW_ROOT/'run/resume_receipt.json',{'status':'verified_before_restore',
        'saved_epoch':SAVED_EPOCH,'next_epoch':SAVED_EPOCH+1,'saved_optimizer_steps':old_steps,
        'checkpoint_sha256':common.sha_file(new/'last_epoch_state.pt'),
        'exact_restore':'original trainer restores model, optimizer, RNG, loader generator and strict earliest selector',
        'parent_contract_sha256':PARENT_SHA,'recovery_contract_sha256':common.sha_json(contract)},exclusive=True)
    old_history = deepcopy(history); del payload; gc.collect()
    arms, exposures = {r.ARMS[0]:baseline}, {r.ARMS[0]:baseline_exposure}
    for arm in r.ARMS[1:]:
        budget(); verify_preserved(contract)
        require(common.runtime_check(parent,'5090') == qualification['runtime'], 'Runtime drift')
        clock = ArmClock(contract,arm,budget)
        args = r.training_args(parent,data,dest); run = arm_dir(NEW_ROOT,arm)
        if arm == r.ARMS[2]: require(not run.exists(), 'Fresh candidate required')
        def status(epoch, records):
            common.write_json(NEW_ROOT/'run/status.json',{'status':'training','host':'5090','dataset':DATASET,
                'backbone':arm,'epoch':epoch,'global_steps':sum(x['batches'] for x in records['train']),
                'deadline_unix':TOTAL_DEADLINE,'recovery_contract_sha256':common.sha_json(contract)})
            common.write_json(run/'exposure.json',records)
            if arm == r.ARMS[1]:
                current = read(run/'epoch_timing.json')
                previous = read(old/'epoch_timing.json')
                common.write_json(run/'combined_epoch_timing.json',{
                    'scope':current['scope'],'epochs':previous['epochs']+current['epochs'],
                    'interruption_wall_time_excluded':True})
        with shared.audited_training(training,data,clock,status) as exposure:
            if arm == r.ARMS[1]: seed_exposure(exposure,saved_exposure,old_history,data)
            summary,_,_ = training.train_one(args=args,frame=frame,quantity_contract=q,
                interface_meta=interface,backbone=arm,quantity_variant=r.VARIANT,seed=42)
        require(summary['status'] == 'success' and summary['completed_epochs'] == 120
                and not summary['stopped_early'] and summary['initial_state_sha256'] == initial[arm], 'Incomplete arm')
        steps = shared.audit_exposure(exposure,data); exposures[arm] = exposure
        common.write_json(run/'exposure.json',exposure)
        history = read(run/'history.json')['history']
        require([x['epoch'] for x in history] == list(range(1,121)), 'History is not continuous')
        if arm == r.ARMS[1]: require(history[:SAVED_EPOCH] == old_history, 'Saved epoch history changed')
        best = training.earliest_strict_minimum(history,metric_key='val_qty_rmse')
        require(summary['best_epoch'] == best['epoch'], 'Strict earliest selector changed')
        identity = training._resume_identity(args=args,backbone=arm,quantity_variant=r.VARIANT,seed=42,
            monitor='validation_raw_quantity_rmse',interface_meta=interface,quantity_contract=q)
        replay = {}
        for label,path in (('selected',Path(summary['checkpoint_path'])),('last',run/'last_epoch_state.pt')):
            epoch = summary['best_epoch'] if label == 'selected' else 120
            replay[label] = r.replay_checkpoint(path,data,frame,clock,expected_arm=arm,
                expected_identity=identity,expected_initial=initial[arm],expected_epoch=epoch)
            require(all(math.isclose(replay[label][metric],history[epoch-1][field],rel_tol=1e-10,abs_tol=1e-8)
                for metric,field in (('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll'))),
                'Checkpoint replay/history mismatch')
        arms[arm] = {**replay,'last30':shared.last30_summary(history),'global_steps':steps,
            'best_epoch':summary['best_epoch'],'initial_state_sha256':initial[arm]}
        common.write_json(run/'endpoint_replays.json',arms[arm],exclusive=True)
        gc.collect(); torch.cuda.empty_cache(); clock.finish()
    require(all(e == exposures[r.ARMS[0]] for e in exposures.values()), 'Actual batch exposure differs')
    require(sum(arms[a]['global_steps'] for a in r.ARMS[1:])-old_steps == SCOPE['new_optimizer_steps'], 'Remaining budget changed')
    result = {'status':'complete','dataset':DATASET,'host':'5090','arms':arms,'exposure_equal':True,
        'acceptance':r.compare_arms(arms,parent['acceptance']),'time_metric':r.TIME_METRIC,
        'evaluation_scope':'validation_only','held_out_test_evaluated':False,
        'parent_contract_sha256':PARENT_SHA,'recovery_contract_sha256':common.sha_json(contract),
        'resumed_arm':r.ARMS[1],'resumed_from_epoch':SAVED_EPOCH,'deadline_unix':TOTAL_DEADLINE,
        'baseline_reused_from':str(arm_dir(OLD_ROOT,r.ARMS[0])),
        'compute_cost_interpretation':'GPU sharing allowed; wall-time ratios are contextual, not controlled isolated throughput' }
    common.write_json(dest/'paired_comparison.json',result,exclusive=True)
    return result


def check_storage(parent):
    limits = r.storage_limits(parent)
    total = sum(shared.check_storage(root/'run',limits) for root in (SOURCE_ROOT,OLD_ROOT,NEW_ROOT))
    require(total <= limits['per_host_output_bytes'], 'Combined host output budget reached')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','_worker'))
    parser.add_argument('--contract',type=Path,required=True)
    parser.add_argument('--approval',type=Path,required=True)
    parser.add_argument('--owner-pid',type=int)
    parser.add_argument('--authority-fd',type=int)
    args = parser.parse_args()
    contract, approval = read(args.contract), read(args.approval)
    parent = validate(contract,approval,verify_source=True)
    require(Path(sys.executable).resolve() == Path(parent['hosts']['5090']['python']).resolve(), 'Wrong host Python')
    require(Path(__file__).resolve() == NEW_ROOT/'source'/ENTRYPOINT and Path.cwd() == SOURCE_ROOT/'source', 'Wrong isolated wrapper/cwd')
    preflight=read(NEW_ROOT/'preflight.json')
    require(preflight.get('status')=='passed' and preflight.get('contract_sha256')==common.sha_json(contract)
            and preflight.get('saved_control_epoch')==SAVED_EPOCH and preflight.get('baseline_complete') is True
            and preflight.get('optimizer_updates')==0 and preflight.get('cuda_initialized') is False,
            'Actual checkpoint preflight is missing')
    permit, qualification = verify_parent_authority(parent)
    verify_start_resources(parent['hosts']['5090'])
    old_tmux = subprocess.run([parent['hosts']['5090']['tmux_binary'],'has-session','-t',parent['hosts']['5090']['tmux']+'_train'],capture_output=True)
    require(old_tmux.returncode != 0, 'Original training session still exists')
    previous_tmux = subprocess.run([parent['hosts']['5090']['tmux_binary'],'has-session','-t',
        'multilag_detail_instacart_resume_20260921_v1_5090_train'],capture_output=True)
    require(previous_tmux.returncode != 0, 'Previous recovery is still running')
    r.apply_process_environment(parent,'5090')
    output = NEW_ROOT/'run'
    if args.command == '_worker':
        require(args.owner_pid == os.getppid() and os.getsid(0) == os.getpid(), 'Owned worker required')
        with os.fdopen(args.authority_fd,'rb') as f: authority=json.loads(f.read())
        require(authority == {'owner_pid':args.owner_pid,'contract_sha256':common.sha_json(contract)}, 'Worker authority differs')
        import resource
        resource.setrlimit(resource.RLIMIT_FSIZE,(64*1024**2,)*2)
        budget = SharedBudget(parent,permit,args.owner_pid,output)
        result = production(contract,parent,permit,qualification,budget)
        verify_preserved(contract)
        budget(); check_storage(parent)
        common.write_json(output/'partition_summary.json',result,exclusive=True)
        common.write_json(output/'status.json',{'status':'complete','host':'5090','deadline_unix':TOTAL_DEADLINE})
        return
    require(args.owner_pid is None and args.authority_fd is None, 'Unexpected worker authority')
    output.mkdir(parents=False,exist_ok=False)
    common.write_json(output/'wrapper_manifest.json',{'contract_sha256':common.sha_json(contract),
        'approval_sha256':common.sha_json(approval),'parent_contract_sha256':PARENT_SHA,
        'explicit_resume':True,'automatic_retry':False,'owner_pid':os.getpid(),
        'started_at_unix':time.time(),'deadline_unix':TOTAL_DEADLINE},exclusive=True)
    deadline = shared.fixed_start(permit,wall=time.time,monotonic=time.monotonic,total_seconds=172800)
    read_fd,write_fd = os.pipe(); child=None
    previous = signal.signal(signal.SIGTERM,shared._supervisor_terminated)
    try:
        with (output/'worker.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-u',__file__,'_worker','--contract',str(args.contract.resolve()),
                '--approval',str(args.approval.resolve()),'--owner-pid',str(os.getpid()),'--authority-fd',str(read_fd)],
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(read_fd,))
            os.close(read_fd);read_fd=None
            with os.fdopen(write_fd,'wb') as pipe:
                write_fd=None;pipe.write(json.dumps({'owner_pid':os.getpid(),'contract_sha256':common.sha_json(contract)}).encode())
            while child.poll() is None:
                require(time.time() < TOTAL_DEADLINE and time.monotonic() < deadline,'Original common deadline reached')
                path=output/'active_arm.json'
                if path.exists():
                    active=read(path)
                    require(active['recovery_contract_sha256'] == common.sha_json(contract),'Arm authority changed')
                    if active['status']=='active':
                        require(time.time() < active['deadline_unix'] and time.monotonic() < active['deadline_monotonic'],
                                'Original per-arm deadline reached')
                check_storage(parent);time.sleep(1)
            require(child.returncode == 0,'Owned recovery worker failed: '+str(child.returncode))
            require(read(output/'partition_summary.json')['status']=='complete','Missing completion audit')
    except BaseException as error:
        if child is not None:shared.kill_owned_process_group(child)
        common.write_json(output/'status.json',{'status':'stopped','error':str(error),'automatic_retry':False})
        raise
    finally:
        signal.signal(signal.SIGTERM,previous)
        for fd in (read_fd,write_fd):
            if fd is not None:os.close(fd)


if __name__ == '__main__':
    main()
