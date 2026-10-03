"""Explicit remaining-arm recovery, one fresh process per fit and no automatic retry.

Training and replay arithmetic below are retained from the frozen benchmark.
The changes concern scope, worker lifetime, progress diagnostics and lineage.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import faulthandler
import gc
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from paper.scripts import run_local_detail_benchmark as original
from paper.scripts.run_local_detail_benchmark import (
    common, shared, base, require, read, ARMS, VARIANT, TIME_METRIC,
    verify_process_environment, scheduled_jobs, time_interface, initialization,
    validate_execution_contract, ArmBudget, training_args, audit_arm,
    replay_checkpoint, audit_replay_accounting, audit_batch_prefixes)
from paper.scripts.local_detail_instacart_remaining_contract import REMAINING


def run_arm(contract, permit, host, qualification, budget, requested_arm):
    import torch
    from paper.scripts.count_aware_tpp_backbone import training
    verify_process_environment(contract, host, active=True)
    runtime = common.runtime_check(contract, host)
    require(runtime == qualification["runtime"], "Qualified runtime changed")
    output = Path(contract["hosts"][host]["output_dir"])
    require(requested_arm in contract["execution_arms"], "Unapproved arm")
    for dataset_id, seed in scheduled_jobs(contract, host):
        data = next(d for d in contract["datasets"] if d["dataset_id"] == dataset_id)
        frame, metadata = base.prepare_admitted_data(data)
        interface, initial = time_interface(data, frame, contract), initialization(data, seed)
        if recovery := contract.get("recovery"):
            require(host == recovery["host"] and dataset_id == recovery["dataset"]
                    and initial == recovery["initial_state_sha256"], "Recovery initialization/scope changed")
        job_id = f"{dataset_id}/seed_{seed}" if "replication" in contract else dataset_id
        dest = output / job_id; dest.mkdir(parents=True, exist_ok=True)
        require(initial == contract['remaining_recovery']['initial_state_sha256'], 'Seed62 initialization changed')
        for name, value in (('input_receipt.json', metadata), ('initialization.json', initial)):
            path = dest/name
            if path.exists():
                require(read(path) == value, 'Shared input or initialization changed')
            else:
                common.write_json(path, value, exclusive=True)
        quantity = {"boundaries": data["quantity_boundaries_all_train_rows"],
                    "strata": [{"label": f"frozen_quantity_bin_{i}"} for i in range(5)]}
        arms, exposures = {}, {}
        for arm in (requested_arm,):
            budget(); validate_execution_contract(contract)
            require(common.runtime_check(contract, host) == runtime, "Runtime drift")
            arm_budget = ArmBudget(contract, host, dataset_id, arm, budget, seed=seed)
            args = training_args(contract, data, dest, arm)
            args.seeds = str(seed)
            run = dest / "runs" / arm / VARIANT / f"seed_{seed}"
            require(not run.exists(), "Fresh arm required; no implicit resume or retry")
            def status(epoch, records):
                common.write_json(output / "status.json", {"status":"training", "host":host,
                    "dataset":dataset_id, "backbone":arm, "seed":seed, "epoch":epoch,
                    "global_steps":sum(r["batches"] for r in records["train"]),
                    "deadline_unix":permit["deadline_unix"]})
                common.write_json(run / "exposure.json", records)
            with shared.audited_training(training, data, arm_budget, status) as exposure:
                summary, _, _ = training.train_one(args=args, frame=frame, quantity_contract=quantity,
                    interface_meta=interface, backbone=arm, quantity_variant=VARIANT, seed=seed)
            history = read(run / "history.json")["history"]
            steps = audit_arm(history, summary, exposure, data, contract["training"])
            require(summary["initial_state_sha256"] == initial[arm], "Initial model state mismatch")
            identity = training._resume_identity(args=args, backbone=arm, quantity_variant=VARIANT, seed=seed,
                monitor="validation_raw_quantity_rmse", interface_meta=interface, quantity_contract=quantity)
            endpoints = {}
            for label, path in (("selected",Path(summary["checkpoint_path"])), ("last",run/"last_epoch_state.pt")):
                epoch = summary["best_epoch"] if label == "selected" else len(history)
                endpoints[label] = replay_checkpoint(path, data, frame, arm_budget, expected_arm=arm,
                    expected_identity=identity, expected_initial=initial[arm], expected_epoch=epoch, expected_seed=seed)
                audit_replay_accounting(endpoints[label])
                require(all(math.isclose(endpoints[label][m], history[epoch-1][f],rel_tol=1e-10,abs_tol=1e-8)
                    for m,f in (("qty_rmse","val_qty_rmse"),("qty_mae","val_qty_mae"),("time_nll","val_time_nll"))),
                    "Checkpoint evaluation differs from recorded history")
            first40 = [r["val_qty_rmse"] for r in history[:40]]
            arms[arm] = {**endpoints, "last30":shared.last30_summary(history),
                "first40":{"mean":statistics.mean(first40), "sd":statistics.stdev(first40), "count":40},
                "best_epoch":summary["best_epoch"], "completed_epochs":len(history), "global_steps":steps,
                "stopped_early":summary["stopped_early"], "initial_state_sha256":initial[arm],
                "parameter_count":summary["parameter_count"], "elapsed_seconds":summary["elapsed_seconds"]}
            exposures[arm] = deepcopy(exposure)
            common.write_json(run/"exposure.json",exposure)
            common.write_json(run/"endpoint_replays.json",arms[arm],exclusive=True)
            arm_budget.finish(); gc.collect(); torch.cuda.empty_cache()
        return arms[requested_arm]


class ProgressBudget:
    """A pulse is emitted by genuine batch/phase progress, never by a timer thread."""
    def __init__(self, outer, path, arm, contract, *, clock=time.time, mono=time.monotonic):
        self.outer, self.path, self.arm, self.contract = outer, path, arm, contract
        self.clock, self.mono, self.last = clock, mono, -math.inf
        self.calls = 0

    def __call__(self):
        self.outer(); self.calls += 1
        if self.mono() - self.last >= self.contract['execution_isolation']['progress_write_interval_seconds']:
            common.write_json(self.path, {'at_unix':self.clock(), 'monotonic':self.mono(),
                'pid':os.getpid(), 'arm':self.arm, 'calls':self.calls,
                'contract_sha256':common.sha_json(self.contract)})
            self.last = self.mono()
            faulthandler.cancel_dump_traceback_later()
            faulthandler.dump_traceback_later(
                self.contract['execution_isolation']['stack_dump_after_stall_seconds'], repeat=True)


def check_progress(record, *, child_pid, arm, digest, started, now, timeout):
    require(record.get('pid') == child_pid and record.get('arm') == arm
            and record.get('contract_sha256') == digest, 'Worker progress identity changed')
    pulse = record.get('monotonic')
    require(type(pulse) in (int, float) and math.isfinite(pulse)
            and started <= pulse <= now, 'Invalid progress clock')
    require(now - pulse < timeout, 'Owned worker made no progress; no automatic retry')


def arm_worker(a):
    c, approval, permit = read(a.contract), read(a.approval), read(a.permit)
    spec = original.verify_authorization(c, approval, permit, '5090', training=True)
    require(a.arm in REMAINING and a.arm in c['execution_arms'], 'Unapproved fit')
    original.source_and_host(c, '5090'); original.verify_previous_run_complete(c, '5090')
    require(os.getppid() == a.owner_pid and os.getsid(0) == os.getpid(), 'Owned process group required')
    with os.fdopen(a.authority_fd, 'rb') as pipe:
        authority = json.loads(pipe.read(4096))
    expected = {'contract_sha256':common.sha_json(c), 'approval_sha256':common.sha_json(approval),
        'permit_sha256':common.sha_json(permit), 'owner_pid':a.owner_pid, 'arm':a.arm}
    require(authority == expected, 'Inherited arm authority mismatch')
    output = Path(spec['output_dir'])
    require(read(output/'workers'/f'{a.arm}.json')['authority'] == expected, 'Worker manifest mismatch')
    original.apply_process_environment(c, '5090'); original.verify_start_memory(c, '5090')
    import resource
    resource.setrlimit(resource.RLIMIT_FSIZE, (c['limits']['max_file_bytes'],)*2)
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    outer = original.Budget(c, permit, '5090', a.owner_pid, output)
    budget = ProgressBudget(outer, output/'progress'/f'{a.arm}.json', a.arm, c)
    budget()
    common.write_json(output/'status.json', {'status':'training', 'host':'5090', 'seed':62,
        'dataset':'insta_market_basket', 'backbone':a.arm, 'epoch':0, 'global_steps':0,
        'phase':'initializing', 'deadline_unix':permit['deadline_unix']})
    result = run_arm(c, permit, '5090', original.verify_local_qualification(c, permit, '5090'), budget, a.arm)
    budget()
    common.write_json(output/'arm_receipts'/f'{a.arm}.json', {
        'status':'complete', 'arm':a.arm, 'pid':os.getpid(), 'contract_sha256':common.sha_json(c),
        'result':result}, exclusive=True)
    faulthandler.cancel_dump_traceback_later()


def collect_final(c, permit):
    """Join JSON evidence only; no new fit or checkpoint replay for reused arms."""
    output = Path(c['hosts']['5090']['output_dir'])
    old = Path(c['remaining_recovery']['old_root'])
    data = next(d for d in c['datasets'] if d['dataset_id'] == 'insta_market_basket')
    arms, exposures, provenance = {}, {}, {}
    for arm in ARMS:
        reused = arm not in REMAINING
        root = old/'run' if reused else output
        run = root/'insta_market_basket/seed_62/runs'/arm/VARIANT/'seed_62'
        history = read(run/'history.json')['history']; summary = read(run/'summary.json')
        exposure = read(run/'exposure.json'); endpoints = read(run/'endpoint_replays.json')
        steps = audit_arm(history, summary, exposure, data, c['training'])
        require(endpoints['global_steps'] == steps and endpoints['best_epoch'] == summary['best_epoch'],
                'Reused/recovered endpoint metadata mismatch')
        require(summary['initial_state_sha256'] == c['remaining_recovery']['initial_state_sha256'][arm],
                'Initial state differs from predecessor seed62')
        for label in ('selected','last'):
            endpoint = endpoints[label]; audit_replay_accounting(endpoint)
            epoch = summary['best_epoch'] if label == 'selected' else len(history)
            require(endpoint['evaluation_scope'] == 'validation_only' and endpoint['held_out_test_evaluated'] is False,
                    'Endpoint split changed')
            require(all(math.isclose(endpoint[m], history[epoch-1][f], rel_tol=1e-10, abs_tol=1e-8)
                for m,f in (('qty_rmse','val_qty_rmse'),('qty_mae','val_qty_mae'),('time_nll','val_time_nll'))),
                'Endpoint checkpoint selection differs')
        arms[arm], exposures[arm] = endpoints, exposure
        provenance[arm] = {'reused':reused, 'run':str(run),
            'contract_sha256':c['parent_contract_sha256'] if reused else common.sha_json(c)}
    prefix = audit_batch_prefixes(exposures)
    comparisons = {reference:base.compare_arms({base.ARMS[0]:arms[reference],base.ARMS[1]:arms[reference],
        base.ARMS[2]:arms['titantpp_local_detail']},c['comparison']['acceptance'])['against'][base.ARMS[0]]
        for reference in ARMS if reference != 'titantpp_local_detail'}
    result = {'status':'complete', 'dataset':'insta_market_basket', 'host':'5090', 'seed':62,
        'arms':arms, 'provenance':provenance, 'batch_prefix_equal':prefix, 'comparisons':comparisons,
        'time_metric':TIME_METRIC, 'evaluation_scope':'validation_only','held_out_test_evaluated':False,
        'claim':'single_seed controlled encoder comparison; parameter capacity not isolated'}
    common.write_json(output/'insta_market_basket/seed_62/paired_comparison.json',result,exclusive=True)
    return {'status':'complete','host':'5090','contract_sha256':common.sha_json(c),
        'deadline_unix':permit['deadline_unix'],'datasets':{'insta_market_basket/seed_62':result},
        'new_training_arms':4,'new_endpoint_replays':8,'reused_seed62_arms':2,'reused_seed62_replays':4,
        'completed_predecessor_seed52_arms':6,'completed_predecessor_seed52_replays':12,
        'total_optimizer_steps':sum(arms[a]['global_steps'] for a in REMAINING),
        'combined_seed62_optimizer_steps':sum(a['global_steps'] for a in arms.values()),
        'endpoint_replays':8, 'combined_seed62_endpoint_replays':12, 'held_out_test_evaluated':False,
        'discarded_unsaved_attempt':c['remaining_recovery']['discarded_attempt']}


def supervise(cp, ap, tp):
    c, approval, permit = read(cp), read(ap), read(tp)
    spec = original.verify_authorization(c, approval, permit, '5090', training=True)
    original.source_and_host(c, '5090'); original.verify_previous_run_complete(c, '5090')
    original.verify_local_qualification(c, permit, '5090'); original.apply_process_environment(c, '5090')
    output = Path(spec['output_dir']); output.mkdir(parents=False, exist_ok=False)
    common.write_json(output/'wrapper_manifest.json', {'contract_sha256':common.sha_json(c),
        'approval_sha256':common.sha_json(approval),'permit_sha256':common.sha_json(permit),
        'started_at_unix':time.time(),'deadline_unix':permit['deadline_unix'],
        'isolation':'fresh_exec_process_per_arm','automatic_retry':False,'arms':list(REMAINING)},exclusive=True)
    deadline = shared.fixed_start(permit, total_seconds=c['limits']['total_wall_seconds'])
    child = None
    old_signal = signal.signal(signal.SIGTERM, shared._supervisor_terminated)
    try:
        for arm in REMAINING:
            original.verify_start_memory(c, '5090')
            authority = {'contract_sha256':common.sha_json(c),'approval_sha256':common.sha_json(approval),
                'permit_sha256':common.sha_json(permit),'owner_pid':os.getpid(),'arm':arm}
            common.write_json(output/'workers'/f'{arm}.json', {'authority':authority},exclusive=True)
            rd, wr = os.pipe()
            try:
                started = time.monotonic()
                log = output/'workers'/f'{arm}.log'
                with log.open('xb') as stream:
                    child = subprocess.Popen([spec['python'], str(Path(__file__).resolve()), '_arm',
                        '--contract',str(cp),'--approval',str(ap),'--permit',str(tp),'--arm',arm,
                        '--owner-pid',str(os.getpid()),'--authority-fd',str(rd)],cwd=spec['source_root'],
                        stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(rd,))
                    os.close(rd); rd = None
                    with os.fdopen(wr,'wb') as pipe:
                        wr = None; pipe.write(json.dumps(authority).encode())
                common.write_json(output/'workers'/f'{arm}.json', {'authority':authority,
                    'pid':child.pid,'started_at_unix':time.time(),'started_monotonic':started})
                progress = output/'progress'/f'{arm}.json'
                while child.poll() is None:
                    now = time.monotonic()
                    require(time.time() < permit['deadline_unix'] and now < deadline, 'Common deadline reached')
                    original.verify_active_arm_deadline(c, '5090')
                    if progress.exists():
                        check_progress(read(progress), child_pid=child.pid, arm=arm, digest=common.sha_json(c),
                            started=started, now=time.monotonic(), timeout=c['execution_isolation']['no_progress_timeout_seconds'])
                    else:
                        require(now-started < c['execution_isolation']['no_progress_timeout_seconds'], 'Worker initialization stalled')
                    shared.check_storage(output, original.storage_limits(c)); time.sleep(1)
                require(child.returncode == 0, f'Owned {arm} worker failed: {child.returncode}; no retry')
                receipt = read(output/'arm_receipts'/f'{arm}.json')
                require(receipt['status'] == 'complete' and receipt['pid'] == child.pid
                        and receipt['contract_sha256'] == common.sha_json(c), 'Invalid arm receipt')
            finally:
                for fd in (rd, wr):
                    if fd is not None: os.close(fd)
        original.verify_previous_run_complete(c, '5090')
        require(time.time() < permit['deadline_unix'] and time.monotonic() < deadline, 'Deadline before final audit')
        result = collect_final(c, permit)
        require(time.time() < permit['deadline_unix'] and time.monotonic() < deadline, 'Deadline during final audit')
        shared.check_storage(output, original.storage_limits(c))
        common.write_json(output/'partition_summary.json',result,exclusive=True)
        common.write_json(output/'status.json', {'status':'complete','host':'5090','deadline_unix':permit['deadline_unix']})
        return result
    except BaseException as error:
        if child is not None: shared.kill_owned_process_group(child)
        common.write_json(output/'status.json', {'status':'stopped','error':str(error),'automatic_retry':False})
        raise
    finally:
        signal.signal(signal.SIGTERM, old_signal)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('_arm',))
    for name in ('contract','approval','permit','arm'):p.add_argument('--'+name,required=True)
    p.add_argument('--owner-pid',type=int,required=True);p.add_argument('--authority-fd',type=int,required=True)
    arm_worker(p.parse_args())


if __name__ == '__main__':main()
