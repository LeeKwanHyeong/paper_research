#!/usr/bin/env python3
"""Supervise six serial joint workers for 18 frozen Intermittent model-split passes.

Only this isolated bundle is writable. Child exits and unsuccessful claims are
preserved; a second invocation cannot restart or resume an evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def safe(root, relative):
    root = Path(root).resolve()
    require(not Path(relative).is_absolute(), 'Absolute sealed path')
    target = (root / relative).resolve()
    require(target.is_relative_to(root), 'Sealed path escapes root')
    return target


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def latest(root, value):
    print(json.dumps({'phase':value.get('phase'),'status':value.get('status'),
          'completed_joint_workers':len(value.get('completed_joint_workers', [])),
          'active_count':len(value.get('active', [])),'observed_unix':time.time()}),flush=True)
    target = root / 'control' / 'pipeline_latest.json'
    temp = target.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, target)


def verify_seals(root, seal_sha):
    seal_path = root / 'operation_integrity.json'
    require(sha(seal_path) == seal_sha, 'Operation seal raw SHA differs from launch authorization')
    seal = read(seal_path)
    for rel, expected in seal['files'].items():
        require(sha(safe(root, rel)) == expected, 'Operation file changed: ' + rel)
    inputs = read(root / 'input_integrity.json')
    for rel, expected in inputs['files'].items():
        require(sha(safe(root, rel)) == expected, 'Frozen input file changed: ' + rel)
    require(inputs['contract_sha256'] == seal['contract_sha256'], 'Input/operation contract differs')
    return seal


def load_adapter(root):
    path = root / 'control' / 'evaluate.py'
    spec = importlib.util.spec_from_file_location('frozen_test_operational_adapter', path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    return adapter


def child_environment(contract, root, job):
    env = dict(os.environ)
    env.update(contract['runtime']['environment'])
    # Prior training-root values are process-local history only. Every import and
    # runtime cache for this evaluation is confined to its own sealed bundle.
    env['SOURCE_REVISION'] = contract['sources'][job['study']]['revision']
    env['PYTHONPATH'] = str(safe(root, contract['sources'][job['study']]['root']))
    env['MPLCONFIGDIR'] = str(root / 'results' / job['id'] / 'runtime_cache' / 'matplotlib')
    env['XDG_CACHE_HOME'] = str(root / 'results' / job['id'] / 'runtime_cache')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def gpu_health(contract):
    run = lambda args: subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()
    row = run(['nvidia-smi', '--query-gpu=uuid,name,driver_version', '--format=csv,noheader']).splitlines()
    require(len(row) == 1 and row[0].split(',')[0].strip() == contract['runtime']['gpu_uuid'],
            'Pinned physical GPU differs')
    compute = run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'])
    require(not compute, 'GPU is already in use; no evaluation launched')
    require(not Path('/var/run/reboot-required').exists(), 'Host reboot is pending')
    return {'observed_unix': time.time(), 'gpu': row[0], 'compute_processes': compute,
            'python_executable': sys.executable, 'reboot_required': False}


def native_probes(contract, root):
    routes = {}
    for job in contract['jobs']:
        routes.setdefault((job['study'], job['arm']), job)
    receipts = {}
    for (study, arm), job in sorted(routes.items()):
        name = study + '__' + arm
        dest = root / 'control' / 'native_probes' / name
        dest.mkdir(parents=True, exist_ok=False)
        argv = [contract['runtime']['python'], str(root / 'control' / 'evaluate.py'),
                '--root', str(root), '--native-probe', '--study', study, '--arm', arm, '--device', 'cuda']
        start = time.time()
        with (dest / 'probe.log').open('x', encoding='utf-8') as log:
            proc = subprocess.Popen(argv, cwd=root, env=child_environment(contract, root, job),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                write_new(dest / 'launch.json', {'pid': proc.pid, 'ppid': os.getpid(), 'argv': argv,
                          'started_unix': start, 'contract_sha256': contract['contract_sha256']})
                rc = proc.wait(timeout=180)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                write_new(dest / 'worker_exit.json', {'exit_code': proc.returncode,
                          'actual_child_exit_observed': True, 'pid': proc.pid, 'ppid': os.getpid(),
                          'argv': argv, 'started_unix': start, 'completed_unix': time.time(),
                          'contract_sha256': contract['contract_sha256']})
        require(rc == 0, 'Native synthetic route probe failed: ' + name)
        lines = (dest / 'probe.log').read_text().splitlines()
        result = json.loads(lines[-1])
        require(result['status'] == 'PASS' and result['contract_sha256'] == contract['contract_sha256']
                and result['native_pinned_contract_checked'] is True
                and result['real_data_loaded'] is False and result['checkpoint_reads'] == 0
                and result['optimizer_updates'] == 0 and all(result['qualification'].values())
                and result.get('hooked_and_unhooked_all_original_outputs_bitwise_equal') is True
                and result.get('telemetry_finite') is True
                and result.get('actual_integer_diagonal_eventwise_bitwise_equal') is True
                and result['pid'] == proc.pid and result['ppid'] == os.getpid()
                and argv == [contract['runtime']['python'], *result['argv']],
                'Incomplete native probe qualification: ' + name)
        write_new(dest / 'receipt.json', result)
        receipts[name] = sha(dest / 'receipt.json')
    require(len(receipts) == 3, 'Missing native history source/arm route')
    write_new(root / 'control' / 'native_qualification.json', {
        'status': 'PASS', 'contract_sha256': contract['contract_sha256'],
        'routes': receipts, 'synthetic_only': True, 'completed_unix': time.time()})


def joint_identity(seed, split):
    require(seed in (42,52,62) and split in ('train','validation'), 'Foreign joint worker')
    return 'seed' + str(seed) + '__' + split


def run_joint(contract, root, seed, split, total_start, state):
    """One actual child only; each resident three-model evaluation shares its PID."""
    identity = joint_identity(seed, split)
    dest = root / 'results' / ('seed' + str(seed)) / (split + '_joint')
    dest.mkdir(parents=True, exist_ok=False)
    job = next(j for j in contract['jobs'] if j['seed'] == seed)
    argv = [contract['runtime']['python'], str(root / 'control' / 'evaluate.py'),
            '--root', str(root), '--seed', str(seed), '--split', split, '--device', 'cuda']
    start, mono_start = time.time(), time.monotonic()
    log = (dest / 'worker.log').open('x', encoding='utf-8')
    proc = None
    exit_recorded = False
    timed_out = aborted = False

    def stop_owned():
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    def record_exit():
        nonlocal exit_recorded
        require(proc is not None, 'No actual launched joint child')
        rc = proc.wait()
        write_new(dest / 'worker_exit.json', {'joint_worker_id': identity,
            'contract_sha256': contract['contract_sha256'], 'seed': seed, 'split': split,
            'exit_code': rc, 'actual_child_exit_observed': True, 'pid': proc.pid,
            'ppid': os.getpid(), 'argv': argv, 'started_unix': start,
            'completed_unix': time.time(), 'timed_out': timed_out, 'supervisor_aborted': aborted,
            'model_split_pass_count_expected': 3})
        exit_recorded = True
        return rc

    try:
        require(state.get('active') == [], 'Only one joint worker may be active')
        proc = subprocess.Popen(argv, cwd=root, env=child_environment(contract, root, job),
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        launch = {'joint_worker_id': identity, 'contract_sha256': contract['contract_sha256'],
            'seed': seed, 'split': split, 'pid': proc.pid, 'ppid': os.getpid(),
            'argv': argv, 'started_unix': start, 'resident_models_max': 3}
        write_new(dest / 'launch.json', launch)
        state.update(phase=split, active=[launch], observed_unix=time.time())
        latest(root, state)
        while proc.poll() is None:
            timed_out = (time.monotonic()-mono_start > contract['execution']['per_job_timeout_seconds']
                or time.monotonic()-total_start > contract['execution']['total_timeout_seconds'])
            if timed_out:
                stop_owned()
                break
            time.sleep(1)
        rc = record_exit()
        state.update(active=[], observed_unix=time.time())
        if rc == 0:
            state['completed_joint_workers'].append(identity)
        latest(root, state)
        require(rc == 0, 'Joint worker failed (no automatic retry): '+identity+' exit='+str(rc))
    finally:
        if proc is not None and not exit_recorded:
            aborted = True
            try:
                stop_owned()
                record_exit()
            except BaseException as error:
                write_new(dest / 'cleanup_failure.json', {'error':repr(error),
                    'pid':proc.pid,'observed_exit_code':proc.poll(),'joint_worker_id':identity})
        log.close()


def process_bridge(actual, receipt, contract):
    require(actual['exit_code'] == 0 and actual['actual_child_exit_observed'] is True
        and actual['pid'] == receipt['pid'] and actual['ppid'] == receipt['ppid']
        and actual['contract_sha256'] == receipt['contract_sha256'] == contract['contract_sha256']
        and actual['argv'] == [contract['runtime']['python'], *receipt['argv']]
        and actual['completed_unix'] >= receipt['completed_unix'], 'Actual joint process bridge differs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--operation-seal-sha256', required=True)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise InterruptedError('Joint supervisor received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    root = args.root.resolve()
    seal = verify_seals(root,args.operation_seal_sha256)
    adapter = load_adapter(root)
    contract = adapter.verify_contract(root)
    require(contract['contract_sha256'] == seal['contract_sha256'], 'Canonical joint contract differs')
    require(str(root) == contract['execution']['remote_root'], 'Not the authorized isolated remote root')
    require(Path(sys.executable).resolve() == Path(contract['runtime']['python']).resolve(), 'Wrong native Python')
    require(contract['execution']['maximum_workers'] == 1 and contract['execution']['automatic_retry'] is False,
            'Only one serial joint worker, no retries')
    prerequisite = contract['prerequisite_H1']
    require(prerequisite['status'] == 'complete' and prerequisite['actual_pipeline_exit0'] is True
            and prerequisite['actual_all_worker_exit0'] is True, 'RAF completion must precede H2 native inference')
    claim = {'contract_sha256':contract['contract_sha256'],'operation_seal_sha256':args.operation_seal_sha256,
             'pid':os.getpid(),'ppid':os.getppid(),'argv':sys.argv,'started_unix':time.time()}
    write_new(root / 'control/pipeline_claim.json',claim)
    state = {**claim,'status':'running','phase':'native_health','completed_joint_workers':[],
             'active':[],'observed_unix':time.time(),'model_split_passes_completed':0}
    total_start = time.monotonic()
    try:
        write_new(root / 'control/native_health.json',gpu_health(contract))
        state.update(phase='native_qualification',observed_unix=time.time())
        latest(root,state)
        native_probes(contract,root)
        adapter.verify_operations(root,contract)
        for split in ('train','validation'):
            for seed in (42,52,62):
                run_joint(contract,root,seed,split,total_start,state)
                state['model_split_passes_completed'] += 3
                latest(root,state)
        identities,model_receipts,joint_receipts,joint_exits = {},{},{},{}
        for split in ('train','validation'):
            for seed in (42,52,62):
                identity = joint_identity(seed,split)
                dest = root/'results'/('seed'+str(seed))/(split+'_joint')
                joint = adapter.verify_joint_receipt(root,contract,seed,split)
                child_exit = read(dest/'worker_exit.json')
                process_bridge(child_exit,joint,contract)
                joint_receipts[identity] = sha(dest/'receipt.json')
                joint_exits[identity] = sha(dest/'worker_exit.json')
                for job in [j for j in contract['jobs'] if j['seed'] == seed]:
                    path = root/'results'/job['id']/(split+'_metrics.json')
                    r = read(path)
                    identity_hash=(r['target_identity_sha256'],r['truth_sha256'])
                    require(identities.setdefault(split,identity_hash) == identity_hash,'Joint population differs across seeds/arms')
                    require(r['pid'] == joint['pid'] and r['argv'] == joint['argv'], 'Model pass must share actual worker')
                    model_receipts.setdefault(job['id'],{})[split]=sha(path)
        require(len(state['completed_joint_workers']) == len(joint_receipts) == len(joint_exits) == 6
                and len(model_receipts) == 9 and state['model_split_passes_completed'] == 18,
                'Six physical joints and18 model passes required')
        verify_seals(root,args.operation_seal_sha256)
        write_new(root/'control/pipeline_complete.json',{'status':'complete',
            'contract_sha256':contract['contract_sha256'],'model_split_passes':18,'joint_worker_count':6,
            'actual_all_joint_worker_exit0':True,'new_fits':0,'test_inference':0,'calibration_fits':0,
            'model_receipts':model_receipts,'joint_receipts':joint_receipts,'joint_worker_exits':joint_exits,
            'completed_joint_workers':state['completed_joint_workers'],'completed_unix':time.time(),
            'elapsed_seconds':time.monotonic()-total_start})
        state.update(status='complete',phase='complete',active=[],observed_unix=time.time())
        latest(root,state)
        return 0
    except BaseException as error:
        write_new(root/'control/pipeline_failure.json',{'status':'failed','phase':state['phase'],
            'contract_sha256':contract['contract_sha256'],'error':type(error).__name__,
            'message':str(error),'traceback':traceback.format_exc(),'completed_unix':time.time(),
            'automatic_retry':False})
        state.update(status='failed',observed_unix=time.time())
        latest(root,state)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
