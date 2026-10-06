"""Single owned worker for the approved, separate two-scalar head experiment."""
from pathlib import Path
import argparse
import datetime
import fcntl
import hashlib
import json
import os
import math
import signal
import subprocess
import sys
import time
import traceback


def read(p):
    return json.loads(Path(p).read_text())


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix+'.tmp')
    temporary.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(p)


def validate(c, root):
    assert c['schema'] == 'titantpp_time_head_bias_refit_v1'
    assert c['host'] == '5080' and c['approved'] is True
    assert Path(c['root']).resolve() == root
    assert Path(__file__).resolve() == root/'operation/controller.py'
    assert c['evaluation_scope'] == 'Train_fit_Validation_development_selection'
    assert c['held_out_test_evaluated'] is False
    assert c['fitting']['parameters'] == ['b_t', 'w_raw']
    assert c['limits']['workers_per_host'] == 1 and not c['limits']['automatic_retry']
    assert len(c['jobs']) == 6 and len({j['id'] for j in c['jobs']}) == 6
    for j in c['jobs']:
        assert Path(j['output_dir']).resolve() == root/'run'/j['id']
    assert {(j['dataset'], j['seed']) for j in c['jobs']} == {
        (d, s) for d in ('yellow_trip_hourly', 'raf_spare_parts') for s in (42, 52, 62)}
    for rel, expected in c['operation']['files'].items():
        p = (root/rel).resolve(); assert p.is_relative_to(root)
        assert sha(p) == expected, rel
    assert digest(c['operation']['files']) == c['operation']['files_sha256']
    approval = read(root/'approval.json')
    assert approval['approved'] and approval['contract_sha256'] == digest(c)
    assert approval['hosts'] == ['5080'] and approval['user_instruction']
    start = read(root/'start_permit.json')
    assert start['contract_sha256'] == digest(c) and start['approval_sha256'] == digest(approval)
    assert start['deadline_unix']-start['issued_at_unix'] == c['limits']['total_wall_seconds']
    assert start['issued_at_unix'] <= time.time() < start['deadline_unix']
    assert not (root/'failure.json').exists(), 'Previous failure requires manual review'
    parent = Path(c['parent_root'])
    assert digest(read(parent/'execution_contract.json')) == c['parent_contract_sha256']
    assert read(parent/'status.json')['status'] == 'complete'
    assert read(parent/'supervisor_process_exit.json')['returncode'] == 0
    return start


def free_gpu(c):
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name',
        '--format=csv,noheader'], text=True)
    assert c['runtime']['gpu_uuid'] in rows and '5080' in rows
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).strip()
    assert not pids, 'GPU is occupied; do not displace another process'


def verify_result(c, j, jobroot):
    """Bind scientific success to sealed evidence, rather than status flags."""
    terminal = read(jobroot/'terminal_manifest.json')
    assert terminal['status'] == 'complete' and terminal['scientific_success'] is True
    assert terminal['job'] == j and terminal['contract_sha256'] == digest(c)
    assert terminal['held_out_test_evaluated'] is False and terminal['evaluation_scope'] == 'validation_only'
    assert terminal['trainable_scalar_count'] == 2 and terminal['original_checkpoint_sha_preserved'] is True
    required = {'result.json', 'history.json', 'endpoint_replays.json', 'startup_gate.json',
        'selected_refit.pt', 'last_refit.pt', 'input_receipt.json'}
    assert required <= set(terminal['files'])
    for rel, expected in terminal['files'].items():
        p = (jobroot/rel).resolve(); assert p.is_relative_to(jobroot)
        assert sha(p) == expected, rel
    r = read(jobroot/'result.json')
    assert r['status'] == 'complete' and r['job'] == j['id'] and r['contract_sha256'] == digest(c)
    assert r['original_checkpoint'] == j['checkpoint'] and r['source_files_sha256'] == c['source']['files_sha256']
    assert r['held_out_test_evaluated'] is False and r['quantity_output_unchanged'] is True
    assert r['trainable_tensors'] == ['b_t', 'w_raw'] and r['trainable_scalar_count'] == 2
    before = r['quantity_prediction_sha256_before']; after = r['quantity_prediction_sha256_after']
    assert before == after and set(before) == {'train', 'validation'}
    assert all(len(v) == 64 for v in before.values())
    assert r['frozen_state_sha256_before'] == r['frozen_state_sha256_after'] == terminal['frozen_state_sha256']
    assert len(r['frozen_state_sha256_before']) == 64
    gate = read(jobroot/'startup_gate.json')
    assert gate['status'] == 'passed' and gate['job'] == j['id'] and gate['contract_sha256'] == digest(c)
    assert gate['original_checkpoint'] == j['checkpoint'] and gate['E0'] == r['E0']
    assert gate['quantity_prediction_sha256'] == before and gate['frozen_state_sha256'] == r['frozen_state_sha256_before']
    history = read(jobroot/'history.json')['history']
    assert [row['epoch'] for row in history] == list(range(r['completed_epochs']+1))
    assert 1 <= r['completed_epochs'] <= 40
    assert all(math.isfinite(row['validation']['time_nll']) for row in history)
    best = min(history, key=lambda row: row['validation']['time_nll'])
    assert best['epoch'] == r['best_epoch'] and best['validation'] == r['selected']
    assert history[0]['validation'] == r['E0'] and history[-1]['validation'] == r['last']
    assert r['selected_is_identity'] == (r['best_epoch'] == 0)
    assert r['selected']['time_nll'] <= r['E0']['time_nll']
    for row in history:
        for key in ('qty_rmse', 'qty_mae'):
            assert row['validation'][key] == r['E0'][key]
            assert row['validation']['tail'][key] == r['E0']['tail'][key]
    assert sha(j['checkpoint']['path']) == j['checkpoint']['sha256']
    return r


def run_owned(cmd, log, deadline, environment, pidpath=None):
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=environment)
        if pidpath:
            write(pidpath, {'pid': proc.pid, 'ppid': os.getpid(), 'command': cmd, 'time': time.time()})
        code = proc.wait(timeout=max(1, deadline-time.time()))
        assert code == 0, 'Owned worker failed; exit '+str(code)
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=15)
            except subprocess.TimeoutExpired: proc.kill(); proc.wait()


def interrupted(signum, frame):
    raise RuntimeError('Owned supervisor interrupted by signal '+str(signum))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--contract', required=True)
    parser.add_argument('--mode', choices=('qualify', 'dispatch'), required=True)
    args = parser.parse_args(); path = Path(args.contract).resolve(); c = read(path)
    root = path.parent; start = validate(c, root); free_gpu(c)
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    runtime = root/'operation/titantpp_time_head_refit_runtime.py'
    python = c['runtime']['python']; base = [python, '-u', str(runtime), '--contract', str(path)]
    environment = {**os.environ, **c['runtime']['environment']}
    with (root/(args.mode+'.lock')).open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == 'qualify':
            assert not (root/'qualification/receipt.json').exists(), 'Do not repeat qualification'
            with (root/'qualification/claim.json').open('x') as f:
                json.dump({'started_unix': time.time(), 'owner_pid': os.getpid()}, f)
            with (root/'logs/qualification.log').open('x') as f:
                run_owned(base+['--job', c['jobs'][0]['id'], '--qualify'], f,
                    min(time.time()+1800, start['deadline_unix']), environment,
                    root/'qualification/worker_process.json')
            q = read(root/'qualification/receipt.json')
            assert q['status'] == 'passed' and q['contract_sha256'] == digest(c)
            write(root/'qualification/controller_exit.json', {'returncode': 0, 'time': time.time()})
            return
        qpath = root/'qualification/receipt.json'; q = read(qpath)
        assert q['status'] == 'passed' and q['contract_sha256'] == digest(c)
        assert read(root/'qualification/controller_exit.json')['returncode'] == 0
        assert q['representative_job'] == c['jobs'][0]['id']
        assert q['checkpoint_sha256'] == c['jobs'][0]['checkpoint']['sha256']
        assert q['trainable_tensors'] == ['b_t', 'w_raw'] and q['trainable_scalar_count'] == 2
        assert q['quantity_output_unchanged'] is True and q['real_optimizer_updates'] == 0
        launch = read(root/'training_permit.json')
        assert launch['approved'] is True and launch['contract_sha256'] == digest(c)
        assert launch['qualification_sha256'] == sha(qpath)
        with (root/'dispatch.claim').open('x') as f:
            json.dump({'owner_pid': os.getpid(), 'contract_sha256': digest(c),
                'started_unix': time.time()}, f)
        write(root/'supervisor.json', {'pid': os.getpid(), 'argv': sys.argv,
            'contract_sha256': digest(c), 'starts_training': True,
            'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        done = []
        for j in c['jobs']:
            free_gpu(c)
            assert time.time() < start['deadline_unix']
            jobroot = Path(j['output_dir']).resolve(); assert jobroot == root/'run'/j['id']
            jobroot.mkdir(parents=True, exist_ok=False)
            deadline = min(time.time()+c['limits']['per_condition_seconds'], start['deadline_unix'])
            permit = {'schema': 'time_head_refit_training_permit_v1', 'contract_sha256': digest(c),
                'job': j['id'], 'owner_pid': os.getpid(), 'issued_at_unix': time.time(),
                'deadline_unix': deadline, 'qualification': {'path': str(qpath), 'sha256': sha(qpath)}}
            permitpath = jobroot/'training_permit.json'; write(permitpath, permit)
            write(root/'status.json', {'status': 'running', 'active_job': j['id'],
                'complete': done, 'owner_pid': os.getpid(), 'time': time.time()})
            cmd = base+['--job', j['id'], '--training-permit', str(permitpath),
                '--owner-pid', str(os.getpid())]
            with (root/'logs'/('fit_'+j['id']+'.log')).open('x') as log:
                run_owned(cmd, log, deadline, environment, jobroot/'worker_process.json')
            result = verify_result(c, j, jobroot)
            files = {str(p.relative_to(jobroot)): sha(p) for p in jobroot.rglob('*') if p.is_file()}
            write(jobroot/'supervisor_terminal_receipt.json', {'status': 'complete', 'scientific_success': True,
                'job': j['id'], 'contract_sha256': digest(c), 'files': files,
                'completed_unix': time.time(), 'held_out_test_evaluated': False})
            done.append(j['id'])
        write(root/'status.json', {'status': 'complete', 'complete': done,
            'active_job': None, 'owner_pid': os.getpid(), 'time': time.time()})
        write(root/'supervisor_process_exit.json', {'returncode': 0, 'time': time.time(),
            'contract_sha256': digest(c)})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        p = Path(sys.argv[sys.argv.index('--contract')+1]).resolve().parent
        write(p/'failure.json', {'error_type': type(exc).__name__, 'error': str(exc),
            'traceback': traceback.format_exc(), 'time': time.time(), 'automatic_retry': False,
            'held_out_test_evaluated': False})
        raise
