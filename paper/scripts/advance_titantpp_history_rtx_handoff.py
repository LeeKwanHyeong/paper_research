"""Advance only the already-approved RTX history handoff; never repair Linux.

An unlaunched5080 receives one read-only health query. Healthy/idle permits its
one-attempt sealed deploy/launch. Each launched owner is observed once; final
preservation is delegated once to the separate immutable-original finalizer.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone, timedelta
import fcntl
import json
import inspect
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from paper.scripts import observe_titantpp_history_rtx_handoff as observer
from paper.scripts import deploy_titantpp_history_rtx_handoff as deployment

BUNDLE = observer.BUNDLE


def verify_monitor_seal(bundle):
    path = bundle / 'control/monitor_integrity.json'
    observer.require(path.is_file(), 'Monitor seal missing; remote0 and activation0')
    seal = observer.read(path)
    observer.require(seal['contract_sha256'] == observer.CONTRACT, 'Monitor contract seal drift; remote0')
    for base, files in [(PROJECT, seal.get('project_files', {})), (bundle, seal.get('files', {}))]:
        for name, digest in files.items():
            observer.require(observer.sha(observer.safe(base, name)) == digest, 'Monitor raw SHA mismatch: ' + name)
    return {'status': 'verified', 'raw_sha256': observer.sha(path)}


def maintenance_evidence(process_text, lock_text, lock_identities):
    """Distinguish Ubuntu's exact shutdown waiter from active package work."""
    active, benign = [], []
    waiter = ['/usr/bin/python3', '/usr/share/unattended-upgrades/unattended-upgrade-shutdown', '--wait-for-signal']
    for line in process_text.splitlines():
        fields = line.split(None, 3)
        if len(fields) != 4:
            continue
        pid, ppid, comm, argv = fields
        if comm not in ('apt', 'apt-get', 'dpkg', 'unattended-upgr', 'unattended-upgrades') and 'unattended-upgrade' not in argv:
            continue
        row = {'pid': int(pid), 'ppid': int(ppid), 'comm': comm, 'argv': argv}
        (benign if argv.split() == waiter else active).append(row)
    held = []
    for line in lock_text.splitlines():
        for field in line.split():
            parts = field.split(':')
            if len(parts) != 3:
                continue
            try:
                identity = [int(parts[0], 16), int(parts[1], 16), int(parts[2])]
            except ValueError:
                continue
            for name, expected in lock_identities.items():
                if identity == expected:
                    held.append({'path': name, 'kernel_lock': line})
    return {'maintenance_processes': active, 'benign_shutdown_waiters': benign, 'package_locks_held': held}


def health_code(c, host):
    spec = c['hosts'][host]
    runtime = "import json,torch,numpy,polars; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'numpy':numpy.__version__,'polars':polars.__version__,'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}))"
    return '''import json,os,platform,subprocess,time
from pathlib import Path
spec=SPEC
host=HOST
def query(argv,env=None):
    try:
        r=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=40)
        return {'rc':r.returncode,'stdout':r.stdout.strip(),'stderr':r.stderr.strip()}
    except Exception as exc:
        return {'rc':None,'stdout':'','stderr':type(exc).__name__+': '+str(exc)}
env=dict(os.environ)
for key in ('LD_LIBRARY_PATH','CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES'):
    if key in spec['environment']:env[key]=spec['environment'][key]
gpu=query(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,driver_version','--format=csv,noheader'])
gpu_pids=query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'])
runtime=query([spec['python'],'-c',RUNTIME],env)
ps=query(['ps','-eo','pid=,ppid=,comm=,args='])
MAINTENANCE_FUNCTION
lock_rc=0
try:
    lock_ids={}
    for name in ('/var/lib/dpkg/lock','/var/lib/dpkg/lock-frontend','/var/lib/apt/lists/lock','/var/cache/apt/archives/lock'):
        path=Path(name)
        if path.exists():
            st=path.stat();lock_ids[name]=[os.major(st.st_dev),os.minor(st.st_dev),st.st_ino]
    maintenance=maintenance_evidence(ps['stdout'],Path('/proc/locks').read_text(),lock_ids)
except Exception as exc:
    lock_rc=1
    maintenance={'maintenance_processes':[],'package_locks_held':[],'package_lock_error':str(exc)}
driver=Path('/proc/driver/nvidia/version')
out={'host':host,'contract_sha256':CONTRACT,'root':spec['root'],'observed_unix':time.time(),'kernel':platform.release(),'driver_kernel':driver.read_text() if driver.exists() else None,'reboot_required':Path('/var/run/reboot-required').exists(),'gpu':gpu,'gpu_pids':gpu_pids,'runtime':runtime,'python_exists':Path(spec['python']).is_file(),'tmux_exists':Path(spec['tmux_binary']).is_file(),**maintenance,'package_lock_query_rc':lock_rc,'maintenance_query_rc':ps['rc'],'read_only':True,'shared_runtime_changes':False}
print(json.dumps(out,allow_nan=False))
'''.replace('MAINTENANCE_FUNCTION', inspect.getsource(maintenance_evidence)).replace('spec=SPEC', 'spec=' + repr(spec)).replace('host=HOST', 'host=' + repr(host)).replace('RUNTIME', repr(runtime)).replace('CONTRACT', repr(observer.CONTRACT))


def query_health(c, host):
    observer.require(host == '5080', 'Only unlaunched5080 health activation is authorized')
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', host, 'python3 -'], input=health_code(c, host), text=True, capture_output=True, timeout=150)
    observer.require(result.returncode == 0, 'Read-only health SSH failed: ' + result.stderr[-1200:])
    return json.loads(result.stdout)


def health_decision(c, host, health, now=None):
    spec = c['hosts'][host]
    now = time.time() if now is None else now
    observer.require(health['host'] == host and health['contract_sha256'] == observer.CONTRACT and health['root'] == spec['root'], 'Health identity mismatch')
    observer.require(-10 <= now - health['observed_unix'] <= 120, 'Fresh native health required')
    if health.get('reboot_required'):
        return 'external_wait_reboot_required'
    if health['gpu']['rc'] != 0 or health['gpu_pids']['rc'] != 0:
        return 'external_wait_NVML_or_driver_unhealthy'
    uuid = [line.split(',')[0].strip() for line in health['gpu']['stdout'].splitlines()]
    observer.require(uuid == [spec['gpu_uuid']], 'GPU UUID changed; no activation')
    if health['gpu_pids']['stdout'].strip():
        return 'external_wait_GPU_busy'
    if health.get('maintenance_query_rc') != 0 or health.get('package_lock_query_rc') != 0 or health.get('maintenance_processes') or health.get('package_locks_held'):
        return 'external_wait_OS_maintenance'
    if not health.get('python_exists') or not health.get('tmux_exists') or health['runtime']['rc'] != 0:
        return 'external_wait_existing_runtime_unavailable'
    runtime = json.loads(health['runtime']['stdout'])
    if any(runtime.get(k) != v for k, v in spec['runtime_expected'].items()) or runtime.get('cuda_available') is not True or runtime.get('device_count') != 1:
        return 'external_wait_existing_runtime_mismatch'
    return 'healthy_idle'


def preserve_health(bundle, host, health):
    directory = bundle / 'deployment'
    directory.mkdir(parents=True, exist_ok=True)
    latest = directory / ('health_' + host + '.json')
    if latest.exists():
        raw = latest.read_bytes()
        previous = json.loads(raw)
        stamp = datetime.fromtimestamp(previous['observed_unix'], timezone(timedelta(hours=9))).strftime('%Y%m%d_%H%M%SKST')
        old = directory / ('health_' + host + '_' + stamp + '_' + observer.sha(latest)[:12] + '.json')
        if old.exists():
            observer.require(old.read_bytes() == raw, 'Historical health preservation mismatch')
        else:
            with old.open('xb') as stream:
                stream.write(raw)
    history = directory / ('health_' + host + '_' + str(time.time_ns()) + '.json')
    observer.write(history, health, True)
    observer.write(latest, health)
    return {'latest': str(latest), 'history': str(history), 'raw_sha256': observer.sha(history)}


def prior_attempt(bundle, host):
    paths = [bundle / ('start_permit_' + host + '.json')]
    paths += [bundle / 'deployment' / (kind + '_' + host + '.json') for kind in ('deploy_intent', 'deploy_receipt', 'launch_intent', 'launch_failure', 'deploy_failure')]
    paths += list((bundle / 'deployment').glob('*failure*' + host + '*')) if (bundle / 'deployment').exists() else []
    return [str(path) for path in paths if path.exists()]


def activate_5080(c, *, bundle=BUNDLE, health_query=query_health, deploy=deployment.deploy, launch=deployment.launch):
    host = '5080'
    if (bundle / 'deployment/launch_receipt_5080.json').exists():
        return {'status': 'already_launched', 'health_SSH_calls': 0, 'deploy_calls': 0, 'launch_calls': 0}
    attempts = prior_attempt(bundle, host)
    if attempts:
        return {'status': 'user_action_required_preserved_incomplete_attempt', 'evidence': attempts, 'health_SSH_calls': 0, 'deploy_calls': 0, 'launch_calls': 0, 'automatic_retry': False}
    try:
        health = health_query(c, host)
        saved = preserve_health(bundle, host, health)
        decision = health_decision(c, host, health)
    except Exception as exc:
        error = {'status': 'health_observation_failed_not_training_failure', 'message': str(exc), 'observed_unix': time.time(), 'health_SSH_calls': 1, 'deploy_calls': 0, 'launch_calls': 0}
        observer.write(bundle / 'deployment' / ('health_error_5080_' + str(time.time_ns()) + '.json'), error, True)
        return error
    result = {'status': decision, 'health_SSH_calls': 1, 'deploy_calls': 0, 'launch_calls': 0, 'health_evidence': saved, 'observed_kst': observer.kst(health['observed_unix'])}
    if decision != 'healthy_idle':
        return result
    # Existing one-attempt helpers preserve their exclusive permits/intents.
    # The heartbeat never repeats either helper after partial/unknown outcomes.
    try:
        result['deploy_calls'] = 1
        result['deployment'] = deploy(host)
        result['launch_calls'] = 1
        result['launch'] = launch(host)
        result['status'] = 'native_pipeline_launched_not_fit_confirmation'
    except Exception as exc:
        result.update(status='user_action_required_preserved_incomplete_attempt', message=str(exc), automatic_retry=False)
        observer.write(bundle / 'deployment' / ('activation_failure_5080_' + str(time.time_ns()) + '.json'), result, True)
    return result


def finalize_once():
    from paper.scripts.finalize_titantpp_history_rtx_handoff import finalize
    return finalize()


def advance(*, observe_only=False, bundle=BUNDLE, verifier=observer.verify_local,
            instacart_check=observer.verify_instacart, a100_check=observer.local_a100_rows,
            activate=activate_5080, observe=observer.observe_host, finalize=finalize_once):
    c = verifier(bundle)
    monitor_seal = verify_monitor_seal(bundle)
    instacart = instacart_check()
    a100_rows, a100 = a100_check()
    lock_path = bundle / 'control/advance.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        activation = {'status': 'observe_only', 'health_SSH_calls': 0, 'deploy_calls': 0, 'launch_calls': 0} if observe_only else activate(c, bundle=bundle)
        hosts = {}
        for host in ('5080', '5090'):
            try:
                hosts[host] = observe(host, bundle=bundle)
            except Exception as exc:
                jobs = [j for j in c['jobs'] if j['host'] == host]
                hosts[host] = {'host': host, 'status': 'local_validation_failed_no_additional_query', 'message': str(exc), 'remote_calls': 0, 'rows': [{'job': j, 'state': 'unconfirmed'} for j in jobs]}
        try:
            completion = finalize()
        except Exception as exc:
            completion = {'status': 'finalization_failed_user_action_required', 'message': str(exc), 'automatic_retry': False}
        counts = observer.unique_counts(a100_rows, [row for host in hosts.values() for row in host['rows']])
        cleanup_path = observer.A100 / 'control/cleanup_receipt.json'
        cleanup = observer.read(cleanup_path) if cleanup_path.exists() else {'status': 'not_yet_locally_confirmed'}
        result = {'contract_sha256': observer.CONTRACT, 'local_report_unix': time.time(), 'monitor_seal': monitor_seal, 'Instacart': instacart, 'A100': a100, 'A100_cleanup_local': cleanup, 'A100_SSH_calls': 0, 'Pod_API_calls': 0, 'activation_5080': activation, 'hosts': hosts, 'counts': counts, 'finalization': completion, 'fixed_retrieval': 'waiting_no_automatic_launch'}
        path = bundle / 'hourly_monitor' / ('advance_' + str(time.time_ns()) + '.json')
        observer.write(path, result, True)
        observer.write(bundle / 'hourly_monitor/latest_advance.json', {'path': str(path), 'sha256': observer.sha(path)})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observe-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(advance(observe_only=args.observe_only), ensure_ascii=False, allow_nan=False))

if __name__ == '__main__':
    main()
