"""Observe one authorized Taxi fitting process; never resume or retry it."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import sys
import time
import traceback


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def under(root: Path, relative: str) -> Path:
    p = PurePosixPath(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('Unsafe payload path: ' + relative)
    candidate = root.joinpath(*p.parts)
    if not candidate.resolve().is_relative_to(root.resolve()) or candidate.is_symlink():
        raise ValueError('Escaping payload path: ' + relative)
    return candidate


def process_identity(pid: int) -> dict:
    p = Path('/proc') / str(pid)
    s = (p / 'stat').read_text()
    fields = s[s.rfind(')') + 2:].split()
    return {'pid': pid, 'start_ticks': int(fields[19]), 'state': fields[0],
            'pgid': int(fields[2]), 'cwd': os.readlink(p / 'cwd'),
            'argv': (p / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')}


def verify_inputs(root: Path, config: dict) -> None:
    if not config.get('input_files'):
        raise ValueError('Empty input inventory')
    for name, expected in config['input_files'].items():
        p = under(root, name)
        if not p.is_file() or sha(p) != expected:
            raise ValueError('Input mismatch: ' + name)


def gpu_guard(config: dict) -> dict:
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != config['boot_id']:
        raise ValueError('Host boot identity changed')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader,nounits'],
                         capture_output=True, text=True, check=True, timeout=15)
    if gpu.stdout.strip().splitlines() != [config['gpu_uuid']]:
        raise ValueError('GPU identity changed')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                          capture_output=True, text=True, check=True, timeout=15)
    if apps.stdout.strip():
        raise ValueError('GPU acquired by another task; no process will be stopped')
    return {'gpu_uuid': config['gpu_uuid'], 'compute_processes': [],
            'boot_id': config['boot_id'], 'checked_unix': time.time()}


def validated_argv(value: object) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(x, str) and '\0' not in x for x in value):
        raise ValueError('Command must be a nonempty argv list')
    return value


def runner_command(value: object, allowed_modes: set[str]) -> list[str]:
    argv = validated_argv(value)
    if argv.count('--mode') != 1:
        raise ValueError('Expected one explicit runner mode')
    index = argv.index('--mode')
    if index + 1 >= len(argv) or argv[index + 1] not in allowed_modes:
        raise ValueError('Command mode outside authorized stage')
    if not any(x == 'paper.scripts.run_taxi_internal_holdout_diagnostic'
               or Path(x).name == 'run_taxi_internal_holdout_diagnostic.py' for x in argv):
        raise ValueError('Unexpected experiment entry point')
    return argv


def prelaunch_command(root: Path, config: dict, value: object) -> list[str]:
    argv = validated_argv(value)
    native = root / 'native_smoke.py'
    prefix = [config['python'], str(native)]
    if argv[:3] == [config['python'], '-u', str(native)]:
        prefix = argv[:3]
    if argv[:len(prefix)] != prefix:
        return runner_command(argv, {'synthetic', 'prepare'})
    expected = config['input_files'].get('native_smoke.py')
    if not expected or sha(under(root, 'native_smoke.py')) != expected:
        raise ValueError('Native smoke must match its audited payload SHA')
    rest = argv[len(prefix):]
    if len(rest) != 6 or set(rest[::2]) != {'--prepared', '--device', '--output'}:
        raise ValueError('Unexpected native smoke arguments')
    options = dict(zip(rest[::2], rest[1::2]))
    prepared = under(root, config.get('output_relative_path', 'run')) / 'prepared'
    if options != {'--prepared': str(prepared), '--device': 'cuda:0',
                   '--output': str(root / 'NATIVE_SMOKE.json')}:
        raise ValueError('Native smoke paths or device differ from authorized scope')
    return argv


def stop_own_child_after_timeout(child: subprocess.Popen) -> int:
    # Only this supervisor's newly created process group can be signalled.
    os.killpg(child.pid, signal.SIGTERM)
    try:
        return child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        return child.wait()


def run_child(root: Path, config: dict, argv: list[str], label: str, timeout: int) -> dict:
    started = time.time()
    out_path, err_path = root / (label + '.stdout.log'), root / (label + '.stderr.log')
    with out_path.open('xb') as out, err_path.open('xb') as err:
        child = subprocess.Popen(argv, cwd=config['source_root'], env=os.environ.copy(),
                                 stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 start_new_session=True)
        identity = process_identity(child.pid)
        write_new(root / (label + '.STARTED.json'), {
            'parent_observer': process_identity(os.getpid()), 'child': identity,
            'started_unix': started, 'timeout_seconds': timeout, 'command': argv,
        })
        timed_out = False
        try:
            code = child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            code = stop_own_child_after_timeout(child)
    result = {'actual_child_exit_code': code, 'hard_timeout': timed_out,
              'elapsed_seconds': time.time() - started, 'child': identity,
              'streams': {p.name: {'bytes': p.stat().st_size, 'sha256': sha(p)} for p in [out_path, err_path]}}
    write_new(root / (label + '.ACTUAL_EXIT.json'), result)
    return result


def main() -> int:
    root = Path(__file__).resolve().parent
    config = json.loads((root / 'EXECUTION.json').read_text())
    if str(root) != config['remote_root'] or config.get('host') != '5090':
        raise ValueError('Unexpected experiment location/host')
    if config.get('test_access') is not False or config.get('fresh_fit_count', 1) != 1:
        raise ValueError('Scope must be one fit with Test locked')
    if not Path(config['source_root']).resolve().is_relative_to(root):
        raise ValueError('Source must be the isolated payload copy')
    timeout = int(config['timeout_seconds'])
    if timeout != 86400:
        raise ValueError('Expected the fixed 24-hour hard timeout')
    with (root / 'ATTEMPT.claim').open('x') as f:
        f.write(str(os.getpid()) + '\n')
    write_new(root / 'SUPERVISOR_STARTED.json', {
        'identity': process_identity(os.getpid()), 'boot_id': config['boot_id'],
        'started_unix': time.time(), 'scope': 'one fresh Taxi seed42 fit; Test locked',
    })
    verify_inputs(root, config)
    write_new(root / 'GPU_GUARD_PRELAUNCH.json', gpu_guard(config))
    for i, argv in enumerate(config.get('prelaunch_commands', []), 1):
        checked_argv = prelaunch_command(root, config, argv)
        result = run_child(root, config, checked_argv, f'prelaunch_{i:02d}',
                           int(config.get('prelaunch_timeout_seconds', 120)))
        if result['actual_child_exit_code'] != 0 or result['hard_timeout']:
            raise RuntimeError(f'Prelaunch {i} failed; fit was not started')
        if str(root / 'native_smoke.py') in checked_argv:
            smoke = json.loads((root / 'NATIVE_SMOKE.json').read_text())
            if not (smoke.get('status') == 'PASS' and smoke.get('synthetic_events') == 4
                    and smoke.get('optimizer_steps') == 1
                    and smoke.get('real_data_training') is False
                    and smoke.get('Test_access') is False and smoke.get('device') == 'cuda:0'):
                raise RuntimeError('Native synthetic smoke receipt failed the scope check')
        verify_inputs(root, config)
    write_new(root / 'GPU_GUARD_BEFORE_FIT.json', gpu_guard(config))
    result = run_child(root, config, runner_command(config['command'], {'fit'}), 'fit', timeout)
    mismatches = [n for n, digest in config['input_files'].items()
                  if not under(root, n).is_file() or sha(under(root, n)) != digest]
    result.update(input_unchanged=not mismatches, input_mismatches=mismatches,
                  test_access=False, fresh_fit_count=1)
    write_new(root / 'PROCESS_EXIT.json', result)
    if mismatches:
        raise RuntimeError('Frozen inputs changed during fit')
    if result['hard_timeout']:
        return 124
    return result['actual_child_exit_code'] if result['actual_child_exit_code'] >= 0 else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        root = Path(__file__).resolve().parent
        detail = {'exception': type(exc).__name__, 'message': str(exc),
                  'traceback': traceback.format_exc(), 'failure_unix': time.time(),
                  'automatic_retry': False}
        failure = root / 'SUPERVISOR_FAILURE.json'
        if not failure.exists():
            write_new(failure, detail)
        traceback.print_exc()
        raise SystemExit(1)
