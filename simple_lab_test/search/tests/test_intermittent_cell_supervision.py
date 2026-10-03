"""Stdlib fake children exercise supervision; no model/data/GPU work is loaded."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SCRIPTS = Path(__file__).resolve().parents[3] / 'paper/scripts'
SPEC = importlib.util.spec_from_file_location('cell_supervision_controls', SCRIPTS / 'intermittent_cell_controls.py')
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)


def limits(total=2.0, **stages):
    caps = {'preflight_and_finalize': 1.5, 'validation': 1.5, 'train_gradient': 1.5}
    caps.update(stages)
    return {'max_wall_seconds': total, 'stage_seconds': caps,
            'max_validation_forward_batches': 4050, 'max_train_forward_batches': 384,
            'max_autograd_calls': 18816, 'max_cuda_allocated_bytes': 16 * 1024**3}


def child(body, *, authenticated=True):
    prefix = f"import sys; sys.path.insert(0, {str(SCRIPTS)!r})\n"
    prefix += "import os, time, signal\nfrom intermittent_cell_controls import WorkerAuthority\nassert 'torch' not in sys.modules\n"
    if authenticated:
        prefix += "authority = WorkerAuthority.from_environment(started=float(os.environ['INTERMITTENT_CELL_PARENT_START']))\n"
    return [sys.executable, '-c', prefix + body]


def run_child(body, bounds=None):
    return controls.supervise_child(child(body), bounds or limits(), poll_interval=0.01, kill_grace=0.02)


def test_budget_accumulates_reentered_stages_and_reports_transitions():
    now = [0.0]
    reported = []
    budget = controls.Budget(limits(validation=0.3), clock=lambda: now[0], stage_reporter=reported.append)
    now[0] = 0.1
    budget.set_stage('validation')
    now[0] = 0.3
    budget.set_stage('preflight_and_finalize')
    now[0] = 0.4
    budget.set_stage('validation')
    now[0] = 0.51
    with pytest.raises(TimeoutError, match='validation cumulative'):
        budget.check()
    assert reported == ['preflight_and_finalize', 'validation', 'preflight_and_finalize', 'validation']
    assert budget.snapshot()['stage_seconds']['validation'] == pytest.approx(0.31)


@pytest.mark.parametrize('stage', ['preflight_and_finalize', 'validation', 'train_gradient'])
def test_parent_expires_each_stage_when_worker_stops_cooperating(stage):
    body = f"authority.set_stage({stage!r})\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(30)\n"
    started = time.monotonic()
    with pytest.raises(TimeoutError, match=stage + ' cumulative') as caught:
        run_child(body, limits(**{stage: 0.25}))
    assert time.monotonic() - started < 1.5
    assert caught.value.supervision_budget['stage_seconds'][stage] >= 0.25
    with pytest.raises(ProcessLookupError):
        os.getpgid(caught.value.supervision_pid)


def test_parent_total_deadline_survives_stage_switches():
    body = "authority.set_stage('validation')\ntime.sleep(.12)\nauthority.set_stage('train_gradient')\ntime.sleep(30)\n"
    with pytest.raises(TimeoutError, match='Total diagnostic') as caught:
        run_child(body, limits(total=0.3))
    assert caught.value.supervision_budget['elapsed_seconds'] >= 0.3
    assert all(value < 1.5 for value in caught.value.supervision_budget['stage_seconds'].values())


def test_parent_reentry_does_not_reset_validation_budget():
    body = ("authority.set_stage('validation')\ntime.sleep(.16)\n"
            "authority.set_stage('train_gradient')\ntime.sleep(.05)\n"
            "authority.set_stage('validation')\ntime.sleep(30)\n")
    with pytest.raises(TimeoutError, match='validation cumulative') as caught:
        run_child(body, limits(validation=0.25))
    snapshot = caught.value.supervision_budget
    assert 0.25 <= snapshot['stage_seconds']['validation'] < 0.6
    assert snapshot['stage_seconds']['train_gradient'] >= 0.05


def test_normal_authenticated_completion_records_parent_stage_accounting():
    result = run_child("authority.set_stage('validation')\ntime.sleep(.02)\n"
                       "authority.set_stage('train_gradient')\ntime.sleep(.02)\n"
                       "authority.set_stage('preflight_and_finalize')\nauthority.finish()\nauthority.close()\n")
    assert result['authenticated'] and result['exit_code'] == 0
    assert result['stage_transitions'] == 3
    durations = result['budget']['stage_seconds']
    assert durations['validation'] >= 0.02 and durations['train_gradient'] >= 0.02
    assert sum(durations.values()) == pytest.approx(result['budget']['elapsed_seconds'], abs=1e-6)


def test_exit_without_handshake_or_done_is_not_success():
    with pytest.raises(ValueError, match='authority channel closed|without authenticated completion'):
        controls.supervise_child(child('pass\n', authenticated=False), limits(), poll_interval=0.01, kill_grace=0.02)
    with pytest.raises(ValueError, match='authority channel closed|without authenticated completion'):
        run_child('authority.close()\n')


def test_direct_worker_flags_denied_before_contract_or_model_load():
    command = [sys.executable, str(SCRIPTS / 'run_intermittent_cell_diagnosis.py'),
               '--contract', '/nonexistent/cell-contract', '--binding', '/nonexistent/cell-binding',
               '--worker', '--parent-start', str(time.monotonic())]
    clean_env = {key: value for key, value in os.environ.items() if key not in controls._AUTH_ENV}
    result = subprocess.run(command, env=clean_env, text=True, capture_output=True, timeout=5)
    assert result.returncode != 0
    assert 'inherited supervisor authority' in result.stderr
    assert 'FileNotFoundError' not in result.stderr


def test_forged_parent_identity_denied_before_inherited_fd_use():
    env = dict(os.environ)
    started = time.monotonic()
    env.update(zip(controls._AUTH_ENV, ['9999', 'f' * 64, '0', str(started), 'a' * 64, 'b' * 64], strict=True))
    command = child(f'WorkerAuthority.from_environment(started={started!r})\n', authenticated=False)
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=5)
    assert result.returncode != 0 and 'process ownership differs' in result.stderr


@pytest.mark.parametrize('change', ['authority.started += 100', "authority.context_sha256 = 'f' * 64"])
def test_worker_cannot_replace_parent_deadline_or_execution_context(change):
    with pytest.raises(ValueError, match='Invalid worker authority'):
        run_child(change + "\nauthority.set_stage('validation')\n")


def test_timeout_does_not_terminate_another_process_group():
    other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
    try:
        with pytest.raises(TimeoutError):
            run_child("authority.set_stage('validation')\ntime.sleep(30)\n", limits(validation=0.25))
        assert other.poll() is None
    finally:
        other.terminate()
        other.wait(timeout=3)


def test_preflight_elapsed_before_launch_is_charged_to_parent():
    with pytest.raises(TimeoutError, match='preflight_and_finalize cumulative'):
        controls.supervise_child(child('authority.finish()\n'), limits(preflight_and_finalize=.25),
                                 started=time.monotonic() - .3, poll_interval=.01, kill_grace=.02)


def parent_fixture(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    spec = importlib.util.spec_from_file_location('cell_supervision_test_runner', SCRIPTS / 'run_intermittent_cell_diagnosis.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'verify_binding', lambda *args, **kwargs: {})
    output = tmp_path / 'output'
    contract = {'server': {'output_dir': str(output)}, 'limits': {**limits(), 'max_output_bytes': 65536}}
    args = SimpleNamespace(contract=tmp_path / 'contract', binding=tmp_path / 'binding', approval=tmp_path / 'approval')
    return runner, output, contract, args


def fake_complete(output, context_sha256, started):
    output.mkdir(exist_ok=True)
    (output / 'worker_terminal_status.json').write_text(json.dumps({'status': 'complete', 'measurement_sha256': controls.MEASUREMENT_SHA}))
    elapsed = time.monotonic() - started
    return {'authenticated': True, 'exit_code': 0, 'execution_context_sha256': context_sha256,
            'stage_transitions': 3, 'budget': {'elapsed_seconds': elapsed,
                'stage_seconds': {'preflight_and_finalize': elapsed, 'validation': 0, 'train_gradient': 0}, 'counts': {}}}


def test_parent_alone_publishes_complete_after_authenticated_exit(monkeypatch, tmp_path):
    runner, output, contract, args = parent_fixture(monkeypatch, tmp_path)
    contexts = []
    def supervised(command, limits, *, started, context_sha256):
        contexts.append(context_sha256)
        return fake_complete(output, context_sha256, started)
    monkeypatch.setattr(runner, 'supervise_child', supervised)
    binding, approval = {'binding': 1}, {'approval': 2}
    runner.supervised(args, contract, binding, approval, time.monotonic())
    terminal = json.loads((output / 'terminal_status.json').read_text())
    expected_context = controls.sha_json({'measurement': controls.sha_json(contract), 'binding': controls.sha_json(binding), 'approval': controls.sha_json(approval)})
    assert contexts == [expected_context]
    assert terminal['status'] == 'complete' and terminal['authority'] == 'owning_supervisor'
    assert terminal['parent_budget']['elapsed_seconds'] >= 0


def test_worker_complete_cannot_override_parent_timeout(monkeypatch, tmp_path):
    runner, output, contract, args = parent_fixture(monkeypatch, tmp_path)
    def supervised(command, limits, *, started, context_sha256):
        fake_complete(output, context_sha256, started)
        raise TimeoutError('parent deadline after worker receipt')
    monkeypatch.setattr(runner, 'supervise_child', supervised)
    with pytest.raises(TimeoutError):
        runner.supervised(args, contract, {}, {}, time.monotonic())
    assert json.loads((output / 'worker_terminal_status.json').read_text())['status'] == 'complete'
    terminal = json.loads((output / 'terminal_status.json').read_text())
    assert terminal['status'] == 'timeout' and terminal['authority'] == 'owning_supervisor'


def test_failed_supervision_before_worker_output_still_has_parent_terminal(monkeypatch, tmp_path):
    runner, output, contract, args = parent_fixture(monkeypatch, tmp_path)
    def supervised(*args, **kwargs):
        raise ValueError('fake child preflight failure')
    monkeypatch.setattr(runner, 'supervise_child', supervised)
    with pytest.raises(ValueError, match='preflight failure'):
        runner.supervised(args, contract, {}, {}, time.monotonic())
    assert json.loads((output / 'terminal_status.json').read_text())['status'] == 'failed'


def test_competing_supervisor_cannot_write_winners_output(monkeypatch, tmp_path):
    runner, output, contract, args = parent_fixture(monkeypatch, tmp_path)
    def supervised(command, limits, *, started, context_sha256):
        receipt = fake_complete(output, context_sha256, started)
        with pytest.raises(BlockingIOError):
            runner.supervised(args, contract, {}, {}, started)
        assert not (output / 'terminal_status.json').exists()
        return receipt
    monkeypatch.setattr(runner, 'supervise_child', supervised)
    runner.supervised(args, contract, {}, {}, time.monotonic())
    assert json.loads((output / 'terminal_status.json').read_text())['status'] == 'complete'
