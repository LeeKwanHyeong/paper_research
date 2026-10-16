"""Lease/owner safety checks for isolated inference supervision, no GPU work."""
import importlib.util
import json
from pathlib import Path
import time

import pytest


PATH = Path(__file__).parents[1] / 'scripts' / 'run_titantpp_history_instacart_test_pipeline.py'
spec = importlib.util.spec_from_file_location('test_inference_pipeline', PATH)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class Child:
    def __init__(self, pid):
        self.pid, self.rc = pid, None

    def poll(self):
        return self.rc

    def wait(self, timeout=None):
        assert self.rc is not None, 'A live mock child must be stopped before wait'
        return self.rc


@pytest.fixture
def setup(tmp_path, monkeypatch):
    contract = {'contract_sha256': 'a' * 64,
                'runtime': {'python': '/frozen/python', 'environment': {'PYTHONPATH': '/old/root'}},
                'sources': {'history': {'revision': 'r', 'root': 'sources/history'}},
                'execution': {'per_job_timeout_seconds': 100, 'total_timeout_seconds': 100}}
    jobs = [dict(id=f'arm{i}', study='history', dataset='dataset', seed=42, arm=str(i)) for i in range(3)]
    children, killed = [], []
    def popen(*args, **kwargs):
        child = Child(800000 + len(children))
        children.append(child)
        return child
    def killpg(pid, signal):
        killed.append(pid)
        next(c for c in children if c.pid == pid).rc = -signal
    monkeypatch.setattr(p.subprocess, 'Popen', popen)
    monkeypatch.setattr(p.os, 'killpg', killpg)
    state = {'validation_completed': [], 'test_completed': []}
    return tmp_path, contract, jobs, children, killed, state


def test_second_launch_failure_reaps_first_child_and_preserves_exit(setup, monkeypatch):
    root, contract, jobs, children, killed, state = setup
    original = p.subprocess.Popen
    def second_fails(*args, **kwargs):
        if children:
            raise OSError('mock launch failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(p.subprocess, 'Popen', second_fails)
    with pytest.raises(OSError, match='mock launch failure'):
        p.run_group(contract, root, jobs, 'validation', time.monotonic(), state)
    assert killed == [children[0].pid]
    result = json.loads((root / 'results/arm0/validation_worker_exit.json').read_text())
    assert result['exit_code'] == -15 and result['actual_child_exit_observed']
    assert result['supervisor_aborted']


def test_launch_receipt_failure_also_tracks_and_reaps_new_child(setup, monkeypatch):
    root, contract, jobs, children, killed, state = setup
    original = p.write_new
    def launch_write_fails(path, value):
        if Path(path).name == 'validation_launch.json':
            raise OSError('mock disk failure')
        return original(path, value)
    monkeypatch.setattr(p, 'write_new', launch_write_fails)
    with pytest.raises(OSError, match='mock disk failure'):
        p.run_group(contract, root, jobs, 'validation', time.monotonic(), state)
    assert killed == [children[0].pid]
    assert (root / 'results/arm0/validation_worker_exit.json').exists()


def test_status_failure_reaps_exact_three_owned_children(setup, monkeypatch):
    root, contract, jobs, children, killed, state = setup
    monkeypatch.setattr(p, 'latest', lambda *args: (_ for _ in ()).throw(OSError('status unavailable')))
    with pytest.raises(OSError, match='status unavailable'):
        p.run_group(contract, root, jobs, 'validation', time.monotonic(), state)
    assert killed == [child.pid for child in children] and len(killed) == 3
    assert all((root / f'results/arm{i}/validation_worker_exit.json').exists() for i in range(3))


def test_sealed_environment_removes_old_training_root(setup):
    root, contract, jobs, *_ = setup
    env = p.child_environment(contract, root, jobs[0])
    assert env['PYTHONPATH'] == str(root / 'sources/history')
    assert str(root) in env['MPLCONFIGDIR'] and str(root) in env['XDG_CACHE_HOME']
    assert env['SOURCE_REVISION'] == 'r'


def test_foreign_seed_group_never_starts_children(setup):
    root, contract, jobs, children, *_ = setup
    jobs[2]['seed'] = 52
    with pytest.raises(ValueError, match='one dataset/seed'):
        p.run_group(contract, root, jobs, 'validation', time.monotonic(), {})
    assert children == []
