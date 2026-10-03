"""Stdlib execution controls; importing this module never loads model libraries."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import select
import signal
import socket
import subprocess
import time

MEASUREMENT_SHA = 'ca08be3e7070aaedac7279d46af4629dc59447769350caa6b4f8dbaad4872838'
OVERLAY_NAMES = frozenset({'run_intermittent_cell_diagnosis.py', 'intermittent_cell_controls.py',
    'intermittent_cell_gradients.py', 'intermittent_cell_statistics.py',
    'run_quantity_checkpoint_diagnosis.py', 'validate_intermittent_cell_contract.py'})


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_binding(contract, binding, overlay_root, *, approval=None, execute=False):
    require(sha_json(contract) == MEASUREMENT_SHA, 'Frozen measurement contract changed')
    require(binding['schema'] == 'intermittent_cell_execution_binding_v1', 'Wrong execution binding')
    require(binding['measurement_canonical_sha256'] == MEASUREMENT_SHA, 'Wrong measurement binding')
    require(binding['status'] == 'implemented_cpu_verified_pending_gpu_approval', 'Binding is not ready for approval')
    require(binding['gpu_execution_authorized'] is False, 'Approval must be a separate user receipt')
    require(binding['limits'] == contract['limits'], 'Bound limits differ from measurement contract')
    require(binding['server'] == contract['server']['ssh_alias'], 'Wrong server binding')
    require(set(binding['overlay_files']) == OVERLAY_NAMES, 'Incomplete or expanded overlay import closure')
    root = Path(overlay_root).resolve()
    for name, expected in binding['overlay_files'].items():
        path = root / name
        require(path.resolve().is_relative_to(root) and sha_file(path) == expected, f'Overlay hash differs: {name}')
    if execute:
        require(isinstance(approval, dict), 'Separate GPU diagnostic approval is required before execution')
        require(approval.get('schema') == 'intermittent_cell_execution_approval_v1'
                and approval.get('status') == 'approved', 'Unapproved diagnostic')
        require(approval.get('binding_canonical_sha256') == sha_json(binding)
                and approval.get('measurement_canonical_sha256') == MEASUREMENT_SHA, 'Approval does not bind this exact execution')
        require(approval.get('server') == '5090' and approval.get('limits') == contract['limits'], 'Approval resource scope differs')
        require(approval.get('checkpoint_model_replay_authorized') is True
                and approval.get('optimizer_updates') == 0 and approval.get('held_out_access_allowed') is False, 'Approval scope differs')
        require(isinstance(approval.get('user_message'), str) and approval['user_message'].strip()
                and isinstance(approval.get('recorded_at'), str) and approval['recorded_at'].strip(), 'Missing user authorization provenance')
    return {'binding_canonical_sha256': sha_json(binding), 'measurement_canonical_sha256': MEASUREMENT_SHA,
            'overlay_files_verified': len(OVERLAY_NAMES), 'execution_approved': bool(execute)}


class Budget:
    def __init__(self, limits, *, started=None, clock=time.monotonic, memory_bytes=None, stage_reporter=None):
        self.limits, self.clock, self.memory_bytes = limits, clock, memory_bytes
        self.stage_reporter = stage_reporter
        self.started = clock() if started is None else started
        self.stage = 'preflight_and_finalize'
        self.stage_started = clock()
        require(math.isfinite(self.started) and self.started <= self.stage_started, 'Invalid diagnostic start time')
        require(math.isfinite(limits['max_wall_seconds']) and limits['max_wall_seconds'] > 0
                and self.stage in limits['stage_seconds']
                and all(math.isfinite(value) and value > 0 for value in limits['stage_seconds'].values()), 'Invalid time budgets')
        self.elapsed = {name: 0.0 for name in limits['stage_seconds']}
        self.elapsed[self.stage] = max(0.0, self.stage_started - self.started)
        self.counts = {'validation_forward_batches': 0, 'train_forward_batches': 0, 'autograd_calls': 0}
        if self.stage_reporter is not None:
            self.stage_reporter(self.stage)

    def check(self):
        now = self.clock()
        if now - self.started >= self.limits['max_wall_seconds']:
            raise TimeoutError('Total diagnostic deadline reached; no automatic retry')
        if self.elapsed[self.stage] + now - self.stage_started >= self.limits['stage_seconds'][self.stage]:
            raise TimeoutError(f'{self.stage} cumulative deadline reached')
        if self.memory_bytes is not None:
            require(self.memory_bytes() <= self.limits['max_cuda_allocated_bytes'], 'CUDA allocation limit exceeded')

    def set_stage(self, stage):
        self.check()
        require(stage in self.elapsed, 'Unknown diagnostic stage')
        # Parent acknowledges the transition before the worker starts new work.
        if self.stage_reporter is not None:
            self.stage_reporter(stage)
        now = self.clock()
        self.elapsed[self.stage] += now - self.stage_started
        self.stage, self.stage_started = stage, now
        self.check()

    def remaining(self):
        self.check()
        now = self.clock()
        return min(self.limits['max_wall_seconds'] - (now - self.started),
                   self.limits['stage_seconds'][self.stage] - self.elapsed[self.stage] - (now - self.stage_started))

    def consume(self, kind, amount=1):
        require(type(amount) is int and amount >= 0 and kind in self.counts, 'Invalid work counter')
        require(self.counts[kind] + amount <= self.limits['max_' + kind], f'{kind} budget exceeded')
        self.counts[kind] += amount
        self.check()

    def snapshot(self):
        now = self.clock()
        stages = dict(self.elapsed)
        stages[self.stage] += now - self.stage_started
        return {'elapsed_seconds': now - self.started, 'stage_seconds': stages, 'counts': dict(self.counts)}


@contextmanager
def phase_alarm(budget):
    """Worker-side Python backstop; the separate parent bounds native/CUDA hangs."""
    remaining = budget.remaining()
    def expired(signum, frame):
        raise TimeoutError(f'{budget.stage} hard deadline reached')
    previous = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    timer_started = time.monotonic()
    if previous_timer[0] > 0:
        remaining = min(remaining, previous_timer[0])
    signal.setitimer(signal.ITIMER_REAL, max(0.001, remaining))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, previous_timer[0] - (time.monotonic() - timer_started)), previous_timer[1])


_AUTH_ENV = ('INTERMITTENT_CELL_PARENT_FD', 'INTERMITTENT_CELL_PARENT_NONCE',
             'INTERMITTENT_CELL_PARENT_PID', 'INTERMITTENT_CELL_PARENT_START',
             'INTERMITTENT_CELL_LIMITS_SHA256', 'INTERMITTENT_CELL_CONTEXT_SHA256')
_MAX_IPC_BYTES = 8192


class WorkerAuthority:
    """An inherited socket and live supervisor handshake, never a CLI flag alone."""
    def __init__(self, channel, nonce, parent_pid, started, limits_sha256, context_sha256):
        self.channel, self.nonce, self.parent_pid = channel, nonce, parent_pid
        self.started, self.limits_sha256 = started, limits_sha256
        self.context_sha256 = context_sha256
        self.sequence = 0
        self.closed = False

    @classmethod
    def from_environment(cls, *, started):
        require(all(name in os.environ for name in _AUTH_ENV), 'Worker requires inherited supervisor authority')
        values = [os.environ.pop(name) for name in _AUTH_ENV]
        fd, nonce, parent_pid, recorded_start, limits_sha, context_sha = values
        fd, parent_pid, recorded_start = int(fd), int(parent_pid), float(recorded_start)
        require(fd >= 3 and len(nonce) == len(context_sha) == len(limits_sha) == 64 and math.isfinite(recorded_start)
                and recorded_start == started, 'Invalid inherited worker authority')
        require(os.getppid() == parent_pid and parent_pid != os.getpid()
                and os.getpgrp() == os.getpid(), 'Worker process ownership differs from supervisor launch')
        channel = socket.socket(fileno=fd)
        channel.set_inheritable(False)
        channel.settimeout(5.0)
        authority = cls(channel, nonce, parent_pid, started, limits_sha, context_sha)
        try:
            authority._request('hello')
        except BaseException:
            channel.close()
            raise
        return authority

    def _request(self, kind, **fields):
        require(not self.closed and os.getppid() == self.parent_pid and os.getpgrp() == os.getpid(),
                'Worker lost its owning supervisor')
        message = {'type': kind, 'sequence': self.sequence, 'nonce': self.nonce,
                   'pid': os.getpid(), 'ppid': os.getppid(), 'pgid': os.getpgrp(),
                   'started': self.started, 'limits_sha256': self.limits_sha256,
                   'context_sha256': self.context_sha256, **fields}
        self.channel.sendall(json.dumps(message, allow_nan=False).encode() + b'\n')
        data = bytearray()
        while not data.endswith(b'\n'):
            block = self.channel.recv(1024)
            require(bool(block), 'Supervisor authority channel closed')
            data.extend(block)
            require(len(data) <= _MAX_IPC_BYTES, 'Oversized supervisor response')
        response = json.loads(data)
        require(response == {'type': 'ack', 'sequence': self.sequence, 'nonce': self.nonce,
                             'parent_pid': self.parent_pid, 'worker_pid': os.getpid(),
                             'started': self.started, 'limits_sha256': self.limits_sha256,
                             'context_sha256': self.context_sha256},
                'Supervisor handshake mismatch')
        self.sequence += 1

    def check_owner(self, *, started, limits, context_sha256):
        require(started == self.started and sha_json(limits) == self.limits_sha256
                and context_sha256 == self.context_sha256,
                'Worker deadline/limits/execution context differ from its supervisor')
        self._request('check')

    def set_stage(self, stage):
        self._request('stage', stage=stage)

    def finish(self):
        self._request('done')

    def close(self):
        if not self.closed:
            self.closed = True
            self.channel.close()


def _terminate_owned_group(process, grace):
    """Only the group created by this Popen(start_new_session=True) is signalled."""
    require(process.pid != os.getpgrp(), 'Refusing to terminate supervisor process group')
    if process.poll() is None:
        require(os.getpgid(process.pid) == process.pid, 'Owned worker changed process group')
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    # Also clean up owned descendants when their leader exits before them.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=max(1.0, grace))


def supervise_child(command, limits, *, started=None, context_sha256=None, env=None, cwd=None, poll_interval=0.05, kill_grace=1.0):
    """Enforce total and cumulative stage deadlines from the parent's own clock.

The worker sends only stage names. It cannot supply elapsed time, reset a cap,
or proceed past a stage boundary without the parent's acknowledgement.
"""
    budget = Budget(limits, started=started)
    budget.check()
    require(0 < poll_interval <= 1 and 0 <= kill_grace <= 3, 'Invalid supervision polling/cleanup bound')
    context_sha256 = sha_json({}) if context_sha256 is None else context_sha256
    require(isinstance(context_sha256, str) and len(context_sha256) == 64, 'Invalid execution context hash')
    parent_channel, child_channel = socket.socketpair()
    nonce = secrets.token_hex(32)
    child_env = dict(os.environ if env is None else env)
    authority_values = (str(child_channel.fileno()), nonce, str(os.getpid()), str(budget.started), sha_json(limits), context_sha256)
    child_env.update(zip(_AUTH_ENV, authority_values, strict=True))
    process = None
    sequence, authenticated, done, transitions = 0, False, False, 0
    buffered = bytearray()
    try:
        process = subprocess.Popen(command, env=child_env, cwd=cwd, start_new_session=True,
                                   pass_fds=(child_channel.fileno(),))
        child_channel.close()
        parent_channel.setblocking(False)
        channel_open = True
        while True:
            budget.check()
            code = process.poll()
            readable = []
            if channel_open:
                readable, _, _ = select.select([parent_channel], [], [], min(poll_interval, max(0, budget.remaining())))
            elif code is None:
                # No cooperative calls are needed for the parent to expire a cap.
                time.sleep(min(poll_interval, max(0, budget.remaining())))
            if readable:
                block = parent_channel.recv(4096)
                if not block:
                    channel_open = False
                    require(not buffered, 'Truncated worker authority message')
                else:
                    buffered.extend(block)
                    require(len(buffered) <= _MAX_IPC_BYTES, 'Oversized worker authority message')
                    while b'\n' in buffered:
                        line, _, remaining = buffered.partition(b'\n')
                        buffered = bytearray(remaining)
                        message = json.loads(line)
                        budget.check()
                        require(not done and message.get('sequence') == sequence and message.get('nonce') == nonce
                                and message.get('pid') == process.pid and message.get('ppid') == os.getpid()
                                and message.get('pgid') == process.pid and message.get('started') == budget.started
                                and message.get('limits_sha256') == sha_json(limits)
                                and message.get('context_sha256') == context_sha256, 'Invalid worker authority message')
                        if process.poll() is None:
                            require(os.getpgid(process.pid) == process.pid, 'Worker process group ownership changed')
                        kind = message.get('type')
                        require((sequence == 0 and kind == 'hello') or (authenticated and kind in {'check', 'stage', 'done'}),
                                'Worker handshake/order mismatch')
                        if kind == 'hello':
                            authenticated = True
                        elif kind == 'stage':
                            budget.set_stage(message['stage'])
                            transitions += 1
                        elif kind == 'done':
                            require(budget.stage == 'preflight_and_finalize', 'Worker completion must include finalization stage')
                            done = True
                        response = {'type': 'ack', 'sequence': sequence, 'nonce': nonce,
                                    'parent_pid': os.getpid(), 'worker_pid': process.pid,
                                    'started': budget.started, 'limits_sha256': sha_json(limits), 'context_sha256': context_sha256}
                        # Small bounded acknowledgements cannot wait on the worker.
                        parent_channel.sendall(json.dumps(response).encode() + b'\n')
                        sequence += 1
            code = process.poll()
            if code is not None:
                require(code == 0, f'Diagnostic worker failed with code {code}; no retry')
                require(authenticated and done, 'Worker exited without authenticated completion')
                budget.check()
                _terminate_owned_group(process, kill_grace)
                return {'exit_code': code, 'pid': process.pid, 'authenticated': True,
                        'execution_context_sha256': context_sha256, 'stage_transitions': transitions, 'budget': budget.snapshot()}
            require(channel_open or done, 'Worker authority channel closed before completion')
    except BaseException as error:
        if process is not None:
            _terminate_owned_group(process, kill_grace)
        error.supervision_budget = budget.snapshot()
        error.supervision_pid = process.pid if process is not None else None
        raise
    finally:
        parent_channel.close()
        child_channel.close()


class EvidenceWriter:
    """Exclusive JSON writes with one cumulative byte cap and terminal reserve."""
    def __init__(self, root, maximum, *, reserve=65536):
        self.root = Path(root)
        self.root.mkdir(exist_ok=False)
        self.maximum, self.reserve, self.used = maximum, min(reserve, maximum // 8), 0

    def _path(self, name):
        require(Path(name).name == name and not name.startswith('.'), 'Unsafe evidence filename')
        return self.root / name

    def write(self, name, value, *, terminal=False):
        data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        bound = self.maximum if terminal else self.maximum - self.reserve
        require(self.used + len(data) <= bound, 'Output byte budget exceeded')
        with self._path(name).open('xb') as handle:
            handle.write(data)
        self.used += len(data)

    def append(self, name, value):
        data = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
        require(self.used + len(data) <= self.maximum - self.reserve, 'Output byte budget exceeded')
        path = self._path(name)
        require(not path.is_symlink(), 'Evidence stream must not be a symlink')
        with path.open('ab') as handle:
            handle.write(data)
        self.used += len(data)


def verify_payload(payload, state, arm_contract, hash_state):
    require(state.get('scope') in {'primary', 'last120'}, 'Unknown checkpoint scope')
    require(payload.get('schema_version') == 'quantity_comparison_engine_v1', 'Wrong checkpoint schema')
    require(payload.get('contract_sha256') == sha_json(arm_contract), 'Checkpoint arm contract mismatch')
    require(payload.get('global_step') == state['global_step'], 'Checkpoint step mismatch')
    if state['scope'] == 'primary':
        require(payload.get('condition') == arm_contract['condition'] and payload.get('selector') == 'raw_quantity_rmse'
                and payload.get('applicable') is True and payload.get('best_epoch') == state['epoch']
                and payload.get('best_value') == state['expected_validation']['metrics']['raw_quantity_rmse'], 'Wrong primary selector/condition')
        expected_hash = payload.get('state_sha256')
    else:
        require(payload.get('contract') == arm_contract and payload.get('epoch') == 120, 'Wrong last checkpoint identity')
        history = payload.get('history', [])
        require(len(history) == 120 and [r['epoch'] for r in history] == list(range(1, 121)), 'Incomplete checkpoint history')
        for key, value in state['expected_history_metrics'].items():
            require(history[-1].get(key) == value, f'Last history mismatch: {key}')
        expected_hash = payload.get('model_state_sha256')
    require(hash_state(payload['model_state_dict']) == expected_hash == state['tensor_state_sha256'], 'Tensor state mismatch')
    return payload['model_state_dict']


def check_replay(actual, expected, policy):
    for key, value in expected.items():
        require(key in actual and math.isfinite(actual[key])
                and math.isclose(actual[key], value, rel_tol=policy['replay_relative_tolerance'], abs_tol=policy['replay_absolute_tolerance']), f'Checkpoint replay mismatch: {key}')
    return {'passed': True, 'metric_count': len(expected)}
