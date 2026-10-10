"""Operational GPU ownership checks for a separately approved recovery bundle.

No training, process termination, network calls or mutation happen on import.
A disappearing process is accepted only after a fresh GPU inventory excludes it.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time


class GPUInspectionError(ValueError):
    """Fail closed while retaining the observations needed to diagnose the failure."""

    def __init__(self, message, observations):
        self.observations = observations
        super().__init__(message + ': ' + json.dumps(observations, sort_keys=True))


class ProcessTransition(Exception):
    pass


def read_process(pid, proc_root=Path('/proc')):
    """Read stat around cmdline, rejecting exit and PID reuse transitions."""
    folder = Path(proc_root) / str(pid)

    def stat():
        fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
        return {'state': fields[0], 'ppid': int(fields[1]),
                'pgrp': int(fields[2]), 'start_ticks': int(fields[19])}

    before = stat()
    argv = (folder / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    after = stat()
    if before['start_ticks'] != after['start_ticks']:
        raise GPUInspectionError('GPU PID reused during inspection', {'pid': pid, 'before': before, 'after': after})
    if (before['ppid'], before['pgrp']) != (after['ppid'], after['pgrp']):
        raise ProcessTransition('Process ancestry changed during inspection')
    if before['state'] in ('Z', 'X', 'x') or after['state'] in ('Z', 'X', 'x') or not any(argv):
        raise ProcessTransition('Process exited or command line is unavailable')
    return {**after, 'argv': argv}


def collect_gpu_processes(inventory, reader=read_process, *, attempts=3, pause=time.sleep):
    """Bounded observation retry, never a training retry or ownership exemption.

All PIDs are checked again on each attempt. A persistent zombie/empty cmdline
remains an error; a real foreign live process is passed to the strict verifier.
    """
    if attempts < 1:
        raise ValueError('At least one inspection attempt is required')
    observations = []
    identities = {}
    for attempt in range(attempts):
        pids = set(inventory())
        processes, transitions = {}, {}
        for pid in sorted(pids):
            try:
                proc = reader(pid)
                identity = proc['start_ticks']
                if pid in identities and identities[pid] != identity:
                    raise GPUInspectionError('GPU PID reused between inspections', observations + [{'pid': pid}])
                identities[pid] = identity
                processes[pid] = proc
            except (FileNotFoundError, ProcessTransition) as exc:
                transitions[pid] = str(exc)
        observations.append({'gpu_pids': sorted(pids), 'processes': processes, 'transitions': transitions})
        if not transitions:
            return processes, observations
        # A fresh inventory is necessary even on the last attempt. A vanished
        # worker needs no argv inspection, while a new GPU PID must be inspected.
        fresh = set(inventory())
        observations.append({'fresh_gpu_pids': sorted(fresh)})
        if fresh <= set(processes):
            return {pid: processes[pid] for pid in fresh}, observations
        if attempt + 1 < attempts:
            pause(.1)
    raise GPUInspectionError('GPU process identity remained unavailable', observations)


def check_active_gpu(campaign, contract, host, root, active):
    """Drop-in check for a new launcher; existing sealed campaign stays untouched."""
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.total,memory.used',
                          '--format=csv,noheader,nounits'], check=True,
                         capture_output=True, text=True, timeout=15)
    rows = [row.strip().split(',') for row in gpu.stdout.splitlines() if row.strip()]
    matches = [r for r in rows if r[0].strip() == contract['hosts'][host]['gpu_uuid']]
    if len(matches) != 1:
        raise GPUInspectionError('GPU UUID missing or ambiguous', {'rows': rows})
    total, used = (float(matches[0][i].strip()) * 1024 ** 2 for i in (1, 2))
    processes, observations = collect_gpu_processes(lambda: campaign.common.gpu_pids(contract['hosts'][host]))
    try:
        campaign.verify_gpu_membership(set(processes), processes, active, root=root,
                                       used_bytes=used, total_bytes=total)
        for pid, proc in processes.items():
            owner = next(item for item in active.values() if item['child'].pid == proc['ppid'])
            # The existing verifier checks group, job and script; also bind the
            # full argv, including contract, host, mode and deadline.
            if proc['argv'] != owner['command'][4:]:
                raise ValueError('GPU worker exact command mismatch')
    except (ValueError, StopIteration) as exc:
        raise GPUInspectionError(str(exc), observations) from exc
    return True
