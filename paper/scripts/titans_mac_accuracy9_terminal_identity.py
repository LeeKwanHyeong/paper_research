"""Bound the terminal /proc-to-waitpid race for an already owned MAC9 child.

This operational adapter does not launch, retry, terminate, or declare success
for a scientific worker. The queue remains responsible for checking the actual
return code and receipt. Existing sealed runtime files are left unchanged.
"""
from __future__ import annotations

import math
import subprocess


def make_terminal_identity(*, proc_identity, initial_identity, exit_wait_seconds=0.5):
    """Return the queue's identity callback with a single bounded exit wait.

    ``initial_identity`` must be the existing bounded initial-exec callback.
    ``process`` is the queue's own Popen instance, never a newly opened PID.
    A missing /proc identity alone is never accepted as proof of OS exit.
    """
    if (isinstance(exit_wait_seconds, bool)
            or not isinstance(exit_wait_seconds, (int, float))
            or not math.isfinite(exit_wait_seconds)
            or not 0.25 <= exit_wait_seconds <= 1.0):
        raise ValueError("Terminal exit wait must be between 0.25 and 1.0 seconds")

    def identity(process, command, source, start_ticks=None):
        if start_ticks is None:
            return initial_identity(process, command, source, start_ticks)
        live = proc_identity(process.pid)
        if live is not None:
            if not (live["pgid"] == live["pid"] == process.pid
                    and live["start_ticks"] == start_ticks
                    and live["argv"] == command and live["cwd"] == str(source)):
                raise ValueError("Owned child identity changed")
            return live

        code = process.poll()
        if code is None:
            try:
                code = process.wait(timeout=exit_wait_seconds)
            except subprocess.TimeoutExpired as exc:
                raise ValueError(
                    "Child identity disappeared without observable OS exit "
                    "after bounded terminal wait"
                ) from exc
        if type(code) is not int or process.returncode != code:
            raise ValueError("Child OS exit was not recorded by its owning Popen")
        # In particular, a nonzero code is preserved for run_stage to reject.
        return None

    return identity
