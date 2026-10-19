"""Preserved shared-history queue with a user-owned predecessor process scan.

The frozen v1 attempt and runtime remain unchanged. Only the first process
enumeration inside its predecessor gate is scoped to the current Unix user.
GPU admission and stopped-campaign descendant checks retain their full scope.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import traceback

from paper.scripts.titantpp_cross_fusion_precheck_runtime import require, sha_file

_V1_PATH = Path(__file__).with_name("titantpp_shared_history_cap_execution_runtime.py")
_V1_SHA = "0a3cd1f203d4f41b9b483dbfc2b1b94623e69543fb21ca26fd7daceac98918ae"
require(sha_file(_V1_PATH) == _V1_SHA, "Frozen shared-history v1 runtime changed")
_spec = importlib.util.spec_from_file_location("paper.scripts._shared_history_cap_v2_private", _V1_PATH)
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)
_impl = _v1._impl
_original_gate = _impl.predecessor_gate
_impl.MODULE = "paper.scripts.titantpp_shared_history_cap_execution_v2_runtime"


def owned_process_parents(proc_root=Path("/proc")):
    """Read only current-user processes; unreadable owned processes fail closed."""
    uid = os.getuid()
    parents = {}
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            fields = (entry / "stat").read_text().rpartition(")")[2].split()
            parents[int(entry.name)] = int(fields[1])
        except (FileNotFoundError, ProcessLookupError):
            # A process that exited during enumeration has no live identity.
            continue
    return parents


def predecessor_gate(contract, host, *, require_complete=False):
    r = _impl
    original_parents = r.process_parents

    def predecessor_parents_once():
        try:
            return owned_process_parents()
        finally:
            # Restore before the frozen gate invokes gpu_state/paused_processes.
            r.process_parents = original_parents

    r.process_parents = predecessor_parents_once
    try:
        return _original_gate(contract, host, require_complete=require_complete)
    finally:
        r.process_parents = original_parents


_impl.predecessor_gate = predecessor_gate
for _name, _value in tuple(_impl.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value

if __name__ == "__main__":
    try:
        raise SystemExit(_impl.main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
