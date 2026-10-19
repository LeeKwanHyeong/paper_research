"""Frozen queue with predecessor-scoped command-line process discovery.

Only processes referring to the completed predecessor root enter its detailed
identity scan. The old dispatcher PID is still checked directly; GPU admission
and stopped-campaign descendant checks retain their original full scope.
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
_spec = importlib.util.spec_from_file_location("paper.scripts._shared_history_cap_v3_private", _V1_PATH)
_v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1)
_impl = _v1._impl
_original_gate = _impl.predecessor_gate
_impl.MODULE = "paper.scripts.titantpp_shared_history_cap_execution_v3_runtime"


def predecessor_process_candidates(predecessor_root, proc_root=Path("/proc")):
    """Read command lines before cwd; inspect predecessor root byte matches.

The frozen gate uses only mapping keys. Parent IDs are deliberately not read
here, avoiding unrelated process stat/cwd accesses. Permission errors reading
cmdline are not hidden because relevance could not then be established.
"""
    root = Path(predecessor_root)
    require(root.is_absolute() and str(root) != "/", "Invalid predecessor process scope")
    # Match the successful native read-only preflight exactly. This deliberately
    # includes similar path names containing the full marker; the frozen gate
    # subsequently checks actual argv/cwd ownership. Never hide such candidates.
    marker = os.fsencode(root)
    candidates = {}
    for entry in Path(proc_root).iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if marker in raw:
            candidates[int(entry.name)] = None
    return candidates


def predecessor_gate(contract, host, *, require_complete=False):
    r = _impl
    original_parents = r.process_parents

    def predecessor_candidates_once():
        try:
            return predecessor_process_candidates(contract["hosts"][host]["completed_predecessor"]["root"])
        finally:
            # Restore before frozen GPU and paused-descendant checks execute.
            r.process_parents = original_parents

    r.process_parents = predecessor_candidates_once
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
