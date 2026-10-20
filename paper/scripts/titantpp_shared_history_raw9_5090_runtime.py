"""Fresh5090 admission for the unchanged original-data joint shared9 science.

The earlier 5080 wait, Cap18 experiment and all sealed originals are preserved.
This isolated adapter has no dependency on Cap18 completion and no resume path.
"""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import traceback

from paper.scripts.titantpp_cross_fusion_precheck_runtime import require, sha_file

_PATH = Path(__file__).with_name("titantpp_shared_history_raw9_runtime.py")
_SHA = "1e06302ee81eb2bbeb37bf1d4d55d6d046e01cbdad6a2a37bdcadf173ed55515"
require(sha_file(_PATH) == _SHA, "Frozen raw9 Runtime changed")
_spec = importlib.util.spec_from_file_location(__name__ + "_private_raw", _PATH)
_raw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_raw)
_impl = _raw._impl
_impl.SCHEMA = "titantpp_shared_history_raw9_5090_v1"
_impl.MODULE = "paper.scripts.titantpp_shared_history_raw9_5090_runtime"
_impl.WORKER_MODULE = "paper.scripts.run_titantpp_shared_history_raw9_5090"
_raw._HOST_DEADLINE = 1792888885.815062
_GPU_UUID = "GPU-c9adc246-ef66-7906-bc07-6e152fa9952f"


def _replace_exact(source, before, after, count=1):
    require(source.count(before) == count, "Unexpected frozen raw9 adaptation: " + before)
    return source.replace(before, after)


_validator = inspect.getsource(_raw.validate_contract)
_validator = _replace_exact(_validator, 'host == "5080" and set(contract["hosts"]) == {"5080"}',
    'host == "5090" and set(contract["hosts"]) == {"5090"}')
_validator = _replace_exact(_validator, '"Only approved 5080 is assigned"', '"Only approved 5090 is assigned"')
_validator = _replace_exact(_validator,
    'require(len(stopped) == len({p["pid"] for p in stopped}) == 4 and all(p["state"] in ("T", "t") for p in stopped), "Stopped process4 identities missing")',
    'require(stopped == [], "5090 cannot inherit stopped 5080 process identities")')
exec(compile(_validator.replace("def validate_contract(", "def _validate_5090(", 1), str(_PATH), "exec"), _raw.__dict__)
_receipt = inspect.getsource(_raw.verify_receipt)
_receipt = _replace_exact(_receipt, '"titantpp_shared_history_raw9_stage_v1"', '"titantpp_shared_history_raw9_5090_stage_v1"')
exec(compile(_receipt.replace("def verify_receipt(", "def _verify_5090_receipt(", 1), str(_PATH), "exec"), _raw.__dict__)


def validate_contract(contract, host):
    jobs = _raw._validate_5090(contract, host)
    h = contract["hosts"][host]
    require(h["gpu_uuid"] == _GPU_UUID and h["runtime_expected"] == {
        "python": "3.12.13", "torch": "2.11.0+cu130", "cuda": "13.0", "cudnn": 92000,
        "threads": 4, "interop_threads": 1}, "5090 native GPU/Runtime contract differs")
    require(contract.get("maximum_parallel_workers") == 3, "5090 parallel limit differs")
    return jobs


def paused_processes(contract, host):
    require(host == "5090" and contract["hosts"][host]["paused_processes"] == [], "No inherited paused process allowance on5090")
    return []


def gpu_state(contract, host, active_pids=()):
    paused_processes(contract, host)
    gpu = _impl.gpu_snapshot(contract["hosts"][host]["gpu_uuid"])
    require(set(gpu["compute_pids"]) <= set(active_pids), "Unidentified5090 GPU process; no new work")
    return gpu, []


def predecessor_gate(contract, host, *, require_complete=False):
    """The latest approval replaces the earlier Cap5080 wait with fresh idle5090."""
    validate_contract(contract, host)
    gpu, _ = gpu_state(contract, host)
    require(not gpu["compute_pids"], "5090 GPU is not idle")
    return {"status": "passed", "host": host, "observed_at": _impl.now(), "gpu": gpu,
        "fresh_idle_verified": True, "cap5080_completion_required": False,
        "old_campaigns_changed_or_resumed": False, "paused_processes": []}


verify_receipt = _raw._verify_5090_receipt
_raw._adapter.validate_contract = validate_contract
_impl.validate_contract = validate_contract
_impl.verify_receipt = verify_receipt
_impl.paused_processes = paused_processes
_impl.gpu_state = gpu_state
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
