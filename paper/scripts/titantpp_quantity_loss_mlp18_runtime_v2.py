"""v2 operational repair: admit only the verified qualification process's own CUDA context.

The v1 runtime and scientific worker remain byte-identical. A private namespace
keeps the repair isolated from imported v1 globals. No loss/model/data changes.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import traceback

_V1_PATH=Path(__file__).with_name("titantpp_quantity_loss_mlp18_runtime.py")
_V1_SHA256="03a4d8c07e7ee1af2164e2a7b1cfe088a86bc189dc1e35ba014a4e6b2f28e9ec"
from paper.scripts.titantpp_cross_fusion_precheck_runtime import require as _require, sha_file as _sha_file
_require(_sha_file(_V1_PATH)==_V1_SHA256,"Sealed v1 runtime changed")
_spec=importlib.util.spec_from_file_location("paper.scripts._quantity_loss_runtime_v2_private",_V1_PATH)
_impl=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_impl)
_impl.MODULE="paper.scripts.titantpp_quantity_loss_mlp18_runtime_v2"
_original_three_concurrent=_impl.qualify_three_concurrent


def qualification_identity(contract,host):
    """Bind the one permitted extra PID to this exact qualification invocation."""
    h=contract["hosts"][host]
    expected=[h["python"],"-u","-m",_impl.MODULE,"--contract",str(Path(h["root"])/"execution_contract.json"),"--host",host,"--mode","qualify"]
    live=_impl.proc_identity(os.getpid())
    _impl.require(live is not None and live["pid"]==os.getpid() and live["pgid"]==os.getpid()
        and live["cwd"]==str(Path(h["source_root"]).resolve()) and live["argv"]==expected,
        "Current qualification PID/cwd/argv/session identity mismatch")
    attempt=_impl.read(Path(h["root"])/"qualification_attempt.json")
    _impl.require(attempt["host"]==host and attempt["contract_sha256"]==_impl.sha_json(contract),
        "Current qualification attempt does not match its contract")
    return live


def qualify_three_concurrent(contract,host):
    """Reuse v1's six native checks; scope the extra allowlist to this invocation."""
    _impl.require(_sha_file(_V1_PATH)==_V1_SHA256,"Sealed v1 runtime changed")
    identity=qualification_identity(contract,host)
    root=Path(contract["hosts"][host]["root"])
    _impl.write(root/"qualification_owner_identity.json",{"host":host,"registered_at":_impl.now(),
        "contract_sha256":_impl.sha_json(contract),"qualification_owner_identity":identity},exclusive=True)
    original_gpu_state=_impl.gpu_state
    def qualification_gpu_state(c,h,active_pids=()):
        _impl.require(c is contract and h==host,"Qualification GPU allowance escaped its contract/host")
        current=qualification_identity(contract,host)
        _impl.require(all(current[k]==identity[k] for k in ("pid","start_ticks","pgid","argv","cwd")),
            "Qualification PID changed during admission")
        original_snapshot=_impl.gpu_snapshot
        def observed_snapshot(uuid):
            snapshot=original_snapshot(uuid)
            _impl.write(root/"qualification_gpu_state_observation.json",{"host":host,"observed_at":_impl.now(),
                "contract_sha256":_impl.sha_json(contract),"qualification_owner_identity":current,"gpu":snapshot,
                "qualification_owner_present_in_gpu_observation":identity["pid"] in snapshot["compute_pids"]},exclusive=True)
            return snapshot  # The complete original list is returned unchanged to the original validator.
        _impl.gpu_snapshot=observed_snapshot
        try:
            return original_gpu_state(c,h,(*active_pids,identity["pid"]))
        finally:
            _impl.gpu_snapshot=original_snapshot
    _impl.gpu_state=qualification_gpu_state
    try:
        result=_original_three_concurrent(contract,host)
        result["qualification_process_identity"]=identity
        result["qualification_owner_identity"]=identity
        result["qualification_gpu_observation"]=result["gpu_after"]
        result["qualification_owner_present_in_gpu_observation"]=identity["pid"] in result["gpu_after"]["compute_pids"]
        result["qualification_self_context_allowed"]=True
        result["qualification_self_context_scope"]="only the verified current qualification child; dispatcher admission retains the original GPU allowlist"
        return result
    finally:
        _impl.gpu_state=original_gpu_state
        _impl.require(_sha_file(_V1_PATH)==_V1_SHA256,"Sealed v1 runtime changed during qualification")


_impl.qualify_three_concurrent=qualify_three_concurrent
# Export the original stdlib API while all inherited function globals remain private.
for _name,_value in tuple(_impl.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name]=_value

if __name__=="__main__":
    try:
        raise SystemExit(_impl.main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
