"""Isolated v2 campaign for the operational qualification-context repair.

All v1 source and failed attempts remain unchanged. The scientific worker,
18-fit matrix, selected-Validation policy and stored alpha0 baseline are reused.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from paper.scripts import titantpp_quantity_loss_mlp18_runtime_v2 as _runtime

_V1_PATH=Path(__file__).with_name("manage_titantpp_quantity_loss_mlp18.py")
_V1_SHA256="331aa17a98532217bb4cdda8865cf09530cc54b5c31f4968456655e93be27e8d"
_runtime.require(_runtime.sha_file(_V1_PATH)==_V1_SHA256,"Sealed v1 manager changed")
_spec=importlib.util.spec_from_file_location("paper.scripts._quantity_loss_manager_v2_private",_V1_PATH)
_impl=importlib.util.module_from_spec(_spec)
# The sole textual adaptation gives the new isolated launch its own tmux name.
_source=_V1_PATH.read_text()
_needle='tmux=f"quantity_loss18_{host}_20261019_v1"'
_runtime.require(_source.count(_needle)==1,"Unexpected v1 tmux adaptation point")
exec(compile(_source.replace(_needle,'tmux=f"quantity_loss18_{host}_20261019_v2"'),str(_V1_PATH),"exec"),_impl.__dict__)
_impl.runtime=_runtime
_impl.NAME="titantpp_quantity_loss_mlp18_20261019_v2"
_impl.BUNDLE=_impl.ROOT/"search_artifacts"/_impl.NAME
_impl.REPORT=_impl.ROOT/"reports"/_impl.NAME
_impl.MODULE=_runtime.MODULE
_impl.REQUIRED_NEW_SOURCE=("paper/scripts/manage_titantpp_quantity_loss_mlp18_v2.py",
    "paper/scripts/titantpp_quantity_loss_mlp18_runtime_v2.py")
_V1_BUNDLE=_impl.ROOT/"search_artifacts/titantpp_quantity_loss_mlp18_20261019_v1"
_V1_REPORT=_impl.ROOT/"reports/titantpp_quantity_loss_mlp18_20261019_v1"


def source_snapshot():
    """Preserve the prepared v1 source950 exactly and add only two v2 operators."""
    path=_V1_BUNDLE/"execution_contract.json";prepared=_impl.read(_V1_REPORT/"PREPARED.json")
    _impl.require(_impl.sha(path)==prepared["contract_sha256"],"Prepared v1 contract changed")
    parent=_impl.read(path)
    _impl.require(_impl.canonical(parent)==prepared["canonical_contract_sha256"],"Prepared v1 canonical contract changed")
    original=parent["source"]["files"]
    _impl.require(len(original)==950 and _impl.canonical(original)==parent["source"]["files_sha256"],"Expected sealed v1 source950")
    for name,digest in original.items():
        _impl.require(_impl.sha(_impl.ROOT/name)==digest,"Sealed v1 source changed: "+name)
    required=set(_impl.REQUIRED_NEW_SOURCE)
    tracked=set(_impl.subprocess.check_output(["git","ls-files","--",*required],cwd=_impl.ROOT,text=True).splitlines())
    _impl.require(tracked==required,"Commit new v2 operators before prepare")
    _impl.require(not _impl.subprocess.check_output(["git","diff","--name-only","HEAD","--",*required],cwd=_impl.ROOT,text=True).strip(),"Commit new v2 operator changes")
    files={**original,**{name:_impl.sha(_impl.ROOT/name) for name in required}}
    revision=_impl.subprocess.check_output(["git","rev-parse","HEAD"],cwd=_impl.ROOT,text=True).strip()
    return {"base_git_revision":revision,"files":files,"files_sha256":_impl.canonical(files),
        "model_science_source_revision":parent["source"]["model_science_source_revision"],
        "model_science_source_files_sha256":parent["source"]["model_science_source_files_sha256"],
        "parent_source_files_sha256":parent["source"]["files_sha256"],"parent_source_file_count":950,
        "operational_repair_parent":{"contract_path":str(path),"contract_raw_sha256":_impl.sha(path),
            "canonical_contract_sha256":_impl.canonical(parent),"prepared_receipt_path":str(_V1_REPORT/"PREPARED.json"),
            "prepared_receipt_sha256":_impl.sha(_V1_REPORT/"PREPARED.json"),
            "repair":"allow verified qualification self CUDA context only; model/loss/inputs unchanged",
            "v1_attempts_preserved":True,"v1_root_reused":False}}


_impl.source_snapshot=source_snapshot
for _name,_value in tuple(_impl.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name]=_value

if __name__=="__main__":
    raise SystemExit(_impl.main())
