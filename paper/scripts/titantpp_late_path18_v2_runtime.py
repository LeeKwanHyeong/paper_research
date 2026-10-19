"""Late-path18 v2: logged inputs after the completed quantity-loss comparison.

Only the completed predecessor changes. The two scientific candidates, worker,
Validation-only matrix, bounded queue and original deadlines remain unchanged.
The raw-time campaign is neither inspected as a prerequisite nor started here.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import traceback

from paper.scripts.titantpp_cross_fusion_precheck_runtime import require as _require, sha_file as _sha_file

_V1_PATH = Path(__file__).with_name("titantpp_late_path18_runtime.py")
_V1_SHA256 = "66d7d739b112347164c7fbfb867d4e024f7d739c86d35c9e0dd096fb0f43d0b2"
_require(_sha_file(_V1_PATH) == _V1_SHA256, "Preserved late-path v1 runtime changed")
_spec = importlib.util.spec_from_file_location("paper.scripts._late_path18_v2_outer_private", _V1_PATH)
_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_adapter)
_impl = _adapter._impl
_impl.MODULE = "paper.scripts.titantpp_late_path18_v2_runtime"
# The same scientific worker/contract schema retains its exact candidate IDs.
_impl.WORKER_MODULE = "paper.scripts.run_titantpp_late_path18"
_PARENT_RAW_SHA256 = "bca53c976981c168bd0c7850c89a178e154e8e851498d1a2646756febd36441b"
_PARENT_CANONICAL_SHA256 = "4b574670362e6469826fcbb4c1e355508594b76a00a125d4e418ca03a0ca71a2"
_original_validate_contract = _adapter.validate_contract


def validate_contract(contract, host):
    """One shared loss is admitted, with both input channels fixed to log1p."""
    r = _impl
    common, architecture = contract.get("common_condition", {}), contract.get("architecture", {})
    r.require(common.get("time_input") == "log1p"
        and architecture.get("time_input_transform") == "log1p_nonnegative_float32"
        and architecture.get("quantity_input_transform") == "log1p_nonnegative_float32",
        "Late-path v2 fixes observed time and quantity inputs to log1p")
    r.require("parent_quantity_loss27" in contract and "parent_raw_time27" not in contract,
        "Late-path v2 must continue from the quantity-loss comparison only")
    binding = contract["parent_quantity_loss27"]
    r.require(binding.get("contract_raw_sha256") == _PARENT_RAW_SHA256
        and binding.get("contract_canonical_sha256") == _PARENT_CANONICAL_SHA256,
        "Approved quantity-loss v2 predecessor changed")
    # Reuse the preserved matrix validator through an ephemeral lineage-name
    # view. No persisted contract, checkpoint, job or scientific identity changes.
    view = {**contract, "parent_raw_time27": binding}
    return _original_validate_contract(view, host)


def predecessor_gate(contract, host, *, require_complete=False):
    """Read verified loss18 terminal originals; never modify or signal them."""
    from paper.scripts import titantpp_quantity_loss_mlp18_runtime_v2 as parent

    r = _impl
    validate_contract(contract, host)
    h = contract["hosts"][host]["completed_predecessor"]
    root = Path(h["root"])
    cp = root / "execution_contract.json"
    r.require(r.sha_file(cp) == h["contract_raw_sha256"] == _PARENT_RAW_SHA256,
        "Predecessor contract bytes changed")
    c = r.read(cp)
    r.require(parent.sha_json(c) == h["contract_canonical_sha256"] == _PARENT_CANONICAL_SHA256
        and c["source"]["files_sha256"] == h["source_files_sha256"],
        "Predecessor contract/source binding changed")
    r.require(c["hosts"][host]["absolute_host_deadline_unix"]
        == contract["hosts"][host]["absolute_host_deadline_unix"], "Predecessor deadline changed")
    jobs = parent.validate_contract(c, host)
    ids = {job["id"] for job in jobs}
    r.paused_processes(contract, host)
    r.require(not (root / "dispatch_failure.json").exists(),
        "Failed predecessor; no automatic continuation")
    fp, xp = root / "final_receipt.json", root / "dispatcher_exit.json"
    r.require(r.sha_file(fp) == h["final_receipt_sha256"]
        and r.sha_file(xp) == h["dispatcher_exit_sha256"], "Predecessor terminal SHA changed")
    final, exit_record = r.read(fp), r.read(xp)
    r.require(final["status"] == "complete" and final["host"] == host
        and final["contract_sha256"] == parent.sha_json(c)
        and final["completed_fit_count"] == final["validation_replay_count"] == len(jobs)
        and final["test_count"] == 0 and final["held_out_test_evaluated"] is False
        and final["owned_worker_pids_absent"] is True, "Predecessor incomplete")
    r.require(exit_record["actual_os_returncode"] == 0
        and exit_record["pid"] == h["dispatcher_identity"]["pid"],
        "Missing predecessor actual dispatcher exit0")
    pids = [exit_record["pid"]]
    r.require(set(final["stages"]) == set(parent.STAGES), "Predecessor stages changed")
    for stage in parent.STAGES:
        stage_result = final["stages"][stage]
        r.require(stage_result["status"] == "complete"
            and set(stage_result["exits"]) == set(stage_result["receipts"]) == ids,
            "Predecessor stage incomplete")
        for job in jobs:
            child, receipt_binding = stage_result["exits"][job["id"]], stage_result["receipts"][job["id"]]
            r.require(child["actual_os_returncode"] == 0 and child["timeout"] is False,
                "Predecessor scientific child failed")
            pids.append(child["pid"])
            receipt_path = r.local_path(root, receipt_binding["path"])
            r.require(r.sha_file(receipt_path) == receipt_binding["sha256"],
                "Predecessor scientific receipt SHA changed")
            parent.verify_receipt(c, host, job, stage)
    manifest = final["artifact_manifest"]
    mp = r.local_path(root, manifest["path"])
    r.require(r.sha_file(mp) == manifest["sha256"], "Predecessor manifest changed")
    inventory = r.read(mp)["files"]
    r.require(inventory and len(inventory) == manifest["file_count"],
        "Predecessor inventory incomplete")
    for name, digest in inventory.items():
        r.require(r.sha_file(r.local_path(root, name)) == digest, "Predecessor artifact changed")
    r.require(all(r.proc_identity(pid) is None for pid in pids), "Predecessor process remains")
    gpu, paused = r.gpu_state(contract, host)
    r.require(not set(pids).intersection(gpu["compute_pids"]), "Predecessor GPU owner remains")
    return {"status": "complete", "predecessor_campaign": "titantpp_quantity_loss_mlp18_20261019_v2",
        "parent_root": str(root), "parent_final_receipt_sha256": r.sha_file(fp),
        "parent_dispatcher_exit_sha256": r.sha_file(xp), "parent_artifact_manifest_sha256": r.sha_file(mp),
        "parent_owned_pids_absent": True, "parent_gpu_pids_absent": True,
        "paused_processes": paused, "gpu": gpu}


_impl.validate_contract = validate_contract
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
