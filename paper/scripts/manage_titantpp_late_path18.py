"""Prepare late-path18 only after both existing comparisons are verified complete."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import time

from paper.scripts import titantpp_late_path18_runtime as _runtime

_BASE = Path(__file__).with_name("manage_titantpp_quantity_loss_mlp18.py")
_runtime.require(_runtime.sha_file(_BASE) == "331aa17a98532217bb4cdda8865cf09530cc54b5c31f4968456655e93be27e8d",
                 "Preserved manager source changed")
_spec = importlib.util.spec_from_file_location("paper.scripts._late_path18_manager_private", _BASE)
_impl = importlib.util.module_from_spec(_spec)
_text = _BASE.read_text().replace("titantpp_quantity_loss_mlp18", "titantpp_late_path18")
_text = _text.replace("quantity_loss18_", "late_path18_")
exec(compile(_text, str(_BASE), "exec"), _impl.__dict__)
_impl.runtime = _runtime
_impl.NAME = "titantpp_mlp16_late_path18_20261019_v1"
_impl.BUNDLE = _impl.ROOT / "search_artifacts" / _impl.NAME
_impl.REPORT = _impl.ROOT / "reports" / _impl.NAME
_impl.MODULE = "paper.scripts.titantpp_late_path18_runtime"
_impl.REQUIRED_NEW_SOURCE = (
    "models/TPPs/CountAwareTitanLateObservationPath.py",
    "paper/scripts/run_titantpp_late_path18.py",
    "paper/scripts/titantpp_late_path18_runtime.py",
    "paper/scripts/manage_titantpp_late_path18.py",
    "paper/scripts/select_compare_titantpp_late_path18.py",
)
PARENT_NAME = "titantpp_raw_time_mlp27_20261019_v1"
RAW_READY_SHA256 = "59ee0a1b1e2a2a261093a7766cf49a49861922c5814b99c5219da1eda0b03937"
MODEL_SHA256 = "a8759bd93c265c486289f14ebe34d8ae14d693b2a75e070d04c655406e599a0e"
TRANSFORMS = {"log1p": "log1p_nonnegative_float32", "raw": "raw_nonnegative_float32"}


def verify_parent_completion():
    """Local originals only; never rerun either prior finisher or collector."""
    from paper.scripts import manage_titantpp_raw_time_mlp27 as parent
    from paper.scripts.compare_titantpp_raw_time_mlp27 import aggregate
    from paper.scripts.select_compare_titantpp_late_path18 import select_common_condition

    ready = parent.REPORT / "LOCAL_READY.json"
    _impl.require(parent.sha(ready) == RAW_READY_SHA256, "Authorized raw-time preparation changed")
    c = parent.verify_local(archive=True)
    cp = parent.BUNDLE / "execution_contract.json"
    fp = parent.BUNDLE / "retrieval/final_receipt.json"
    done_path = parent.REPORT / "followup_v1/complete.json"
    final, done = parent.read(fp), parent.read(done_path)
    expected = {"new_fit_count": 27, "validation_replay_count": 27,
        "baseline_reference_count": 27, "comparison_model_count": 54, "seed_count": 3,
        "time_alpha": 0., "time_head_no_training_verified_count": 27,
        "actual_scientific_child_exit0_count": 54, "actual_dispatcher_exit0_count": 2}
    _impl.require(final["status"] == "complete" and final["held_out_test_evaluated"] is False
                  and final["baseline_reevaluated"] is False
                  and all(final.get(k) == v for k, v in expected.items()), "Raw-time parent incomplete")
    _impl.require(done["status"] == "complete" and done["final_receipt_path"] == str(fp)
                  and done["final_receipt_sha256"] == parent.sha(fp), "Raw-time completion binding missing")
    for group in ("source_bindings", "archive_bindings", "report_files"):
        _impl.require(bool(final[group]), "Empty raw-time completion bindings")
        for name, digest in final[group].items():
            _impl.require(parent.sha(name) == digest, "Parent completed evidence changed: " + name)
    refs = parent.baseline_validation_references()
    _impl.require(len(refs) == 27, "Expected all log-time27 originals")
    for ref in refs:
        ref.update(time_input="log1p", time_input_transform=TRANSFORMS["log1p"])
    for host in ("5080", "5090"):
        folder = parent.BUNDLE / "retrieval" / host
        originals = folder / "originals"
        collected = parent.read(folder / "receipt.json")
        _impl.require(collected["status"] == "complete" and collected["host"] == host
            and parent.sha(folder / "original.tar.gz") == collected["archive_sha256"]
            and parent.sha(folder / "manifest.json") == collected["manifest_sha256"], "Parent collection differs")
        inventory = parent.read(folder / "manifest.json")["files"]
        _impl.require(inventory and len(inventory) == collected["file_count"], "Empty parent originals")
        for name, digest in inventory.items():
            _impl.require(parent.sha(parent.runtime.local_path(originals, name)) == digest,
                          "Parent original changed: " + name)
        parent.verify_collected(c, host, originals)
        for job in (j for j in c["jobs"] if j["host"] == host):
            path = originals / "run" / job["id"] / "validation_receipt.json"
            receipt = parent.read(path)
            _impl.require(final["source_bindings"].get(str(path)) == parent.sha(path), "Endpoint binding missing")
            _impl.require(receipt["validation_replay_passed"] is True, "Parent Validation not replayed")
            selected = Path(receipt["selected_checkpoint"])
            remote_root = Path(c["hosts"][host]["root"])
            _impl.require(selected.is_absolute() and selected.is_relative_to(remote_root), "Parent selected path differs")
            local_cp = parent.runtime.local_path(originals, str(selected.relative_to(remote_root)))
            _impl.require(parent.sha(local_cp) == receipt["checkpoint_sha256"], "Parent selected binary changed")
            histories = list((path.parent / "runs").glob("*/*/seed_*/history.json"))
            _impl.require(len(histories) == 1, "Missing parent selected history")
            finite = [(r["val_qty_rmse"], r["epoch"]) for r in parent.read(histories[0])["history"]
                      if type(r.get("val_qty_rmse")) in (float, int) and math.isfinite(r["val_qty_rmse"])]
            _impl.require(finite and min(finite, key=lambda r: r[0])[1] == receipt["selected_epoch"],
                          "Parent first finite RMSE selection differs")
            refs.append({"dataset": job["dataset"], "host": host, "seed": job["seed"],
                "arm": job["arm"], "loss_mode": job["loss_mode"], "time_input": "raw",
                "time_input_transform": TRANSFORMS["raw"], "time_alpha": 0.,
                "quantity_only": True, "time_head_trained": False,
                "selected_epoch": receipt["selected_epoch"], "selected_checkpoint": str(local_cp),
                "checkpoint_sha256": receipt["checkpoint_sha256"], "state_sha256": receipt["state_sha256"],
                "target_identity": receipt["target_identity"], "metrics": receipt["metrics"],
                "time_head_no_training_verification": receipt["time_head_no_training_verification"],
                "path": str(path), "sha256": parent.sha(path), "completion_receipt_path": str(fp),
                "completion_receipt_sha256": parent.sha(fp), "reevaluated": False,
                "retrained": False, "held_out_test_evaluated": False})
    comparison_path = parent.REPORT / "comparison_v1/PAIRED_TIME_INPUT_VALIDATION.json"
    comparison = parent.read(comparison_path)
    _impl.require(final["report_files"].get(str(comparison_path)) == parent.sha(comparison_path)
                  and comparison["status"] == "complete" and comparison["held_out_test_evaluated"] is False,
                  "Parent comparison not bound")
    original = {(r["dataset"], r["seed"], r["loss_mode"], r["time_input"]): r
                for r in comparison["records"]}
    _impl.require(len(refs) == len(original) == 54, "Expected all54 verified endpoints")
    for ref in refs:
        old = original[ref["dataset"], ref["seed"], ref["loss_mode"], ref["time_input"]]
        _impl.require(all(old[k] == ref[k] for k in ("host", "time_alpha", "selected_epoch",
                      "time_head_trained", "target_identity", "metrics")), "Parent endpoint/aggregation mismatch")
    for key, result in aggregate(comparison["records"]).items():
        _impl.require(comparison[key] == result, "Parent pure aggregation differs: " + key)
    selection = select_common_condition(refs)
    binding = {"name": PARENT_NAME, "contract_path": str(cp), "contract_raw_sha256": parent.sha(cp),
        "contract_canonical_sha256": parent.canonical(c), "source_files_sha256": c["source"]["files_sha256"],
        "completion_path": str(done_path), "completion_sha256": parent.sha(done_path),
        "final_receipt_path": str(fp), "final_receipt_sha256": parent.sha(fp),
        "comparison_path": str(comparison_path), "comparison_sha256": parent.sha(comparison_path),
        "retrained": False, "reevaluated": False}
    return c, final, refs, selection, binding


def baseline_validation_references(*, verified=None):
    _, _, refs, selection, _ = verified if verified is not None else verify_parent_completion()
    condition = selection["selected_condition"]
    chosen = [r for r in refs if all(r[k] == condition[k] for k in ("time_input", "loss_mode", "time_alpha"))]
    _impl.require(len(chosen) == 9, "Expected one shared9-reference condition")
    return sorted(chosen, key=lambda r: (r["dataset"], r["seed"]))


def source_snapshot():
    from paper.scripts import manage_titantpp_raw_time_mlp27 as parent
    previous = parent.source_snapshot()
    _impl.require(len(previous["files"]) == 955, "Expected preserved source955")
    required = set(_impl.REQUIRED_NEW_SOURCE)
    tracked = set(_impl.subprocess.check_output(["git", "ls-files", "--", *required], cwd=_impl.ROOT, text=True).splitlines())
    _impl.require(tracked == required and not _impl.subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD", "--", *required], cwd=_impl.ROOT, text=True).strip(),
        "Commit the isolated late-path sources before prepare")
    files = {**previous["files"], **{name: _impl.sha(_impl.ROOT / name) for name in required}}
    _impl.require(_impl.sha(_impl.ROOT / "models/TPPs/CountAwareTitanLateObservationPath.py") == MODEL_SHA256,
                  "Prepared model changed")
    revision = _impl.subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_impl.ROOT, text=True).strip()
    return {"base_git_revision": revision, "files": files, "files_sha256": _impl.canonical(files),
        "model_science_source_revision": revision, "model_science_source_files_sha256": _impl.canonical(files),
        "parent_source_files_sha256": previous["files_sha256"], "parent_source_file_count": 955}


def build_jobs(condition):
    jobs = []
    for dataset in _runtime.DATASETS:
        for seed in (42, 52, 62):
            host = "5080" if dataset == "yellow_trip_hourly" or (dataset == "intermittent_frozen_5000" and seed == 42) else "5090"
            for arm in _runtime.ARMS:
                jobs.append({"id": f"{dataset}__{arm}__seed{seed}__loss{condition['loss_mode']}",
                    "dataset": dataset, "seed": seed, "host": host, "arm": arm, **condition})
    return jobs


def prepare():
    _impl.require(not _impl.BUNDLE.exists(), "Existing late-path bundle/attempt; no overwrite or retry")
    verified = verify_parent_completion()
    parent, _, _, selection, binding = verified
    _impl.require(all(time.time() < h["absolute_host_deadline_unix"] for h in parent["hosts"].values()),
                  "Inherited host deadline expired; no new preparation or training")
    condition = selection["selected_condition"]
    baseline = baseline_validation_references(verified=verified)
    source = source_snapshot()
    inputs, hosts = deepcopy(parent["input_files"]), deepcopy(parent["hosts"])
    for host, h in hosts.items():
        root = f"/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/{_impl.NAME}_{host}"
        original = _impl.ROOT / "search_artifacts" / PARENT_NAME / "retrieval" / host / "originals"
        fp, xp = original / "final_receipt.json", original / "dispatcher_exit.json"
        h.update(root=root, source_root=root + "/source", tmux=f"late_path18_{host}_20261019_v1",
            completed_predecessor={"root": parent["hosts"][host]["root"],
                **{k: binding[k] for k in ("contract_raw_sha256", "contract_canonical_sha256", "source_files_sha256")},
                "dispatcher_identity": {"pid": _impl.read(xp)["pid"]},
                "final_receipt_sha256": _impl.sha(fp), "dispatcher_exit_sha256": _impl.sha(xp)})
        h["environment"].update(SOURCE_REVISION=source["base_git_revision"], PYTHONPATH=h["source_root"],
                               MPLCONFIGDIR=root + "/cache/matplotlib", XDG_CACHE_HOME=root + "/cache")
    architecture = {"arms": list(_runtime.ARMS), "time_alpha": 0., "quantity_loss_mode": condition["loss_mode"],
        "quantity_loss_modes": [condition["loss_mode"]], "quantity_loss_normalization": "none",
        "time_input_transform": TRANSFORMS[condition["time_input"]], "quantity_input_transform": "log1p_nonnegative_float32",
        "late_path_parameter_count": 1024, "model_architecture_changed": True, "checkpoint_warmstart": False,
        "bottleneck_width": 8, "insertion": "after_encoder2_before_existing_static_hard_lmm"}
    contract = {"schema": _runtime.SCHEMA, "created_at": _impl.now(),
        "science_spec_path": str(_impl.REPORT / "SCIENCE_SPEC.json"), "science_spec_sha256": _impl.sha(_impl.REPORT / "SCIENCE_SPEC.json"),
        "source": source, "hosts": hosts, "jobs": build_jobs(condition), "datasets": deepcopy(parent["datasets"]),
        "input_files": inputs, "data_receipt": deepcopy(parent["data_receipt"]), "architecture": architecture,
        "design_sha256": _impl.canonical(architecture), "common_condition": condition,
        "common_condition_selection": selection, "evaluation_scope": "validation_only",
        "authorization": "2026-10-19 user approved two late-path candidates x Cap3datasets x3seeds after loss/time comparisons, same one common condition, max3 per5080/5090; CNN unchanged; no Test",
        "held_out_test_authorized": False, "historical_cnn_retrained": False, "historical_cnn_reevaluated": False,
        "training": deepcopy(parent["training"]), "limits": deepcopy(parent["limits"]),
        "execution": {"maximum_concurrent_per_host": 3, "completed_predecessor_required_before_gpu_work": True,
            "validation_gate_scope": "all assigned jobs on each host", "existing_campaign_remains_stopped": True,
            "recurring_automation": False},
        "comparison": {"scope": "two candidate arms18 versus same-condition storedMLP16 references9",
            "model_count": 27, "seeds": [42, 52, 62], "sample_sd_ddof": 1},
        "baseline_validation_references": baseline, "parent_raw_time27": binding,
        "stored_cap_mlp_reference": deepcopy(parent["stored_cap_mlp_reference"]),
        "packaging": {"required_empty_source_directories": ["sample_data"], "test_inputs_packaged": False}}
    for host in hosts:
        _runtime.validate_contract(contract, host)
    origin_root = _impl.ROOT / "search_artifacts" / PARENT_NAME / "input_snapshot"
    for name, digest in inputs.items():
        _impl.require(_impl.sha(_runtime.local_path(origin_root, name)) == digest, "Parent TrainVal input changed")
    _impl.BUNDLE.mkdir(parents=True, exist_ok=False)
    for name in inputs:
        origin, target = _runtime.local_path(origin_root, name), _runtime.local_path(_impl.BUNDLE / "input_snapshot", name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with origin.open("rb") as src, target.open("xb") as dst:
            _impl.shutil.copyfileobj(src, dst)
    _impl.write(_impl.BUNDLE / "execution_contract.json", contract)
    archive = _impl.package_archive(contract, source)
    _impl.write(_impl.REPORT / "COMMON_CONDITION.json", selection)
    receipt = {"status": "prepared", "created_at": _impl.now(), "source_revision": source["base_git_revision"],
        "source_file_count": len(source["files"]), "contract_sha256": _impl.sha(_impl.BUNDLE / "execution_contract.json"),
        "canonical_contract_sha256": _impl.canonical(contract), "archive_sha256": _impl.sha(archive),
        "planned_fit_count": 18, "host_fit_counts": {h: sum(j["host"] == h for j in contract["jobs"]) for h in hosts},
        "baseline_reference_count": 9, "common_condition": condition, "held_out_test_evaluated": False,
        "historical_cnn_retrained": False, "parent_completion_sha256": binding["completion_sha256"]}
    _impl.write(_impl.REPORT / "PREPARED.json", receipt)
    return receipt


_original_verify_collected = _impl.verify_collected


def verify_collected(contract, host, originals):
    result = _original_verify_collected(contract, host, originals)
    condition = contract["common_condition"]
    for job in _runtime.validate_contract(contract, host):
        for stage in _runtime.STAGES:
            receipt = _impl.read(Path(originals) / "run" / job["id"] / f"{stage}_receipt.json")
            _impl.require(receipt["arm"] == job["arm"] and receipt["loss_mode"] == condition["loss_mode"]
                and receipt["time_input_transform"] == TRANSFORMS[condition["time_input"]], "Collected arm/condition mismatch")
    return result


_impl.source_snapshot = source_snapshot
_impl.baseline_validation_references = baseline_validation_references
_impl.prepare, _impl.build_jobs, _impl.verify_collected = prepare, build_jobs, verify_collected
for _name, _value in tuple(_impl.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value

if __name__ == "__main__":
    raise SystemExit(_impl.main())
