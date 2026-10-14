"""Local-only compatibility repair proposal; never deploys or starts/retries work.

The failed sealed execution is preserved. This diagnostic proposes only the two
legacy keys consumed by the frozen time interface before the cap adapter replaces
its backbone metadata, and validates the real admitted Train/Validation inputs.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT = Path(__file__).resolve().parents[2]
PREPARED = PROJECT / "search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1"
EXECUTION = PREPARED / "execution"
PARENT = PROJECT / "search_artifacts/titantpp_cnn_gru_components_20261012_v5"
OUTPUT = PROJECT / "reports/titantpp_gru_residual_cap_validation_20261014_v1/qualification_failure/local_repair_v2"
EXECUTION_RAW = "efa5dfb71cb38df205f8ae1fa641b938da15b0d8a981e00a38cb22fe9c1477f9"
PARENT_RAW = "591a31ed862148c1f7343fef33a65da9d263ec9b80769a240d1ff43cdb6b49f4"
INTEGRITY_RAW = "4bee280e4777ca2c13ed97674074e232d9359f0c57cfbbb3a5bb4ff49039b82b"
KEYS = ("design_sha256", "architecture")


def require(value, message):
    if not value:
        raise ValueError(message)


def raw_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def propose_contract(original, parent):
    require(all(key not in original for key in KEYS), "Proposal only accepts the known missing-key failure")
    require(isinstance(parent.get("design_sha256"), str) and len(parent["design_sha256"]) == 64,
            "Missing parent design provenance")
    require(isinstance(parent.get("architecture"), dict), "Missing parent architecture provenance")
    proposed = deepcopy(original)
    for key in KEYS:
        proposed[key] = deepcopy(parent[key])
    validate_delta(original, proposed, parent)
    return proposed


def validate_delta(original, proposed, parent):
    require(set(proposed) - set(original) == set(KEYS), "Only two inherited compatibility keys may be added")
    require(all(proposed.get(key) == value for key, value in original.items()),
            "Existing science, ownership, authorization or budget changed")
    require(all(proposed[key] == parent[key] for key in KEYS), "Parent compatibility metadata changed")


def main():
    require(not OUTPUT.exists(), "Local proof already exists; no overwrite")
    require(raw_sha(EXECUTION / "execution_contract.json") == EXECUTION_RAW, "Failed execution changed")
    require(raw_sha(EXECUTION / "execution_integrity.json") == INTEGRITY_RAW, "Execution integrity changed")
    require(raw_sha(PARENT / "execution_contract.json") == PARENT_RAW, "Pinned parent changed")
    spec = importlib.util.spec_from_file_location("_cap_sealed_launcher",
        PROJECT / "paper/scripts/launch_titantpp_gru_cap_validation.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    original, integrity = launcher.check_bundle(EXECUTION)
    parent = json.loads((PARENT / "execution_contract.json").read_text())
    proposed = propose_contract(original, parent)

    # This file runs as a script so the sealed package path wins before imports.
    source = PREPARED / "source"
    require("paper.scripts.run_titantpp_gru_residual_cap" not in sys.modules, "Already imported live adapter")
    sys.path.insert(0, str(source))
    from paper.scripts import run_titantpp_gru_residual_cap as runner
    import torch
    require(Path(runner.__file__).resolve().is_relative_to(source), "Adapter is not the frozen source")
    require(Path(runner.base.__file__).resolve().is_relative_to(source), "Interface is not the frozen source")
    require(not torch.cuda.is_initialized(), "Local proof cannot use CUDA")
    # The native pipeline runs from its admitted source/input root. Preserve the
    # same relative-path resolution here without rewriting any input contract.
    os.chdir(Path(proposed["inherited_input_root"]))
    OUTPUT.mkdir(parents=True)
    write(OUTPUT / "metadata_only_contract.proposed.json", proposed)
    write(OUTPUT / "proposal.json", {
        "status": "local_diagnostic_only_not_a_launch_permit", "deployment_authorized_by_this_file": False,
        "automatic_retry": False, "failed_contract_raw_sha256": EXECUTION_RAW,
        "parent_contract_raw_sha256": PARENT_RAW, "added_keys": list(KEYS),
        "all_original_contract_values_preserved": True,
        "proposed_raw_sha256": raw_sha(OUTPUT / "metadata_only_contract.proposed.json"),
        "operational_note": "A separately authorized new isolated attempt is required; this proposal preserves failed-root references and must not be deployed.",
        "source_files_sha256": proposed["source"]["files_sha256"],
    })
    rows = []
    for data in proposed["datasets"]:
        dataset = data["dataset_id"]
        frame, receipt = runner.base.prepare_admitted_data(data, data_root=Path(proposed["inherited_input_root"]))
        require(receipt["held_out_materialized"] is False and receipt["populations"] ==
                data["inherited_data_identity"]["populations"], "Admitted population drift")
        # Reproduce the two missing-key conditions against the actual frozen path.
        reproduced = []
        for missing, value in (("design_sha256", original),
                ("architecture", {**original, "design_sha256": parent["design_sha256"]})):
            try:
                runner.time_interface(data, frame, value, arm=runner.CAPPED)
            except KeyError as error:
                require(error.args == (missing,), "Unexpected interface failure")
                reproduced.append(missing)
            else:
                raise ValueError("Expected missing-key regression did not reproduce")
        interface = runner.time_interface(data, frame, proposed, arm=runner.CAPPED)
        require(interface["data_scope"] == "verified_train_validation_only" and
                interface["backbone_design"]["intervention"] == runner.CAPPED and
                interface["backbone_design"]["training"] == proposed["training"], "Wrong cap interface")
        require("design_sha256" not in interface["backbone_design"] and
                "architecture" not in interface["backbone_design"], "Legacy metadata leaked into cap design")
        with tempfile.TemporaryDirectory(prefix="cap_metadata_only_") as temporary:
            args = runner.training_args(proposed, data, temporary, runner.CAPPED)
            arguments = {key: getattr(args, key) for key in (
                "epochs", "min_epochs", "early_stopping_patience", "batch_size", "checkpoint_monitor",
                "lambda_log_qty", "lambda_tail", "time_head_mode", "time_sigma_floor",
                "execution_role", "model_role", "lr", "weight_decay", "grad_clip", "force_rerun")}
            require((args.epochs, args.min_epochs, args.early_stopping_patience, args.batch_size) ==
                    (300, 40, 40, 128) and args.force_rerun is False, "Budget or fresh-fit drift")
        row = {"dataset": dataset, "status": "passed", "missing_keys_reproduced": reproduced,
            "input_receipt": receipt, "interface": interface, "training_arguments": arguments,
            "held_out_materialized": False, "model_constructed": False,
            "optimizer_updates": 0, "remote_calls": 0, "cuda_initialized": torch.cuda.is_initialized()}
        require(row["cuda_initialized"] is False, "Unexpected CUDA initialization")
        write(OUTPUT / (dataset + ".json"), row)
        rows.append({"dataset": dataset, "status": "passed", "path": dataset + ".json",
            "raw_sha256": raw_sha(OUTPUT / (dataset + ".json")), "interface_sha256": canonical(interface)})
        print(json.dumps({"dataset": dataset, "status": "passed", "remote_calls": 0}), flush=True)
    require(raw_sha(EXECUTION / "execution_contract.json") == EXECUTION_RAW, "Failed execution was modified")
    receipt = {"status": "local_train_validation_integration_passed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_delta": list(KEYS), "datasets": rows, "frozen_execution_integrity": integrity,
        "script_raw_sha256": raw_sha(__file__), "remote_calls": 0, "training_fit_count": 0,
        "test_materialized": False, "native_gpu_qualification_passed": False,
        "all_original_science_values_preserved": True, "not_a_launch_permit": True}
    write(OUTPUT / "receipt.json", receipt)
    print(json.dumps({"receipt": str(OUTPUT / "receipt.json"), "raw_sha256": raw_sha(OUTPUT / "receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
