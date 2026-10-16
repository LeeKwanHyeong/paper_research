"""Local preparation only: no dataset reader, candidate registration, fit or SSH."""
import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path

from models.TPPs.CountAwareTitanMomentCNN import CNN, ARMS, REVIN, CONDITIONED, CAPACITY, CONTRACT_ID, CountAwareTitanMomentCNN, metadata
from paper.scripts import run_titantpp_cnn_gru_components as original


def build_model(data, arm):
    if arm == CNN:
        return original.build_model(data, arm)
    if arm not in ARMS:
        raise ValueError("Unknown preparation arm")
    original.install_hooks()
    from models.TPPs import CountAwareFactory as factory
    signature = inspect.getclosurevars(factory.build_count_aware_model).nonlocals.get("signature")
    if not isinstance(signature, inspect.Signature):
        raise ValueError("Original model factory signature unavailable")
    config = {k: v for k, v in data["model"].items() if k not in
              ("backbone", "lambda_log_qty", "lambda_tail", "time_head_lr_multiplier")}
    bound = signature.bind(CNN, **config, train_log_mean=data["statistics"]["train_log_mean"],
                           train_log_std=data["statistics"]["train_log_std"], max_seq_len=data["loader"]["max_seq_len"])
    bound.apply_defaults()
    values = dict(bound.arguments)
    values.pop("backbone")
    if values.pop("titans_memory_gradient_clip") is not None:
        raise ValueError("Original memory clipping must remain unchanged")
    model = CountAwareTitanMomentCNN(**values, moment_arm=arm)
    return factory.with_time_metadata(model, {**metadata(arm=arm), "max_len": values["max_seq_len"]})


def run_fit(*args, **kwargs):
    raise PermissionError("Preparation only: new training requires the user's later decision and native verification")


def raw_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_bundle(project, destination):
    """Verify old frozen source then copy it, without changing any old artifact."""
    project, destination = Path(project).resolve(), Path(destination).resolve()
    parent_root = project / "search_artifacts/titantpp_cnn_history_adaptation_matched_20261015_v1"
    parent_path = parent_root / "contract.json"
    expected_parent = "c5ae7ffed7e89c798dcb7bf7264e15a781adf10c6162a632501c3b8019bcdb3b"
    if raw_sha(parent_path) != expected_parent:
        raise ValueError("Frozen parent preparation contract mismatch")
    parent = json.loads(parent_path.read_text())
    files, checkout_differences = {}, []
    for name, expected in parent["source_files"].items():
        frozen = parent_root / "source" / name
        if raw_sha(frozen) != expected:
            raise ValueError("Frozen source mismatch: " + name)
        files[name] = frozen.read_bytes()
        current = project / name
        if not current.is_file() or raw_sha(current) != expected:
            checkout_differences.append(name)
    new_files = ("models/TPPs/CountAwareTitanMomentCNN.py", "paper/scripts/prepare_titantpp_moment_cnn.py")
    for name in new_files:
        if name in files:
            raise ValueError("New module would replace frozen source")
        files[name] = (project / name).read_bytes()
    source_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    contract = {
        "schema": CONTRACT_ID, "status": "prepared_not_trained", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authorization": {"local_contract_implementation_synthetic_verification": True, "new_training": False,
                          "remote_calls": False, "real_data_inference": False, "test_evaluation": False},
        "research_question": "Does observed log-quantity mean/std conditioning improve causal CNN history processing beyond latent block normalization and matched capacity?",
        "historical_test_informed_exploration": True, "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False, "shared_science_modified": False,
        "arms": [CNN, *ARMS], "arm_metadata": {arm: metadata(arm=arm) for arm in ARMS},
        "primary_contrast": [CONDITIONED, CAPACITY], "secondary_contrasts": [[CONDITIONED, REVIN], [REVIN, CNN], [CONDITIONED, CNN]],
        "reference_parent": {"path": str(parent_path), "raw_sha256": expected_parent},
        "source_files": source_hashes, "frozen_parent_source_count": len(parent["source_files"]),
        "current_checkout_differences_preserved": checkout_differences,
        "dataset_binding": parent["dataset"],
        "dataset_scope": "Taxi reference binding only; implementation is dataset-independent; RAF/Intermittent bindings require a later execution decision",
        "seeds": [42, 52, 62],
        "baseline_references": [{"id": job["id"], "seed": job["seed"], "dataset": job["dataset"], "host": job["host"],
            "checkpoint": job["original"]["checkpoint"], "selected_epoch": job["original"]["selected_epoch"],
            "checkpoint_state_sha256": job["original"]["checkpoint_state_sha256"]} for job in parent["baseline_jobs"]],
        "baseline_binary_reverified_in_this_preparation": False,
        "training": {**parent["training"], "authorized": False},
        "execution": {"state": "not_authorized", "new_fit_count_if_all_three_arms_and_three_seeds_approved": 9,
            "baseline_reused_count": 3, "proposed_host_for_taxi": "5080", "maximum_parallel_per_host": 3,
            "native_gpu_runtime_input_sha_verified": False, "permit_created": False, "training_entrypoint_installed": False},
        "host_identity_reference_only": parent["runtimes"],
        "claim_boundaries": ["latent block RevIN, not standard end-to-end RevIN", "log-domain moments, not raw-domain normalization",
            "plain linear block scale cancels; mean-centering remains", "no new independent Test or causal/generalization claim"],
        "metrics": ["validation_raw_quantity_rmse", "validation_raw_quantity_mae", "large_quantity_rmse", "large_quantity_mae",
                    "recorded_positive_integer_time_nll", "large_quantity_recorded_positive_integer_time_nll"],
        "aggregation": "dataset-separated 3seed mean and sample SD ddof1; paired absolute/relative differences; all adverse seeds retained",
        "decision_rule": "Do not claim conditional filter benefit unless CONDITIONED improves mean Validation raw RMSE over both CAPACITY and REVIN; report CNN comparison, MAE and adverse secondary results regardless",
    }
    # All integrity checks precede an exclusive new directory. Partial writes are
    # retained if interrupted; never delete or overwrite an existing attempt.
    destination.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        out = destination / "source" / name
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("xb") as handle:
            handle.write(content)
    contract_bytes = (json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with (destination / "contract.json").open("xb") as handle:
        handle.write(contract_bytes)
    manifest = {"contract_raw_sha256": hashlib.sha256(contract_bytes).hexdigest(), "source_files": source_hashes,
                "source_closure_sha256": hashlib.sha256(json.dumps(source_hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}
    with (destination / "source_manifest.json").open("x") as handle:
        json.dump(manifest, handle, sort_keys=True, indent=2); handle.write("\n")
    return {"bundle": str(destination), "status": contract["status"], "source_count": len(files), **manifest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_bundle(Path(__file__).resolve().parents[2], args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "source_files"}, ensure_ascii=False, indent=2))
