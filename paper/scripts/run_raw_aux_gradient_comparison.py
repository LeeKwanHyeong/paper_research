#!/usr/bin/env python3
"""Connect the frozen raw-auxiliary design to the trainer and qualify on CPU.

The CLI only runs bounded synthetic CPU qualification and freezes local source.
It does not load research datasets, connect to a server, or launch GPU training.
Research GPU supervision and its execution contract remain a separate step.
"""
from __future__ import annotations

import argparse
import ast
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper.scripts import run_quantity_comparison as base

DESIGN_SHA256 = "34a16247a55c370c5b5dc9c50abd70a6818d8ec2e19ee6addc492425eeb3dde2"
DESIGN_PATH = ROOT / "paper/contracts/raw_aux_gradient_control_design_v1.json"
ARMS = ("B_log_original", "mixed_original", "mixed_raw_capped_original")
require, read_json, write_json = base.require, base.read_json, base.write_json
sha_file, sha_json = base.sha_file, base.sha_json


def load_design(path=DESIGN_PATH):
    require(sha_file(path) == DESIGN_SHA256, "Frozen raw auxiliary design changed")
    design = read_json(path)
    require([arm["id"] for arm in design["arms"]] == list(ARMS), "Arm registry changed")
    return design


def source_hashes(root=ROOT):
    """Resolve this entrypoint's complete first-party static import closure."""
    root = Path(root).resolve()
    pending = [root / "paper/scripts/run_raw_aux_gradient_comparison.py"]
    visited = set()

    def add(module):
        if module.split(".")[0] not in base.PACKAGES:
            return
        stem = root.joinpath(*module.split("."))
        for path in (stem.with_suffix(".py"), stem / "__init__.py"):
            if path.is_file():
                pending.append(path)
        parent = stem.parent
        while parent != root:
            if (parent / "__init__.py").is_file():
                pending.append(parent / "__init__.py")
            parent = parent.parent

    while pending:
        path = pending.pop().resolve()
        require(path.is_relative_to(root), "Source dependency escapes snapshot")
        if path in visited:
            continue
        visited.add(path)
        package = list(path.relative_to(root).parts[:-1])
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                prefix = package[:len(package) - node.level + 1] if node.level else []
                module = ".".join(prefix + ([node.module] if node.module else []))
                add(module)
                for alias in node.names:
                    if alias.name != "*":
                        add(f"{module}.{alias.name}")
    return {str(path.relative_to(root)): sha_file(path) for path in sorted(visited)}


def objective_for_arm(arm_id, design=None):
    from paper.scripts.mixed_quantity_objective import MixedQuantityObjective
    from paper.scripts.raw_aux_gradient_control import RawAuxGradientControl

    design = load_design() if design is None else design
    require(arm_id in ARMS, "Unknown raw auxiliary arm")
    measurement = read_json(ROOT / design["evidence"]["measurement_contract"]["path"])
    require(sha_file(ROOT / design["evidence"]["measurement_contract"]["path"])
            == design["evidence"]["measurement_contract"]["sha256"], "Coefficient receipt changed")
    objective = MixedQuantityObjective(
        "B_log_original" if arm_id == ARMS[0] else "mixed_original",
        0.0 if arm_id == ARMS[0] else design["losses"]["alpha"],
        1.0, measurement["train_probe"]["calibration_sha256"],
    )
    return objective, RawAuxGradientControl() if arm_id == ARMS[2] else None


def run_arm(*, arm_id, model, train_loader, validation_loader, statistics,
            output_dir, identity, synthetic=False, device="cpu", epochs=120,
            resume=False, stop_after_epochs=None, budget_check=None,
            cuda_qualification=None, design_path=DESIGN_PATH):
    """Exercise the real trainer only on bounded synthetic CPU inputs here."""
    from paper.scripts.quantity_comparison_engine import run_case
    from paper.scripts.quantity_objective_comparison import QuantityCase

    design = load_design(design_path)
    require(identity.get("raw_aux_design_sha256") == DESIGN_SHA256, "Design identity missing")
    require(identity.get("source", {}).get("files") == source_hashes(), "Source identity changed")
    require(identity["source"].get("files_sha256") == sha_json(identity["source"]["files"]),
            "Source manifest digest differs")
    require(synthetic is True, "Research execution is disabled until a separate GPU execution contract is implemented")
    from torch.utils.data import TensorDataset
    require(device == "cpu" and type(epochs) is int and 1 <= epochs <= 2,
            "CPU qualification budget exceeded")
    for loader in (train_loader, validation_loader):
        require(type(loader.dataset) is TensorDataset and 1 <= len(loader.dataset) <= 8,
                "Qualification requires <=8 synthetic TensorDataset rows")
        require(len(loader.dataset.tensors) == 3
                and all(tensor.ndim == 2 and 2 <= tensor.shape[1] <= 8
                        for tensor in loader.dataset.tensors),
                "Qualification sequence shape budget exceeded")
    objective, control = objective_for_arm(arm_id, design)
    return run_case(
        model=model, train_loader=train_loader, validation_loader=validation_loader,
        case=QuantityCase.B_LOG_ORIGINAL, statistics=statistics, output_dir=output_dir,
        epochs=epochs, seed=42, identity=identity, device=device, lr=.001,
        weight_decay=.01, grad_clip=1., resume=resume, stop_after_epochs=stop_after_epochs,
        budget_check=budget_check, cuda_qualification=cuda_qualification,
        mixed_objective=objective, raw_aux_control=control,
    )


def synthetic_inputs():
    import torch
    model, train, validation = base.synthetic_inputs()
    # Exercise all five frozen quantity bins without reading any real targets.
    train.dataset.tensors[2][:, -1] = torch.tensor([1., 2., 10., 32., 46., 47., 187., 188.])
    return model, train, validation


def _probe(manifest_path, arm_id, stage):
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    from paper.scripts.quantity_objective_comparison import QuantityStatistics

    manifest = read_json(manifest_path)
    configure_runtime("cpu", threads=1)
    require(runtime_identity("cpu") == manifest["identity"]["runtime"], "CPU runtime changed")
    model, train, validation = synthetic_inputs()
    return run_arm(
        arm_id=arm_id, model=model, train_loader=train, validation_loader=validation,
        statistics=QuantityStatistics(1.0, load_design()["losses"]["raw_scale"]),
        output_dir=Path(manifest["output_dir"]) / arm_id / ("full" if stage == "full" else "split"),
        identity=manifest["identity"], synthetic=True, epochs=2,
        resume=stage == "resume", stop_after_epochs=1 if stage == "first" else None,
    )


def audit_pairs(summaries):
    require(set(summaries) == set(ARMS), "All three arms required")
    baseline = summaries[ARMS[0]]
    for arm_id in ARMS:
        result = summaries[arm_id]
        require(result["status"] == "complete" and result["initial_state_sha256"] == baseline["initial_state_sha256"]
                and result["global_step"] == baseline["global_step"], "Arm initialization/steps differ")
        require(len(result["history"]) == len(baseline["history"]), "Arm epochs differ")
        for b, c in zip(baseline["history"], result["history"], strict=True):
            for key in ("epoch", "global_step", "train_count", "train_batches", "validation_count",
                        "validation_batches", "train_batch_order_sha256"):
                require(b[key] == c[key], "Paired exposure mismatch: " + key)
    return {"passed": True, "arms": 3, "epochs": len(baseline["history"]),
            "global_steps_per_arm": baseline["global_step"]}


def check_production_shape():
    """Check the production architecture on two artificial 256-position rows."""
    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.mixed_quantity_objective import mixed_joint_causal_batch_objective
    from paper.scripts.quantity_objective_comparison import QuantityStatistics
    from paper.scripts.run_time_quantity_diagnostic import reset_seed
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    design = load_design()
    exposure = design["initialization_and_exposure"]
    stats = QuantityStatistics(exposure["statistics"]["train_log_mean"], design["losses"]["raw_scale"])
    reset_seed(42)
    model, _ = build_count_aware_model(**exposure["model"], train_log_mean=stats.mu,
        train_log_std=exposure["statistics"]["train_log_std"], max_seq_len=256)
    initial_sha = canonical_state_dict_sha256(model.state_dict())
    # The real initialization has a zero quantity-head weight. Its constant
    # predictions cannot detect encoder leakage, so make this synthetic probe
    # sensitive after recording the untouched production initialization.
    with torch.no_grad():
        model.quantity_head.weight.copy_(torch.linspace(-.02, .03, model.quantity_head.weight.numel())
                                         .reshape_as(model.quantity_head.weight))
    probe_sha = canonical_state_dict_sha256(model.state_dict())
    dts = torch.ones((2, 256))
    mask = torch.ones_like(dts, dtype=torch.bool)
    mask[0, :64], mask[1, :128] = False, False
    quantities = torch.full_like(dts, 3.)
    quantities[:, -1] = torch.tensor([46., 188.])
    model.eval()
    objective, _ = objective_for_arm(ARMS[0], design)
    with torch.no_grad():
        before = mixed_joint_causal_batch_objective(model, dts, mask, quantities,
                                                    statistics=stats, objective=objective)
        dts[~mask], quantities[~mask] = 999., 999.
        dts[:, -1], quantities[:, -1] = 42., 123.
        after = mixed_joint_causal_batch_objective(model, dts, mask, quantities,
                                                   statistics=stats, objective=objective)
        dts[:, -2], quantities[:, -2] = 7., 100.
        changed_history = mixed_joint_causal_batch_objective(model, dts, mask, quantities,
                                                             statistics=stats, objective=objective)
    require(torch.equal(before["pred_qty"], after["pred_qty"]), "Production-shape target/padding causality differs")
    require(not torch.equal(after["pred_qty"], changed_history["pred_qty"]),
            "Causality probe is insensitive to valid history")
    require(canonical_state_dict_sha256(model.state_dict()) == probe_sha, "Evaluation changed probe model state")
    return {"passed": True, "hidden_dim": 64, "sequence_length": 256,
            "synthetic_rows": 2, "optimizer_steps": 0, "initial_state_sha256": initial_sha,
            "historical_initial_state_sha256": exposure["initial_state_sha256"],
            "historical_initialization_matches_cpu": initial_sha == exposure["initial_state_sha256"],
            "synthetic_probe_nonzero_head": True, "probe_state_sha256": probe_sha,
            "valid_history_sensitivity": True,
            "quantity_prediction_target_padding_causality": True,
            "production_batch128_cuda_qualification": False}


def qualify_cpu(output_dir):
    from paper.scripts.quantity_comparison_runtime import configure_runtime, runtime_identity
    from paper.scripts.quantity_objective_comparison import QuantityStatistics
    from paper.scripts.raw_aux_gradient_acceptance import assess_acceptance, evaluate_checkpoints
    from simple_lab_test.search.common.runner import torch_load_checkpoint

    design = load_design()
    output = Path(output_dir).resolve()
    require(not output.exists(), "Qualification output must be a new directory")
    configure_runtime("cpu", threads=1)
    files = source_hashes()
    contract = {"schema": "raw_aux_cpu_integration_v1", "design_sha256": DESIGN_SHA256,
                "source": {"files": files, "files_sha256": sha_json(files)},
                "runtime": runtime_identity("cpu"), "epochs": 2, "samples": 8,
                "hidden_dim": 8, "max_seq_len": 8, "arms": list(ARMS)}
    identity = {"execution_contract_sha256": sha_json(contract), "source": contract["source"],
                "runtime": contract["runtime"], "data": {"kind": "synthetic", "targets": 8},
                "raw_aux_design_sha256": DESIGN_SHA256, "purpose": "synthetic_qualification"}
    manifest = {"output_dir": str(output), "contract": contract, "identity": identity}
    manifest_path = output / "manifest.json"
    write_json(manifest_path, manifest)
    summaries, evaluations, replay = {}, {}, {}
    began = time.monotonic()
    for arm_id in ARMS:
        for stage in ("full", "first", "resume"):
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "_probe",
                       "--manifest", str(manifest_path), "--arm", arm_id, "--stage", stage]
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
            write_json(output / arm_id / (stage + "_process.json"),
                       {"command": command, "exit_code": result.returncode,
                        "stdout": result.stdout[-2000:], "stderr": result.stderr[-2000:]})
        folder = output / arm_id
        full = read_json(folder / "full/summary.json")
        split = read_json(folder / "split/summary.json")
        base.compare_replay(full, split)
        left = torch_load_checkpoint(folder / "full/last_epoch_state.pt", map_location="cpu")
        right = torch_load_checkpoint(folder / "split/last_epoch_state.pt", map_location="cpu")
        replay[arm_id] = {}
        for key in ("model_state_sha256", "optimizer_state_sha256", "rng_state_sha256"):
            require(left[key] == right[key], "Fresh-process replay differs: " + key)
            replay[arm_id][key] = left[key]
        summaries[arm_id] = full
        model, _, validation = synthetic_inputs()
        objective, _ = objective_for_arm(arm_id, design)
        evaluation = evaluate_checkpoints(
            model=model, validation_loader=validation,
            statistics=QuantityStatistics(1.0, design["losses"]["raw_scale"]), objective=objective,
            arm_dir=folder / "full", arm_id=arm_id, synthetic=True,
        )
        evaluations[arm_id] = evaluation
        write_json(folder / "validation_diagnosis.json", evaluation)
        del left, right, model
        gc.collect()
    paired = audit_pairs(summaries)
    acceptance = assess_acceptance(evaluations, synthetic=True)
    write_json(output / "acceptance_pipeline.json", acceptance)
    production_shape = check_production_shape()
    write_json(output / "production_shape_cpu.json", production_shape)
    require(source_hashes() == files, "Source changed during CPU qualification")
    proof_files = {str(path.relative_to(output)): sha_file(path)
                   for path in sorted(output.rglob("*")) if path.is_file()}
    receipt = {"schema": "raw_aux_cpu_integration_receipt_v1", "passed": True,
               "qualifies_cuda": False, "design_sha256": DESIGN_SHA256,
               "source_files_sha256": sha_json(files), "runtime": contract["runtime"],
               "paired_exposure": paired, "fresh_process_replay": replay,
               "production_shape_cpu": production_shape,
               "validation_checkpoint_replays": 6, "synthetic_optimizer_updates": 24,
               "proof_files": proof_files, "manifest_sha256": sha_file(manifest_path),
               "elapsed_seconds": time.monotonic() - began,
               "quality_claim": "Synthetic pipeline qualification, not research performance",
               "research_data_or_checkpoints_read": False, "new_gpu_execution": False}
    write_json(output / "receipt.json", receipt)
    return receipt


def freeze_binding(qualification_path, output_dir):
    """Preserve reviewed source; this binding does not authorize execution."""
    design = load_design()
    receipt = read_json(qualification_path)
    files = source_hashes()
    require(receipt["passed"] is True and receipt["qualifies_cuda"] is False
            and receipt["design_sha256"] == DESIGN_SHA256
            and receipt["source_files_sha256"] == sha_json(files), "Fresh CPU qualification required")
    require(receipt.get("schema") == "raw_aux_cpu_integration_receipt_v1"
            and receipt.get("paired_exposure") == {"passed": True, "arms": 3, "epochs": 2, "global_steps_per_arm": 4}
            and set(receipt.get("fresh_process_replay", {})) == set(ARMS)
            and receipt.get("validation_checkpoint_replays") == 6
            and receipt.get("synthetic_optimizer_updates") == 24,
            "Incomplete CPU qualification evidence")
    evidence = Path(qualification_path).resolve().parent
    proofs = receipt.get("proof_files", {})
    required_proofs = {"manifest.json", "acceptance_pipeline.json", "production_shape_cpu.json"}
    for arm in ARMS:
        required_proofs.add(f"{arm}/validation_diagnosis.json")
        for stage in ("full", "first", "resume"):
            required_proofs.add(f"{arm}/{stage}_process.json")
        for stage in ("full", "split"):
            required_proofs.update(f"{arm}/{stage}/{name}" for name in ("summary.json", "last_epoch_state.pt", "contract.json"))
    require(required_proofs <= set(proofs), "Missing CPU qualification artifacts")
    for relative, digest in proofs.items():
        target = (evidence / relative).resolve()
        require(target.is_relative_to(evidence) and target.is_file() and sha_file(target) == digest,
                "CPU qualification evidence changed: " + relative)
    manifest = read_json(evidence / "manifest.json")
    require(sha_file(evidence / "manifest.json") == receipt["manifest_sha256"]
            and manifest["contract"]["source"]["files"] == files
            and manifest["identity"]["execution_contract_sha256"] == sha_json(manifest["contract"])
            and manifest["identity"]["raw_aux_design_sha256"] == DESIGN_SHA256,
            "Qualification manifest identity differs")
    destination = Path(output_dir).resolve()
    require(not destination.exists(), "Refusing to replace an implementation binding")
    snapshot = destination / "source"
    for relative, digest in files.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
        require(sha_file(target) == digest, "Snapshot content differs")
    for path in (DESIGN_PATH, ROOT / design["evidence"]["measurement_contract"]["path"]):
        target = snapshot / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    require(source_hashes(snapshot) == files, "Frozen source closure differs")
    binding = {"schema": "raw_aux_implementation_binding_v1", "design_sha256": DESIGN_SHA256,
               "source": {"files": files, "files_sha256": sha_json(files), "snapshot": str(snapshot)},
               "cpu_qualification": {"path": str(Path(qualification_path).resolve()),
                                      "sha256": sha_file(qualification_path)},
               "readiness": {"runner_connected": True, "cpu_small_actual_titan_qualified": True,
                             "production_cuda_qualified": False, "execution_approved": False},
               "remaining": ["5090 production-shape throughput/memory qualification under a bound execution contract",
                             "resource/wall-time ceiling and separate GPU training approval"],
               "no_gpu_started": True}
    write_json(destination / "implementation_binding.json", binding)
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    qualify = commands.add_parser("qualify-cpu")
    qualify.add_argument("--output", type=Path, required=True)
    freeze = commands.add_parser("freeze-binding")
    freeze.add_argument("--qualification", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    probe = commands.add_parser("_probe")
    probe.add_argument("--manifest", type=Path, required=True)
    probe.add_argument("--arm", choices=ARMS, required=True)
    probe.add_argument("--stage", choices=("full", "first", "resume"), required=True)
    args = parser.parse_args()
    if args.command == "qualify-cpu":
        result = qualify_cpu(args.output)
    elif args.command == "freeze-binding":
        result = freeze_binding(args.qualification, args.output)
    else:
        result = _probe(args.manifest, args.arm, args.stage)
    print(json.dumps({k: result[k] for k in ("passed", "qualifies_cuda", "elapsed_seconds", "no_gpu_started", "status")
                      if k in result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
