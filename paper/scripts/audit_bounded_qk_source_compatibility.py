"""Replay existing B/FULL checkpoints on synthetic inputs through the K=1 builder.

No benchmark rows, model optimization, or time performance evaluation are used.
Run in an isolated source tree with --artifact-root pointing to existing artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from models.TPPs.CountAwareFactory import validate_checkpoint_route
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.run_backbone_normalized_duration import validate_source_artifacts
from paper.scripts.run_hard_lmm_frozen_lognormal_duration import (
    build_frozen_lognormal_candidate, freeze_time_head_only, state_partition_sha256,
)
from paper.scripts.run_matched_frozen_lognormal_duration import build_source_model
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.artifact_root / "paper/contracts/hard_lmm_qkv_path_diagnostic_v1.json"
    assert sha(contract_path) == "a4ae310d254ed124fc8be4c340a10d22c462a33fb08d669afd38504cf978ddea"
    contract = json.loads(contract_path.read_text())
    aligned_path = args.artifact_root / "paper/contracts/aligned_frozen_lognormal_duration_v1.json"
    assert sha(aligned_path) == "677cc3af91d84dfea8e3b4e71b38697fb467059f680678f0b8370a50f2366f16"
    aligned = {d["dataset"]: d for d in json.loads(aligned_path.read_text())["datasets"]}
    torch.set_num_threads(2)
    rows = []
    for dataset in contract["datasets"]:
        for role, key in (("B", "B"), ("FULL", "candidate")):
            binding = dataset[key]
            path = args.artifact_root / binding["checkpoint_path"]
            assert sha(path) == binding["checkpoint_file_sha256"]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            summary_path = args.artifact_root / binding["summary_path"]
            assert sha(summary_path) == binding["summary_sha256"]
            summary = json.loads(summary_path.read_text())
            history_path = path.parent / "history.json"
            source_history = json.loads(history_path.read_text())
            source_spec = {
                "backbone": binding["backbone"],
                "training_source_revision": payload["source_revision"],
                "training_source_revision_history": payload["source_revision_history"],
                "checkpoint_file_sha256": sha(path),
                "checkpoint_state_sha256": payload["model_state_sha256"],
                "source_non_time_state_sha256": state_partition_sha256(payload["model_state_dict"], time_head=False),
                "history_path": str(history_path),
                "history_sha256": sha(history_path),
            }
            validate_source_artifacts(
                payload, summary, source_history, role=role, source_spec=source_spec,
                checkpoint_file_sha256=sha(path), aligned_dataset=aligned[dataset["dataset"]],
            )
            validate_checkpoint_route(payload, binding["backbone"])
            source = build_source_model(payload, max_seq_len=dataset["max_seq_len"])
            normalized, _metadata, proof = build_frozen_lognormal_candidate(
                payload,
                # Arbitrary synthetic initializer for quantity identity only.
                # These are NOT fitted train-duration statistics or NLL results.
                train_time_statistics={"time_scale": 1., "target_log_scaled_mean": 0.,
                                       "target_log_scaled_std": 1.},
                time_sigma_floor=0.001, source_backbone=binding["backbone"],
                max_seq_len=dataset["max_seq_len"],
                training_stage="synthetic_source_compatibility_audit",
            )
            # The shared fit routine applies this freeze after construction.
            freeze_time_head_only(normalized)
            assert sum(p.numel() for p in normalized.parameters() if p.requires_grad) == 130
            source.eval()
            normalized.eval()
            dt = torch.tensor([[1., 2., 4., 3.], [1., 3., 2., 0.]])
            quantity = torch.tensor([[2., 5., 8., 11.], [3., 6., 9., 0.]])
            mask = torch.tensor([[True] * 4, [True] * 3 + [False]])
            with torch.no_grad():
                left = target_outputs(source, dt, mask, quantity, lambda_log_qty=1.)["pred_qty"]
                right = target_outputs(normalized, dt, mask, quantity, lambda_log_qty=1.)["pred_qty"]
            assert torch.equal(left, right)
            assert state_partition_sha256(source.state_dict(), time_head=False) == state_partition_sha256(
                normalized.state_dict(), time_head=False)
            rows.append({"dataset": dataset["dataset"], "role": role,
                         "checkpoint_relative_path": binding["checkpoint_path"],
                         "checkpoint_file_sha256": sha(path),
                         "checkpoint_state_sha256": canonical_state_dict_sha256(source.state_dict()),
                         "source_non_time_state_sha256": proof["source_non_time_state_sha256"],
                         "strict_source_restore": True, "source_dataset_and_training_contract_validated": True,
                         "source_history_sha256": sha(history_path),
                         "source_raw_rmse_selector_and_termination_replayed": True,
                         "synthetic_quantity_bitwise_identical": True,
                         "normalized_trainable_parameter_count": 130})
    result = {"status": "passed", "scope": "existing_checkpoint_synthetic_quantity_replay_only",
              "source_root": str(ROOT), "torch": torch.__version__, "device": "cpu",
              "audit_script_sha256": sha(Path(__file__)),
              "evaluation_runner_sha256": sha(ROOT / "paper/scripts/run_backbone_normalized_duration.py"),
              "new_benchmark_rows": False, "optimization_steps": 0,
              "time_performance_evaluated": False, "checks": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "checkpoint_pairs": len(rows),
                      "output": str(args.output)}))


if __name__ == "__main__":
    main()
