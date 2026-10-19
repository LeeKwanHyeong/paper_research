"""CPU/fabricated-only preparation check; no deployment or training entry point."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch

from models.TPPs.CountAwareTitanLateObservationPath import (
    ARMS, MODE_BY_ARM, CountAwareTitanLateObservationPath, metadata,
)


def fabricated_model(arm):
    if arm not in ARMS:
        raise ValueError("Unknown late-path arm")
    # Same interface as the existing MLP16 fabricated condition; no data loader.
    return CountAwareTitanLateObservationPath(
        hidden_dim=64, train_log_mean=1.2, train_log_std=.8, max_seq_len=256,
        late_path_mode=MODE_BY_ARM[arm],
        quantity_variant="count_only_log_regression",
        time_head_mode="heteroscedastic_lognormal_duration", time_scale=1.,
        time_initial_location=.2, time_initial_scale=.8,
        time_observation_contract={"mode": "positive_integer_round_clamp_v1",
                                   "unit": "hour", "top_code": None},
    )


def check_synthetic_integration():
    """Exercise common heads/target blocking using fabricated optimizer steps.

    The three objectives are compatibility checks, not an experimental choice.
    No fitted model or checkpoint is persisted and no research rows are read.
    """
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs
    from paper.scripts.run_titantpp_time_zero_mlp9 import TimeHeadGuard
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    dts = torch.tensor([[1., 2., 1., 3., 1.], [1., 1., 2., 1., 3.]])
    quantities = torch.tensor([[1., 3., 2., 4., 2.], [2., 4., 1., 3., 1.]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    conditions = []
    with torch.random.fork_rng(devices=[]):
        for arm in ARMS:
            for loss_mode in ("log_mse", "raw_mse", "raw_mae"):
                torch.manual_seed(42)
                model = fabricated_model(arm).train()
                guard = TimeHeadGuard(model)
                optimizer = guard.bind_optimizer(torch.optim.AdamW(
                    model.parameters(), lr=.001, weight_decay=.01))
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    value = target_outputs(model, dts, mask, quantities,
                                           lambda_log_qty=1.)
                    residual = value["pred_qty"] - value["true_qty"]
                    objective = (value["quantity_train_loss"] if loss_mode == "log_mse"
                                 else residual.square() if loss_mode == "raw_mse"
                                 else residual.abs())
                    loss = objective.mean()
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("Nonfinite fabricated objective")
                    loss.backward()
                    if not all(p.grad is None or bool(torch.isfinite(p.grad).all())
                               for p in model.parameters()):
                        raise ValueError("Nonfinite fabricated gradient")
                    if model.late_path.output_projection.weight.grad is None:
                        raise ValueError("New path is disconnected from common quantity head")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                proof = guard.proof()
                if not bool(model.late_path.output_projection.weight.detach().ne(0).any()):
                    raise ValueError("New path output never learned in fabricated check")
                model.eval()
                before = canonical_state_dict_sha256(model.state_dict())
                altered_dt, altered_q = dts.clone(), quantities.clone()
                altered_dt[:, -1] += 10.
                altered_q[:, -1] += 100.
                with torch.no_grad():
                    original = target_outputs(model, dts, mask, quantities,
                                              lambda_log_qty=1.)
                    altered = target_outputs(model, altered_dt, mask, altered_q,
                                             lambda_log_qty=1.)
                    single = target_outputs(model, dts[:1], mask[:1], quantities[:1],
                                            lambda_log_qty=1.)
                if not torch.equal(original["pred_qty"], altered["pred_qty"]):
                    raise ValueError("Future target leaked into prediction")
                torch.testing.assert_close(original["pred_qty"][:1], single["pred_qty"],
                                           rtol=1e-5, atol=1e-5)
                if canonical_state_dict_sha256(model.state_dict()) != before:
                    raise ValueError("Evaluation changed parameters or buffers")
                conditions.append({
                    "arm": arm, "loss_mode": loss_mode,
                    "fabricated_optimizer_steps": 2,
                    "target_blocking_and_batch_consistency_passed": True,
                    "evaluation_state_unchanged": True,
                    "new_path_quantity_gradient_and_update_verified": True,
                    "time_head_no_training_verification": proof,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                    "late_path_parameter_count": sum(p.numel() for p in model.late_path.parameters()),
                })
    return {
        "status": "passed_cpu_fabricated_integration",
        "device": "cpu", "condition_count": len(conditions), "conditions": conditions,
        "scientific_fit_count": 0, "native_gpu_verified": False,
        "real_data_loaded": False, "remote_calls": 0,
        "held_out_test_evaluated": False,
        "existing_models_or_queues_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New local result file; never overwrite an existing result")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing preparation evidence")
    torch.set_num_threads(1)
    result = check_synthetic_integration()
    root = Path(__file__).resolve().parents[2]
    spec = root / "reports/titantpp_mlp16_late_observation_path_20261019_v1/SCIENCE_SPEC.json"
    paths = [Path(__file__), root / "models/TPPs/CountAwareTitanLateObservationPath.py", spec]
    result.update(
        created_at=datetime.now(timezone.utc).isoformat(),
        model_metadata={arm: metadata(mode=MODE_BY_ARM[arm]) for arm in ARMS},
        source_bindings={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in paths},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "condition_count": result["condition_count"],
                      "output": str(args.output), "new_scientific_fits": 0}))


if __name__ == "__main__":
    main()
