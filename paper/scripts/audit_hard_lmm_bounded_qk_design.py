"""Audit existing artifacts and a synthetic formula; never import/run a TPP model.

Run from the repository root. No dataset files are read, no model is trained,
and no new validation prediction is made. Checkpoints supply kernel tensors only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper/results/hard_lmm_bounded_qk_design_20260908"
INPUTS: dict[str, str] = {}


def pin(relative: str, expected: str | None = None) -> Path:
    path = ROOT / relative
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None:
        assert digest == expected, (relative, digest, expected)
    INPUTS[relative] = digest
    return path


def read(relative: str, expected: str | None = None):
    return json.loads(pin(relative, expected).read_text())


def aggregate(rows):
    n = sum(row["count"] for row in rows)
    return {
        "count": n,
        "mae": sum(r["count"] * r["qty_mae"] for r in rows) / n,
        "mse": sum(r["count"] * r["qty_rmse"] ** 2 for r in rows) / n,
        "bias": sum(r["count"] * r["qty_bias"] for r in rows) / n,
    }


def formula_checks():
    # A mathematical prototype only, not the proposed production backbone.
    torch.manual_seed(20260908)
    cases = []
    for dtype in (torch.float32, torch.float64):
        p = torch.randn(2, 5, 4, 16, dtype=dtype, requires_grad=True)
        r = torch.zeros_like(p, requires_grad=True)
        s = (p.square().mean(-1, keepdim=True) + 1e-8).sqrt()
        gain = s / (s.square() + r.square().mean(-1, keepdim=True)).sqrt()
        output = p + r * gain
        upstream = torch.randn_like(p)
        gp, gr = torch.autograd.grad((output * upstream).sum(), (p, r))
        assert torch.equal(output, p)
        assert torch.equal(gp, upstream)
        assert torch.equal(gr, upstream)
        max_ratio = 0.0
        for scale in (0.0, 1e-6, 1.0, 1e3, 1e6):
            a = (torch.randn(2, 5, 4, 16, dtype=dtype) * scale).requires_grad_()
            b = (torch.randn_like(a) * scale * 100).requires_grad_()
            s = (a.square().mean(-1, keepdim=True) + 1e-8).sqrt()
            gain = s / (s.square() + b.square().mean(-1, keepdim=True)).sqrt()
            bounded = b * gain
            ratio = bounded.square().mean(-1, keepdim=True).sqrt() / s
            max_ratio = max(max_ratio, ratio.max().item())
            grads = torch.autograd.grad((a + bounded).square().mean(), (a, b))
            assert all(torch.isfinite(x).all() for x in (bounded, *grads))
            assert ratio.max().item() <= 1.0 + 5e-7
        cases.append({"dtype": str(dtype), "zero_output_exact": True,
                      "zero_base_gradient_exact": True,
                      "zero_residual_gradient_exact": True,
                      "max_rms_ratio": max_ratio, "finite_checked": True})
    return {"scope": "standalone formula only; no model/CUDA/causality/resume certification",
            "seed": 20260908, "epsilon": 1e-8, "reduction": "per_event_per_attention_head_16_features",
            "rho": 1.0, "cases": cases}


def main():
    diagnostic = read("paper/contracts/hard_lmm_qkv_path_diagnostic_v1.json",
                      "a4ae310d254ed124fc8be4c340a10d22c462a33fb08d669afd38504cf978ddea")
    aligned = read("paper/contracts/aligned_causal_duration_scale_adapter_v1.json")
    read("paper/contracts/aligned_frozen_lognormal_duration_v1.json",
         "677cc3af91d84dfea8e3b4e71b38697fb467059f680678f0b8370a50f2366f16")
    for rel in (
        "paper/results/hard_lmm_qkv_path_diagnostic_20260908/legacy_time_score_audit.json",
        "paper/results/aligned_causal_duration_scale_adapter_seed42_20260907/validation_audit.json",
    ):
        pin(rel)
    results = {}
    for data in diagnostic["datasets"]:
        name = data["dataset"]
        summaries = {role: read(data[role]["summary_path"], data[role]["summary_sha256"])
                     for role in ("B", "candidate")}
        full_path = Path(data["candidate"]["summary_path"])
        history = read(str(full_path.with_name("history.json")))["history"]
        npz_path = pin(f"search_artifacts/hard_lmm_qkv_path_diagnostic_20260908_01f6c3a/{name}/extraction.npz")
        npz = np.load(npz_path, allow_pickle=False)
        checkpoint = torch.load(pin(data["candidate"]["checkpoint_path"],
                                    data["candidate"]["checkpoint_file_sha256"]),
                                map_location="cpu", weights_only=False)
        kernels = {}
        for channel in ("q", "k", "v"):
            kernel = checkpoint["model_state_dict"][f"encoder.layers.0.attn.causal_{channel}_kernel"].double()
            dc = kernel.sum(0)
            kernels[channel] = {"row_l2": kernel.norm(dim=1).tolist(),
                                "dc_sum_mean": dc.mean().item(),
                                "dc_sum_min": dc.min().item(), "dc_sum_max": dc.max().item()}
        train = {}
        quantity = npz["true_quantity"]
        p95, p99 = data["quantity_boundaries"][-2:]
        for label, mask in {"all": np.ones(len(quantity), bool), "body": quantity <= p95,
                            "p95_p99": (quantity > p95) & (quantity <= p99),
                            "gt_p99": quantity > p99}.items():
            train[label] = {"count": int(mask.sum()),
                            "series": len(np.unique(npz["series_id"][mask])),
                            "variants": {}}
            for variant in ("B", "FULL", "QK"):
                train[label]["variants"][variant] = {
                    "mae": float(npz[f"{variant}__absolute_error"][mask].mean()),
                    "rmse": float(np.sqrt(npz[f"{variant}__squared_error"][mask].mean())),
                    "legacy_time_loss": float(npz[f"{variant}__legacy_time_loss"][mask].mean()),
                }
        validation = {}
        group_map = {"body": {"le_p50", "p50_p90", "p90_p95"},
                     "p95_p99": {"p95_p99"}, "gt_p99": {"gt_p99"}}
        total = sum(r["count"] for r in summaries["B"]["quantity_rows"])
        for label, strata in group_map.items():
            grouped = {role: aggregate([r for r in summary["quantity_rows"] if r["stratum"] in strata])
                       for role, summary in summaries.items()}
            assert grouped["B"]["count"] == grouped["candidate"]["count"]
            grouped["contribution_to_overall_mse_change"] = (
                grouped["B"]["count"] / total * (grouped["candidate"]["mse"] - grouped["B"]["mse"]))
            validation[label] = grouped
        total_delta = (summaries["candidate"]["best_val_qty_rmse"] ** 2
                       - summaries["B"]["best_val_qty_rmse"] ** 2)
        assert abs(sum(r["contribution_to_overall_mse_change"] for r in validation.values()) - total_delta) < 1e-8
        selected = summaries["candidate"]["best_epoch"]
        epochs = {"first": history[0], "selected": next(r for r in history if r["epoch"] == selected),
                  "last": history[-1], "minimum_legacy_time": min(history, key=lambda r: r["val_time_nll"])}
        if name == "yellow_trip_hourly":
            time_train = {}
            for variant in ("B", "FULL"):
                loss = npz[f"{variant}__legacy_time_loss"]
                time_train[variant] = {
                    "mean_legacy_loss": float(loss.mean()),
                    "wd_saturated_count": int(npz[f"{variant}__wd_saturated"].sum()),
                    "intercept_saturated_count": int(npz[f"{variant}__intercept_saturated"].sum()),
                    "intercept_min": float(npz[f"{variant}__time_intercept_raw"].min()),
                    "intercept_max": float(npz[f"{variant}__time_intercept_raw"].max()),
                    "max_exp_intercept": float(npz[f"{variant}__exp_intercept"].max()),
                }
        else:
            time_train = None
        results[name] = {"validation_selected": {
            role: {key: summary[key] for key in ("best_epoch", "best_val_qty_mae", "best_val_qty_rmse", "best_val_time_nll")}
            for role, summary in summaries.items()}, "existing_train_groups": train,
            "existing_validation_groups": validation, "overall_validation_mse_change": total_delta,
            "full_history": epochs, "full_kernel_statistics": kernels, "taxi_existing_train_time": time_train}
    output = {"status": "existing_evidence_and_synthetic_math_audit_only",
              "new_model_forward": False, "new_dataset_rows": False, "training": False,
              "baseline_commit": "f0cc1fd", "runtime": {"torch": torch.__version__, "device": "cpu"},
              "datasets": results,
              "aligned_B_reuse": [{key: value for key, value in d.items()
                                    if key == "dataset" or key.startswith("aligned_B_")}
                                   for d in aligned["datasets"]],
              "formula_checks": formula_checks(), "input_sha256": INPUTS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evidence_audit.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"result": str(OUT / "evidence_audit.json"), "pinned_inputs": len(INPUTS),
                      "formula_checks": output["formula_checks"]}, indent=2))


if __name__ == "__main__":
    main()
