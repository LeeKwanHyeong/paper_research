#!/usr/bin/env python3
"""Independent event/metric reconciliation of the inference-only device audit."""

import argparse
from pathlib import Path
import sys

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts.run_hard_lmm_frozen_probe import load_json, save_json, sha256_file, verify_hashes
from paper.scripts.validate_hard_lmm_frozen_probe import finite_json


def paired_difference(a, b, route):
    for key in ("target_index", "series_index", "context_end", "quantity", "history_length"):
        if not torch.equal(a[key], b[key]):
            raise ValueError(f"Event alignment differs: {key}")
    pred = (a[f"{route}_prediction"].double() - b[f"{route}_prediction"].double()).abs()
    ai, bi = a[f"{route}_prototype_indices"], b[f"{route}_prototype_indices"]
    return {"events": len(pred), "prediction_max_abs_difference": float(pred.max()),
        "prediction_mean_abs_difference": float(pred.mean()), "predictions_not_bitwise_equal": int((pred > 0).sum()),
        "prototype_order_different_events": int((ai != bi).any(1).sum()),
        "prototype_set_different_events": int((ai.sort(1).values != bi.sort(1).values).any(1).sum()),
        "time_max_abs_difference": float((a[f"{route}_time_nll"] - b[f"{route}_time_nll"]).abs().max())}


def read_verified(root):
    digests = load_json(root / "output_digests.json")
    assert set(digests) == {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()
        and p.name not in ("output_digests.json", "probe_status.json")}
    verify_hashes(root, digests)
    manifest = load_json(root / "source_manifest.json")
    verify_hashes(ROOT, manifest["files"])
    summary = load_json(root / "summary.json")
    assert summary["status"] == load_json(root / "probe_status.json")["status"] == "complete"
    assert summary["base_unchanged"] and summary["training_steps"] == 0 and not summary["held_out_test_evaluated"]
    assert not list(root.rglob("*test*"))
    events = torch.load(root / "events.pt", map_location="cpu", weights_only=True)
    assert sha256_file(root / "events.pt") == summary["events_sha256"]
    assert all(torch.isfinite(v).all() for v in events.values())
    for route in ("official", "probe"):
        error = events[f"{route}_prediction"].numpy().astype(np.float64) - events["quantity"].numpy().astype(np.float64)
        metrics = {"qty_mae": float(np.abs(error).mean()), "qty_rmse": float(np.sqrt(np.square(error).mean())),
            "time_nll": float(events[f"{route}_time_nll"].numpy().astype(np.float64).mean()),
            "log_qty_mse": float(events[f"{route}_log_loss"].numpy().astype(np.float64).mean())}
        metrics["joint_objective"] = float(events["official_joint_loss"].double().mean()) if route == "official" else metrics["time_nll"] + metrics["log_qty_mse"]
        for key, value in metrics.items():
            np.testing.assert_allclose(value, summary["metrics"][route][key], atol=1e-12, rtol=1e-12)
    finite_json(summary)
    return events, summary, manifest


def main(root, output):
    if output.exists():
        raise FileExistsError(output)
    runs = {f"{device}_{phase}": read_verified(root / f"{device}_{phase}")
        for phase in ("train", "validation") for device in ("cpu", "cuda")}
    manifests = [r[2] for r in runs.values()]
    assert all(m == manifests[0] for m in manifests)
    reference_path = ROOT / "search_artifacts/hard_lmm_readout_seed_replication_20260903/seed_62/train_cache.pt"
    assert sha256_file(reference_path) == "887b65fe81bfe1e2cce2828026bb05651749607607718599f2cb35e9cf6194d5"
    reference = torch.load(reference_path, map_location="cpu", weights_only=True)
    cpu_train = runs["cpu_train"][0]
    for key in ("target_index", "series_index", "context_end", "quantity", "history_length"):
        assert torch.equal(cpu_train[key], reference[key][:256])
    assert torch.equal(cpu_train["probe_z"], reference["z"][:256])
    comparisons, rows = {}, []
    historical = load_json(ROOT / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/runs/titantpp/count_only_log_regression/seed_62/summary.json")
    for phase in ("train", "validation"):
        for route in ("official", "probe"):
            comparisons[f"{phase}_{route}_cpu_vs_cuda"] = paired_difference(runs[f"cpu_{phase}"][0], runs[f"cuda_{phase}"][0], route)
        for device in ("cpu", "cuda"):
            events, summary, _ = runs[f"{device}_{phase}"]
            assert summary["device"] == device and summary["phase"] == phase
            assert len(events["quantity"]) == summary["targets"] == (256 if phase == "train" else 86285)
            delta = (events["official_prediction"].double() - events["probe_prediction"].double()).abs()
            comparisons[f"{phase}_{device}_official_vs_probe"] = {"prediction_max_abs_difference": float(delta.max()),
                "prototype_set_different_events": int((events["official_prototype_indices"].sort(1).values != events["probe_prototype_indices"].sort(1).values).any(1).sum())}
            for route in ("official", "probe"):
                row = {"phase": phase, "device": device, "torch": summary["runtime"]["torch"], "route": route,
                    "targets": summary["targets"], **summary["metrics"][route]}
                if phase == "validation":
                    difference = {k: abs(row[k] - historical[f"best_val_{k}"]) for k in ("qty_mae", "qty_rmse", "time_nll", "joint_objective")}
                    expected = summary["reference_replay"][route]
                    assert difference == expected["absolute_differences"] and expected["tolerance"] == 1e-5
                    assert expected["all_pass"] == all(v <= 1e-5 for v in difference.values())
                    row.update({f"{k}_reference_abs_difference": v for k, v in difference.items()}, replay_pass=expected["all_pass"])
                rows.append(row)
    output.mkdir(parents=True)
    pl.DataFrame(rows, infer_schema_length=None).write_csv(output / "metric_comparison.csv")
    save_json(output / "event_comparison.json", comparisons)
    save_json(output / "verification.json", {"status": "inference_evidence_verified", "performance_candidate_evaluated": False,
        "training_steps": 0, "held_out_test_evaluated": False, "original_checkpoints_unchanged": True,
        "source_revision": manifests[0]["source_revision"], "all_metrics_independently_reconciled": True,
        "event_alignment_verified": True, "local_train_logits_match_prior_cache_exactly": True,
        "source_files_verified": len(manifests[0]["files"]), "reference_tolerance_unchanged": 1e-5,
        "runs": {name: summary for name, (_, summary, _) in runs.items()}})
    print(pl.DataFrame(rows, infer_schema_length=None))
    print(comparisons)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args.artifact, args.output)
