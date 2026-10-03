"""Read-only research checks; write evidence only beside this script.

Run from the repository root with PYTHONDONTWRITEBYTECODE=1.
This is a validation-artifact and synthetic CPU check, not checkpoint replay,
held-out evaluation, or a comparison between language models.
"""

from __future__ import annotations

import csv
import hashlib
import json
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    started = time.perf_counter()
    source_paths: set[Path] = set()
    publication = runpy.run_path(str(ROOT / "paper/scripts/verify_v0_7_paper_artifacts.py"))
    verifier = publication["Verification"]()
    # Record the exact inputs without invoking either existing writer/main().
    def read_csv(path: Path):
        source_paths.add(path)
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    publication["reconstruct_validation_rows"].__globals__["read_csv"] = read_csv
    rows = publication["reconstruct_validation_rows"](verifier)
    for name, args in [
        ("verify_t3", (verifier, rows)),
        ("verify_t4", (verifier, rows)),
        ("verify_t5", (verifier,)),
        ("verify_publication_content", (verifier,)),
    ]:
        publication[name](*args)

    # Verify that the frozen Taxi/Instacart export still agrees with its
    # underlying validation-only run summaries, rather than only with itself.
    freeze = ROOT / "paper/results/titantpp_v0_7_validation_freeze_20260905"
    bundle = runpy.run_path(str(freeze / "independent_verify.py"))
    source_check = bundle["Verification"]()
    source_seed_rows, _, _ = bundle["load_validation_sources"](source_check)
    for check in source_check.checks:
        verifier.check(check["passed"], "upstream." + check["name"], check["detail"])
    for error in source_check.errors:
        verifier.check(False, "upstream.error", error)
    frozen_index = {(r["dataset"], r["model"], int(r["seed"])): r for r in rows}
    for row in source_seed_rows:
        key = (row["dataset"], row["model"], int(row["seed"]))
        for metric in publication["METRICS"]:
            verifier.numeric(row[metric], frozen_index[key][metric], f"upstream.{key}.{metric}")
    for spec in bundle["DATASETS"].values():
        source_paths.update(spec["root"] / name for name in (
            "launch_contract.json", "run_summaries.csv",
            "quantity_seed_metrics.csv", "history_seed_metrics.csv",
        ))

    table = {(r["dataset"], r["model"]): r for r in read_csv(ROOT / "paper/tables/T3_v0_7_backbone_validation.csv")}
    def improvement(dataset: str, baseline: str, metric: str) -> float:
        ref = float(table[(dataset, baseline)][metric + "_mean"])
        candidate = float(table[(dataset, "titantpp")][metric + "_mean"])
        return (ref - candidate) / abs(ref) * 100

    claims = {}
    for label, baseline, metric, expected in (
        ("intermittent_mae_vs_rmtpp", "rmtpp", "qty_mae", 74.3),
        ("intermittent_rmse_vs_rmtpp", "rmtpp", "qty_rmse", 81.9),
        ("intermittent_rmse_vs_thp", "thp", "qty_rmse", 10.8),
        ("intermittent_mae_vs_thp", "thp", "qty_mae", -12.1),
    ):
        claims[label] = improvement("intermittent", baseline, metric)
        verifier.numeric(claims[label], expected, "claim." + label, abs_tol=0.05, rel_tol=0)
    claims["taxi_rmse_vs_rmtpp"] = improvement("taxi", "rmtpp", "qty_rmse")
    verifier.numeric(claims["taxi_rmse_vs_rmtpp"], 0.57, "claim.taxi_rmse", abs_tol=0.005, rel_tol=0)
    for metric in publication["METRICS"]:
        values = {m: float(table[("instacart", m)][metric + "_mean"]) for m in publication["MODEL_ORDER"]}
        verifier.equal(min(values, key=values.get), "rmtpp", "claim.instacart_best." + metric)

    import torch
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    torch.manual_seed(42)
    model, metadata = build_count_aware_model("titantpp", hidden_dim=64, train_log_mean=1.0, max_seq_len=8)
    model.eval()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    dts = torch.tensor([[0., 1., 2., 3.]])
    qty = torch.tensor([[0., 2., 5., 7.]])
    mask = torch.tensor([[False, True, True, True]])
    changed = qty.clone()
    changed[:, -1] = 999.
    with torch.no_grad():
        original = target_outputs(model, dts, mask, qty, lambda_log_qty=1.0)
        alternate = target_outputs(model, dts, mask, changed, lambda_log_qty=1.0)
    verifier.check(torch.allclose(original["pred_qty"], alternate["pred_qty"], atol=1e-7), "code.titan_target_excluded")
    verifier.check(all(torch.equal(value, before[key]) for key, value in model.state_dict().items()), "code.eval_does_not_update_state")
    verifier.check(torch.isfinite(original["pred_qty"]).all().item(), "code.finite_quantity")
    verifier.check(torch.allclose(original["joint_loss"], original["time_loss"] + original["log_qty_loss"]), "code.joint_loss_contract")
    verifier.equal(type(model.lmm).__name__, "HardLocalMemoryMatcher", "code.original_static_matcher")
    verifier.equal((model.lmm.mem_size, model.lmm.topk), (64, 4), "code.memory_dimensions")
    verifier.equal(model.time_intercept_limit, 30., "code.current_cap_30_not_frozen_300")
    verifier.equal(hasattr(model, "mark_head"), False, "code.no_mark_head")

    source_paths.update(ROOT / p for p in (
        "README.md", "paper/titantpp_short_paper_draft_v0_7_manuscript.md",
        "paper/scripts/verify_v0_7_paper_artifacts.py",
        "paper/results/titantpp_v0_7_validation_freeze_20260905/independent_verify.py",
        "models/TPPs/CountAwareFactory.py", "models/TPPs/CountAwareTPP.py",
        "models/Titan/backbone.py", "models/Titan/common/memory.py",
        "paper/scripts/count_aware_tpp_backbone/core.py",
        "paper/scripts/count_aware_tpp_backbone/training.py",
        "paper/scripts/run_count_aware_tpp_backbone_control.py",
    ))
    report = verifier.report(None)
    report.pop("target_manifest", None)  # Full publication manifest was not checked.
    report.update({
        "audit_driver": str(Path(__file__).relative_to(ROOT)),
        "scope": "validation_only_tables_publication_and_synthetic_cpu_contract",
        "held_out_predictions_or_metrics_read": False,
        "checkpoint_replay_performed": False,
        "full_dataset_or_publication_manifest_verified": False,
        "upstream_validation_seed_rows_compared": len(source_seed_rows),
        "claim_improvement_percent": claims,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__, "device": "cpu"},
        "duration_seconds": round(time.perf_counter() - started, 3),
        "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_paths)},
    })
    (OUT / "research_verification.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "checks_total", "checks_passed", "numeric_comparisons", "duration_seconds", "failures")}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
