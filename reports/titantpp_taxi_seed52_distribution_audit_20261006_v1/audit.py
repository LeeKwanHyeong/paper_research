"""Local aggregate-only Taxi seed audit; no torch, inference, or remote calls.

Run from any directory with /usr/local/bin/python3 audit.py.
Only this report directory is written. Original evidence is never changed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
BUNDLE = ROOT / "search_artifacts/titantpp_cnn_gru54_test_3seed_20261006_v1"
SEEDS = (42, 52, 62)
DATASET = "yellow_trip_hourly"
BOUNDARIES = (7.0, 686.0, 1562.0, 3449.0)
BIN_LABELS = ("1–7", ">7–686", ">686–1562", ">1562–3449", ">3449")
BINDINGS = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bound(path, expected=None):
    actual = sha(path)
    require(expected is None or actual == expected, f"SHA mismatch: {path.name}")
    BINDINGS[str(path.relative_to(ROOT))] = actual
    return path


def read(path, expected=None):
    return json.loads(bound(path, expected).read_text())


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                      allow_nan=False) + "\n")


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def profile(values, prefix):
    names = ("min", "median", "q90", "q95", "q99", "max")
    return {f"{prefix}_{k}": float(v) for k, v in
            zip(names, np.quantile(values, [0, .5, .9, .95, .99, 1]))} | {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_population_sd": float(values.std(ddof=0)),
    }


def main():
    contract = read(BUNDLE / "execution_contract.json",
                    "a66e0fb42cb70a9f188288a358adba3e0c040bc5c1b4087787f70a9e7b00e56b")
    registry = read(BUNDLE / "evaluation_registry.json", contract["registry_sha256"])
    read(BUNDLE / "dataset_manifest.json", contract["dataset_manifest_sha256"])
    bound(BUNDLE / "operation/prior_evaluate.py", contract["evaluator_sha256"])
    bound(BUNDLE / "operation/evaluate_titantpp_cnn_gru_frozen_test.py", contract["operation_adapter_sha256"])
    retrieval = read(BUNDLE / "retrieved/retrieval_receipt.json")
    inputs, seed_rows, histories, exposures = {}, [], [], {}
    for seed in SEEDS:
        row, = [r for r in registry["rows"] if r["dataset"] == DATASET and r["seed"] == seed]
        job = BUNDLE / "original/run" / f"{DATASET}__{seed}__titantpp_cnn_gru54"
        terminal = read(BUNDLE / row["original_binding"]["terminal_manifest_path"],
                        row["original_binding"]["terminal_manifest_sha256"])
        require(terminal["scientific_success"] is True, "Original terminal is not scientific success")
        inputs[seed] = read(job / "input_receipt.json", terminal["files"]["input_receipt.json"])
        init = read(job / "initialization.json", terminal["files"]["initialization.json"])
        require(init["titantpp_cnn_gru54"] == row["initial_state_sha256"], "Initialization binding drift")
        prefix = f"runs/titantpp_cnn_gru54/count_only_log_regression/seed_{seed}/"
        exposures[seed] = read(job / (prefix+"exposure.json"), terminal["files"][prefix+"exposure.json"])
        require(all(r["count"] == 38393 and r["batches"] == 300 for r in exposures[seed]["train"]),
                "Train exposure population drift")
        require(all(r["count"] == 8268 and r["batches"] == 65 for r in exposures[seed]["validation"]),
                "Validation exposure population drift")
        suffix = f"runs/titantpp_cnn_gru54/count_only_log_regression/seed_{seed}/history.json"
        history = read(job / suffix, terminal["files"][suffix])["history"]
        require([r["epoch"] for r in history] == list(range(1, len(history)+1)), "History discontinuity")
        require(all(r["train_all_finite"] is True for r in history), "Recorded nonfinite training")
        require(all(math.isfinite(v) for r in history for v in r.values()
                    if isinstance(v, (int, float))), "Nonfinite value in history")
        selected = min(history, key=lambda r: r["val_qty_rmse"])
        require(selected["epoch"] == row["selected_epoch"], "Selected epoch differs from first Validation minimum")
        seed_rows.append({"seed": seed, "selected_epoch": row["selected_epoch"],
                          "saved_epoch": history[-1]["epoch"],
                          "data_sha256": inputs[seed]["input_identity"]["data_sha256"],
                          "split_manifest_sha256": inputs[seed]["input_identity"]["split_manifest_sha256"],
                          "train_targets": inputs[seed]["populations"]["train"]["target_count"],
                          "validation_targets": inputs[seed]["populations"]["validation"]["target_count"],
                          "initial_state_sha256": row["initial_state_sha256"],
                          "first_train_batch_digest": exposures[seed]["train"][0]["batch_order_sha256"]})
        for h in history:
            histories.append({"seed": seed} | {k: h[k] for k in
                ("epoch", "train_time_nll", "train_quantity_loss", "val_time_nll", "val_qty_rmse", "val_qty_mae",
                 "train_gradient_clip_fraction", "train_pre_clip_grad_norm_max", "train_max_per_event_time_nll")})
    identity_keys = ("input_identity", "all_train_rows", "populations", "raw_rms", "raw_scale",
                     "train_log_mean", "train_log_std")
    require(all(inputs[seed][k] == inputs[42][k] for seed in SEEDS for k in identity_keys),
            "Seed-specific input population or normalization differs")
    require(len({r["batch_order_sha256"] for seed in SEEDS for r in exposures[seed]["validation"]}) == 1,
            "Validation batch contents/order differs")
    require(len({r["first_train_batch_digest"] for r in seed_rows}) == len(SEEDS),
            "First Train batch digests do not differ across seeds")
    require(len({r["initial_state_sha256"] for r in seed_rows}) == len(SEEDS),
            "Initial model states do not differ across seeds")

    data = BUNDLE / "data/yellow_trip_hourly/yellow_trip_hourly_with_split.parquet"
    bound(data, inputs[42]["input_identity"]["data_sha256"])
    data_sha = BINDINGS[str(data.relative_to(ROOT))]
    bound(data.with_name("yellow_trip_hourly_split_manifest.json"),
          inputs[42]["input_identity"]["split_manifest_sha256"])
    frame = pl.read_parquet(data).sort(["oper_part_no", "seq"]).with_columns(
        (pl.col("seq").cum_count().over("oper_part_no")-1).alias("entity_position"))
    null_counts = frame.null_count().to_dicts()[0]
    duplicate_keys = frame.height - frame.select(["oper_part_no", "seq"]).unique().height
    split_rank = {"train": 0, "validation": 1, "test": 2}
    checks = {"null_cells": sum(null_counts.values()), "duplicate_entity_seq_rows": duplicate_keys,
              "nonfinite_or_negative_raw_quantities": int((~np.isfinite(frame["demand_qty"].to_numpy()) |
                                                          (frame["demand_qty"].to_numpy() < 0)).sum()),
              "negative_raw_gaps": int((frame["delta_t"].to_numpy() < 0).sum()),
              "nonpositive_target_gaps": frame.filter((pl.col("entity_position") > 0) &
                                                     (pl.col("delta_t") <= 0)).height,
              "initial_zero_gap_rows": frame.filter((pl.col("entity_position") == 0) &
                                                    (pl.col("delta_t") == 0)).height,
              "split_reversals_within_entity": 0, "nonincreasing_time_pairs": 0,
              "gap_vs_timestamp_mismatch_targets": 0, "gap_vs_seq_mismatch_targets": 0,
              "first_entity_row_not_train": 0, "nonconsecutive_seq_pairs": 0}
    parts = frame.partition_by("oper_part_no", maintain_order=True)
    for g in parts:
        splits = g["chronological_split"].to_list()
        checks["split_reversals_within_entity"] += sum(
            split_rank[splits[i]] < split_rank[splits[i-1]] for i in range(1, len(splits)))
        checks["first_entity_row_not_train"] += splits[0] != "train"
        hours = np.diff(g["time_bucket"].cast(pl.Int64).to_numpy()) / 3_600_000_000
        gaps = g["delta_t"].to_numpy()[1:]
        seq_diff = np.diff(g["seq"].to_numpy())
        checks["nonincreasing_time_pairs"] += int((hours <= 0).sum())
        checks["gap_vs_timestamp_mismatch_targets"] += int((hours != gaps).sum())
        checks["gap_vs_seq_mismatch_targets"] += int((seq_diff != gaps).sum())
        checks["nonconsecutive_seq_pairs"] += int((seq_diff != 1).sum())
    # seq gaps are real elapsed hours, not missing loader targets: both timestamp and delta_t agree.
    require(checks["initial_zero_gap_rows"] == len(parts), "Unexpected initial-row gap encoding")
    require(all(v == 0 for k, v in checks.items() if k not in
                ("nonconsecutive_seq_pairs", "initial_zero_gap_rows")),
            "Raw data quality check failed; review before interpreting seed effects")
    train_max = frame.filter(pl.col("chronological_split") == "train")["demand_qty"].max()
    distributions, target_bins, receipts, metrics = [], {}, {}, []
    for split in ("train", "validation", "test"):
        raw = frame.filter(pl.col("chronological_split") == split)
        target = raw.filter(pl.col("entity_position") > 0)
        quantity = target["demand_qty"].to_numpy()
        gaps = target["delta_t"].to_numpy()
        require(np.isfinite(quantity).all() and (quantity >= 0).all(), "Invalid target quantities")
        require(np.isfinite(gaps).all() and (gaps > 0).all(), "Invalid target gaps")
        bins = np.searchsorted(BOUNDARIES, quantity, side="left")
        target_bins[split] = [int((bins == i).sum()) for i in range(5)]
        distributions.append({"split": split, "raw_rows": raw.height, "targets": target.height,
                              "entities": target["oper_part_no"].n_unique(),
                              "first_recorded_date": int(target["demand_dt"].min()),
                              "last_recorded_date": int(target["demand_dt"].max()),
                              **profile(quantity, "quantity"), **profile(gaps, "gap_hours"),
                              "tail_gt3449": int((quantity > 3449).sum()),
                              "tail_percent": float(100*(quantity > 3449).mean()),
                              "above_train_max": int((quantity > train_max).sum())})
        if split in ("train", "validation"):
            parts_names = [str(g["oper_part_no"][0]) for g in parts]
            part_index = dict(zip(parts_names, range(len(parts_names))))
            native_id = hashlib.sha256(b"hard_lmm_target_identity_v1\0" + split.encode()+b"\0")
            native_id.update(json.dumps(parts_names, ensure_ascii=False, separators=(",", ":")).encode())
            for values in (np.array([part_index[p] for p in target["oper_part_no"].to_list()]),
                           target["entity_position"].to_numpy(), target["seq"].to_numpy()):
                native_id.update(values.astype("<i8", copy=False).tobytes())
            native_qty = hashlib.sha256(b"hard_lmm_target_quantity_v1\0")
            native_qty.update(quantity.astype(np.float32).astype("<f8").tobytes())
            require(native_id.hexdigest() == inputs[42]["populations"][split]["target_identity_sha256"],
                    "Raw-to-training identity mismatch")
            require(native_qty.hexdigest() == inputs[42]["populations"][split]["target_quantity_sha256"],
                    "Raw-to-training quantity mismatch")
        if split == "train":
            require(target.height == inputs[42]["populations"][split]["target_count"], "Train count drift")
            continue
        for seed in SEEDS:
            rel = f"runs/{split}/{DATASET}__titantpp_cnn_gru54__seed{seed}/receipt.json"
            r = read(BUNDLE / "retrieved" / rel, retrieval["files_SHA"][rel])
            require(r["status"] == "complete" and r["full_population"] is True, "Incomplete evaluation")
            require(r["registry_sha256"] == contract["registry_sha256"], "Evaluation registry drift")
            require(r["dataset_manifest_sha256"] == contract["dataset_manifest_sha256"], "Evaluation manifest drift")
            require(r["contract_sha256"] == BINDINGS[str((BUNDLE / "execution_contract.json").relative_to(ROOT))],
                    "Evaluation contract drift")
            require(r["data_file_sha256"] == data_sha, "Evaluation data drift")
            m = r["metrics"]
            require(m["count"] == target.height, "Evaluation population count drift")
            require(m["quantity_boundaries"] == list(BOUNDARIES), "Quantity boundary drift")
            require([c["count"] for c in m["quantity_cells"]] == target_bins[split], "Bin population drift")
            require(math.isclose(sum(c["qty_sse"] for c in m["quantity_cells"]), m["qty_sse"], rel_tol=1e-12),
                    "SSE decomposition mismatch")
            require(math.isclose(math.sqrt(m["qty_sse"]/m["count"]), m["qty_rmse"], rel_tol=1e-12),
                    "RMSE calculation mismatch")
            receipts[split, seed] = r
            for name, mm in [("overall", m)] + [(BIN_LABELS[c["bin"]], c) for c in m["quantity_cells"]]:
                metrics.append({"split": split, "seed": seed, "quantity_bin": name,
                                **{k: mm[k] for k in ("count", "qty_sse", "qty_rmse", "qty_mae", "qty_bias", "time_nll")}})
        require(len({receipts[split,s]["target_identity_sha256"] for s in SEEDS}) == 1,
                "Different evaluation target identities")
        require(len({receipts[split,s]["truth_sha256"] for s in SEEDS}) == 1, "Different evaluation truth")
        # Independently reconstruct evaluator target identity without importing torch or its loader.
        identity = hashlib.sha256()
        for row in target.iter_rows(named=True):
            payload = [DATASET, data_sha, row["oper_part_no"], int(row["seq"]), split, str(row["demand_dt"])]
            target_id = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":"),
                                                   allow_nan=False).encode()).hexdigest()
            identity.update((target_id + "\n").encode())
        require(identity.hexdigest() == receipts[split,42]["target_identity_sha256"], "Raw-to-evaluator identity mismatch")

    sse52 = receipts["test",52]["metrics"]["qty_sse"]
    reference_sse = sum(receipts["test",s]["metrics"]["qty_sse"] for s in (42,62))/2
    extra = sse52-reference_sse
    contributions = []
    for i in range(5):
        c52 = receipts["test",52]["metrics"]["quantity_cells"][i]
        ref = sum(receipts["test",s]["metrics"]["quantity_cells"][i]["qty_sse"] for s in (42,62))/2
        contributions.append({"quantity_bin": BIN_LABELS[i], "count": c52["count"],
                              "seed52_sse": c52["qty_sse"], "seed42_62_mean_sse": ref,
                              "extra_sse": c52["qty_sse"]-ref,
                              "extra_sse_share_percent": 100*(c52["qty_sse"]-ref)/extra})
    summary = {"inputs_identical_across_seeds": True, "raw_rows": frame.height,
               "quantity_boundaries": list(BOUNDARIES), "quality_checks": checks,
               "seed52_test_extra_sse_vs_seed42_62_mean": extra,
               "extra_sse_share_gt686_percent": sum(c["extra_sse_share_percent"] for c in contributions[2:]),
               "target_bin_counts": target_bins,
               "fresh_evaluation_truth_identical_across_seeds": True,
               "raw_evaluation_target_identity_reconstructed": True,
               "raw_training_target_identity_and_quantity_reconstructed": True,
               "validation_batch_digest_identical_across_all_epochs_and_seeds": True,
               "first_train_batch_digest_differs_across_seeds": True,
               "recorded_history_epochs": len(histories),
               "inference_run": False, "training_run": False, "checkpoint_loaded": False,
               "raw_predictions_written": False,
               "limitation": "Aggregate bins cannot identify a single erroneous prediction or establish initialization/order/selector causality."}
    write_csv("data_distribution.csv", distributions)
    write_csv("seed_inputs.csv", seed_rows)
    write_csv("seed_metrics.csv", metrics)
    write_csv("test_sse_decomposition.csv", contributions)
    write_csv("training_history.csv", histories)
    write_json("audit_result.json", summary)
    plot(distributions, metrics, histories)
    write_notebook()
    write_sources(distributions, metrics, summary)
    write_json("analysis_receipt.json", {"schema_version": 1,
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Taxi frozen raw input and existing Train/Validation/Test aggregate evidence; no new evaluation",
        "source_files_sha256": BINDINGS,
        "analysis_script_sha256": sha(Path(__file__)),
        "outputs_sha256": {p.name: sha(p) for p in OUT.iterdir()
                           if p.is_file() and p.name != "analysis_receipt.json"},
        "checks": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def plot(distributions, metrics, histories):
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="taxi-seed-audit-mpl-"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout="constrained")
    bins = np.arange(5)
    colors = {42: "#3870b5", 52: "#cf5928", 62: "#2c8a71"}
    for offset, seed in enumerate(SEEDS):
        rows = [r for r in metrics if r["split"] == "test" and r["seed"] == seed and r["quantity_bin"] != "overall"]
        axes[0].bar(bins+(offset-1)*.24, [r["qty_rmse"] for r in rows], .24, color=colors[seed], label=f"Seed {seed}")
        h = [r for r in histories if r["seed"] == seed]
        axes[1].plot([r["epoch"] for r in h], [r["val_qty_rmse"] for r in h], color=colors[seed], label=f"Seed {seed}")
        axes[2].bar(bins+(offset-1)*.24, [r["qty_bias"] for r in rows], .24, color=colors[seed])
    axes[0].set(title="Test RMSE on identical targets", ylabel="Quantity RMSE", xticks=bins, xticklabels=BIN_LABELS)
    axes[1].set(title="Original full Validation history", ylabel="Quantity RMSE", xlabel="Epoch")
    axes[2].set(title="Test signed bias on identical targets", ylabel="Prediction minus target", xticks=bins, xticklabels=BIN_LABELS)
    axes[2].axhline(0, color="#777", linewidth=.7)
    for ax in axes:
        ax.grid(axis="y", alpha=.2)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Taxi CNN+GRU54: identical data, different fitted predictions")
    fig.savefig(OUT / "taxi_seed_comparison.png", dpi=180)
    plt.close(fig)


def write_notebook():
    cells = [{"id": "scope", "cell_type": "markdown", "metadata": {}, "source": [
        "# Taxi seed52 distribution audit\n",
        "Reproduces the local aggregate-only audit. No training, model inference, or remote calls.\n",
        "Train/Validation/Test remain separate; seed52 is retained in every comparison.\n"]},
        {"id": "reproduce-audit", "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "from pathlib import Path\n", "import runpy\n",
            "audit = Path.cwd() / 'audit.py'\n",
            "if not audit.is_file():\n",
            "    audit = Path.cwd() / 'reports/titantpp_taxi_seed52_distribution_audit_20261006_v1/audit.py'\n",
            "runpy.run_path(str(audit), run_name='__main__')\n"]}]
    write_json("audit.ipynb", {"cells": cells, "metadata": {"kernelspec": {
        "display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5})


def write_sources(distributions, metrics, summary):
    preview = [{k: r[k] for k in ("split", "targets", "quantity_median", "quantity_max", "tail_gt3449")}
               for r in distributions]
    errors = [{k:r[k] for k in ("split", "seed", "quantity_bin", "count", "qty_rmse", "qty_bias")}
              for r in metrics if r["quantity_bin"] == "overall" or
              (r["split"] == "test" and r["quantity_bin"] == ">3449")]
    write_json("inline_sources.json", {"schemaVersion": 1, "items": [
        {"id": "taxi-identical-inputs", "title": "Taxi 세 seed의 데이터와 평가 대상은 동일합니다", "queries": [{
            "id": "frozen-input-profile", "source": {"label": "Taxi 동결 데이터·원본 입력 영수증",
                "files": ["yellow_trip_hourly_with_split.parquet", "input_receipt.json", "data_distribution.csv"],
                "filters": ["Dataset: Taxi", "Seeds: 42, 52, 62", "기존 Train·Validation·Test 분할별 비교"],
                "caveats": ["Test는 기존에 노출된 분할의 보존 결과이며 새 독립 평가가 아닙니다."],
                "evidenceFlow": [{"kind": "validation", "title": "파일과 모집단 대조", "body":
                    "Parquet·split manifest·원본 입력 영수증의 SHA를 검산했습니다. 세 seed의 Train/Validation 모집단·정규화값과 fresh Validation/Test target·truth SHA가 동일하며 raw 대상 identity 재구성도 일치했습니다."}]},
            "rows": preview, "columns": list(preview[0]), "preview": {"kind":"aggregate", "note":"세 seed에 공통인 분할별 수량 분포입니다."}}]},
        {"id": "taxi-seed52-error", "title": "Seed52의 Test 손해는 큰 수량 과소예측에 집중됐습니다", "queries": [{
            "id": "same-target-error-decomposition", "source": {"label":"동결 checkpoint의 기존 Test 집계 영수증",
                "files": ["receipt.json", "seed_metrics.csv", "test_sse_decomposition.csv"],
                "filters": ["Dataset: Taxi", "Validation과 Test를 구분해 집계", "원래 Validation 수량 RMSE 선택 checkpoint"],
                "caveats": ["수량 구간 집계만으로 개별 이상 예측이나 초기화·학습 순서·선택 기준의 인과관계를 확정하지 않습니다."],
                "evidenceFlow": [{"kind":"calculation", "title":"오차 증가분 분해", "body":
                    f"Seed52와 seed42·62 평균의 Test 제곱오차 차이 중 수량 686 초과 구간이 {summary['extra_sse_share_gt686_percent']:.4f}%를 차지합니다. 구간별 SSE 합과 전체 RMSE를 재계산해 대조했습니다."}]},
            "summary":"평균 편향은 예측 수량에서 실제 수량을 뺀 평균이며 음수는 과소예측입니다.",
            "rows": errors, "columns": list(errors[0]), "preview":{"kind":"aggregate", "note":"Validation/Test 전체와 Test 큰 수량의 집계이며 예측 행은 포함하지 않습니다."}}]}
    ]})


if __name__ == "__main__":
    main()
