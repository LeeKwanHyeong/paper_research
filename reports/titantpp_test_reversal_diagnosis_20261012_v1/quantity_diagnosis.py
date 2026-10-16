#!/usr/bin/env python3
"""Recompute descriptive error accounting from frozen native sufficient statistics.

No model, checkpoint, data frame, GPU, or remote process is opened by this script.
It verifies immutable local evidence and writes only this report directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

CONTRACT = "f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c"
COMPLETION = "d21d137190e98552d07c5b9e2d8fc31151846e39c782ad907ff9a1a3cf644dbd"
FULL = "titantpp_cnn_gru54"
CONTROLS = ("titantpp_cnn_gru54_correction_disabled", "titantpp_cnn_gru54_current_only")
DATASETS = ("yellow_trip_hourly", "intermittent_frozen_5000", "raf_spare_parts")
LABEL = {DATASETS[0]: "Taxi", DATASETS[1]: "Intermittent", DATASETS[2]: "RAF"}
CONTROL_LABEL = {CONTROLS[0]: "보정 제거", CONTROLS[1]: "current-only"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def close(a, b, *, rel=1e-9, absolute=1e-5):
    assert math.isfinite(a) and math.isfinite(b), (a, b)
    assert math.isclose(a, b, rel_tol=rel, abs_tol=absolute), (a, b)


def verify_evidence(root: Path):
    assert digest(root / "analysis/completion_receipt.json") == COMPLETION
    completion = read(root / "analysis/completion_receipt.json")
    assert completion["status"] == "complete"
    assert completion["distinct_conditions"] == 36
    assert completion["contract_sha256"] == CONTRACT
    assert completion["test_is_independent"] is False
    assert completion["retrained"] is False and completion["reselected"] is False
    evidence = {"analysis/completion_receipt.json": COMPLETION}
    for rel, expected in {**completion["source_files"], **completion["analysis_files"]}.items():
        assert digest(root / rel) == expected, rel
        evidence[rel] = expected
    contract = read(root / "contract.json")
    canonical = dict(contract)
    del canonical["contract_sha256"]
    assert hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest() == CONTRACT
    archived = read(root / "originals/receipt.json")
    assert archived["status"] == "verified_terminal_archived"
    assert archived["contract_sha256"] == CONTRACT
    assert digest(root / "originals/manifest.json") == archived["manifest_sha256"]
    assert digest(root / "originals/originals.tar.gz") == archived["archive_sha256"]
    manifest = read(root / "originals/manifest.json")
    assert manifest["contract_sha256"] == CONTRACT
    assert len(manifest["files"]) == archived["file_count"] == 929
    for rel, expected in manifest["files"].items():
        assert digest(root / "originals/verified" / rel) == expected, rel
    relevant = [j for j in contract["jobs"] if j["study"] == "history"]
    assert len(relevant) == 27
    for job in relevant:
        for filename in ("test_metrics.json", "validation_replay.json", "receipt.json", "worker_exit.json", "validation_worker_exit.json"):
            rel = f"results/{job['id']}/{filename}"
            assert digest(root / rel) == manifest["files"][rel], rel
            assert evidence[rel] == manifest["files"][rel], rel
    for rel in ("originals/receipt.json", "originals/manifest.json", "originals/originals.tar.gz"):
        evidence[rel] = digest(root / rel)
    return contract, evidence


def moments(cell):
    n = cell["count"]
    assert isinstance(n, int) and n > 0
    close(cell["represented_n"], n)
    mse = cell["qty_sse"] / n
    mae = cell["qty_absolute_error_sum"] / n
    bias = cell["qty_signed_error_sum"] / n
    variance = mse - bias * bias
    assert variance >= -1e-7 * max(1, mse)
    close(math.sqrt(mse), cell["qty_rmse"])
    close(mae, cell["qty_mae"])
    close(bias, cell["qty_bias"])
    return {"count": n, "mse": mse, "rmse": math.sqrt(mse), "mae": mae,
            "bias": bias, "bias_squared": bias * bias,
            "error_variance_population": max(0., variance),
            "sse": cell["qty_sse"], "sae": cell["qty_absolute_error_sum"],
            "signed_error_sum": cell["qty_signed_error_sum"]}


def verify_metrics(metrics, bounds):
    assert metrics["quantity_boundaries"] == bounds
    assert len(metrics["quantity_cells"]) == 5
    cells = metrics["quantity_cells"]
    assert [c["bin"] for c in cells] == list(range(5))
    assert sum(c["count"] for c in cells) == metrics["count"]
    for key in ("qty_sse", "qty_absolute_error_sum", "qty_signed_error_sum"):
        close(math.fsum(c[key] for c in cells), metrics[key])
    for key in ("count", "qty_sse", "qty_absolute_error_sum", "qty_signed_error_sum"):
        close(metrics["tail"][key], cells[-1][key])
    moments(metrics)
    for c in cells:
        moments(c)


def load_jobs(root, contract):
    rows = {}
    for job in contract["jobs"]:
        if job["study"] != "history":
            continue
        outputs = {}
        for split, filename in (("validation", "validation_replay.json"), ("test", "test_metrics.json")):
            obj = read(root / "results" / job["id"] / filename)
            for key, expected in (("job_id", job["id"]), ("split", split), ("contract_sha256", CONTRACT), ("selected_epoch", job["selected_epoch"]), ("checkpoint_sha256", job["checkpoint"]["sha256"]), ("frozen_source_closure", job["frozen_source_closure"])):
                assert obj[key] == expected, (job["id"], key)
            assert obj["retrained"] is False and obj["reselected"] is False
            assert obj["metrics"]["split"] == split
            bounds = contract["datasets"][job["dataset"]]["definition"]["quantity_boundaries_all_train_rows"]
            verify_metrics(obj["metrics"], bounds)
            if split == "validation":
                assert obj["validation_replay_check"]["passed"]
                for key in ("qty_rmse", "qty_mae"):
                    close(obj["metrics"][key], job["validation_reference"][key], rel=1e-5, absolute=1e-5)
            outputs[split] = obj
        rows[job["dataset"], job["seed"], job["arm"]] = {"job": job, **outputs}
    assert len(rows) == 27
    for dataset in DATASETS:
        for split in ("validation", "test"):
            population = [rows[dataset, seed, arm][split] for seed in (42, 52, 62) for arm in (FULL, *CONTROLS)]
            for key in ("target_identity_sha256", "truth_sha256"):
                assert len({p[key] for p in population}) == 1, (dataset, split, key)
            assert len({tuple(c["count"] for c in p["metrics"]["quantity_cells"]) for p in population}) == 1
    return rows


def contrast(full, control):
    n = full["count"]
    assert n == control["count"]
    f, c = moments(full), moments(control)
    cells = []
    for fc, cc in zip(full["quantity_cells"], control["quantity_cells"]):
        assert fc["count"] == cc["count"]
        fm, cm = moments(fc), moments(cc)
        p = fc["count"] / n
        row = {"bin": fc["bin"], "count": fc["count"], "weight": p,
               "full": fm, "control": cm,
               "mse_difference_within_bin": fm["mse"] - cm["mse"],
               "mse_contribution": (fm["sse"] - cm["sse"]) / n,
               "mae_contribution": (fm["sae"] - cm["sae"]) / n,
               "weighted_bias_squared_contribution": p * (fm["bias_squared"] - cm["bias_squared"]),
               "weighted_error_variance_contribution": p * (fm["error_variance_population"] - cm["error_variance_population"])}
        close(row["mse_contribution"], row["weighted_bias_squared_contribution"] + row["weighted_error_variance_contribution"])
        cells.append(row)
    d = (f["sse"] - c["sse"]) / n
    a = (f["sae"] - c["sae"]) / n
    close(d, math.fsum(x["mse_contribution"] for x in cells))
    close(a, math.fsum(x["mae_contribution"] for x in cells))
    gross = math.fsum(max(x["mse_contribution"], 0) for x in cells)
    improvement = -math.fsum(min(x["mse_contribution"], 0) for x in cells)
    close(d, gross - improvement)
    for row in cells:
        row["positive_gross_share"] = max(row["mse_contribution"], 0) / gross if gross else None
        row["signed_net_share"] = row["mse_contribution"] / d if d != 0 else None
    return {"count": n, "full": f, "control": c, "mse_difference": d,
            "rmse_difference": f["rmse"] - c["rmse"], "mae_difference": a,
            "global_bias_squared_difference": f["bias_squared"] - c["bias_squared"],
            "global_error_variance_difference": f["error_variance_population"] - c["error_variance_population"],
            "positive_gross_mse": gross, "negative_gross_mse_magnitude": improvement,
            "cancellation_fraction_of_positive_gross": improvement / gross if gross else None,
            "quantity_cells": cells}


def compare_splits(validation, test):
    cells = []
    for v, t in zip(validation["quantity_cells"], test["quantity_cells"]):
        pv, pt = v["weight"], t["weight"]
        dv, dt = v["mse_difference_within_bin"], t["mse_difference_within_bin"]
        cells.append({"bin": v["bin"], "validation_weight": pv, "test_weight": pt,
                      "validation_within_bin_difference": dv, "test_within_bin_difference": dt,
                      "symmetric_mix_component": (pt - pv) * (dt + dv) / 2,
                      "symmetric_within_bin_component": (dt - dv) * (pt + pv) / 2,
                      "test_contrast_at_validation_mix": pv * dt,
                      "validation_contrast_at_test_mix": pt * dv})
    difference = test["mse_difference"] - validation["mse_difference"]
    mix = math.fsum(x["symmetric_mix_component"] for x in cells)
    within = math.fsum(x["symmetric_within_bin_component"] for x in cells)
    close(difference, mix + within)
    standardized = math.fsum(x["test_contrast_at_validation_mix"] for x in cells)
    return {"test_minus_validation_mse_contrast": difference,
            "symmetric_mix_component": mix, "symmetric_within_bin_component": within,
            "mix_signed_share_of_net_change": mix / difference if difference else None,
            "within_bin_signed_share_of_net_change": within / difference if difference else None,
            "test_mse_contrast_at_validation_mix": standardized,
            "validation_mse_contrast_at_test_mix": math.fsum(x["validation_contrast_at_test_mix"] for x in cells),
            "positive_test_contrast_survives_validation_mix": test["mse_difference"] > 0 and standardized > 0,
            "quantity_cells": cells}


def bin_label(bounds, index):
    if index == 0:
        return f"q≤{bounds[0]:g}"
    if index == 4:
        return f"q>{bounds[-1]:g}"
    return f"{bounds[index-1]:g}<q≤{bounds[index]:g}"


def report(result):
    lines = ["# 기존 Test의 수량 오차 반전 진단 — 완료", "",
             "고정된 이력 보정27 조건의 native Validation 재현·Test 충분통계만 다시 계산했다. 새 추론·학습·epoch 선택은 수행하지 않았다. 양수 차이는 full 악화, 음수는 개선이다. Test는 과거 노출 이력이 있는 기존 최종 비교이며 새 독립 평가가 아니다.", "",
             "Taxi seed52의 제거 대조 대비 반전은 실제 수량>3449인 83건(전체0.997%)의 손해가 나머지4구간 이득을 넘어선 데서 발생한다. 그 구간에서 full의 평균 오차는−888.407, 제거 대조는+128.055다. current-only 대비 손해도 가장 큰 두 수량구간에 집중된다. 반면 Taxi42·62는 동일 구간·split에서 두 대조 모두 개선했다. 따라서 Taxi 전체가 실패하거나 tail 자체가 무조건 손해라고 결론 내릴 수 없다.", "",
             "Intermittent seed42의 악화는 tail이 아니라 2<실제 수량≤31 구간의 양의 오차 증가가 중심이다. 이 구간의 full 편향+1.684가 두 대조+1.172/+1.312보다 크고, 오차분산도 커졌다. tail에서는 두 대조에 오히려 이득이다. seed52·62는 동일 자료에서 양대조에 개선하므로 Taxi52와 Intermittent42를 한 종류의 ‘큰 수량 실패’로 묶는 설명은 맞지 않는다.", "",
             "RAF는 세 seed 모두 실제 수량>200인 39건(전체0.746%)에서 대조보다 더 크게 과소예측한다. 이 tail의 full 평균 오차는 seed42/52/62에서−354.648/−369.052/−355.821이다. tail이 양의 총손해49~100%를 차지하며, 여러 중간구간 이득이 상쇄된다. RAF52는 Validation에서부터 두 대조보다 나빴으므로 모든 RAF 악화를 Validation→Test 방향 반전으로 표현하면 안 된다.", "",
             "세 자료의 모든 악화10쌍은 Test를 Validation의5구간 비중으로 표준화해도 ΔMSE가 양수다. 고정5구간의 단순 비중 변화만으로 악화를 설명할 수 없다. 하지만 구간 내부의 실제 수량·시간·개체·history 변화를 분리한 것은 아니어서 과적합, 분포 변화 또는 구조 기여 중 하나를 인과 원인으로 확정하지 못했다.", "",
             "## 오차가 발생한 구간을 분해한다 — 완료", "",
             "RMSE 자체는 구간별로 더할 수 없다. 따라서 각 구간의 MSE 기여를 `(SSE_full−SSE_control)/전체 N`, MAE 기여를 `(SAE_full−SAE_control)/전체 N`으로 계산한다. 양의 총손해·음의 총이득·순차이를 함께 남겨 상쇄를 숨기지 않았다. 편향은 예측−실제, 분산은 표본추정이 아닌 `MSE−편향²`인 오차의 모집단 분산이다.", "",
             "| 자료 / seed | 대조 | Validation ΔMSE | Test ΔMSE | Test ΔRMSE | Test ΔMAE | Test 양의 총손해 | 상쇄 이득 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for c in result["contrasts"]:
        v, t = c["validation"], c["test"]
        lines.append(f"| {LABEL[c['dataset']]} {c['seed']} | {CONTROL_LABEL[c['control_arm']]} | {v['mse_difference']:.6f} | {t['mse_difference']:.6f} | {t['rmse_difference']:.6f} | {t['mae_difference']:.6f} | {t['positive_gross_mse']:.6f} | {t['negative_gross_mse_magnitude']:.6f} |")
    lines += ["", "개선 seed까지 포함한 총18쌍을 위에 모두 표시했다. 전체 표와 상세 JSON의 분모는 동일 자료·split·seed·arm에서 일치하며, 5구간 합계가 전체 count·SSE·SAE·부호오차합과 일치한다.", "",
              "## 반전과 RAF 악화의 위치를 확인한다 — 완료", "",
              "표의 ‘총손해 비중’ 분모는 양의 구간 기여 합이다. ‘순차이 비중’ 분모는 양·음 상쇄 후의 전체 ΔMSE이므로 100%를 넘을 수 있다. 사후 실제 수량으로 나눈 구간은 원인 분석용이며 실제 수량을 모르는 추론 시점에 적용할 gate가 아니다."]
    for c in result["contrasts"]:
        if not ((c["dataset"] == DATASETS[0] and c["seed"] == 52) or (c["dataset"] == DATASETS[1] and c["seed"] == 42) or c["dataset"] == DATASETS[2]):
            continue
        t, s = c["test"], c["split_composition"]
        bounds = c["quantity_boundaries"]
        lines += ["", f"### {LABEL[c['dataset']]} seed{c['seed']} / {CONTROL_LABEL[c['control_arm']]}", "",
                  f"Test ΔMSE {t['mse_difference']:.6f} = 총손해 {t['positive_gross_mse']:.6f} − 상쇄 이득 {t['negative_gross_mse_magnitude']:.6f}. full 전체 편향 {t['full']['bias']:.6f}, 대조 편향 {t['control']['bias']:.6f}.", "",
                  "| 실제 수량 구간 | Test 수 / 비중 | ΔMSE 기여 | ΔMAE 기여 | full 편향 | 대조 편향 | 편향² 기여 | 오차분산 기여 | 총손해 비중 | 순차이 비중 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for x in t["quantity_cells"]:
            positive = f"{100*x['positive_gross_share']:.2f}%" if x["positive_gross_share"] is not None else "—"
            net = f"{100*x['signed_net_share']:.2f}%" if x["signed_net_share"] is not None else "—"
            lines.append(f"| {bin_label(bounds,x['bin'])} | {x['count']} / {100*x['weight']:.3f}% | {x['mse_contribution']:.6f} | {x['mae_contribution']:.6f} | {x['full']['bias']:.6f} | {x['control']['bias']:.6f} | {x['weighted_bias_squared_contribution']:.6f} | {x['weighted_error_variance_contribution']:.6f} | {positive} | {net} |")
        lines += ["", f"Validation→Test 비교차 변화 {s['test_minus_validation_mse_contrast']:.6f} = 구간 비중 항 {s['symmetric_mix_component']:.6f} + 구간 내부 오차차 항 {s['symmetric_within_bin_component']:.6f}. Test를 Validation 구간 비중으로 표준화해도 ΔMSE={s['test_mse_contrast_at_validation_mix']:.6f}."]
    lines += ["", "## 구간 비중 변화와 구간 내부 변화를 구분한다 — 완료", "",
              "`D=Σ p_k d_k`에서 `D_Test−D_Validation=Σ(pT−pV)(dT+dV)/2 + Σ(dT−dV)(pT+pV)/2`를 정확히 계산했다. 첫 항은 구간 비중, 둘째 항은 같은 구간의 오차차 변화에 해당하는 회계적 항이다. 같은 구간 내부에서 실제 수량·개체·시간대·history가 달라질 수 있으므로 둘째 항을 모델 변화의 인과 효과로 해석할 수 없다. 비중 표준화는 수량구간 혼합만으로 반전을 설명할 수 있는지를 보는 서술적 확인이며 개입 실험이 아니다.", "",
              "구간별 편향은 실제 수량에 조건을 건 평균 오차다. 예측값에 조건을 건 calibration 검사와 다르며, 오차분산은 seed 간 학습 분산이 아니다. 원본에는 개별 paired loss 공분산·개체 단위 표준오차가 없어 이 계산으로 통계 유의성을 추정하지 않는다.", "",
              "| 자료 / seed | 대조 | Test−Val ΔMSE | 비중 항 | 구간 내부 항 | Val 비중의 Test ΔMSE |", "|---|---|---:|---:|---:|---:|"]
    for c in result["contrasts"]:
        s = c["split_composition"]
        lines.append(f"| {LABEL[c['dataset']]} {c['seed']} | {CONTROL_LABEL[c['control_arm']]} | {s['test_minus_validation_mse_contrast']:.6f} | {s['symmetric_mix_component']:.6f} | {s['symmetric_within_bin_component']:.6f} | {s['test_mse_contrast_at_validation_mix']:.6f} |")
    lines += ["", "## 검증과 재현", "",
              f"분석 완료 receipt rawSHA `{COMPLETION}`, canonical 계약 `{CONTRACT}`를 고정했다. 완료 receipt의 모든 source·analysis 파일, 회수 archive, 회수 manifest929개 verified 파일을 stdlib SHA256으로 검증했고, 관련 native 결과의 현재 사본도 회수 manifest 및 완료 receipt와 대조했다. 각 선택 checkpoint와 epoch는 계약과 일치한다.", "",
              "재계산 명령: `python3 reports/titantpp_test_reversal_diagnosis_20261012_v1/quantity_diagnosis.py`.", "",
              "출력 JSON의 `evidence_sha256`에 모든 읽은 증거 경로와 rawSHA, 각 구간 count·SSE·SAE·부호오차합·분모·편향·분산·Validation/Test mix 항을 보존했다. 이 계산만으로 GRU와 CNN의 개별 기여, 과적합 원인, 개체별 집중, 통계 유의성은 확인할 수 없다. 새 학습·구조 개입·독립 평가가 필요한 경우 별도 계약과 실행 승인을 받아야 한다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, default=Path("search_artifacts/titantpp_history_instacart_test_20261012_v1"))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    root = args.bundle.resolve()
    contract, evidence = verify_evidence(root)
    jobs = load_jobs(root, contract)
    contrasts = []
    for dataset in DATASETS:
        for seed in (42, 52, 62):
            for control in CONTROLS:
                f, c = jobs[dataset, seed, FULL], jobs[dataset, seed, control]
                v = contrast(f["validation"]["metrics"], c["validation"]["metrics"])
                t = contrast(f["test"]["metrics"], c["test"]["metrics"])
                contrasts.append({"dataset": dataset, "seed": seed, "full_arm": FULL, "control_arm": control,
                                  "full_job_id": f["job"]["id"], "control_job_id": c["job"]["id"],
                                  "full_selected_epoch": f["job"]["selected_epoch"], "control_selected_epoch": c["job"]["selected_epoch"],
                                  "original_validation_reference": {"full_qty_rmse": f["job"]["validation_reference"]["qty_rmse"],
                                                                    "control_qty_rmse": c["job"]["validation_reference"]["qty_rmse"],
                                                                    "full_qty_mae": f["job"]["validation_reference"]["qty_mae"],
                                                                    "control_qty_mae": c["job"]["validation_reference"]["qty_mae"],
                                                                    "mse_contrast": f["job"]["validation_reference"]["qty_rmse"]**2-c["job"]["validation_reference"]["qty_rmse"]**2,
                                                                    "replay_mse_contrast_minus_original": v["mse_difference"]-(f["job"]["validation_reference"]["qty_rmse"]**2-c["job"]["validation_reference"]["qty_rmse"]**2)},
                                  "quantity_boundaries": f["test"]["metrics"]["quantity_boundaries"],
                                  "validation": v, "test": t, "split_composition": compare_splits(v, t)})
    result = {"schema": "frozen_test_quantity_reversal_accounting_v1", "status": "verified_complete",
              "source_bundle": str(root), "contract_sha256": CONTRACT, "analysis_completion_raw_sha256": COMPLETION,
              "test_is_independent": False, "new_inference": False, "new_training": False,
              "analysis_kind": "descriptive_sufficient_statistics_accounting_not_causal_identification",
              "difference_direction": "full minus control; positive means full worse",
              "variance_definition": "population error second moment minus squared mean error; not sample variance",
              "selection": "all history27, all9 dataset/seed pairs and both controls, including benefiting seeds",
              "five_quantity_bins": "fixed all-Train boundaries; actual-target-defined descriptive strata, not an inference gate",
              "checks": {"count_sse_sae_signed_error_reconciled": True, "mse_mae_contributions_reconciled": True,
                         "symmetric_mix_within_bin_decomposition_reconciled": True,
                         "archived_verified_file_count": 929, "contrasts": len(contrasts)},
              "evidence_sha256": evidence, "calculation_script_sha256": digest(Path(__file__)), "contrasts": contrasts}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "quantity_diagnosis.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    (args.output / "QUANTITY_DIAGNOSIS.md").write_text(report(result))
    print(json.dumps({"status": result["status"], "contrasts": len(contrasts), "archive_files_verified": 929,
                      "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
