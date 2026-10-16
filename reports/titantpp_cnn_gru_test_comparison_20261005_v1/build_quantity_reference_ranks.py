"""Rank the observed quantity metrics, retaining seed-free deterministic controls.

This reads aggregate evidence only and never performs inference or training.
Native NB and quantity-only references remain labelled; their time scores are not
pooled with the common-output-head comparison.
"""
from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parent
LABELS = {
    "titantpp_history_mlp": "MLP4",
    "titantpp_history_mlp_width8": "MLP8",
    "titantpp_history_mlp_width12": "MLP12",
    "titantpp_history_mlp_width16": "MLP16",
    "titantpp_all_available_history_mlp": "모든 가용 분기 MLP",
    "titantpp_current_only_param_matched": "현재 상태만 · 파라미터 대조",
    "titantpp": "TitanTPP B",
    "titantpp_cnn_mlp16": "CNN+MLP16",
    "titantpp_gru54": "GRU54",
    "titantpp_cnn_gru54": "CNN+GRU54",
    "s2p2_matched_head": "S2P2 · 공통 출력부",
    "attnhp_matched_head": "AttNHP · 공통 출력부",
    "deep_renewal_event_native_nb": "Deep Renewal · native NB",
    "last_observed_quantity": "마지막 관측 수량 · 수량 전용",
    "mean_quantity_in_same_observed_window": "관측 이력 평균 · 수량 전용",
}
DATASETS = {
    "yellow_trip_hourly": "Taxi",
    "intermittent_frozen_5000": "Intermittent",
    "raf_spare_parts": "RAF",
}


def main():
    rows = list(csv.DictReader((ROOT / "combined_metrics_per_seed.csv").open()))
    rows = [r for r in rows if r["seed"] in ("42", "")]
    if len(rows) != 224:
        raise ValueError("Expected 224 seed42 or deterministic-control aggregate rows")
    ranked = []
    doc = [
        "# 모든 모델의 수량 지표 순위",
        "",
        "학습 모델은 동일 seed42, 결정론 수량 기준선은 seed가 없다. "
        "공통 사건 모집단에서 전체·큰 수량 RMSE와 MAE를 비교한다. "
        "수량 전용 기준선과 native NB를 포함한 관측 순위이며 시간 예측까지 포함한 종합 우월성 순위가 아니다.",
        "",
        "각 표는 해당 범위의 RMSE 오름차순이다. MAE 순위는 별도 열이다. "
        "숫자가 작은 RMSE/MAE가 좋다. 시간 NLL은 출력부를 구분한 [주 비교표](TEST_SEED42.md)에서 확인한다.",
        "",
        "모델 ID는 총19개지만 RAF TitanTPP B는 원본 평가가 없어 RAF에는18개다. 빈 결과를 만들어 채우지 않았다.",
        "",
    ]
    for split in ("validation", "test"):
        doc += [f"## {split}", ""]
        for dataset, name in DATASETS.items():
            for view in ("overall", "tail"):
                group = [r for r in rows if (r["dataset"], r["split"], r["view"]) == (dataset, split, view)]
                if len(group) != (18 if name == "RAF" else 19):
                    raise ValueError(f"Unexpected model coverage: {dataset}/{split}/{view}")
                if len({r["count"] for r in group}) != 1:
                    raise ValueError("Population counts differ")
                values = {m: sorted(float(r[m]) for r in group) for m in ("qty_rmse", "qty_mae")}
                doc += [f"### {name} · {view} · target {group[0]['count']}", "",
                        "| RMSE 순위 | MAE 순위 | 모델 | RMSE | MAE | 비교 범위 |",
                        "|---:|---:|---|---:|---:|---|"]
                for r in sorted(group, key=lambda r: (float(r["qty_rmse"]), r["model"])):
                    rmse_rank = 1 + sum(v < float(r["qty_rmse"]) for v in values["qty_rmse"])
                    mae_rank = 1 + sum(v < float(r["qty_mae"]) for v in values["qty_mae"])
                    scope = ("수량 전용" if r["seed"] == "" else
                             "native NB · 참고" if r["model"] == "deep_renewal_event_native_nb" else
                             "공통 출력부 학습 모델")
                    ranked.append({**r, "all_quantity_rmse_rank": rmse_rank,
                                   "all_quantity_mae_rank": mae_rank, "quantity_rank_scope": scope})
                    doc.append(f"| {rmse_rank} | {mae_rank} | {LABELS.get(r['model'], r['model'].upper())} | "
                               f"{float(r['qty_rmse']):.6f} | {float(r['qty_mae']):.6f} | {scope} |")
                doc += [""]
    with (ROOT / "all_model_quantity_ranks.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    (ROOT / "ALL_MODEL_QUANTITY_RANKS.md").write_text("\n".join(doc) + "\n")
    print({"rows": len(ranked), "model_ids": len({r['model'] for r in ranked}), "inference_executed": False})


if __name__ == "__main__":
    main()
