# TitanTPP core 최종 validation 결과 — 완료

- [최종 보고서](final_report.md): 대표 MLP, 전체 구조·외부 비교, 3seed 평균±표본표준편차, MAE/RMSE/시간 NLL, 반례와 비용.
- [기계판독 결과](final_results.json) · [원본 감사](verification.json) · [독립 집계 검증](independent_verification.json).
- [3seed 최종표](final_three_seed_metrics.csv) · [조건별 selected/last](final_conditions.csv) · [구간별 결과](final_strata.csv).
- [안정성](final_stability.csv) · [원래 성능기준](final_gate_checks.csv) · [MLP seed별 비교](final_mlp_paired.csv) · [실측 비용](final_costs.csv).

core54조건108역할(신규36+재사용18)의 학습·재평가·최종 원본 감사 완료. 외부36조건을 더한90개 고유 조건은 기존 B/Full18개 중복을 제거했다. Gate 탐색과 단발 MAC 효율 측정은 별도 보고서다.

전체 표는 validation이며 held-out 결과가 아니다. 기존1차 분석과 과거 중간 보고서는 당시 기록으로 보존했다. 새 학습이나 평가 없이 `finalize.py`와 `verify_final_tables.py`로 저장 기록의 집계를 재현할 수 있다.
