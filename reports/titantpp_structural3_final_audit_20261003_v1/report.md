# 최종 구조 대조 원본 감사와 3seed 집계

평가 범위는 validation-only다. 새 구조 대조 3조건/6 checkpoint가 감사에 통과했고 기존 구조 대조 21조건과 원래 MLP 12조건을 재사용했다. 4개 데이터 × 3개 모델 × 3seed의 36개 행이다. 이는 원래 후속 36조건 캠페인 전체 감사 완료를 뜻하지 않는다.

| 데이터 | 모델 | MAE 평균 ± 표본 SD | RMSE 평균 ± 표본 SD | 시간 NLL 평균 ± 표본 SD |
|---|---|---:|---:|---:|
| Taxi | Original MLP | 25.623661 ± 0.832006 | 79.711120 ± 2.825983 | 1.038668 ± 0.284331 |
| Taxi | Current-only | 28.813019 ± 0.468834 | 90.048143 ± 1.513093 | 0.881758 ± 0.182704 |
| Taxi | All-available | 25.329012 ± 0.294516 | 78.121235 ± 0.711441 | 1.190739 ± 0.354153 |
| Intermittent | Original MLP | 0.700523 ± 0.054055 | 1.673034 ± 0.081887 | 0.497111 ± 0.209463 |
| Intermittent | Current-only | 0.735678 ± 0.045445 | 1.744653 ± 0.034595 | 0.464801 ± 0.103922 |
| Intermittent | All-available | 0.697524 ± 0.031214 | 1.683666 ± 0.009952 | 0.367282 ± 0.011788 |
| RAF | Original MLP | 9.250602 ± 0.085484 | 33.950679 ± 0.058804 | 3.544626 ± 0.066929 |
| RAF | Current-only | 9.304358 ± 0.080607 | 34.207615 ± 0.249548 | 3.454769 ± 0.095354 |
| RAF | All-available | 9.252190 ± 0.116989 | 33.994099 ± 0.120212 | 3.524878 ± 0.071817 |
| Instacart | Original MLP | 3.991574 ± 0.002592 | 5.882742 ± 0.004771 | 2.807087 ± 0.004149 |
| Instacart | Current-only | 3.993312 ± 0.002966 | 5.882187 ± 0.016217 | 2.815348 ± 0.013820 |
| Instacart | All-available | 3.990216 ± 0.003112 | 5.878243 ± 0.005615 | 2.806875 ± 0.004286 |

## 해석 범위

- 각 seed의 최초 최소 raw 수량 RMSE epoch에서 MAE·RMSE·시간 NLL을 함께 읽었다. 평균은 seed별 지표의 산술평균이며 표준편차는 ddof=1이다.
- Intermittent 구조 대조 seed62는 PRO4500, seed42/52 및 원래 MLP는 5080이다. 이종 GPU 출처를 유지하고 효율 측정에 혼합하지 않는다.
- 학습·forward·validation replay를 새로 실행하지 않았다. strict CPU 모델 로드, optimizer/RNG/shuffle 복원, frozen source109 및 원본 SHA, 선택 epoch, 저장 지표, selected/last replay와 exposure를 검증했다.
- Deep Renewal·A100의 추가 CPU 감사와 독립적인 최종 일반화 평가는 이 작업의 완료 주장에 포함하지 않는다.

## 재현

프로젝트 root에서 새 scope에 해당하는 최초 회수는 `collect.py 5090` 및 `collect_local_pro4500.py`, 감사는 `audit.py 5090`과 `audit.py pro4500`, 집계는 `summarize.py`를 사용했다. 수집·감사 스크립트는 기존 완료 결과를 덮어쓰지 않는다. 삭제된 PRO4500 Pod에 접속하지 않았다.

원본과 감사 경로, contract/source SHA, seed별 짝 비교와 방향별 변화율은 `comparison.json`, 36개 조건별 지표는 `condition_registry.csv`, 감사 증거는 `verification.json`에 보존한다.
