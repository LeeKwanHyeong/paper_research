# 후속 36조건 중 완료 30조건 비교 — 2026-10-02 19:53 KST

Validation-only. 완료 조건의 selected validation replay만 비교한다. 모든 지표는 최초 최소 raw 수량 RMSE 선택 epoch의 값이다. 세 seed가 모두 끝난 그룹만 평균과 표본표준편차(ddof=1)를 계산했다. 기존 MLP는 재학습하지 않고 확정 결과를 재사용한다.

## 3seed 완료 그룹

| 데이터 | 모델 | MAE 평균±SD | RMSE 평균±SD | Time NLL 평균±SD | GPU |
|---|---|---:|---:|---:|---|
| Taxi | Original MLP | 25.623661 ± 0.832006 | 79.711120 ± 2.825983 | 1.038668 ± 0.284331 | 5080 |
| Taxi | Current-only | 28.813019 ± 0.468834 | 90.048143 ± 1.513093 | 0.881758 ± 0.182704 | 5080 |
| Taxi | All-available | 25.329012 ± 0.294516 | 78.121235 ± 0.711441 | 1.190739 ± 0.354153 | 5080 |
| Taxi | Deep Renewal | 91.474695 ± 1.046666 | 404.786354 ± 2.978370 | 0.790247 ± 0.015938 | 5080 |
| Intermittent | Original MLP | 0.700523 ± 0.054055 | 1.673034 ± 0.081887 | 0.497111 ± 0.209463 | 5080 |
| Intermittent | Current-only | 0.735678 ± 0.045445 | 1.744653 ± 0.034595 | 0.464801 ± 0.103922 | 5080,pro4500 |
| RAF | Original MLP | 9.250602 ± 0.085484 | 33.950679 ± 0.058804 | 3.544626 ± 0.066929 | 5080 |
| RAF | Current-only | 9.304358 ± 0.080607 | 34.207615 ± 0.249548 | 3.454769 ± 0.095354 | 5080 |
| RAF | All-available | 9.252190 ± 0.116989 | 33.994099 ± 0.120212 | 3.524878 ± 0.071817 | 5080 |
| RAF | Deep Renewal | 10.261750 ± 0.063932 | 34.653847 ± 0.109843 | 6.650869 ± 0.332432 | 5080 |
| Instacart | Original MLP | 3.991574 ± 0.002592 | 5.882742 ± 0.004771 | 2.807087 ± 0.004149 | 5090 |

Intermittent Current-only seed62는 PRO4500, seed42/52는5080이다. 이종 GPU 출처를 보존하며 계산 효율 비교와 섞지 않는다.

## 완료 30조건과 같은 seed MLP 비교

변화율은 `(후속 / 기존 MLP - 1) × 100`이며 음수는 후속 모델의 오차가 낮음을 뜻한다.

| 데이터 | seed | 모델 | 완료/선택 epoch | MAE | RMSE | Time NLL | 변화율 MAE/RMSE/NLL (%) |
|---|---:|---|---:|---:|---:|---:|---|
| Taxi | 42 | Current-only | 75/35 | 28.891639 | 89.942490 | 0.715675 | +9.028 / +9.042 / -15.077 |
| Taxi | 42 | All-available | 203/163 | 25.071719 | 77.334671 | 1.205044 | -5.387 / -6.243 / +42.992 |
| Taxi | 42 | Deep Renewal | 300/300 | 90.337847 | 402.498289 | 0.785984 | +240.906 / +387.972 / -6.734 |
| Taxi | 52 | Current-only | 156/116 | 28.309845 | 88.590647 | 1.077463 | +13.952 / +15.300 / -21.052 |
| Taxi | 52 | All-available | 99/59 | 25.265072 | 78.719786 | 0.829650 | +1.696 / +2.453 / -39.210 |
| Taxi | 52 | Deep Renewal | 300/300 | 91.687884 | 403.706818 | 0.776874 | +269.060 / +425.422 / -43.077 |
| Taxi | 62 | Current-only | 94/54 | 29.237573 | 91.611293 | 0.852138 | +14.531 / +14.780 / -6.202 |
| Taxi | 62 | All-available | 300/298 | 25.650246 | 78.309247 | 1.537523 | +0.479 / -1.886 / +69.240 |
| Taxi | 62 | Deep Renewal | 300/300 | 92.398354 | 408.153955 | 0.807883 | +261.949 / +411.377 / -11.074 |
| Intermittent | 42 | Current-only | 61/21 | 0.745193 | 1.773053 | 0.352289 | +16.714 / +12.326 / +1.573 |
| Intermittent | 42 | All-available | 50/10 | 0.675898 | 1.682415 | 0.358399 | +5.861 / +6.584 / +3.334 |
| Intermittent | 42 | Deep Renewal | 153/113 | 0.572327 | 2.379105 | 1.075626 | -10.361 / +50.721 / +210.126 |
| Intermittent | 52 | Current-only | 100/60 | 0.775612 | 1.754782 | 0.484922 | +5.174 / +2.069 / +18.818 |
| Intermittent | 52 | All-available | 51/11 | 0.733308 | 1.694184 | 0.362791 | -0.563 / -1.455 / -11.107 |
| Intermittent | 52 | Deep Renewal | 154/114 | 0.569690 | 2.451200 | 1.057593 | -22.750 / +42.578 / +159.136 |
| RAF | 42 | Current-only | 45/5 | 9.248576 | 34.467761 | 3.534701 | -0.411 / +1.677 / -0.681 |
| RAF | 42 | All-available | 49/9 | 9.385523 | 33.868021 | 3.605488 | +1.063 / -0.092 / +1.308 |
| RAF | 42 | Deep Renewal | 299/259 | 10.333441 | 34.599357 | 6.396626 | +11.270 / +2.065 / +79.734 |
| RAF | 52 | Current-only | 45/5 | 9.396776 | 33.970223 | 3.480382 | +0.910 / +0.095 / +0.250 |
| RAF | 52 | All-available | 45/5 | 9.204316 | 34.107428 | 3.501431 | -1.157 / +0.500 / +0.856 |
| RAF | 52 | Deep Renewal | 300/300 | 10.210654 | 34.780283 | 7.027052 | +9.650 / +2.482 / +102.410 |
| RAF | 62 | Current-only | 48/8 | 9.267722 | 34.184863 | 3.349225 | +1.254 / +0.500 / -7.050 |
| RAF | 62 | All-available | 59/19 | 9.166732 | 34.006848 | 3.467714 | +0.150 / -0.023 / -3.761 |
| RAF | 62 | Deep Renewal | 284/244 | 10.241154 | 34.581903 | 6.528931 | +11.889 / +1.667 / +81.196 |
| Instacart | 42 | Current-only | 70/30 | 3.990017 | 5.872238 | 2.803320 | -0.037 / -0.110 / +0.004 |
| Instacart | 42 | All-available | 70/30 | 3.992629 | 5.872499 | 2.803145 | +0.028 / -0.106 / -0.003 |
| Instacart | 42 | Deep Renewal | 53/13 | 4.024150 | 5.779987 | 2.848954 | +0.818 / -1.680 / +1.632 |
| Instacart | 52 | Current-only | 55/15 | 3.994152 | 5.873423 | 2.812281 | -0.001 / -0.137 / +0.029 |
| Instacart | 52 | All-available | 55/15 | 3.991316 | 5.878511 | 2.811556 | -0.072 / -0.050 / +0.003 |
| Intermittent | 62 | Current-only | 82/42 | 0.686230 | 1.706125 | 0.557191 | -5.430 / -0.888 / -24.333 |

## 진행 중·미시작 보존

- Instacart / Deep Renewal / seed52 / 5090: running, completed/best epoch 21/13. 최종 성능 집계에서 제외.
- Instacart / Current-only / seed62 / 5090: not_started, completed/best epoch 0/None. 최종 성능 집계에서 제외.
- Instacart / All-available / seed62 / 5090: not_started, completed/best epoch 0/None. 최종 성능 집계에서 제외.
- Instacart / Deep Renewal / seed62 / 5090: not_started, completed/best epoch 0/None. 최종 성능 집계에서 제외.
- Intermittent / All-available / seed62 / pro4500: running, completed/best epoch 5/5. 최종 성능 집계에서 제외.
- Intermittent / Deep Renewal / seed62 / pro4500: not_started, completed/best epoch 0/None. 최종 성능 집계에서 제외.

## 해석

- Taxi: 기존 MLP는 Current-only보다 평균 MAE·RMSE가 낮다. All-available은 기존 MLP보다 평균 MAE·RMSE가 낮지만 Time NLL은 높다. 단계적 활성화의 우월성은 지지되지 않는다.
- Intermittent: Current-only 3seed 평균 MAE·RMSE는 기존 MLP보다 높다. seed62는 역방향이다. All-available은 완료 seed42에서 기존보다 나쁘고 seed52에서 좋다. Deep Renewal은 완료 seed42/52에서 MAE가 낮지만 RMSE·Time NLL은 높다. 이 두 모델의3seed 결론은 아직 없다.
- RAF: 기존 MLP는 두 내부 변형보다 평균 MAE·RMSE가 소폭 낮으며 Deep Renewal보다 세 지표 모두 낮다. 작은 평균 차이를 통계적 우월성으로 해석하지 않는다.
- Instacart: 내부 변형의 RMSE는 완료 seed42/52에서 소폭 낮다. Deep Renewal seed42는 RMSE가 낮으나 MAE·Time NLL이 높다.3seed 결론은 보류한다.
- Deep Renewal은 event-native adapter로 고유 NB 수량·시간 손실을 사용한다. 모든 모델이 동일 수량 손실로 학습됐다는 설명이나 원 논문 전체에 대한 일반화는 하지 않는다.

## 검증 범위

종료 manifest와 selected/last 재평가 소형 기록 SHA 검증을 통과한30조건이다. 기존13조건 CPU감사는 재사용하며 신규 완료 checkpoint의 원본 회수·CPU감사는 본 분석에서 실행하지 않았다. held-out/test 성능은 열람하지 않았다.

## 원본

- [관측 기록](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/hourly_monitor/20261002T105307983521Z/analysis.json)
- [기존 MLP 선택값](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_external_comparison_20261001_v1/selected_conditions.csv)
