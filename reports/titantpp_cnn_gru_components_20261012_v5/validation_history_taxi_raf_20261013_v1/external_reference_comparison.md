## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-10-13T14:22:05.414488+09:00
- Verification Status: ANALYZED
- Version Label: external_validation_reference_v1

## 비교 범위

현재 구성요소 CNN+GRU와 기존 완료 외부6종의 공통 출력부 Validation 참조 비교다. 외부 모델을 현 source·Runtime에서 새로 학습한 비교가 아니다. Taxi·RAF의 seed42/52/62 산술평균±표본SD를 제시한다. 현재 CNN+GRU는 구성요소 대조24개의 선택 Validation history를, 외부6종은 기존 감사된 selected endpoint 집계를 사용했다. Test는 열람하지 않았다.

## 전체 raw 수량 RMSE 평균 ± 표본SD

| 모델 | Taxi | RAF |
|---|---:|---:|
| 현재 CNN+GRU54 | 81.923 ± 5.027 | 34.182 ± 0.234 |
| RMTPP | 94.728 ± 3.401 | 34.550 ± 0.251 |
| THP | 107.598 ± 4.504 | 34.745 ± 0.320 |
| NHP | 318.815 ± 5.301 | 36.273 ± 0.319 |
| SAHP | 120.527 ± 2.000 | 35.419 ± 0.213 |
| S2P2 | 106.354 ± 1.914 | 33.995 ± 0.126 |
| AttNHP | 127.129 ± 8.992 | 34.877 ± 0.068 |

## 관측 결과

- RAF는 현재 구성요소 대조에서 FULL 평균 RMSE34.182가 NONE34.224보다0.121% 낮다. 작은 수치 개선이라는 사실을 유지한다.
- Taxi는 외부6종보다 현재FULL의 평균 RMSE와MAE가 모두 낮다. 외부군에서 가장 낮은RMSE인 RMTPP94.728 대비FULL81.923은13.518% 낮다.
- RAF는 외부6종 중5종보다평균RMSE가낮지만S2P2의33.995보다FULL34.182가0.552% 높다. MAE는FULL9.181보다RMTPP9.136과NHP9.179가낮다.
- 과거 내부TitanTPP-MLP4는TaxiRMSE79.711±2.826, RAF33.951±0.059다. 외부6종에대한위치와모든TitanTPP변형중최고라는주장은구분한다.

## 비교 가능성과 한계

Train/Validation target identity·quantity 및parquet/split SHA, B128/min40/max300/patience40, 3seed와최초strict finite 최소raw수량RMSE 선택규칙은일치한다. 그러나source세대·GPU배치·전체Runtime/OS/driver까지같은새외부대조가아니다. 현재FULL은선택history이고외부는기존감사된selected endpoint라는증거단계차이도남긴다.

외부모델은동일관측수치형입력과공통수량·시간출력부·손실에적응한비교구현이다. 원논문native head·개별최적튜닝전체보다우월한것으로확대하지않는다. 10월8일baseline parity는CNN+GRU와retrieval_mean의제한적alias구현검사로외부성능parity가아니다. 이번비교는새학습·재추론·SSH·원본회수·Test열람0이다. 통계유의성검정을수행하지않았다.

## 출처

- 신규구성요소선택history 집계: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_components_20261012_v5/validation_history_taxi_raf_20261013_v1/analysis.json
- 신규집계rawSHA: b616c1d58e89a29986b996f3ace26d31f370e4df42efb26f15cb3e922405343a
- 기존외부Validation 집계: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_completed_external_comparison_20261001_v1/comparison.json
- 기존외부집계rawSHA: 587d035451f4b5a2cb36b4e444b924b126c745a3257cd6f7d941df0514591480
- 비교계약감사: reports/titantpp_external_panel_restoration_20261008_v1/contract_audit.md
- Validation전용기존전체표: reports/titantpp_three_dataset_final_comparison_20261005_v1/VALIDATION_TABLES.md
