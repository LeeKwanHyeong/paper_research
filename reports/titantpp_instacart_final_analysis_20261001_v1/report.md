# Instacart 최종 비교: MAE의 작은 이득과 RMSE의 열세

분석 완료: 2026-10-01T19:50:30.870455+09:00. 5090 학습 큐 종료: 2026-10-01 16:33:08 KST.

**5090의 6조건 원본 회수와 CPU 검증을 완료했다. Instacart에서 TitanTPP MLP가 외부 최상위 모델보다 우수하다는 결론은 나오지 않았다.** S2P2·AttNHP보다 평균 MAE는 조금 낮지만 평균 RMSE는 높고, RMSE는 두 모델에 대해 세 seed 모두 열세다. 7모델 중 TitanTPP의 평균 순위는 MAE 2위, RMSE 6위, 시간 NLL 5위다.

대표 모델은 기존 `titantpp_history_mlp`다. 모든 값은 동일한 최초 최소 raw 수량 RMSE checkpoint의 validation 결과이며, 평균과 표본표준편차는 seed42·52·62를 동일 가중치로 계산했다. 표본 수는 seed당 503,733개이고 같은 validation 표본을 세 번 독립 데이터셋처럼 합산하지 않는다.

## 1. 7모델의 최종 3seed 비교 — 완료

모든 지표는 낮을수록 좋다. 외부 모델은 동일 입력·head·loss·선택 기준으로 연결한 비교군이다.

| 모델 | 수량 MAE | 수량 RMSE | 시간 NLL |
|---|---:|---:|---:|
| TitanTPP MLP | 3.991574 ± 0.002592 | 5.882742 ± 0.004771 | 2.807087 ± 0.004149 |
| RMTPP | 3.985184 ± 0.005197 | 5.877541 ± 0.021322 | 2.805148 ± 0.004840 |
| THP | 4.001303 ± 0.002485 | 5.858178 ± 0.025304 | 2.805649 ± 0.001553 |
| NHP | 4.492282 ± 0.030410 | 6.761279 ± 0.053816 | 2.813234 ± 0.008020 |
| SAHP | 3.997910 ± 0.009945 | 5.877591 ± 0.006729 | 2.801849 ± 0.001259 |
| S2P2 | 3.998695 ± 0.006282 | 5.852254 ± 0.015040 | 2.797041 ± 0.001358 |
| AttNHP | 3.995799 ± 0.002080 | 5.864779 ± 0.019441 | 2.807970 ± 0.002779 |

**MAE 최저는 RMTPP, RMSE와 시간 NLL 최저는 S2P2다.** TitanTPP는 RMTPP보다 MAE가 0.1604% 높고, S2P2보다 RMSE가 0.5210% 높다. 작은 차이라는 점과 방향이 불리하다는 점을 함께 기록한다. 세 seed의 표준편차가 작거나 겹친다는 사실만으로 통계적 동등성·유의성을 판정하지 않는다.

## 2. 추가 비교군 대비 차이와 seed별 일관성 — 완료

개선율 = (비교군 평균 − TitanTPP 평균) / 비교군 평균 × 100. 양수는 TitanTPP 이득, 음수는 열세다. 승수는 동일 seed에서 TitanTPP의 지표가 더 낮은 횟수다.

| 비교군 | MAE 개선율 | RMSE 개선율 | MAE 승 | RMSE 승 | 시간 NLL 승 |
|---|---:|---:|---:|---:|---:|
| RMTPP | -0.1604% | -0.0885% | 0/3 | 1/3 | 0/3 |
| THP | +0.2431% | -0.4193% | 3/3 | 1/3 | 1/3 |
| NHP | +11.1459% | +12.9937% | 3/3 | 3/3 | 3/3 |
| SAHP | +0.1585% | -0.0876% | 2/3 | 1/3 | 0/3 |
| S2P2 | +0.1781% | -0.5210% | 2/3 | 0/3 | 0/3 |
| AttNHP | +0.1057% | -0.3063% | 2/3 | 0/3 | 2/3 |

S2P2 대비 평균 MAE는 0.1781%, AttNHP 대비 0.1057% 낮다. 두 비교 모두 MAE 승수는 2/3이다. 반면 평균 RMSE는 각각 0.5210%, 0.3063% 높으며 승수는 모두 0/3이다. 따라서 “추가 TPP 대비 수량 예측이 개선됐다”보다 “MAE와 RMSE에서 서로 다른 결과를 보였다”가 정확하다. NHP에 대해서는 두 수량 지표와 시간 NLL이 세 seed 모두 낮지만, 이것을 외부 비교군 전체의 우위로 넓히지 않는다.

| 모델 | seed | 선택 / 종료 epoch | MAE | RMSE | 시간 NLL |
|---|---:|---:|---:|---:|---:|
| TitanTPP MLP | 42 | 30 / 70 | 3.991502 | 5.878730 | 2.803218 |
| S2P2 | 42 | 72 / 112 | 4.005765 | 5.838054 | 2.796148 |
| AttNHP | 42 | 72 / 112 | 3.995990 | 5.847649 | 2.811169 |
| TitanTPP MLP | 52 | 15 / 55 | 3.994202 | 5.881479 | 2.811469 |
| S2P2 | 52 | 11 / 51 | 3.993754 | 5.868013 | 2.798603 |
| AttNHP | 52 | 11 / 51 | 3.993630 | 5.860780 | 2.806147 |
| TitanTPP MLP | 62 | 35 / 75 | 3.989019 | 5.888017 | 2.806575 |
| S2P2 | 62 | 35 / 75 | 3.996564 | 5.850693 | 2.796372 |
| AttNHP | 62 | 35 / 75 | 3.997777 | 5.885909 | 2.806595 |

## 3. MAE와 RMSE의 방향이 다른 이유 — 관측된 오차 분포

수량 구간은 원래 동결된 경계 8·20·25·35를 사용한다. `searchsorted(..., side="left")`에 따라 아래 구간의 상한이 포함된다. 수량은 Instacart에서 기록된 같은 사용자·활동일의 product-row 수로 정의한 basket-size proxy다.

| 실제 수량 구간 | 표본 수 | 비중 | TitanTPP MAE / RMSE | S2P2 MAE / RMSE | AttNHP MAE / RMSE |
|---|---:|---:|---:|---:|---:|
| ≤8 | 247,651 | 49.16% | 2.7184 / 3.8081 | 2.8722 / 4.0783 | 2.8268 / 3.9210 |
| 8 < q ≤20 | 202,534 | 40.21% | 3.6871 / 4.6741 | 3.7605 / 4.8145 | 3.6216 / 4.6209 |
| 20 < q ≤25 | 27,322 | 5.42% | 7.9877 / 9.0866 | 7.4211 / 8.6749 | 7.8505 / 8.9679 |
| 25 < q ≤35 | 20,190 | 4.01% | 11.8469 / 13.1561 | 10.6341 / 12.2964 | 11.5741 / 12.9685 |
| >35 | 6,036 | 1.20% | 22.0811 / 24.6419 | 20.5236 / 23.4686 | 21.7169 / 24.4131 |

S2P2와 비교하면 TitanTPP는 수량 20 이하의 89.37%에서 평균 MAE·RMSE가 낮다. 그러나 수량 20 초과의 10.63%에서는 두 지표가 모두 높다. 가장 큰 수량 구간(>35)은 1.20%에 불과하지만 평균 RMSE가 TitanTPP 24.6419, S2P2 23.4686이다.

저수량 두 구간의 이득은 전체 표본당 평균 제곱오차 차이에 −1.5882, 고수량 세 구간의 열세는 +1.9458을 기여한다. 합계 +0.3576 때문에 전체 RMSE는 S2P2보다 높아진다. 반면 절대오차 합계에서는 저수량 이득이 조금 더 커 전체 MAE는 낮다. 이 계산은 세 seed의 구간별 SSE를 전체 표본 수로 나누어 합한 값이며, 구간 RMSE를 단순 평균해 전체 RMSE를 만든 것이 아니다.

AttNHP에 대해서는 가장 작은 수량 구간(≤8)에서 TitanTPP가 유리하고, 나머지 네 구간의 평균 수량 지표는 AttNHP가 유리하다. 따라서 단순히 “대부분 구간에서 더 좋다”는 설명도 비교군에 따라 달라진다.

이력 길이별 수치도 `analysis.json`과 `strata.csv`에 보존했다. S2P2는 모든 비어 있지 않은 이력 길이 구간에서 평균 RMSE가 낮다. 따라서 현재 결과만으로 긴 이력이 부족한 문제 하나를 원인으로 확정할 수 없다. 위 구간 분석은 오차가 어디에서 발생했는지 설명하며, 모델 구조가 왜 그 오차를 만들었는지에 대한 인과 검증은 아니다.

## 4. 활성 분기 정규화의 Instacart 결과 — seed42 탐색으로 분리

이미 감사된 4090→5080 복구 결과를 재사용한다. 선택30/종료70epoch, MAE 3.990506, RMSE 5.871661, 시간 NLL 2.802719다. 동일 seed의 plain MLP 대비 MAE 0.02495%, RMSE 0.12025% 감소했지만, S2P2·AttNHP의 같은 seed RMSE보다 여전히 높다. 정규화의 3seed 우위 또는 대표 모델 교체를 뒷받침하는 결과로 쓰지 않는다. 보류 중인 seed52·62를 새로 실행하지 않았다.

## 5. 선택·마지막 checkpoint와 실측 비용을 보존한다 — 완료

종료 후 마지막 checkpoint도 별도로 남긴다. 주 결과는 위의 RMSE 선택 checkpoint이며, 마지막 지표로 유리한 결과를 바꿔 고르지 않는다.

| 모델 | 마지막 MAE | 마지막 RMSE | 마지막 시간 NLL |
|---|---:|---:|---:|
| TitanTPP MLP | 4.032520 ± 0.056782 | 5.987658 ± 0.118539 | 2.803180 ± 0.003374 |
| RMTPP | 4.051423 ± 0.057192 | 6.031676 ± 0.118660 | 2.801172 ± 0.003316 |
| THP | 4.039574 ± 0.052774 | 5.983700 ± 0.117496 | 2.805433 ± 0.003982 |
| NHP | 4.691720 ± 0.091319 | 7.029410 ± 0.124388 | 2.815284 ± 0.006689 |
| SAHP | 4.046423 ± 0.086462 | 5.997267 ± 0.179594 | 2.798505 ± 0.001559 |
| S2P2 | 4.079135 ± 0.096115 | 6.067935 ± 0.198320 | 2.798967 ± 0.005501 |
| AttNHP | 4.113689 ± 0.124400 | 6.136155 ± 0.245854 | 2.818746 ± 0.011273 |

| 5090 조건 | 선택 / 종료 | 실측 fit 시간(h) | epoch 기록 합(h) | peak allocated(MiB) |
|---|---:|---:|---:|---:|
| S2P2 seed42 | 72 / 112 | 7.6858 | 7.6803 | 368.439 |
| AttNHP seed42 | 72 / 112 | 5.3549 | 5.3499 | 171.246 |
| S2P2 seed52 | 11 / 51 | 3.5107 | 3.5053 | 368.439 |
| AttNHP seed52 | 11 / 51 | 2.4964 | 2.4913 | 171.246 |
| S2P2 seed62 | 35 / 75 | 5.2321 | 5.2266 | 368.439 |
| AttNHP seed62 | 35 / 75 | 3.6780 | 3.6728 | 171.246 |

6조건의 `train_one` 실측 합은 27.9580시간이다. 이는 epoch 기록 합·캠페인 전체 경과시간과 다르다. 준비·최종 재평가·대기 시간을 추정 보정하지 않았다. 개인 5090 추가 cloud 임대료는 0달러이며 전기료는 측정하지 않았다. 다른 종료 epoch와 다른 모델의 이 시간을 기존 TitanTPP 대비 공정한 단독 속도 비교로 해석하지 않는다.

## 6. 원본 회수·검증 범위 — 완료

- 5090 원본 223파일·동결 소스 106개·checkpoint 12개를 회수·검증했다. 서버 원본은 변경하지 않았다.
- 계약·동결 source closure·approval/permit·실제 native qualification·초기 tensor SHA·모델 경로를 대조했다. CPU strict loading, 모델 tensor SHA, optimizer 상태와 전체 step, RNG·shuffle 복원을 확인했다.
- 전체 train/validation 노출, 기존 B/Full과 공통 배치 prefix, 최초 최소 RMSE 선택과 patience40 종료, selected/last 재평가 기록·구간별 합계를 확인했다. native 합성60updates는 실제 fit step에서 제외했다.
- 추가 6조건의 optimizer step은 seed42 각1,742,384, seed52 각793,407, seed62 각1,166,775다. 마지막 epoch와 종료 후 추가 validation의 exposure SHA를 구분해 검증했다.
- 기존 Core 및 외부4종의 완료 감사와 선택/마지막 endpoint를 재사용했다. 7모델×3seed×2역할=42기록을 원래 동결 CSV 및 tensor SHA와 대조했다. 현재 checkout의 과학 코드로 결과를 다시 생성하지 않았다.
- 회수 시 해당 캠페인 소유 프로세스는 없었다. 첫 CPU 도구 선택에서 번들 Python에 torch가 없어 기존 5080 감사와 같은 `/usr/local/bin/python3` 환경을 사용했다. 패키지 설치·원격 Runtime 변경은 없다.
- 로컬 CPU 감사는 새 CUDA 예측 재현이 아니다. 저장된 CUDA RNG 바이트와 기존 native 검증 기록을 확인했다. raw 데이터와 held-out/test는 읽지 않았고 새 학습·재평가·GPU 측정은 실행하지 않았다.

## 논문 해석과 남은 순서

**Instacart 결과 분석 — 완료**

현재 결론은 “TitanTPP는 Instacart에서 일부 비교군보다 MAE가 조금 낮지만, 최상위 비교군 대비 수량 정확도 전반의 우위는 확보하지 못했다”다. Taxi·Intermittent에서의 기존 개선은 별도 데이터 결과로 유지하고, Instacart는 고수량 오차와 지표 간 상충을 보여주는 결과로 포함한다.

**원고와 공통 비교표에 최종 수치를 반영한다 — 다음 작업**

이번 요청에 따라 분석을 먼저 완료했으며 원고와 기존 보고서는 변경하지 않았다. 다음에는 Table 4의 Instacart Pending 두 행을 이번 3seed 결과로 교체하고, “미완료” 문구·관련 해석·주장 근거를 함께 갱신한다. 원고의 기존 수치가 오래된 상태라는 사실을 이 분석 완료와 혼동하지 않는다.

**독립 평가 규칙을 고정한다 — 이후 작업**

모델·비교군·선택 checkpoint·지표·평가 범위를 고정한 뒤 별도 승인 범위에서 독립 평가를 진행한다. 현재 validation 결과만으로 최종 일반화·통계적 유의성을 주장하지 않는다.

## 증거 파일

- [5090 원본 CPU 감사](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5090_20261001_v1/terminal_audit.json)
- [회수 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5090_20261001_v1/retrieval_receipt.json)
- [원본 파일 목록](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_additional_tpp_20260930_v1/retrieved/terminal_5090_20261001_v1/original/collection_manifest.json)
- [기계판독 분석](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/analysis.json)
- [선택·마지막 조건별 표](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/conditions.csv)
- [수량·이력 길이 구간별 표](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/strata.csv)
- [검증 결과](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/verification.json)
- [입력 파일 SHA 목록](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/source_manifest.json)
