# RAF 24조건 결과 — TitanTPP와 외부 TPP 비교

**24조건 학습·48개 selected/last validation 재평가와 원본 CPU 감사를 완료했다.** 2026-10-01 08:55:56 KST 큐 종료. 원본 회수는 2026-10-01T09:12:05.998600+09:00이다.

## 결과의 의미

대표 TitanTPP MLP의 3seed 평균 RMSE는 외부 최저 S2P2보다 0.130% 낮지만 MAE는 외부 최저 RMTPP보다 1.252% 높다. 정규화 변형은 8모델 중 평균 RMSE가 가장 낮으며 기존 MLP보다 0.155% 낮다. RAF는 큰 폭의 수량 우위를 추가 입증하는 결과가 아니라 지표에 따라 장단점이 갈리는 추가 데이터 결과다.

모든 값은 seed42·52·62의 평균 ± 표본표준편차이며 낮을수록 좋다. 세 지표는 동일한 최초 최소 validation raw 수량 RMSE checkpoint에서 얻었다.

| 모델 | MAE ↓ | RMSE ↓ | 시간 NLL ↓ |
|---|---:|---:|---:|
| TitanTPP MLP | 9.250602 ± 0.085484 | 33.950679 ± 0.058804 | 3.544626 ± 0.066929 |
| 활성 분기 정규화 | 9.232175 ± 0.097352 | 33.897914 ± 0.080386 | 3.563070 ± 0.097796 |
| RMTPP | 9.136205 ± 0.055678 | 34.550327 ± 0.250627 | 3.519108 ± 0.146304 |
| THP | 9.301358 ± 0.181900 | 34.744517 ± 0.320413 | 3.392755 ± 0.018745 |
| NHP | 9.179253 ± 0.088159 | 36.273078 ± 0.318641 | 3.388947 ± 0.055638 |
| SAHP | 9.233171 ± 0.020876 | 35.419269 ± 0.213047 | 3.741497 ± 0.074537 |
| S2P2 (공통 head) | 9.272988 ± 0.051684 | 33.994704 ± 0.126119 | 3.418729 ± 0.028388 |
| AttNHP (공통 head) | 9.191659 ± 0.043397 | 34.876644 ± 0.067567 | 3.970710 ± 0.060931 |

## 같은 seed의 비교

승수는 세 seed 중 기준 모델의 지표가 더 낮은 횟수다. 평균 차이나 승수는 통계적 유의성 검정이 아니다.

| 기준 모델 | 비교군 | MAE 승 | RMSE 승 | 시간 NLL 승 |
|---|---|---:|---:|---:|
| TitanTPP MLP | 활성 분기 정규화 | 1/3 | 1/3 | 2/3 |
| TitanTPP MLP | RMTPP | 1/3 | 3/3 | 1/3 |
| TitanTPP MLP | THP | 1/3 | 3/3 | 0/3 |
| TitanTPP MLP | NHP | 1/3 | 3/3 | 0/3 |
| TitanTPP MLP | SAHP | 1/3 | 3/3 | 3/3 |
| TitanTPP MLP | S2P2 (공통 head) | 2/3 | 2/3 | 0/3 |
| TitanTPP MLP | AttNHP (공통 head) | 1/3 | 3/3 | 3/3 |
| 활성 분기 정규화 | TitanTPP MLP | 2/3 | 2/3 | 1/3 |
| 활성 분기 정규화 | RMTPP | 1/3 | 3/3 | 1/3 |
| 활성 분기 정규화 | THP | 1/3 | 3/3 | 0/3 |
| 활성 분기 정규화 | NHP | 1/3 | 3/3 | 0/3 |
| 활성 분기 정규화 | SAHP | 1/3 | 3/3 | 3/3 |
| 활성 분기 정규화 | S2P2 (공통 head) | 3/3 | 2/3 | 0/3 |
| 활성 분기 정규화 | AttNHP (공통 head) | 1/3 | 3/3 | 3/3 |

## 조건별 선택과 성능

| 모델 | seed | 선택 / 종료 epoch | MAE | RMSE | 시간 NLL |
|---|---:|---:|---:|---:|---:|
| TitanTPP MLP | 42 | 9 / 49 | 9.286776 | 33.899339 | 3.558940 |
| 활성 분기 정규화 | 42 | 9 / 49 | 9.300672 | 33.818344 | 3.569101 |
| RMTPP | 42 | 30 / 70 | 9.075063 | 34.728736 | 3.687925 |
| THP | 42 | 50 / 90 | 9.216271 | 34.818224 | 3.379420 |
| NHP | 42 | 4 / 44 | 9.105754 | 36.046264 | 3.349053 |
| SAHP | 42 | 30 / 70 | 9.255864 | 35.542307 | 3.790944 |
| S2P2 (공통 head) | 42 | 21 / 61 | 9.311775 | 33.962798 | 3.386572 |
| AttNHP (공통 head) | 42 | 50 / 90 | 9.201472 | 34.856080 | 3.904763 |
| TitanTPP MLP | 52 | 5 / 45 | 9.312051 | 33.937865 | 3.471698 |
| 활성 분기 정규화 | 52 | 5 / 45 | 9.275119 | 33.979091 | 3.462398 |
| RMTPP | 52 | 15 / 55 | 9.149561 | 34.263786 | 3.429176 |
| THP | 52 | 15 / 55 | 9.177595 | 35.021655 | 3.384659 |
| NHP | 52 | 15 / 55 | 9.155006 | 36.135588 | 3.365283 |
| SAHP | 52 | 9 / 49 | 9.228865 | 35.542238 | 3.655766 |
| S2P2 (공통 head) | 52 | 27 / 67 | 9.292874 | 33.887602 | 3.440316 |
| AttNHP (공통 head) | 52 | 80 / 120 | 9.144196 | 34.952104 | 4.024916 |
| TitanTPP MLP | 62 | 20 / 60 | 9.152978 | 34.014833 | 3.603240 |
| 활성 분기 정규화 | 62 | 20 / 60 | 9.120734 | 33.896308 | 3.657710 |
| RMTPP | 62 | 8 / 48 | 9.183990 | 34.658460 | 3.440225 |
| THP | 62 | 20 / 60 | 9.510208 | 34.393674 | 3.414188 |
| NHP | 62 | 8 / 48 | 9.276998 | 36.637381 | 3.452505 |
| SAHP | 62 | 20 / 60 | 9.214783 | 35.173264 | 3.777781 |
| S2P2 (공통 head) | 62 | 22 / 62 | 9.214315 | 34.133711 | 3.429299 |
| AttNHP (공통 head) | 62 | 26 / 66 | 9.229309 | 34.821748 | 3.982451 |

## 데이터와 해석 범위

RAF는 월별 양의 부품 수요를 사건열로 표현했다. train 25,779개, validation 6,690개 target을 사용하며 최대 길이84에는 target이 포함된다. lookback84의 단위는 월이고 시간 스케일은6개월이다. 월별 합계나 미래 고정기간 수요를 직접 예측한 결과와 구분한다. held-out/test는 읽거나 평가하지 않았다.

정규화는 기존 고정 /8을 max(활성 분기 수,1)로 바꾼 변형이다. 기존 MLP와 같은 GPU·split·head·loss·seed에서 비교했지만 데이터에 따라 모델을 교체하지 않는다. 다른 데이터의 정규화 seed52·62 보류와 RAF에서 승인된 3seed를 구분한다.

quantity/history/additional_history별 모든 구간, 빈 구간, selected/last 기록은 comparison.json과 endpoint_roles.csv에 보존했다. B·Full·Gate는 이 RAF24조건에 포함되지 않아 RAF의 보완모듈 유무 효과로 해석할 수 없다.

## 원본 감사와 실측 비용

473개 원본 파일·104개 동결 source·48개 checkpoint를 SHA와 CPU strict loading으로 검증했다. optimizer step, RNG/shuffle, 공통 초기화 증거, 전체 표본 노출과 seed별 batch prefix, 최초 RMSE 선택과 early stop, 재평가 기록 및 구간 합계를 확인했다. 새 학습·forward·GPU 재평가는 수행하지 않았다. CPU와 학습 Runtime 차이는 terminal_audit.json에 기록했다.

첫 본학습 epoch 시작부터 큐 종료까지 87.449분, train_one 실측합 5123.965초, epoch 기록합 5112.595초다. 원래 중단·준비 시간까지 포함한 전체 운영 시간과 구분한다. 개인 RTX5080 추가 cloud 임대료는 $0, 전기료는 미측정이다. 서로 다른 모델의 종료 epoch가 달라 전체 fit 시간으로 속도 우위를 주장하지 않는다.

User interruption after qualification and pre-fit parent-guard correction preserved. All 24 scientific fits completed. Collector v1 rejected an operational test Python filename before streaming; read-only collection v2 completed. No scientific rerun.

## 증거

- [terminal_audit.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/terminal_audit.json) — SHA256 `70296dedeaf67d1da8f6df945ce2d7bfbf507998f0d57bca5357b21dab80df14`
- [retrieval_receipt.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/retrieval_receipt.json) — SHA256 `f84798ce6409142e4496efb1a6d29a8ddcc792091b6324809aae9cb1e6136abe`
- [execution_contract.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/original/execution_contract.json) — SHA256 `7b83022702d244688d13e8afbd404ed38975b9f09f533de69995b4bd7fd3d9fc`
- [collection_manifest.json](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_raf_5080_20260930_v1/retrieved/terminal_20261001_v2/original/collection_manifest.json) — SHA256 `4d972e0927534d560c51a5ea2ec4bfbc9479bd867dea56d0231077d3b5bb109e`
- [summarize.py](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_raf_execution_20261001_v1/summarize.py) — SHA256 `24e9f4760d49595986e0d10031cb42af621019064322531c1431e017e1741271`
