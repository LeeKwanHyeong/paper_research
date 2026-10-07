# Test에서는 고정 비중 검색이 수량에서 앞섰고, 가변 비중의 추가 효과는 재현되지 않았다

Taxi·seed42에서 5가지 방식 × 기존 두 GPU의 **10개 Validation 선택 checkpoint를 모두 Test 평가**했습니다. 5090·5080에서 fixed가 전체 수량 RMSE와 MAE가 가장 낮았습니다. 5090 Validation에서 앞섰던 adaptive는 Test에서 fixed·weighted보다 전체·큰 수량 RMSE가 모두 높았습니다. 현재 결과는 가변 검색을 대표 모델이나 확정적인 추가 Contribution으로 채택할 근거를 지지하지 않습니다.

Test 계산 완료와 원본 SHA·모델 state 불변·종료 확인을 마쳤습니다. 학습·checkpoint 재선택은 없었습니다. 두 결과는 같은 seed42의 GPU/Runtime 반복으로, 두 seed 평균이나 독립 표본으로 합산하지 않습니다.

## 전체 Test 결과

전체 8,327건, 큰 수량 83건입니다. 큰 수량은 Train에서 정한 수량 >3,449를 유지했습니다. 아래 6지표는 모두 각 방식의 고정된 **Validation 선택 epoch**에서 Test로 계산했으며 낮을수록 좋습니다. Test를 보고 새 epoch를 고르지 않았습니다.

| GPU | 방식 | 고정 선택 epoch | 전체 RMSE | 전체 MAE | TimeNLL | 큰 수량 RMSE | 큰 수량 MAE | 큰 수량 TimeNLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 5090 | mean | 80 | 149.245 | 41.511 | 1.1838 | 1051.682 | 902.231 | 4.078e-38 |
| 5090 | weighted | 59 | 126.887 | 37.864 | 0.9837 | 814.235 | 706.076 | 1.85e-42 |
| 5090 | fixed | 120 | 126.369 | 37.730 | 1.5533 | 814.309 | 674.457 | 1.629e-122 |
| 5090 | adaptive | 124 | 157.206 | 47.182 | 1.6189 | 948.317 | 853.189 | 8.039e-54 |
| 5090 | generic | 57 | 166.734 | 47.957 | 0.9422 | 1124.119 | 1010.472 | 4.527e-76 |
| 5080 | mean | 46 | 190.858 | 55.258 | 0.9129 | 1315.099 | 1216.279 | 1.223e-28 |
| 5080 | weighted | 80 | 141.108 | 41.263 | 1.1199 | 945.547 | 792.874 | 4.398e-52 |
| 5080 | fixed | 111 | 107.743 | 35.609 | 1.4585 | 550.547 | 416.318 | 1.937e-14 |
| 5080 | adaptive | 59 | 187.573 | 51.250 | 1.0107 | 1384.852 | 1165.797 | 6.001e-31 |
| 5080 | generic | 69 | 130.163 | 38.923 | 1.2062 | 834.686 | 659.574 | 2.621e-36 |

5090의 큰 수량 RMSE는 weighted 814.234950, fixed 814.309108입니다. fixed가 약 0.0091% 높으므로 두 방식이 정확히 동률이거나 fixed가 모든 수량 지표 1위라고 표현하지 않습니다. 5080은 fixed가 전체·큰 수량 RMSE/MAE 모두 가장 낮습니다.

## 가변 비중의 추가 효과와 시간 손해

변화율은 100 × (adaptive/비교 방식−1)입니다. 양수는 악화, 음수는 개선입니다. TimeNLL은 같은 Test epoch의 절대 차이(adaptive−비교 방식)이며 양수는 악화입니다.

| GPU | 비교 방식 | 전체 RMSE 변화 | 큰 수량 RMSE 변화 | 전체 MAE 변화 | 큰 수량 MAE 변화 | TimeNLL 차이 |
|---|---|---:|---:|---:|---:|---:|
| 5090 | mean | +5.33% | -9.83% | +13.66% | -5.44% | +0.4352 |
| 5090 | weighted | +23.89% | +16.47% | +24.61% | +20.84% | +0.6353 |
| 5090 | fixed | +24.40% | +16.46% | +25.05% | +26.50% | +0.0657 |
| 5090 | generic | -5.71% | -15.64% | -1.62% | -15.57% | +0.6767 |
| 5080 | mean | -1.72% | +5.30% | -7.25% | -4.15% | +0.0978 |
| 5080 | weighted | +32.93% | +46.46% | +24.20% | +47.03% | -0.1092 |
| 5080 | fixed | +74.09% | +151.54% | +43.93% | +180.03% | -0.4478 |
| 5080 | generic | +44.11% | +65.91% | +31.67% | +76.75% | -0.1955 |

adaptive는 fixed보다 전체/큰 수량 RMSE가 5090에서 **24.40%/16.46%**, 5080에서 **74.09%/151.54%** 높습니다. 5090에서는 generic보다 두 RMSE가 낮지만 mean보다 전체 RMSE가 높고, fixed·weighted를 넘지 못했습니다. 5080은 mean보다 전체 RMSE가 1.72% 낮아도 큰 수량 RMSE가 5.30% 높습니다. 유리한 비교만으로 이력에 따른 검색 기준 조절의 추가 효과를 주장할 수 없습니다.

fixed는 mean보다 전체/큰 수량 RMSE를 5090에서 15.33%/22.57%, 5080에서 43.55%/58.14% 낮췄습니다. 하지만 fixed의 전체 TimeNLL은 mean보다 5090에서 +0.3695(31.22%), 5080에서 +0.5456(59.77%) 높습니다. 수량 개선과 시간 손해가 함께 나타났습니다. adaptive는 5090에서 fixed보다 TimeNLL도 +0.0657 높고, 5080에서는 fixed보다 −0.4478 낮습니다.

TimeNLL은 동결된 heteroscedastic lognormal duration head의 positive_integer_round_clamp_v1 관측 계약, hour 단위, top-code 없음, sigma floor 0.001에서 계산했습니다. 시간 예측 RMSE나 다른 모델의 별도 시간 지표로 바꾸어 해석하지 않습니다. 큰 수량 TimeNLL은 거의 0이므로 비율로 우열을 주장하지 않습니다.

## Validation에서의 기대와 Test에서의 결과

아래는 서로 다른 split의 지표를 나란히 기록한 것이며 섞어 집계한 성과가 아닙니다.

| GPU | 방식 | Validation 전체 RMSE | Test 전체 RMSE | Validation 큰 수량 RMSE | Test 큰 수량 RMSE |
|---|---|---:|---:|---:|---:|
| 5090 | mean | 84.878 | 149.245 | 427.843 | 1051.682 |
| 5090 | weighted | 86.253 | 126.887 | 411.523 | 814.235 |
| 5090 | fixed | 83.420 | 126.369 | 412.485 | 814.309 |
| 5090 | adaptive | 81.436 | 157.206 | 380.695 | 948.317 |
| 5090 | generic | 85.959 | 166.734 | 465.053 | 1124.119 |
| 5080 | mean | 86.900 | 190.858 | 473.329 | 1315.099 |
| 5080 | weighted | 83.153 | 141.108 | 408.028 | 945.547 |
| 5080 | fixed | 83.541 | 107.743 | 386.151 | 550.547 |
| 5080 | adaptive | 86.117 | 187.573 | 441.273 | 1384.852 |
| 5080 | generic | 84.905 | 130.163 | 449.328 | 834.686 |

5090 Validation의 adaptive는 fixed보다 전체 RMSE 2.38%, 큰 수량 RMSE 7.71% 낮았으나 Test에서는 반대였습니다. 5080에서 이미 보였던 반복 안정성 문제에 Test 일반화 문제까지 확인됐습니다. 원인(과적합, 시기별 데이터 차이, 최적화 민감도 등)은 이 계산만으로 특정하지 않습니다.

## 비교한 방법과 Contribution 범위

CNN+GRU54 이력 보정, 학습된 정적 prototype64, top4, 과거32 관측 사건 요약은 공통 기준입니다. mean은 top4 평균, weighted는 유사도 가중합, fixed는 시간·수량 검색 점수의 0.5/0.5 결합, adaptive는 이력에 따른 가변 비중, generic은 같은 추가 parameter 예산의 일반 학습 검색입니다. 추가 parameter는 각각 0/0/512/521/521입니다. 실제 과거 사건 bank나 prototype 내용 자체를 새로 개선했다고 표현하지 않습니다.

현재 수량 결과는 fixed를 더 검토할 대상으로 지지합니다. fixed에는 mean·weighted보다 추가 projection parameter가 있으므로 이 결과만으로 검색 정보를 나눈 설계의 독립적인 인과 효과나 참신성을 확정할 수 없습니다. adaptive 고유의 추가 효과는 이번 Test에서 지지되지 않았습니다. 기존 MLP8 54조건 이력 보정 대조는 별도 증거로 보존하고 CNN+GRU54 검색부의 기여로 옮기지 않습니다.

## 비용과 실행 상태

기존 native qualification의 adaptive 비용을 원본 Validation 분석에서 재사용했습니다. B128, warm5/measure20, T64/T256에서 mean 대비 추론 +20%, 학습 step +30%, peak allocation +20%의 사전 한도는 양 GPU에서 통과했습니다. 이번 Test에서 비용을 재측정하거나 사후 기준을 바꾸지 않았습니다.

| GPU | 길이 | adaptive 추론 증가 | 학습 step 증가 | peak allocation 증가 | 기존 비용 판정 |
|---|---:|---:|---:|---:|---|
| 5090 | 64 | 8.43% | 4.82% | 3.16% | 적격 |
| 5090 | 256 | 2.43% | 1.74% | 2.59% | 적격 |
| 5080 | 64 | 8.37% | 4.79% | 3.16% | 적격 |
| 5080 | 256 | 2.57% | 1.67% | 2.59% | 적격 |

비용 적격은 수량 성능의 추가 효과와 별도입니다. Test 파이프라인의 실제 시간은 5080 22.50초, 5090 21.54초로, 각 GPU의 모델 5개 로드·추론·검증 전체를 포함합니다. 방식별 순수 추론 지연이나 학습 효율 비교로 사용하지 않습니다. 전체 파이프라인 CUDA peak allocation은 양쪽 408,809,472 bytes이며 방식별 비용 비교가 아닙니다. 기존 실제 학습 epoch 시간은 별도 Validation 보고서에 보존했습니다.

| GPU | 실제 종료 관측 KST | 완료 | 실패 | 진행 | 대기 | 미확정 | 소유 평가 PID |
|---|---|---:|---:|---:|---:|---:|---:|
| 5090 | 2026-10-08T08:03:20.416119+09:00 | 5 | 0 | 0 | 0 | 0 | 0 |
| 5080 | 2026-10-08T08:03:20.313136+09:00 | 5 | 0 | 0 | 0 | 0 | 0 |

## 검증과 Test 노출 이력

평가 전에 10개 selected checkpoint의 전체/큰 수량 Validation 6지표를 원래 GPU·Runtime에서 먼저 재현했습니다. 최대 절대 편차는 1.4210854715202e-14이며 사전 허용오차 1e-5 안입니다. 양쪽 10개 재현 완료 gate를 확인한 다음 Test를 각 한 번 실행했습니다. 평가 전후 모델 state와 checkpoint bytes가 동일하고, source126·전체 parquet·원래 split manifest SHA가 일치했습니다. 독립 검토에서도 5개 수량 구간의 충분통계 합과 전체/큰 수량 RMSE·MAE·TimeNLL 재계산이 일치했습니다.

평가는 torch.no_grad 및 eval 모드에서 target 수량·target memory write를 가린 동결 코드로 실행했습니다. 시계열 Test의 각 예측에는 그 시점 이전에 관측된 사건만 입력했고 앞선 Test 사건은 관측 후 이후 예측의 prefix로 사용할 수 있습니다. Test로 parameter·통계·시간 head를 학습하지 않았습니다. 원시 예측 행은 저장하지 않았고 집계 충분통계만 보존했습니다.

**이 Taxi Test는 과거 연구에서 이미 열람된 데이터입니다.** 이번 10개 checkpoint의 Test 평가는 처음이지만 untouched 독립 Test를 새로 확보한 것은 아닙니다. 단일 데이터·단일 seed의 고정 checkpoint 평가이며 통계적 유의성, 새 독립 평가, 외부 모델 우월성이나 최종 모델 선택을 확정하지 않습니다. 외부 비교군과 MLP8 모델의 Test를 이번에 추가 실행·열람하지 않았습니다.

CPU binary metadata 검증과 native GPU Validation/Test 재추론은 완료했습니다. 별도 CPU binary 재추론 감사와 새 독립 평가는 미완료입니다. 확대 학습은 실행하지 않았습니다.

## 원본 및 보고서 파일

- [고정 평가 계약](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_retrieval_test_20261008_v1/execution_contract.json)
- [10실행 Test 지표 CSV](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_test_20261008_v1/selected_Test_metrics.csv)
- [계산·비교·원본 SHA JSON](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_test_20261008_v1/analysis.json)
- [5090 원본 Test 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_retrieval_test_20261008_v1/results/test_5090.json)
- [5080 원본 Test 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_retrieval_test_20261008_v1/results/test_5080.json)
- [전체 Validation 재현 gate](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_retrieval_test_20261008_v1/validation_gate.json)
- [Test 완료·출력 SHA 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_retrieval_test_20261008_v1/analysis/completion_receipt.json)
- [보존된 기존 Validation 보고서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contextual_retrieval_pilot_results_20261008_v1/README.md)

동결 science revision은 `4274043717cf24640d9a18fb35231a1f7ba50c8e`이고 이번 평가 도구의 operational revision은 `0f6c6e96820f5c0d6e108d2f7a320428a7794078`입니다. 기존 학습 계약·소스·선택을 변경하지 않았습니다.
