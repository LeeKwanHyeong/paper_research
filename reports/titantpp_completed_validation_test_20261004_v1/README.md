# 완료 조건의 Validation·Test 전체 비교

보고서 생성 시각: 2026-10-04T05:54:09.195968+00:00. 원래 Validation에서 선택한 체크포인트를 유지한 기존 분할 재평가다.

**학습 조건 153개를 빠짐없이 연결했고, Validation·Test 306행과 결정적 수량 기준선 16행을 산출했다. 전체·큰 수량을 합쳐 원시 지표 644행이다. 폭16을 모든 데이터의 대표 설정으로 고정할 근거는 부족하다.**

## 완료 범위와 비교 기준

| 묶음 | 학습 조건 | seed | 이번 처리 |
|---|---:|---|---|
| 기존 공통 출력부 MLP·외부 비교군·구조 대조 | 108 | 42·52·62 | 원본·예측·선택 기록을 검증해 재사용 |
| 기존 TitanTPP B | 9 | 42·52·62 | Taxi·Intermittent·Instacart의 비교 가능한 결과 재사용 |
| MLP 폭16 | 12 | 42·52·62 | 24 split 중 검증된 4개 재사용, 나머지 20개 추론 완료 |
| A100 후보 두 캠페인 | 12 | 42만 | 원래 후보 12개의 Validation 재현 후 Test 12개 완료 |
| Deep Renewal native NB | 12 | 42·52·62 | Validation 12개 재현 후 Test 12개 완료; 원고 제외 유지 |

수량만 예측하는 마지막 관측 수량·관측 이력 평균도 네 데이터의 두 split에 포함했다. RAF의 비교 가능한 B와 A100 후보의 Instacart는 미편성이므로 수치를 만들지 않았다. 폭8·12와 CNN·GRU는 완료 실험에 포함되지 않는다.

- 모든 모델은 같은 target·truth·관측 이력과 고정 전처리를 사용했다. 다음 사건의 target이나 미래 이력은 입력하지 않았다.
- 체크포인트·선택 epoch·동결 source·loader·데이터와 결과 영수증의 SHA를 대조했다. 새 학습, 분할 변경, Test에 따른 epoch 재선택은 수행하지 않았다.
- 각 seed 안에서 사건별 오차를 집계하고, seed42·52·62의 산술 평균과 표본 표준편차를 구했다. seed 평균은 ensemble 점수가 아니다. A100은 같은 seed42끼리만 비교한다.
- Test는 이미 접근했던 기존 분할이다. 새 독립 평가나 미접근 자료로 표현하지 않는다. 표준편차는 seed 간 변동이며 유의성 검정이나 일반화 보장을 대신하지 않는다.
- Time NLL은 같은 기록 정수 간격의 확률질량 점수다. 공통 출력부는 round/clamp lognormal, Deep Renewal은 native shifted-NB다. 데이터별 단위는 Taxi 시간·Intermittent 주·RAF 월·Instacart 일이며, Instacart만 30일 상한 코딩을 적용했다. 서로 다른 데이터의 NLL을 합쳐 순위를 만들지 않는다. Native 모델과의 차이는 출력부로 통제된 backbone 효과가 아니다.

## 전체 비교표

- [Validation 전체: 모든 3seed 비교군의 RMSE·MAE·Time NLL](VALIDATION_TABLES.md)
- [Test 전체: 모든 3seed 비교군의 RMSE·MAE·Time NLL](TEST_TABLES.md)
- [Validation seed42: 기존 모델·폭16·A100 네 후보](VALIDATION_SEED42_TABLES.md)
- [Test seed42: 기존 모델·폭16·A100 네 후보](TEST_SEED42_TABLES.md)
- [Validation 큰 수량 · 3seed](VALIDATION_TAIL_TABLES.md) / [Test 큰 수량 · 3seed](TEST_TAIL_TABLES.md)
- [Validation 큰 수량 · seed42](VALIDATION_SEED42_TAIL_TABLES.md) / [Test 큰 수량 · seed42](TEST_SEED42_TAIL_TABLES.md)
- [모든 seed 원시 지표](results/metrics_per_seed.csv) / [평균·표본 표준편차](results/metrics_aggregated.csv)
- [폭4 대비 전체 비교 차이](results/all_vs_width4.csv) / [폭4·16의 seed별 차이](results/width16_vs_width4_per_seed.csv)

## 폭 확대의 효과

아래는 raw 수량 RMSE 평균 ± 표본 표준편차다. MAE·Time NLL을 포함한 전체 결과는 위 표에 연결했다.

| 데이터 | Validation 폭4 | Validation 폭16 | Test 폭4 | Test 폭16 | Test S2P2 · 공통 출력부 |
|---|---:|---:|---:|---:|---:|
| Taxi | 79.711 ± 2.826 | 77.826 ± 4.161 | 118.624 ± 19.570 | 113.393 ± 16.502 | 115.542 ± 6.402 |
| Intermittent | 1.673 ± 0.082 | 1.687 ± 0.048 | 2.761 ± 0.079 | 2.936 ± 0.083 | 3.215 ± 0.274 |
| Instacart | 5.883 ± 0.005 | 5.874 ± 0.003 | 5.980 ± 0.005 | 5.970 ± 0.004 | 5.947 ± 0.011 |
| RAF | 33.951 ± 0.059 | 34.083 ± 0.087 | 39.795 ± 0.075 | 40.248 ± 0.169 | 39.773 ± 0.161 |

| 데이터 | 폭16 Test RMSE 변화 | MAE 변화 | Time NLL 변화 | 큰 수량 Test RMSE 폭4 → 폭16 |
|---|---:|---:|---:|---:|
| Taxi | -4.41% | -3.12% | -4.75% | 646.423 → 594.683 |
| Intermittent | +6.32% | +7.87% | +197.76% | 14.962 → 13.785 |
| Instacart | -0.17% | -0.07% | +0.03% | 24.133 → 23.982 |
| RAF | +1.14% | +0.92% | -3.96% | 406.965 → 414.097 |

변화의 음수는 해당 지표 개선, 양수는 악화다. 보정 파라미터는 6,144→24,576이지만 전체 파라미터는 96,003→114,435로 약 19.2% 증가한다.

**Taxi:** 폭16 평균 수량 지표는 개선됐지만 Test RMSE의 폭16−폭4 차이는 seed42 −44.530, seed52 +26.054, seed62 +2.783이다. 평균 이득이 seed42에 의존한다. 기존 모든 가용 분기 MLP의 Test 평균 RMSE 99.304, MAE 32.125, tail RMSE 503.640도 폭16보다 낮다. 따라서 단계적 분기 활성화나 폭 확대가 일관되게 우월하다는 결론은 지원되지 않는다.

**Intermittent:** 폭16 Test 전체 RMSE·MAE·Time NLL은 세 seed 모두 악화했다. 평균 tail RMSE는 개선됐으므로 큰 수량 개선을 누락하지 않되, 전체 손해와 함께 보고한다. 기존 B의 Test RMSE 2.636과 모든 가용 분기 MLP의 2.660이 폭16보다 낮다. 폭16이 S2P2 평균 3.215보다 낮다는 사실만으로 기존 MLP보다 낫다고 판단할 수 없다.

**RAF:** 폭16의 Validation·Test·tail RMSE는 세 seed 모두 악화했다. 평균 Time NLL 개선은 별도 장점으로 유지한다. 현재 상태만 사용하는 파라미터 대조의 Test RMSE 39.464도 반드시 함께 제시한다.

**Instacart:** 폭16의 Test 전체 RMSE·MAE는 세 seed 모두 소폭 개선됐다. Validation 전체 RMSE도 세 seed에서 개선됐지만 모든 수량 구간·지표가 일관되게 개선된 것은 아니다. S2P2와 다른 일부 공통 출력부 비교군은 더 낮은 Test RMSE를 보였다. 별도 연구군 Deep Renewal의 Test RMSE 5.880은 더 낮으나 MAE 4.087과 Time NLL 2.881은 폭16보다 높다. Deep Renewal의 Taxi·Intermittent·RAF 수량 결과가 불리한 것도 모두 보존했다. 기존 사용자 결정에 따라 원고 비교에서는 제외한 상태다.

## A100 후보 · 같은 seed42 비교

표는 단일 seed의 raw RMSE다. 네 후보는 이종 GPU에서 학습한 탐색이며 3seed 확정 결과와 합치지 않는다. 전체 MAE·Time NLL과 큰 수량 결과는 seed42 표에서 확인한다.

| 모델 | Taxi Val | Taxi Test | Intermittent Val | Intermittent Test | RAF Val | RAF Test |
|---|---:|---:|---:|---:|---:|---:|
| TitanTPP MLP 폭4 | 82.484 | 140.462 | 1.578 | 2.748 | 33.899 | 39.866 |
| TitanTPP MLP 폭16 | 82.499 | 95.932 | 1.632 | 2.899 | 33.984 | 40.407 |
| Cross-product · A100 seed42 | 82.435 | 103.926 | 1.994 | 2.715 | 34.200 | 41.650 |
| Recent-four mean · A100 seed42 | 87.371 | 115.523 | 1.635 | 2.681 | 34.479 | 39.753 |
| Recent-four attention · A100 seed42 | 81.614 | 129.563 | 1.613 | 2.515 | 34.634 | 39.764 |
| Post-block · A100 seed42 | 86.991 | 162.128 | 1.659 | 2.922 | 34.094 | 39.767 |

Recent-four attention의 Intermittent Test RMSE 2.515와 MAE 1.058은 같은 seed42 비교에서 유리하지만 Time NLL 21.878은 기존 MLP의 14.768보다 높다. Cross-product는 같은 Intermittent Test에서 RMSE 2.715·MAE 1.095·Time NLL 6.990으로 세 지표가 개선됐지만 Validation 수량 지표는 악화했다. RAF의 작은 수량 차이와 Taxi post-block 악화도 보존한다. 어느 후보도 여러 데이터·모든 지표에서 일관된 우위를 확보한 상태가 아니다.

Recent-four attention은 참조 범위와 가중 방식을 함께 바꿨으므로 attention 단독 효과로 해석하지 않는다. A100 학습 성능과 이번 5080 평가 시간을 논문 계산 효율 비교로 혼합하지 않는다.

## Intermittent Time NLL 악화의 관측 분해

폭4·16의 동일 Test target 88,019개를 대조했다. 폭16 NLL 증가는 거의 모두 quantity body에 집중됐다. 세 seed의 손실 증가분 중 49.63%는 5–13주, 24.11%는 14–52주에서 발생했다. >52주 관측은 없다. 폭16 상위 NLL 1% 관측이 총 손실의 43.96–62.69%를 차지했다. 이것은 손실 집중도이며, 과신·분산 축소·분포 이동 등의 원인을 증명하지 않는다.

[고정 gap·수량 구간 분해와 증적](diagnostics/README.md). Test 결과를 사용한 이 진단은 사후 개발 분석으로 구분한다.

## 실행·검증 증적

승인된 5080 전용 작업 폴더와 기존 Runtime에서 한 번에 GPU worker 한 개만 실행했다. 새 학습·임대·공용 Runtime 변경은 없었다. 폭16·Deep Renewal·A100 평가의 소유 프로세스와 GPU PID 종료를 확인했고 결과 원본을 SHA 검증해 회수했다.

A100 첫 기술 시도는 기존 CUDA NVRTC 라이브러리 경로 누락으로 Validation 첫 조건의 예측 저장 전에 중단됐다. Test는 시작되지 않았다. 실패 증거를 보존한 뒤 byte-identical evaluator와 동일 과학 설정을 새 전용 폴더에서 실행했고, 기존 라이브러리의 per-process 경로만 명시해 12 Validation gate와 12 Test를 완료했다. 이 오류를 학습 실패나 새로운 과학 조건으로 중복 집계하지 않았다.

| 증적 | 확인 범위 |
|---|---|
| [기존 모델 원본 검증](existing/verification.json) | 117 체크포인트, 기존 Test 117·Validation raw 13조건, 나머지 Validation 선택 endpoint 104조건의 SHA 연결 |
| [폭16 최종 검증](width16/final_verification.json) | 12조건·24 split, 20개 신규 추론·4개 재사용, 구간 합 검증 |
| [Deep Renewal 완료](deep_renewal/completion_receipt.json) | 12 Val·12 Test, 원본 560파일 SHA, 동일 population·native score 정의 |
| [A100 완료 검증](a100_attempt2/final_verification.json) | 12 Val·12 Test, 원본 273파일 SHA, 동일 source·selected·population |
| [A100 첫 기술 실패](a100/first_attempt_failure_evidence.json) | 예측 0행·Test 미진입, 원래 실패 기록 보존 |
| [전체 집계 영수증](comparison_receipt.json) | 153조건·644행 정확한 범위, 중복 없음·count/threshold 일치 |
| [독립 집계 검토](comparison_review.json) | 평균·표본 표준편차·population·출력 비교 독립 확인 |

이번 작업은 원고에 연결할 selected checkpoint의 검증·추론 범위다. 모든 last checkpoint의 CPU 감사나 전체 연구 감사를 완료했다고 주장하지 않는다. 원고 main.tex는 수정하지 않았다.

## 남은 작업 순서

**1. 공통 대표 설정과 열화 기준을 정한다 — 다음 작업 / 최우선**

- 대상: paper_research의 후속 실험 계약. 폭4를 기존 비교 기준선으로 보존하고 폭16은 완료된 용량 대조군으로 남긴다. 폭16 일괄 고정은 보류하는 편이 타당하다.
- 수량 전체·tail·시간 지표와 seed별 방향을 함께 판단하는 기준을 고정한다. 기존에 확인한 Test로 대표 모델을 다시 골라 독립 최종 결과로 포장하지 않는다.
- Intermittent의 시간 예측 악화는 큰 수량 오차와 분리해, train·validation에서 시간 분포·손실·공유 표현을 확인하는 진단을 설계한다.
- 완료 조건: 무엇을 개선으로 인정하고 무엇을 허용하지 않을지, checkpoint 선택과 평가 범위를 문서로 설명할 수 있는 상태.

**2. 계획된 폭8·12 대조로 용량 효과를 확인한다 — 다음 작업 / 1번 이후**

- 대상: paper_research와 5080·5090 실험 계획. 기존 폭4·16 원본은 재사용하고, 같은 분기·/8·초기화·출력부·손실·선택 기준에서 폭8·12만 바꾼다.
- 이미 정한 네 데이터·seed42·52·62 범위를 구체적인 실행 계약에 연결한다. 서로 다른 서버의 독립 조건은 병렬로 배치할 수 있다.
- 큰 수량만의 개선과 전체·시간 손해, Taxi의 seed 의존성을 구분한다. Test를 폭 선택 지표로 사용하지 않는다.
- 완료 조건: 용량 효과가 중간 폭에서도 나타나는지 Validation 3seed와 train 진단으로 확인한 상태. 이번 비교 작업에서 새 학습을 시작하지 않았다.

**3. CNN·GRU의 기여를 분리한 구조 대조를 확정한다 — 다음 작업 / 기준 계약 이후**

- 대상: paper_research의 Encoder 1·중간 보정 구현. CNN만, GRU 보정만, CNN+GRU를 기존 기준선과 비교해 두 요소의 효과를 구분한다.
- 기존 64차원 잔차 경로, 과거 참조 범위, CNN 채널·kernel, GRU 잠재 폭, 삽입 위치·초기화·파라미터 예산·padding·개체별 상태 reset을 먼저 확정한다.
- 단일 GRU로 8개 MLP 분기 전체를 대체하면 가용성·/8까지 달라지는 구조 변경이므로 단순 연산자 교체나 폭 대조와 구분한다. 결합 성능이 좋아질 것이라고 미리 단정하지 않는다.
- 완료 조건: 기존 구조 대비 변경 표와 인과성·계약 테스트가 마련되고, 정확한 실행 서버·자원·조건 범위가 검토 가능한 상태.

**4. 확정된 주장과 원고를 갱신한다 — 다음 작업 / 검증 결과 이후**

- 대상: paper/titantpp_pakdd_2027_draft/main.tex. 현재 결과는 데이터·수량 구간별 조건부 이득과 한계를 함께 설명한다.
- 인접 상태 보정, 분기 활성화, 용량, CNN·GRU 효과를 별개 주장으로 연결한다. 효율은 같은 환경의 실제 측정 범위만 사용한다.
- 이번 기존 분할 재평가와 향후 미접근 평가를 구분하고, A100 seed42·native NB 연구 기록·불리한 결과를 누락하지 않는다.
- 완료 조건: 초록·본문·Appendix의 숫자와 주장이 검증된 원본 및 split에 일치하는 상태.
