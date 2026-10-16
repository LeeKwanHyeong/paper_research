# CNN과 외부 비교군의 Validation→Test 순위 변화

Material Passport: 기존 저장 endpoint·Test 예측만 재집계했다. 신규 학습·추론·원격 조회·checkpoint 재선택은 없다. CNN은 source131 구성요소 캠페인, 외부 6개는 기존 공통 head encoder 캠페인이다. 기존 접근 이력이 있는 chronological Test이다.

## 평균 결과와 seed별 방향

각 차이는 CNN−외부 모델의 3seed 평균이다. 음수는 CNN 우세다. RMSE 평균과 평균 MSE는 별도 값이며, 아래 오류 분해는 MSE에 수행했다.

| 데이터셋 | 비교 모델 | Validation RMSE 차이 | Test RMSE 차이 | Validation MAE 차이 | Test MAE 차이 | RMSE seed 순위 반전 수 |
|---|---|---:|---:|---:|---:|---:|
| Taxi | s2p2_matched_head | -19.7040 | 19.7053 | -5.2304 | 3.7794 | 2 / 3 |
| Taxi | attnhp_matched_head | -40.4790 | 3.9271 | -8.1883 | 1.8625 | 2 / 3 |
| Taxi | rmtpp | -8.0783 | -111.4004 | -0.7196 | -24.6617 | 0 / 3 |
| Taxi | thp | -20.9476 | -66.9271 | -4.6167 | -17.7333 | 1 / 3 |
| Taxi | sahp | -33.8773 | -13.0026 | -6.4332 | -0.2855 | 1 / 3 |
| Taxi | nhp | -232.1647 | -245.8294 | -66.4766 | -75.7171 | 0 / 3 |
| RAF | s2p2_matched_head | 0.4367 | -0.1723 | -0.0547 | -0.1237 | 3 / 3 |
| RAF | attnhp_matched_head | -0.4452 | -0.6030 | 0.0266 | -0.0118 | 0 / 3 |
| RAF | rmtpp | -0.1189 | -0.8625 | 0.0821 | 0.0658 | 1 / 3 |
| RAF | thp | -0.3131 | -0.9544 | -0.0831 | -0.1508 | 1 / 3 |
| RAF | sahp | -0.9879 | -1.4487 | -0.0149 | -0.2895 | 0 / 3 |
| RAF | nhp | -1.8417 | -2.5757 | 0.0390 | 0.0030 | 0 / 3 |
| Intermittent | s2p2_matched_head | -0.8529 | -0.5992 | -0.2002 | -0.2580 | 0 / 3 |
| Intermittent | attnhp_matched_head | -1.3570 | -9.6730 | -0.3056 | -3.2876 | 0 / 3 |
| Intermittent | rmtpp | -0.8270 | -1.5953 | -0.1592 | -0.6306 | 0 / 3 |
| Intermittent | thp | -1.0304 | -0.8678 | -0.1637 | -0.2506 | 0 / 3 |
| Intermittent | sahp | -5.3069 | -9.6363 | -1.1263 | -2.6153 | 0 / 3 |
| Intermittent | nhp | -11.1525 | -3.2903 | -3.7831 | -1.2661 | 0 / 3 |

## 공통 기준선 S2P2 대비 오류 반전의 산술 위치

전체 차이 변화는 Train 고정 수량 구간의 구성비 변화와 각 구간 내 상대 오류 변화로 정확히 분해된다. 이 분해는 오류 위치를 설명하며 원인의 인과적 증명이 아니다. 구성비 항에는 구간 내부 값·관측 이력의 분포 변화가 포함되지 않는다.

| 데이터셋 | 평균 MSE 차이 변화 | 구성비 변화 항 | 구간 내 오류 변화 항 | q>Train 경계의 변화 기여 |
|---|---:|---:|---:|---:|
| Taxi | 9388.82973 | 43.80065 | 9345.02908 | 3830.45211 |
| RAF | -43.56569 | 1.21795 | -44.78365 | -50.80252 |
| Intermittent | 0.12625 | 3.26334 | -3.13709 | 1.96539 |

## 확인된 사실과 해석의 한계

- Taxi의 CNN은 Validation RMSE·MAE에서 외부 6개 평균보다 좋았지만 Test에서는 S2P2와 AttNHP에 두 지표 모두 뒤졌다. 다른 네 외부 모델에는 Test 평균 두 지표가 여전히 좋다.
- Intermittent의 CNN은 Validation과 Test에서 외부 6개보다 평균 RMSE·MAE가 낮다. 외부 비교 기준의 전체 순위 반전으로 일반화할 수 없다.
- RAF는 Validation RMSE에서 S2P2에 뒤졌고, Test에서는 CNN의 평균 RMSE가 외부 6개 중 최저다. 이 방향은 악화 반전과 다르며 MAE도 별도 비교해야 한다.
- Test의 모든 외부 parquet 부분 SHA·대상 순서·truth SHA를 검증했다. CNN과 공통 10개 truth 필드 및 recorded time SHA도 재검증했다. Validation은 고정 선택 tensor·대상 수·5개 구간별 대상 수를 확인했으나 외부의 원시 Validation 대상 지문은 endpoint에서 복구할 수 없다.
- 5개 수량 구간의 구성비 변화가 작아도 구간 내부의 수량·이력·개체·시간 조건 변화는 남을 수 있다. 구성비 항이 작다는 사실은 모든 분포 변화가 없다는 증거가 아니다.
- CNN의 원시행 예측이 저장되지 않아 두 모델의 같은 행 오류를 시간·개체별로 연결할 수 없다. 외부 예측만으로 CNN의 특정 시점 실패를 추정하지 않았다.
- 외부 Validation의 signed error가 없어 외부의 Validation→Test 과소·과대 예측 변화는 계산할 수 없다. Test에서만 구간별 bias를 확인했다.
- Train 로그의 log 수량 손실은 raw 수량 RMSE와 직접 비교할 수 없다. 이 자료만으로 과적합, GRU 특성 또는 분포 변화 중 하나를 확정할 수 없다.
- 외부 학습 source·캠페인·Runtime과 CNN source131은 다르다. 같은 입력·선택·공통 head 비교이며 동일 Runtime 재학습 대조가 아니다. Intermittent seed42/52는 5090, seed62는 5080이므로 시간은 합치지 않았다.
- 3seed 평균 순위와 seed별 예외는 기술 통계다. 통계적 유의성·모든 데이터셋 우월성·원인 규명 완료를 주장하지 않는다.

전체 six-metric seed별 차이, 모든 외부 모델의 5개 bin 기여, 검증 SHA 및 보존 endpoint 학습 말기 지표는 EXTERNAL_ANALYSIS.json에 기록했다.
