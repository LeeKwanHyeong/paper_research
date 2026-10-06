# CNN+GRU54 3seed 고정 Test 평가와 비교

**현재 상태: 완료.** 2026-10-06 5080 전용 폴더에서 고정 checkpoint의 fresh Validation9 전역 gate를 통과한 뒤 Test9를 완료했다. 결과 원본 135파일과 archive SHA, 실제 프로세스 부재 및 supervisor exit0를 검증했다. 신규 학습·checkpoint 재선택은 없다.

CNN+GRU54는 Test에서 S2P2보다 세 데이터에 걸쳐 우세한 대표 모델이라고 확정할 수 없다. Intermittent 수량 개선은 남았지만 Taxi에서 Validation 수량 이득이 유지되지 않았고, 전체 TimeNLL 평균은 세 데이터 모두 손해다.

| Test 3seed 평균 | CNN+GRU54 전체 RMSE ± 표본 SD | S2P2 전체 RMSE ± 표본 SD | 전체 RMSE 변화 | 큰 수량 RMSE 변화 | 전체 TimeNLL 변화 |
|---|---:|---:|---:|---:|---:|
| Taxi | 130.4002 ± 30.3146 | 115.5415 ± 6.4023 | +12.86% | +41.42% | +92.74% |
| Intermittent | 2.8245 ± 0.2227 | 3.2154 ± 0.2743 | −12.16% | −41.01% | +32.21% |
| RAF | 39.9563 ± 0.4074 | 39.7729 ± 0.1609 | +0.46% | +0.85% | +6.95% |

변화는 CNN/S2P2의 같은 split·수량 구간 평균 차이이며 음수가 좋다. 큰 수량은 원래 Train의 Taxi3449·Intermittent187·RAF200 초과다. 개선 seed 수와 동일 seed 차이, MAE, TimeNLL 및 수량 구간의 절대 값은 아래 산출물에 모두 보존했다.

- [Test 전체·큰 수량 정렬표](TEST_TABLES.md): 외부 six, MLP4/8/12/16, 강한 내부 대조, 연구 기록 및 결정적 수량 기준선.
- [Validation 정렬표](VALIDATION_TABLES.md): 이번 fresh replay 결과. [원래 최종 Validation 비교](../titantpp_cnn_gru54_final_validation_comparison_20261006_v1/README.md)는 별도 동결 자료로 보존한다.
- [Contribution 판단](CONTRIBUTION_DECISION.md): 우위를 주장할 수 있는 범위, 실패한 범위와 다음 검증.
- [시간 일반화 위치 확인](TIME_GENERALIZATION_READOUT.md): 수량 구간별 aggregate NLL 분해.
- [seed별 값](metrics_per_seed.csv), [평균·표본 SD](metrics_aggregated.csv), [동일 seed 차이](paired_candidate_deltas.csv), [지표별 순위](metric_ranks.csv), [고정된 Test 판단 기준](strict_test_gate.json).
- [비교 영수증](comparison_receipt.json), [독립 QA 기록](QA_RECEIPT.json), [실제 종료 관측](native_terminal_observation.json).

**현재 기준선 — 완료**

- 원래 CNN+GRU54의 Taxi·Intermittent·RAF × seed42·52·62 9조건은 scientific_success, selected/last full Validation, checkpoint 원본 SHA 및 서버 종료가 검증됐다.
- 이전 시간 보정값 2개와 시간 출력부 130개 재적합은 이 Test 평가에 포함하지 않는다.

**고정된 checkpoint의 전체 Validation 재현 — 완료**

- 5080 전용 평가 폴더에서 원본 source117/123, seed별 source revision·선택 epoch·state SHA를 유지한다.
- full Validation9 gate는 14:11:55.826 KST에 완료됐고, 첫 Test claim은 14:11:55.897 KST 이후다. 모두 동일 수량-selected epoch·state SHA를 사용했다.

**동일 3seed의 Test와 외부·내부 비교 — 완료**

- 전체·큰 수량 RMSE/MAE/TimeNLL, 각 seed와 평균·표본 SD, 같은 seed의 차이를 구분한다.
- 기존 Test 분할은 이미 노출됐다. 이번 결과는 새 독립 미접근 평가의 증거가 아니다.
- Validation 비교에서 고정한 판단 규칙을 그대로 적용하고 불리한 지표와 seed를 보존한다.
- 14:14:36.458 KST의 실제 terminal 관측에서 소유 프로세스 및 GPU PID 부재, supervisor exit0를 확인했다. 회수 archive SHA는 `347149697709e0ea57f45d762362818b1f4b9f1ba42ea8c68806423fab544bc3`다.
- 기존 학습 조건123개(Deep Renewal 연구 기록9 포함)와 새 후보9개는 canonical 조건별 한 번 집계했다. 별도 결정적 수량 조건6개를 구분하고, 지표552행·집계200행을 검증했다. Deep Renewal은 원고/공통 출력부 우월성 주장에서는 제외한다.
- native 평가·회수와 독립 검산을 완료했고, gate·SHA·종료·승인 누락을 거부하는 오프라인 mutation8개를 Python `-O`에서도 통과했다. 사전 평가 어댑터 guard94개도 통과했다.

**Contribution의 주장 범위를 판단한다 — 완료**

- 수량 우위와 TimeNLL 손해를 별도로 판단한다. 세 데이터·모든 지표의 우월성이나 CNN/GRU 독립 기여를 임의로 확정하지 않는다.
- S2P2와 MLP16에 대한 고정된 강한 지배 기준은 Test에서도 세 데이터 모두 실패했다. 통합 승자·대표 개선 모델 채택은 확정하지 않는다.

**수량과 시간 일반화를 함께 개선할 대조 계약을 정한다 — 다음 작업**

- Taxi Test 수량 불안정과 Intermittent의 작은 수량 구간 TimeNLL 증가를 우선 다룬다. 현재 결과와 강한 MLP·S2P2를 기준선으로 보존한다.
- 이번 결과만으로 구조를 정하거나 추가 학습을 시작하지 않았다. 새 학습은 목적·고정 선택 규칙·데이터 사용 범위·3seed 대조를 계약으로 정한 뒤 실행한다.

**논문 주장에 대한 독립 평가와 인과 대조를 완료한다 — 다음 작업**

- 이미 개발에 노출된 Test와 구분된 독립 평가, CNN/GRU 및 문맥·잔차·파라미터 예산을 통제한 기여 대조, CPU 전체 binary 재추론 감사는 별도 미완료다.
- n=3의 표본 SD는 학습 seed 변동의 기술 통계이며 통계적 유의성 또는 독립 데이터 일반화 증거가 아니다. Instacart는 후속이고 이번 보고서에는 포함하지 않는다.

평가 구현은 `paper_research/master`의 `6f15e71c0e4be8e5c592f08295c1abed6d37b7e0`에 별도 커밋했다. 공용 Runtime·인증정보·다른 서버 작업·자동화를 변경하거나 저장소를 Push하지 않았다.
