# TitanTPP 후속 완료 13조건 원본 감사 및 validation 비교

작성: 2026-10-02 11:06:04 KST

**완료 13조건의 checkpoint 26개와 서버별 동결 source 109개를 회수하여 CPU 검증을 통과했다.** 범위는 2026-10-02 10:51:58 KST 관측에서 완료된 Taxi 9조건, Intermittent 2조건, Instacart 2조건이다. 학습 전체 완료 보고가 아니며, 이후 완료 조건은 이 감사에 포함하지 않았다.

## 1. 회수 및 검증 결과

| 서버 | 완료 조건 | binary checkpoint | 회수 파일¹ | 회수 byte | 결과 |
|---|---:|---:|---:|---:|---|
| 5080 | 11 | 22 | 283 | 27,419,183 | CPU 검증 통과 |
| 5090 | 2 | 4 | 145 | 6,162,227 | CPU 검증 통과 |

¹ collection_manifest.json 자체를 제외한 SHA 검증 파일 수. 두 서버의 동일 source 사본은 각각 검증했다.

- 계약 SHA, source 109개 closure, 승인·실행 permit, native qualification, 종료 명세와 회수 전후 파일 SHA를 연결했다.
- selected/last checkpoint를 frozen source로 구성한 모델에 CPU strict loading했다. tensor SHA·유한성·선택 epoch·embedded best를 대조했다.
- last checkpoint의 optimizer tensor/step·RNG·shuffle state를 복원했고 전체 train 노출·기존 MLP와 공통 batch prefix·validation 모집단을 확인했다.
- 최초 최소 raw 수량 RMSE 선택, 종료 epoch, selected/last validation 재평가 기록의 MAE·RMSE·시간 NLL 및 구간 합계를 대조했다.
- 마지막 epoch의 exposure 영수증은 학습 종료 후 추가 validation 1회를 제외한 당시 상태와 일치한다. 원본 파일은 수정하지 않았다.
- 서버에서는 종료된 조건의 파일만 읽었다. 새 학습·모델 forward·재평가·GPU 측정은 수행하지 않았다. 기존 84조건 및 효율 감사는 재사용했다.

CPU strict loading과 기록 대조를 완료한 것이며 예측을 새로 계산한 것은 아니다. CUDA 초기화·정확성 검증은 SHA로 연결된 기존 native 증거를 재사용했다. 서버별 qualification의 합성 75updates(정확성 15 + 비용 측정 60)는 scientific fit에 포함하지 않는다.

## 2. Taxi — 세 seed 완료 비교

세 seed 42·52·62의 validation 지표 평균 ± 표본 표준편차(ddof=1). 모든 지표는 같은 RMSE 선택 epoch에 대응하며 낮을수록 좋다.

| 모델 | MAE | RMSE | 시간 NLL |
|---|---:|---:|---:|
| 기존 History MLP | 25.6237 ± 0.8320 | 79.7111 ± 2.8260 | 1.0387 ± 0.2843 |
| 현재 상태 전용·파라미터 일치 | 28.8130 ± 0.4688 | 90.0481 ± 1.5131 | 0.8818 ± 0.1827 |
| 전체 분기 사용·고정 /8 | 25.3290 ± 0.2945 | 78.1212 ± 0.7114 | 1.1907 ± 0.3542 |
| Deep Renewal (event-native NB) | 91.4747 ± 1.0467 | 404.7864 ± 2.9784 | 0.7902 ± 0.0159 |

- 기존 MLP는 현재 상태 전용 변형 대비 MAE 11.07%, RMSE 11.48% 낮다. 수량 두 지표에서 세 seed 모두 같은 방향이며, 인접 상태 결합의 유용성을 뒷받침한다. 시간 NLL은 현재 상태 전용 변형이 평균적으로 낮다.
- 전체 분기 사용 변형은 기존 MLP보다 평균 MAE 1.15%, RMSE 1.99% 낮다. seed별 개선은 MAE 1/3, RMSE 2/3이며 시간 NLL 평균은 악화한다. 단계적 분기 가용성 규칙이 항상 우월하다는 주장은 지지하지 않는다.
- 기존 MLP는 이번 Deep Renewal adapter보다 수량 오차가 크게 낮고, Deep Renewal은 시간 NLL이 낮다. Deep Renewal은 세 seed 모두 최대 300epoch에서 선택됐으므로 수렴 완료나 모든 Deep Renewal 설정에 대한 우월성으로 확대하지 않는다.
- 두 내부 대조군은 기존 MLP와 파라미터 수·공통 head·수량 loss를 맞췄다. Deep Renewal은 고유 NB 시간·수량 손실을 쓰므로 동일 head/loss 비교로 표현하지 않는다.

## 3. Intermittent·Instacart — seed42 완료분

| 데이터 | 모델 | 선택/완료 epoch² | MAE | RMSE | 시간 NLL |
|---|---|---:|---:|---:|---:|
| Intermittent | 기존 History MLP | 10/기존 감사 | 0.638480 | 1.578487 | 0.346835 |
| Intermittent | 현재 상태 전용·파라미터 일치 | 21/61 | 0.745193 | 1.773053 | 0.352289 |
| Intermittent | 전체 분기 사용·고정 /8 | 10/50 | 0.675898 | 1.682415 | 0.358399 |
| Instacart | 기존 History MLP | 30/기존 감사 | 3.991502 | 5.878730 | 2.803218 |
| Instacart | 현재 상태 전용·파라미터 일치 | 30/70 | 3.990017 | 5.872238 | 2.803320 |
| Instacart | 전체 분기 사용·고정 /8 | 30/70 | 3.992629 | 5.872499 | 2.803145 |

² 기존 MLP의 완료 epoch는 이번 신규 감사 범위에 포함하지 않았으며 이미 확인된 선택 결과를 재사용했다.

- Intermittent seed42에서는 기존 MLP가 두 변형보다 MAE·RMSE·시간 NLL 모두 낮다. 아직 한 seed 결과이므로 세 seed 결론을 만들지 않는다.
- Instacart seed42에서 두 변형의 RMSE는 기존 MLP보다 약 0.11% 낮은 수준이다. MAE·시간 NLL의 방향은 변형별로 다르며, 전반적인 개선으로 묶지 않는다.
- 이 두 데이터의 Deep Renewal과 seed52·62, RAF는 이번 고정 13조건 분석 대상이 아니다. 결과를 누락한 성공 집계가 아니라 별도 대기 범위로 보존한다.

## 4. 주장에 미치는 영향

Taxi의 같은 파라미터 수 대조는 기존 이력 결합 구조의 수량 예측 효과를 보강한다. 반면 전체 분기 사용 변형이 평균 수량 오차에서 앞서므로, 특정 단계적 활성화 규칙을 개선의 필수 원인으로 제시하지 않는다. 기존 외부 여섯 비교군 및 효율 측정 결과는 이번 감사로 변경되지 않았다. 구조 전체의 유용성, 분기 규칙의 효과, 데이터별 적용 범위를 분리해 원고에 반영할 수 있다. 통계적 유의성이나 독립 test 결과를 새로 확보한 것은 아니다.

## 5. 원본 및 재현 경로

- 고정 범위: [scope.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_extension_completed13_audit_20261002_v1/scope.json).
- 조건별 표와 원본 경로: [conditions.csv](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_extension_completed13_audit_20261002_v1/conditions.csv); 수치·짝지은 비교: [comparison.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_extension_completed13_audit_20261002_v1/comparison.json).
- 5080: [회수 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/retrieved/completed13_20261002_v1/5080/retrieval_receipt.json) · [최종 CPU 감사](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/retrieved/completed13_20261002_v1/5080/terminal_audit.json) · [원본 파일 SHA 목록](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/retrieved/completed13_20261002_v1/5080/original/collection_manifest.json).
- 5090: [회수 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/retrieved/completed13_20261002_v1/5090/retrieval_receipt.json) · [최종 CPU 감사](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/retrieved/completed13_20261002_v1/5090/terminal_audit.json) · [원본 파일 SHA 목록](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/retrieved/completed13_20261002_v1/5090/original/collection_manifest.json).
- 기존 84조건: [완료된 회수·검증](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_independent_evaluation_preparation_20261001_v1/report.md) 및 [validation 기준선](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_dataset_appendix_20261002_v1/selected_conditions.csv). 효율: [기존 검증 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_efficiency_5080_20260930_v1/verification.json).

최초 retrieval_receipt의 binary_cpu_audit=pending은 전송 직후 이력이다. 후속 terminal_audit.json이 현재 완료 판정이며, 최초 영수증은 덮어쓰지 않았다.

## 남은 작업 순서

**1. 이번 확정 비교를 원고에 반영한다 — 다음 작업**
- 대상: paper/titantpp_pakdd_2027_draft/main.tex의 Appendix A 및 연결 본문. Taxi 3seed와 다른 데이터 seed42를 구분하고, 단계적 분기 규칙에 대한 해석을 위 결과에 맞춘다. 기존 주 비교표 84조건과 효율 결과는 기준선으로 유지한다.

**2. 진행 중인 후속 조건을 관측하고 완료분을 같은 방식으로 검증한다 — 진행 중**
- 대상: 승인된 5080·5090 후속 큐. 이번 범위 밖 23조건의 학습 상태는 기존 시간별 관측을 따르며, 이번 보고서는 새로운 전체 상태 조회가 아니다. 데이터·모델별 3seed가 모두 확인된 뒤 평균과 표본 표준편차를 확정한다. 원고 반영과 병행할 수 있다.

**3. 최종 평가 범위와 원고의 핵심 주장을 확정한다 — 이후 작업**
- 남은 후속 비교와 기존 데이터 계보 검토를 연결한다. 최종 평가의 성격·실행 계약을 확인하고, 필요한 평가 후 초록·결론을 정리한다. 이번 원본 감사는 독립 평가나 새 평가 실행 승인을 대신하지 않는다.
