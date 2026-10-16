# Cap36 실제 시작 기록

**5090 학습 시작·5080 대기 연결 — 진행 중**

2026-10-17 01:05:20–21 KST 실제 관측이며 이 시각을 이후 상태로 갱신해 해석하지 않는다.

| 서버 | 승인 배정 | 관측 상태 |
|---|---:|---|
| 5090 | RAF12 + Intermittent52·62의8 =20 | RAF42 MLP4·MLP16·AllAvailable 세 worker/GPU 진행, 각3epoch 저장 |
| 5080 | Taxi12 + Intermittent42의4 =16 | 기존 raw 학습 2개가 GPU 사용 중. 새 Cap 학습0, 준비 작업 종료와 유휴를 기다리는 큐 연결 |

총 Cap 완료0·진행3·대기33. Validation 재현과 Test는 아직 실행하지 않았다. 네 번째 CurrentOnly와 나머지 seed는 같은 고정 큐에서 차례대로 시작한다. 각 서버 동시 최대3개다.

v1 패키징 오류는 기존 모델·과학 조건을 바꾸지 않고 v2 새 경로에서 해결했다. 실패 원문은 보존했다. 실제 v2 archive의 변경 없는 추출·의존성 import 검사, 5090 GPU/Runtime·합성 검사와 실제worker의 PID·시작시각 식별자·명령·GPU를 검증했다. 이는 초기 시작 검증이며 이후 완료를 뜻하지 않는다.

**고정 checkpoint를 Validation 재현 후 Test로 평가한다 — 다음 작업**

각 서버의 native 큐는 모든 배정 학습 완료 → Validation 최저 rawRMSE checkpoint 재현 → 해당 서버 전체 재현 통과 → Test 순서로 연결돼 있다. 전36개 전역 gate가 아니라 서버별 gate다. 새로운 학습을 추가하거나 실패를 자동 재시도하지 않는다. 이전 반복개선 자동화는 재생성하지 않았다.

**결과 원본을 회수하고 3seed를 비교한다 — 다음 작업**

- 완료 전에 collect를 호출하지 않는다. 다음 확인 시 `manage_titantpp_mlp_cap36 --mode status --host 5080|5090`을 호스트별 한 번 실행하고 현재 완료/실패를 구분한다. prepare/deploy는 재실행하지 않는다.
- 양쪽 큐의 완료, worker/dispatcher exit0, 해당 작업의 프로세스 종료를 확인하면 `--mode collect --host 5080|5090`으로 호스트별 원본을 한 번 회수한다. 미완료·실패 회수 경로는 덮어쓰거나 재시도하지 않는다.
- 양쪽 회수 완료 뒤 `python3 -m paper.scripts.compare_titantpp_mlp_cap36`을 실행한다. 전체36 Train/Validation/Test·선택 checkpoint와 원본 SHA를 확인하고 데이터셋별 RMSE·MAE 평균±표본SD, 다섯 seed별 구조 대조, cap집단·시간 지표를 보고한다.
- 기존 CNN/원본 결과는 변경·재평가하지 않는다. raw 결과와 Cap 결과는 서로 다른 목표이므로 같은 성능 표에 섞지 않는다.

**코드·기록 보존 — 완료**

- 저장소: paper_research, 브랜치: codex/contextual-retrieval-pilot-20261007.
- 실제 학습 source revision: e33f540c02b5de69fcd75e84375ec0f5c6a89bd3. 기존 Cap36 구현은 ab8d2f20068e526c4696b6e71250adf2aed3b847.
- 계약·입력·archive·실제 상태 SHA: PREPARED.json, START_RECEIPT.json. v1 실패: ../titantpp_mlp_cap36_20261017_v1/PACKAGING_FAILURE_QA.json.
- Notion의 TitanTPP History MLP16 페이지에 범위·시작·실패 수정·한계·남은 순서를 추가하고 기존 하위 페이지 보존 및 기록 내용을 다시 확인했다.
