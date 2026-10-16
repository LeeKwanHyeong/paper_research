# 시간·수량 결합 두 구조의 seed52·62 추가

## 기존 seed42와 과학 조건을 보존한다 — 현재 기준선

- 사용자 2026-10-16 승인: “그러면 지금 RAF, Intermittent, Taxi 관련해서 52, 62 모두 실험에 추가하자.”
- 기존 `titantpp_time_quantity_cross_precheck6_20261016_v1`의 두 구조 × 3데이터셋 × seed42는 그대로 진행하고 종료 결과를 재사용한다. 재학습·중단·계약 수정은 하지 않는다.
- 모델은 이력 MLP16 + 시점별 MLP 분기와 이력 MLP16 + 시간↔수량 Cross Attention이다. 추가 용량은 각각6,176가중치이며 입력 앞 결합 위치가 같다. 기존 후단 메모리를 유지한다. 전단 메모리의 효과는 이번 범위에 없다.
- 모델·runner·기존 933개 source와 원수량 Train/Validation 입력은 부모 archive의 동일 바이트로 상속한다. Q3 Cap 자료는 사용하지 않는다.

## 동일 구조에 두 seed를 추가한다 — 진행 중

- 신규12개: RAF·Taxi·Intermittent × 두 구조 × seed52/62. 기존 seed42 6개와 합쳐 총18개, 데이터셋·구조별3seed가 된다.
- 원수량 입력/log1p·전처리·공통 손실·head·메모리·batch128·AdamW lr0.001/weight_decay0.01·gradient clip1을 유지한다.
- 최대300epoch, 최소40epoch, patience40, 전체 Validation raw 수량 RMSE의 최초 strict finite 최소 epoch를 선택한다. MAE는 같은 checkpoint 값이다. fit별36시간을 유지한다. 큐 자체는 시작 후72시간으로 제한하며 기존 seed42의 제한을 바꾸지 않는다.
- 실제 `train_one(seed)`와 각 worker의 `PYTHONHASHSEED`를52/62로 설정한다. 기존 checkpoint를 불러오지 않는 fresh fit이다.
- 비교 기준선은 기존 이력 MLP16의 저장된 동일seed Validation9개다. 기존 CNN이나 MLP16을 다시 학습·추론하지 않는다. Test는 실행하지 않는다.

## 기존 작업과 합쳐 서버당 세 개를 넘지 않게 실행한다 — 다음 작업

| 서버 | 신규 배정 | 신규 fit 수 |
|---|---|---:|
| 5080 / cuDNN91900 | Taxi 두 구조 ×52/62, Intermittent 시점별 MLP ×52/62 | 6 |
| 5090 / cuDNN92000 | RAF 두 구조 ×52/62, Intermittent Cross ×52/62 | 6 |

- 부모 seed42의 실제 worker와 신규 worker를 함께 세어 최대3개만 실행한다. 시작 중이지만 GPU에 아직 표시되지 않은 worker도 슬롯을 차지한다.
- 빈 슬롯은 Taxi/RAF seed52 두 구조, seed62 두 구조, Intermittent52/62 순으로 채운다. 기존 학습 파일·프로세스를 수정하지 않는다.
- 검증 순서: 로컬 계약/seed/슬롯 합성 검사 → 코드 커밋 → 부모 source·입력·계약·qualification 일치 → GPU/기존 worker 신원 확인 → 별도 root의 tmux 큐 한 번 시작 → 실제 신규 worker/GPU/seed 확인.
- 같은 source·입력·GPU의 부모 native 합성 및 메모리 admission 증적을 재사용하고, 새 각 worker는 실제 Runtime을 확인한다. GPU 합성 검사를 중복 실행하지 않는다.
- unknown GPU 작업·부모 실패·새 worker 실패·timeout이면 추가 시작을 멈추고 기록한다. 자동 retry/resume·기존 claim 삭제·기존 작업 kill은 없다. 새 child의36시간 timeout 정리만 자기 process group에 적용한다.
- 새 자동화나 반복 개선 스케줄러는 만들지 않는다. 큐는 이번12개만 처리하고 종료한다. 사용자 요청 시 상태를 조회한다.

## 종료 원본을 회수하고 세 seed를 비교한다 — 다음 작업

- 각 host의 신규6개 실제 worker exit0와 dispatcher exit0·종료 파일 SHA를 확인한 뒤 별도 원본 archive/manifest로 회수한다. 기존 seed42도 해당 host 종료 후 한 번 회수하여 원본과 선택 checkpoint를 검증한다.
- 같은 대상의 seed42/52/62를 묶어 데이터셋별 평균±표본SD(ddof1), 모든 seed의 후보−MLP16과 Cross−시점별MLP RMSE/MAE 절대·상대차이를 기록한다. 기존 기준선과 신규 결과를 혼합하여 하나의 모델 평균으로 만들지 않는다.
- 부가 수량/시간 지표는 실제 저장된 범위에서 보고하며 없는 지표를 추정하거나 새 추론으로 채우지 않는다. 이번에는 Test를 열거나 실행하지 않는다.
- Intermittent 대조군은5080, Cross는5090으로 모든seed의 환경을 기존과 맞춘다. 구조 차이와 GPU/cuDNN 차이가 섞인 한계를 명시한다. 원수량 결과와 Cap 결과는 별개다.

작업 대상: `paper_research` 저장소, `codex/contextual-retrieval-pilot-20261007` 브랜치.
새 root: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_time_quantity_cross_seed52_62_20261016_v1_{5080,5090}`.
모델/운영 검증은 병렬 검토하며 계약 생성·커밋·실제 배포는 순서대로 진행한다.
