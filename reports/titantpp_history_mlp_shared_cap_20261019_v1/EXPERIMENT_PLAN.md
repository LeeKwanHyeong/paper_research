# Cap 공유형 이력 MLP와 매개변수 대조 준비

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: approved implementation and local reproducibility checks only
- Origin Date: 2026-10-19
- Verification Status: 구현·합성 통합 검증·독립 검토 완료; 실데이터 결과 없음
- Version Label: shared_history_cap_preparation_v1

**현재 기준선과 승인 범위 — 완료**

- 사용자는 공유형 이력 MLP와 용량 대조군의 계약·구현·검증만 승인했다. 새 학습은 준비 결과를 보고 결정한다. 현재5080/5090의 학습·복구·완료 대기 절차를 변경하지 않는다.
- 세 데이터셋은 같은 Cap 입력을 사용한다: Taxi65, RAF10, Intermittent20. 기존 Train/Validation parquet·분할 manifest6개와 원래 Train 통계·모델/시간 head 설정을 같은 SHA로 재사용한다.
- 두 입력은 모두 기존 log1p, 수량 손실은 log-MSE, 시간 가중치는0이다. 시간 head는 그대로 두며 시간 손실의 gradient 기여를 제외한다. TimeNLL은 미학습 시간 head의 진단값이다.
- 기존 MLP16의 같은 조건 Validation9개만 저장 기준선으로 재사용한다. 기존 CNN·MLP16·완료 결과·선택 checkpoint를 수정·재학습·재추론하지 않는다.
- 출발 근거인 10월17일 이력 MLP 분석은 원래 수량 데이터의 사후 분석이다. 현재 Cap에서 공유가 유익하다는 직접 증거나 과적합 원인 확정으로 읽지 않는다.

**검증 가능한 공유 가설 고정 — 완료**

질문은 “현재·직전 관측의 관계 특징을 공통으로 학습하는 구조가, 이력 길이마다 같은 수의 가중치를 사용할 수 있는 독립 MLP보다 세 데이터셋의 Validation을 개선하는가?”다.

입력은 첫 encoder 뒤의 현재/직전 유효 관측64차원 쌍이다. 두 번째 encoder와 후단 Hard-LMM·persistent memory·수량/시간 head를 유지한다. 새 전단/후단 우회 경로·gate·RevIN·GRU·QKV 필터·시간 입력 변형은 추가하지 않는다.

| 구조 | 입력 변환 | 출력 변환 | 보정부 전체 가중치 |
|---|---|---|---:|
| 기존 MLP16, 저장 기준선 | 분기별128→16 | 분기별16→64 | 24,576 |
| 공유형15 | 공통128→15 하나 | 분기별15→64 여덟 개 | 9,600 |
| 독립 prefix 대조군 | 분기별128→[15,5,5,5,5,5,5,5] | 각 폭→64 | 9,600 |

원래 이력 문턱1·2·4·8·16·32·64·128과 직전 관측의 선택·padding 건너뛰기·withheld 경계 초기화·고정 /8·bias 없는 정확 GELU를 유지한다. 각 문턱은 다른 과거 거리를 읽는 것이 아니다. 두 후보 모두 같은 직전 관측을 받는다.

공유형은 `r=(1/8)Σ A_j B_j GELU(W_shared [c;p])`, 대조군은 `r=(1/8)Σ A_j B_j GELU(W_j [c;p])`다. 공유형의 모든 출력 분기가 같은 비선형 특징을 사용한다.

**이력 노출에 따른 가중치 수까지 맞추기 — 완료**

처음 검토한 공유12/독립균일5는 등록 매개변수 수만7,680으로 같았다. RAF의 최대 길이84에서는128 분기가 활성화되지 않아 실제 관여 가능한 가중치 수가 달랐다. 이 초기 검토는 DESIGN_QA.md에 보존하고, 최종 결정은 DESIGN_QA_V2.md에 남겼다. 성능을 본 설계 변경은 아니며 실제 학습을 실행하지 않았다.

최종 공통 폭 r=15는 기존 폭16 이하의 가장 가까운 양의3배수라는 산술 규칙으로 고정했다. 독립 폭 `[r,r/3×7]`를 사용하면 활성 prefix k≥1에서 두 구조 모두 `64×r×(k+2)`개가 수식에 관여한다. k=0은0개다. k=7이면8,640, k=8이면9,600이다. 같은 규칙을 세 데이터셋에 적용한다.

이는 **동일 매개변수 수**이지 같은 함수 공간이나 통계적 용량의 증명이 아니다. 공유형의 국소 Jacobian rank 상한은15, 독립형은 min(64,15+5(k−1))이며 gradient 합산과 분기별 폭 배분도 다르다. 따라서 향후 결과도 이 전체 구조 제약의 효과로 해석한다. 초기 출력이0이라 첫 step의 입력 가중치 gradient가0인 것은 정상이며, output update 이후의 경로 연결을 검사한다. 실데이터에서 모든 문턱이 관측된다고 보장하지 않는다.

**별도 모델·실행 진입점 준비 — 완료**

- paper_research 저장소 codex/contextual-retrieval-pilot-20261007 브랜치의 신규 모델 파일과 신규 합성 실행 진입점만 추가한다. 기존 factory·training engine·봉인 source는 수정하지 않는다.
- 후보는 `titantpp_history_mlp_shared15_v1`, `titantpp_history_mlp_prefix_capacity_v1`이다. 이전 MLP16 이름이나 checkpoint identity를 재사용해 바꿔 부르지 않는다.
- 초기 출력은0이며, 같은 seed에서 encoder·memory·head의 초기 tensor와 호출자 난수 상태는 기존 MLP16과 같게 유지한다. 새 학습이 승인돼도 기존 checkpoint warm start는 하지 않는다.
- 합성 진입점은 CPU 합성검사만 제공한다. 실제 Cap 학습·추론·원격 설치·GPU qualification·Test를 여는 명령은 준비 단계에 포함하지 않는다.

**로컬 통합 검증 — 완료**

- 129개 이상 합성 관측으로 모든 분기 경로와 gradient 사용을 검사하고 각 활성 prefix의 실제 연결 가중치 수를 대조한다.
- 인과성·target 차단·padding/withheld 처리·batch 독립성, 공통 초기화·RNG·head 보존, 잘못된 arm/metadata/tensor checkpoint 거부를 검사한다.
- 기존 실제 training.train_one 엔진을 작은 합성 frame에서만2epoch 실행해 최초 strict finite fullValidation rawRMSE 최소 checkpoint를 저장하고, 같은 target의 예측과 여섯 지표를 다시 계산한다. 이2epoch는 실제 연구 학습 예산이 아니며 실데이터 fitting 횟수는0이다.
- 시간 head의 gradient·optimizer state·가중치 불변, optimizer 및 양쪽 arm의 동일 objective를 검사한다. 모든 hook은 예외 때도 복원한다.
- 원 source963·입력6·기준선9의 SHA를 검증하고 독립 검토 후 신규 파일과 증적을 대상 브랜치에 커밋한다. 공용 Runtime이나 원격 서버는 변경하지 않는다.
- 최종 독립 검사72개와 하위 사례106개가 통과했다. 검사 전후 기존 source·입력·기준선에 연결된 고유990개 파일의 SHA가 같았다. 최종 원문과 검사 범위는 INDEPENDENT_QA_V2.json/.md에 보존한다. 초기 검사 증적은 수정하지 않았다.
- 실제 합성 CLI에서 진행 로그가 결과 JSON에 섞인 출력 오류를 발견했다. 최초 child exit0와 양쪽 checkpoint 재현 통과 기록, 로컬 캡처 오류를 CLI_FIRST_ATTEMPT.json에 남겼다. 새 진입점의 로그 출력만 수정하고 회귀 검사 및 실제 합성 CLI 재확인까지 통과했다. 최초 로그와16개 합성 산출물도 보존했다.
- 합성 검사는 CPU에서만 수행했다. native GPU 동등성이나 Cap 데이터의 성능 개선을 검증한 결과가 아니다.

**향후 비교 규칙 — 계약에 고정, 실제 학습은 승인 필요**

- 계획 행렬은 후보2개×Cap3데이터셋×seed42·52·62=신규18fit다. 이는 현재 실행 승인이 아니다. 기존 MLP16 기준선9개는 재사용한다.
- 학습 조건은 기존300epoch 상한·최소40·patience40·batch128·AdamW lr0.001/weight_decay0.01/clip1이다. 입력·Train 통계·head·memory·선택 기준을 공통으로 유지한다.
- 선택은 전체 fullValidation raw 수량 RMSE의 최초 엄격 유한 최소 epoch다. MAE·cap-hit·시간 지표도 같은 checkpoint에서 구한다. MAE를 따로 골라 최적값을 합치지 않는다.
- 데이터셋별3seed 평균·표본SD(ddof1), 공유−MLP16/대조−MLP16/공유−대조의 모든 seed 절대·상대차이를 보고한다. 서로 다른 데이터셋의 raw 오류를 평균내지 않는다.
- 다음 단계의 공통 개선 근거는 세 데이터셋 각각에서 공유형의 RMSE·MAE3seed 평균이 기존 MLP16과 매개변수 대조군보다 모두 낮은지다. seed별 악화와 예외도 모두 공개한다. 대조군과 같거나 더 나쁘면 공유형의 추가 근거가 부족하다.
- Cap 자료의 `q>cap` tail은 비어 있으므로 N/A_empty로 명시한다. `q==cap` cap-hit RMSE/MAE·전체/cap-hit recorded positive-integer TimeNLL을 보존한다. 시간 가중치0의 TimeNLL을 학습된 시간 예측 성능으로 주장하지 않는다.
- SD 감소만으로 통계적 안정성·유의성을 주장하지 않으며, Test 성능·세 데이터셋의 보편적 개선·독창적 공유 원리를 주장하지 않는다.

**남은 실행 순서**

1. 완료된 실제 엔진 합성 save/replay·독립 검토 증적과 새 소스·입력·기준선 계약을 로컬에서 봉인한다.
2. 사용자에게 준비 결과와 남은 native 검증 범위를 보고한다.
3. 사용자가 새 학습을 결정한 뒤에만 별도 실행 계약·서버 배정·GPU/Runtime admission·실데이터 학습을 준비한다. 이번에는 원격 실행이나 현재 큐 변경을 하지 않는다.
