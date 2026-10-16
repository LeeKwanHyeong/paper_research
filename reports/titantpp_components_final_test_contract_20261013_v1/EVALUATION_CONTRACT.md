## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-10-13T20:41:42.600782+09:00
- Verification Status: UNVERIFIED（실행 미검증; 계약·registry 메타데이터 정적 확인）
- Version Label: components_final_Test_contract_v1

**현재 기준선과 승인 범위를 구분한다 — 완료 / 실행 봉인 대기**

대상 프로젝트는 `/Users/igwanhyeong/PycharmProjects/paper_research`다. 현재 관측 근거는 2026-10-13 19:40:29 KST의 구성요소36 중 완료32·진행2·대기2·미확정0이며, Taxi·RAF24는 08:27:20 KST 종료 캐시다. 새 원격 조회를 수행하지 않았다.

이번 계약은 비교군·선택 기준·지표·실행 순서를 확정한 설계 문서다. 신규 source131 선택 binary36의 로컬 immutable 원본 회수·SHA 결속과 신규 평가 연결부가 아직 준비되지 않았으므로 `execution_ready=false`다. 실행 명령·학습·Test 열람/평가·원고 편집·자동화 변경은0이다. 설계 문서는 start/training/Test permit이 아니며, 종료 heartbeat가 Test를 자동 시작하도록 변경하지 않는다.

기존 Test에는 생성·접근 및 분석 이력이 있다. 새 모델의 최종 Test 비교도 **기존 chronological Test의 후속 최종 비교**로 설명하며 새 미노출 독립 평가라고 부르지 않는다.

**논문의 질문별 비교군을 고정한다 — 설계 확정**

모든 학습 모델은 seed42/52/62를 사용한다. 새 구성요소 FULL을 기존 대표 모델이나 기존 이력 FULL과 섞지 않는다.

| 패널 | 데이터·비교군 | 모델 조건 수 | 처리 |
|---|---|---:|---|
| 신규 구성요소 | Taxi·Intermittent·RAF × NONE/CNN/GRU/FULL ×3seed |36| 새 선택 모델의 Validation 재현 후 Test 추론 |
| 기존 이력 보정 | 같은 세 자료 × FULL/NONE/current-only ×3seed |27| 기존 Test 원본의 신원·평가 계약이 같으면 재사용 |
| 기존 외부군 | 같은 세 자료 × RMTPP/THP/NHP/SAHP/S2P2/AttNHP ×3seed |54| 기존 common-head Test 원본 조건부 재사용 |
| 단순 내부 구조 | 같은 세 자료 × MLP4 ×3seed |9| 기존 Test 원본 조건부 재사용; 신규 동시 학습 비교로 설명하지 않음 |
| 기존 대표 모델 | 같은 세 자료 × 이전 대표 CNN+GRU54 ×3seed |9| 기존 원고 패널 보존; 신규 FULL9 대신 사용하지 않음 |
| Instacart 보충 | Instacart × CNN+GRU54/MLP8/MLP12 ×3seed |9| 기존 Test 원본 조건부 재사용 |
| 비학습 수량 기준선 | 네 자료 × 직전 수량/같은 관측 창의 과거 평균 |8 결정적 벡터| 같은 target/window 확인 후 재사용; seed 없음 |

학습 모델의 계획상 패널 행은144개 = 신규36 + 기존 재사용 후보108이다. **144회 새 추론이나144개 독립 모델을 뜻하지 않는다.** 같은 이름이라도 cohort를 구분하며, 실제 같은 binary의 재사용은 byte/state/source/target/metric의 전체 신원 결속으로만 결정한다. Deterministic 기준선을 독립3seed 평가로 세지 않는다.

Instacart 외부6종·MLP4의 기존21개도 메타데이터가 있지만 이번 범위에 추가하지 않는다. 원고의 핵심 외부 비교는 세 자료이고 Instacart는 구조 보충 비교다. 검색·새 독립자료·새 학습·MLP8/16 등 다른 과거 패널을 자동 추가하지 않는다. 확장하려면 결과를 보기 전에 별도 범위 수정이 필요하다.

기존 Test 재사용은 selected epoch·binary byte SHA·canonical tensor SHA·frozen source·evaluator·전처리·target identity/order/truth/time SHA·time bin/top-code·tail 정의·실제 완료 receipt와 원본 archive/result SHA를 모두 묶을 때만 허용한다. 이번에는 registry/receipt 메타데이터 가용성을 확인했으며 Test 수치/예측은 읽지 않았다. 불일치가 발견되면 해당 패널을 미확정으로 두고, 자동 재추론·유리한 조건만 제외·checkpoint 교체하지 않는다.

**Test 전에 모델 선택과 데이터 입력을 고정한다 — 설계 확정 / 실제 checkpoint 봉인 대기**

- 각 fit의 저장된 전체 Validation history 중 전체 raw 수량 RMSE가 finite인 최초 strict 최소 epoch를 선택한다. 동률은 최초 epoch다. 현재 진행 fit의 잠정 best를 최종 selected로 쓰지 않는다.
- `best_val_qty_rmse_model.pt`가 평가 대상이며 `last_epoch_state.pt`는 원본 감사용이다. 지표마다 epoch를 다르게 고르거나 Test 점수로 모델·seed·checkpoint를 고르지 않는다.
- 새36은 source131 closure `15f6d2b482c5f76c0488d017f5c923c76bc54fd75543aab838c477266a608b53`, frozen science `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`를 유지한다. CNN/GRU의 source131 component hooks와 identity가 필요하며 FULL/과거 구조로 치환하지 않는다.
- 신규 원본/handoff canonical SHA는 각각 `e60badde6d9d902500aa67764f6834213ea1775797508f31d204ae67a5e63074` / `81972d7b774c3b924d4bf6517252a0d0bed8291da9165100ab5b8932128ef1c1`다. 기존 이력/Instacart/외부는 자체 frozen source를 사용한다.
- Train에서 정한 변환·통계·time_scale·sigma floor·tail 임계값·모델 출력 규칙을 유지한다. Test에 맞춘 bias correction·rounding·추가 clipping·분포 폭/위치 보정·학습/통계 재추정은0이다.

| 자료 | 기존 Test 예상 target 수 | 큰 수량 truth 조건 | 시간 단위 / scale | lookback seq 단위 / max sequence |
|---|---:|---|---|---|
| Taxi |8327|q >3449|hour /1|168 /256|
| Intermittent |88019|q >187|week /3|520 /256|
| RAF |5226|q >200|month /6|84 /84|
| Instacart |578387|q >35|day /7; top-code30|52 /64|

이는 기존 동결 계약의 **대상 metadata**이며 Test 성과가 아니다. full-data/split-manifest SHA는 JSON에 고정했다. 특히 RAF full-data의 Train/Validation slice가 실제 학습 export와 같은지 검증한다. 신규 Test target/truth/time identity hashes는 실행 승인 이후 입력검증 단계에서 결속하며 임의 분할·대상 삭제를 하지 않는다.

기존 `oper_part_no,seq` 순서 및 full chronology 문맥을 유지하는 rolling one-step 평가다. 점수는 Test target에서만 계산한다. 이전 Test 사건은 해당 예측이 끝나고 실제 관측됐을 때 후속 prefix로 들어갈 수 있다. 모델 파라미터 업데이트·미래/target event 입력은0이다. max sequence에는 target이 포함되므로 target quantity0·memory target write false·직전 history hidden 사용과 target gap 비유입을 검증한다. 다단계 미래 무관측 예측으로 설명하지 않는다.

직전 수량은 모델과 같은 허용 관측 창의 마지막 raw 수량이고, 과거 평균은 같은 창에서 target/padding을 뺀 raw 수량 평균이다. Train-global/series 평균으로 바꾸지 않는다.

**수량 중심 지표와 집계를 고정한다 — 설계 확정**

- 주 지표: 전체 raw 수량 RMSE = sqrt(SSE/N).
- 보조 수량: 전체 MAE, 큰 수량 RMSE·MAE. 큰 수량은 true raw quantity가 Train 고정 상한을 strict 초과하는 동일 대상 집합이다.
- 보조 시간: 전체·큰 수량 recorded positive-integer TimeNLL = mean(−log P(D=d)). 위 수량 선택 checkpoint에서 여섯 지표를 함께 낸다. source131은 d=1에 P(T≤1.5), d≥2에 P(d−0.5<T≤d+0.5)를 쓰며 Instacart는 원래 top-code30 계약을 유지한다. 과거 v0.7 clamped loss나 continuous-density NLL과 혼합하지 않는다.
- 수량 예측은 원래 `expm1(softplus(quantity_head(hidden)))` 그대로다. 없는 시간 모델이나 tail N=0의 지표는 N/A이며0으로 채우지 않는다. 데이터셋별 시간 단위가 달라 합산 TimeNLL 순위를 만들지 않는다.
- target 단위 pooled SSE/절대오차/N를 사용한다. series별 macro 평균으로 평가 정의를 바꾸지 않는다.
- 각 지표의3seed 산술평균·표본SD(ddof1), 같은 dataset/seed의 FULL−대조 차이, 평균 절대·상대 개선과 wins/3를 모두 보고한다. 모델 앙상블이나 독립3회 baseline 성과로 표현하지 않는다.
- 기존 Q1 및 구성요소 joint strict 기준을 변경하지 않고 Test에서의 같은 방향 여부를 평균 개선과 별도로 보고한다. 통계 유의성·보편적 결합 필요성·참신성·독립 일반화의 증명으로 확대하지 않는다.
- NONE/CNN/GRU/FULL의 활성 파라미터는89859/90435/114591/115167이며 용량 통제가 아니다. M11−M10−M01+M00은 지표 척도에 의존하는 기술 통계로만 다룬다. 불리한 seed·큰 수량·시간 결과를 제외하지 않는다.

**source131 평가 연결부를 준비한다 — 다음 작업 / 구현 미완료**

기존 `evaluate_titantpp_history_instacart_frozen_test.py`는 이력27+Instacart9의 exact jobs와 source별 경로를 강제한다. 이전 CNN·GRU/대표 모델 evaluator도 다른 구조/조건 전용이다. 신규4arm36을 이 도구들에 숫자만 바꿔 실행하지 않는다.

별도 운영 adapter에 exact36 contract, source131 component install_hooks·identity·strict state loading, immutable artifact adapter, target/split 검증, 여섯 지표 재현, native Runtime gate를 연결해야 한다. science source·모델·학습 조건은 바꾸지 않는다. Synthetic/contract 테스트로 causal target/gap perturbation·single-vs-batch·state SHA 불변·잘못된 cohort/epoch/source 거부를 먼저 확인한다. 실제 Test로 도구를 디버깅하지 않는다. 이 단계의 검증된 실행 명령은 아직 없다.

**실제 종료와 원본을 확인한 뒤 Validation을 재현한다 — 외부 작업 대기 / 다음 작업**

1. 구성요소 고유36 scientific worker actual exit0와 binary manifest를 확인한다. 원/new5080은 dispatcher actualexit0·train_complete, 원5090은 retained8 scientific success와 계획된 claim FileExistsError/actualexit1·failure 보존을 따로 인정한다. 숫자 실패/부모12완료로 바꾸지 않는다. pipeline wrapper 자체 OS exit는 직접 관측하지 않았다는 한계를 유지한다.
2. 원본 archive·manifest·selected/last binary를 로컬 immutable 경로로 회수하고 byte SHA·canonical tensor SHA·selected epoch·metadata·source·Validation endpoint를 결속한다. 현 학습 root·failure/claim·기존 원본을 덮어쓰지 않는다.
3. 승인된 실행 회차에서 host별 현재 GPU UUID·실제 전역 idle·기존 Runtime fingerprint를 확인한다. 과거 원5080 종료 캐시는 현재 전역 idle을 증명하지 않는다.
4. 두 host에서 각 fit의 원 GPU/Runtime으로 selected 모델의 전체 Validation을 replay한다. **36개 모두 통과하기 전 신규 Test는0**이다. 전체/tail count·target/truth/time hashes는 정확 일치, 여섯 지표 각각 abs_tol=1e−5·rel_tol=1e−5로 고정한다. 불일치는 중단하며 허용오차 확대·source 수정·checkpoint 교체·자동 retry하지 않는다.
5. 실제 selected manifest·평가 adapter/source/runtime/input hash와 global Validation gate를 봉인한 실행 계약으로 묶고, 그 대상의 Test 실행 승인을 적용한다. 이번 설계 문서는 실행 permit이 아니다.

**두 GPU에서 고정 모델의 Test를 한 번 평가한다 — 이후 작업 / 실행 승인 필요**

| 평가 host | 신규 조건 | 기존 Runtime | 동시 worker |
|---|---|---|---:|
|5080|Taxi·RAF24 + Intermittent62 네 구조 =28|기존 ai_env / GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6 / cuDNN91900|1|
|5090|Intermittent42·52 네 구조 =8|기존 전용 resume runtime_env / GPU-c9adc246-ef66-7906-bc07-6e152fa9952f / cuDNN92000|1|

Validation 재현은 서로 독립적으로 병렬 준비할 수 있지만 Test release는 전체36 global gate에 의존한다. 이후 Test는 두 host에서 병렬로 진행하고 각 host 내 fit은 직렬로 평가한다. 격리 evaluation root·자체 source를 쓰며 공용 Runtime·driver·인증·학습 dispatcher를 변경하지 않는다.

각 모델은 model.eval/no_grad/grad false, 전후 tensor SHA 동일을 확인한다. 입력통계·학습가능 상태 업데이트·calibration·새 학습은0이다. raw prediction export는 끄고, 감사 가능한 target hash·count·sums·여섯 지표와 immutable 종료 receipt를 보존한다. 최종 성과는 전체 작업 종료/실제 exit 확인 후 일괄 집계한다.

기한은 job3시간·host campaign12시간으로 고정한 운영 한계이며 ETA가 아니다. 학습의36h/168h lease를 연장하지 않는다. 이 평가는 별도 승인 root/permit/기한이고 자동 retry/resume·kill·claim/failure 삭제는0이다. 기존Test cache 불일치로 추가 추론이 필요하면 범위를 명시하며 자동 확대하지 않는다. 별도 CPU binary 감사는 이후 승인 작업이며 native replay 실패를 CPU PASS로 대체하지 않는다.

**Test 결과와 논문 근거를 연결한다 — 이후 작업**

Test-only 결과를 cohort별·dataset별·seed별로 분리하고, Validation과의 방향 비교는 별도 표로 작성한다. primary quantity의 평균 이득·seed 예외·큰 수량 손해·외부/MLP4/단순 기준선 대비 위치를 함께 보고한다. 역사적 외부 common-head 적응과 source/Runtime 차이를 설명하며 native 원 논문 우월성으로 확대하지 않는다.

Test를 보고 모델/선택 기준을 바꾸면 그 작업은 후속 개발로 별도 기록한다. 동일 Test를 다시 새 독립 평가라고 부르지 않는다. 이번에는 열린 main.tex 및 다른 자동화를 변경하지 않았다. 원고 집필은 결과 확정과 별도다.

**증적과 남은 작업을 보존한다 — 계획 기록 완료**

`evaluation_design.json`에는 fresh36 및 조건부 재사용108의 정확한 cohort/ID/epoch/binary metadata, 데이터 SHA, 선택/지표/Runtime/승인 경계와 source rawSHA를 기록했다. 신규36의 아직 회수되지 않은 checkpoint SHA/epoch는 null로 두었으며 이를 봉인 완료라고 주장하지 않는다.

바로 다음 작업은 science 변경 없이 **source131 평가 adapter 구현·synthetic/contract 검증**이다. 이 작업은 남은 학습과 병렬로 준비할 수 있다. 이후 원본 회수·SHA 결속→Validation36 global gate→봉인된 Test 실행 승인→Test36/기존 결과 재사용→최종 집계 순서로 진행한다.
