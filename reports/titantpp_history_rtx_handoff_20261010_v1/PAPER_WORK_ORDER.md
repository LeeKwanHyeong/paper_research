# 완료 결과 반영 후 논문 작업 순서

2026-10-12 RAF의 log/raw 진단18개와 Intermittent의 시간 위치·폭 교차18개 완료를 반영해 재정렬했다. 사용자가 승인한 순서대로 RAF 원본·분석·독립 QA를 먼저 확정한 뒤 Intermittent를 실행했다. 두 단계 총36개 모델·split 평가와 실제24개 작업(18+6)·pipeline OS exit0, immutable 원본·로컬 SHA 및 독립 QA가 확인됐다. 새 학습·새 Test 평가·calibration fit·원고 편집은0이다. 원고의 Validation 기준선은 그대로다. 2026-10-12 최신 사용자 지시에 따라 프로세스 판정 수정·검증 후5080부터 Taxi·RAF24개 학습을 시작한다. 5090 Intermittent12개는 최신사용자 “5090도 진행하자” 승인에따라동일v5의host별native검증뒤시작한다. 현재 열린 main.tex는 이번에도 수정하지 않았다. 수정 전 순서는 `reports/titantpp_intermittent_time_cross_20261012_v1/before/PAPER_WORK_ORDER.md`에 보존했다.

**현재 기준선의 완료 결과와 원본을 확정한다 — 완료**

- Instacart9와 이력 보정27의 실제 worker·supervisor 종료, immutable 원본·selected/last checkpoint SHA와 최종 Validation 집계가 확인됐다. 보정27은 A10015+5080 RAF62 세 조건+5090 Intermittent9를 원래 ID로 한 번씩 센다. 이미 완료된 학습은 다시 실행 대상으로 잡지 않는다.
- A100 부모·복구 Pod와 전용 volume 삭제 증거가 확보됐고 완료 관측 자동화도 종료됐다. 최종 청구 확정 여부는 과학 결과와 별개다. 추가 Pod를 만들거나 삭제된 Pod에 접속하지 않는다.
- 최초 strict finite 최소 전체 raw 수량 RMSE checkpoint에서 전체/큰 수량 MAE·RMSE·TimeNLL을 묶었다. 3seed 평균과 표본 SD, 불리한 paired 차이는 보존된다.
- 완료 근거는 RTX 인계 bundle의 `analysis/completion_receipt.json`, `selected_conditions.json`, `reports/titantpp_history_rtx_handoff_20261010_v1/FINAL_VALIDATION_AUDIT_20261011.json` 및 Instacart 완료 bundle의 같은 분석 기록이다. 2026-10-11 Validation 원고 편집에서는 원격 관측이나 새 학습을 하지 않았다.

**Validation 기준 CNN+GRU54 결과와 방법·주장의 범위를 맞춘다 — 완료 / 현재 기준선**

- 본문은 기존 CNN+GRU54/외부 인코더 비교 → 보정27 → Instacart9 → 대기 중 검색 순서다. 기존 외부 비교 full9와 새 보정 full9를 합치거나 교체하지 않는다. 과거 MLP4/MLP8 수치는 부록에 유지했다.
- Taxi·Intermittent의 세 seed가 각각 두 대조보다 strict 전체 RMSE 개선을 보여 Q1 개발 일관성 기준을 충족한다. RAF52는 두 대조보다 나빠 같은 기준에 실패한다. 전체 TimeNLL11/18, 큰 수량 RMSE·MAE 각각10/18 손해도 기록했다.
- Instacart는 작은 평균 수량 이득, seed52 MLP12 우위와 평균 전체 TimeNLL 악화를 함께 적었다. 다른 파라미터 예산의 구조 비교이므로 보정 경로의 인과 효과를 대체하지 않는다.
- Methods 수식·Algorithm 1은 유지했다. bypass 등록 모듈의 동결·실행0·활성 학습 파라미터0, full/current-only 활성25308 동일과 계산 경로·깊이 차이를 명확히 했다. 현재 hidden의 prefix 이력 때문에 모델 전체에서 이력을 제거한 대조로 설명하지 않는다.
- A100/RTX 분담, RAF42 full/current-only 물리 A100 이관과 seed 간 Runtime 차이를 공개했다. 동시 fit wall time은 구조 효율 근거가 아니다. Abstract부터 Conclusion까지 같은 제한을 유지한다.
- 부록 G/H는 checkpoint epoch와 여섯 지표를 추가했다. 숫자·기존 표·28개 참고문헌 보존과 컴파일·페이지 배치를 이번 편집 증적으로 검증한다. 새 독립 평가·CPU 재추론 감사 완료를 주장하지 않는다.

**검색 실험이 현재 핵심 주장에 필요한지 판단한다 — 완료**

- 권고는 신규 검색 학습0개다. 현재 제목·기여·결론은 국소·순환 보정의 결합 경로와 외부/Instacart 성능 위치이며, 검색 규칙 우위를 요구하지 않는다. 보정27은 prototype mean readout이 유지된 backbone에서의 조건부 효과다. Prototype 자체 효과와 없어도 같은 보정 이득이 유지되는지는 별도 질문이다.
- 검색 추가는 시간·큰 수량 지표 손해나 독립 평가의 공백을 자동으로 해결하지 않는다. 기존 Validation을 독립 확인으로 바꾸지 않는다. 현재 주장에 맞춰 완료 근거를 우선 정리한다.
- [판단·조건부 최소 구성](../titantpp_retrieval_necessity_decision_20261012_v1/README.md)에 주장–근거, 12/24/36조건의 범위와 재사용·비용 계약 문제를 기록했다. 이번 계획 요청에서는 열린 main.tex나 실험 코드를 수정하지 않았다.

**선택 모델의 기존 Test 최종 비교를 마친다 — 완료**

- 사용자가 2026-10-12 승인한 이력 보정27·Instacart9의 선택 checkpoint36개를 고정했다. 5080의 격리된 평가 root에서 native6검증 → 전체 Validation36 재현과 실제 exit0 → 전역 gate → Test36을 실행했다. 재학습·Test epoch 재선택·Test 튜닝은 수행하지 않았다.
- 09:32:08 KST에 실제 파이프라인 exit0, 평가 worker72개의 exit0와 소유 GPU/PID 부재를 확인했다. 원본929파일의 archive·manifest·로컬SHA가 일치했고 36조건을 중복 없이 집계했다. 원고·공용 Runtime·인증은 변경하지 않았다.
- Test에서 Taxi·Intermittent의 평균 전체 RMSE는 두 대조보다 좋아졌지만, 같은 seed에서 양대조를 이긴 조건은 각각2/3이다. 모든3seed·양대조 개선을 보인 이력 보정 Test dataset은0개다. RAF는3seed 모두 양대조보다 나쁘다. Validation의 Q1 개발 일관성 판정과 이 Test 결과를 구분한다.
- Instacart는 MLP8에3/3, MLP12에2/3 전체 RMSE 개선이고 평균 이득은 작다. 이력 보정의 전체 TimeNLL은18쌍 중12쌍 손해다. RAF 전체 RMSE는 입력 이력 평균보다 나쁘고 Taxi·RAF·Instacart의 큰 수량 RMSE는 직전 수량 반복보다 나쁘다. 불리한 결과와 부호 반전을 제외하지 않는다.
- [최종 관측·해석 범위](../titantpp_history_instacart_test_20261012_v1/FINAL_OBSERVATION.md), [원수치·여섯 지표·평균/표본SD](../../search_artifacts/titantpp_history_instacart_test_20261012_v1/analysis/TEST_FINAL_COMPARISON.md), [완료 증적](../../search_artifacts/titantpp_history_instacart_test_20261012_v1/analysis/completion_receipt.json)을 기준선으로 사용한다. 기존 Test의 생성·접근 이력을 유지하며 새 미노출 독립 확인으로 부르지 않는다.

**기존 Test의 수량 오류·대상 구성·학습 기록을 진단한다 — 완료**

- [종합 진단](../titantpp_test_reversal_diagnosis_20261012_v1/README.md)에18 paired 수량 분해,27개 원본 학습의 선택·종료와 frozen loader 대상·truth/history digest를 연결했다. 새 fit·모델 추론·원격 호출은0이고 원본은 보존했다.
- Taxi52는 큰 수량83건의 과소예측 손해, Intermittent42는 주로2<q≤31의 과대예측, RAF는 모든 seed의 큰 수량39건 과소예측이 중심이다. 좋은 seed와 불리한 시간 지표도 남겼다. RAF52는 Validation부터 악화해서 모든 RAF를 방향 반전으로 부르지 않는다.
- Test 악화10쌍은 Validation의5개 수량 구간 비중으로 재가중해도 악화한다. Intermittent의 긴 이력·작은 현재 수량, RAF의 series 구성·이력 지원 차이는 확인했지만 특정 경로의 인과 원인은 미확인이다. 늦은 선택이나 RAF42 A100 이관 하나로 설명할 근거는 없다.
- CNN+GRU54의54는 hidden 폭이고 GRU 조건은 관측2개 이상이다. RAF14→19는 recurrent 입력 지원 차이 후보이며 lag16 경로 새 활성화가 아니다. 기존 CNN+MLP16은 진짜 CNN-only가 아니다.

**선택27개 저장 모델을 Validation·Test에서 계측하고 원래 지표를 재현한다 — 완료**

- 2026-10-12 사용자 “진행해보자” 승인 범위인 Taxi·Intermittent·RAF의3arm×3seed, 선택27checkpoint를 기존5080 native Runtime의 격리 v2 root에서 평가했다. native synthetic3route 불변성 확인 → Validation27 실제 exit0·원래 여섯 지표·대상 digest 재현 → 전역 gate → Test27 재현 순서를 지켰다. 신규 fit·경로 개입·checkpoint 재선택은0이다.
- **10:51:06 KST**에 pipeline의 실제 OS exit0을 확인했고 worker54개도 모두 실제 exit0이었다. 종료 후 소유 GPU/PID 부재를 별도로 확인했다. 원본746개·archive·manifest·두 로컬 사본의 SHA와 독립 QA가 통과했다. contract `70e8e22dbe85aa93cb5dc555b06303511b0e5e7f511771b455b4b243ddb3ad1d`, source129 closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`를 보존한다. [완료 상태](../titantpp_history_diagnostics_20261012_v2/RUN_STATUS_COMPLETE.json), [최종 QA](../titantpp_history_diagnostics_20261012_v2/FINAL_QA.md).
- Taxi52의 큰 수량83건은4series 모두 양 대조보다 손해이며, 좋은 seed42보다 보정 norm이 더 크지는 않았다. Intermittent42의2<q≤31 손해는33,203건 중 비절단33,183건에 집중돼 cap 절단 단독 원인은 지지되지 않는다. 작은 log-duration scale과 긴 목표 gap이 큰 시간 손해에 동반됐으나 수량 반전과 시간 손해의 인과 원인은 분리해 다룬다.
- RAF의 큰 수량39건은 모두 입력 이력14 이내이고32건은 Validation 공유 series다. 긴 이력 지원 초과 또는 Validation 비등장 series만으로 큰 수량 손해를 설명하지 않는다. 기존14→19 이력 차이는 전체 입력 특성의 관찰로 남기되 tail 원인처럼 쓰지 않는다. [핵심 해석](../titantpp_history_diagnostics_20261012_v2/FOCUSED_INTERPRETATION.md), [전체 split·seed·대조 수치](../../search_artifacts/titantpp_history_diagnostics_20261012_v2/analysis/DIAGNOSTIC_REPORT.md).
- v1의 metadata 집합을 None으로 덮어쓴 운영 오류, Validation3worker 부분 시도·완료0/Test0·실제 exit1·claim/failure를 보존했다. 수정 회귀 검증 뒤 별도 v2에서54개 평가를 완료했다. 첫 오프라인 보고서의 구간 강조 오류도 이전 분석을 보존해 수정했으며27개 행·18개 대조·source 파일은 불변, 추가 모델 실행은0이다. 이를 학습의 수치 실패나 새 독립 확인으로 바꾸지 않는다.

**같은 선택 모델의 Train·Validation·Test 적합 상태를 확인한다 — 완료**

- 같은 선택 history27 checkpoint를 Train에서만 추가 평가하고 이전 같은 checkpoint의 Validation27·Test27을 SHA로 재사용했다. native3route → Train27 실제 worker·pipeline OS exit0(2026-10-12 **11:47:48 KST**) → immutable 원본559개·archive/manifest·두 로컬 사본 SHA →81split 연결과 독립 QA를 완료했다. 신규 fit·새 Validation/Test 추론·checkpoint 재선택은0이다.
- 현재 Train canonical `76248e2174d7bd651d821f649ad860030da985d59da88a8ce50725b476f55ccf`, operation rawSHA `157791af8ce197cfc858659e195eda314b2d1de7ca139971d627e51d6b91819b`와 기존 source129/revision·선택 state를 유지한다. Train raw 지표는 학습 중 batch loss와 다르며 과거 선택 checkpoint Train 평가의 재현이라고 부르지 않는다. [완료 상태](../titantpp_history_train_fit_20261012_v1/RUN_STATUS_COMPLETE.json), [최종 QA](../titantpp_history_train_fit_20261012_v1/FINAL_QA.md), [모든81split·여섯 지표·대조·평균/표본SD](../../search_artifacts/titantpp_history_train_fit_20261012_v1/analysis/TRAIN_FIT_REPORT.md).
- RAF full은 Train tail205건부터 RMSE458.95~468.43·평균 오차−350.86~−364.10으로 직전 수량426.56·과거 평균412.18보다 tail RMSE가 나쁘다. 전체 Train RMSE는 두 단순 기준선보다 좋다. 큰 수량 point 재현 부족은 학습 대상에도 있지만 true-tail 조건부 평균회귀·확률성·목표 상충·희소성 등의 원인을 분리한 증명은 아니다.
- Taxi52의 tail RMSE는 Train413.31·Validation370.76·Test1017.49로 Train에서는 두 대조·두 단순 기준선보다 좋고 Test 증가가 두드러진다. Taxi62는 Train tail에서도 양 대조보다 나쁘므로 모든 Taxi를 같은 상태로 묶지 않는다. Intermittent42의 전체 수량 RMSE는1.4146→1.6667→2.9429, TimeNLL은0.2193→0.6187→62.5721로 평가 구간 악화가 크다. 전체/큰 수량 MAE·bias·시간 손해도 함께 보존한다.
- split별 수량 분포·tail 비중이 다르다. RAF Train tail은 Validation/Test보다 분산이 크며 Train raw RMSE가 Test보다 큰 사실을 순수 과적합으로 해석하지 않는다. 출력에는 유한 raw 상한 cap·raw_scale 재곱셈이 없다. [기존 증거와 실제 적합 상태의 해석](../titantpp_history_train_fit_20261012_v1/EXISTING_EVIDENCE_DIAGNOSIS.md).
- 원본 회수 검사에서 launch argv의 Python 실행 경로와 worker sys.argv의 표현 차이로 멈춘 운영 오류를 보존했다. 실제27개 PID/PPID·인자·exit0가 정확히 일치하고 claim/archive 생성 전이었다. 별도 행정 코드·독립 QA·회귀4개로 회수만 마쳤으며 원래 sealed finalizer·operation·worker는 수정0, 추가 추론은0이다.

**RAF의 실제 log 수량 목표와 raw 큰 수량 오류를 연결한다 — 완료 / 현재 기준선**

- 선택9모델×Train/Validation18개를 같은5080 eval Runtime으로 평가했다. native3·실제18worker/pipeline OS exit0·357개 원본 archive/manifest와 두 로컬 복사본 SHA·독립 최종 QA가 통과했다. v1의 합성 CUDA eval RNN backward 검증 실패는 실제 평가0개로 보존하고, 과학 forward는 같은 별도 운영 수정v2에서 완료했다.
- Train 큰 수량205/25779는0.7952%이며 full 수량 log 손실5.93–6.78%·raw SSE80.43–81.48%를 차지했다. Train 큰 수량에서 log/raw 오류는 직전·평균 기준선보다 세 seed 모두 나쁘다. Validation은 직전 값보다 log 오류가 좋지만 raw 오류는 나쁘고 평균보다 둘 다 나쁘다.
- 전체 log 이득과 희소 큰 수량 실패를 분리한다. 같은 부분집합의 목표 상충·전체/큰 수량 모집단 차이·seed52의 손해를 함께 보고하며 학습 gradient 비중·용량 부족·최적화 실패·손실 변경 효과를 확정하지 않는다. [원본 결과](../../search_artifacts/titantpp_raf_log_raw_20261012_v2/analysis/RAF_LOG_RAW_REPORT.md).

**Intermittent의 시간 위치와 폭의 점수 의존성을 확인한다 — 완료 / 현재 기준선**

- RAF의 실제 완료·원본·분석·QA 원문 SHA를 계약에 결합한 뒤18모델·split 평가를 실제6joint 작업으로 마쳤다. joint1개씩·resident3모델, native3·실제6exit/pipeline OS exit0·309개 원본과 두 로컬 사본 SHA·독립 QA가 통과했다. 원래18개 여섯 지표·대상·history/state를 재현했다.
- Validation6비교에서 full 위치를 유지한 대조 σ 교체는 모두 NLL 개선, μ만 교체는 모두 손해였다. Train에서는 두 단일 교체 모두6비교에서 손해였다. Validation seed42는0.618713→σ 교체0.243837/0.253718이며 μ 교체0.858383/0.686191이다.
- 기록 간격(1,3]의 σ 교체 개선과 ≤1·일부 >7 구간 손해, 모든 seed·두 대조·상호작용을 공개한다. 48paired 셀/24새 교차 셀은 서로 다른 저장 모델 출력을 섞은 점수이며 새 calibration 모델·기존 Test 반전 해결·CNN/GRU 필요성·독립 일반화의 증명이 아니다. [원본 결과](../../search_artifacts/titantpp_intermittent_time_cross_20261012_v1/analysis/TIME_CROSS_REPORT.md), [두 진단과 후속 판단](../titantpp_intermittent_time_cross_20261012_v1/DIAGNOSTIC_DECISION.md).

**1. CNN과 GRU 결합 대조를 두 서버에서 실행할 준비를 마친다 — 완료 / 현재 기준선**

- 기존 CNN+MLP16은 CNN-only가 아니며, 이력 보정27도 네 구조의 구성요소 대조가 아니다. 이를 확인한 뒤 사용자가 승인한 fresh36 비교의 모델·실행 코드·계약을 준비했다. 과거 checkpoint를 새 학습 초기값으로 사용하거나 과거 A100/RTX 결과를 새 matched fit으로 재명명하지 않는다.
- [두 서버 실행 준비](../titantpp_cnn_gru_components_20261012_v1/README.md)에서 5080 Taxi·RAF24학습, 5090 Intermittent12학습을 고정했다. 각 데이터셋의 세 seed와 같은 seed의 네 구조는 한 GPU·Runtime에 둔다. NONE/CNN/GRU 세 작업 동시 실행 뒤 FULL 하나를 실행하고 다음 seed로 넘어간다.
- 총56개 로컬 검증과 source131·input7·operation4·로컬12파일·배포 archive의 SHA 검사가 통과했다. 계약 canonical은593501da72c8628a50c5158835e0f877a5423b01bb5911872f3058a0f02e1ed3이다. native GPU 검증·원격 접속·실제 학습·Test 평가·원고 편집은0이다.
- 최신 사용자는 5090의 다른 작업이 있으므로 다음 “진행하자” 지시 전에는 실행하지 말고 준비하도록 변경했다. launch hold 원문을 보존하고 새 authorization·release·permit은 아직 발급하지 않았다.

**2. 5080 Taxi·RAF24개를 이어가고5090 Intermittent12개를 검증·시작한다 — 진행 중**

- 최신 사용자는 “마지막 프로세스 판정 오류를 수정하고 검증해서 5080 학습부터 바로 진행하자. 5090은 내가 진행하자고 할 때까지 대기해.”라고 승인했다. 5080의 건강·고정 GPU·기존 Runtime·SHA와 해당 host의 native 검증만 학습 gate로 사용한다. 5090은 최신사용자의별도 “5090도 진행하자” 승인으로ownhealth/봉인/native gate를거쳐시작하며5080실행/permit을변경하지않는다.
- v2는 필수 설계 metadata 누락으로 실제 qualification exit1, 과학 학습0이었다. 수정 v3의 native 네 구조 검증은 실제 exit0였으나 동시 수용 단계의 프로세스 판정에서 멈췄다. CNN-only 정상 종료 증적과 다른 두 작업의 owned cleanup, 실패 원본을 보존한다. 실패 순간 proc argv가 미기록이므로 종료 경계의 경합은 추론이며 성공으로 변경하지 않는다.
- v4에서 종료한CNN PID가proc에서소멸해도NVML에잠깐남는원인을확인했다. v5의최대5초·25회관측settling과현재소유·외부PID차단회귀를확정하고로컬고유55개검증을통과했다. source131·input7·모델·초기화·손실·optimizer·selector와최초5080 절대기한을유지하며이전v2/v3/v4의실패·claim을보존했다.
- 5080 v5의native·동시3합성worker·supervisedqualification campaign 실제exit0와GPU idle을확인하고2026-10-12 18:28:26 KST에trainphase를시작했다. 2026-10-12 18:32:30 KST에Taxi42 NONE/CNN/GRU 실제3GPU PID와정확argv/PPID·lease·세checkpoint저장을확인했다. 세arm종료→FULL1개→Taxi52/62→RAF42/52/62순서로24학습을진행한다. 지속원격polling과새자동화는0이다. [실제시작관측](../titantpp_cnn_gru_components_20261012_v5/START_OBSERVATION.md).

- 5090은 최신 “5090도 진행하자” 승인후같은v5의ownhealth/native/admission 실제exit0·GPU유휴를확인하고2026-10-12 18:39:56 KST에trainphase를시작했다. 2026-10-12 18:44:38 KST에Intermittent42 실제3GPU worker1059784/1059786/1059788·정확argv/lease를확인했다. 세arm첫checkpoint저장전,12개완료0/진행3이다. [5090 실제관측](../titantpp_cnn_gru_components_20261012_v5/START_OBSERVATION_5090.md).

**2.1 두서버 checkpoint와조건부ETA를매시간확인한다 — 적용완료 / 관측진행중**

- 최신사용자의1시간모니터링승인으로 `cnn-gru36-checkpoint-eta` heartbeat ACTIVE를등록했다. 호스트별읽기전용status1회/검증terminalcache0회, 실제저장·history best·각fit최근최대10완료epoch 속도의조건부잔여를보고한다. 원본수정·새학습/Test·retry/resume0이며전체큐ETA미확정이다. [스케줄/종료조건](../titantpp_cnn_gru_components_20261012_v5/hourly_monitor/SCHEDULER.md).

**3. 모든 종료와 원본을 확인한 뒤 결합 경로의 효과를 판정한다 — 다음 작업 / 학습 종료 후**

- 고유36개 ID, 실제 worker·supervised dispatcher 종료와 마지막 phase complete, 소유 process/GPU 부재, selected/last binary·history·timing·manifest/archive·로컬 SHA를 확인한다. 부분 완료·조회 실패·식별자 미확정을 완료로 바꾸지 않는다.
- 같은 선택 epoch의 전체/큰 수량 RMSE·MAE·TimeNLL, seed별 paired 차이와3seed 평균·표본 SD를 모두 보고한다. FULL이 NONE·CNN-only·GRU-only보다 같은 두 개 이상 데이터셋의 각각 모든3seed에서 전체 raw 수량 RMSE를 strict 개선할 때만 결합 경로의 개발 지지로 기록한다.
- 파라미터 용량 차이, 데이터셋 사이의 GPU·Runtime 차이, 세 작업 동시/FULL 단독 부하 차이를 공개한다. 상호작용은 지표 척도에 의존하며 통계 유의성·기계적 시너지·보편적 필수성·참신성·새 독립 일반화의 증명으로 확대하지 않는다. 시간·큰 수량의 손해와 실패한 seed를 제외하지 않는다.

**4. 핵심 근거를 기준으로 CNN+GRU54 원고와 필요한 최소 개선을 연결한다 — 다음 작업 / 공통 원고 직렬 편집**

- 구성요소 대조 결과를 먼저 판단하고 Results·Methods·Abstract·Conclusion의 주장과 예외를 맞춘다. 기존 mean prototype 설명, 확정된 H1/H2 진단과 Test/Train 예외를 보존한다. 단순한 원고 축소를 핵심 기여 강화로 표현하지 않는다.
- 현재 열린 main.tex는 이번 준비에서 수정하지 않았다. 새 구조 대조36과 기존 보정27·외부 full9를 중복 집계하지 않는다. 원고 편집 뒤에는 내장 LaTeX 컴파일과 배치·인용을 검증한다. 독립 문헌·출처 검토는 병렬 가능하고 공통 원고는 직렬 진행한다.
- RAF 수량 목표나 Intermittent 시간 폭의 실제 개선이 꼭 필요하면 이번 구조 결과와 연결되는 최소 가설·대조·학습 계약을 따로 정한다. 진단의 연관만으로 원인을 확정하거나 head/loss 학습·seed 검색을 자동 실행하지 않는다. 추가 실행에는 별도 승인이 필요하다.

**5. 최종 모델을 동결한 뒤 미노출 확인·감사·문헌과 투고 점검을 마친다 — 다음 작업 / 추가 평가 승인 필요**

- 선택한 설계 작업 완료 또는 생략 뒤 최종 모델·전처리·대조·selector·여섯 지표를 동결한다. 이미 접근한 기존 Test는 설계 판단에 사용된 개발 증거다. 새 자료의 사용 이력·호환성을 성과 열람 전에 확인하고 미노출 확인·CPU binary 감사의 별도 대상·자원·정지·원본 계약을 정한다.
- 검색은 최종 핵심 주장에 꼭 필요한 경우만 새 공통 계약·native 비용·동일 GPU 재사용 조건을 검토한다. 현재 실행 대기이며 빈 GPU나 과거 prepared bundle을 승인으로 해석하지 않는다.
- 최종 주장에 맞춰 기존28개 참고문헌·수식·부록·재현성 증적을 확인하고, 해당 연도·트랙의 공식 제출 규정에 맞춰 컴파일·저자 검토를 진행한다. 실제 투고는 별도 명시적 승인 이후다.

현재바로이어가는단계는시작된5080 Taxi·RAF24와5090 Intermittent12의자연진행이다. 매시간실제진행/checkpoint/조건부ETA를관측하고모든종료후원본·36ID·여섯Validation지표를확정한다. 이후 모든 종료·원본·기여 판정→원고 정합과 필요한 최소 개선→모델 동결·미노출 확인/CPU 감사→문헌·부록·투고 점검 순서로 진행한다.
