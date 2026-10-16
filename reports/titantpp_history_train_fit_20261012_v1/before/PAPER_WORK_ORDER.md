# 완료 결과 반영 후 논문 작업 순서

2026-10-12 기존 Test36과 선택 모델27개의 Validation/Test 진단 계측54 완료를 반영해 재정렬했다. Validation 원고 반영은 현재 기준선이며 Test와 이번 진단 수치는 아직 원고에 반영하지 않았다. 대상은 paper_research의 `paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1/main.tex`다. 사용자가 승인한 저장 모델 진단 계측은 구현·실제 평가·원본 회수·분석·독립 QA까지 완료했다. 신규 학습·경로 개입·추가 검색·새 자료 평가는0이며 열린 main.tex는 수정하지 않았다. 수정 전 작업 순서는 `reports/titantpp_history_diagnostics_20261012_v2/before/PAPER_WORK_ORDER.md`에 보존했다. 이전 계획 및 Test 완료 이전 순서는 각 기존 before 파일에 유지한다.

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

**1. 보정 중심 원고에 실제 Test 예외와 진단 범위를 반영한다 — 다음 작업 / 공통 main.tex는 직렬 편집**

- 다음에 바로 진행할 대상은 열린 동일 main.tex의 Results·Methods·주장 정합이다. 실제 mean top4 prototype 설명은 유지하고 미완료 fixed/general512의 현재형·빈 Results 비중을 줄인다. 이번 진단 실행에서는 main.tex를 수정하지 않았다.
- Validation과 Test를 split·선택 checkpoint 출처별로 구분하고 Taxi/Intermittent 평균 개선과 seed 반전·RAF3seed 악화·시간/큰 수량 손해·단순 기준선 우위를 함께 쓴다. 계측의 확인 사실·연관·미확인을 나눠 cap·긴 이력·norm 크기를 확정 원인으로 표현하지 않는다.
- 구성 학습을 생략하면 결합 경로의 prototype 조건부 효과로 범위를 좁힌다. 모든3seed Test 일관성, CNN/GRU 각 필요성, 광범위한 우월성이나 새 자료 일반화를 주장하지 않는다. 외부 인코더 full9와 보정 full9는 별도 학습 기준선으로 보존한다.
- 완료 조건은 Abstract·Introduction·Methods·Results·Conclusion과 부록의 숫자·주장·한계가 같은 범위로 맞고 내장 컴파일·배치 검증이 통과한 상태다. 독립 문헌·증적 점검은 병렬 가능하고 공통 원고는 현재 세션에서 직렬 편집한다.

**2. 시간 head와 큰 수량 손해에서 검증할 단일 가설을 정한다 — 다음 작업 / 추가 평가·학습 실행은 승인 필요**

- Intermittent42의 좁은 시간 scale과 긴 gap 손해, Taxi52·RAF의 큰 수량 과소예측을 서로 다른 질문으로 정리한다. 원래 log 수량 objective와 raw RMSE selector, 시간 scale의 Validation 안정성·위치 오차 및 수량 head 출력을 기존 증적과 코드에서 먼저 점검한다.
- 논문 핵심 주장을 유지하는 데 필요한 질문 하나만 골라 현상·예측·비교군·판정 지표·불리한 결과·정지 조건을 성과 열람 전에 고정한다. 관측만으로 임의 σ floor·tail weight·LR 값을 정하거나 여러 값의 검색을 시작하지 않는다. cap 또는 큰 norm을 이미 입증한 원인으로 전제하지 않는다.
- 완료 조건은 새 학습이 필요한지, 저장 모델의 추가 추론이면 충분한지 또는 서술 제한으로 충분한지 판단한 최소 계약이다. 이 문서의 재정렬은 추가 실행 승인이 아니며 현재 신규 검색·학습은0개 권고다.

**3. CNN·GRU 각 기여가 핵심 주장에 필요할 때만 구성 실험을 검토한다 — 조건부 다음 작업 / 추가 실행 승인 필요**

- 저장 full9의 CNN/GRU 각 경로 우회18설정(Validation/Test36pass)은 출력 의존성 확인이고 독립 학습한 구조 ablation이 아니다. 단일 가설에 필요할 때만 새 격리 계약을 검토하며 이번에는 실행하지 않았다. 기존54 계측 평가를 우회 실험 완료로 세지 않는다.
- CNN·GRU 각각의 기여를 논문 주장으로 유지할 때에만 CNN-only/GRU-only/full/둘 다 제거×3seed=RAF12fresh를 같은5080 GPU/Runtime의 새 계약으로 검토한다. 결합 경로의 조건부 관찰로 좁히면 생략 가능하다. 기존 A100/RTX full·제거를 재사용한6신규를 동일 환경의 네 구조 비교로 세지 않는다.
- RAFcap84·공통 초기 tensor·RNG·shuffle·loss·optimizer·selector, 최대3동시·lease·예산·정지·회수를 고정한다. 예상 활성 수78,851/79,427/103,583/104,159는 새 runner의 native 검증으로 확정한다. 여섯 지표·개별 seed·평균/표본SD·불리한 결과와 구조별 용량 차이를 함께 기록한다.
- RAF는 Test를 보고 선택한 탐색적 사례로 공개한다. Taxi·Intermittent24fresh 확대·실패 seed만 재학습·loss/LR sweep은 자동 추가하지 않는다. 완료 조건은 해당 구성 주장을 유지할지 결정하고 필요한 경우 승인된 최소 계약의 실제 종료·원본SHA와 주장 범위를 확인한 상태다.

**4. 선택한 설계 작업 완료 또는 생략 후 모델을 동결하고 미노출 확인 계약을 정한다 — 다음 작업 / 자료·평가 실행 승인 필요**

- 필요한 head/구조 작업 완료 또는 생략을 결정한 뒤 모델·전처리·selector·비교군·여섯 지표를 고정한다. 기존 Test를 설계에 사용하면 후속 개발 자료로 표시한다. 새 자료의 사용 이력·전처리 호환성과 적격 대상 범위를 성과 열람 전에 확인한다.
- 모델을 유지한다면 full/제거/current-only×3seed의9선택 모델과 직전/평균 기준선을 새 기간에 추론만 하는 후보를 검토한다. 구조·head를 바꾸면 checkpoint 목록도 새로 동결한다. Taxi 한 기간 결과를 모든 dataset의 독립 일반화로 확대하지 않는다.
- CPU binary 감사는 저장 모델의 CPU 재현이라는 별도 목적이다. native GPU의 원래 Validation36와 이번 진단 Validation27 재현·SHA·표 검산은 완료했지만 전체 CPU 감사는 미완료이고 미노출 확인을 대신하지 않는다.
- 완료 조건은 최소 자료와 checkpoint·평가·자원·회수·정지 계약이 성과 열람 전에 고정되고 승인 범위가 명시된 상태다. 이번 진단에서는 새 자료 다운로드·새 자료 평가·학습을 실행하지 않았다.

**5. 검색을 별도 기여로 유지할 경우에만 새 계약을 확정한다 — 승인 필요 / 현재 실행 대기**

- mean·weighted·fixed·general512의 기존 3core×3seed×4arm=36과학 조건을 기본으로 한다. 재사용 확정 전에는36fresh를 예산으로 잡는다. 최신 full9는 A1005+RTX4다. RTX의 Intermittent42/52/62와 RAF62만 재사용 후보이며, GPU/Runtime·동결 초기화·입력·Train/Validation 순서와 native bridge가 검증되면32신규(검색27+새 mean5)와 과거 mean4로36조건을 구성한다. 현재 확정 재사용은0이다.
- 단일 dataset12조건은 파일럿이며 Q2를 충족하지 않는다. 2dataset24조건은 원래3dataset 범위를 축소하는 새 계약과 사후 선택 공개가 필요하다. 이미 보정 결과가 좋다는 이유로 dataset을 고르고 새 독립 확인이라고 쓰지 않는다.
- 각 arm은 동일 초기값과 입력 계약에서 독립 학습하며 최종 encoder·prototype 값을 공유하지 않는다. 시간·수량32개 관측 요약은 fixed/general512에만 query 입력으로 사용된다.
- 기존 prepared bundle의 prior_stage 포인터와 workers1을 최신 RTX handoff/parallel3 계약으로 간주하지 않는다. 과학 source·공통 mean·GPU/Runtime·최대3worker·예산과 회수·중단 조건을 고정하고 승인 전에는 launch하지 않는다.
- 비용 문서의 T64/T256과 frozen RAF 측정 T64/T84가 다르다. Frozen position 보간 경로 덕분에 과학 모델·Train cap84를 유지한 operation의 synthetic256 측정은 코드상 가능하지만 native finite·메모리·비용은 미검증이다. 모델 생성의 max_seq_len을256으로 바꾸면11,008개 parameter와 RNG가 달라져 같은 모델이 아니다. 기존 gate를 조용히 완화하지 말고 새 운영 측정 계약과 native 확인 전에는 launch를 차단한다.
- 성과는 같은 두 개 이상 dataset 각각 모든3seed에서 fixed가 세 대조보다 strict 전체 RMSE 개선인지 평가한다. 비용은 단독 GPU에서 B128/warmup5/measure20과 mean 대비 추론1.20·학습 step1.30·peak1.20 상한을 검증한다. 각 dataset/seed군은 같은 GPU에서 mean 완료/재사용 확인 후 세 검색 fit을 최대3개 동시 실행하고 모두 자연종료한 뒤 다음 군으로 진행한다. 같은 선택 epoch의 여섯 지표·표본 SD·손해도 함께 보존한다.
- 이 조건부 계획은 실행 승인이 아니다. 새 학습·추론·자원 계약을 승인받은 뒤에만 진행하고, 빈 GPU를 이유로 시작하지 않는다.

**6. 선행연구와 재현성·부록을 최종 주장에 맞춘다 — 다음 작업 / 독립 문서 점검은 병렬 가능**

- 검색 포함 여부와 평가 범위에 맞춰 Related Work·References·Appendix의 주장과 출처를 대조한다. S2P2·AttNHP, residual/adapters, prototype memory, numerical marks와 간헐 수요를 중심 질문과 연결한다. 이번에 유지한28개 참고문헌은 현재 편집의 보존 기준선이며 새 참신성 검증을 대신하지 않는다.
- Train lag 상관은 데이터 특성으로, 과거 MLP 결과는 별도 구조의 관찰로 유지한다. 데이터·seed·선택 epoch와 여섯 지표, source revision·Runtime 기록을 서로 연결한다.
- 문헌 조사와 독립 문서 검토는 병렬 가능하다. 공통 `main.tex` 편집은 확정된 분석과 포함 결정에 따라 한 세션에서 순서대로 반영한다.
- 완료 조건은 최종 주장마다 직접 근거가 있고 출처·split·재현성·한계가 일관된 상태다.

**7. 저자 편집과 실제 제출 규정을 반영해 최종본을 만든다 — 다음 작업 / 평가·범위 확정 후**

- 현재 확장 초안은 저자 검토용이다. 실제 PAKDD 투고 연도·트랙의 공식 분량·익명성·보충자료·마감 규정을 해당 시점의 공식 출처에서 확인한 뒤 본문과 부록을 줄인다.
- 표·그림·수식·인용을 다시 컴파일·렌더링하고 최종 source/PDF SHA와 증적을 남긴다. 기존 내장 컴파일 성공과 정적 PDF 검증을 실제 제출 규정 충족으로 해석하지 않는다.
- 완료 조건은 저자가 최종 검토한 원고와 제출 점검 기록이다. 실제 투고는 별도의 명시적 제출 승인 이후다.

현재 학습·원본·Validation 원고 기준선·검색 필요성 판단·기존 Test36 최종 비교·수량/입력/학습 진단과 선택27checkpoint의 실제54split 계측·원래 지표 재현·독립 QA는 완료다. 다음에 바로 진행할 대상은 열린 main.tex의 Test·진단 예외 반영이며, 그 뒤 단일 가설을 고정해 추가 증거의 최소 범위를 결정한다. 경로 우회18설정과 RAF12fresh는 필요할 때만 별도 승인으로 검토하며 신규 검색은0개 권고로 대기한다. 미노출 확인과 전체 CPU 감사는 별도 미완료다. 독립 문헌·증적 검토는 병렬 가능하지만 공통 계약·GPU 실행·main.tex 편집은 의존관계에 따라 직렬로 진행한다.
