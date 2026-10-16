# 완료 결과 반영 후 논문 작업 순서

2026-10-12 기존 Test36조건의 실제 실행·원본 회수·집계 완료를 반영해 재정렬했다. Validation 원고 반영은 2026-10-11의 현재 기준선이며 이번 Test 수치는 아직 원고에 반영하지 않았다. 대상은 paper_research의 `paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1/main.tex`다. 사용자가 승인한 Instacart·이력 보정 반영과 CNN+GRU54 중심 Results·Methods·주장 정합을 완료한 상태에서 재정렬했다. 이번 수정 전 순서는 `reports/titantpp_history_instacart_test_20261012_v1/before/PAPER_WORK_ORDER.md`에 보존했다. 2026-10-11 원고 반영 이전 순서는 `reports/titantpp_completed_results_integration_20261011_v1/before/PAPER_WORK_ORDER.md`에 남아 있다.

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

**1. Test 결과와 예외를 반영해 보정 중심 원고의 주장 범위를 맞춘다 — 다음 작업**

- 대상은 열린 paper_research의 동일 main.tex다. 이번 Test의 원수치·3seed 평균/표본SD·paired 예외를 저자 편집 기준으로 연결한다. 실제 mean top4 prototype readout은 구조·재현성 설명으로 유지하고, 미실행 fixed/general512 비교의 현재형·빈 Results와 별도 검색 가설 비중을 줄이거나 후속 연구로 분리한다.
- 기존 Validation과 이번 Test를 별도 split·선택 checkpoint 출처로 표시한다. Taxi/Intermittent의 평균 개선과 seed 실패, RAF의3seed 악화, 시간 손해, 단순 기준선의 우위를 함께 반영한다. 모든3seed의 Test 일관성이나 전반적 우월성으로 확대하지 않고 Abstract·Introduction·Methods·Results·Conclusion을 같은 범위로 맞춘다.
- 완료 조건은 이력 보정의 평균 이득과 적용 예외를 prototype 조건부 효과로 설명하고, Validation에서 관찰한 일관성이 기존 Test에도 유지됐다고 오인할 문장과 미완료 검색의 완료형 표현을 제거한 상태다. 다음에 바로 진행할 일이다. 공통 원고 편집은 한 세션에서 직렬 진행한다.

**2. 추가 증거의 목적과 새 평가·학습 계약을 먼저 확정한다 — 다음 작업 / 실제 추가 실행은 승인 필요**

- 현재 Test36은 완료 기준선으로만 사용한다. 같은 Test를 반복하거나 그 결과로 모델을 조정한 뒤 새 독립 확인이라고 부르지 않는다. 아직 보지 않은 자료·기간의 확보 가능성, 원래 전처리 호환성·사용 이력을 먼저 확인한다.
- 새 자료를 실제 평가하기 전에 비교군·선택 checkpoint·전처리·전체/큰 수량/시간 지표·정지 조건을 고정한다. 구조 구성요소의 원인을 별도 기여로 주장할 경우 필요한 최소 대조를 먼저 정하고, 추가 학습이 있다면 완료·모델 동결 후 새 자료의 최종 성과를 확인한다. 검색 추가가 현재의 seed 실패·시간/큰 수량 손해를 자동으로 해결한다고 가정하지 않는다.
- CPU binary 감사는 저장 모델의 CPU 수치 재현을 확인하는 별도 목적이다. 이번 native GPU Validation36 재현과 SHA·표 검산은 완료했지만 전체 CPU 재추론 감사는 미완료다. 독립 자료 평가를 대신하지 않는다.
- 완료 조건은 논문 주장에 직접 필요한 추가 확인을 최소 범위로 정하고, 새 학습·자료 접근·자원·회수 계약과 승인 대상을 구분한 계획이다. 이번 승인은 기존 Test36에 한정되며 새 미노출 자료 평가·추가 학습을 자동 실행하지 않는다.

**3. 검색을 별도 기여로 유지할 경우에만 새 계약을 확정한다 — 승인 필요 / 현재 실행 대기**

- mean·weighted·fixed·general512의 기존 3core×3seed×4arm=36과학 조건을 기본으로 한다. 재사용 확정 전에는36fresh를 예산으로 잡는다. 최신 full9는 A1005+RTX4다. RTX의 Intermittent42/52/62와 RAF62만 재사용 후보이며, GPU/Runtime·동결 초기화·입력·Train/Validation 순서와 native bridge가 검증되면32신규(검색27+새 mean5)와 과거 mean4로36조건을 구성한다. 현재 확정 재사용은0이다.
- 단일 dataset12조건은 파일럿이며 Q2를 충족하지 않는다. 2dataset24조건은 원래3dataset 범위를 축소하는 새 계약과 사후 선택 공개가 필요하다. 이미 보정 결과가 좋다는 이유로 dataset을 고르고 새 독립 확인이라고 쓰지 않는다.
- 각 arm은 동일 초기값과 입력 계약에서 독립 학습하며 최종 encoder·prototype 값을 공유하지 않는다. 시간·수량32개 관측 요약은 fixed/general512에만 query 입력으로 사용된다.
- 기존 prepared bundle의 prior_stage 포인터와 workers1을 최신 RTX handoff/parallel3 계약으로 간주하지 않는다. 과학 source·공통 mean·GPU/Runtime·최대3worker·예산과 회수·중단 조건을 고정하고 승인 전에는 launch하지 않는다.
- 비용 문서의 T64/T256과 frozen RAF 측정 T64/T84가 다르다. Frozen position 보간 경로 덕분에 과학 모델·Train cap84를 유지한 operation의 synthetic256 측정은 코드상 가능하지만 native finite·메모리·비용은 미검증이다. 모델 생성의 max_seq_len을256으로 바꾸면11,008개 parameter와 RNG가 달라져 같은 모델이 아니다. 기존 gate를 조용히 완화하지 말고 새 운영 측정 계약과 native 확인 전에는 launch를 차단한다.
- 성과는 같은 두 개 이상 dataset 각각 모든3seed에서 fixed가 세 대조보다 strict 전체 RMSE 개선인지 평가한다. 비용은 단독 GPU에서 B128/warmup5/measure20과 mean 대비 추론1.20·학습 step1.30·peak1.20 상한을 검증한다. 각 dataset/seed군은 같은 GPU에서 mean 완료/재사용 확인 후 세 검색 fit을 최대3개 동시 실행하고 모두 자연종료한 뒤 다음 군으로 진행한다. 같은 선택 epoch의 여섯 지표·표본 SD·손해도 함께 보존한다.
- 이 조건부 계획은 실행 승인이 아니다. 새 학습·추론·자원 계약을 승인받은 뒤에만 진행하고, 빈 GPU를 이유로 시작하지 않는다.

**4. 선행연구와 재현성·부록을 최종 주장에 맞춘다 — 다음 작업 / 독립 문서 점검은 병렬 가능**

- 검색 포함 여부와 평가 범위에 맞춰 Related Work·References·Appendix의 주장과 출처를 대조한다. S2P2·AttNHP, residual/adapters, prototype memory, numerical marks와 간헐 수요를 중심 질문과 연결한다. 이번에 유지한28개 참고문헌은 현재 편집의 보존 기준선이며 새 참신성 검증을 대신하지 않는다.
- Train lag 상관은 데이터 특성으로, 과거 MLP 결과는 별도 구조의 관찰로 유지한다. 데이터·seed·선택 epoch와 여섯 지표, source revision·Runtime 기록을 서로 연결한다.
- 문헌 조사와 독립 문서 검토는 병렬 가능하다. 공통 `main.tex` 편집은 확정된 분석과 포함 결정에 따라 한 세션에서 순서대로 반영한다.
- 완료 조건은 최종 주장마다 직접 근거가 있고 출처·split·재현성·한계가 일관된 상태다.

**5. 저자 편집과 실제 제출 규정을 반영해 최종본을 만든다 — 다음 작업 / 평가·범위 확정 후**

- 현재 확장 초안은 저자 검토용이다. 실제 PAKDD 투고 연도·트랙의 공식 분량·익명성·보충자료·마감 규정을 해당 시점의 공식 출처에서 확인한 뒤 본문과 부록을 줄인다.
- 표·그림·수식·인용을 다시 컴파일·렌더링하고 최종 source/PDF SHA와 증적을 남긴다. 기존 내장 컴파일 성공과 정적 PDF 검증을 실제 제출 규정 충족으로 해석하지 않는다.
- 완료 조건은 저자가 최종 검토한 원고와 제출 점검 기록이다. 실제 투고는 별도의 명시적 제출 승인 이후다.

현재 학습·원본·Validation 집계와 원고 기준선, 검색 필요성 판단, 기존 Test36 실행·원본 회수·최종 집계는 완료다. 다음 작업은 이번 Test의 예외를 반영해 보정 중심 원고의 수치·주장과 미완료 검색 비중을 정리하는 것이다. 고정 검색은 현재 실행 대기를 유지한다. 전체 CPU binary 감사·새 미노출 독립 평가·추가 구조 학습은 별도 미완료이며, 필요성과 계약을 확정한 뒤 승인 범위에서 진행한다. 독립 문헌·증적 검토는 병렬 가능하지만 공통 원고와 평가 계약 변경은 현재 세션에서 직렬로 정리한다.
