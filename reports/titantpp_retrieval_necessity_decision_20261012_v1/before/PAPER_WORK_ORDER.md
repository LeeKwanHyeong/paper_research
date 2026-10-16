# 완료 결과 반영 후 논문 작업 순서

2026-10-11 갱신. 대상은 paper_research의 `paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1/main.tex`다. 사용자가 승인한 Instacart·이력 보정 반영과 CNN+GRU54 중심 Results·Methods·주장 정합을 완료한 상태에서 재정렬했다. 수정 전 순서는 `reports/titantpp_completed_results_integration_20261011_v1/before/PAPER_WORK_ORDER.md`에 보존했다.

**현재 기준선의 완료 결과와 원본을 확정한다 — 완료**

- Instacart9와 이력 보정27의 실제 worker·supervisor 종료, immutable 원본·selected/last checkpoint SHA와 최종 Validation 집계가 확인됐다. 보정27은 A10015+5080 RAF62 세 조건+5090 Intermittent9를 원래 ID로 한 번씩 센다. 이미 완료된 학습은 다시 실행 대상으로 잡지 않는다.
- A100 부모·복구 Pod와 전용 volume 삭제 증거가 확보됐고 완료 관측 자동화도 종료됐다. 최종 청구 확정 여부는 과학 결과와 별개다. 추가 Pod를 만들거나 삭제된 Pod에 접속하지 않는다.
- 최초 strict finite 최소 전체 raw 수량 RMSE checkpoint에서 전체/큰 수량 MAE·RMSE·TimeNLL을 묶었다. 3seed 평균과 표본 SD, 불리한 paired 차이는 보존된다.
- 완료 근거는 RTX 인계 bundle의 `analysis/completion_receipt.json`, `selected_conditions.json`, `reports/titantpp_history_rtx_handoff_20261010_v1/FINAL_VALIDATION_AUDIT_20261011.json` 및 Instacart 완료 bundle의 같은 분석 기록이다. 이번 원고 편집에서 원격 관측이나 새 학습은 하지 않았다.

**CNN+GRU54 중심 결과와 방법·주장의 범위를 맞춘다 — 완료**

- 본문은 기존 CNN+GRU54/외부 인코더 비교 → 보정27 → Instacart9 → 대기 중 검색 순서다. 기존 외부 비교 full9와 새 보정 full9를 합치거나 교체하지 않는다. 과거 MLP4/MLP8 수치는 부록에 유지했다.
- Taxi·Intermittent의 세 seed가 각각 두 대조보다 strict 전체 RMSE 개선을 보여 Q1 개발 일관성 기준을 충족한다. RAF52는 두 대조보다 나빠 같은 기준에 실패한다. 전체 TimeNLL11/18, 큰 수량 RMSE·MAE 각각10/18 손해도 기록했다.
- Instacart는 작은 평균 수량 이득, seed52 MLP12 우위와 평균 전체 TimeNLL 악화를 함께 적었다. 다른 파라미터 예산의 구조 비교이므로 보정 경로의 인과 효과를 대체하지 않는다.
- Methods 수식·Algorithm 1은 유지했다. bypass 등록 모듈의 동결·실행0·활성 학습 파라미터0, full/current-only 활성25308 동일과 계산 경로·깊이 차이를 명확히 했다. 현재 hidden의 prefix 이력 때문에 모델 전체에서 이력을 제거한 대조로 설명하지 않는다.
- A100/RTX 분담, RAF42 full/current-only 물리 A100 이관과 seed 간 Runtime 차이를 공개했다. 동시 fit wall time은 구조 효율 근거가 아니다. Abstract부터 Conclusion까지 같은 제한을 유지한다.
- 부록 G/H는 checkpoint epoch와 여섯 지표를 추가했다. 숫자·기존 표·28개 참고문헌 보존과 컴파일·페이지 배치를 이번 편집 증적으로 검증한다. 새 독립 평가·CPU 재추론 감사 완료를 주장하지 않는다.

**1. 검색이 핵심 논증에 필요한지 먼저 결정한다 — 다음 작업**

- 대상은 현재 원고의 prototype retrieval 질문과 기존 mean·weighted·fixed·general512 설계다. 보정 결과를 설명하는 데 독립적인 근거를 더하는지 판단하고 포함/제외 이유를 기록한다. 검색이 다른 결과의 손해를 만회할 것이라고 전제하지 않는다.
- 제외한다면 미검증 검색 기여를 연구 질문·기여·결론에서 정리하고 필요한 구조 설명과 과거 탐색의 범위만 남긴다. 유지한다면 미완료 상태를 명확히 하고 다음 비교 계약을 만든다.
- 완료 조건은 원고의 중심 주장과 검색 질문의 관계, 필요한 증거와 추가 실험 여부가 구체적으로 정리된 상태다. 다음에 바로 진행할 일은 이 결정이며, 빈 GPU를 이유로 검색을 시작하지 않는다.

**2. 검색을 유지할 경우 공통 기준선과 자원 계약을 확정한다 — 포함 결정 후 다음 작업 / 실행 승인 필요**

- 같은 데이터·seed·GPU·Runtime의 mean과 세 검색 대조를 고정한다. A100 full을 RTX 동일 GPU mean으로 혼합하지 않는다. 기존 증거의 실제 재사용 가능성을 확인한 뒤 필요한 fit 수와 자원·비용·학습 예산을 정한다. 27개나36개를 자동 실행하지 않는다.
- Q2, 전체/큰 수량/시간 지표, T64/T256 추론·학습 step·peak 비용 기준을 결과 확인 전에 고정한다. fixed/general512는 같은512개 query 파라미터를 사용한다. 정적 prototype 유무의 효과나 새로운 메모리 원리의 발명으로 해석하지 않는다.
- 완료 조건은 비교·checkpoint 선택·비용과 실행 대상이 검토 가능한 계약이다. 신규 학습·추론·자원 실행은 해당 계약의 명시적 승인 후 진행한다. 이번 완료 학습 승인을 새로운 실행 범위로 확대하지 않는다.

**3. 미노출 평가와 CPU binary 재추론 감사를 각각 설계한다 — 다음 작업 / 실행 승인 필요**

- 아직 결과를 관찰하지 않은 기간·데이터의 실제 확보 가능성을 먼저 확인한다. 기존 Validation이나 이미 개발 중 노출된 Test를 새 독립 평가로 이름만 바꾸지 않는다.
- 구조·비교군·checkpoint·데이터 접근 시점과 전체/큰 수량/시간 지표를 사전에 고정한다. 확보가 어려우면 Validation 개발 근거라는 한계를 유지한다.
- CPU 감사는 보존 checkpoint와 frozen source가 저장 지표를 재현하는지 확인하는 별도 작업이다. SHA·표 검산과 추론 재현을 구분한다. 현재 기본 Runtime으로 과거 실행 재현을 가정하지 않는다.
- 완료 조건은 확보 가능성·동결 범위·실행 대상·필요 승인까지 분리한 평가 계획이다. Test/held-out 또는 혼합 파일을 지금 열지 않는다.

**4. 선행연구와 재현성·부록을 최종 주장에 맞춘다 — 다음 작업 / 독립 문서 점검은 병렬 가능**

- 검색 포함 여부와 평가 범위에 맞춰 Related Work·References·Appendix의 주장과 출처를 대조한다. S2P2·AttNHP, residual/adapters, prototype memory, numerical marks와 간헐 수요를 중심 질문과 연결한다. 이번에 유지한28개 참고문헌은 현재 편집의 보존 기준선이며 새 참신성 검증을 대신하지 않는다.
- Train lag 상관은 데이터 특성으로, 과거 MLP 결과는 별도 구조의 관찰로 유지한다. 데이터·seed·선택 epoch와 여섯 지표, source revision·Runtime 기록을 서로 연결한다.
- 문헌 조사와 독립 문서 검토는 병렬 가능하다. 공통 `main.tex` 편집은 확정된 분석과 포함 결정에 따라 한 세션에서 순서대로 반영한다.
- 완료 조건은 최종 주장마다 직접 근거가 있고 출처·split·재현성·한계가 일관된 상태다.

**5. 저자 편집과 실제 제출 규정을 반영해 최종본을 만든다 — 다음 작업 / 평가·범위 확정 후**

- 현재 확장 초안은 저자 검토용이다. 실제 PAKDD 투고 연도·트랙의 공식 분량·익명성·보충자료·마감 규정을 해당 시점의 공식 출처에서 확인한 뒤 본문과 부록을 줄인다.
- 표·그림·수식·인용을 다시 컴파일·렌더링하고 최종 source/PDF SHA와 증적을 남긴다. 이번 내장 컴파일 성공과 정적 PDF 검증을 실제 제출 규정 충족으로 해석하지 않는다.
- 완료 조건은 저자가 최종 검토한 원고와 제출 점검 기록이다. 실제 투고는 별도의 명시적 제출 승인 이후다.

현재 학습·원본·수치 집계와 원고 반영은 완료다. 검색은 대기이며 CPU binary 재추론·Test·새 독립 평가는 별도 미완료다. 이번 실행 요청은 논문 반영 범위에서 완료했다.
