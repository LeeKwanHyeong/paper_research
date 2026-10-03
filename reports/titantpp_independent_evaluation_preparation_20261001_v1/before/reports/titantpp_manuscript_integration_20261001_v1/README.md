**초록·결론 작성 — 완료:** 확정된 네 데이터 validation 비교와 동일 입력 효율 자료로 초록212단어·결론237단어를 추가했다. 기존 본문·수치·그림·참고문헌은 유지했다. [작성·근거 기록](abstract_conclusion_20261001/report.md).

**Instacart 최종 결과 원고 반영 — 완료:** 5090 원본·CPU 감사가 끝난 S2P2·AttNHP의 3seed 평균·표본표준편차로 표4의 Pending 두 행을 채웠다. 네 데이터 × 7모델 × 3seed의 84조건이 모두 표에 연결됐다. 5.2절의 지표별 상충·수량 구간 해석과 7절의 상태를 갱신했다. [변경 기록](instacart_final_integration_20261001/report.md) · [확정 분석](../titantpp_instacart_final_analysis_20261001_v1/report.md). 아래 개별 편집 기록의 Pending 서술은 당시 이력이다.

**ICLR 5편의 공식 링크 정리 — 완료:** AttNHP·Intensity-Free·PatchTST·AdamW·EasyTPP의 본문 인용·참고문헌·BibTeX를 OpenReview로 연결했다. 저자 순서·연도·학회명과 본문 내용은 유지했다. [변경 기록](official_links_20261001/report.md).

**참고문헌 32편 재대조 — 완료:** 공식 출판·저자 기록과 DOI를 다시 대조했다. 잘못된 논문 식별·저자 순서·출판 연도·기존 DOI는 발견하지 않았고 8편의 누락 정보·출판본 링크·BibTeX 성씨 처리를 보완했다. [항목별 대조와 변경](reference_doublecheck_20261001/report.md). 아래 보강 작업 기록은 수정 당시의 이력이다.

**참고문헌·관련 연구 보강 — 완료:** 기존 9편에 23편을 추가해 32편으로 확장하고 서론·관련 연구·방법·평가 절의 인용과 BibTeX를 연결했다. [보강 내용과 남은 작업](reference_expansion_review.md) · [수정 전 원고](before_reference_expansion/titantpp_history_mlp_manuscript_20261001_v1.md).

**구조 그림 개편 — 완료:** Figure 1을 전체 구조와 History correction 확대의 두 패널로 교체했다. 고정 /8·동일 직전 사건·8개 독립 분기·잔차 삽입 위치를 표현한다. 원래 그림과 [수정 전 원고](before_architecture_revision/titantpp_history_mlp_manuscript_20261001_v1.md)는 보존했다. [그림 제작·검증](../titantpp_architecture_figure_20261001_v2/README.md) · [참고문헌 보강 검토](reference_coverage_review.md).

**RAF 결과 배치 조정 — 완료:** RAF를 5.2절 네 데이터 공통 결과표(표4)에 통합하고 지표별 차이를 한 문단으로 설명했다. 5.3절은 구성 비교로 바꾸어 정규화 2구조 표(표5)를 배치했다. 대표 TitanTPP·원본 결과·수식9개·그림2개·효율표6·7은 유지했다. Instacart 추가 비교군의 미완료 3seed 평균은 Pending으로 표시했다. [변경 전 원고](before_raf_repositioning/titantpp_history_mlp_manuscript_20261001_v1.md) · [RAF 보고서](../titantpp_raf_execution_20261001_v1/report.md).

# TitanTPP 방법·관련 연구·효율 원고 통합

**후속 문체 수정 — 완료:** 서론의 수요 예측 동기와 설계·성과를 강화하고, 반복되는 방어적 문장을 7절에 정리했다. 수식·표·그림·참고문헌은 유지했다. [수정 기록](claim_evidence_and_writing_notes.md) · [수정 전 사본](before_prose_revision/titantpp_history_mlp_manuscript_20261001_v1.md).

**영문 원고를 연결한다 — 완료**

- [통합 원고](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_history_mlp_manuscript_20261001_v1.md)에 서론·기여, 관련 연구, 방법 수식9개, 실험 설정, 완료된 근거, 직접 효율 측정, 한계를 연결했다.
- 대표는 `titantpp_history_mlp` 한 구조이며 논문에서는 TitanTPP로 표기한다. Full·Gate·활성 분기 정규화는 별도 비교·탐색으로 유지한다.
- 기존 구조 그림과 비용 그림 각1개, 표7개, 기호표, 본문에 연결한 참고문헌32편를 포함한다. 과거 v0.7 및 원본 방법·효율 산출물은 보존했다.
- [주장–근거표·작성 검토](claim_evidence_and_writing_notes.md), [수치 원본 추출](tables.json), [문헌 기록](references.json), [BibTeX](references.bib), [통합 검증](verification.json), [입력 SHA](source_manifest.json)를 함께 남겼다.

**기존 측정과 원고의 일치를 확인한다 — 완료**

- 회수된 동결 source 중 방법 관련8파일과 기존 검증 해시를 대조했다. 식의 CPU 동치·초기화 검사는 완료 증거를 재사용했다.
- 원고의 표를 기존 집계·반복별 실측값과 대조하고, 36개 같은 seed 쌍의 수량 지표 승패와 반례를 확인했다. 신규 학습·GPU 측정·checkpoint 추론은 없다.
- 관련 연구의 제목·저자·발표 정보와 설명을 공식 proceedings/저자 논문으로 확인했다. “TPP는 discrete mark만 가능”, “최초 continuous mark” 주장을 제거하고 기존 연구와 비교 범위를 명시했다.
- 학습 step 시간 개선뿐 아니라 평가 GPU 메모리 증가, 공통 head 적응의 한계, validation 기반 개발, 데이터 구성의 제한도 본문에 넣었다.
- 이 검증은 원고와 기존 증거의 일치 확인이다. 독립 재실험이나 새로운 성능 검증을 의미하지 않는다.

**네 데이터의 최종 validation 비교를 원고에 연결한다 — 완료**

- 5090 Instacart6조건의 원본223파일·source106개·checkpoint12개 회수와 CPU 감사를 마쳤다. 기존5080 추가TPP12조건의 감사를 재사용해 추가비교군18조건을 모두 확인했다.
- 표4는28개3seed 그룹이다. MAE·RMSE·시간NLL은 동일 RMSE 선택checkpoint에서 집계했고, Instacart의 불리한RMSE와 seed 결과도 포함했다.
- 기존08:59 보고서와 이전 분석의 ‘원고 미반영’ 기록은 당시 증거로 보존했다. 이번 원고 반영은 별도 변경 기록으로 연결한다.

**확정 결과에 맞춰 초록·결론을 작성한다 — 완료**

- [원고](../../paper/titantpp_history_mlp_manuscript_20261001_v1.md)의 Abstract와8절에 구조, Taxi·Intermittent의 수량 개선, RAF·Instacart의 차이, 실측 비용을 연결했다.
- 초록과 결론 모두 validation 비교를 명시하고, 학습 메모리 감소와 평가 메모리 증가를 함께 설명했다. 과거 ‘초록·결론 미작성’ 메모는 각 편집 당시의 이력이다.

**독립 평가 기준을 문서로 고정한다 — 완료**

- [독립 평가 기준](../titantpp_independent_evaluation_protocol_20261001_v1/protocol.md)에 고정84조건, RMSE·MAE·시간NLL, seed SD와paired bootstrap 절차를 기록했다. 이번에는 원고와 validation 수치를 변경하지 않았다.
- 과거 test 접근 이력 때문에 미접근 독립성은 아직 확정하지 않았다. 로컬48개checkpoint의 byte SHA를 대조했으며, 기존 외부4종36개 selected binary는 원본 회수·CPU검증이 남았다. [목록과 남은 작업](../titantpp_independent_evaluation_protocol_20261001_v1/README.md).

**독립 평가와 제출본으로 이어간다 — 이후 작업 / 평가 실행·열람 및 제출은 승인 필요**

- 다음은 split 계보·실제 접근 이력과36개binary 준비를 마무리하고, 합성·validation 기반 실행기 검증과 실제 평가 계약을 확정하는 작업이다. 별도 승인된 평가 후 현재 초록·결론의 주장 범위를 다시 대조한다.
- 제출 당시 공식 분량·익명성을 확인하여 로컬 경로를 제거하고 부록을 편집한다. 이번 원고 통합은 외부 게시·제출·커밋·Push를 포함하지 않는다.
- 현재 작업 순서는 [기준선과 남은 작업](../titantpp_baseline_reset_20261001_v1/README.md)에 반영한다.
