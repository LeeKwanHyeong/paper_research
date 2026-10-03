**RAF 결과 반영 — 완료:** 4.1절 데이터 표와5.3절 RAF8모델3seed 표·해석을 추가했다. 이후 효율표는6·7번으로 정리했고 수식9개와그림2개는 유지했다. [RAF 보고서](../titantpp_raf_execution_20261001_v1/report.md).

# TitanTPP 방법·관련 연구·효율 원고 통합

**후속 문체 수정 — 완료:** 서론의 수요 예측 동기와 설계·성과를 강화하고, 반복되는 방어적 문장을 7절에 정리했다. 수식·표·그림·참고문헌은 유지했다. [수정 기록](claim_evidence_and_writing_notes.md) · [수정 전 사본](before_prose_revision/titantpp_history_mlp_manuscript_20261001_v1.md).

**영문 원고를 연결한다 — 완료**

- [통합 원고](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_history_mlp_manuscript_20261001_v1.md)에 서론·기여, 관련 연구, 방법 수식9개, 실험 설정, 완료된 근거, 직접 효율 측정, 한계를 연결했다.
- 대표는 `titantpp_history_mlp` 한 구조이며 논문에서는 TitanTPP로 표기한다. Full·Gate·활성 분기 정규화는 별도 비교·탐색으로 유지한다.
- 기존 구조 그림과 비용 그림 각1개, 표7개, 기호표, 검증된 1차 문헌9개를 포함한다. 과거 v0.7 및 원본 방법·효율 산출물은 보존했다.
- [주장–근거표·작성 검토](claim_evidence_and_writing_notes.md), [수치 원본 추출](tables.json), [문헌 기록](references.json), [BibTeX](references.bib), [통합 검증](verification.json), [입력 SHA](source_manifest.json)를 함께 남겼다.

**기존 측정과 원고의 일치를 확인한다 — 완료**

- 회수된 동결 source 중 방법 관련8파일과 기존 검증 해시를 대조했다. 식의 CPU 동치·초기화 검사는 완료 증거를 재사용했다.
- 원고의 표를 기존 집계·반복별 실측값과 대조하고, 36개 같은 seed 쌍의 수량 지표 승패와 반례를 확인했다. 신규 학습·GPU 측정·checkpoint 추론은 없다.
- 관련 연구의 제목·저자·발표 정보와 설명을 공식 proceedings/저자 논문으로 확인했다. “TPP는 discrete mark만 가능”, “최초 continuous mark” 주장을 제거하고 기존 연구와 비교 범위를 명시했다.
- 학습 step 시간 개선뿐 아니라 평가 GPU 메모리 증가, 공통 head 적응의 한계, validation 기반 개발, 데이터 구성의 제한도 본문에 넣었다.
- 이 검증은 원고와 기존 증거의 일치 확인이다. 독립 재실험이나 새로운 성능 검증을 의미하지 않는다.

**남은 캠페인의 결과를 회수하고 최종 표를 확정한다 — 다음 작업 / 서버 작업 대기**

- RAF24조건은 원본 회수·CPU 감사와 결과표 통합을 완료했다. 남은 대상은5090 Instacart 추가TPP다. 08:59 KST 저장 관측에서4/6완료, seed62 두 조건 미완료였다. 이번 RAF 통합에서5090을 새로 조회하거나 실행기를 변경하지 않았다.
- 캠페인 종료 확인 후 실패·미완료를 포함해 원본 회수·CPU 감사를 수행하고, 네 데이터의 전체 결과표·selected/last·구간별 반례를 확정한다.
- 그동안 이 원고는 문장 검토가 가능하다. 최종 초록·결론은 미완료 실험 결과로 채우지 않는다.

**독립 평가와 제출본으로 이어간다 — 이후 작업 / 평가 실행·열람 및 제출은 승인 필요**

- 최종 대표·checkpoint·지표·불확실성 계산·split 독립성을 먼저 고정한다. 별도 승인된 held-out 평가 후 초록·결론·주장 범위를 확정한다.
- 제출 당시 공식 분량·익명성을 확인하여 로컬 경로를 제거하고 부록을 편집한다. 이번 원고 통합은 외부 게시·제출·커밋·Push를 포함하지 않는다.
- 현재 작업 순서는 [기준선과 남은 작업](../titantpp_baseline_reset_20261001_v1/README.md)에 반영한다.
