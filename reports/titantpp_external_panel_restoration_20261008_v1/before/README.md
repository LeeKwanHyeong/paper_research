# TitanTPP 논문 초안 — 도입·선행연구·부록 개정

2026-10-08 개정. [main.tex](main.tex)와 [main.pdf](main.pdf)는 기존 [PAKDD 원고](../titantpp_pakdd_2027_draft/main.tex)의 조판과 서술 방식을 기준으로 다시 작성한 저자 검토용 초안이다. 기존 원고의 성능 수치를 새 원고로 옮기지 않았다.

## 이번 개정 — 완료

- Introduction에 간헐 수요의 발생 간격과 수량 예측 문제를 추가했다. 대표적인 neural TPP의 범주형 사건 유형과 수치형 수량의 차이를 설명하되, TPP 전체가 수치형 mark를 다루지 못한다는 주장은 피했다. Deep Renewal과 FlexTPP를 연결하고, 핵심 질문을 인코더의 이력 요약에 명시적 보정이 주는 추가 이득으로 좁혔다.
- 참고문헌을 **9개에서 28개로 확장**했다. 간헐 수요, neural TPP, 시간 분포, residual/adapter, 메모리, 평가 지표의 선행연구 19개를 복원하고 모두 본문 설명과 연결했다. 기존 원고의 32개 문헌 중 현재 논증과 직접 연결되는 25개와 현 초안의 기본 방법 문헌 3개를 수록했다. [문헌별 대조 기록](../../reports/titantpp_submission_references_appendix_20261008_v1/reference_audit.md)에 복원·제외 이유와 확인 범위를 남겼다.
- 부록 A의 MLP8 Validation 6지표는 그대로 보존했다. 부록 B에 데이터 정의·Train 분포와 이력 길이 그림, C에 Train의 사건 lag 상관 그림, D에 수량·이력 길이별 오차 분해 방법을 추가했다. 데이터 설명은 현 계약의 Train 식별자와 대조했다. [재사용 근거](../../reports/titantpp_submission_references_appendix_20261008_v1/appendix_evidence.md)를 확인할 수 있다.
- 과거 폭4 MLP의 성능표는 현재 MLP8·CNN+GRU54 결과로 옮기지 않았다. Train lag 상관은 데이터 특성이며 보정 경로의 성능 기여를 증명하는 결과로 해석하지 않았다.

## 앞선 재작성에서 유지한 내용 — 현재 기준선

- 응용 문제 → 연구 질문 → 방법 → 대조 설계 → 결과 → 해석의 흐름으로 다시 썼다. 반복적인 상태 설명과 빈 결과표를 제거했다.
- 기존 원고의 LNCS 클래스, Times 계열 글꼴, 표·수식 스타일과 차분한 그림 색상을 따랐다. 익명 저자는 유지했다.
- CNN+GRU54 전체 구조와 fixed/general512 검색 대조를 설명하는 벡터 그림 2개를 작성했다.
- 완료된 **MLP8 구조 54조건의 Validation 결과**를 본문에 반영했다. 부록에는 18개 dataset/arm 조합의 전체·큰 수량 6지표 평균과 표본 SD를 수록했다.
- 수량 개선과 함께 Taxi의 TimeNLL 악화, 일부 큰 수량 오류, 제한된 비용 측정도 적었다. MLP8 결과를 CNN+GRU54 효과로 해석하지 않았다.
- Instacart는 **2026-10-08 14:38 KST 보존 관측**의 단일 완료 fit만 기술했다. 최신 GPU 상태 또는 최종 3seed 비교로 제시하지 않았다.
- 초록부터 결론까지 영어 문장을 정리하고 수식·사실·주장 범위를 별도로 검토했다. 구체적인 변경과 근거는 [Writing Notes](WRITING_NOTES.md)에 있다.

## 문서 검증 — 완료

- PDF **18쪽**, 그림 4개, 표 4개, 참고문헌 28개, 부록 A–D. 저자 검토를 위한 확장 초안이며 제출 분량에 맞춘 축약본은 아니다. 기본 LNCS 본문 크기와 여백을 줄이지 않았다.
- 로컬 TeX Live 2026 / 제공된 LNCS v2.25로 컴파일했다. 정의되지 않은 인용·참조, 중복 링크 목적지, overfull/underfull box가 없다.
- Codex 내장 LaTeX 컴파일도 성공했다. 최종 18쪽을 이미지로 렌더링해 그림·표·수식·페이지 배치를 확인했다. 28개 문헌이 모두 인용되며 누락·중복 인용 키가 없다.
- `amsmath`의 기존 `\vec` 재정의 경고 1건은 사용하지 않는 악센트 정의에 관한 것으로 남아 있다.
- [이번 검증 기록](../../reports/titantpp_submission_references_appendix_20261008_v1/validation.json)과 [수정 전 12쪽 원고 보존 기록](../../reports/titantpp_submission_references_appendix_20261008_v1/before_revision_manifest.json)을 남겼다. 기준 원고 디렉토리의 모든 기존 파일 SHA와 Results·Discussion·Conclusion·기존 부록 A의 내용이 유지됐는지 확인했다. 새 학습·원격 관측·Test 평가·자동화 변경은 수행하지 않았다.

로컬 PDF 재생성 예시:

```sh
cd /Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd2027_cnn_gru_submission_draft_20261008_v1
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.tex`에 수식·TikZ 그림·참고문헌이 들어 있으며 BibTeX 실행은 필요 없다. `llncs.cls`와 `LNCS_README.txt`는 기존 원고의 파일을 수정 없이 복사했다.

## 남은 작업

**CNN+GRU54의 실험 결과를 연결한다 — 진행 중인 실험의 결과 대기**
- 현재 원고의 경험적 근거는 완료된 MLP 연구다. Instacart 9조건, 대표 구조의 보정 27조건, 검색 27조건은 각 단계의 최종 원본과 Validation 결과가 확보되면 반영한다.
- 같은 선택 epoch의 6지표와 3seed 결과, 파라미터 대조, 계산비용을 함께 연결한 뒤 초록·기여·결론을 갱신한다.

**외부 비교와 독립 평가 범위를 확정한다 — 다음 작업**
- S2P2 등 비교군의 입력·출력부·학습·모델 선택 조건을 맞춘 근거를 확인한다. 지금 원고에는 외부 모델 대비 우월성 표를 넣지 않았다.
- 미노출 자료의 확보와 대표 모델 동결 후 독립 평가를 별도로 진행한다.

**저자 최종 편집과 제출 요건을 확인한다 — 외부 작업 대기**
- 저자가 논문 기여의 강도, 문헌 범위, 데이터 출처·재현성 정보, 익명화를 최종 점검한다. 새 결과와 핵심 기여가 정리된 뒤 본문·부록 분량과 당시 제출 요건에 맞춰 축약한다.
