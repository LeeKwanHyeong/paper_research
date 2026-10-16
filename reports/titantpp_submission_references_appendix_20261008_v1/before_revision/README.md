# TitanTPP 논문 초안 — 기존 LNCS 원고 형식으로 재작성

2026-10-08 개정. [main.tex](main.tex)와 [main.pdf](main.pdf)는 기존 [PAKDD 원고](../titantpp_pakdd_2027_draft/main.tex)의 조판과 서술 방식을 기준으로 다시 작성한 저자 검토용 초안이다. 기존 원고의 성능 수치를 새 원고로 옮기지 않았다.

## 이번 개정 — 완료

- 응용 문제 → 연구 질문 → 방법 → 대조 설계 → 결과 → 해석의 흐름으로 다시 썼다. 반복적인 상태 설명과 빈 결과표를 제거했다.
- 기존 원고의 LNCS 클래스, Times 계열 글꼴, 표·수식 스타일과 차분한 그림 색상을 따랐다. 익명 저자는 유지했다.
- CNN+GRU54 전체 구조와 fixed/general512 검색 대조를 설명하는 벡터 그림 2개를 작성했다.
- 완료된 **MLP8 구조 54조건의 Validation 결과**를 본문에 반영했다. 부록에는 18개 dataset/arm 조합의 전체·큰 수량 6지표 평균과 표본 SD를 수록했다.
- 수량 개선과 함께 Taxi의 TimeNLL 악화, 일부 큰 수량 오류, 제한된 비용 측정도 적었다. MLP8 결과를 CNN+GRU54 효과로 해석하지 않았다.
- Instacart는 **2026-10-08 14:38 KST 보존 관측**의 단일 완료 fit만 기술했다. 최신 GPU 상태 또는 최종 3seed 비교로 제시하지 않았다.
- 초록부터 결론까지 영어 문장을 정리하고 수식·사실·주장 범위를 별도로 검토했다. 구체적인 변경과 근거는 [Writing Notes](WRITING_NOTES.md)에 있다.

## 문서 검증 — 완료

- PDF **12쪽**, 그림 2개, 표 3개, 참고문헌 9개. 기본 LNCS 본문 크기와 여백을 줄이지 않았다.
- 로컬 TeX Live 2026 / 제공된 LNCS v2.25로 컴파일했다. 정의되지 않은 인용·참조, 중복 링크 목적지, overfull/underfull box가 없다.
- Codex 내장 LaTeX 컴파일도 성공했다. 최종 12쪽을 이미지로 렌더링해 그림·표·수식·페이지 배치를 확인했다.
- `amsmath`의 기존 `\vec` 재정의 경고 1건은 사용하지 않는 악센트 정의에 관한 것으로 남아 있다.
- [검증 기록](../../reports/titantpp_submission_rewrite_20261008_v1/validation.json)과 [이전 버전 보존 기록](../../reports/titantpp_submission_rewrite_20261008_v1/before_revision_manifest.json)을 남겼다. 새 학습·원격 관측·Test 평가·자동화 변경은 수행하지 않았다.

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
- 저자가 논문 기여의 강도, 문헌 범위, 데이터 출처·재현성 정보, 익명화를 최종 점검한다. 새 결과가 들어오면 12쪽 제한도 다시 확인해야 한다.
- 이 파일은 AI가 재작성한 내부 초안이다. 기존 README에서 확인한 [PAKDD 2027 Author's Kit](https://pakdd2027.org/pages/calls/authors-kit)의 생성형 AI 사용 제한과 신고 요건은 별도 제출 조건이며, 저자 수정만으로 준수가 자동 보장되는 것은 아니다.
