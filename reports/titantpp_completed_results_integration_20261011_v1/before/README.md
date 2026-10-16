# TitanTPP 논문 초안 — CNN+GRU54 중심 결과 구성

2026-10-08 개정. [main.tex](main.tex)는 기존 [PAKDD 원고](../titantpp_pakdd_2027_draft/main.tex)의 조판과 서술 방식을 따른 저자 검토용 초안이다. 최신 내용은 열려 있는 LaTeX 편집기의 소스와 PDF 미리보기에서 확인한다. 디렉터리의 `main.pdf`는 앞선 18쪽 개정의 정적 출력본이며, 이번 소스 변경을 반영해 다시 내보내지 않았다.

## CNN+GRU54 중심 Results 재배치 — 완료

2026-10-08 사용자 승인에 따라 대표 모델과 본문 성능표의 대상을 일치시켰다.

- **5.1절·Table 2:** 완료된 CNN+GRU54의 Taxi·Intermittent·RAF × 3seed 결과와 외부 6개 공통 출력부 인코더 비교를 제시한다. 평균·표본 SD는 9개 대표 모델 fit과 기존 외부 54개 fit의 선택 Validation 기록에서 검산했다.
- **5.2절:** CNN+GRU54 보정 없음·전체 보정·현재 상태 대조의 질문과 미완료 상태를 구분한다. **5.3절:** mean·weighted·fixed·general512 검색 대조를 이어 배치한다. 두 대조의 신규 54 fit을 기존 대표 모델 완료 9 fit으로 대체하지 않는다.
- **부록 A:** CNN+GRU54의 선택/최종 저장 epoch·실행 GPU 및 전체/큰 수량 6지표. **부록 B:** 과거 MLP4 외부 비교와 단순 규칙. **부록 C:** MLP8 구조·파라미터 대조·전체 수치·시간/큰 수량/비용의 불리한 결과. 기존 Train 분포·lag·오차 분해 부록은 D–F로 이어진다.
- Methods·Algorithm 1·28개 참고문헌·기존 MLP 표 수치·Train 부록 본문을 보존했다. 초록·기여·Discussion·Conclusion은 CNN+GRU54의 완료 성능과 아직 미확인인 기여를 중심으로 정리했다.
- 데이터·출력부·손실·학습 예산·선택 규칙은 비교 가능하지만, 별도 캠페인과 GPU·Runtime 차이는 남는다. seed42 Taxi/RAF는 A100 원본, Intermittent는 A100 E1–12→RTX5090 E13–52의 재개 원본이다. 동일 초기화나 이종 GPU의 구조 효율을 주장하지 않는다.
- Taxi·Intermittent의 수량 이득과 함께 RAF의 S2P2 RMSE 우위, 시간 점수 손해, Intermittent 직전 수량 반복의 MAE/RMSE 우위를 본문에 보존했다. Instacart 대표 모델 3seed 결과는 아직 본문에 없으며 과거 MLP 행으로 채우지 않는다.
- [수치·출처 근거](../../reports/titantpp_cnn_gru_results_reorganization_20261008_v1/evidence.json), [편집·검증 기록](../../reports/titantpp_cnn_gru_results_reorganization_20261008_v1/REVIEW.md), [수정 전 원본](../../reports/titantpp_cnn_gru_results_reorganization_20261008_v1/before_manifest.json)을 남겼다. 별도 PDF는 내보내지 않고 열린 편집기의 PDF 미리보기를 갱신한다.

아래는 이전 개정의 기록이며 현재 본문 위치는 위 구성을 따른다.

## 앞선 외부 비교군 결과 복원 — 완료

- **5.1절**에 RMTPP·NHP·THP·SAHP·S2P2·AttNHP와 과거 TitanTPP의 네 데이터 Validation 비교를 추가했다. 84개 선택 실행에서 28개 그룹의 평균·표본 SD를 다시 계산했으며, 원고의 168개 mean/SD 값이 기존 표와 소수점 네 자리까지 일치한다.
- 과거 TitanTPP 행은 **MLP4**(폭4·8분기·명목 보정 파라미터6,144)로 명명했다. MLP8 내부 대조와 현재 CNN+GRU54의 실험 결과로 대체하지 않았다. 외부 여섯 모델도 수치 입력·공통 출력부로 적응한 인코더 비교임을 명시했다.
- 동일 관측창의 직전 수량 반복과 수량 평균을 단순 기준선으로 추가했다. Instacart의 S2P2/AttNHP RMSE 우세, timing 손해, Intermittent 직전 수량 반복의 MAE 우세를 보존했다.
- 과거와 현재 계약19쌍의 데이터 모집단·입력창·출력부·손실·optimizer·선택 규칙을 대조했다. Runtime·초기 파라미터·학습 경로·checkpoint 동일성을 주장하지 않는다. 동결 소스691개와 Methods manifest8개 raw SHA가 일치했다.
- 초록·Introduction·실험 설계·Discussion·Conclusion을 새 증거 범위에 맞췄다. Methods 수식·Algorithm1·기존 MLP8 표의 수치는 보존했다. 내장 LaTeX 컴파일이 성공했다.
- [수치 근거](../../reports/titantpp_external_panel_restoration_20261008_v1/panel_evidence.md), [계약 감사](../../reports/titantpp_external_panel_restoration_20261008_v1/contract_audit.md), [최종 검증](../../reports/titantpp_external_panel_restoration_20261008_v1/validation.json)에 출처·SHA·검증 범위를 남겼다. 새 학습·추론·원격 관측·Test 평가·자동화 변경은 수행하지 않았다.

## 앞선 도입·선행연구·부록 개정 — 완료

- Introduction에 간헐 수요의 발생 간격과 수량 예측 문제를 추가했다. 대표적인 neural TPP의 범주형 사건 유형과 수치형 수량의 차이를 설명하되, TPP 전체가 수치형 mark를 다루지 못한다는 주장은 피했다. Deep Renewal과 FlexTPP를 연결하고, 핵심 질문을 인코더의 이력 요약에 명시적 보정이 주는 추가 이득으로 좁혔다.
- 참고문헌을 **9개에서 28개로 확장**했다. 간헐 수요, neural TPP, 시간 분포, residual/adapter, 메모리, 평가 지표의 선행연구 19개를 복원하고 모두 본문 설명과 연결했다. 기존 원고의 32개 문헌 중 현재 논증과 직접 연결되는 25개와 현 초안의 기본 방법 문헌 3개를 수록했다. [문헌별 대조 기록](../../reports/titantpp_submission_references_appendix_20261008_v1/reference_audit.md)에 복원·제외 이유와 확인 범위를 남겼다.
- 부록 A의 MLP8 Validation 6지표는 그대로 보존했다. 부록 B에 데이터 정의·Train 분포와 이력 길이 그림, C에 Train의 사건 lag 상관 그림, D에 수량·이력 길이별 오차 분해 방법을 추가했다. 데이터 설명은 현 계약의 Train 식별자와 대조했다. [재사용 근거](../../reports/titantpp_submission_references_appendix_20261008_v1/appendix_evidence.md)를 확인할 수 있다.
- 과거 폭4 MLP의 성능표는 이번 5.1절에서 독립된 역사적 비교로 복원했으며, 현재 MLP8·CNN+GRU54 결과로 옮기지 않았다. Train lag 상관은 데이터 특성이며 보정 경로의 성능 기여를 증명하는 결과로 해석하지 않았다.

## 앞선 재작성에서 유지한 내용 — 현재 기준선

- 응용 문제 → 연구 질문 → 방법 → 대조 설계 → 결과 → 해석의 흐름으로 다시 썼다. 반복적인 상태 설명과 빈 결과표를 제거했다.
- 기존 원고의 LNCS 클래스, Times 계열 글꼴, 표·수식 스타일과 차분한 그림 색상을 따랐다. 익명 저자는 유지했다.
- CNN+GRU54 전체 구조와 fixed/general512 검색 대조를 설명하는 벡터 그림 2개를 작성했다.
- 완료된 **MLP8 구조 54조건의 Validation 결과**를 본문에 반영했다. 부록에는 18개 dataset/arm 조합의 전체·큰 수량 6지표 평균과 표본 SD를 수록했다.
- 수량 개선과 함께 Taxi의 TimeNLL 악화, 일부 큰 수량 오류, 제한된 비용 측정도 적었다. MLP8 결과를 CNN+GRU54 효과로 해석하지 않았다.
- Instacart는 **2026-10-08 14:38 KST 보존 관측**의 단일 완료 fit만 기술했다. 최신 GPU 상태 또는 최종 3seed 비교로 제시하지 않았다.
- 초록부터 결론까지 영어 문장을 정리하고 수식·사실·주장 범위를 별도로 검토했다. 구체적인 변경과 근거는 [Writing Notes](WRITING_NOTES.md)에 있다.

## 앞선 18쪽 출력본의 검증 기록 — 완료

- 외부 표·Algorithm 추가 이전의 PDF **18쪽**, 그림 4개, 표 4개, 참고문헌 28개, 부록 A–D. 저자 검토를 위한 확장 초안이며 제출 분량에 맞춘 축약본은 아니다. 기본 LNCS 본문 크기와 여백을 줄이지 않았다.
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

**대표 구조의 보정·검색 대조 결과를 연결한다 — 외부 작업 대기**
- 승인된 63 fit 중 Instacart 9 fit, 보정 27 fit, 검색 27 fit의 완료 원본·동일 선택 epoch의 6지표·비용을 확보한 순서대로 반영한다. 보정 기여 검증이 검색 기여 해석보다 먼저다.
- 현재 CNN+GRU54 외부 비교는 완료된 이전 9 fit의 성능 위치이며, 새 대조에서 재사용할 mean은 새 보정 단계의 full 9 fit이다.

**미노출 독립 평가와 저자 최종 편집을 준비한다 — 다음 작업**
- 대표 모델과 주장 범위를 동결한 뒤 미노출 자료의 확보·평가 계약을 정리한다. 기존 Validation을 독립 평가로 표기하지 않는다.
- 저자는 결과가 들어온 원고의 기여·출처·재현성·익명화와 제출 규정을 확인하고 본문·부록 분량을 조정한다.
