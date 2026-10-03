# TitanTPP 익명 LNCS 초안 작성

> **후속 갱신(2026-10-02):** Appendix B·C와 본문 5.4절 보강 후 내장 컴파일이 통과했다.
> 현재 원고는 표 9개·그림 4개이며, 기존 수치·구조 검사 41개와 새 Appendix 검사 69개를 통과했다.
> 전체 원고 PDF의 지면 검토는 아직 남아 있다. 최신 작업 기록은
> `reports/titantpp_dataset_appendix_20261002_v1/report.md`와 해당 검증 파일을 따른다.
> 아래 내용은 최초 초안 작성 시점의 기록이며, 당시의 컴파일 대기는 현재 상태가 아니다.

## 현재 기준선 — 완료

- 대상은 로컬 `paper_research`의 `paper/titantpp_pakdd_2027_draft/main.tex`다.
  기존 Markdown 원고를 보존하고, 공식 Springer LNCS v2.25 클래스에 맞춘
  별도 익명 LaTeX 편집본을 만들었다.
- 초록, 서론, 관련 연구, 방법, 실험 설정, 결과, 효율, 논의·한계, 결론과
  완료된 후속 실험 일부를 기록한 부록을 작성했다. 벡터 그림 2개, 표 6개,
  실제 인용된 참고문헌 32개를 포함한다.
- 본문은 네 데이터 × 7모델 × 3seed의 완료된 validation 84조건을 기준으로
  작성했다. 진행 중인 학습을 기다리지 않아도 원고 전체를 편집할 수 있다.
- 구조·인용·수치 40개 점검이 통과했다. 이는 LaTeX 컴파일 성공이나 PDF
  지면 검증을 뜻하지 않는다. 학습·재평가·원격 실행은 하지 않았다.

## 주장과 근거 연결 — 완료

| 원고 내용 | 사용한 근거 | 반영 방식 |
| --- | --- | --- |
| 네 데이터 공통 head 비교 | 기존 외부 비교와 Instacart 최종 분석 | 84조건, MAE·RMSE·시간 NLL, 3seed 평균·표본표준편차 |
| 이력 보정 유무 | Core `final_conditions.csv`의 selected validation | B와 원래 MLP 비교; 용량 차이도 있는 대조임을 설명 |
| 단순 기준선 | 후속 준비의 `nonlearned_comparison.csv` | 네 데이터 모두 포함; Intermittent 마지막 수량 MAE 우세와 MLP RMSE 2.04% 개선을 함께 기록 |
| 계산 효율 | 5080 효율 `analysis.json` | 학습 step 시간 5.58–5.74배 차이와 allocated memory 약 75.5% 감소; 평가 메모리 역전도 기록 |
| 진행 중 후속 실험의 완료 부분 | `20261001T221422059513Z/analysis.json` | 10월 2일 07:14 KST의 완료 11조건만 부록에 포함; 실행 중 수치 제외·binary 감사 미완료 표시 |

정확한 근거 경로:

- `reports/titantpp_completed_external_comparison_20261001_v1/comparison.json`
- `reports/titantpp_instacart_final_analysis_20261001_v1/conditions.csv`
- `reports/titantpp_instacart_final_analysis_20261001_v1/verification.json`
- `reports/titantpp_core_ablation_execution_20260928_v1/final_conditions.csv`
- `reports/titantpp_pakdd_extension_preparation_20261001_v1/nonlearned_comparison.csv`
- `reports/titantpp_efficiency_5080_20260930_v1/analysis.json`
- `search_artifacts/titantpp_pakdd_extension_20261001_v1/prelaunch_v2/hourly_monitor/20261001T221422059513Z/analysis.json`

초기 외부 비교의 Instacart 미완료 표시는 이후 최종 분석 자료로 대체했다.
본문 표의 28개 조합과 84조건은 최종 완료 값이다. 후속 부록은 별도 시점의
관측 자료이며, 그 이후 학습 상태를 새로 조회한 결과가 아니다.

## PDF 생성·지면 검증 — 차단됨

내장 `compile_latex_document`는 최초 호출과 3회 재시도 모두
`busy: A LaTeX document is already compiling`을 반환했다. 소스 오류에 대한
컴파일 진단도 받지 못했다. 편집기에 `main.tex`를 여는 요청은 queued로
접수되었다. 실제 PDF 생성·페이지 수·표 넘침·그림 배치는 확인하지 못했다.
별도 TeX 배포판이나 플러그인은 설치하지 않았다.

현재 확인된 학회 규정은 Springer LNCS와 double-blind review다.
공식 PAKDD 2027 연구 트랙 페이지의 분량 제한과 제출 시스템은 초안 작성
시점에 미공지 상태였다. 임의로 12쪽/14쪽 제한을 확정하지 않았고 공식
클래스·여백을 바꾸지 않았다.

## 남은 작업 순서

**1. LaTeX 컴파일과 실제 지면을 확인한다 — 외부 작업 대기**
- 내장 컴파일러가 사용 가능해지면 같은 `main.tex`를 컴파일한다.
- PDF의 모든 페이지에서 표·수식 넘침, 그림 글자 크기, 인용과 참조를 확인한다.
  완료 조건은 실제 출력과 지면 검토를 통과한 PDF다.

**2. 핵심 주장과 본문 분량을 다듬는다 — 다음 작업**
- 위 소스를 기준으로 동기·방법·결과 흐름을 검토한다. 공식 분량 제한이
  공지되면 표와 부록의 배치를 조정한다. 이 작업은 남은 학습과 병행 가능하다.
- 현재 결과를 독립 최종 평가나 모든 데이터의 우월성으로 표현하지 않는다.

**3. 완료되는 후속 결과와 최종 평가를 반영한다 — 진행 중 / 후속 작업**
- 기존 36조건 학습은 별도 승인된 큐에서 계속된다. 원본 회수·검증 후
  완성된 3seed 그룹으로 부록과 해석을 갱신한다.
- 최종 평가 데이터 성격과 실행 계약이 확정되면 해당 절차에 따라 평가하고
  validation 결과와 구분해 반영한다. 초안 작성의 선행 조건은 아니다.

공식 자료: [PAKDD 연구 트랙](https://pakdd2027.org/pages/calls/research),
[Springer proceedings author instructions](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines).
