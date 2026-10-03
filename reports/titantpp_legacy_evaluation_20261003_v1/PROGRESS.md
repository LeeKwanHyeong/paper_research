# 기존 test 분할 재평가 진행 기록

**현재 기준선과 승인 범위 — 완료**

- 기존 4개 데이터셋, 9개 학습 모델 × 3개 시드의 108개 선택 체크포인트를 고정했습니다. 새 학습이나 test 기반 모델 재선택은 하지 않았습니다.
- 이번 결과는 기존 test 분할의 사후 재평가(retrospective reevaluation)입니다. 독립 미접근 자료 평가나 v0.7 validation 기준선의 대체로 표현하지 않습니다.

**전체 처리 환경과 validation 재현 확인 — 완료**

- CPU 36조건·CUDA 36조건의 표본 검증과 대표 모델의 네 데이터셋 전체 validation 재현을 통과했습니다.
- 첫 배포의 빈 `sample_data` 디렉터리 누락으로 발생한 CPU import 실패 36개는 `failed_deployment_1`에 보존했습니다. 당시 예측·test 생성은 0개였고, 빈 폴더 복원 후 코드·데이터·체크포인트와 최초 24시간 마감을 유지했습니다.

**기존 test 전체 평가와 원본 회수 — 완료**

- 108조건 모두 성공했으며 실패는 0개입니다. 조건별 대상 수는 Taxi 8,327개, Intermittent 88,019개, Instacart 578,387개, RAF 5,226개입니다.
- 원본 3,094개 파일을 회수하고 SHA를 대조했습니다. 독립 감사에서 체크포인트·선택 epoch·고정 소스·행 수·파라미터 불변 증적이 일치했습니다. 보존된 초기 실패의 font cache 36개는 명시적 회수 제외 대상으로 구분했습니다.

**고정 통계 집계와 보고 — 완료**

- 학습 모델 108조건과 결정적 기준 8개 벡터를 집계했습니다. 산출 가능한 paired bootstrap은 10,000회 완료했고 빈 재표집은 없었습니다. Taxi 168시간 기본 구간은 기간 부족으로 미산출이며 민감도 구간으로 대체하지 않았습니다.
- 외부 모델의 실제 test RMSE 최저는 네 데이터셋 모두 S2P2입니다. validation 고정 anchor는 RMTPP/RMTPP/S2P2/S2P2 그대로 유지했습니다.
- Intermittent의 수량 결과와 시간 NLL 악화를 함께 기록했습니다. 다른 데이터셋에서의 불리하거나 불확실한 비교와 구조·결정적 기준 결과도 보존했습니다.
- [최종 보고서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/FINAL_REPORT.md), [전체 결과표](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/review/summary.md), [독립 감사](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/final_artifact_audit.json)에 근거를 연결했습니다.

**원고 본문 갱신과 수치 대조 — 완료**

- [main.tex](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex)의 초록·방법·test 결과표·논의·결론·Appendix A를 갱신했습니다. 기존 validation·효율·분포·구간별 수치표 8개를 보존했습니다.
- 원본 집계와 새 표시 수치 272개가 일치하며 내부 참조와 참고문헌 연결도 확인했습니다. 이중 반올림 4개 표시값은 원본 정밀도에서 직접 반올림해 수정했습니다.
- 최종 소스에서 Codex 내장 LaTeX 컴파일이 성공했습니다. PDF 전체 페이지의 시각적 조판 검수는 별도로 수행하지 않았습니다.

**남은 작업 — 승인 범위 완료**

- 이번 기존 분할 재평가의 미완료 조건은 없습니다. 추가 학습·대표 모델 재선택 없이 결과와 해석 한계를 원고에 연결했습니다.
- 투고용 분량·형식·최종 PDF 조판 검토는 별도 편집 작업입니다. [전체 완료 기록](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/FINAL_REPORT.md)을 현재 기준선으로 사용합니다.
