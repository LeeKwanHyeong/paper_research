# Instacart 최종 회수·검증·분석 — 완료

5090 추가 TPP 6조건의 원본 회수와 CPU 검증, 기존 TitanTPP 및 외부6종의 3seed 비교를 완료했다. 현재 원고와 기존 전체 비교 보고서는 이번 작업에서 수정하지 않았다.

- [결과 분석](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/report.md)
- [기계판독 결과](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/analysis.json)
- [검증](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_instacart_final_analysis_20261001_v1/verification.json)

`build_analysis.py`와 `write_report.py`는 회수된 validation 원본으로 분석을 재생성한다. `collect.py additional`은 5090 읽기 전용 회수이며 기존 목적지를 덮어쓰지 않는다. `audit.py additional`은 CPU 상태 복원·기록 검증이며 forward를 호출하지 않는다.
