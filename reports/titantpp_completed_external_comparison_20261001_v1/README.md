# 완료 실험과 외부 비교군 전체 비교

RAF24조건을 포함한 **139개 고유 완료조건, 42개 3seed 그룹**을 통합했다. RAF는 2026-10-01 08:55:56 KST 종료 후 원본 SHA·CPU 감사 결과다. 추가 TPP는08:59 관측의 완료16/18조건을 반영했으며 진행 중·미시작 seed62 두 조건은 완료표에 넣지 않았다.

[전체 보고서](report.md) · [기계판독 집계](comparison.json) · [선택 checkpoint 수치](selected_conditions.csv) · [검증](verification.json) · [RAF 상세](../titantpp_raf_execution_20261001_v1/report.md)

[기존 로컬 보고서](http://127.0.0.1:4181/?view=1)에서 RAF를 선택하면 8모델의 평균±표본표준편차와 동일 seed42 상세를 볼 수 있다. 기존 앱 ID·표·사용자 편집 구조를 유지했다. 공개 게시하지 않았다.

5080 추가TPP12조건·Instacart정규화1조건·RAF24조건의 원본 CPU 감사는 완료다. 5090 추가 TPP 최종 binary 감사는 남아 있다. 기존114조건의 수치는 보존했으며 이전 보고서는 before_raf_integration/에 있다.

build_analysis.py로 원본 집계와 Markdown을 재생성하고, sync_app_data.py로 기존 앱의 reviewed rows를 갱신한다. RAF 원본 감사·세부 집계는 ../titantpp_raf_execution_20261001_v1/에 있다.
