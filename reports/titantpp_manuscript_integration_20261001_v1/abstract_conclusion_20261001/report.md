# TitanTPP 초록·결론 작성 기록

**확정 근거를 초록·결론으로 연결한다 — 완료**

사용자 요청:

> 확정된 네 데이터의 validation 결과와 효율 근거를 연결해 논문의 핵심 메시지를 정리합니다. 진행하자

- [원고](../../../paper/titantpp_history_mlp_manuscript_20261001_v1.md)에 Abstract와 8. Conclusion을 추가했다. 공백 기준 초록212단어, 결론237단어다. 별도로 지정된 단어 수나 학회 서식 제한은 없었으며 이번 분량은 편집상 선택이다.
- 기존 서론·관련 연구·방법·평가 설정·결과·비용·한계·기호·참고문헌은 수정하지 않았다. [기록된 두 삽입](changes.json)과 [수정 전 SHA](before_sha256.json)로 확인한다.
- 영어 학술 문장 작성에 `oma-academic-writer`를 적용했다. 국문 메모는 작업 기록이며 별도 번역본이 아니다.

## 핵심 메시지와 근거

| 초록·결론의 주장 | 근거 | 범위 |
|---|---|---|
| 인과적 gap–quantity encoder 사이에 인접 contextual state의 bottleneck correction을 삽입하며 static bank를 함께 사용 | 원고3절 식1–9, [동결 코드 대응](../../titantpp_method_efficiency_20260930_v1/code_equation_map.md) | 구조 설명. 여러 lag를 읽거나 모든 memory를 제거했다는 주장이 아님 |
| 네 데이터·6개 외부 encoder·3seed 비교이며 공통 input/head/loss/checkpoint 선택 기준을 적용 | 원고4절·표4, [기계판독 표](../tables.json), [Instacart 확정 분석](../../titantpp_instacart_final_analysis_20261001_v1/analysis.json) | validation 결과. 원논문의 native likelihood 결과와 구분 |
| Taxi·Intermittent의 RMSE가 최상 외부 비교군보다15.85%·34.39% 낮고, 모든 같은 seed 쌍에서 MAE·RMSE가 낮음 | [기존 비교](../../titantpp_completed_external_comparison_20261001_v1/comparison.json), 표4, 36개 같은 seed 비교 | 해당 두 데이터에 한정. 유의성·일반적 SOTA 주장 없음 |
| RAF는 근소한 RMSE 이득과 최상 MAE 비교군 대비 열세, Instacart는 추가 두 비교군 대비 작은 MAE 이득과 RMSE 열세 | [RAF 결과](../../titantpp_raf_execution_20261001_v1/comparison.json), [Instacart 결과](../../titantpp_instacart_final_analysis_20261001_v1/analysis.json) | 고정된 대표 TitanTPP의 지표별 차이를 유지 |
| Instacart의 고수량 오차가 작은 수량에서의 이득을 상쇄 | 같은 Instacart 분석의 quantity strata·SSE 기여도 | 관측된 오차 분포 설명. 아키텍처의 인과 효과로 단정하지 않음 |
| MAC adapter의 학습 step 시간이5.58–5.74배이며 TitanTPP의 peak allocated 학습 메모리가약75.5% 낮음 | [동일 입력 비용 실측](../../titantpp_efficiency_5080_20260930_v1/analysis.json), 원고표6·7 | Taxi·Intermittent의 warmed batch·동일GPU 실측. 전체epoch·수렴시간·전체데이터 비용으로 확대하지 않음 |
| 평가 batch 속도는 개선되지만 평가 메모리 할당은 높음 | 같은 비용 기록의 evaluation 행 | 학습/평가 경로를 구분해 효율 반례 포함 |

## 문단 흐름과 문체 검토

- 초록은 예측 문제 → 제안 구조 → 비교 조건 → 두 데이터의 정량적 성과 → RAF·Instacart·시간 likelihood의 차이 → batch 비용 → 적용 메시지 순서다.
- 결론은 구조 요약, 네 데이터의 결과 해석, 비용과 후속 평가의 세 문단이다. `combines`, `relates`, `reduces`, `identifies`, `complement`처럼 실제 동작과 관측을 설명하는 동사를 사용했다.
- 확정된 실측은 직접 서술하고, 논문의 넓은 해석은 `support`로 표현했다. 비선형 신호나 인과 원인을 확정하거나 모든 데이터에서 우월하다고 쓰지 않았다.
- 반복적인 면책 문장을 새로 늘리지 않았다. `validation`과 `matched Taxi and Intermittent batch measurements`로 적용 범위를 문장 안에 적고, 상세 제한은 기존7절에 유지했다.
- 긴 비교 문장 사이에 짧은 문장으로 데이터 범위와 시간 likelihood·평가 메모리 차이를 배치했다. 과장된 최초성, 광고성 수식어, 이중 완곡 표현, 허구의 인용은 추가하지 않았다.
- 초록의 수치와 결론의 주장을 기존 표·paired 비교·비용 반복 측정과 대조한다. [통합 검증](../verification.json)은 현재 원고를 검사하고, 새 절을 역적용한 사본으로 앞선 수정 이력의 검증도 유지한다.

**독립 평가 기준을 문서로 고정한다 — 다음 작업**

- split 접근 이력, 대표 모델/checkpoint, 최종 지표와 불확실성 계산을 정리한다. 독립 평가 실행·held-out 열람은 별도 승인 범위다.

**제출 형식으로 편집한다 — 이후 작업**

- 평가 범위에 맞춰 최종 문구를 확정하고, 학회 양식·분량·익명성·부록·자료공개 범위를 정리한다. 이번 작업은 로컬 원고 작성이며 학습·GPU·원격 관측·스케줄러·commit/push·외부 게시·제출은 없다.
