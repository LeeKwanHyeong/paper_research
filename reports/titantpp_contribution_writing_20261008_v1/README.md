# TitanTPP 본문 초안과 기여 근거 정리

2026-10-08 작성. **Methods·실험 설계·표적 선행연구 대조와, 완료 근거의 Results·Discussion 작성을 완료했다.** 작성에 사용한 관측 기준으로 승인된 63조건 캠페인은 Instacart 단계가 진행 중이며, Q1/Q2 결과는 대기 상태다. 그 성과를 본문에 미리 확정하지 않았다.

- [영문 연구 초안](/Users/igwanhyeong/PycharmProjects/paper_research/paper/drafts/titantpp_contribution_methods_20261008_v1.md): 중심 질문, 관련 방법, 정확한 모델 수식, 대조군,63조건 설계, 지표·비용·판정 기준.
- [주장–근거 지도와 작성 메모](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_writing_20261008_v1/WRITING_NOTES.md): 기존 근거·새 검증·미확인의 경계, 코드 위치와 작성 시점 상태.
- [S2P2·선행연구 비교](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_writing_20261008_v1/LITERATURE_COMPARISON.md): S2P2, deep renewal, RMTPP, A-NHP, persistent memory, product-key memory의 방법 대조.
- [검토 결과](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_writing_20261008_v1/REVIEW.md), [원본 해시 근거](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_writing_20261008_v1/evidence_manifest.json).

- [완료 결과의 본문 반영](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contribution_writing_20261008_v1/RESULTS_UPDATE_20261008.md): MLP8 구조54 최종 Validation과 Instacart 완료1조건의 중간 기록, 시간·비용 손해.

## 현재 논문 기여의 구성

| 주제 | 지금 쓸 수 있는 설명 | 성과 주장 상태 |
|---|---|---|
| 주 기여: 명시적 이력 보정 | 이력을 학습하는 대표 구조 안에서 국소·순환 보정 경로의 추가 효과를 비교한다. 보정 없음과 같은 활성 파라미터 수의 현재 위치 전용 대조를 둔다. | 새27조건 대기. 기존 MLP8 이득은 동기 근거. |
| 보조 기여: 고정 비중 prototype 검색 | 시간·수량 query를 별도 정규화하는 제약이 weighted·generic512보다 유리한지 검증한다. | 새27조건과 mean9 재사용 대기. 통과 전 두 번째 확정 기여가 아님. |
| 외부 모델과의 성능 관계 | 동일 입력·출력 계약의 adaptation 비교로 설명하고 원논문 native 모델과 구분한다. | 현재63에는 외부 재학습 없음. 최종 대표 모델 기준의 정합과 평가가 남음. |

실제 CNN+GRU54는 세 위치 Q/K/V 필터와 GRU 누적 상태를 함께 사용한다. 따라서 “직전 상태 하나만의 효과”라고 단정하지 않는다. 또한 prototype64는 학습된 정적 벡터이며 실제 사건을 저장한 bank가 아니다. 이 두 사항은 원고 전체에 반영했다.

## 주제별 완료 여부

| 작업 | 상태 | 완료 조건/범위 |
|---|---|---|
| Methods와 실험 설계 초안 | 완료 | 동결 source 수식·대조군·63조건·선택 규칙 연결 |
| 기존 근거와 미확인 주장 구분 | 완료 | MLP8/parity/새Validation/독립평가 분리 |
| S2P2 adaptation 및 6편 방법 대조 | 완료 | primary source에 기반한 표적 비교. 포괄적 최초성 조사는 아님. |
| 완료 MLP8 Results·Discussion 및 Instacart 중간 행 | 완료 | 54조건 평균/표본SD 재계산 및 원본 SHA 대조 |
| 새 Q1/Q2 성과 판정 | 다음 작업 | 자동화의 원본 검증·최종Validation 집계 이후 |
| 투고 가능한 최종 원고 | 외부 작업 대기 | 저자 작성 과정·학회 AI 정책·최종 근거 반영 필요 |

## 남은 작업 순서

**1. 승인된 실험의 단계별 원본과 결과를 확정한다 — 진행 중**
- `paper_research`의 기존63조건 자동화가 Instacart→보정→검색 순으로 진행한다. 이 문서 작업에서는 원격 명령·재학습·자동화 변경을 하지 않았다.
- 각 단계의 실제 종료·원본 SHA·같은 선택 epoch Validation을 확인한 뒤 수량 이득과 시간·큰수량·계산비용 손해를 함께 본문에 반영한다.

**2. 대표 구조의 주장을 판정하고 외부 비교의 근거를 맞춘다 — 다음 작업**
- Q1을 우선 판정하고 Q2는 별도 기준으로 채택 여부를 결정한다. 통과하지 않은 가설을 기여 수에 포함하지 않는다.
- 기존 외부 모델의 source·출력부·seed·Runtime·선택 규칙을 정리한다. 현재 코드가 같다는 사실만으로 과거 checkpoint 성과를 결합하지 않는다.

**3. 독립 평가 자료와 가장 가까운 선행연구를 보강한다 — 다음 작업**
- 자료·문헌 확인은 학습과 병렬로 진행할 수 있다. 미노출 기간·데이터의 접근 이력을 확인한다. 현재 Validation 재사용과 새 독립 평가를 구분한다.
- time–quantity multi-view query의 인용망과 단순 상태 연결 대조를 더 확인한다. 추가 GPU 조건을 이 문서에서 자동으로 늘리지 않는다.

**4. 저자 중심의 최종 투고문과 표를 작성한다 — 다음 작업**
- 근거가 확정된 범위에서 대표 모델·주장·지표 표를 동결한다. 문서의 개인 경로와 개발 증적은 익명화·공개 가능한 식별자로 전환한다.
- 현재 [PAKDD2027 Author Kit](https://pakdd2027.org/pages/calls/authors-kit)는 생성형 AI 사용을 문법 교정으로 제한하고 공개를 요구한다. 이 AI-assisted 초안은 연구 검토용이며 그대로 투고하거나 단순 검토만으로 정책을 충족한다고 간주하지 않는다.
