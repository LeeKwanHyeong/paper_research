# 결과를 반영한 기여 범위와 원고 작업 순서

## Material Passport

source131 신규 구성요소36 Test와 이전 대표9·외부54, 보정27, Instacart9는 각 출처·선택 모델·split을 유지했다. 아래는 완료된 원본 지표에 근거한 해석이며 모델 우월성이나 수락 가능성의 판정은 아니다. n=3 평균·표본SD는 기술통계이고 이전 Test 접근 이력이 있으므로 새 독립 검증으로 부르지 않는다.

**신규 구성요소36의 결합 효과를 판단한다 — 완료**

| 데이터셋 | Validation FULL RMSE | Test FULL RMSE | Test 최저 평균 구조 | 그 구조의 Test RMSE | FULL의 Test 순위 |
|---|---:|---:|---|---:|---:|
| Taxi | 81.922727 | 158.706128 | CNN | 135.246858 | 3/4 |
| RAF | 34.182346 | 39.895919 | NONE | 39.399819 | 3/4 |
| Intermittent | 1.700536 | 2.914264 | CNN | 2.616125 | 3/4 |

- FULL은 세 데이터셋의 Validation 평균 RMSE가 최저였지만 Test에서는 어느 데이터셋에서도 평균 1위가 아니었다. 따라서 “CNN과 GRU를 함께 써야 두 단독 구조보다 Test 수량 성능이 항상 낫다”는 주장은 이 대조로 뒷받침되지 않는다.
- Taxi Test에서 FULL은 NONE보다 평균 RMSE가 16.869% 낮지만 CNN만보다 17.346%, GRU만보다 5.062% 높다. NONE 대비 개선은 2/3 seed이며 CNN/GRU 대비 각각 1/3 seed다.
- RAF Test에서 FULL은 NONE보다 1.259%, CNN만보다 0.746% 높다. GRU만 대비 평균 0.112% 낮지만 seed별 개선은 1/3이다. Validation의 작은 개선을 Test 개선으로 표현할 수 없다.
- Intermittent Test에서 FULL은 GRU만보다 평균 RMSE가 9.648% 낮고 3/3 seed 개선이다. NONE·CNN만 대비 각각 5.370%·11.396% 높으며 개선은 각각 1/3 seed다.
- 평균 순위는 같은 source131 캠페인 안에서만 해석한다. 성능이 좋은 CNN만 또는 과거 다른 FULL을 사후에 대표 모델로 바꿔 전체 결과를 다시 쓰지 않는다. 변경안을 검토한다면 Test를 본 뒤 선택했다는 이력을 유지하고 별도 모델·후속 연구로 기록해야 한다.

**기존 대표 모델·보정·Instacart의 근거를 구분한다 — 완료**

- 기존 대표 모델9(source117/123)의 외부 비교는 별도로 보존한다. Test의 CNN+GRU 전체 RMSE는 Taxi 130.400, Intermittent 2.824482, RAF 39.95632다. S2P2 common-head 대비 Intermittent는 개선되지만 Taxi·RAF는 개선되지 않았다. 신규 FULL과 교체하지 않는다.
- 보정27(source129)은 FULL이 NONE·current-only 대비 Taxi·Intermittent에서 평균 Test RMSE를 개선한다. 각 대조의 개선은 2/3 seed이며 RAF는 두 대조보다 악화하고 0/3 seed다. 원고의 기존 “모든 seed”라는 문장은 Validation에만 한정하고 Test까지 확대하지 않는다.
- Instacart9(source126)의 FULL은 두 MLP보다 평균 Test RMSE가 작지만 폭은 작다. MLP8 대비 3/3 seed, MLP12 대비 2/3 seed이며 seed52 예외를 보존한다. 원래 시간·큰 수량 불리 결과를 함께 보고한다.
- 외부6개는 공유 출력 head를 사용하는 encoder adaptation이다. 원래 논문 전체 모델/native marked-event likelihood의 일반적 우월성으로 확대하지 않는다. 외부 Validation은 기존 selected endpoint 재사용이라는 출처 차이도 명시한다.

**main.reviewed.tex의 세 기여를 완료 근거에 맞춘다 — 다음 작업**

1. 방법 제안은 관측된 history만 사용하는 local/recurrent residual 구조와 target 차단 규칙의 명확한 기술로 유지할 수 있다. 이 설계의 존재와 모든 데이터셋의 최적 성능은 서로 다른 주장이다.
2. history 보정 비교의 기여는 같은 선택 기준에서 우회·동일 활성 파라미터 current-only 대조를 수행하고, 효과가 데이터셋·seed·평가 split에 따라 달라짐을 분석하는 범위로 쓴다. “모든 seed 개선”은 해당 Validation 사실로만 남긴다. 신규 CNN/GRU 단독 대조는 활성 파라미터 수가 다르므로 용량 효과를 완전히 분리한 근거로 쓰지 않는다.
3. 외부 비교와 Instacart의 기여는 검증된 수량 성능·큰 수량·시간 tradeoff를 같은 checkpoint에서 보고하는 것으로 유지한다. 모든 데이터셋/지표/구조에 대한 보편 우월성은 주장하지 않는다.
4. Abstract·Introduction·Results·Discussion·Conclusion의 split과 캠페인별 FULL 출처를 일치시킨다. 현재 main.reviewed.tex는 Validation 중심 문구를 유지하고 있으므로 완료 Test를 추가할 때 위 반전·예외도 함께 반영해야 한다. 이번 승인은 결과 평가·표와 범위 정리까지이며 원고 파일은 변경하지 않았다.

**교수님 보고용 완성본 이후 PAKDD 제출본을 만든다 — 다음 작업**

- 교수님 보고용 main.reviewed.tex에 캠페인별 Test 결과와 제한사항을 먼저 반영하고 표·주장·원본 링크를 확인한다.
- 그 다음 별도 12페이지 PAKDD용 LaTeX에 핵심 방법·근거를 압축한다. 본문 분량과 부록·참고문헌·익명화 등은 실제 투고 규정으로 확인한다. 추가 실험은 현재 완료 평가의 요구가 아니며, 기여 문구를 고친 뒤 필요한 근거가 있을 때 별도 계약으로 결정한다.

근거: [구성요소36 여섯 지표·seed별 결과](COMPONENTS36_ANALYSIS.md), [기존 세 캠페인 별도 표](REUSED_COHORT_TABLES.md). 모든 수치 원본은 각 JSON의 SHA 결속과 평가 영수증에 보존한다.
