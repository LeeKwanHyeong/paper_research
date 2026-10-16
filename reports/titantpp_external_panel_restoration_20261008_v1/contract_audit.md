# 외부 비교 패널의 계약 재사용 검토 — 완료

이 검토는 기존 원고의 네 데이터 Validation 비교표를 현재 원고의 역사적 성능 패널로 복원하기 위한 로컬 계약·소스 감사다. 새 모델의 성능이나 학습 완료 여부를 평가하지 않았다. 수치 행·선택 epoch·집계 검증은 같은 디렉터리의 별도 패널 증적에서 담당한다.

## 판정

**복원할 수 있다.** 기존 패널은 동일한 수치형 입력과 공통 수량·시간 출력부로 적응한 여섯 외부 인코더와 과거 TitanTPP MLP4를 비교한 Validation 개발 결과다. 현재 CNN+GRU54의 결과처럼 이름을 바꾸거나, 현재 63 fit과 같은 새 실행·동일 초기화·동일 Runtime 대조로 합칠 수는 없다.

| 항목 | 확인 결과 | 본문에 필요한 구분 |
|---|---|---|
| 과거 TitanTPP | `titantpp_history_mlp`, r=4, 8개 분기, 명목 보정 파라미터 6,144, 고정 divisor 8 | `TitanTPP-MLP4 (historical)` 등으로 표시 |
| 후속 MLP8 구조 패널 | r=8, 8개 분기, 명목 보정 파라미터 12,288 | 과거 패널의 MLP4와 별도 |
| 현재 CNN+GRU54 | 보정 파라미터 25,308, 다른 국소·순환 보정 구조 | 자체 완료 결과가 있어야 별도 행으로 추가 |
| Instacart 폭16 | 20261008 실행 계약의 별도 역사참고 | 복원하는 20261001 원고 패널의 Titan 행은 폭16이 아니라 MLP4 |
| RAF 대표 행 | `titantpp_history_mlp` | 같은 배치의 `titantpp_history_mlp_active_norm`을 대표 행과 바꾸지 않음 |

분기 파라미터 수는 모듈 전체의 명목 수다. 실제 가용 분기와 동적 참여는 입력 이력 길이에 따라 달라진다.

## 데이터·학습 계약 대조

현재 Instacart/history/retrieval의 일곱 dataset 계약과 과거 benchmark seed42·Instacart 반복/잔여 배치·core MLP·추가 TPP·RAF 계약 사이의 **19쌍**을 비교했다. 아래 필드는 모두 일치한다.

- Train 및 Validation target count, target identity SHA256, target quantity SHA256
- batch, lookback, sequence cap, shuffle, padding/target 포함 창 정의에 연결된 loader 계약
- quantity variant, log-quantity 손실 가중치, heteroscedastic lognormal duration head, time scale, sigma floor, recorded time 단위·round/clamp·top-code
- AdamW, learning rate 0.001, weight decay 0.01, gradient clip 1
- Train 수량 및 시간 통계, 시간 target SHA, 수량 구간 경계
- 최소 40/최대 300 epoch, patience 40, batch 128, 최초 strict finite 최소 Validation raw 수량 RMSE 선택, fresh fit 정책

| 데이터 | Train target | Validation target | 최대 행 수 | Lookback/단위 | 큰 수량 경계 |
|---|---:|---:|---:|---|---:|
| Taxi | 38,393 | 8,268 | 256 | 168 hour | >3,449 |
| Intermittent | 393,824 | 86,285 | 256 | 520 week | >187 |
| Instacart | 1,991,192 | 503,733 | 64 | 52 day | >35 |
| RAF | 25,779 | 6,690 | 84 | 84 month | >200 |

최대 행 수에는 withheld target이 포함된다. Instacart 시간 top-code는 30이고 나머지는 없다. parquet 원시 행이나 예측을 새로 읽지 않았다. 이 판정은 동결 population metadata와 소스·계약 일치에 근거하며 새로운 데이터 재구성은 아니다.

## 외부 모델의 의미

RMTPP, NHP, THP, SAHP, S2P2, AttNHP는 각각의 원 논문이 보고한 native categorical-mark/intensity 성능을 그대로 인용한 행이 아니다. 동일한 관측 `log1p(gap), log1p(quantity)` 입력, 수량 회귀 및 recorded-duration mass 출력, 공통 손실과 선택 규칙으로 적응한 비교다.

S2P2/AttNHP는 `observed_event_right_limit_common_heads` 계약을 사용하며 다음 target time을 입력하지 않는다. S2P2는 64차원 2층, state size 16, complex dynamics, backward ZOH, relative time, affine-doubling recurrence를 사용한다. AttNHP는 64차원 2층, attention head 1, time dimension 16이고 관측 사건 직후 상태를 질의한다. 각각 native mark-intensity head는 비활성화되어 있다. 따라서 표 캡션은 common-head adaptations라고 명시해야 한다.

## 소스 및 Runtime 근거

- 과거 core 소스 97개, additional 소스 106개, RAF 원본 소스 104개와 현재 Instacart 126개·history 129개·retrieval 129개를 각 동결 계약의 raw SHA와 대조했고 모두 일치했다.
- 원고 통합 source manifest에 묶인 8개 Methods 소스의 raw SHA도 모두 일치했다.
- 과거 core와 현재 history 사이의 loader, target loss 경로, 학습 경로, integer-time likelihood, MLP4 모듈, Titan backbone, prototype memory 소스는 byte-identical이다.
- `CountAwareTPP.py`의 유일한 과거 core→현재 변경은 지원하는 시간 단위 집합에 `month`를 추가한 한 줄이다. RAF에 사용된 month 지원 파일은 현재 파일과 byte-identical이다.
- 과거 core·additional·RAF는 base revision `1de2c31e8e3febda7ff20d0527402ea9959b4098`와 각 content closure에 연결된다. 이 base commit만으로 후속 additive source 전체를 대표하지 않는다.
- 현재 Instacart는 science revision `4274043717cf24640d9a18fb35231a1f7ba50c8e`, 후속 history/retrieval는 `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`다. 현재 MLP8 패널은 별도로 `a699448f2991465b4fc17c75fafcc1b85845d1d6`에 연결된다.
- 과거 5090 계약은 기존 `/opt/miniconda3/envs/ai_env`와 NumPy 2.1.3, 현재 5090은 전용 resume Runtime과 NumPy 2.4.4를 가리킨다. PyTorch 2.11.0+cu130 등 일부 버전과 GPU UUID가 같더라도 전체 Runtime 동일성으로 보고하지 않는다. 5080/5090의 dataset·seed 배치도 캠페인마다 달라진다.

Runtime은 보존된 계약의 정보이며 이번에 서버를 조회해 실제 환경을 재검증하지 않았다. seed 42/52/62가 같다는 이유로 서로 다른 모델·캠페인의 초기 파라미터나 최적화 경로가 같다고 볼 수 없다. 각 행은 그 실행에서 선택된 checkpoint를 유지한다. 따라서 세대 간 시간 차이를 구조 효율 효과로 합치거나 paired seed 검증을 새로 주장하지 않는다.

## 본문 적용 범위

기존 패널을 “Historical common-head Validation comparison”으로 먼저 제시하고, 정확한 MLP4 버전을 표시한다. 현재 대표 모델의 이력 보정 검증 및 검색 검증은 그다음에 별도로 제시한다. 패널은 외부 방법 대비 성능의 위치를 설명하지만 현재 CNN+GRU54 보정 경로의 효과를 대신 입증하지 않는다. 동일 frozen Validation을 개발에 사용했다는 사실도 유지한다.

이번 작업은 원격 호출·GPU 학습·재추론·새 평가 0회이며 Test 성능과 예측 행을 읽지 않았다. 이전 데이터 감사 문서에서 split identity metadata와 소스 동등성 설명을 참고했으나 그 문서의 과거 초기화/성과를 새 비교 근거로 옮기지 않았다. 원고 및 다른 작업자의 파일은 수정하지 않았다.

다음 작업은 별도 담당자가 원본 Validation endpoint의 선택 epoch·수치 집계·명칭을 확정한 뒤 본문 5.1에 통합하는 것이다. 상세 field 값·raw SHA·검증 결과는 `contract_audit.json`에 보존했다.
