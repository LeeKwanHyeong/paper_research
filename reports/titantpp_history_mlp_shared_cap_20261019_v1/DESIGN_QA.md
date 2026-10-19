# Shared History MLP 설계 검토

작성일: 2026-10-19

- 상태: **설계 검토 완료 — 아래 조건으로 SCIENCE_SPEC 확정 및 로컬 구현 진행 가능**
- 검토 역할: Architecture Specialist / architecture-reviewer. 프로젝트 역할 설정은 `.codex/agents/architecture-reviewer.toml`의 `gpt-6-astra`, `high`로 확인했다. 실행 중인 모델의 정확한 식별자는 별도 런타임 조회로 확인하지 않았다.
- 방법: 해당 모듈의 코드와 기존 가설을 읽고 두 대안을 비교하는 간단한 설계 검토. UI 설계·PM 계획·인프라 변경은 범위 밖이다.
- 범위: 계약·구현 전 검토. 이 검토에서는 코드 수정, 학습, SSH, 배포, 실제 데이터 평가, checkpoint loading, held-out Test 열람을 수행하지 않았다.

## 문제와 판단

기존 History MLP의 입력 변환을 공유할 때, 매개변수 감소만으로 설명되는 이득과 공유 구조를 적용한 모델의 이득을 비교할 수 있는 대조군이 필요하다. 같은 매개변수 수가 같은 함수 공간이나 같은 학습 난도를 뜻하지 않는다는 해석 한계도 계약에 남겨야 한다.

**shared width12와 independent width5를 주 비교군으로 채택한다.** 두 군은 보정 모듈의 학습 가능한 가중치 수가 정확히 7,680개다. 이 비교는 동일 매개변수 예산에서 두 설계의 성능을 비교하는 데 타당하다. 표현 폭·공유 제약·최적화가 함께 달라지므로, 결과를 공유만의 순수 인과효과나 동일 표현 용량의 대조라고 주장할 수 없다.

기존 independent width16의 24,576개 보정 가중치는 참고 기준으로만 쓴다. 기존 결과를 가져올 경우 source revision, split, 입력, loss, alpha, seed, selector와 학습 조건 일치를 먼저 확인해야 한다. 이번 검토에서는 기존 성능 자료를 새로 읽지 않았다.

## 확인한 코드 계약

근거는 다음 현재 소스와 문서다.

- `models/TPPs/CountAwareTitanHistoryWidth.py`: 독립적인 bias-free 입력·출력 투영, exact GELU, 출력 투영 0 초기화, CPU RNG fork, /8.
- `models/TPPs/CountAwareTitanCoreAblation.py`: 기존 encoder와 static/persistent memory를 유지하는 MLP 교체 지점.
- `models/TPPs/CountAwareTitanMultiLagDetail.py`: `lag_source_indices(mode="local")`에서 모든 활성 분기는 같은 직전 유효 관측을 읽는다. 문턱은 1, 2, 4, 8, 16, 32, 64, 128이다. padding은 관측 수에 포함하지 않고 withheld valid 행은 이후 이력 가용성을 재설정한다.
- `simple_lab_test/search/tests/test_titantpp_history_width.py`: 기존 수식 대조·mask·causality·공통 초기화 검증의 참고 구현. 해당 테스트를 이번 검토에서 실행하지 않았다.
- `reports/titantpp_mlp_train_validation_mechanism_20261017_v1/HYPOTHESIS.md`: 공유 가설, 기존 근거의 한계, 사전 반증 조건.

현재 폭 구현은 width5를 허용하지 않는다. 따라서 새로운 두 군의 계약과 식별자를 별도로 만들고, 기존 registry나 reference arm의 의미를 바꾸지 않는 방식이 적절하다.

## 매개변수 계산과 대안

입력 차원 128, 출력 차원 64, 분기 8개, 모든 bias 없음으로 고정한다.

```text
P_shared(s)      = 128s + 8·64s = 640s
P_independent(r) = 8·(128r + 64r) = 1536r
P_shared = P_independent  ⇔ 5s = 12r
양의 정수 해: s = 12k, r = 5k
k=1: s=12, r=5, P=7,680
```

공유 폭을 기존 관심 폭16 이하로 두는 설계 제약에서는 12/5가 유일한 양의 정수 해다. 이는 성능을 보고 선택한 최적 폭이라는 뜻이 아니다. 24/10도 정확히 맞지만 폭16 상한을 벗어난다.

| 대안 | 판단 가능한 질문 | 구현·운영 비용 | 팀 복잡도·향후 변경 비용 | 판단 |
|---|---|---|---|---|
| 공유12 대 독립5 | 같은 가중치 수에서 두 구조가 어떻게 다른가 | 작은 별도 보정 모듈과 계약 필요. 실제 속도·메모리는 측정 전 미확인 | 균일 폭이라 계산과 검증이 단순하며 계약을 고정하기 쉬움 | 주 비교로 채택 |
| 공유16 대 독립16 | 동일 폭에서 공유로 바꾼 전체 효과가 무엇인가 | 공유군 10,240개, 독립군 24,576개. 구현은 단순하나 크기 차이가 큼 | 설명은 쉽지만 크기 감소를 분리하려면 추가 군 필요 | 이번 동일 매개변수 대조를 대체하지 못함 |
| 공유16 대 독립 혼합 폭6/7 | 가까운 가중치 예산에서 비교 가능한가 | 독립 폭 합53이면 10,176개, 합54이면 10,368개로 불일치 | 어느 문턱에 폭7을 배치할지 추가 설계 선택과 검증 필요 | 불필요한 혼합 요인을 추가하므로 배제 |

동일 매개변수 수는 동일 실행 비용을 보장하지 않는다. 모든 분기를 계산하는 단순 수식에서 두 군의 선형 연산량은 각각 7,680 MAC 상당이지만 GELU 차원과 tensor 배치·중간 메모리·kernel 호출이 다르다. 속도나 메모리 개선은 구현 후 별도 측정 없이 주장하지 않는다.

## 해석에 남는 교란 요인

활성 분기 수를 k, 고정된 availability mask 아래 pair 입력을 x라 두면 다음과 같다.

```text
공유: (Σ_active B_j) GELU(W_shared x) / 8
독립: Σ_active B_j GELU(W_j x) / 8
```

이 보정항의 x에 대한 Jacobian rank 상한은 공유에서 12, 독립에서 min(64, 5k)다. 이는 실제 rank를 측정한 결과도, encoder를 포함한 전체 모델의 rank 상한도 아니다. k=1/2에서는 공유군의 단일 관계 표현 폭이 더 크고, k≥3에서는 독립군의 합성 숨은 폭 상한이 더 크다. 두 함수 공간을 단순한 포함 관계로 가정하지 않는다. 이력 길이에 따라 성능 차이가 달라질 가능성을 사전에 기록한다.

공유 W는 서로 다른 문턱의 학습 신호를 함께 받는다. 초기 출력 B가 모두 0이므로 첫 backward에서 W의 gradient가 0인 것은 정상이다. 이후에는 공유군의 gradient 합산, 독립군의 분기별 노출 빈도, optimizer와 정규화의 영향이 다를 수 있다. 같은 seed·초기 출력·공통 model state는 이 차이를 없애지 않는다.

분기별 B_j는 코드상 모두 사용될 수 있어야 한다. 다만 실제 학습 샘플이 129개 이상의 유효 연속 관측을 포함하지 않으면 마지막 분기는 활성화되지 않는다. 따라서 “모든 매개변수가 사용된다”는 주장은 synthetic 경로 검증과 실제 데이터의 문턱별 노출을 구분해야 한다. 이 검토는 실제 노출 수를 확인하지 않았다.

## SCIENCE_SPEC 확정 조건

1. 주 비교군은 공유12와 독립5이며, correction parameter count와 full-model trainable parameter count 모두 정확히 일치한다. dummy parameter, 사용하지 않는 학습 모듈, 다른 부위의 폭 보정은 금지한다.
2. 두 군은 concat(current64, previous64), immediate predecessor, 원래 8개 문턱, exact GELU, bias-free, /8, encoder 삽입 위치, head와 memory를 유지한다. 공유는 입력 W에만 적용하고 출력 B_j는 8개를 별도로 둔다.
3. 공통 log1p 입력·logMSE·time alpha0·Cap3 데이터셋 계약·seed·selector·학습 조건은 바꾸지 않는다. 현재 실행 중인 5090/5080 작업과 기존 reference 파일을 보존한다.
4. RNG 보존, 공통 가중치 동일성, 출력 0 초기화를 유지한다. 두 구조의 서로 다른 shape를 억지로 동일 초기 가중치라고 표현하지 않는다.
5. 각 arm의 구조, 폭, 공유 방식, 수식, parameter count를 metadata와 identity에 포함한다. 잘못된 relabel과 다른 arm의 state를 거부한다. 실제 checkpoint를 읽지 않고 synthetic state로 검증 가능하다.
6. 가설 문서의 기존 반증 조건을 보존하되, 공유12 대 독립5의 차이가 주 비교라는 점을 명시한다. width16은 참고군이며 순수 공유효과의 근거로 쓰지 않는다. 이번 승인에 새 학습과 실제 Test 평가는 포함되지 않는다.

## 구현 후 필요한 로컬 검증

- 두 보정 모듈 7,680개 및 전체 모델 학습 매개변수 동일성을 실제 등록된 parameters에서 계산한다. 출력 B 8개와 공유 W 1개 또는 독립 W 8개가 예상 경로에 등록됐는지 확인한다.
- 독립적인 단순 수식 구현과 forward·gradient를 비교한다. 문턱 직전·직후, 길이0/1, padding, withheld reset, future perturbation, 비유효 입력 NaN 격리를 포함한다.
- 유효 연속 관측 129개 이상 synthetic 입력으로 8개 분기를 모두 활성화한다. 0 초기화 검증 뒤 synthetic nonzero B를 넣어 각 parameter tensor의 유한한 gradient와 경로 연결을 확인한다. 첫 backward부터 W의 nonzero gradient를 요구하지 않는다.
- 같은 seed에서 구성 전후 CPU RNG state와 공통 model state가 동일한지 확인한다. eval에서 초기 보정 출력 0 및 전체 모델 출력의 일치를 확인한다.
- target withholding, input transform, logMSE, time alpha0, memory/head metadata, arm identity·shape 검증을 확인한다. 실제 checkpoint나 held-out 데이터를 쓰지 않는다.
- frozen contract와 실행 진입점이 불허 데이터셋·arm·time alpha·Test 요청을 거부하는지 확인한다. 기존 reference 경로에 대한 회귀 검증도 범위에 포함한다.

## 산출물과 남은 순서

생성 산출물은 이 `DESIGN_QA.md` 하나다. 프로젝트에서 지정한 `.agents/skills/_shared/runtime/execution-protocols/codex.md`는 존재하지 않아 읽지 못했다. `.agents/` 수정 금지와 이번 지정 산출물 경로를 따라 결과를 `reports/`에 기록했다.

**두 실험군의 과학적 비교 범위를 확정한다 — 다음 작업**
- 대상: 현재 paper_research 세션의 SCIENCE_SPEC. 위 계산·해석 한계·보존 계약을 고정한 뒤 구현자가 동일 계약을 사용한다.

**고정한 계약을 구현하고 synthetic 입력으로 검증한다 — 다음 작업**
- 대상: 공유/독립 보정 모듈과 로컬 검증. 공통 계약 확정 이후에 진행한다. 이 설계 검토 자체는 구현 완료를 뜻하지 않는다.

**새 학습·SSH·배포·실제 Test 평가를 시작한다 — 승인 필요**
- 이번 범위 밖이다. 현재 실행 중인 5090/5080 학습의 상태 조회나 변경도 수행하지 않았다.
