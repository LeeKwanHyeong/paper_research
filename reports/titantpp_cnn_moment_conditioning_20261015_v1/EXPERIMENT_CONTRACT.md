# 관측 μ·σ를 이용한 CNN 이력 처리 — 준비 계약

Material Passport: academic-research-suite / experiment-agent / local implementation preparation / 2026-10-15 v1.

## 목적과 승인 범위

관측된 수량의 수준·변동폭에 따라 CNN의 과거 관측 조합을 바꾸면, 단순 정규화나 파라미터 증가를 넘어 이력이 더 유용하게 표현되는지 검증할 준비다. 효과는 미확인이다. 기존 Test를 본 뒤 나온 탐색적 가설이며, Test 일반화 개선이나 원인을 확정한 결과가 아니다.

사용자는 실험 계약·구현·로컬 검증까지만 승인했다. 새 학습은 준비 보고 후 결정한다. 이번 작업에서 원격 연결, 설치, native GPU 실행, 실제 데이터 추론, 새 학습, Test 평가, checkpoint 재선택은 하지 않는다. 기존 결과·실패·claim·source·원고와 진행 중인 자동화를 변경하지 않는다.

## 비교 구조

| 구조 | 내부 처리 | 기존 CNN 대비 추가 학습 파라미터 |
|---|---|---:|
| 기존 CNN | 원래 Q/K/V의 현재·이전 두 관측에 고정 학습 커널 적용 | 0 |
| 블록 RevIN CNN | 각 query의 관측 prefix로 Q/K/V를 정규화하고 CNN 뒤 원래 잠재 단위로 복원 | 0 |
| μ·σ 조건 CNN | 같은 정규화 + 관측 log 수량의 μ·σ로 세 tap 비중 조절 | 195 |
| μ·σ 용량 대조군 | 같은 통계·같은 MLP로 정규화 특징에 보정값을 더함; tap 조절 없음 | 195 |

이는 **affine=False인 causal latent block RevIN**이다. 표준 모델 입출력 RevIN을 그대로 적용한 실험으로 부르지 않는다. 시간·수량 입력과 출력 head의 의미를 유지하려고 적용 위치를 첫 attention block의 event Q/K/V CNN 내부로 한정했다. 두 번째 block, attention 참조 범위, persistent memory 및 prototype 검색·평균 더하기는 그대로다. GRU는 기존 CNN처럼 동결·우회한다.

## 서로 다른 두 종류의 μ·σ

1. **정규화 통계:** 각 Q/K/V 채널의 관측 segment prefix 평균 `m[t,c]`와 `s[t,c]=sqrt(var+1e-5)`. 현재 query의 같은 통계를 현재·lag1·lag2 모두에 적용한다.
2. **조절 통계:** 원래 관측 수량 `log1p(q)`의 segment prefix 평균 `mu_q[t]`와 `sigma_q[t]=sqrt(var+1e-5)`. MLP 입력은 `[mu_q, log1p(sigma_q)]` 두 값이다. 원수량 공간의 통계가 아니며 로그 압축을 되돌리는 실험도 아니다.

통계는 sample별·현재 시점까지의 관측만 사용하고 ddof=0이다. padding은 건너뛰며 valid-but-withheld 행에서 segment 통계와 CNN tap 이력을 초기화한다. parent attention의 과거 참조 범위는 바꾸지 않는다. 정답 수량·시간은 통계에 포함하지 않는다. float64 누적합으로 계산한 통계를 feature dtype으로 변환하고 gradient를 끊는다. 같은 forward 안에서 명시적으로 전달하며 다음 batch를 위한 mutable cache를 두지 않는다. 상수·단일 관측 이력은 eps로 처리한다.

## 연산과 중요한 상쇄 조건

Q/K/V 중 하나를 `v`, 현재 query의 정규화된 lag를 `z[t,l]=(v[p_l(t)]−m[t])/s[t]`라 한다. 없는 lag는 0으로 가린다.

- 블록 RevIN: `v_out[t]=v[t]+s[t]*sum_l(w[l]*z[t,l])`.
- μ·σ 조건: `v_out[t]=v[t]+s[t]*sum_l(w[l]*a[t,l]*z[t,l])`.
- 용량 대조군: `v_out[t]=v[t]+s[t]*(sum_l(w[l]*z[t,l])+0.5*tanh(MLP(stats[t])))`. 단, 두 관측 이상일 때만 MLP 보정을 더한다.

첫 두 식에서 **잠재 특징의 s[t]는 대수적으로 상쇄**된다. 블록 RevIN은 `v[t]+sum_l(w[l]*(v[p_l(t)]−m[t]))`라는 평균 중심화 필터와 같다. 이것을 표준편차 정규화의 독립 효과라고 해석하지 않는다. 조건 CNN에서는 관측 **수량** sigma_q가 MLP와 tap 계수를 통해 별도로 영향을 준다. 용량 대조군에는 동일한 통계·파라미터·CNN 수용범위를 제공하지만, 복원 scale이 가산 보정 크기에 영향을 주므로 함수군과 최적화까지 같지는 않다.

Q/K/V별 `2→16→1`, tanh MLP를 하나씩 두고 64채널에 공유한다. 각각 65개, 총195개 파라미터다. `u=tanh(MLP)`로 현재 tap 비중은 `1+0.5u`, 사용 가능한 과거 tap 각각은 `1−0.5u/m`으로 한다. m은 과거 tap 개수다. 비중은 [0.5,1.5], 가용 tap 평균은1이며, 한 관측일 때 조절하지 않는다. 이는 전체 CNN 잔차 norm 상한이 아니다.

MLP 마지막 층은 0이고, 생성 시 기존 RNG를 소비하지 않는다. 모든 공통 초기 tensor를 보존한다. 기존 CNN 커널도 0으로 시작하므로 네 구조의 초기 출력은 동일하다. 비영 커널에서는 조건·용량 MLP가 0일 때 블록 RevIN과 동일하다. 초기 0층 때문에 일부 gradient가 0인 현상을 숨기지 않으며, 합성 비영 가중치에서도 연결성을 별도 검증한다.

## 고정 과학 조건과 데이터 결속

기존 log1p 수량·시간 입력, hidden64, 공통 수량 head와 log 수량 MSE, recorded positive-integer 시간 head, 손실 계수, loader, optimizer, 초기화와 선택 기준을 유지한다. head는 상속된 기존 함수를 그대로 호출하며 미래 평균·분산을 출력하는 head를 새로 만들지 않는다.

준비 bundle은 기존 Taxi 계약의 입력/대상/전처리 지문과 CNN seed42·52·62의 selected checkpoint 식별자를 재사용한다. 기존 binary와 실제 데이터의 native SHA 재검증은 아직 하지 않았다. 구현은 데이터셋에 종속되지 않지만 RAF·Intermittent의 실행 계약과 데이터 결속을 완료했다고 표시하지 않는다.

후속 **Taxi 첫 비교를 선택한다면** 새3구조×seed42·52·62=9 fit, 기존 CNN3 재사용이 된다. 이것은 제안된 실행 규모이며 현재 학습 승인이 아니다. 기존 조건은 batch128, AdamW lr0.001/weight_decay0.01/clip1, epoch당300step, 최대300epoch, 최소40/patience40, fresh fit이다. 선택은 전체 fullValidation raw 수량 RMSE 최초 strict finite 최소 epoch다. 동일 checkpoint의 MAE 및 나머지 지표도 보존한다. 기존 조건36h/host168h 기한을 임의로 연장하지 않는다.

실행을 결정할 때 같은 비교의 seed들은 같은 GPU·Runtime에서 수행한다. Taxi 기준선과 맞추는 5080을 우선안으로 두고 서버당 최대3 병렬을 유지한다. 현재 서버 상태·성능·가용성을 확인했다고 주장하지 않는다. 5090이나 다른 데이터셋으로 확장하면 별도 입력·환경 결속과 비교 조건을 먼저 기록한다.

## 증적과 구현 완료 기준

기존 동결 source134를 검증·재사용하며 기존 작업 폴더와 차이가 있는 평가 도구는 보존한다. 새 모델과 준비 adapter만 추가해 별도 source136을 만든다. 준비 adapter에는 candidate 학습 CLI나 전역 등록이 없고 `run_fit`은 거부한다.

합성 CPU 검증: prefix 통계 수계산, 정규화·역변환과 중심화 동치, scale 상쇄, μ/σ 각각의 실제 영향, tap bounds, 공통 초기 tensor/RNG/head·loss 동일성, 각 신규 파라미터 gradient, 미래/target/padding 차단, batch 불변, 평가 중 상태 불변, checkpoint identity·selector 혼동 거부, source SHA 보존, 기존 준비 경로 덮어쓰기 거부.

로컬 합성 통과는 학습 성능이나 native GPU 통합 통과가 아니다. 일반화·우월성 결론은 아직 없다.

## 관련 방법과 차이

- RevIN 공식 구현: https://github.com/ts-kim/RevIN/blob/master/RevIN.py — 입력 통계로 정규화·복원하는 원리. 이번 적용은 causal prefix/latent block/affine=False로 범위를 제한했다.
- Non-stationary Transformers: https://arxiv.org/abs/2205.14415 — 정규화로 약해지는 정보를 내부 연산에 다시 반영하는 관련 선행 연구. 관측 통계 활용만으로 신규성을 주장하지 않는다.
