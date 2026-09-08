# Q/K 이력 혼합의 크기를 제한하는 Hard-LMM — 설계 계약 v1

상태: **로컬 후보 선정 완료, 모델 구현·실데이터 후보 평가 전**.
2026-09-08, `paper_research / codex/hard-lmm-causal-qkv`.
설계 시작 기준은 `f0cc1fd`, 기존 Causal-QKV 구현 기준은
`84668e207d5121f211a6a93af12ca2f96068a25e`다.
식별자: `hard_lmm_bounded_qk_causal_v_v1`.

## 해결할 문제와 단일 가설

기존 Causal-QKV FULL은 Taxi의 수량 오차를 줄였지만 legacy 시간 점수의
train–validation 격차가 컸다. Intermittent는 validation body를 개선하면서
상위 5% 수량의 제곱오차를 늘렸고, 저장된 train 표본에서는 그 방향이 달랐다.
Instacart의 공통 개선도 확인되지 않았다. 따라서 필요한 것은 표현 용량을 더
늘리는 것보다 **이미 배운 국소 이력의 효과가 평가 구간에서도 유지되는지** 검증하는 것이다.

선택한 가설은 다음 하나다.

> 추가 causal-V 경로를 유지하고 Q/K의 추가 이력 혼합을 각 attention head의
> 원 projection 크기 안으로 제한하면, FULL의 수량 이득을 보존하면서 일반화
> 실패를 줄일 수 있는가?

V 제거 개입의 실패는 V 경로를 유지할 근거다. 그러나 Q/K residual 크기나
attention logit의 validation 이상은 아직 측정하지 않았다. 이 설계는
**원인이 입증된 수리안이 아니라, 변경 범위를 제한한 검증 후보**다.

여기서 Q/K/V는 첫 encoder block의 사건 attention projection이다.
마지막 static Hard-LMM bank의 key/value를 새로 분리하거나 바꾸는 설계가 아니다.

## 하나의 고정 구조

첫 block에서 기존 projection을 `p`, event mask를 적용한 convolution 입력을 `p_masked`,
기존 kernel3 causal residual을 `r=C_p(p_masked)`라 한다.
현재 사건·직전 사건·두 번째 이전 사건을 사용한다. Q와 K 각각을
`[batch, events, 4 heads, 16 features]`로 보고 마지막 16개 feature만 평균한다.

\[
s=\sqrt{\operatorname{mean}_{16}(p^2)+10^{-8}},\quad
R^2=\operatorname{mean}_{16}(r^2),\quad
g=\frac{s}{\sqrt{s^2+R^2}},\quad p'=p+g r.
\]

- Q/K에 같은 식을 적용한다. 제한 비율 `rho=1`, epsilon `1e-8`을 고정한다.
- `s`를 detach하지 않는다. 학습 가능한 gate나 추가 통계 buffer는 없다.
- FP16/BF16 입력이면 norm과 gain은 FP32로 계산한다. FP32/FP64는 해당 dtype을
  유지한다. gain을 먼저 구해 residual에 곱하고 **기존 FULL의 `p+r` 연산 dtype**
  (`torch.result_type(p,r)`)으로 돌린 뒤 더한다. `p.dtype`으로 무조건 낮추지 않는다.
  현재 학습 비교는 기존 FP32 경로를 기준으로 하며 AMP는 CUDA에서 dtype·초기 값·
  gradient 동일성을 별도 확인하기 전에는 사용하지 않는다. 임의로 큰 finite tensor까지
  overflow가 없다고 주장하지 않으며, 운영 입력 범위의 finite 검증은 구현 단계에서 수행한다.
- V는 기존 FULL과 **같은 연산 `v'=v+C_v(v)`**을 사용한다.
- hidden64 기준 kernel 파라미터는 Q/K/V 각각 `3×64`, 총 576개다.
  B보다 576개 많고 FULL과 파라미터 수가 같다. 세 kernel 모두 정확히 0으로 초기화한다.
- RNG를 추가 소비하지 않는다. Q/K/V 선형 projection, 두 번째 encoder block,
  persistent tokens, static 64-vector top-4 arithmetic-mean memory, 결합 방식,
  시간·수량 head는 기존 경로를 유지한다. 기존 파라미터는 계속 학습한다.
- causal convolution 입력 `p_masked`는 event mask를 적용하여 padding 위치의
  projection bias까지 제거한다. 원 Q/K/V projection과 최종 output masking은
  기존 FULL 경로를 유지한다. RMS 기준 `s`는 원 projection `p`에서 계산하며,
  유효 사건에서는 `p=p_masked`다. 왼쪽 이력 밖은 0, 미래 사건은 사용하지 않는다.
  persistent K/V는 사건 혼합 이후에 추가하므로 이 제한을 받지 않는다.
- dataset 이름·특정 duration·target 수량 구간에 따른 분기나 수동 계수는 없다.

64개 feature 전체에 하나의 gain을 쓰면 큰 head가 작은 head의 제한까지 결정한다.
실제 attention dot product 단위에 맞춰 head별 계산으로 고정한다.

## 보존의 의미와 한계

`RMS(gr)=R*s/sqrt(s²+R²) < s`다. 이는 epsilon을 포함한 원 projection RMS에
대한 **추가 residual의 크기 제한**이다. Q/K 전체 크기, 방향 변화, attention logit,
top-4 선택 또는 전체 모델의 안정성을 보장하는 정리는 아니다.
원 projection이 0이어도 epsilon에 의한 작은 허용량이 있다.

kernel이 0이면 출력은 B 및 zero-init FULL과 같고, residual에 대한 미분은 1이다.
따라서 같은 seed·공통 초기 상태·dropout RNG에서 다음을 확인해야 한다.

1. 초기 전체 출력과 공통 파라미터 gradient가 B 및 FULL과 일치한다.
2. 새 Q/K/V kernel gradient가 zero-init FULL과 일치하고, 과거 사건 계수도
   finite한 nonzero gradient를 받는다.
3. 같은 projection과 V kernel을 넣은 V 연산은 FULL과 bitwise하게 같다.
4. 학습 뒤에는 attention weight와 downstream hidden이 달라질 수 있다.
   **V 연산 유지가 학습 후 quantity prediction의 보존을 뜻하지는 않는다.**

기존 학습된 FULL checkpoint에 제한을 켜는 것은 초기 identity가 아니다.
이 후보는 B와 동일한 초기화에서 새로 학습하는 대조 실험으로 평가한다.
학습된 FULL의 V tensor를 고정하거나 다른 모델의 V를 이식하지 않는다.

## 구현 후 로컬에서 반드시 검증할 계약

- zero-init 출력·공통 gradient·kernel gradient, RNG와 파라미터 수.
- Q/K 각 head의 residual bound, 실제 학습 후 gain과 검색·예측 변화.
- prefix causality, 미래 target 시간·수량 변경, padding 값 변경, H1/H2,
  batch 간 독립성, persistent token 불변 경로.
- 운영 범위의 극단 입력에서 finite loss/gradient. 단순 norm식의 synthetic 검증을
  전체 모델이나 CUDA 검증으로 대체하지 않는다.
- 전용 모델 route, 기존 checkpoint strict 복원, 새 checkpoint·optimizer·selector
  roundtrip, resume 후 동일한 다음 step. B/FULL checkpoint를 새 모델로 오인하지 않는다.
- 비용은 batch128/hidden64/L8,64,256에서 기존 FULL과 B 모두에 대해 측정한다.
  B 대비 median training-step ratio ≤1.5, peak allocated memory ratio ≤1.25를
  기존 계약처럼 적용한다. 5 warmup·15 measured steps·3 repeats, 순서를 교대한다.
  추가 norm 연산은 O(Ld)이지만 **전체 attention은 여전히 O(L²)**다.

이번 완료 범위는 독립 수식의 FP32/FP64 zero-init 출력·미분·bound·finite 확인까지다.
위 전체 모델 계약, CUDA, 실제 성능, 속도·메모리는 아직 검증하지 않았다.

## 학습·평가에서 바꾸지 않을 것

Backbone 학습은 기존 Causal-QKV의 train/validation split, mask, log1p quantity MSE,
lambda_qty1, tail0, legacy time objective, AdamW lr0.001, batch128, clip1을 유지한다.
선택·조기 종료는 validation raw quantity RMSE의 가장 이른 strict minimum,
seed42 최대300/min40/patience40이다. 정상화 head를 Backbone과 함께 학습하지 않는다.

비교 행은 **B / 기존 FULL / 새 후보**다. 기존 B와 FULL의 quantity 결과는 source·data·
target identity·selector가 일치할 때 재사용한다. FULL과 후보의 파라미터 수가 같아서
이번 비교는 추가 파라미터 수 자체의 효과와 구분할 수 있다. 기존 validation을 반복
참고해 후보를 골랐으므로 다음 seed42 결과도 탐색적 screening이다.

정상화 시간 비교는 각 선택 Backbone을 고정하고 동일한 K=1 head만 학습하는 별도
평가다. 공통 계약은 `aligned_frozen_lognormal_duration_v1.json`을 재사용한다.
130개 head 파라미터, train-only initializer, epoch0 포함 max100/min20/patience20,
validation proper observation NLL의 earliest strict finite minimum을 유지한다.
수량 SHA는 **각 Backbone 내부에서 head fit 전후 동일**해야 한다.
새 Backbone과 B의 수량 출력이 같아야 한다는 뜻은 아니다.

- Intermittent: 기존 관측 계약의 원 단위 continuous log-normal density.
- Taxi: 양의 정수 PMF, `P(1)=F(1.5)`, `P(d)=F(d+0.5)-F(d-0.5)`.
- Instacart: 같은 PMF, top-code30은 기존 censoring 계약에 따라 `S(29.5)`.
- float64, probability clamp/loss cap 없음. 데이터셋 간 NLL 평균을 만들지 않는다.
- median 오차는 기존 observation/censoring 정의를 그대로 쓰고 함께 보고한다.
  NLL 개선이 median MAE/RMSE 개선을 보장하지 않는다.

기존 A/B runner는 model role과 source route가 고정되어 있다. 내부 likelihood·cache·
fit 함수를 재사용하되 B/FULL/새 후보의 source를 검증하는 **새 평가 route와 실행 계약**이
필요하다. 기존 계약을 수정해 이미 끝난 결과의 의미를 바꾸지 않는다.

### 후속 screening에서 적용할 공통 기준

아래는 다음 실행 전에 고정하는 연구 기준이며, 과거 FULL의 실패 기록을 바꾸지 않는다.
실행용 manifest에는 source commit·runtime·artifact 위치를 추가로 고정해야 한다.

- 세 데이터셋 모두 B 대비 raw RMSE가 strictly 감소해야 한다.
- B 대비 전체 MAE 악화 ≤1%, body(≤train p95) 및 >train p99 MAE 악화 각각 ≤2%.
- FULL의 기존 이득 보존: 세 데이터셋 모두 FULL 대비 raw RMSE·전체 MAE 악화 ≤1%,
  body·>p99 MAE 악화 각각 ≤2%. 이는 특히 Taxi의 기존 수량 개선을 보존하는 조건이다.
- 같은 K=1 평가에서 candidate proper NLL은 aligned-B+0.01 이내여야 한다.
  Legacy time score는 별도 보고하며 이 proper NLL과 수치 비교하지 않는다.
- FULL 대비 시간 일반화 개선 주장은 동일 K=1 조건의 Taxi proper NLL이 FULL보다
  0.005 이상 낮아야 사용한다. FULL의 해당 값은 아직 없어 현재 판정할 수 없다.
- 원본 T0 대비 body MAE ≥5% 개선 목표, 세 데이터셋 raw RMSE 공통 개선,
  RMTPP·THP·NHP·SAHP 대비 우위는 각각 별도 주장이다. 위 screening만으로 모두를
  달성했다고 쓰지 않는다. 추가 seed와 최종 held-out 확인 전에는 채택하지 않는다.

## 다음 작업과 실행 경계

**다음 작업 / 로컬 — 독립 모델 경로와 정규화 평가 route를 구현한다.**
- 이 설계에 따라 모델·계약 테스트를 완료하고 source commit을 만든다.
- 정확한 B aligned artifact 위치를 찾고 SHA를 확인한다. 회수되면 B head를 재학습하지 않는다.
- 기존 FULL과 새 후보는 각각 자기 hidden cache를 생성해야 한다. B hidden cache를 대체
  입력으로 사용하지 않는다. FULL의 수량 학습 자체를 다시 할 필요는 없다.

**후속 작업 / GPU — 실행 계약이 준비된 뒤 계약·성능을 순서대로 검증한다.**
- 서버·source·data·시간 평가 artifact를 고정하고 CUDA·비용·full-data e1을 확인한다.
- 기존 FULL의 정상화 시간 기준선과 고정 후보 seed42를 평가한 뒤 공통 기준을 판정한다.
- 이번 로컬 후보 선정으로 GPU job을 시작하거나 `paper_research/develop`,
  `paper_research/master`에 병합하지 않는다.

근거와 기존 결과의 한계는
`paper/results/hard_lmm_bounded_qk_design_20260908/README.md`에 기록했다.
