# Frozen raw-affine calibration 계약 v1

- 동결일: 2026-09-07 KST
- 대상 저장소·작업 branch: `paper_research / codex/raw-affine-calibration`
- 구현 시작 revision: `9440609673cfdf7bcfec5c77d7f677affd04ae49`
- 실행 장비: RTX 5080 한 대만 사용
- 평가 범위: seed42 validation-only
- Held-out test: 잠금 유지

## 현재 기준선

B는 기존 `log1p` MSE로 학습한 공통 Hard-LMM T0를 validation raw RMSE로
선택한 checkpoint다. 이 checkpoint는 Intermittent, Taxi, Instacart에서 각각
raw RMSE `1.499555`, `88.194997`, `5.872217`을 기록했다. Instacart의 같은
선택 규칙 비교군은 RMTPP `5.836961`, THP `5.852342`로, B가 근소하게 뒤진다.

이번 후보는 frozen checkpoint의 raw 수량 예측 뒤에 두 파라미터만 추가한다.
Encoder, Hard-LMM memory, top-4 검색, quantity head, time head와 base checkpoint
선택은 바꾸지 않는다. 따라서 이 실험은 **output-interface ablation**이다.
성공하더라도 Backbone이나 memory가 개선됐다고 주장하지 않는다. Backbone
효과를 비교하려면 동일한 calibration 절차를 각 backbone에 적용한 뒤 남는
차이를 봐야 한다.

기존 frozen calibration과도 구분한다. 이전 실험은 오래된 joint-selected
checkpoint의 quantity logit에 bounded MLP correction을 더했고 Instacart와
Intermittent에서 identity가 선택됐다. Taxi 개선은 train-only 상수 logit
offset이 거의 전부 재현했다. 이후 readout factorial도 raw MAE 또는 log-MSE
목표의 logit correction을 검토했지만 공통 후보를 만들지 못했다. 이번 후보는
최신 raw-RMSE-selected checkpoint의 **raw prediction에 positive-slope OLS와
nonnegative projection을 적용**한다는 점이 다르지만, global bias 가설을 다시
검증하는 실험이다.

## 단일 후보와 적합 방식

Frozen source가 출력한 nonnegative raw quantity를 `q_hat`이라 할 때

\[
q_{cal}=\max(0, a q_{hat}+b),\qquad a>0
\]

로 고정한다. `a=1, b=0`은 원본과 정확히 같은 identity다. 각 fitting
partition에서 float64 centered ordinary least squares로 clamp 전 affine map의
squared error를 최소화한다. Unconstrained slope가 양수가 아니면 사전에 고정한
`1e-12`로 투영하고, 그 slope에서 절편을 다시 계산한다. 마지막으로 inference와
metric 계산에서만 음수 affine output을 0으로 투영한다.

따라서 이 fitting은 **clamp 전 affine SSE의 positive-slope OLS**이며, clamp가
적용된 `q_cal` MSE의 정확한 optimum이라고 주장하지 않는다. Clamp 전·후 train
SSE와 clamp된 event 수를 모두 기록한다. Dataset마다 학습되는 `a,b` 값은 달라도
되지만 함수, OLS 규칙, minimum slope와 판정 규칙은 모든 dataset과 backbone에서
같다.

Validation과 held-out target은 parameter나 train-only activation을 정하는 데
사용하지 않는다. Frozen source는 eval mode와 `requires_grad=False`를 유지한다.
OLS는 optimizer나 gradient path를 만들지 않으며 자동 재시도나 대체 fitting은
허용하지 않는다.

## 다음 작업 / 5080 — Train-only 필요조건 확인

각 dataset의 canonical series identifier를 고정된 SHA-256 규칙으로 두 fold에
배정한다. 한 series의 모든 train target은 같은 fold에 남는다. Fold 0에서
적합해 fold 1을 평가하고 반대 방향도 평가한다. 세 B dataset의 여섯 방향
모두에서 held-fold raw RMSE가 identity보다 엄격히 낮아야 한다. 두 방향의
cross-fitted prediction을 합쳤을 때 body(`<= train p95`)와
`> train p99` MAE 악화는 각각 2% 이하여야 하며, dataset 내부의 합산 OOF
raw RMSE 개선율은 최소 1%여야 한다.

이 fold는 calibration parameter의 series 간 전달만 확인한다. Frozen B가 원래
전체 train split으로 학습됐으므로 독립적으로 재학습한 Backbone fold라고
해석하지 않는다. B가 한 dataset에서라도 필요조건을 통과하지 못하면 full
validation prediction을 읽기 전에 중단한다. 이 단계가 모두 통과한 source만 전체 train
target으로 한 번 다시 적합한다.

## 조건부 다음 작업 / 5080 — B의 세 dataset validation gate

Train-only 필요조건을 모두 통과한 뒤에만 전체 validation을 한 번 평가한다.
Intermittent, Taxi, Instacart 각각에서 다음 조건을 모두 충족해야 한다.

- Raw RMSE가 uncalibrated B보다 엄격히 낮아야 한다.
- 전체 MAE 악화는 uncalibrated B 대비 2% 이하여야 한다.
- Body MAE와 `>p99` MAE 악화는 각각 2% 이하여야 한다.
- Target identity와 target 수가 원본 B replay와 정확히 같아야 한다.
- Base replay와 calibrated replay의 event-time 출력과 per-event Time NLL은
  같은 5080 실행 안에서 정확히 같아야 한다.
- 모든 parameter, prediction과 metric이 finite여야 한다.

Raw RMSE는 dataset마다 단위와 규모가 다르므로 세 dataset의 squared error를
하나로 합친 pooled RMSE를 판정에 사용하지 않는다. Dataset별 상대 변화와 그
비가중 macro 평균만 보조 통계로 보고한다. Raw RMSE, 전체·body·`>p99` MAE,
`log1p` MSE, signed raw bias와 다섯 quantity 구간을 모두 남긴다. 같은 series의
event가 독립이라고 가정하지 않도록 series-clustered paired bootstrap 구간도
보고하되, 고정 seed `20260907`로 series 500회 재표집하고 draw buffer는 최대
8,192개로 제한한다. 이미 알려진 seed42 validation을 새로운 확증 표본으로
표현하지 않는다.

한 dataset이라도 실패하면 공통 calibration 후보를 기각한다. 성공한 dataset만
골라 공통 개선이라고 해석하지 않는다.

## 조건부 다음 작업 / 5080 — Instacart 공정성 비교

B가 세 dataset validation gate를 모두 통과할 때만 Instacart의 RMTPP와 THP를
평가한다. 두 source에도 동일한 series fold, raw-MSE 적합, train-only activation,
identity fallback과 full-train fit을 적용한다. Base checkpoint를 calibration
결과에 맞춰 다시 고르지 않는다.

최종 비교 행은 calibrated-pipeline B, RMTPP, THP 세 개다. B의 MAE와 RMSE가
각 비교군보다 모두 엄격히 낮을 때만 Instacart 공정성 gate를 통과한다. 한
비교군에 하나의 metric이라도 지면 추가 seed를 실행하지 않고 held-out test를
열지 않는다. 통과 결과는 calibrated predictor의 우위만 뜻하며 Hard-LMM
Backbone이 보정 없이 우수하다는 근거가 아니다.

## 로컬·CUDA 계약 검증

실제 full-data fit 전에 다음을 검증한다.

- `a=1, b=0`에서 모든 raw quantity output이 source output과 정확히 같다.
- Calibration을 켜거나 적합해도 time output과 per-event Time NLL이 정확히 같다.
- Source model state SHA-256가 prediction 추출 전후에 변하지 않는다.
- OLS fitting은 optimizer나 source-model gradient를 생성하지 않는다.
- Target·future event·padding을 바꿔도 고정된 `q_hat`에 대한 inference 결과는
  변하지 않는다.
- `a`는 strictly positive이고 output은 nonnegative이며 숨은 upper clipping은 없다.
- 극단 입력에서도 parameter, loss와 prediction이 finite다.
- 저장한 `a`, `b`, OLS 통계와 모든 provenance를 복원했을 때
  prediction vector와 metric이 재현된다.
- 잘못된 dataset/source/checkpoint/fold digest, 부분 checkpoint와 손상 artifact는
  거부하며 기존 output directory를 덮어쓰지 않는다.

5080에서는 먼저 source·data·split·target identity·checkpoint file/state hash를
검증하고 B 세 dataset의 train CUDA real-batch identity/finite smoke 3건을
수행한다. 이어서 B 세 dataset의 full train prediction으로 train-only gate를
판정한다. 세 dataset이 모두 통과한 뒤에만 validation CUDA smoke 3건과 full
validation을 순서대로 수행한다. 이 단계는 frozen
inference와 2-parameter fit이며 e300 Backbone 재학습이 아니다.
이 순서는 committed `paper/scripts/control_frozen_raw_affine_5080.py`가 한 번에 한
process만 실행하며, 앞 gate가 실패하면 뒤 단계의 prediction을 materialize하기
전에 종료한다.

## 고정된 source와 데이터

| Dataset | Source | Best epoch | Checkpoint file SHA-256 | State SHA-256 |
| --- | --- | ---: | --- | --- |
| Intermittent | B | 77 | `63d38ee4401c208dcfa24206682edaca56e2d2a5ca1037b09decdc2ffa567c08` | `4b984d415d83a479fdacb49b12d964afa668adad136511590a54293df763c0d0` |
| Taxi | B | 45 | `46b456816dbc3c36e4f9c115c05b480fc030a8c2004646d179416d528bd897d5` | `eddc8b4a11b233606bf74508f0ef721e05e55bbcb047d18edab4aa61f2f43c41` |
| Instacart | B | 72 | `e594f8df0dcd66249c3724063f44eee3de213bc83a780ca81fdf39a094bfed07` | `a74e1f3055b03db5ac85adf2fdb63711890ea9d06449965e3ca749aefccc2f02` |
| Instacart | RMTPP | 40 | `64ec26afba13d3117e0d3cf0cc1c554d2a6de10b5a4181834dd0096fd290aacc` | `2af7eb20d81c802096963c6301fa8dfd46cdb7ba44502d978ad124a4cfb4fd38` |
| Instacart | THP | 50 | `f52310f70e19bc2a9e7f7b63d256e684b9aee7daa8711e5505e4521f3582df37` | `6c9a686c27a89e0ded3ccc8ee2cb920a137a526081aa14e87d516e7343b9883b` |

Data, split, train/validation target identity와 quantity hash, history hash,
checkpoint reference metric의 full precision 값은 JSON 계약에 고정한다.

## 완료 조건과 이후 경계

이번 작업의 완료 조건은 5080에서 prospective contract를 그대로 실행하고,
train-only gate와 조건부 validation·fairness gate 결과를 독립 감사 가능한
artifact로 남기는 것이다. 결과를 본 뒤 calibration 식, OLS 규칙, minimum slope,
fold, metric 또는 gate를 바꾸지 않는다. Seed42가 모두 통과해도 seeds52·62와
held-out test는 이 계약에 포함되지 않으며 별도 계약 없이 실행하지 않는다.
