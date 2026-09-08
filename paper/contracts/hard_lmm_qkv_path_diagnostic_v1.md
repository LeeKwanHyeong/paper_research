# Frozen Q/K/V 경로 진단 계약

상태: **로컬 진단 실행 전 동결**. 기계 판정 기준은 같은 이름의 JSON이 우선한다.
최종 사전 검토 JSON SHA-256:
`a4ae310d254ed124fc8be4c340a10d22c462a33fb08d669afd38504cf978ddea`.

## 목적과 범위

Causal-QKV seed42 결과에서 수량 개선과 시간 손실 악화가 함께 나타났다.
이번에는 학습이 끝난 checkpoint에서 추가 Q/K/V 경로를 선택적으로 끄고,
QK를 유지하면서 V를 제거하는 후속 Backbone 후보를 검토할 근거가 있는지 확인한다.
모델 구현·loss·head·selector를 바꾸거나 다시 학습하지 않는다.
CPU만 사용하며 validation 및 held-out 행은 읽지 않는다.

대상은 `paper_research / codex/hard-lmm-causal-qkv`다.
모델과 loader는 source `84668e207d5121f211a6a93af12ca2f96068a25e`에서 복원한다.
405개 Python 파일의 기존 manifest와 입력 데이터·checkpoint·summary hash를 검증한다.
현재 작업 폴더의 수정된 모델 파일은 실행에 사용하지 않는다.

## 표본과 대조군

Taxi, Intermittent, Instacart의 train target을 dataset당 4,096개 선택한다.
Series ID의 고정 SHA-256 규칙으로 두 fold를 만들고 각각 2,048개를 뽑는다.
Seed는 20260908이며 예측값이나 오차를 보기 전에 target identity를 저장한다.
Fold마다 고유 series 30개 이상을 요구한다. 이 조건을 맞추기 위해 오차나 수량을
기준으로 다시 뽑지 않는다. 수량 body와 tail은 기존 train p95/p99 경계를 사용한다.
Duration 구간은 전체 train target의 사분위수로 고정한다.

| 표기 | 동결 checkpoint와 추가 경로 |
| --- | --- |
| B | 별도로 학습된 기존 Hard-LMM raw-RMSE 선택 checkpoint |
| FULL | Causal-QKV의 Q/K/V 모두 유지: 111 |
| QK | Q/K 유지, 추가 V 제거: 110 — 유일한 주가설 |
| QV, KV, Q, K, V | 개별 경로와 상호작용을 설명하기 위한 비교 |
| ZERO | 동일 Causal-QKV checkpoint의 추가 Q/K/V를 모두 제거: 000 |

ZERO의 encoder·memory·head는 FULL 학습에 함께 적응했으므로 B와 같지 않다.
어떤 switch도 재학습된 별도 모델이 아니다. B와 FULL은 이미 해당 train series로
학습되었으므로 두 fold는 train 내부 일관성 확인이며 일반화 검증이 아니다.

## 출력과 검증

모든 variant를 동일 target·batch에서 평가한다. 마지막 관측 위치의 H1, H2,
Hard-LMM residual, 결합 상태와 top-4 index를 저장한다. 수량 MAE, raw RMSE,
log-MSE와 기존 float32 시간 loss를 동일한 함수로 계산한다.
시간 head의 intercept, exp(intercept), w, wΔt, 누적 항 및 clamp 활성 여부는
별도 기록한다. Float64 분해값은 설명용이며 기존 float32 loss를 대체하지 않는다.

첫 고정 batch에서 기존 `target_outputs`와 출력이 일치해야 한다.
미래 target의 수량·간격 및 padding을 바꿔도 마지막 관측 표현, 수량 예측,
top-4와 시간 head 입력은 그대로여야 한다. Target 의존 loss의 변화는 허용한다.
각 switch는 지정한 추가 kernel의 세 row만 0으로 만들고 원본을 복원한다.
모든 state 및 checkpoint 파일 hash는 실행 전후 같아야 한다.
Nonfinite 출력·loss는 제거하거나 평균에서 누락하지 않고 오류로 중단한다.

B와 FULL의 각 fold에서 미리 정한 첫 네 batch에 대해 시간·수량 loss의 gradient를
별도로 측정한다. 공유 layer, Hard-LMM bank, 추가 Q/K/V의 norm·내적·cosine을
보고한다. Optimizer step은 없으며 gradient 부호만으로 경로 분리를 선택하지 않는다.

## 사전 판정 기준

QK가 FULL 대비 raw RMSE·전체 MAE를 1% 이내, body·tail MAE를 2% 이내로
보존하고 시간 loss 증가는 0.01 이내여야 한다. ZERO 대비로도 raw RMSE 1%,
body·tail MAE 2%, 시간 loss +0.01 이내여야 한다. 이 조건은 세 데이터셋의
두 fold에 동일하게 적용한다.

적어도 한 데이터셋의 두 fold에서 QK가 ZERO보다 raw RMSE를 1% 이상 줄여야
QK 경로의 수량 기여가 남아 있다고 본다. FULL의 시간 loss가 B보다 0.01 넘게
나쁜 fold에서는 V 제거가 `max(0.005, FULL−B 격차의 20%)` 이상을 줄여야 한다.
어느 fold에서도 시간 문제가 재현되지 않으면 이 진단으로 원인 가설을 통과시키지 않는다.
별도 학습된 B 대비 격차와 Taxi 수량 개선 보존율은 함께 보고하되, frozen 개입의
B 격차를 QK 재학습 성공의 필요조건으로 해석하지 않는다.

Tail은 fold당 10 targets 및 5 series 이상이 있어야 보호 여부를 판정한다.
충분히 측정된 수치 기준이 하나라도 실패하면 ‘현재 개입 결과로는 구현 근거 부족’이다.
다른 수치 기준이 통과해도 표본이나 시간 문제 재현이 부족하면 ‘판정 유보’다.
전부 통과하면 ‘QK-only 학습 계약을 작성할 근거 확보’이며 성능 채택이 아니다.
다른 switch의 수치가 좋아 보여도 이번 gate의 대체 후보로 선택하지 않는다.

## 사전 검토에서 반영한 변경

예측값 생성 전에 독립 검토를 받아 표본 수를 2,048에서 4,096으로 늘리고
tail series 수를 추가했다. 또한 단순 V 제거로 전체 모델이 약해지는 효과와
QK 자체의 기여를 구분하기 위해 ZERO 비교를 판정에 포함했다.
별도 학습된 B의 격차를 강제로 회복하는 초기 안은 coadaptation 한계에 맞게
서술 비교로 옮겼다. 이 변경 뒤 JSON을 동결하며 결과 확인 후 기준을 수정하지 않는다.

기존 20260820의 T1 tail-shared gradient 감사는 이번 B 또는 Causal-QKV의
증거가 아니다. 현재 checkpoint에서 새로 계산한 값만 이번 진단에 사용한다.

## 다음 실행 경계

이번 승인 범위는 계약, 진단 구현, 단위·계약 검증, 로컬 train 진단과 증적 커밋이다.
새 Backbone 학습, GPU 실행, 추가 seed, held-out 평가 및 master 병합은 포함하지 않는다.
진단 실패는 QK-only 재학습이 불가능하거나 Backbone 개선 여지가 없다는 뜻이 아니다.
