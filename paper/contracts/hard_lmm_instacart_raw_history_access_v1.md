# Instacart raw history 접근성 진단 계약

## 목적

Instacart에서 다음 수량의 남은 오차를 설명할 수 있는 신호가 관측 raw 이력에는 있지만 현재 Hard-LMM encoder 상태 `h`에서는 접근하기 어려운지 확인한다. 이 진단은 Backbone을 학습하거나 후보를 채택하는 실험이 아니다.

## 표본과 입력 경계

- 앞선 시간 진단이 결과를 보기 전에 선택한, 이전 8,192개 진단에 쓰이지 않은 train series 65,536개를 재사용한다. 각 series에는 context 하나만 있다.
- target label이나 checkpoint를 읽기 전에 이력 길이 `3 <= H <= 32`인 65,525개 행을 고정한다. `H>32`인 11개 행은 자르지 않고 제외한다.
- raw 입력은 관측 사건 전체를 오래된 순서부터 최근 순서로 배치한다. 사건마다 encoder가 실제 받는 `[log1p(delta_t), log1p(quantity)]`를 사용하고 32개 slot의 오른쪽에 맞춘다. 왼쪽 padding은 0이며 총 64차원이다.
- target 사건의 시간 간격과 수량은 입력에 포함하지 않는다. `h`는 같은 observed-only 입력에서 마지막 사건의 64차원 local encoder 상태를 사용한다.

## 동일 용량 비교

`raw64`, `h64`, 이력 길이만 표현하는 `history_only64`, 독립 난수 `sham64`는 모두 64차원이다. 모든 입력에는 같은 두 OOF 보조 예측기를 적용한다.

- Linear: 반대 fold 통계만 사용한 float64 ridge, SSE 기준 `lambda=1`, 절편은 벌점 없이 반대 fold label 평균으로 둔다.
- Random128: 표준화한 원래 64차원과 seed 20260905의 고정 tanh 특성 128개를 결합한 뒤 같은 ridge를 적용한다.
- fold는 기존 series 단위 두 fold를 그대로 사용한다. 표준화와 적합은 항상 반대 fold에서만 수행한다.

주 label은 각 checkpoint의 `log1p(target quantity) - frozen base log prediction`이다. 현재 기준선인 separate-key가 주 판정 대상이고 원본 checkpoint는 같은 방향이 재현되는지 별도로 확인한다. Direct `log1p(target quantity)` 예측은 raw 이력 자체의 제한된 예측 신호를 설명하는 보조 결과일 뿐 병목 판정에 사용하지 않는다.

## 판정

각 비교는 pooled residual MSE를 1% 이상 줄이고, 두 fold의 평균 제곱오차 차이가 모두 양수이며, 10,000회 series bootstrap의 동시 하한이 0보다 커야 한다. 양의 log MSE 결과가 body raw MAE 악화와 충돌하면 실패로 처리한다.

Separate-key에서 두 decoder 모두 다음을 충족할 때만 현재 encoder 경로의 병목 후보로 판정한다.

1. `raw64`가 constant와 `history_only64`를 각각 이긴다.
2. `raw64`가 `h64`를 이긴다.
3. `sham64`가 같은 절차로 constant를 이기는 이상 징후가 없다.

원본에서도 같은 조건을 충족해야 두 checkpoint에 공통인 encoder 병목 후보라고 부른다. 하나라도 불일치하면 구현을 보류한다.

## 해석 한계

이 probe가 실패해도 raw 이력에 정보가 전혀 없거나 데이터셋을 개선할 수 없다는 뜻은 아니다. 고정 linear/random-feature probe가 복잡한 순서 상호작용을 놓칠 수 있다. 반대로 raw가 우세해도 encoder의 문자 그대로의 정보 삭제만을 입증하지는 않는다. Backbone은 이 train series를 이미 보았고 checkpoint도 validation에서 선택됐으므로 결과는 train 내부의 조건부 접근성 증거다. 이번 단계에서는 validation, held-out test, 5090, Backbone 학습을 사용하지 않는다.
