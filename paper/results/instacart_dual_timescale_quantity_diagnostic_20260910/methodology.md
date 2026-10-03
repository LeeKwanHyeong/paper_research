# Instacart current dual-timescale candidate: frozen quantity diagnosis

이 진단은 seed42의 이미 선택된 B, 현재 dual-timescale 후보, RMTPP, THP를 동일한 원본 수량 표본에서 비교한다. 모든 모델의 checkpoint 선택은 validation raw RMSE이고, loss·모델·checkpoint를 바꾸거나 추가 학습하지 않는다. Held-out test는 읽지 않는다. 실행은 로컬 CPU에 한정한다.

## 계약과 모집단

`contract.json`은 새로운 행 단위 예측을 생성하기 전에 동결했다. 계약 SHA256은 `0f7a37b537778ab71d4910ff4a47e624360f9be8767f53ebaf8b5070ab2a1152`다. validation 503,733건과 train 1,991,192건을 사용한다. 수량 구간 경계 8·20·25·35는 기존 train 통계이며, history 구간은 1, 2–3, 4–7, 8–15, 16–31, 32–63이다. History는 고객의 전체 주문 수가 아니라 해당 예측에 실제로 들어간 관측 창의 길이다.

B의 이전 train·validation 캐시는 checkpoint 및 state SHA, 모든 표본의 식별자·수량·순서가 일치하는 것을 독립 재구성으로 확인한 뒤 재사용한다. 이전 FULL 또는 BOUNDED-QK 예측은 현재 후보로 사용하지 않는다. 나머지 세 모델은 `ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c`에서 CPU로 복원한다. 해당 source에서 B·RMTPP·THP의 기존 loader·encoder·수량 경로가 유지됨을 확인했다. 전체 validation MAE·RMSE·log-MSE를 저장된 summary와 대조한다.

처음 고정한 절대 오차 0.00001 기준에서는 RMTPP의 RMSE 차이 +0.0000140398 때문에 실패했다. 이 기록을 유지한다. 행 단위 분석 전에 별도 `numeric_replay_policy.json`을 고정하여 CPU torch 2.14와 기존 CUDA torch 2.11 간 비교에 `atol=0.00001, rtol=0.00001`을 적용하고, 세 RMSE 비교의 방향 보존 및 원래 차이 대비 수치 왜곡 0.1% 이하를 함께 요구했다. 모두 통과했으며 실제 최대 왜곡은 RMTPP 비교의 0.0473%였다. 이는 기술적 재현 허용 오차의 명시적 변경이며 모델 성능 채택 기준 변경이 아니다.

수량 구간의 추가 재현 감사에서도 모든 30개 후보-기준선 MAE·RMSE 차이의 방향이 보존되었다. 구간별 최대 절대 재현 차이는 MAE 0.0006062, RMSE 0.0004378이었다. 따라서 표본별 bitwise 동일성을 주장하지 않는다. 표에는 실제 캐시/CPU 예측으로 다시 집계한 값을 사용한다.

## 분석과 해석의 경계

오차는 `prediction - true_qty`다. 음수 bias는 과소예측이다. 제곱오차 평균은 `MSE = bias² + centered MSE`로 정확히 분해한다. 구간의 전체 차이 기여는 `구간 표본 수 / 전체 표본 수 × 구간 ΔMSE`다. 각 구간의 기여를 더하면 전체 ΔMSE가 재현되어야 한다. RMSE 자체는 구간별로 더하지 않는다.

Series별로 나쁜 차이와 좋은 차이가 상쇄될 수 있으므로, 상위 series의 순기여와 양의 총기여 비중을 구분한다. Series를 묶어서 재표집하는 bootstrap은 현재 고정된 seed42 모델에 대한 표본 변동만 설명한다. 초기화·학습 seed 불확실성이나 최종 held-out 성능을 대신하지 않는다. 이 validation은 이미 checkpoint 선택에도 사용되었으므로 bootstrap 구간을 독립적인 최종 유의성 검정으로 해석하지 않는다.

Validation에서 설명을 좁힌 뒤 `validation_hypotheses.json`을 고정하고 train 분석을 시작한다. 두 train fold는 `SHA256("instacart_dual_timescale_quantity_v1:20260910|" + series_id) mod 2`로 만든다. 같은 series는 한 fold에만 속한다. 그러나 모델 학습에는 두 fold가 모두 사용되었으므로 이는 학습 내 진단 재현이며, out-of-fold 일반화 검증이 아니다.

실제 target 수량 구간은 사후 오차 설명에만 사용한다. 예측 시 알 수 있는 입력 조건이 아니다. Bias 분해가 성립해도 일괄 상향 보정이 미관측 데이터에서 성능을 개선한다는 뜻은 아니다. 관측적 차이만으로 특정 memory 모듈의 인과 효과를 확정하지 않는다.

## 현재 후보의 구조를 읽을 때의 주의점

관측 history가 h이면 쓸 수 있는 인접 전이는 h−1개다. 최근 8개 전이는 local, 더 오래된 전이는 global에 들어간다. 따라서 history 2부터 local 경로가 가능하고, history 10부터 global 경로가 가능하다. 8–15 구간 전체가 global 활성 구간은 아니다. 실제 read에는 query에 따른 support 조건도 필요하다.

History 1에서는 동적 residual이 0이지만, 학습 후 후보의 출력이 B와 같다는 보장은 없다. 두 모델은 공통 초기화에서 각각 전체 학습되어 encoder·static memory·head가 달라질 수 있다. 초기 alpha=0의 출력 일치와 학습 후 출력 일치를 혼동하지 않는다.

## 재현

프로젝트 루트에서 다음 순서로 실행한다. 기존 출력이 있으면 추론기는 덮어쓰지 않는다. 원본 소스·checkpoint·데이터 SHA와 계약이 모두 필요하다.

```sh
python3 paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/run_inference.py --split validation
python3 paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/check_numeric_replay.py
python3 paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/analyze_predictions.py --input-parquet search_artifacts/instacart_dual_timescale_quantity_diagnostic_20260910/validation_predictions.parquet --split validation --output-dir paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/validation --contract paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/contract.json
# validation 설명을 검토하고 validation_hypotheses.json을 기록한 뒤에만:
python3 paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/run_inference.py --split train
python3 paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/analyze_predictions.py --input-parquet search_artifacts/instacart_dual_timescale_quantity_diagnostic_20260910/train_predictions.parquet --split train --output-dir paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/train --contract paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/contract.json
python3 paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/verify_train_hypotheses.py
```

동결 계약은 한 번 생성된 파일을 사용한다. `freeze_contract.py`는 기존 계약을 덮어쓰지 않는다. 큰 checkpoint·행 단위 예측은 `search_artifacts/`에 두며, 이 보고서에는 집계와 감사 자료만 넣는다.
