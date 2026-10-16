# QK-only 고정 모델의 다음 진단 구현 가능성

**Q/K 국소 잔차를 함께 켜거나 끄는 진단은 소스·가중치 변경 없이 가능합니다. 다만 현재 선택 모델의 개별 예측행은 저장돼 있지 않습니다.**

## 완료 — 현재 증적과 최소 변경 위치 확인

선택 epoch는 seed42/52/62 각각 **5/5/22**이며, 선택 binary·metadata·학습 이력과 source143의 SHA를 확인했습니다. 세 native Validation receipt는 모두 `raw_predictions_written=false`입니다. `prediction_rows=6690`은 처리 개수입니다. 여섯 집계 지표와 대상 identity는 있지만, 과거 개별 prediction·time-loss 배열에 대한 재현은 확인할 수 없습니다.

최소 진입점은 `models.TPPs.CountAwareTitanQKOnlyCNN` 모듈 내부의 `causal_depthwise_residual` 바인딩입니다. 이 후보의 `QKOnlyAttention.forward` 108·109행은 그 함수를 Q와 K에 각각 한 번 호출합니다. 공통 `qkv` projection 뒤이며 메모리 토큰 결합 전입니다. 원본 공통 CNN 모듈의 함수나 source 파일은 수정할 필요가 없습니다.

숫자 **0/1만** 받는 새 진단 context에서 원함수를 먼저 호출하고, 1이면 반환값을 그대로, 0이면 같은 모양의 0 residual을 반환할 수 있습니다. 원래 shape·dtype·device·관측값 검사와 gather 오류를 유지하고, 중첩 적용을 막으며, `finally`에서 함수 객체를 복구해야 합니다. 원함수에는 kernel 자체의 유한성 검사가 없으므로 off가 비정상 계산을 숨기지 않도록 반환 residual의 유한성도 별도 확인해야 합니다.

모든 가중치·named buffer·`_ctx_mem` hash와 eval/frozen 상태를 전후 비교합니다. V projection·head·메모리의 연산과 가중치는 유지하지만, Q/K 변화로 attention과 이후 메모리 **출력**은 달라질 수 있습니다. 이 진단은 Q/K의 공동 영향만 보며 Q와 K를 분리하지 않습니다. off는 기존 CNN이나 Q/K 없이 새로 학습한 모델이 아닙니다.

## 다음 작업 — 먼저 current Train·Validation 행 확보를 권장

가정이 더 적은 첫 단계는 branch를 건드리지 않고 현재 모델3개의 Train·Validation 행을 저장하는 것입니다. 기존 행 exporter를 새 QK-only 경로로 옮기면 됩니다. 합성 qualification3 → Validation3 전체 재현 → Train3의 **9개 worker(실데이터6개)**로 구성할 수 있습니다. 여섯 Validation 지표는 기존 `1e-5` 상대·절대 허용오차, 대상6690개는 동일 identity로 확인합니다.

곧바로 Q/K 공동 민감도를 보려면 qualification3 → on Validation3 → on Train3 → off Validation3 → off Train3의 **15개 worker(실데이터12개)**가 필요합니다. 각 단계는 최대3병렬·새 process로 분리합니다. 먼저 얻은 on행을 새 불변 참조로 고정해야 하며, 이를 과거 native 개별행 재현이라고 표현하면 안 됩니다. 새로운 on행에 대해 또 한 번 Validation 행별 재현을 요구하면 별도3개가 추가되어 총18개가 됩니다.

## 다음 작업 — 대상 identity와 기존 15조건 분석을 구분해 연결

native Validation은 pure Train/Validation 파일을 읽었지만 target_id 계산에는 parent full-input SHA를 사용했습니다. 기존 행 exporter는 pure-input SHA를 사용합니다. 이력 평균도 native는 torch float32, 행 진단은 float64 관측 요약입니다. 그러므로 두 target/truth digest를 그대로 같다고 검사하면 안 됩니다. hard_lmm 대상·수량 identity와 여섯 지표를 우선 맞추고, 과거 target/truth/time digest까지 필요하면 native 식을 별도 감사 digest로 그대로 계산하면 됩니다. parent SHA 문자열은 학습 metadata에서 얻을 수 있어 Test 파일을 열 필요가 없습니다.

순수 `analyze_titantpp_value_difference_rows.py`의 join·15조건·가중·가산 검사는 재사용할 수 있습니다. 기존 Train 경계 **Q25=0.4054651081081645, Q75=1.252762968495368**와 동일 입력·loader·대상을 그대로 유지합니다. 전체1 + 이력패턴4 + 패턴×변화폭10 조건, 행 가중 및 각 조건의 Train/Validation 공통 개체 균등 가중, 제외된 행과 seed 예외를 모두 보존합니다.

다만 기존 operational caller는 old source141·ValueDifference·epoch9/5/22·old ON receipt를 고정하므로 그대로 실행할 수 없습니다. 새 caller에서 source143, QK-only, epoch5/5/22, local384·active79235·registered103967, 새 receipt 경로를 결속해야 합니다. on/off 비교에서는 순수 계산기의 candidate/cnn 별칭을 off/on으로 명시적으로 바꿔 보고합니다. 기존 CNN 추론은 필요하지 않습니다.

## 외부 작업 대기 — 새 계약 이후 구현·검증·진단

이번 검토는 **새 코드0·모델 생성0·checkpoint load0·추론0·원격0·Test 열람0**입니다. 읽기 전용으로 binary SHA와 Train/Validation metadata를 확인했고 두 메모만 작성했습니다. 실제 분기 효과·오류 원인·새 학습 후보의 성능은 아직 측정하지 않았습니다. 새 임계값이나 학습 진입 기준도 선택하지 않았습니다. 부모가 진단 범위를 정한 뒤 새 계약·파일·합성 검증을 거쳐 진행해야 합니다.
