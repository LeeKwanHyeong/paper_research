# Taxi의 시간 일반화 문제: Train/Validation 진단

전체 학습이 끝나기 전에 완료된 Taxi·RAF 6개 원본으로 진단을 수행했습니다. **Taxi의 시간 출력부 재적합은 고정된 표현 위에서도 Train 손실을 줄이고 Validation 손실을 늘렸으며, 분포 퍼짐을 줄이는 파라미터 변화가 그 손해의 대부분을 설명합니다.** 이는 재적합 단계의 계산적 설명입니다. 원래 공동 학습의 근본 원인이나 CNN/GRU의 독립 효과까지 확정하지 않습니다.

**현재 기준선 — 완료**

- 원래 CNN+GRU54: 보존된 관측에서 8/9 조건 완료(신규5/6 + 재사용seed42 3). 5090 Intermittent62만 최종 검증 대기입니다.
- bias2/full130 재적합: Taxi·RAF 각3seed 완료, 원본 archive SHA와 서버 종료 검증 완료. Taxi full130은 세 seed 모두 E0를 선택했습니다. 아래 악화한 마지막 상태는 채택하지 않은 진단 대상입니다.
- 이 분석은 원본 캐시와 Train/Validation 행의 정합을 확인한 CPU 계산입니다. GPU encoder 재추론, optimizer update, checkpoint 재선택, Test 열람은 수행하지 않았습니다.

**동일한 고정 가중치에서 Train/Validation을 비교한다 — 완료**

학습 중 기록된 Train loss는 batch마다 가중치가 변하는 평균이므로 직접적인 endpoint 비교에 쓰지 않았습니다. 보존된 동일 표현과 전체 Train/Validation 모집단에서 원래 E0 및 full130 마지막 상태를 각각 고정해 native 양성정수 시간 likelihood를 계산했습니다. TimeNLL은 낮을수록 좋습니다.

| 데이터·seed | 원래 수량-selected epoch | 새 head selected/last epoch | 고정 Train TimeNLL E0→last | 고정 Validation TimeNLL E0→last | 채택한 Validation TimeNLL |
|---|---:|---:|---:|---:|---:|
|RAF·42|9|1/11|3.050714→3.045850|3.548010→3.559690|3.505588|
|RAF·52|5|2/12|3.054318→3.055618|3.479681→3.489568|3.460704|
|RAF·62|20|7/17|3.043048→3.038257|3.611548→3.665207|3.563462|
|Taxi·42|126|0/10|0.345847→0.301620|1.365979→4.507146|1.365979|
|Taxi·52|157|0/10|0.302600→0.278794|2.163854→6.274360|2.163854|
|Taxi·62|54|0/10|0.483419→0.470188|0.875252→1.244418|0.875252|

Taxi 세 seed의 고정 Train은 개선됐지만 Validation은 악화됐습니다. RAF의 마지막 상태는 일부 Train/Validation이 악화됐으며, 선택된 head만 비교하면 3seed 평균은 3.546413→3.509918(1.0291% 감소)입니다. RAF seed52에서는 bias2의 선택 값3.439869가 full130의3.460704보다 좋았다는 불리한 결과도 유지합니다.

**분포의 중심과 퍼짐 변화가 손실에 얼마나 기여했는지 확인한다 — 완료**

시간 분포는 μ=v_t·h+b_t, σ=0.001+softplus(time_scale_weight·h+w_raw)입니다. σ는 잠재 log-duration의 퍼짐이며 시간 단위의 표준편차와 같지 않습니다. 기존/마지막 중심 block과 퍼짐 block을 서로 조합한 네 가지 고정 계산을 비교하고, 두 적용 순서를 대칭 평균해 전체 변화로 분해했습니다. 새 checkpoint 후보를 저장하거나 선택하지 않았습니다.

| Taxi seed | Validation σ 중앙값 E0→last | 전체 NLL 증가 | 중심 block 기여 | 퍼짐 block 기여 |
|---|---:|---:|---:|---:|
|42|0.235020→0.145228|+3.141167|-0.270809|+3.411976|
|52|0.240118→0.162045|+4.110506|-0.451546|+4.562052|
|62|0.468610→0.291635|+0.369166|+0.034639|+0.334527|

세 seed 모두 퍼짐 block이 Validation 손해를 늘렸습니다. seed42/52에서는 중심 block이 일부 손해를 상쇄했습니다. 분포를 좁히면 실제 긴 간격에 부여하는 확률이 매우 작아져 NLL 벌점이 커질 수 있습니다. 이 계산은 고정된 표현과 저장된 파라미터에 대한 정확한 분해이며, 어떤 학습 요인이 그 파라미터를 만들었는지는 별도 문제입니다.

σ의 실제 마지막 최솟값은0.017 이상으로 floor0.001에 붙지 않았습니다. native likelihood는 정수 관측 구간의 확률 질량을 float64 안정 log-CDF 차이로 계산하며, 이번 진단에서 NLL clipping을 적용하지 않았습니다. 따라서 현재 증거로 sigma-floor 붕괴나 NLL clipping 버그를 원인으로 보고하지 않습니다.

**시간 간격과 관측 이력별로 손해가 집중되는 대상을 찾는다 — 완료**

원본 loader의 retained history와 cache 순서를 맞췄고, 대상 (part,seq) 및 dt/qty를 대조했습니다. Train 기준 수량 경계[7,686,1562,3449]를 이력의 직전 수량 구간에도 고정 사용했습니다. target 간격 구간은 사후 분석용이며 미래 target으로 모델을 분기하는 규칙이 아닙니다.

- Taxi의 target 간격1은 Train82.4395%(31,651/38,393), Validation82.1239%(6,790/8,268)입니다. raw target dt0은 두 split 모두0개여서 0→1 clip이 이번 Taxi 모집단을 만든 것은 아닙니다.
- Validation의 직전 수량≤7인4,348행(52.5883%)에 full130 마지막 상태의 전체 NLL 증가 중97.4–98.5%가 집중됐습니다.
- 그중 실제 target 간격≥2인1,468행(17.7552%)에 전체 증가의86.2–95.0%가 집중됐습니다. 서로 겹치는 이력 구간들의 기여는 합산하지 않습니다.

| Taxi seed | 전체 NLL 증가 | 직전 수량≤7 증가 기여 | 해당 비중 | 직전 수량≤7·target 간격≥2 증가 기여 | 해당 비중 |
|---|---:|---:|---:|---:|---:|
|42|3.141167|3.093768|98.49%|2.708641|86.23%|
|52|4.110506|4.030050|98.04%|3.754742|91.35%|
|62|0.369166|0.359727|97.44%|0.350726|95.00%|

**간격 구성 차이와 같은 구간 안의 손실 차이를 구분한다 — 완료**

원래 E0의 Validation−Train 격차를 raw target 간격 구간별로 대칭 분해했습니다. 구간 비중 차이와 구간 내 평균 손실 차이의 합은 실제 전체 격차와 일치합니다. 인과 추정이 아니며 더 세밀한 구간의 숨은 구성 차이까지 제거했다는 뜻은 아닙니다.

| Taxi seed | E0 Validation−Train | 구간 비중 차이 | 구간 내 손실 차이 |
|---|---:|---:|---:|
|42|1.020131|0.016422|1.003709|
|52|1.861254|0.025639|1.835615|
|62|0.391833|0.015197|0.376635|

간격1 비중이 비슷하고, 이 구간 정의에서는 격차의96.1–98.6%가 같은 간격 구간 안의 손실 차이입니다. 단순히 Validation에 긴 간격이 더 많다는 설명만으로는 부족합니다. 이력 길이 중앙값은Train129/Validation157로 다르지만, 이 관측만으로 긴 이력이 원인이라고 단정하지 않습니다.

**원래 학습 이력과 Intermittent를 연결한다 — 진행 중 / 외부 작업 대기**

아래는 원래 공동 학습 history의 수량-selected epoch와 이력 내 TimeNLL 최소 epoch입니다. 최소 TimeNLL은 설명용이며 checkpoint를 바꾸지 않습니다. history Train loss는 online 평균이고 위 고정 head 계산과 다릅니다.

| 데이터·seed | 저장 epoch | 수량-selected epoch·TimeNLL | 이력 내 시간 최소 epoch·TimeNLL | 시간 최소 epoch의 수량 RMSE 손해 | 최종 여부 |
|---|---:|---:|---:|---:|---|
|Intermittent·42|52|E12·0.296848|E11·0.237343|+32.11%|완료|
|Intermittent·52|51|E11·0.287827|E2·0.216284|+263.00%|완료|
|Intermittent·62|31|E13·0.322681|E2·0.211881|+78.07%|진행 중·최종 아님|
|RAF·42|49|E9·3.548010|E2·3.324068|+10.23%|완료|
|RAF·52|45|E5·3.479681|E2·3.322293|+10.27%|완료|
|RAF·62|60|E20·3.611548|E7·3.340312|+4.21%|완료|
|Taxi·42|166|E126·1.365979|E2·0.653553|+39.95%|완료|
|Taxi·52|197|E157·2.163854|E2·0.651991|+57.17%|완료|
|Taxi·62|94|E54·0.875252|E2·0.654926|+60.01%|완료|

- 원래5080 실제 terminal 관측:2026-10-06 04:37:49KST. 5090 보존된 실제 관측:2026-10-06 10:29:41KST. 이 진단에서 새 SSH를 실행하지 않았으며 과거 시각을 현재 관측으로 표시하지 않습니다.
- Taxi·RAF는 모든 seed에서 시간 최소 epoch가 수량-selected epoch보다 이릅니다. Intermittent42/52도 같은 방향이며62는 미완료라서 세 seed 최종 결론을 내리지 않습니다.
- Intermittent seed42의 과거 A100 history와 이종 GPU Validation replay의 작은 차이는 원래 계약대로 모두 보존합니다. 최종 연결에서는 summary↔endpoint, history↔history anchor를 각각 대조하고 tolerance를 임의 확대하지 않습니다.
- [Intermittent 현재 연결 상태](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_time_generalization_diagnosis_20261006_v1/INTERMITTENT_STATUS.md)는 보존된 로컬 관측만 읽습니다. 원래9조건 terminal·새 원본 archive SHA·선택/마지막 Validation 완료 receipt가 확보되면 최종3seed Validation 요약과 연결합니다. Intermittent full130 재적합은 아직 수행하지 않았습니다.

**현재 증거가 의미하는 범위를 정리한다 — 완료**

확인된 사실은 Taxi head 재적합의 Train 개선/Validation 악화, 퍼짐 축소 block의 손해 기여, 작은 직전 수량·긴 target 간격의 집중입니다. 시간 예측이 학습 이력의 조건에 과하게 맞춰지고 평가 이력에서 과신하는 상황과 일치합니다.

아직 미확정인 것은 원래 CNN/GRU 표현 자체의 원인, 시간·수량 gradient 충돌, S2P2와 이력 조건별 차이, 시간 손해의 불가피성입니다. 이번 계산은 head 재적합 단계의 메커니즘을 좁혔지만 원래 공동 학습의 sigma 변화 전체를 직접 추적하지는 않았습니다. 외부 모델 우월성이나 논문 Contribution을 확정하는 증거로 사용할 수 없습니다.

**남은 작업을 목적에 맞춰 정렬한다**

1. **Intermittent 최종 결과와 회수를 연결한다 — 외부 작업 대기.** 기존5090 학습과 승인된 hourly 원본 회수는 계속합니다. 실제 terminal·manifest/archive SHA·선택/마지막 Validation이 확인된 뒤 세 데이터 방향을 확정합니다.
2. **Taxi의 시간 과신을 억제하는 대조 계약을 만든다 — 다음 작업.** 수량 경로를 고정한 채 시간 head를 원래 상태에 가깝게 제한하거나, 시간 전용 이력 경로를 검토합니다. 별도 Train 내부 시간 순서 분할로 선택하고 원래 Validation에서 확인해야 하며, 단순한 sigma-floor 상향을 해결로 단정하지 않습니다. 이번 요청에서는 새 fit을 시작하지 않았습니다.
3. **외부 비교와 독립 평가로 Contribution을 검증한다 — 다음 작업.** 동일한 수정·선택 규칙을 S2P2 등 비교군에도 적용하는 공정한 대조와 이미 노출된 Test/새 독립 평가의 구분이 필요합니다.

**재현성과 증적 — 완료**

- [고정 시간 head 결과](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_time_generalization_diagnosis_20261006_v1/cached_head_results.json) · [시간 분포/이력 구간 결과](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_time_generalization_diagnosis_20261006_v1/context_results.json) · [진단 코드](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_time_generalization_diagnosis_20261006_v1/analysis_code/run_context_diagnosis.py).
- 원본 full130 archive SHA: `1dbf6f033948813b17b8c004f8e43beffc0d276af258b30734199d7b54eb7348`; contract: `f47310a5d2cafa97510f401e1484653023223bf3fa8ec4550dfecbb8ef806187`; 동결source123: `4e94229fae002dc678025ecf056862f832c64459969f403b3cb5ac8cea8ba7b0`.
- 원래 GPU torch2.11.0+cu130의 캐시를 로컬 CPU torch2.14.0으로 계산했습니다. E0 cache행 native NLL 최대 차이는2.842e-13, 고정 Validation endpoint 차이는1.776e-15입니다. 이것은 시간 head 캐시 산술 검증이며 전체 binary encoder CPU 재추론 감사가 아닙니다.
- 큰 수량 기준 Taxi3449/RAF200 초과와 full Train/Validation 수량 prediction byteSHA 불변 증거는 원래 head 완료 receipt를 유지합니다. 독립 calibration이나 새 독립 Test가 아닙니다.

단위/정합/실패 gate 중심 합성 계약 테스트49개 PASS(시간 head12·이력 구간7·후속 연결30).
