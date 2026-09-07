# Frozen-B raw-affine calibration: 5080 train-only 결과

## 결론

공통 식 `q_cal = max(0, a * q_hat + b)`는 **세 데이터셋 공통 개선 후보로
기각**됐다. Frozen-B가 가진 전역 수량 편향을 어느 정도 줄여 raw RMSE를 낮추는
효과는 있었지만, 사전에 고정한 train-only 필요조건을 세 데이터셋 어느 곳에서도
모두 만족하지 못했다. 따라서 controller는 validation prediction을 읽기 전에
정상 종료했고, identity인 `a=1, b=0`을 최종 상태로 저장했다.

이 결과가 기각하는 범위는 global monotone affine output calibration이다. 이번
후보는 encoder와 memory를 완전히 동결한 output-interface ablation이므로,
Hard-LMM Backbone의 개선 가능성 자체를 기각하는 결과로 해석하지 않는다.

## 사전 판정 기준

Canonical train series를 SHA-256 규칙으로 두 fold에 분리하고, 한 fold에서 적합한
두 파라미터를 다른 fold에 적용했다. 데이터셋마다 다음 네 조건을 모두 충족해야
validation으로 진행하도록 계약했다.

- 양방향 fold에서 raw MSE가 각각 엄격히 개선된다.
- 합산 out-of-fold raw RMSE가 최소 1% 개선된다.
- body(`<= train p95`) MAE 악화가 2% 이하다.
- `> train p99` MAE 악화가 2% 이하다.

한 데이터셋이라도 실패하면 공통 후보를 기각하고 validation, baseline 공정성
비교, 추가 seed와 held-out test를 실행하지 않는다.

## Train-only 결과

양수는 개선 또는 악화를 뜻하는 열 이름의 방향을 따른다. MAE 변화는 calibrated
값이 커졌을 때 양수이며, 음수는 개선이다.

| Dataset | Train targets | 양방향 fold 개선 | Raw RMSE (B → affine) | RMSE 개선 | 전체 MAE (B → affine) | Body MAE 악화 | `>p99` MAE 악화 | 판정 |
| --- | ---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| Intermittent | 393,824 | 예 | 1.355153 → 1.345435 | 0.717% | 0.562314 → 0.566282 | 1.337% | 1.906% | 기각 |
| Taxi | 38,393 | 아니요 | 87.948217 → 86.526355 | 1.617% | 26.564959 → 26.628366 | 3.651% | -4.721% | 기각 |
| Instacart | 1,991,192 | 예 | 5.637565 → 5.617745 | 0.352% | 3.891614 → 3.940484 | 2.185% | -2.626% | 기각 |

- Intermittent는 두 fold의 MSE 방향과 MAE guardrail은 통과했지만, RMSE 개선이
  최소 1%에 못 미쳤다.
- Taxi는 합산 RMSE가 1% 넘게 개선됐지만 fold 0에서는 RMSE가 1.748% 악화됐고
  fold 1에서는 2.177% 개선됐다. Body MAE도 3.651% 악화돼 두 조건을 실패했다.
- Instacart는 두 fold 모두 MSE가 개선됐지만 합산 RMSE 개선이 0.352%에
  그쳤으며 body MAE가 2.185% 악화됐다.

세 데이터셋에서 전체 MAE도 각각 0.706%, 0.239%, 1.256% 악화됐다. 전역
affine 보정은 평균적인 squared error와 편향을 조금 줄이는 대신 빈도가 높은
body 구간의 absolute error를 늘렸다. Taxi의 상반된 fold 방향까지 고려하면,
필요한 잔차 보정은 데이터셋 전체에 일정한 상수·기울기로 표현되지 않는다.

## 실행 범위와 무결성

- 실행 장비: **RTX 5080 한 대만 사용**
- 실행 source: `1ba9a439d003cad901a80ef05b9a6b327b4cff2a`
- 실행 source tree: `5470142cbe95b3c47a0d70b1b25c6f7dd9377087`
- source manifest SHA-256:
  `5a9ca8cf32c051ad61218981435c2623768cf14745405ef321fb5142cbccd8fa`
- 계약 SHA-256:
  `6fa3557c5d725755e759a0f231ce01e878a20c4058232e62fb17f94965116f2a`
- 최종 실행 시각: 2026-09-07 10:10:45–10:12:38 KST, 약 112초
- CUDA real-batch train smoke: 3/3 통과
- full train inference: 3 datasets, 211,065 series, 2,423,409 targets
- Source model은 eval mode와 `requires_grad=False`를 유지했고, 실행 전후 state
  hash가 일치했으며 source gradient가 생성되지 않았다.
- Identity quantity 출력과 복사된 per-event Time NLL은 정확히 일치했다.
- Validation smoke, full validation, RMTPP/THP 공정성 비교, 추가 seed와 held-out
  artifact는 생성되지 않았다.

독립 검산은 source manifest의 1,634개 파일, 36개 실행 artifact, NPZ 내부 배열의
dtype·shape·SHA-256, target 수·identity·quantity 경계, checkpoint와 model-state
hash, gate 산식과 identity fallback을 다시 확인했다. 결과는
`PASS_NO_BLOCKING_INTEGRITY_FINDING`이다.

별도 read-only 교차검산에서는 원본 train 데이터에서 target count·identity와
quantity SHA를 다시 만들고, 저장된 fold 배정·계수로 OOF prediction hash를
정확히 복원했다. 로컬 BLAS에서 OLS를 새로 적합했을 때에는 마지막 비트 차이만
있었고 metric 차이는 최대 `1.5e-15`였다.

- 기계 판독 감사 결과: [`independent_audit.json`](./independent_audit.json)
- 재현 가능한 검산기: [`independent_verify.py`](./independent_verify.py)
- 원 실행 artifact: [`search_artifacts/frozen_raw_affine_calibration_v1_1ba9a43_5080`](../../../search_artifacts/frozen_raw_affine_calibration_v1_1ba9a43_5080)
- 사전 계약: [`frozen_raw_affine_calibration_v1.md`](../../contracts/frozen_raw_affine_calibration_v1.md)

현재 checkout에서 별도 결과 파일로 검산하려면 다음을 실행한다.

```bash
python paper/results/frozen_raw_affine_calibration_20260907/independent_verify.py \
  --output /tmp/frozen_raw_affine_independent_audit.json
```

## 실행 중 발견한 문제와 수정

최종 수치는 `1ba9a43` source로 수행한 깨끗한 실행에서만 가져왔다. 그 전에 나온
세 번의 실행 오류와 한 번의 train-only preflight 실패는 validation에 접근하기
전에 중단됐고, 각 수정은 테스트와 독립 commit으로 남겼다.

| Source | 발견 내용 | 수정 |
| --- | --- | --- |
| `0852b6a` | Git archive에 빈 `sample_data/` root sentinel이 빠져 import 전 중단 | runtime empty-directory 계약과 manifest 검증 추가 |
| `39c657b` | PyTorch 2.11이 CUDA peak-memory reset의 device 객체 인수를 거부 | 정수 CUDA index 사용 |
| `29fb906` | 성공 상태 기록에서 지역 변수 이름 오류 | controller context의 status path 사용 |
| `f649415` | 일부 train quantity 경계가 canonical `nearest` quantile과 불일치 | 경계 재계산 및 GPU prediction 전 검증 추가 |

## 해석 한계와 다음 진단

Series-disjoint fold는 calibration parameter가 다른 series로 전달되는지 확인한다.
Frozen-B checkpoint 자체는 전체 train split으로 이미 학습됐으므로 이 fold를
독립적으로 재학습한 Backbone 실험으로 해석할 수 없다. 알려진 validation
결과도 이번 후보 판정에 사용하지 않았다.

일부 result JSON의 참조 경로는 5080의 절대 경로를 보존한다. 독립 검산기는
SHA가 일치하는 로컬 mirror를 직접 읽지만, 기존 summarizer를 로컬에서 그대로
재실행하려면 경로 remapping이 필요하다. 이 이식성 한계는 저장된 수치나 gate
판정에는 영향을 주지 않는다.

다음 로컬 진단은 이미 동기화한 train cache만 사용해, 잔차 보정의 방향이
관측 가능한 causal history 또는 quantity regime에 따라 달라지는지 확인하는
것이다. 두 fold와 세 데이터셋에서 하나의 공통 조건 변수가 일관된 residual
MSE 개선을 보이고 body MAE 방향과 충돌하지 않을 때만 Backbone/readout 후보를
정의할 근거가 생긴다. 그 전에는 추가 5080 학습을 시작하지 않는다.
