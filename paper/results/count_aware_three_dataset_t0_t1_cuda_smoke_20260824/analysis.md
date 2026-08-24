# 세 데이터셋 T0·TitanTPP-T1 CUDA/e1 Smoke 결과

## 판정

- **Runtime gate: PASS**
- **정식 e300 진입: Online Retail II time 안정성 확인 전 보류**
- 실행 서버: 5080, RTX 5080
- source revision: `3f10f15cfc281a9932ab07f423f711be4a19480c`
- held-out test: 사용하지 않음

## Artifact 확인 순서

1. `source_manifest.txt`와 여섯 `launch_contract.json`: source checksum, 데이터셋별 시간 단위, lookback, max sequence length와 T0/T1 역할이 계약과 일치했다.
2. `logs/run.log`: focused test `26 passed`, 여섯 CUDA model-test case finite, Traceback·NaN·Infinity 없음, 여섯 역할 실행 모두 `[complete]`를 기록했다.
3. `run_summaries.csv`: T0 15개와 T1 3개, 총 18개 run이 모두 `success`이며 checkpoint를 생성했다.
4. Held-out test summary: 생성되지 않았다. `cuda_model_test.json`만 파일명에 `test`가 포함되며 평가 데이터 artifact가 아니다.
5. `history.json`: 18개 모두 epoch 1 한 행, `train_all_finite=true`였다.
6. Scale-wise metrics: partial e1 smoke에서는 생성하지 않았다.
7. Plots: partial e1 smoke에서는 생성하지 않았다.

## 핵심 확인 사항

| Dataset | Context | TitanTPP-T0 MAE/RMSE | TitanTPP-T1 MAE/RMSE | 판정 |
| --- | --- | --- | --- | --- |
| Intermittent v2 | 520 weeks / 256 events | 23.2504 / 24.0409 | 23.2504 / 24.0409 | 실행 계약 통과 |
| Online Retail II | 8,760 hours / 256 events | 30.2700 / 106.0559 | 30.2692 / 106.0554 | 실행 통과, time 경고 |
| RAF Spare Parts | 84 months / 84 events | 4.5081 / 9.9604 | 4.5081 / 9.9604 | 실행 계약 통과 |

수치는 train과 validation을 각각 두 batch만 읽은 e1 결과다. 따라서 T0/T1의 성능
차이나 backbone 순위를 해석하지 않는다. 이번 단계에서 확인한 것은 dataset별
tail 상수, 공통 head, forward/backward, checkpoint와 validation summary 경로가
세 데이터셋에서 연결된다는 점이다.

## 발견된 위험

Online Retail II에서 TitanTPP-T0/T1 validation Time NLL이 약 `17,759.9`였고,
모든 backbone의 train gradient clipping 비율이 `1.0`이었다. T0와 T1의 Time NLL은
거의 같으므로 tail loss가 만든 문제라기보다 hourly delta-time과 기존
`legacy_clamped_rmtpp` time head의 scale 부적합 가능성이 높다. 이 상태로 e300을
시작하면 quantity 비교가 time loss 폭증에 가려질 수 있다.

## 다음 결정

Intermittent v2와 RAF는 정식 matched validation runner 준비를 이어갈 수 있다.
Online Retail II는 모델이나 quantity loss를 바꾸지 않은 채 train-only delta-time
분포, per-event Time NLL과 clipping을 먼저 감사한다. 공통 time head를 유지할 수
있는 단위 변환 또는 train-only scaling 계약을 확정한 뒤 세 데이터셋 e300 실행
여부를 결정한다.
