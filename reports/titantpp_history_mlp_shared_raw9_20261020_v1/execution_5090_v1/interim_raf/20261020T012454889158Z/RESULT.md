# 원본 RAF: MLP16과 공유형의 잠정 Validation 비교

학습 3개는 성공 종료했지만, 새 공유형 checkpoint의 별도 native Validation 재현과 최종 회수는 아직 전이다. 2026-10-20 10:24:54 KST에 수집한 학습 기록을 사용했다.

선택 기준은 전체 Validation raw 수량 RMSE의 최초 strict finite 최소 epoch다. MAE는 같은 epoch 값이며, 개별 최소 MAE를 따로 고르지 않았다. 원본 데이터·시간 가중치1·시간/수량 log1p·log 수량 MSE 조건이다.

| 구조 | RMSE 평균 ± 표본SD | MAE 평균 ± 표본SD |
|---|---:|---:|
| 기존 MLP16 | 34.083038 ± 0.086861 | 9.264040 ± 0.080681 |
| 공유형 Shared15 | 33.967318 ± 0.110590 | 9.246784 ± 0.064196 |

차이는 공유형−MLP16이며 음수는 개선이다.

| seed | 선택 epoch MLP16/공유형 | MLP16 RMSE | 공유형 RMSE | ΔRMSE (상대%) | MLP16 MAE | 공유형 MAE | ΔMAE (상대%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 9/9 | 33.983832 | 33.920463 | -0.063368 (-0.1865%) | 9.329766 | 9.303522 | -0.026244 (-0.2813%) |
| 52 | 5/5 | 34.119857 | 34.093622 | -0.026234 (-0.0769%) | 9.288356 | 9.259729 | -0.028628 (-0.3082%) |
| 62 | 19/19 | 34.145425 | 33.887869 | -0.257557 (-0.7543%) | 9.173997 | 9.177102 | +0.003105 (+0.0338%) |

3seed 평균 RMSE 차이 -0.115720 (-0.3395%), MAE 차이 -0.017256 (-0.1863%). RMSE는 3seed 모두 개선했고, MAE는42·52 개선/62 소폭 악화다.

현재는 소폭의 Validation 개선 근거가 있다. 최종 재현을 통과한 성능이나 Test 개선으로 해석하지 않는다.

기존 학습은5080/cuDNN91900, 신규는5090/cuDNN92000이다. MLP16 seed42의 저장 Validation은CPU 재현값이며 나머지 두 seed는5080 CUDA 값이다. 보정 파라미터 수도24576→9600으로 달라, 공유 자체와 용량 감소 효과를 분리할 수 없다.

**남은 검증 — 다음 작업**

- 기존 진행 중인 원본 Taxi·Intermittent 학습을 완료한다.
- 선택 checkpoint9개의 native Validation 재현과 원본 회수를 검증한 뒤 최종 3seed 비교를 확정한다.

## 근거

- 기준선: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_history_mlp_shared_raw9_20261020_v1/BASELINE_REFERENCES.json (SHA256 23311c5ae5918dc2cd51a47e720c26dec87711936b3bd2d1db275a6a0071d8d4)
- 학습 관측: /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_history_mlp_shared_raw9_20261020_v1/execution_5090_v1/observations/20261020T012455297188Z_5090.json (SHA256 edeba5de8440a57da99c9f25430b172cd818d74078f493885d3e8b329b532076)
- 기존 RAF 기준선 원본 연결 28개 SHA를 다시 검증했다. 추가 원격 호출·추론·학습은0회다.
