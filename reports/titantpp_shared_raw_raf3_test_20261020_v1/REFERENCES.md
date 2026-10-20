# 원본 RAF 저장 Test 비교군 — 검증 완료

MLP16 3개와 외부6종×3seed=18개, 총21개 저장 결과만 재사용했습니다. 재학습·재추론·원격 호출·기존 파일 변경은 없습니다.

| 모델 | RMSE 평균±표본SD | MAE 평균±표본SD |
|---|---:|---:|
| titantpp_history_mlp_width16 | 40.247819 ± 0.168908 | 10.148947 ± 0.090736 |
| rmtpp | 40.463166 ± 0.241581 | 9.906347 ± 0.037018 |
| thp | 40.555091 ± 0.295255 | 10.122950 ± 0.230475 |
| nhp | 42.176330 ± 0.300481 | 9.969145 ± 0.093177 |
| sahp | 41.049344 ± 0.156112 | 10.261643 ± 0.103236 |
| s2p2_matched_head | 39.772935 ± 0.160914 | 10.095805 ± 0.145493 |
| attnhp_matched_head | 40.203624 ± 0.204402 | 9.983934 ± 0.028442 |

모든 표본SD는 ddof=1입니다. 큰수량(q>200) RMSE·MAE 및 전체/큰수량 recorded positive-integer TimeNLL, 모든seed 원문 경로·SHA·수치는 REFERENCES.json에 보존했습니다.

## 검증 범위

- 원본 full RAF parquet와 split manifest SHA를 확인했습니다. Test5226개·Validation6690개 hard_lmm 대상/수량/시간 지문을 모델 추론 없이 재계산했습니다.
- 모든21개 receipt·prediction part·checkpoint·학습 history를 검증하고, 저장 예측109746행의 target/truth digest와 원본 수량/시간/순서가 일치함을 확인했습니다.
- 전체Validation RMSE의 최초 유한 최소epoch를 다시 계산해 동일 checkpoint와 연결했습니다. MAE는 그 checkpoint의 값을 사용합니다.
- 공통 log 수량손실+시간NLL 가중치1을 학습history 등식으로 검산했습니다.
- 기존 외부모델은 프로젝트의 공통 수량·시간head로 적응된 비교입니다. 원논문 고유head 결과로 해석하지 않습니다.

## 한계와 다음 작업

- 기존MLP16 seed42는CPU torch2.14 저장평가, 나머지기준선은5080/cuDNN91900이며 새Shared15는5090/cuDNN92000입니다. Runtime 차이를 함께 보고해야 합니다.
- Shared15 Test 결과는 아직 포함하지 않았습니다. 새후보 RAF42/52/62 epoch9/5/19의 Validation 재현 통과 후 Test를 비교합니다.
- 반복 Test 참고 후속 실험이므로 독립 일반화·통계 유의성으로 확대하지 않습니다.
