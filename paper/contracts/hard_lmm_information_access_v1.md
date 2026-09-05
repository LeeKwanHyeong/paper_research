# Hard-LMM 단계별 예측 정보 접근성 진단 계약

**진행 중 / 로컬 — 입력 이력부터 수량 readout까지 정보 접근성을 비교한다.**
이 진단은 관측 이력의 제한된 요약 정보가 고정된 보조 예측기로 얼마나 복원되는지, 그리고 그 정보가 기존 수량 예측의 잔차를 설명하는지 확인한다. 정보의 완전한 소실이나 특정 구조 변경의 성능 개선을 증명하는 실험은 아니다.

## 현재 기준선과 범위

- `paper_research/master`, 기준 `e8fdd11`. Original Hard-LMM과 separate-key의 기존 seed42 checkpoint를 CPU에서 동결한다. 실패한 elapsed-age 후보를 재학습하지 않는다.
- 기존 query 진단의 동일한 train target 8,192개씩과 series 단위 두 분할을 재사용한다. Taxi는131series, Instacart는7,848series다. 모든 모델은 이미 전체 train으로 학습됐고 validation으로 checkpoint가 선택됐다.
- Taxi의 모든 train series가 기존 진단에 사용됐다. 이번 분석은 회고적 train 내부 진단이며, 새 표본·일반화·held-out 성능의 근거로 주장하지 않는다.
- 새로운 추출·보조 예측기 적합 전에 [기계 판독 계약](hard_lmm_information_access_v1.json), 대상 cache·checkpoint hash와 분석 기준을 고정한다. 모델 파라미터·loss·head·데이터·기존 결과는 변경하지 않는다.

## 관측 위치와 입력 요약

실제 encoder 입력, layer1과layer2 출력에서 마지막 관측 사건 상태와 관측 사건 평균을 각각 추출한다. 이어서 최종 local 상태 `h`, 검색된 memory `r`, 실제 결합 `h+r`, 두 상태를 나란히 둔 `[h,r]`를 비교한다. Target와 padding은 모든 pooling에서 제외한다.

입력 요약 F는12개로 고정한다: log 이력 길이, 평균·마지막·표준편차 log수량, 최근3개 평균 log수량, 마지막−평균 log수량, 정규화 사건순서에 대한 수량 기울기, 내부 log간격의 평균·마지막·표준편차, log 내부 총경과시간, 기존 시간배치 차이 J. 첫 retained gap은 시간 요약에서 제외한다. H1의 기울기·내부 시간 요약은0, H≤2 또는 span0의 J는0이다.

F는 원시 이력 전체가 아니다. Mean pooling은 순서를 잃을 수 있고, 마지막 사건의 encoder 입력은 그 사건만 본다. 서로 다른 관측 범위를 비교해 입력 단계의 정보 소실을 주장하지 않는다.

## 보조 예측기와 대조군

- Label: `log1p(다음 수량) − softplus(동결 모델 수량 logit)`. Target 수량은 label·보고에만 쓰며 target gap은 쓰지 않는다.
- 보조 예측기의 학습과 표준화는 반대 series 분할에서만 한다. 전체 데이터 표준화, validation 선택, hyperparameter 탐색은 없다.
- 선형 ridge와, 같은 입력에 label과 무관하게 고정한 tanh random feature128개를 더한 ridge를 각각 사용한다. 두 예측기의 ridge는 sum-SSE 기준1이며 절편은 벌점이 없다. 행렬 계산은float64다.
- 반대 분할 평균 잔차를 상수 대조군으로 둔다. 추가 변수의 수와 같은 IID noise feature를 대조군으로 넣어 차원 증가 효과를 함께 확인한다. 같은 차원에는 같은 random projection을 사용한다.
- 단계별 F 복원은 별도 선형 multi-output ridge의 OOF R²로 보고한다. 상수 특성으로 분모가0이면 R²는 미정의로 표시한다. 복원 저하만으로 유용한 정보 부족을 판정하지 않는다.

## 사전 판정 기준

| 질문 | 비교 | 해석 가능 범위 |
| --- | --- | --- |
| Encoder 상태에 이력 요약을 더하면 도움이 되는가 | `[h,F]` vs `h`, `[h,noise12]` | 고정 decoder에서 이력 특징의 추가 접근성 |
| 최종 결합 상태에도 이력 요약이 필요한가 | `[h+r,F]` vs `h+r`, `[h+r,noise12]` | 최종 표현의 조건부 추가 접근성 |
| 분리된 두 상태가 덧셈보다 잘 읽히는가 | `[h,r]` vs `h+r`, `[h+r,noise64]` | 결합 표현의 접근성 단서 |
| 최종 상태에 현재 head가 활용하지 않은 잔차 신호가 있는가 | `h+r` vs 상수, `noise64` | Head·목적함수·선택법을 분리하지 못한 활용 단서 |

각 비교는 두 고정 decoder 모두에서 양쪽 대조군보다 pooled residual MSE1% 이상 개선, 두 분할 모두 개선, paired series bootstrap10,000회의 보수적 하위 분위수 `.05/32`에서 개선량 양수를 요구한다. 32는4개 질문×2dataset×2model×2decoder다. 이 범위는 고정된 OOF 오차의 조건부 안정성 검사이며, 정식 인과·일반화 유의성 검정이 아니다.

이력 추가 근거는 F 단독도 상수 대비1% 이상·양 분할 개선을 보여야 한다. Log residual MSE 개선과 body raw MAE의 방향이 충돌하면 구조 결함 근거로 승격하지 않는다. 두 데이터셋과 두 모델에서 동일 비교가 통과할 때만 공통 backbone 근거로 분류한다. Dataset별로 다른 승자를 조합하지 않는다.

Raw body MAE·RMSE·tail MAE는 별도로 보고한다. 보정 log수량은 보고용으로만0~20에 clip하고 건수를 남긴다. 진단 MSE는 clip 이전의 residual을 사용한다. 기존 성능 채택 기준과 혼용하지 않는다.

## 완료 조건과 다음 경계

- 추출 위치·공식 출력 동일성, target/padding 차단, hook 제거, short-history·finite 계산, series 분리·train-fold 통계, synthetic 정보 추가/제거 대조군, 판정 조건 테스트를 통과한다.
- 동일 표본의 두 모델·네 조합을 추출·분석하고 OOF 예측·hash·표·독립 감사 기록을 남긴다.
- 공통 단서가 나오면 추가 접근성을 보인 정보와 위치를 구체화한다. 단서가 없거나 decoder/지표 간 충돌이 있으면 어느 위치가 문제인지 아직 식별하지 못했다고 기록한다. 결과만으로 새 구현이나 5090 실행을 시작하지 않는다.

방법 참고: [중간층 probe](https://arxiv.org/abs/1610.01644)는 층별 접근성 측정의 출발점이며, [probe와 control task 연구](https://aclanthology.org/D19-1275/)는 보조모델의 용량·암기 효과를 경계한다. 여기의 수량 회귀·IID noise 대조군은 이 문헌의 분류·언어 control task를 그대로 재현한 것은 아니다.
