# Frozen-B Hard-LMM 검색 credit-assignment 필요조건 검사

## 판정

**공통 gate는 실패했다.** Intermittent만 두 fold의 모든 방향 조건을 통과했고,
Taxi와 Instacart는 raw squared error 방향에서 실패했다. 따라서 계약한
`all-64`, `temperature=1`, unit-strength quantity-only soft surrogate는 모델에
구현하거나 GPU에서 학습하지 않는다.

이 판정은 soft addressing 전체가 불가능하다는 뜻이 아니다. 이번에 고정한 한
후보가 공통 Backbone 개선 후보가 되기 위한 필요조건을 충족하지 못했다는 뜻이다.

## 검사 범위

- 기준선은 seed 42의 B, 즉 기존 Hard-LMM 구조와 raw-RMSE checkpoint
  selector를 사용한 checkpoint이다.
- B를 학습한 revision `f75243473adc25d622319dbca9bda7e076d8240f`를
  별도 경로에 복원했다. 실행 중 로드된 project Python 파일 48개가 모두 해당
  경로 안에 있고 pinned source manifest의 hash와 일치함을 확인했다.
- 데이터셋마다 train target 4,096개를 사용했다. 각 fold는 2,048개이며 series
  교집합은 없다.
- B checkpoint는 전체 train split으로 이미 학습되었다. 따라서 두 fold는
  out-of-fold 성능 평가가 아니라 서로 다른 series 집단 사이의 gradient 방향
  안정성을 확인하는 용도다.
- validation과 held-out test row는 materialize하지 않았고 optimizer step도
  수행하지 않았다.

## 구조 확인

현재 hard top-4는 cosine score로 index를 고른 뒤 선택된 prototype value의
산술평균만 loss에 연결한다. 따라서 encoder에는 직접 residual 경로의 gradient가,
선택된 memory row에는 value gradient가 전달되지만, **순위 선택을 통한 query/key
addressing gradient는 세 데이터셋 모두 정확히 0**이었다.

Forward 값이 B와 bitwise하게 같은 zero-valued soft branch를 붙이면 query와 key
addressing gradient는 모두 finite하고 0이 아니었다. Quantity prediction도 B와
bitwise하게 같았으며, 기존 top-4 index와 memory residual은 cache와 정확히
일치했다.

## 두 fold 방향 결과

아래 값은 source fold의 추가 gradient와 반대 fold metric gradient 사이의 cosine이다.
양수이면 그 추가 gradient의 음의 방향으로 아주 작게 이동할 때 해당 metric이
감소하는 1차 방향이다. 괄호 안은 동일 score·value 집합의 대응만 순환 이동한
shuffled control이다.

| 데이터셋 | 방향 | log-MSE 정상 (shuffled) | raw squared 정상 (shuffled) | body MAE 정상 | 판정 |
|---|---|---:|---:|---:|---|
| Intermittent | fold0→fold1 | 0.0392 (-0.0156) | 0.00489 (-0.00674) | 0.0186 | 통과 |
| Intermittent | fold1→fold0 | 0.0249 (-0.00904) | 0.00499 (0.00014) | 0.0129 | 통과 |
| Taxi | fold0→fold1 | 0.0772 (-0.00040) | -0.0130 (-0.00359) | -0.0184 | 실패 |
| Taxi | fold1→fold0 | 0.1151 (-0.00041) | 0.00787 (-0.00542) | 0.00324 | clip 후 실패 |
| Instacart | fold0→fold1 | 0.2628 (-0.0378) | -0.2531 (0.0346) | 0.2621 | 실패 |
| Instacart | fold1→fold0 | 0.2776 (-0.0341) | -0.2317 (0.0315) | 0.2674 | 실패 |

실제 B runner와 같은 batch size 128 및 global norm clip 1.0을 적용한 상대 update도
검사했다. Intermittent는 계속 통과했지만 Taxi의 raw 방향은 양방향 음수가 됐고,
Instacart의 raw 방향도 양방향 음수를 유지했다. 따라서 strict shuffled-control
조건이나 clip 조건을 제외해도 공통 실패 결론은 바뀌지 않는다.

## Gradient 크기

Soft branch의 추가 gradient norm은 기존 hard log-quantity gradient norm에 비해
다음과 같았다. 범위는 두 fold를 나타낸다.

| 데이터셋 | 전체 trainable parameter | Query encoder | Key memory |
|---|---:|---:|---:|
| Intermittent | 0.0336–0.0349% | 0.0382–0.0454% | 0.70–0.75% |
| Taxi | 0.129–0.177% | 0.184–0.191% | 6.05–7.16% |
| Instacart | 0.202–0.219% | 0.317–0.334% | 2.09–2.19% |

Key 역할에는 구분 가능한 gradient가 생기지만 전체 최적화 경로에서는 매우 작다.
Intermittent도 통과 방향의 cosine이 0.025–0.039 수준이므로 큰 성능 향상을 예고하는
증거로 해석할 수 없다.

Instacart에서는 현재 hard log-MSE gradient 자체와 raw squared-error gradient의
fold 간 cosine도 -0.955와 -0.966이었다. Soft branch는 log-MSE 방향을 더 잘
전달하지만 raw-scale 목표에는 반대 방향이다. 이 데이터셋의 실패는 단순히 hard
top-4의 미분 불가능성만 해소해서 raw RMSE가 개선되기 어렵다는 근거다.

## 한계

- Frozen seed42 checkpoint 한 점의 `eval()` 모드에서 계산한 국소 1차 진단이다.
  Dropout과 AdamW가 포함된 처음부터의 학습 결과를 직접 예측하지 않는다.
- Raw squared gradient는 heavy tail에 민감하다. 다른 seed나 cohort bootstrap은
  이번 계약 범위에 포함하지 않았다.
- Shuffled control은 결과를 보기 전에 고정한 단일 cyclic derangement다. 여러
  permutation의 null distribution보다 약한 대조군이다.
- Time NLL 방향은 기록했지만 사용자가 정한 필요조건 gate에는 포함하지 않았다.
  Instacart fold1→fold0과 Taxi fold1→fold0의 clip 후 방향은 음수였다.

정식 수치는 [analysis.json](analysis.json), 판정은
[evidence_decision.json](evidence_decision.json), 실행·hash 증적은
[execution_manifest.json](execution_manifest.json)에 저장했다.
