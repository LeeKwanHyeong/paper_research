다음 Intermittent class imbalance와 mark confusion 분석 결과를 Notion에 정리해주세요.

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- Model Enhancement 분석이므로 `2. Confirm and Refine Topic`에는 작성하지 않습니다.
- 같은 제목의 페이지가 있으면 업데이트하고, 없을 때만 새 페이지를 만듭니다.
- `TitanTPP V3: Mark-Conditioned Value Head`와
  `TitanTPP V3c Detached Value-Encoder Route Smoke And Intermittent Screening e50`
  페이지를 연결합니다.

## 페이지 제목

- `Intermittent Mark Imbalance And Confusion Diagnostic: V2-V3c`

## 작성 원칙

- 연구자가 실험 결과를 직접 정리한 기술 노트처럼 간결하게 작성합니다.
- 아래 수치와 local artifact에서 확인된 사실만 사용합니다.
- confirmed finding과 interpretation을 구분합니다.
- class imbalance가 V3 regression의 단독 원인이라고 쓰지 않습니다.
- tail class 수치에는 support가 작다는 제한을 같이 적습니다.
- `획기적인`, `강력한`, `유의미한`, `주목할 만한`, `명확히 입증`,
  `종합적으로` 같은 홍보성·상투적 표현은 사용하지 않습니다.
- 결론을 반복하거나 모든 문단을 요약하지 않습니다.
- 상태는 `completed`로 기록합니다.

## 분석 범위

- 분석일: `2026-07-12 KST`
- dataset: `intermittent`
- split: fixed chronological train/validation/test
- target grain: next-event target; 각 series의 첫 event 제외
- variants: TitanTPP V2, V3a, V3b, V3c
- screening condition: seed `42`, e50
- checkpoint selection: `best_val_nll`
- model evidence: validation/test `mark_confusion_best_val_nll.csv`
- 실행 실험이 아니라 완료 artifact의 local diagnostic이므로 GPU/tmux 실행 정보는 없음

## Source Validation

- fixed split actual target count: train `136,256`, validation `41,901`, test `41,344`
- 모든 variant의 validation/test confusion true-class count가 fixed split과 일치
- confusion에서 재계산한 accuracy가 leaderboard accuracy와 모두 일치
- raw rows가 아니라 실제 next-event target 기준으로 class distribution을 계산

## Fixed-Split Distribution

| Split | Targets | Majority share | Marks 0-2 | Marks 4+ | Effective classes | TV vs train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | `136,256` | `39.24%` | `86.60%` | `5.62%` | `4.31` | `0.0000` |
| validation | `41,901` | `41.18%` | `87.19%` | `4.63%` | `4.18` | `0.0233` |
| test | `41,344` | `41.67%` | `87.39%` | `4.74%` | `4.17` | `0.0253` |

Test class support:

| Mark | Count | Share |
| ---: | ---: | ---: |
| 0 | `17,230` | `41.675%` |
| 1 | `11,954` | `28.914%` |
| 2 | `6,946` | `16.801%` |
| 3 | `3,253` | `7.868%` |
| 4 | `1,117` | `2.702%` |
| 5 | `426` | `1.030%` |
| 6 | `215` | `0.520%` |
| 7 | `91` | `0.220%` |
| 8 | `57` | `0.138%` |
| 9 | `39` | `0.094%` |
| 10 | `16` | `0.039%` |

작성 포인트:

- marks `0-2`가 test target의 `87.39%`를 차지해 imbalance가 큽니다.
- validation/test의 train 대비 TV distance는 `0.0233/0.0253`으로 drift는 작습니다.
- 모든 variant가 동일 target으로 평가되므로 imbalance만으로 variant 차이를 설명할 수 없습니다.

## Held-Out Test Diagnostics

| Variant | Accuracy | Balanced acc. | Macro F1 | Adjacent acc. | Adjacent error share | Mark MAE | Pred mark-0 share | Signed bias |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V2 | `54.460%` | `36.400%` | `0.3733` | `93.924%` | `86.658%` | `0.5216` | `42.529%` | `-0.0808` |
| V3a | `53.437%` | `41.703%` | `0.4329` | `91.900%` | `82.604%` | `0.5540` | `59.443%` | `-0.2491` |
| V3b | `53.367%` | `42.697%` | `0.4448` | `92.292%` | `83.470%` | `0.5507` | `57.617%` | `-0.2473` |
| V3c | `53.674%` | `36.373%` | `0.3701` | `92.175%` | `83.110%` | `0.5483` | `52.545%` | `-0.1530` |

## Mark 0/1 Boundary

Head-class recall:

| Mark | True share | V2 recall | V3a recall | V3b recall | V3c recall | V3c contribution vs V2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `41.675%` | `69.41%` | `85.37%` | `83.66%` | `78.80%` | `+3.916%p` |
| 1 | `28.914%` | `47.62%` | `23.06%` | `25.97%` | `29.23%` | `-5.316%p` |
| 2 | `16.801%` | `40.24%` | `32.72%` | `34.02%` | `43.74%` | `+0.588%p` |
| 3 | `7.868%` | `51.25%` | `54.35%` | `48.11%` | `44.79%` | `-0.508%p` |
| 4 | `2.702%` | `20.05%` | `23.10%` | `28.20%` | `40.47%` | `+0.551%p` |

V3c 주요 confusion:

| True -> Pred | Count | Within true | Share of all errors |
| --- | ---: | ---: | ---: |
| `1 -> 0` | `6,713` | `56.16%` | `35.05%` |
| `0 -> 1` | `2,717` | `15.77%` | `14.19%` |
| `2 -> 1` | `1,695` | `24.40%` | `8.85%` |
| `1 -> 2` | `1,515` | `12.67%` | `7.91%` |
| `2 -> 0` | `1,372` | `19.75%` | `7.16%` |
| `3 -> 2` | `1,222` | `37.57%` | `6.38%` |

Confirmed finding에 아래를 반영합니다.

- V3a/V3b는 overall accuracy가 낮아졌지만 balanced accuracy와 macro F1은 V2보다 높습니다.
- 이는 전체 class가 같이 나빠진 결과가 아니라 일부 tail recall을 얻고 high-support mark `1`을 잃은 macro/micro trade-off입니다.
- V3c의 `1 -> 0` confusion은 V2 `40.11%`에서 `56.16%`로 증가했습니다.
- V3c는 mark `0`에서 `+3.916%p`를 얻지만 mark `1`에서 `-5.316%p`를 잃어 net accuracy가 `-0.786%p`입니다.
- 대부분의 오류가 인접 mark에서 발생합니다. V2 error의 `86.66%`, V3c error의 `83.11%`가 distance 1입니다.

## Interpretation And Decision

Interpretation에는 아래처럼 제한해서 적습니다.

- class imbalance는 mark `0/1` 경계를 민감하게 만드는 조건으로 볼 수 있지만, architecture/optimization이 경계를 어느 방향으로 옮겼는지는 별도 문제입니다.
- hard-prediction confusion만 있으므로 probability calibration이나 class-conditional confidence의 원인은 아직 확인하지 못했습니다.
- raw inverse-frequency weighting은 support `16`인 mark `10` 등의 gradient를 과도하게 키울 수 있어 첫 구조 변경으로 선택하지 않습니다.

Decision:

- Intermittent baseline은 V2 유지
- V3c multi-seed 중단 결정 유지
- 추가 encoder gradient detachment는 진행하지 않음
- Intermittent 다음 모델 강화는 V5 ordinal marker objective를 우선
- 첫 V5 prototype은 standard marker CE를 유지하고 작은 ordered-distance auxiliary를 추가
- capped effective-number 또는 logit adjustment는 별도 prior-correction ablation으로 분리
- V5 gate에 overall accuracy, marker NLL, balanced accuracy, macro F1,
  mark `0/1` recall, adjacent error share 포함
- Taxi V3b confirmed decision과 V4 time-head 트랙은 변경하지 않음

## Local Evidence

```text
/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/analyze_intermittent_mark_diagnostics.py
/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_inter_mark_diagnostics_0712/diagnostic_manifest.json
/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_inter_mark_diagnostics_0712/report.md
/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_inter_mark_diagnostics_0712/data/
/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_inter_mark_diagnostics_0712/plots/
```

페이지 마지막에는 다음 작업을 한 줄로 남깁니다.

```text
Next: V5 ordinal marker objective의 loss contract와 acceptance gate 설계
```
