다음 결과를 Notion의 기존 페이지에 업데이트해주세요. 새 페이지를 중복 생성하지
말고, 시작 시 작성한 계획과 실행 정보 아래에 결과 섹션을 추가합니다.

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- 기존 페이지: `TitanTPP V5a Intermittent Seed-42 e50 Validation Screening`
- Model Enhancement 결과이므로 `2. Confirm and Refine Topic`에는 작성하지 않습니다.

## 문체

- 실험 기록처럼 간결하게 작성합니다.
- 확인된 수치와 해석을 분리합니다.
- 과장된 표현, 의미 없는 수식어, 일반론적인 GPT 문장은 넣지 않습니다.
- V5a가 일부 quantity/NLL 지표를 개선했더라도 gate 실패 사실을 먼저 씁니다.
- held-out test 수치는 쓰지 않습니다.

## 상태 업데이트

- 상태: `completed; validation gate failed`
- 시작: `2026-07-12 21:16:02 KST`
- 종료: `2026-07-12 21:22:38 KST`
- 서버: `5090` (`192.168.0.71:22`)
- tmux: `inter_v5a_rps_e50_0712`
- 완료 epoch: `50/50`
- primary checkpoint: `best_val_nll`, epoch `30`
- NaN/Traceback/RuntimeError/ERROR/FAILED: 없음
- test-lock: 유지. held-out test artifact 내용은 읽지 않음

## 실험 조건

| Field | V2 reference | V5a |
| --- | --- | --- |
| dataset/split | Intermittent fixed | same |
| candidate | `small_lmm` | same |
| seed/epochs | `42 / 50` | same |
| lr/batch | `1e-3 / 128` | same |
| lookback/max sequence | `52 / 16` | same |
| value input/loss scope | `residual / target_only` | same |
| quantity loss | `hybrid` | same |
| value/gradient route | `shared/coupled/coupled` | same |
| marker loss | `ce` | `ce_rps` |
| ordinal weight | `0.0` | `0.10` |

## Validation 결과

아래는 두 모델 모두 `best_val_nll` checkpoint의 동일 validation split 결과입니다.

| Metric | V2 | V5a | Change | Gate |
| --- | ---: | ---: | ---: | --- |
| Normalized RPS | `0.035283` | `0.035371` | `+0.251%` regression | Fail |
| Mark MAE | `0.487411` | `0.527028` | `+8.128%` regression | Fail |
| Balanced accuracy | `42.664%` | `41.667%` | `-0.997%p` | Fail |
| Macro F1 | `43.302%` | `41.163%` | `-2.139%p` | Fail |
| Mark accuracy | `57.249%` | `54.820%` | `-2.430%p` | Fail |
| Adjacent accuracy | `94.377%` | `92.976%` | `-1.401%p` | Fail |
| Mark-0 recall | `75.543%` | `86.462%` | `+10.919%p` | Pass |
| Mark-1 recall | `49.616%` | `24.664%` | `-24.953%p` | Fail |
| Marker NLL | `0.991274` | `0.991668` | `+0.040%` regression | Pass |
| Time NLL | `4.675246` | `4.664064` | `-0.239%` improvement | Pass |
| Total NLL | `5.666520` | `5.655732` | `-0.190%` improvement | Pass |
| Quantity MAE | `3.060182` | `2.889382` | `-5.581%` improvement | Pass |
| Value MAE | `0.146300` | `0.130431` | `-10.847%` improvement | Pass |

## Scale-Wise Quantity Safety

validation share가 5% 이상인 bucket만 gate에 사용했습니다.

| Bucket | Share | V2 qty MAE | V5a qty MAE | Change | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| `1-9` | `88.666%` | `0.979752` | `1.031592` | `+5.291%` | Fail |
| `10-99` | `10.723%` | `9.318595` | `8.720051` | `-6.423%` | Pass |

## 해석

- V5a는 total/time NLL과 전체 quantity/value MAE를 개선했지만 ordinal RPS와
  mark MAE는 개선하지 못했습니다.
- mark-0 prediction share가 V2 `45.030%`에서 V5a `58.750%`로 증가했습니다.
  실제 validation mark-0 share는 `41.180%`입니다.
- 이 변화와 함께 mark-0 recall은 높아졌지만 mark-1 recall이 `24.664%`로
  하락했습니다. high-support mark 0/1 경계 문제를 해결하지 못한 결과입니다.
- learning curve에서도 quantity MAE 감소와 달리 RPS/mark MAE의 안정적인 개선은
  확인되지 않았습니다.

## 판정

- Ordinal benefit: `FAIL`
- Classification/task safety: `FAIL`
- Overall seed-42 validation gate: `FAIL`
- 사전 branch rule에서 benefit과 safety가 동시에 실패했으므로
  `lambda_ordinal=0.05` 또는 `0.20` 추가 screening은 실행하지 않습니다.
- V5a는 multi-seed validation으로 승격하지 않습니다.
- Intermittent baseline은 V2를 유지합니다.
- held-out test metric은 읽지 않았으며 이후 V5a 재튜닝 근거로 사용하지 않습니다.

## Artifact

```text
server: /home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_v5a_inter_short_e50_0712
local: /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v5a_inter_short_e50_0712
V2 reference: /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_v2_inter_validation_reference_v5a_0712
```

페이지 마지막 Next는 아래 문장으로 교체합니다.

```text
Next: V5a를 종료하고 V2 baseline에서 분리된 V5b class-prior correction의 설계 필요성을 검토한다.
```
