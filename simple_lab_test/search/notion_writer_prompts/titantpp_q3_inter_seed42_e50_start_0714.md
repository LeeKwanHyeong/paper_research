# TitanTPP Q3 Factorial Intermittent Seed-42 e50 Validation-Only Screening

Notion의 `5. Model Design Enhancement` 아래 기존
`TitanTPP Q3 Factorial Intermittent Seed-42 e50 Validation-Only Screening` 페이지를
업데이트한다. 상위 history에는 제목 2 `2026-07-14 | Q3 Actual-Data Integration`,
제목 3 `Step 2. Q3 Intermittent Seed-42 e50 Validation-Only Screening`으로 연결한다.

## 상태

- 상태: `완료 - V2 유지, Q3a/Q3b/Q3c 미승격`
- 실행 시작: `2026-07-15 08:15:07 KST`
- 실행 종료: `2026-07-15 08:46:16 KST`
- 완료 상태: Q2/Q3a/Q3b/Q3c 모두 `exit_code=0`
- artifact: 로컬 동기화 및 checksum 검증 완료

## 목적

Q2/Q3a/Q3b/Q3c를 Intermittent fixed split에서 같은 e50·seed-42 예산으로 비교한다.
Magnitude gradient 분리와 log2 auxiliary loss가 marker 성능과 수량 예측에 미치는 영향을
각각 분리하고, 두 방법을 함께 적용했을 때의 interaction을 확인한다.

이 실험은 validation-only screening이다. Q3a 또는 Q3b의 중간 결과로 Q3c를 생략하지
않고 네 Variant를 모두 실행한다.

## Factorial 계약

| Variant | Encoder로 전달되는 magnitude gradient | Log2 auxiliary | 확인할 효과 |
| --- | --- | --- | --- |
| Q2 | `coupled` | 없음 | 동일 revision에서 재현하는 control |
| Q3a | `detached` | 없음 | magnitude gradient 분리 효과 |
| Q3b | `coupled` | `log_huber` | log-domain 보조 loss 효과 |
| Q3c | `detached` | `log_huber` | 두 방법의 결합 및 interaction 효과 |

네 Variant는 위 두 축만 다르며 model size, data order, seed와 나머지 학습 설정은 같다.

## 고정 조건

| 항목 | 값 |
| --- | --- |
| 서버 / 환경 | `5090 / ai_env` |
| tmux session | `titantpp_q3_inter_e50_0714` |
| dataset / split | `intermittent / fixed` |
| model / candidate | `TitanTPP / small_lmm` |
| epochs / seed | `50 / 42` |
| learning rate / batch size | `1e-3 / 128` |
| lookback / max sequence | `52 weeks / 16` |
| decoder | `direct_raw_qty` |
| normalization | `causal_shrinkage_revin` |
| train loss scope / mode | `target_only / hybrid` |
| marker objective | plain CE, `lambda_ordinal=0` |
| loss weights | time `1.0`, magnitude `1.0`, raw quantity `0.25` |
| log auxiliary | weight `0.25`, Huber delta `1.0`, floor `1.0` |
| checkpoint selection | `best_val_nll`, `best_score`, `final` |
| artifact | `search_artifacts/model_enhancement_titantpp_q3_inter_seed42_e50_0714` |

## 실행 명령어

```bash
ssh 5090
/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s titantpp_q3_inter_e50_0714 \
  "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_q3_inter_seed42_e50_0714.sh"
```

## 결과

네 Variant 모두 정상 종료했고 validation artifact 기준으로 비교했다.

| Model | Total NLL | Raw qty MAE | Log2 qty MAE | Mark accuracy | 결정 |
| --- | ---: | ---: | ---: | ---: | --- |
| V2 | 5.6665 | 3.0602 | 0.5887 | 57.249% | 유지 |
| Q2 | 5.6709 | 2.7629 | 0.6837 | 55.168% | 미승격 |
| Q3a | 5.6605 | 3.3334 | 0.8757 | 55.178% | 미승격 |
| Q3b | 5.6341 | 2.6493 | 0.6867 | 54.963% | 미승격 |
| Q3c | 5.6659 | 3.3862 | 0.7380 | 55.853% | 미승격 |

- Q3a는 gradient 분리만으로 marker 성능을 회복하지 못했고 수량 오차가 커졌다.
- Q3b는 Total NLL과 raw qty MAE가 가장 낮았지만, NLL 개선은 time head에서 발생했다.
  Log2·1-9 수량 구간과 marker 기준을 통과하지 못했다.
- Q3c는 mark-0 편향과 mark-1 recall을 개선했지만 raw·short-history·log2 수량 오차가
  커져 균형 잡힌 후보로 보기 어렵다.
- Fresh Q2가 기존 Q2 결과를 재현하지 못해 Q3 효과를 인과적으로 확정하지 않았다.

최종적으로 Q3a/Q3b/Q3c는 모두 acceptance 기준을 통과하지 못해 미승격 처리하고
Intermittent 기준 모델은 V2로 유지한다. 현재 결론을 위해 full e50 재실행은 하지 않는다.
향후 Q3를 다시 검토할 때는 strict deterministic control을 적용한 Q2 e3 A/B 재현성
검증을 먼저 통과해야 한다.
