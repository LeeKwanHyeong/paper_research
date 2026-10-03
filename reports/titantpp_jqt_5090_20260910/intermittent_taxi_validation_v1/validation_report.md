# Intermittent·Taxi J/Q/T 검증 결과

**종합 판단 — 조건과 한계를 명시하면 공유 가능(Share with caveats)**

두 데이터셋·6개 학습의 완료 증적과 비교 계산은 검증을 통과했다. 사전 지정된 최적 validation selector 비교에서는 두 데이터셋 모두 Q의 수량 RMSE와 T의 시간 loss가 J보다 나빴다. Intermittent의 수량 MAE만 1.07% 좋아졌다. 다만 **Taxi는 동일 최종 epoch120에서 Q·T가 J보다 좋으므로, 단독 학습이 언제나 나쁘다고 해석해서는 안 된다.**

대상은 이번 5090에서 재학습한 static B의 J(공동)·Q(수량 전용)·T(시간 전용)다. 과거 B 결과표나 dual-timescale·RMTPP·THP 결과를 비교 분모에 섞지 않았다. 증적 기준 시각은 **2026-09-10 19:59:09 KST**다.

**학습 계약과 완료 증적 확인 — 완료**

공통 조건: seed42, arm당120 epoch, batch128, AdamW lr0.001·weight decay0.01·clip1, 수량 log-MSE 가중치1, legacy time cap300. 데이터셋 안에서 동일 초기화·batch 순서·표본 노출을 검증했다. 동일 step은 동일 계산 비용이나 각각 최적으로 조정한 학습률을 뜻하지 않는다.

| 데이터셋 | 완료 arm | train/epoch | validation | step/arm | 종료 KST |
|---|---:|---:|---:|---:|---|
| Intermittent | 3/3 | 393,824 | 86,285 | 369,240 | 18:06:30 |
| Taxi | 3/3 | 38,393 | 8,268 | 36,000 | 18:50:52 |

source revision은 `76426fd981e624144947fd582f794c2832e9fe05`다. 완료 receipt와 summary·wrapper·paired comparison SHA, history 일치, 전체120 epoch의 step·표본·순서, 활성 selector의 strict earliest minimum을 확인했다. 비활성 지표는 null로 보존했다. 원본 표본에서 성능을 다시 계산하거나 checkpoint를 다시 실행한 검증은 아니다.

**최적 validation checkpoint 비교 — 완료, 사전 지정된 주 비교**

수량은 J와 Q 각각 raw-RMSE selector로 고른 checkpoint를 비교한다. MAE도 그 동일 checkpoint에서 읽는다. 시간은 J와 T 각각 시간 selector를 적용한다. 수량·시간의 J checkpoint가 다르므로 하나의 모델이 동시에 달성한 성적으로 합치지 않는다.

| 데이터셋 | 지표 | J | Q 또는 T | 단독 학습의 J 대비 변화 | 선택 epoch J / 단독 |
|---|---|---:|---:|---|---|
| Intermittent | 수량 Raw RMSE | 1.499555 | Q 1.548389 | +3.26% · 악화 | 77 / 24 |
| Intermittent | 수량 MAE | 0.604340 | Q 0.597850 | -1.07% · 개선 | 77 / 24 |
| Intermittent | Legacy time loss | -3.566226 | T -3.520814 | +0.045412 · 악화 | 120 / 120 |
| Taxi | 수량 Raw RMSE | 85.606968 | Q 87.789343 | +2.55% · 악화 | 116 / 97 |
| Taxi | 수량 MAE | 27.470006 | Q 27.941478 | +1.72% · 악화 | 116 / 97 |
| Taxi | Legacy time loss | 1.365333 | T 1.366590 | +0.001257 · 악화 | 16 / 4 |

**동일 최종 epoch120 비교 — 완료, 보조 비교**

두 학습을 같은 최종 step에서 비교한다. 성능을 본 뒤 주 selector를 바꾼 결과가 아니다. 모든120 epoch의 대응 수치는 JSON에 보존했다.

| 데이터셋 | 지표 | J epoch120 | Q/T epoch120 | 단독 학습의 J 대비 변화 |
|---|---|---:|---:|---|
| Intermittent | 수량 Raw RMSE | 1.652868 | Q 1.718703 | +3.98% · 악화 |
| Intermittent | 수량 MAE | 0.619376 | Q 0.647233 | +4.50% · 악화 |
| Intermittent | Legacy time loss | -3.566226 | T -3.520814 | +0.045412 · 악화 |
| Taxi | 수량 Raw RMSE | 93.959049 | Q 91.671297 | -2.43% · 개선 |
| Taxi | 수량 MAE | 29.603272 | Q 28.757767 | -2.86% · 개선 |
| Taxi | Legacy time loss | 17.111530 | T 15.801643 | -1.309887 · 개선 |

Intermittent는 최종epoch에서 Q의 RMSE·MAE와 T의 시간 loss가 모두 J보다 나쁘다. 최적 수량 checkpoint에서의 MAE 개선은 최종epoch에서는 유지되지 않는다. Taxi는 최종epoch에서 Q의 RMSE·MAE와 T의 시간 loss가 모두 J보다 좋다. 그러나 Taxi의 J·T 시간 loss는 각각의 최적값보다 크게 높아져, 최적 성능 비교와 후반 학습 안정성을 함께 살펴야 한다.

**학습 중 clipping 집계 — 완료, 원인 해석에는 제한**

| 데이터셋 | J 횟수·비율 | Q 횟수·비율 | T 횟수·비율 | 전체 batch/arm |
|---|---:|---:|---:|---:|
| Intermittent | 9,179 · 2.4859% | 44 · 0.0119% | 8,895 · 2.4090% | 369,240 |
| Taxi | 21,998 · 61.1056% | 5,651 · 15.6972% | 13,970 · 38.8056% | 36,000 |

Intermittent의 J·T clipping은 거의 초기6 epoch에 집중된다(J epoch108의1건 예외). Taxi는 세 arm 모두120개 epoch 전체에서 clipping이 발생한다. Taxi 마지막10 epoch의 비율은 J95.57%, Q3.33%, T92.63%다. Taxi T도 후반 clipping이 높기 때문에 공동 학습만의 문제로 확정할 수 없다.

clipping은 실제 clip_grad_norm_ 호출 전 norm이 임계값1을 초과한 batch 횟수다. 각 arm의 학습 경로가 달라지므로 횟수 차이는 시간 gradient의 인과 효과나 실제 AdamW update 억제량이 아니다. 선택 checkpoint의 과거 train-only gradient probe와 이번 전체 학습 이력도 구분한다.

**해석 검토와 남은 한계 — 검토 완료**

- 중요한 해석 보정: 기존의 ‘단독 학습의 이득 없음’은 최적 selector의 주 비교 지표에 한정한다. Taxi의 최종epoch 이득과 Intermittent의 선택 checkpoint MAE 이득을 함께 제시해야 한다.
- 확인된 사실: 두 데이터셋 모두 주 selector 기준 수량 RMSE·시간 loss의 분리 이득은 없다. Taxi의 최종epoch에서는 비교 방향이 바뀐다.
- 해석: 현재 B에서 학습 간섭이 공통의 핵심 병목이고 완전 분리가 해결책이라는 주장은 지지되지 않았다. 그렇다고 간섭의 부재나 모든 분리 구조의 열세를 입증한 것은 아니다.
- 미확인: Body/tail MAE, 수량 구간·이력 길이별 오차, 구간별 task gradient 방향, 여러 seed의 재현성, held-out 성능, 별도 최적화 조건에서의 수렴 성능이다.
- Q와 T를 결합한 시스템의 용량·학습/추론 비용·동시 성능과 benchmark 우월성은 검증하지 않았다. J의 수량/시간 최적 checkpoint도 서로 다르다.
- validation은 checkpoint 선택과 기술적 비교에 사용했다. 표본수는 독립 반복실험 수가 아니며 통계적 유의성·보편적 우월성 주장은 하지 않는다. 시간 지표는 정규화된 Time NLL이 아니다.

**남은 작업 순서**

**Instacart 완료 결과를 같은 기준으로 통합 — 외부 작업 대기**

- 승인된 기존 학습을 시간별 scheduler로 관찰하고 완료 증적이 확보되면 주 selector와 동일 최종step 비교를 추가한다.

**두 데이터셋의 오차·후반 안정성 원인 확인 범위 확정 — 다음 작업**

- Intermittent는 수량 구간·이력 길이별 RMSE/MAE 차이, Taxi는 후기 시간 loss와 clipping의 동반 변화에 초점을 둔다. 필요한 표본별 예측과 학습 중 gradient가 없으므로 원인을 완료된 것처럼 보고하지 않는다.

이번 추출의 차단 요소는 없다. 새로운 평가 실행은 이 보고서에 포함하지 않았다. 기존 학습·원본 결과·scheduler는 변경하지 않았다.

**재현 및 원본 증적**

- [검증 결과 JSON](validation_results.json): 전체120 epoch 대응 수치, 지표 정의, clipping, 증적 SHA.
- [추출·검증 코드](build_report.py): 표준 라이브러리만 사용하며 원격·모델·원본 데이터에 접근하지 않는다.
- [19:59 원격 스냅샷](../monitor/20260910T105909Z_snapshot.json), [기존 완료 감사](../monitor/20260910T105909Z_check.json), [동결 실행 계약](../remote_receipts/suite_contract.json).

재계산 명령:

```bash
python3 /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_5090_20260910/intermittent_taxi_validation_v1/build_report.py
```
