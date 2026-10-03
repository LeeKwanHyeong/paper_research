# 수량 단독 학습은 최적 RMSE를 개선하지 못했다

**통합 해석 — 검토 범위 내 공유 가능(Share with caveats)**

세 데이터셋의 결과를 합치면, **현재 B에서 시간·수량의 완전 분리를 다음 구조로 확정할 근거는 부족하다.** 최적 수량 checkpoint에서는 Q가 J보다 RMSE를 개선하지 못했고, Instacart T의 시간 loss에서만 작은 개선이 있었다. 그러나 최종 epoch 120에서는 Taxi와 Instacart의 Q/T가 J보다 좋다. 따라서 최적 성능과 학습 후반 상태를 구분해야 한다.

가장 구체적인 후속 단서는 Taxi다. 후기 구간에서 validation 수량 log-MSE는 **1.90% 감소**했지만 raw RMSE는 **7.35% 증가**했다. 이 결과와 실제 구현을 함께 보면, Backbone을 다시 바꾸기 전에 **수량 학습 목표와 원단위 평가 지표의 관계**, 그리고 **시간 loss의 후반 악화**를 먼저 진단할 이유가 있다. 이는 원인을 확정한 결론이 아니다.

**비교 대상과 완료 기준 — 완료**

J는 시간·수량 공동 학습, Q는 수량만, T는 시간만 학습한다. Q/T도 각각 encoder를 학습했고 비활성 head는 optimizer에서 제외했다. 이번 비교는 동일한 static B 안에서 objective를 바꾼 진단이다. RMTPP·THP 또는 dual-timescale의 성능을 비교 분모로 섞지 않았다.

9개 학습이 각각 120 epoch를 완료했다. seed 42, batch 128, 같은 초기화·batch 순서·표본 노출·AdamW lr 0.001/weight decay 0.01/clip 1 조건이다. 전체 step은 6,816,240이다. 실제 실행은 2026-09-10 11:41:29부터 2026-09-11 08:33:27 KST까지다.

| 데이터셋 | train 표본/epoch | validation 표본 | step/arm | 완료 |
|---|---:|---:|---:|---|
|Intermittent|393,824|86,285|369,240|J/Q/T 각 120 epoch|
|Taxi|38,393|8,268|36,000|J/Q/T 각 120 epoch|
|Instacart|1,991,192|503,733|1,866,840|J/Q/T 각 120 epoch|

비교 우선순위는 [실행 승인 당시 정책](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_5090_20260910/execution_policy.json)의 selector 규칙을 따른다. 전체 결과와 단계별 노출·strict earliest selector·최종 receipt 감사는 [완료 기록](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_5090_20260910/monitor/20260910T233614Z_check.md)에 있다.

**최적 validation checkpoint 비교 — 완료, 주 비교**

수량은 각 arm의 raw RMSE가 가장 낮은 checkpoint를 선택하고 MAE도 그 checkpoint에서 읽었다. 시간은 별도 시간 selector다. 변화율은 (Q/J−1)×100이며 양수는 악화, 음수는 개선이다. 시간은 절대 차이(T−J)만 표시한다.

| 데이터셋 | Raw RMSE J → Q | 변화 | MAE J → Q | 변화 | 수량 epoch J/Q |
|---|---:|---:|---:|---:|---|
|Intermittent|1.499555 → 1.548389|+3.2566%|0.604340 → 0.597850|-1.0740%|77/24|
|Taxi|85.606968 → 87.789343|+2.5493%|27.470006 → 27.941478|+1.7163%|116/97|
|Instacart|5.872217 → 5.877409|+0.0884%|3.993781 → 3.992786|-0.0249%|72/72|

| 데이터셋 | Legacy clamped time loss J → T | T−J | 시간 epoch J/T |
|---|---:|---:|---|
|Intermittent|-3.566226 → -3.520814|+0.045412402|120/120|
|Taxi|1.365333 → 1.366590|+0.001257150|16/4|
|Instacart|3.206416 → 3.204591|-0.001824114|59/2|

**확인된 사실:** Q의 최적 수량 RMSE는 세 데이터셋 모두 J보다 높다. MAE는 Intermittent와 Instacart에서 조금 낮고 Taxi에서는 높다. T의 시간 loss는 Instacart에서만 낮다. 그러므로 “시간 단독 학습도 모두 나쁘다”는 이전 두 데이터셋 기준 해석을 세 데이터셋으로 확대하면 안 된다.

**동일 최종 epoch 120 비교 — 완료, 보조 비교**

같은 최종 optimizer step의 모델끼리 비교했다. 주 selector를 관측 결과에 맞춰 바꾼 것은 아니다. 모든 120 epoch 대응 수치는 JSON에 남겼다.

| 데이터셋 | Raw RMSE J → Q | 변화 | MAE J → Q | 변화 | 시간 loss J → T | T−J |
|---|---:|---:|---:|---:|---:|---:|
|Intermittent|1.652868 → 1.718703|+3.9831%|0.619376 → 0.647233|+4.4975%|-3.566226 → -3.520814|+0.045412402|
|Taxi|93.959049 → 91.671297|-2.4348%|29.603272 → 28.757767|-2.8561%|17.111530 → 15.801643|-1.309886720|
|Instacart|6.213403 → 6.160919|-0.8447%|4.143256 → 4.110572|-0.7889%|3.224450 → 3.223239|-0.001211311|

Taxi와 Instacart는 최종 epoch에서 Q/T가 J보다 낫다. 이는 분리의 효과가 비교 시점에 따라 달라진다는 근거다. 다만 최종 값들은 각자의 최적 값보다 나빠졌으므로, 이 보조 비교만으로 분리가 최적 성능을 높였다고 말할 수 없다. 최적값보다 최종값이 나쁘다는 사실 자체도 과적합의 증명은 아니다.

**Clipping 감소가 최적 수량 RMSE 개선으로 이어지지는 않았다 — 완료**

Clipping은 gradient norm이 임계값 1을 넘어 축소된 train batch의 비율이다. 서로 다른 학습 경로에서 기록한 빈도이므로 실제 parameter update 억제량이나 시간 gradient의 인과 효과를 뜻하지 않는다.

| 데이터셋·arm | 전체 clipping 횟수 / batch | 전체 비율 | 첫 10 epoch | 마지막 10 epoch |
|---|---:|---:|---:|---:|
|Intermittent J|9,179 / 369,240|2.4859%|29.8278%|0.0000%|
|Intermittent Q|44 / 369,240|0.0119%|0.1430%|0.0000%|
|Intermittent T|8,895 / 369,240|2.4090%|28.9080%|0.0000%|
|Taxi J|21,998 / 36,000|61.1056%|83.8000%|95.5667%|
|Taxi Q|5,651 / 36,000|15.6972%|62.5000%|3.3333%|
|Taxi T|13,970 / 36,000|38.8056%|35.8667%|92.6333%|
|Instacart J|310,221 / 1,866,840|16.6174%|33.5945%|14.2425%|
|Instacart Q|31,473 / 1,866,840|1.6859%|3.2378%|1.3872%|
|Instacart T|79,728 / 1,866,840|4.2707%|17.5104%|2.7222%|

Q는 세 데이터셋 모두 J보다 clipping이 적지만 최적 수량 RMSE는 좋아지지 않았다. Intermittent의 clipping은 주로 초기에 집중됐고, Taxi의 마지막 10 epoch에서는 J 95.57%와 T 92.63%가 높다. **T에도 후반 clipping과 시간 loss 악화가 있으므로, Taxi 문제를 시간·수량의 공동 학습 간섭만으로 설명할 수 없다.**

**Taxi에서는 수량 loss와 원단위 성능의 방향이 다르다 — 완료된 기술적 분석**

후기 81–100 epoch 평균과 101–120 epoch 평균을 비교했다. 이는 결과를 본 뒤 추가한 기술적 요약이며 새로운 selector나 통계적 반복실험이 아니다.

| 데이터셋·arm | train 수량 log-MSE 변화 | validation 수량 log-MSE 변화 | validation raw RMSE 변화 |
|---|---:|---:|---:|
|Intermittent J|-0.7884%|-1.2260%|-1.9721%|
|Intermittent Q|-0.7584%|-5.3983%|-5.0031%|
|Taxi J|-0.2453%|-1.8970%|+7.3470%|
|Taxi Q|-7.9665%|-2.1507%|-2.0340%|
|Instacart J|-0.0056%|+0.2181%|-0.0062%|
|Instacart Q|-0.0035%|+0.1364%|+0.0867%|

Taxi J는 train·validation의 수량 log-MSE가 모두 낮아지는데 원단위 RMSE가 높아진다. 따라서 “train만 좋아지고 validation이 나빠지는 단순 과적합”으로 정리하기 전에 loss와 보고 지표가 강조하는 오차의 차이를 확인해야 한다. 어느 수량 구간이나 표본이 이 차이를 만들었는지는 아직 계산하지 않았다.

Taxi 시간 loss도 한 가지 패턴은 아니다. 같은 후기 구간에서 J의 train 시간 loss는 1.3162→1.3484, validation은 7.0670→10.6811로 함께 악화했다. T의 train은 1.3716→1.3572로 낮아졌지만 validation은 6.9263→12.8108로 높아졌다. 이 관측만으로 하나의 과적합·간섭 원인으로 묶지 않는다.

수량 구현은 `z = softplus(Wh+b)`, `Lq = (z − log1p(q))²`, `q̂ = expm1(z)`다. [frozen 모델 코드](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/jqt_cuda_5090_20260910/source/models/TPPs/CountAwareTPP.py:737). 원단위 지표는 `q̂−q`의 절댓값·제곱오차를 표본 수로 집계한다. [평가 코드](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/jqt_cuda_5090_20260910/source/paper/scripts/time_quantity_diagnostic.py:304). 로그 변환은 큰 수량 차이를 압축하므로 log-MSE와 raw RMSE의 개선 방향이 반드시 같지는 않다.

이론적으로 기대 log-MSE를 완전히 최소화한 예측을 역변환한 값과 원단위 MSE를 최소화하는 조건부 평균은 일반적으로 다르다. 그러나 이번 모델이 그 이론적 최적점에 도달했거나 현재 오차가 과소예측 때문이라고 확정할 근거는 없다. 조건부 중앙값이라는 표현도 추가 분포 가정 없이는 사용하지 않는다.

기록 해석상 `history.quantity_train_loss`는 **validation** 수량 loss이고 `history.train.quantity_train_loss`가 실제 train loss다. 또한 train 값은 epoch 도중 바뀌는 가중치의 forward 평균, validation은 epoch 종료 모델의 평가값이다. 같은 checkpoint로 train split을 재평가한 것은 아니다. [학습·평가 순서](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/jqt_cuda_5090_20260910/source/paper/scripts/time_quantity_diagnostic.py:532).

**과거 gradient 진단과 이번 실험의 관계 — 해석 완료**

과거 train-only B probe는 데이터셋당 2,048 target·16 batch에서 encoder gradient cosine 중앙값이 Intermittent −0.0299, Taxi +0.0399, Instacart −0.0213이었다. 음수 batch는 10/16, 6/16, 10/16이었다. 일부 batch의 방향 충돌 가능성과 이번 전체 학습에서의 분리 이득은 다른 질문이다. 음수 cosine만으로 실제 성능 손실의 원인을 확정할 수 없으며 이번 Q RMSE 결과도 그 점을 뒷받침한다.

이 probe는 과거 선택 checkpoint·CPU·별도 frozen source ca8823e6의 0-update 측정이며, 이번 CUDA J/Q/T 학습 궤적과 섞지 않았다. [probe 요약](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_diagnostic_20260910/gradient_summary.json) 및 [계약](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_diagnostic_20260910/contracts/frozen_gradient_probe_v1.json).

**지금 내릴 설계 판단과 논문 주장 범위 — 해석 완료**

- 공동 B를 현재 대조 기준으로 유지하고, 완전 분리를 본모델 구조로 확정하는 판단은 보류한다. 이번 조건에서 분리의 공통 수량 RMSE 이득을 찾지 못했다는 결론이다.
- Backbone의 모든 한계나 모든 분리 구조의 실패를 입증한 것은 아니다. 새 memory 구조가 필요하다는 근거도 이번 J/Q/T 비교에서 나오지 않았다.
- 다음 설계 검토의 우선순위는 수량 목표·원단위 출력의 정합성과 시간 loss의 후반 안정성이다. raw MSE로 바로 교체하거나 clipping만 바꾸면 해결된다고 결론내리지 않는다.
- “TitanTPP가 benchmark보다 우수하다”는 논문 가설은 이 실험으로 검증되지 않았다. 이번 비교에는 같은 조건의 THP·RMTPP 성능, 추가 seed, Q+T의 전체 용량·학습/추론 비용 통제가 없다.

**남은 작업 순서**

**수량 오차가 어느 구간에서 갈리는지 확인할 계약 작성 — 다음 작업**

- 대상은 현재 B의 J/Q 수량 selector다. Intermittent의 수량 구간·이력 길이별 오차와 Taxi·Instacart의 부호 편향 및 log 오차·raw 제곱오차 기여를 비교할 표본·구간·출력 항목을 고정한다. 구간 경계는 train 기준으로 정하고 validation 결과를 보고 유리한 경계를 고르지 않는다.
- 목적은 representation 변경, 수량 head/목표 변경 중 어느 쪽을 먼저 검토할지 결정하는 것이다. 현재 회수본에는 표본별 예측이 없어 이 구간 분석은 미실행이다. 기존 진단 준비 문서를 재사용하되 이번 source와 J/Q identity를 분리한다.

**Taxi 시간 loss의 후반 악화와 기존 관측모형 트랙 정리 — 병렬 가능한 다음 작업**

- J와 T의 후반 시간 loss·gradient 크기·clipping, legacy clamp의 영향을 구분할 범위를 정한다. 기존 [관측 law 정렬 트랙](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_design_review_20260910/README.md:82)을 재사용한다. 정상화된 likelihood로 변경하면 새 J도 함께 학습해야 하며 기존 B나 legacy 허용폭을 그대로 matched control로 쓰지 않는다.
- 수량 진단과 독립적인 문서·코드 검토는 병렬로 진행할 수 있다. 공유 실행 계약·새 head/loss·실제 GPU 비교는 그 결과를 모아 직렬로 확정한다.

**새 모델 평가·GPU 실험 실행 — 실행 계약 확정 후 승인 필요**

- 서버·비용·학습량·비교군·평가 split과 중단 조건을 구체화한 뒤 실행한다. 이번 통합 분석에서는 checkpoint나 원본 데이터를 실행하지 않았고, 완료된 두 Scheduler는 PAUSED를 유지했다.

**검증과 재현 범위**

단일 seed validation 결과이며, validation은 checkpoint 선택에도 사용됐다. J의 수량/시간 selector는 서로 다른 checkpoint다. 시간 지표는 legacy clamped loss로 정규화된 Time NLL이 아니다. 최적 checkpoint끼리의 비교와 최종epoch 120 비교를 별도로 유지했고 데이터셋별 raw RMSE를 단순 평균하지 않았다.

Body/tail 성능·표본별 원인·새 구조의 이득·통계적 유의성은 미확인이다. train/validation 대상 수와 순서·step·결과 SHA 검증은 완료됐지만 원본 표본을 재평가한 검증은 아니다.

- [통합 수치와 전체 120 epoch 대응 결과](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_5090_20260910/three_dataset_validation_v1/validation_results.json)
- [재현 가능한 추출·계산 코드](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_5090_20260910/three_dataset_validation_v1/build_report.py)
- [최종 원격 증적](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_5090_20260910/monitor/20260910T233614Z_terminal/collection_manifest.json)
- [기존 Intermittent·Taxi 보고서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_5090_20260910/intermittent_taxi_validation_v1/validation_report.md)

재계산 명령:

```bash
python3 /Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_jqt_5090_20260910/three_dataset_validation_v1/build_report.py
```
