# 가변 검색은 5090에서 수량을 개선했지만 5080 반복에서는 추가 효과가 재현되지 않았다

10실행의 완료·종료 검증과 20개 selected/last binary 원본의 로컬 SHA 검증을 마쳤습니다. 기존 미확정 8개는 정상 종료 후 Validation 추가 확인 기록을 epoch 영수증과 구분하지 못한 관측기 오류였습니다. 학습 실패로 취급하지 않으며 중단·재개 이력은 원본대로 보존합니다.

5090 주 비교에서는 adaptive가 사전에 정한 수량·비용 판정을 통과했습니다. 그러나 5080 같은 seed 반복에서는 fixed·weighted보다 전체와 큰 수량 RMSE가 모두 나빴습니다. 현재 판단은 **가능성이 확인됐지만 반복 안정성이 부족한 검색 방법 후보**입니다. 대표 모델 채택이나 안정적인 추가 Contribution 확정으로 해석하지 않습니다.

## 전체 Validation 결과

Taxi·seed42, 전체 8,268건·큰 수량 79건(Train에서 정한 수량 >3,449)입니다. 각 방식의 최초 strict finite 최소 fullValidation raw 수량 RMSE epoch에서 6지표를 함께 읽었습니다. 모든 지표는 낮을수록 좋습니다.

| GPU | 방식 | 선택/종료 epoch | 전체 RMSE | 전체 MAE | TimeNLL | 큰 수량 RMSE | 큰 수량 MAE | 큰 수량 TimeNLL |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 5090 | mean | 80/120 | 84.878 | 26.736 | 1.2058 | 427.843 | 333.241 | 6.978e-34 |
| 5090 | weighted | 59/99 | 86.253 | 28.198 | 1.0082 | 411.523 | 317.190 | 3.46e-31 |
| 5090 | fixed | 120/160 | 83.420 | 26.095 | 1.5133 | 412.485 | 306.463 | 1.263e-94 |
| 5090 | adaptive | 124/164 | 81.436 | 26.374 | 1.6182 | 380.695 | 293.576 | 3.269e-42 |
| 5090 | generic | 57/97 | 85.959 | 26.565 | 1.0237 | 465.053 | 348.251 | 7.274e-113 |
| 5080 | mean | 46/86 | 86.900 | 26.839 | 0.9156 | 473.329 | 368.141 | 2.505e-25 |
| 5080 | weighted | 80/120 | 83.153 | 26.419 | 1.1188 | 408.028 | 314.746 | 1.318e-36 |
| 5080 | fixed | 111/151 | 83.541 | 26.158 | 1.4277 | 386.151 | 285.303 | 7.661e-14 |
| 5080 | adaptive | 59/99 | 86.117 | 27.389 | 0.9872 | 441.273 | 340.559 | 2.322e-35 |
| 5080 | generic | 69/109 | 84.905 | 27.224 | 1.1382 | 449.328 | 347.003 | 1.055e-28 |

## 가변 비중의 추가 효과와 불리한 조건

아래 RMSE 변화는 (비교 방식−adaptive)/비교 방식입니다. 양수는 adaptive 개선, 음수는 악화입니다. TimeNLL 변화는 원본의 절댓값 차이이며 양수는 악화입니다.

| GPU | 비교 방식 | 전체 RMSE 개선 | 큰 수량 RMSE 개선 | 전체 MAE 개선 | 큰 수량 MAE 개선 | TimeNLL 변화 |
|---|---|---:|---:|---:|---:|---:|
| 5090 | mean | +4.06% | +11.02% | +1.35% | +11.90% | +0.4123 |
| 5090 | weighted | +5.58% | +7.49% | +6.47% | +7.44% | +0.6100 |
| 5090 | fixed | +2.38% | +7.71% | -1.07% | +4.20% | +0.1049 |
| 5090 | generic | +5.26% | +18.14% | +0.72% | +15.70% | +0.5944 |
| 5080 | mean | +0.90% | +6.77% | -2.05% | +7.49% | +0.0716 |
| 5080 | weighted | -3.57% | -8.15% | -3.67% | -8.20% | -0.1316 |
| 5080 | fixed | -3.08% | -14.27% | -4.71% | -19.37% | -0.4405 |
| 5080 | generic | -1.43% | +1.79% | -0.61% | +1.86% | -0.1510 |

5090 adaptive는 fixed보다 전체 RMSE 2.38%, 큰 수량 RMSE 7.71%를 개선했습니다. generic 대비 개선은 각각 5.26%, 18.14%입니다. 비용도 T64/T256 모두 적격이므로 **5090 단일 데이터·seed의 사전 탐색 판정은 통과**입니다. weighted보다도 두 RMSE가 개선됐지만, 전체 MAE는 fixed보다 1.07% 나쁩니다.

시간 손해를 숨길 수 없습니다. 5090 TimeNLL은 mean 1.2058→adaptive 1.6182로 +0.4123(34.19%) 악화했습니다. fixed 대비 +0.1049(6.93%), generic 대비 +0.5944(58.07%)입니다. 이 값은 frozen legacy clamped time head(cap 300)의 기록 지표이며 외부 모델의 별도 시간 지표와 섞지 않습니다. 큰 수량 TimeNLL은 모두 거의 0이라 비율 개선으로 해석하지 않습니다.

5080에서는 adaptive가 fixed보다 전체 RMSE 3.08%, 큰 수량 RMSE 14.27% 나쁩니다. weighted보다도 3.57%, 8.15% 나쁘고, 전체 MAE는 모든 비교 방식보다 나쁩니다. mean보다 RMSE가 좋아졌어도 고정 비중·일반 검색을 함께 넘어야 하는 판정은 **미통과**입니다.

두 서버는 같은 seed42의 GPU/Runtime 반복입니다. cuDNN도 5090=92000, 5080=91900으로 달라 결과 차이를 GPU 자체의 인과 효과로 단정하지 않습니다. 반복 실패의 원인은 아직 검증하지 않았습니다. 두 결과를 두 seed처럼 평균·표준편차로 합산하지 않습니다.

## 계산비용과 실제 학습 시간

동일 GPU의 native qualification에서 B128, warm5/measure20으로 측정한 평균 검색 대비 증가율입니다. 실제 fit 지연·메모리와 동일한 측정은 아니며, GPU 간 시간을 섞지 않습니다. 사전 한도는 추론 +20%, 학습 step +30%, peak allocation +20%입니다.

| GPU | 길이 | 추론 증가 | 학습 step 증가 | peak allocation 증가 | 비용 판정 |
|---|---:|---:|---:|---:|---|
| 5090 | 64 | 8.43% | 4.82% | 3.16% | 적격 |
| 5090 | 256 | 2.43% | 1.74% | 2.59% | 적격 |
| 5080 | 64 | 8.37% | 4.79% | 3.16% | 적격 |
| 5080 | 256 | 2.57% | 1.67% | 2.59% | 적격 |

| GPU | 방식 | 실제 완료 epoch | epoch 실측 시간 합계 |
|---|---|---:|---:|
| 5090 | mean | 120 | 1.651h |
| 5090 | weighted | 99 | 1.310h |
| 5090 | fixed | 160 | 2.159h |
| 5090 | adaptive | 164 | 2.241h |
| 5090 | generic | 97 | 1.289h |
| 5080 | mean | 86 | 1.305h |
| 5080 | weighted | 120 | 1.868h |
| 5080 | fixed | 151 | 2.288h |
| 5080 | adaptive | 99 | 1.515h |
| 5080 | generic | 109 | 1.664h |

epoch 시간 합계는 train 시작부터 Validation·checkpoint 저장까지의 실측입니다. 종료 후 확인·원본 회수·중단 대기 시간은 제외합니다. 5090 fixed는 E148까지의 원래 시간과 E149–160의 시간을 모두 포함하며, 재개 구간만의 약 594초를 전체 비용으로 제시하지 않습니다. 조기 종료 epoch가 서로 달라 전체 시간 차이를 검색 구조 자체의 효율 차이로 단정하지 않습니다.

## 중단·재개와 미확정 해소

- 원래 5090 mean E120/best80, weighted E99/best59의 완료 원본 2개를 재사용했습니다. 원 관측시각은 2026-10-07 14:08:54 KST입니다.
- 원래 fixed는 E148/best120에서 7,200초 제한으로 worker124/supervisor1 종료한 증거를 보존했습니다. 같은 5090의 optimizer/RNG/loader/history/timing prefix와 원본 SHA를 검증한 뒤 E149부터 이어가 E160/best120에 정상 종료했습니다.
- 5080 다섯 방식과 새 5090 adaptive/generic은 fresh fit 7개입니다. 중단·재개 자체가 전체 8개 미확정의 원인은 아닙니다.
- 원래 epoch receipt의 Train N/Validation N 기록에서, 정상 종료 시 Validation 1회가 추가됐습니다. 수정 관측기는 최종 exposure에서 마지막 Validation 1개를 제외한 원래 형식의 raw SHA가 영수증과 정확히 같은지 검증했습니다. 나머지 4개 파일 SHA, 실제 CPU checkpoint metadata, 전체 terminal manifest SHA, PID 부재와 exit0 검사는 유지했습니다.
- 기존 current·C2·source126·operation·control 영수증은 변경하지 않았습니다. 결과 복구용 코드만 별도 profile로 봉인했습니다. 검증 회귀 테스트 47개와 세부 사례 60개가 통과했습니다.

## Contribution에 대한 판단

이번 후보의 방법론 범위는 **관측된 과거 이력 요약으로 시간·수량 검색 점수의 비중을 조절하는 정적 prototype 검색**입니다. 학습된 prototype64, top4, CNN+GRU54 및 과거32사건 요약은 공통 기준입니다. 실제 과거 사건 bank를 새로 저장·검색하는 모델이나 prototype 내용 자체의 개선으로 표현하지 않습니다. 추가 parameter는 mean/weighted 0, fixed 512, adaptive/generic 521입니다.

5090에서는 fixed와 같은 예산의 generic을 모두 넘었으므로 단순 가중합·추가 parameter보다 좁은 방법 기여를 검토할 개발 증거가 생겼습니다. 다만 5080 결과가 같은 설명을 지지하지 않으므로, 검색부를 확정적인 두 번째 Contribution으로 내세우기에는 이릅니다. 기존 MLP8 54조건의 명시적 이력 보정 결과는 별도 증거이며 이 후보의 CNN+GRU54 결과로 옮겨 주장하지 않습니다.

현재는 논문의 검색 방법 후보/탐색 결과로 보존하고 대표 모델 채택은 보류하는 것이 타당합니다. 고정 Runtime 안에서 독립 seed 재현성 확인이 다음 판단을 바꿀 수 있지만 이번 요청에서 확대 학습은 실행하지 않았습니다. 이 파일럿은 참신성의 선행연구 검증, 통계적 유의성, 외부 우월성, 독립 Test 성능을 증명하지 않습니다.

## 최종 관측과 원본 증거

양 repair GPU의 실제 종료 관측시각은 2026-10-08 07:05:08 KST입니다. 완료 수는 5080=5/5, 5090=재사용2+repair3=5/5이며 실패·진행·대기·미확정은 모두 0입니다. 기존 5090 timeout 종료도 정확한 원인과 소유 PID 부재로 확인했습니다. 학습 ETA는 종료 상태이며 원본 회수와 최종 Validation 집계도 완료했습니다.

최종 원본 검증 완료시각: 2026-10-08 07:05:39 KST. 세 owner의 archive/manifest와 20개 selected/last binary가 로컬에서 raw SHA로 검증됐습니다. CPU binary **metadata 읽기**와 binary **재추론 감사**는 다르며 재추론 감사·Test·새 독립 평가는 미완료입니다.

| owner | archive SHA256 | manifest SHA256 |
|---|---|---|
| repair_5090 | `9de75569f4e8e8d97193f28c639699a6163c3dd7408f207d09047bf26719341a` | `6c68825bf1808e619caf464476c12ca3d630eff6e8f54b4d48b2c5cc29b55384` |
| repair_5080 | `85f46ef3bffc0f1464db4339210238366573ac4688e36d6f11197fac77503221` | `5ccd119665d3781da73052c179fd0161c737b75b9462763552c992d972e271b0` |
| parent_5090 | `d4414325f303451f9c70e6933adb118160103067a782c562a8c5dbbc98c5609a` | `64570c78b3cc7222575a9adb79e05725c7ed4cc53267174889bb7487117d5c1c` |

원본과 집계 출처:

- [최종 Validation 요약 JSON](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_repair_20261007_v1/analysis/Validation_summary.json)
- [10실행 선택 epoch의 원본 지표 CSV](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_repair_20261007_v1/analysis/selected_Validation_metrics.csv)
- [20개 binary 원본 manifest](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_repair_20261007_v1/analysis/binary_originals_manifest.json)
- [최종 완료 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_repair_20261007_v1/analysis/completion_receipt.json)
- [별도 결과 복구 봉인](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_contextual_retrieval_pilot_repair_20261007_v1/control/result_recovery_20261008_v1/profile.json)
- [이 보고서의 계산·출처 JSON](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_contextual_retrieval_pilot_results_20261008_v1/analysis.json)
