# 기존 CNN+GRU54와 새 평균 검색은 구조가 같지만, 같은 학습 결과는 아니다

Taxi·seed42의 기존 CNN+GRU54와 검색부 파일럿의 mean 5090·5080을 로컬 원본으로 대조했습니다. **데이터·공통 과학 소스·학습 설정·배치 순서는 동일합니다. 학습 장비와 실행 환경, 실제 학습 경로·노출 횟수와 선택 checkpoint는 다릅니다.** 현재 mean은 CNN+GRU54를 새로 학습한 내부 기준선입니다.

**fixed는 새 mean보다 수량 Test가 개선됐지만 기존 강한 CNN+GRU54를 전반적으로 개선했다고 주장할 근거는 없습니다.** 기존 전체 Test RMSE는 105.940, fixed5080은 107.743, fixed5090은 126.369입니다. fixed5080의 큰 수량 RMSE는 기존보다 낮지만 전체 RMSE·MAE와 TimeNLL은 높습니다. 확인된 Runtime 차이가 성능 격차의 원인이라는 인과적 결론은 아직 없습니다.

이 감사는 원본 계약·checkpoint metadata·학습 이력·기존 평가 집계를 읽고, 로컬 CPU에서 합성 모델/집계 검증을 수행했습니다. 실제 데이터로 모델 추론, 새 학습, 원격 접속, checkpoint 재선택은 수행하지 않았습니다. 이미 요청한 Test 결과의 비교 범위만 사용했습니다.

## 비교 대상을 정확한 checkpoint에 연결했다 — 완료

| 구분 | 기존 CNN+GRU54 | mean5090 | mean5080 |
|---|---|---|---|
| 학습 계보 | A100 원학습 fresh fit | 부모5090 fresh fit 완료 결과 재사용 | repair5080에서 원래 미시작 조건 fresh fit |
| 선택 / 종료 epoch | 126 / 166 | 80 / 120 | 46 / 86 |
| 선택 binary SHA 앞부분 | `17f90d3eed5d` | `234ce1388208` | `2e1746da3151` |
| 초기 full-state SHA 앞부분 | `592be61554bf` | `23b246f93392` | `23b246f93392` |
| 과학 revision | `38fcc58e45f3` | `4274043717cf` | `4274043717cf` |
| 초기 state 키 수 | 46 | 47 | 47 |

기존 Test 값은 [기존 CNN+GRU54 Test 원본 영수증](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_test_20261005_v1/attempt2/retrieved/runs/test/yellow_trip_hourly__titantpp_cnn_gru54__seed42/receipt.json)과 A100 selectedE126 binary를 SHA로 연결했습니다. 과거 회수 경로 이름에 resume가 포함돼도 **Taxi CNN+GRU54는 재개 학습이 아닙니다.** 새 mean5090 역시 완료된 fresh fit의 재사용입니다. 재개된 것은 파일럿의 fixed5090 E148 이후이며 mean 계보와 구분했습니다. selected/last 총6개 binary의 raw SHA와 실제 tensor state digest를 검증했습니다. 전체 SHA·선택 계보는 [데이터·checkpoint 증적](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/evidence/data_audit.json)에 있습니다.

## 데이터와 구조, 학습 조건을 대조했다 — 완료

| 항목 | 확인 결과 | 해석 |
|---|---|---|
| 원자료 / split manifest | rawSHA `b47e98e9…ca46` / `4a005d4a…ed85` 모두 동일 | 다른 데이터 파일이나 분할의 비교가 아님 |
| 모집단·대상 순서·수량·관측 시간 | Train 대상38,393, Validation8,268 / 큰 수량79, Test8,327 / 큰 수량83; 각 identity 재계산 일치 | 표본·target 정의의 차이 증거 없음 |
| digest 표기 | 과거 JSON identity/truth와 현재 hard_lmm digest 방식이 다름; 같은 frozen loader에서 각각 재계산 후 원본과 일치 | 서로 다른 SHA 문자열을 데이터 차이로 해석하지 않음 |
| source117 → source126 | 공통117개 파일 SHA 전부 동일, 추가9개 | revision/closure 변경이 공통 backbone 변경을 뜻하지 않음 |
| mean 구조 | 기존 CNNGRU54를 생성하고 같은 forward로 위임; `contextual_retrieval_identity` 버퍼 하나 추가 | 추가 학습 parameter0, 둘 다115,167 parameters |
| 공통 모델 | CNN+GRU54 이력 보정, 학습된 정적 prototype64, top4 평균 | 실제 과거 사건 bank로 해석하지 않음 |
| 정규화·시간 초기 통계 | Train log 수량 평균2.837734592747447 / SD2.1840596349132664; Train 시간 통계 동일 | Test로 다시 맞춘 통계가 아님 |
| 학습 조건 | seed42, B128, AdamW lr0.001/wd0.01/betas0.9,0.999/eps1e-8, clip1, maxT256, lookback168, dropout0.1 | metadata와 실제 last optimizer 설정 동일 |
| 종료·선택 규칙 | min40/max300/patience40, 최초 strict finite 최소 fullValidation raw수량RMSE | 선택 정책 동일; 선택 epoch와 실제 학습량은 다름 |
| 공통 epoch의 배치 순서 | A100↔5080 86개, A100↔5090 120개, 5080↔5090 86개 epoch 모두 Train/Validation 순서·개수 일치 | shuffle 차이를 격차 원인으로 확인하지 못함 |

큰 수량 경계는 Train에서 정한 **수량 >3,449**입니다. 전체 Validation/Test 성과이며 N/n 가중 Train 표본 진단과 섞지 않았습니다. 같은 causal history 규칙, loader와 동일한 duration head 계약를 확인했습니다. [소스·설정 증적](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/evidence/code_audit.json)에는 파일117개 각각의 SHA와 코드 위치가 있습니다.

## 초기화 동등성의 확인 범위를 구분했다 — 완료 / native 초기 tensor는 미확인

동일 로컬 CPU Runtime(Python3.12.10/Torch2.14.0)에서 seed42로 두 모델을 생성하면 공통 state/parameter 변경0, 모델 생성 후 RNG 동일, 합성 `target_outputs` 12항목 모두 bitwise 동일입니다. 새 버퍼는 parameter가 아니며 이 검사에서는 RNG를 소비하지 않았습니다.

그러나 이 로컬 실행은 과거 Torch2.11 native 초기화 SHA를 어느 쪽도 재현하지 못했습니다. 원래 GPU 실행의 초기 parameter tensor 원본도 확보돼 있지 않습니다. 따라서 **과거 A100과 새 RTX의 실제 초기 parameter가 정확히 같았다거나 초기 SHA 차이가 버퍼만 때문이라고 확정하지 않습니다.** 반대로 full-state SHA가 다르다는 사실만으로 parameter 초기값이 달랐다고 단정하지도 않습니다. 새 mean5080/5090의 기록된 full-state 초기 SHA는 같지만 학습 곡선은 달랐습니다.

## 실제 학습 Runtime과 경로가 달랐다 — 완료

| 항목 | 기존 CNN+GRU54 | mean5080 | mean5090 |
|---|---|---|---|
| 학습 GPU | A100-SXM4-80GB / CC8.0 | RTX5080 / CC12.0 | RTX5090 / CC12.0 |
| driver | 580.126.16 | 595.84 | 595.84 |
| Python / CPU threads | 3.12.3 / 2 | 3.12.13 / 4 | 3.12.13 / 4 |
| Torch / CUDA | 2.11.0+cu130 / 13.0 | 동일 | 동일 |
| NumPy / Polars | 2.4.4 / 1.39.3 | 동일 | 동일 |
| cuDNN | 91900 | 91900 | 92000 |
| 당시 학습 worker 수 | 3 | 1 | 1 |
| 완료 epoch / optimizer steps | 166 / 49,800 | 86 / 25,800 | 120 / 36,000 |
| 첫 epoch Validation RMSE | 119.308111 | 123.172343 | 121.383105 |
| epoch 실측 중앙값 | 187.34초 | 54.61초 | 49.52초 |
| epoch 실측 합계 | 30,941.70초 | 4,698.12초 | 5,943.85초 |

Float32/highest, deterministic algorithms, cuDNN deterministic/benchmark off, TF32 off 설정은 같습니다. 같은 seed·결정적 실행 옵션이 이종 GPU/커널의 동일 학습 경로까지 입증하지는 않습니다. 학습 곡선은 첫 epoch부터 다릅니다. 동일한 종료 규칙이 달라진 곡선에 적용돼 선택/종료 epoch와 실제 노출 횟수도 달라졌습니다. **종료 정책 차이 또는 새 mean이 중간 중단된 것 때문에 점수가 낮았다는 증거는 없습니다.**

실측 epoch 시간은 Train 반복 시작부터 Validation·checkpoint 저장까지 포함합니다. 장비와 동시 worker 수가 다르므로 위 시간 차이를 CNNGRU/mean 구조 효율 차이로 주장하지 않습니다. 상세 precision flags·exposure·첫10epoch 곡선과 Runtime 증거는 [학습 이력·Runtime 증적](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/evidence/runtime_audit.json)에 있습니다.

## 평가 방식과 수치 차이를 확인했다 — 완료

기존 A100 학습 checkpoint의 Validation/Test 추론은 **RTX5080**에서 수행했습니다. 새 checkpoint는 각각 원래5090/5080에서 평가했습니다. 기존과 새5080 평가의 GPU·주요 package·precision 설정은 일치합니다. 기존 frozen CNNGRU import adapter와 새 retrieval hook은 동일 core `target_outputs`와 eval/no_grad 경로를 사용합니다. 학습 장비와 평가 장비를 혼동하지 않습니다.

두 평가기의 RMSE/MAE/TimeNLL·수량 구간 정의는 같습니다. 과거 NumPy float64 합산과 현재 Python double `math.fsum`의 구현 차이는 있지만, 경계값과 무작위 합성10,017건을 4가지 batch 크기로 확인한 집계 충분통계의 최대 상대 편차는 6.57e-15였습니다. 이는 과거 실제 model forward 재현을 대신하지 않습니다.

기존 선택 당시 A100 Validation RMSE는78.31980241569856이고, 저장 checkpoint의 5080 Validation replay는78.31980934542565입니다. 절대 차이6.9297271e-06로 당시 gate(rel/abs1e-5)를 통과했습니다. 아래 Validation 표는 **replay 값**으로 통일했습니다. 새10개 checkpoint의 native Validation replay 최대 절대 편차는1.42e-14입니다. 감사 과정에서 재선택·Test 재추론을 수행하지 않았습니다.

### 같은 선택 epoch의 Validation 지표

| 모델 / 실행 | 학습 GPU → 평가 GPU | 선택 / 종료 epoch | 전체 RMSE | 전체 MAE | 전체 TimeNLL | 큰 수량 RMSE | 큰 수량 MAE | 큰 수량 TimeNLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 CNN+GRU54 | A100 → 5080 | 126 / 166 | 78.320 | 25.232 | 1.3660 | 370.080 | 276.912 | 9.056e-123 |
| mean 5090 | 5090 → 5090 | 80 / 120 | 84.878 | 26.736 | 1.2058 | 427.843 | 333.241 | 6.978e-34 |
| mean 5080 | 5080 → 5080 | 46 / 86 | 86.900 | 26.839 | 0.9156 | 473.329 | 368.141 | 2.505e-25 |
| fixed 5090 | 5090 → 5090 | 120 / 160 | 83.420 | 26.095 | 1.5133 | 412.485 | 306.463 | 1.263e-94 |
| fixed 5080 | 5080 → 5080 | 111 / 151 | 83.541 | 26.158 | 1.4277 | 386.151 | 285.303 | 7.661e-14 |

### 같은 선택 epoch의 Test 지표

| 모델 / 실행 | 학습 GPU → 평가 GPU | 선택 / 종료 epoch | 전체 RMSE | 전체 MAE | 전체 TimeNLL | 큰 수량 RMSE | 큰 수량 MAE | 큰 수량 TimeNLL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 기존 CNN+GRU54 | A100 → 5080 | 126 / 166 | 105.940 | 33.696 | 1.2753 | 589.041 | 450.037 | 6.098e-115 |
| mean 5090 | 5090 → 5090 | 80 / 120 | 149.245 | 41.511 | 1.1838 | 1051.682 | 902.231 | 4.078e-38 |
| mean 5080 | 5080 → 5080 | 46 / 86 | 190.858 | 55.258 | 0.9129 | 1315.099 | 1216.279 | 1.223e-28 |
| fixed 5090 | 5090 → 5090 | 120 / 160 | 126.369 | 37.730 | 1.5533 | 814.309 | 674.457 | 1.629e-122 |
| fixed 5080 | 5080 → 5080 | 111 / 151 | 107.743 | 35.609 | 1.4585 | 550.547 | 416.318 | 1.937e-14 |

Validation과 Test는 서로 다른 split로 나란히 보고했으며 섞어 집계하지 않았습니다. 같은 epoch의 전체/큰 수량6지표를 모두 보존했습니다. 큰 수량 TimeNLL은 거의0이므로 비율로 우월성을 주장하지 않습니다.

## 검색부에 대해 유지할 수 있는 결론 — 완료

fixed는 새 mean 대비 전체/큰 수량 Test RMSE를5090에서15.33%/22.57%,5080에서43.55%/58.14% 낮췄습니다. 동일 epoch 전체 TimeNLL은 각각+0.3695/+0.5456 높습니다. 각 GPU 실측 학습 epoch 합계는 mean→fixed가5090 5,943.85→7,773.53초,5080 4,698.12→8,238.57초입니다. 실제 종료 epoch 수가 다르고5090fixed는 중단 전·후 실측을 합친 값이므로 순수 step 속도나 추론 비용 차이로 해석하지 않습니다.

기존 CNNGRU54와 비교하면 fixed5080의 전체 Test RMSE는 **+1.70%**, 큰 수량 RMSE는 **-6.54%**입니다. 전체 MAE는+5.68%, TimeNLL은+0.1832입니다. fixed5090은 전체/큰 수량 RMSE가+19.28%/+38.24%입니다. **기존보다 모든 수량·시간 지표가 개선된 결과는 아닙니다.** 이 값들은 다른 학습 run의 고정 checkpoint 관측 비교이며 검색 변경의 통제된 인과 효과를 뜻하지 않습니다.

따라서 “새 파일럿 안에서 fixed가 mean보다 수량 Test에서 좋았다”는 기술은 가능합니다. “fixed가 기존 CNNGRU54를 전반적으로 개선했다”, “adaptive의 독립적 추가 효과가 확인됐다”, “과거 학습 장비 때문에 기존 모델이 더 좋았다”는 확정은 지지되지 않습니다. fixed의512개 추가 projection parameter와 일반 학습 검색 대조도 함께 고려해야 하며, 데이터1개/seed1개 결과로 Contribution이나 참신성을 확정하지 않습니다.

이 Taxi Test는 이전에 열람된 데이터입니다. 이번 감사는 새 독립 Test를 확보한 작업이 아닙니다. 이후 설계 변경을 이 점수에 맞추면 개발 과정의 Test 노출로 기록해야 합니다. MLP8 구조54조건 증거는 별도로 보존합니다. CPU binary 재추론 감사·새 독립 평가는 미완료입니다.

## 남은 작업을 의존 순서로 정리한다

**기존 CNNGRU54와 새 mean의 비교 감사 — 완료 / 현재 기준선**
- paper_research의 기존 selectedE126을 강한 비교 대상으로 유지하고, 새 mean5090/5080을 별도 fresh 기준선으로 기록했습니다. 데이터·설정 동일과 native 초기화/인과 원인 미확인을 구분했습니다.

**동일 실행 환경의 초기화·첫 학습 step 계약을 준비한다 — 다음 작업**
- paper_research에서 CNNGRU54와 mean의 parameter-only initial SHA, CPU/CUDA RNG, 공통 입력·optimizer·단계별 state를 저장하는 비교 계약을 정하면 Runtime과 초기화 차이를 더 좁힐 수 있습니다. 과거 epoch를 덮어쓰지 않고 새 증거로 분리합니다. 이번 감사에서는 새 코드를 모델 학습에 적용하지 않았습니다.

**고정 검색의 재현성 대조를 실행한다 — 승인 필요**
- 사용자가 확대 실행을 선택하면 하나의 지정 GPU/Runtime에서 CNNGRU54/mean과 fixed·같은 예산 generic을 맞춘 여러seed Validation 대조를 수행해야 합니다. 실제 초기 parameter·budget·selector를 고정하고 현재Test로 사후 선택하지 않습니다. common contract에 의존하므로 계약/로컬 검증 후 GPU 작업은 직렬로 연결합니다. 새 학습·확대 실험은 이번 요청에서 실행하지 않았습니다.

**모델·주장을 고정한 뒤 독립 평가를 확보한다 — 외부 작업 대기**
- 향후 개발에 노출하지 않은 기간/데이터를 분리하고 선택 규칙을 동결한 뒤 수량 이득과 시간 손해를 함께 검증해야 독립 일반화와 Contribution 판단을 보강할 수 있습니다. 현재 Taxi Test 재사용과 구분합니다.

## 원본·재현·검증 파일

- [전체 감사 JSON](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/analysis.json)
- [분리된 Validation/Test checkpoint 지표 CSV](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/checkpoint_comparison.csv)
- [소스·초기화 합성 증적](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/evidence/code_audit.json)
- [원자료·평가 대상·checkpoint 계보](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/evidence/data_audit.json)
- [Runtime·설정·실제 이력](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/evidence/runtime_audit.json)
- [평가 정의·합성 집계 검증](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_baseline_audit_20261008_v1/evidence/evaluation_audit.json)
- [기존 CNNGRU54 Test 원본](/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/titantpp_cnn_gru_test_20261005_v1/attempt2/retrieved/runs/test/yellow_trip_hourly__titantpp_cnn_gru54__seed42/receipt.json)
- [새 검색부 Test 보고서](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_retrieval_test_20261008_v1/README.md)

감사 작성 시 paper_research 브랜치는 `codex/contextual-retrieval-pilot-20261007`, HEAD는 `d803d1aa17ec9973888355230717387aae3f5d19`입니다. 과학revision·동결 원본·기존 사용자 변경은 보존했습니다. 이번 보고서의 commit은 별도 완료 응답에서 기록합니다.
