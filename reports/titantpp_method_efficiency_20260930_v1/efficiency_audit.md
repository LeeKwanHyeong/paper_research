# Titans-MAC 대비 효율 근거 — 기존 기록 감사 완료

대상: `paper_research` 로컬 validation 기록과 당시 source revision. 2026-09-30 작성. 이번 작업은 원격 접속·GPU 측정·학습·checkpoint replay를 수행하지 않았다.

**결론: 기존 MAC 기록은 온라인 갱신을 포함한 adapter의 비용이 컸다는 과거 근거로 재사용할 수 있다. 현재 채택한 TitanTPP가 MAC보다 몇 배 빠른지는 아직 직접 입증되지 않았다.** 학습 속도에 초/epoch를 쓰는 것은 타당하지만, 동일한 처리량과 timer 범위를 맞춰야 한다.

## 1. 기존 MAC 자료로 사용할 수 있는 값

| 데이터 | 과거 B0 초/완료 epoch | 과거 MAC 초/완료 epoch | MAC/B0 | B0/MAC 완료 epoch |
|---|---:|---:|---:|---:|
| Taxi | 7.6847 | 52.7067 | 6.8587배 | 42 / 46 |
| Intermittent | 107.0674 | 583.2923 | 5.4479배 | 240 / 229 |

이 수치는 당시 `summary.elapsed_seconds / completed_epochs`다. 과거 trainer revision `08e59880cd61cbd27cec40aa04636452b87bebfc`의 timer를 확인했다. epoch loop 직전에 시작해 매 epoch의 train·validation·기록·저장을 거치고, loop 이후 best-state validation 및 checkpoint 저장 뒤에 끝난다. 따라서 **완료 epoch당 상각된 실행 시간**이며, 순수 train-only 시간이나 warm-up 제외 epoch 중앙값이 아니다. 초기 모델·loader 준비는 timer 시작 전이다. elapsed에 포함되지 않은 외부 작업까지 합산한 전체 캠페인 시간도 아니다.

원본: `paper/results/count_aware_titantpp_mac_b1_audit_20260830_recovery1/historical_cost.csv`. 이번 재계산: [historical_costs.csv](historical_costs.csv). 역사적 RAF 기록은 현재 세 데이터의 비용표에서 제외했다.

별도의 짧은 profiler 기록은 다음과 같다.

| 데이터 | B0 target_outputs 중앙값 | MAC target_outputs 중앙값 | B0 peak allocated | MAC peak allocated |
|---|---:|---:|---:|---:|
| Taxi | 5.284 ms | 35.019 ms | 364.889 MiB | 74.295 MiB |
| Intermittent | 5.276 ms | 33.737 ms | 364.889 MiB | 145.053 MiB |

**이 표는 ‘순수 backbone inference’라고 이름 붙이지 않는다.** profiler source revision `cc0382a21f3d1e37e41692bb8dd673fc01582c59`와 원본 manifest SHA가 일치한다. `benchmark_validation_forward`는 batch를 loader에서 받은 후 GPU 동기화·timer를 시작하고, device 전송과 `target_outputs`의 encoder·head·target loss 계산을 포함해 측정한다. loader 대기는 제외된다. 다섯 batch 중 첫 회를 cold, 나머지 네 회를 steady로 요약한 짧은 표본이다. `estimated_compile_overhead`는 cold−steady 차이일 뿐 compile만 분리한 실측이 아니다. 모델은 no-grad 평가를 하지만 MAC 내부의 관측 메모리 갱신은 그 forward 경로의 일부다.

MAC의 peak allocated가 더 낮았던 반례를 보존한다. 서로 다른 profiler/학습 peak를 섞지 않으며, 이 기록으로 현재 TitanTPP의 메모리 절약을 주장하지 않는다. allocated·reserved·parameter 수는 서로 다른 양이다.

## 2. 현재 TitanTPP 자체의 기록

| 데이터 | GPU / 단독 비교 가능 seed | 실행 초/완료 epoch, 평균 ± 표본 SD | 학습 peak allocated | Parameter |
|---|---|---:|---:|---:|
| Taxi | 5080 / 42·52·62 | 13.015 ± 0.011 | 1,869.468 MiB | 96,003 |
| Intermittent | 5080 / 42·52·62 | 123.096 ± 1.186 | 1,869.468 MiB | 96,003 |
| Instacart | 5090 / 42만 | 245.149, n=1 | 410.045 MiB | 83,715 |

학습 peak는 원래 summary 기록이며 위의 과거 no-grad profiler peak와 직접 비교하지 않는다. 첫 두 행은 seed별 `fit_elapsed/completed_epochs` 세 값의 평균 ± **seed 간** 표본 SD다. 개별 epoch 시간의 변동이나 confidence interval이 아니다. Instacart는 positional embedding 길이가 64여서 parameter 수가 다르다(Taxi/Intermittent256). 동일한 구성 규칙을 사용한다.

현재 timer도 `train_one`의 epoch loop부터 종료 후 내부 selected validation·저장까지 포함한다. 캠페인의 별도 selected/last endpoint replay, 사전 qualification, 배포 시간을 이 수치에 더하지 않았다. 원래 기록 시간을 추정 보정하지 않았다.

Instacart seed52는 병렬 실행, seed62는 병렬·재부팅 복구 조건이다. 둘 다 원래 기록과 함께 [current_mlp_costs.csv](current_mlp_costs.csv)에 남겼지만 단독 속도 평균에서 제외했다. 특히 seed62 `elapsed_seconds`는 복구 후 구간이므로 전체 75epoch로 나누어 속도로 제시하지 않았다.

## 3. 직접 비교에 남는 불일치

| 항목 | 확인 사실 | 현재 MLP↔MAC 주장에 미치는 영향 |
|---|---|---|
| 비교 모델 | 과거 비용의 상대 모델은 B0, 현재 대표는 History MLP | 과거 5.45/6.86배를 현 TitanTPP 가속 배수로 옮길 수 없음 |
| 시간 head·loss | 과거 legacy clamped head, 현재 heteroscedastic lognormal + 정수 관측 질량 | 연산량·gradient·학습 동작이 다름 |
| 선택/조기 종료 | 과거 joint objective, 현재 raw quantity RMSE | 전체 학습 시간 및 상각 평균의 구성과 정확도 비교에 영향 |
| 실행 장치 | 과거 Taxi shard 계약은 RTX5090-server, 현재 Taxi는 5080 | 같은 물리 GPU 비교가 아님. 과거 계약 host 표기는 UUID/부하 실측의 대체물도 아님 |
| 과거 Intermittent 장치 증거 | recovery5080 역할 기록이 있으나 일부 상위 계약 host는 MacBook, 요약에 UUID/runtime 없음 | 배정 기록과 실제 물리 장치·단독 점유 보장을 구분 |
| 데이터/입력 부하 | 안정화 MAC52/62는 같은 data/split SHA·target 수·batch·max sequence 설정 | 주요 workload 설정은 재사용 가능하나 실제 측정 batch 순서·길이·padding까지 맞췄다는 증거는 아님 |
| MAC 내부 정책 | 과거42 unbounded, 안정화52/62 inner clip1 | 하나의 동일 3seed 설정으로 합칠 수 없음 |
| 비교 목적 | 과거 elapsed, 짧은 target_outputs, 현재 train peak가 혼재 | 각 timer와 작업을 별도 표로 보고해야 함 |

데이터 계약 호환성 근거: [기존 6조건 감사](../titans_mac_reuse_audit_20260928_v1/mac_current_compatibility.csv). 이 문서의 과거 학습 진행 상태 서술은 당시 이력이며, 현재 모든 승인 학습이 종료된 사실과 별개다.

## 4. 연산 경로의 비교로 지금 말할 수 있는 것

| 경로 | Titans-MAC 이벤트 adapter | 채택 TitanTPP |
|---|---|---|
| 시간·수량 이벤트 입력 | 이벤트 wrapper로 적응 | 같은 문제의 log1p 입력 |
| 이력 표현 | attention과 neural memory의 retrieval/update 경로 | 두 causal encoder 사이 bottleneck residual |
| 온라인 상태 | 관측에 대한 associative gradient, update/forget/momentum 및 neural-memory parameter 상태 | forward-local hidden/gather, 온라인 parameter 갱신 없음 |
| 고정 학습 bank | MAC 구성의 persistent memory | encoder persistent16 및 static prototype64 유지 |
| 구현 해석 | 원 논문의 메커니즘을 사건 입력에 적용한 로컬 adapter | 원본 Titans의 모든 구성·벤치마크를 그대로 축소했다는 주장 아님 |

‘온라인 갱신 계산 경로를 생략했다’는 구현으로 확인한 사실이다. ‘실행 시간이 감소한다’는 장치·구현별 실측 주장이고, ‘같은 정확도에서 비용이 감소한다’는 별도의 학습 결과 주장이다. 세 주장을 혼동하지 않는다. 병목이 달라질 수 있고 full causal attention은 여전히 길이에 대해 이차 비용을 포함한다.

## 5. 부족한 측정의 최소 범위와 승인 판단

**권장: 먼저 짧은 동일 부하 profile만 보완한다 — 승인 필요, 미실행**

- 목적: 현재 TitanTPP와 MAC adapter의 **step 비용·평가 경로 지연·메모리**를 같은 장치에서 직접 비교한다. 장기 수렴이나 정확도 우열은 측정 목표가 아니다.
- 대상: 개인 5080 한 장 단독, 현 Native Runtime 유지. 모델은 채택 MLP와 inner-clip1을 명시한 MAC adapter 두 개, 데이터는 Taxi·Intermittent의 **train만**. common input/head/loss/batch128/정밀도/입력 mask를 맞춘다. 기존 결과 checkpoint를 덮어쓰지 않는 별도 scratch 경로를 사용한다.
- 고정 표본: 각 train loader에서 동일 seed42 순서의 32 batch와 준비용 5 batch. target ID와 실제 길이·padding histogram을 저장한다. 모델별 같은 순서·같은 입력을 주고 MAC의 sample-local memory reset/write 규칙은 보존한다. MAC update를 끈 비교로 대체하지 않는다.
- 반복: 데이터×모델 각각 3회. 각 반복은 같은 초기 state에서 준비용5+측정32 optimizer step을 실행한다. 총 최대444 step으로 **성능 학습에 합치지 않고 폐기**한다. 평가 모드 forward+head 경로는 동일32 batch×3회, 별도 initial state에서 측정한다. batch128 지연이며 단일 요청 지연이라고 부르지 않는다.
- 측정: host→device 포함/제외 경계를 분리하고 synchronized wall time·처리 targets/s·각 반복 중앙값/사분위수·allocated/reserved peak·parameter 수를 기록한다. loader 대기·checkpoint 저장은 별도이며, cold/compile 시간을 따로 남긴다. 각 모델·반복을 별도 process로 실행해 allocator peak를 분리한다.
- 한도 제안: 준비 포함 **총 GPU 점유 30분**, 동시 worker 없음, 실패 자동 retry 없음. finite 값·mask·source/head 정합성 검증 실패 시 본 측정 차단. 이 한도에서 끝나지 않으면 미완료를 보고하고 자동 연장하지 않는다. 추가 임대비0, 전력은 미측정이다.
- 승인 전 준비할 계약: 정확한 MAC wrapper/source SHA, 두 모델 공통 head·loss, batch manifest, 프로파일러 timer, GPU UUID·Runtime·단독 점유 확인 방법과 중단 guard를 고정한다. 위는 검토 가능한 측정 범위이며 **실행 permit은 아니다**.

이 최소 profile 결과를 **실측 초/epoch로 환산해 표시하지 않는다**. `측정32 batch × 전체 step수`는 추정치에 불과하다.

**논문에 실제 초/epoch가 꼭 필요할 경우 — 별도 후속 승인 판단**

- profile로 측정 가능성을 확인한 뒤, 두 데이터×두 모델에서 **전체 train 1회 cold + 2회 반복**(최대12 epoch)을 별도 scratch에서 계측한다. 이는 최소 전량 반복이며 정밀한 분산 추정은 아니다.
- 전체 target 수/step 수는 Taxi38,393/300, Intermittent393,824/3,077로 동일하게 고정한다. train-only와 train+validation을 섞지 않는다. 반복 epoch마다 실제 처리량과 시간 기록을 남긴다.
- 실제 승인 요청 전 profile 실측으로 최대 점유시간을 산정한다. 현재 과거 MAC 시간으로 새 실행 예산을 확정하거나 30분 한도에 전량12epoch가 들어간다고 약속하지 않는다.

**동일 정확도까지의 효율 또는 MAC보다 높은 정확도 — 이번 범위 밖**

짧은 profile이나 세 epoch로는 검증할 수 없다. 현재 head·선택 규칙의 별도 MAC 장기 비교가 필요하며 기존 9조건 미승인 계획을 자동 실행하지 않는다. 반대로 방법 설명과 기존 TPP 대비 수량 결과를 쓰는 데 장기 MAC9조건을 선행 필수로 둘 이유도 없다.

## 검증·재현

`/usr/local/bin/python3 reports/titantpp_method_efficiency_20260930_v1/audit.py`

새 source나 기존 scientific 결과를 변경하지 않는다. 비용 재계산·8개 방법 source SHA·manifest closure·작은 CPU 수식 동치를 검사한다. 검증 결과 [verification.json](verification.json), 기계 판독 판단 [efficiency_audit.json](efficiency_audit.json), 읽은 파일 [sources.json](sources.json). 5090 최종 binary/source 회수 감사는 이 작업에서 완료한 것으로 표시하지 않는다.
