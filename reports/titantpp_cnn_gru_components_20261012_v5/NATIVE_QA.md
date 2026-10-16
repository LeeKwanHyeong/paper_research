# v5 실제 5080 native 독립 QA

상태: `5080_NATIVE_PASS_INITIAL_THREE_SCIENTIFIC_FITS_VERIFIED`. QA는 로컬 원문을 검증했으며 직접 SSH/GPU 호출은0이다.

## Review Result: PASS — native 및 동시성 gate

### CRITICAL
- 없음.

### HIGH
- 없음.

### MEDIUM
- 없음.

### LOW
- 없음.

2026-10-12 18:27:57.118 KST 원관측의 `control/qualification_cache/5080/snapshot.json` raw SHA `dea1c67f8417a3e1c3bb68c0f989d8b545a5363cb5f4d59efa6425cb5b5a6df9`와 embedded26개 raw SHA를 확인했다. 봉인 `qualification_binding`을 독립 재실행해 PASS했다.

- qualification receipt raw SHA: `b97158aa56d8f16bbb86d8c42a8811ec6c6088ad096fceafa7dd4195a753592f`.
- native child guard40812/parent40798 실제 exit0, 정확 qualification.py argv/contract binding 확인.
- 세 capacity GPU PID40906/40908/40909가 동시에 존재했으며 정확 guard40904/40905/40907 및 argv, PPID/process group/start ticks를 확인했다.
- FULL/CNN-only/GRU-only 세 guard가 모두 actual child exit0를 남겼다.
- 최대 관측 GPU memory fraction은0.47850088940685764이며 최종 GPU PID는 비었다.
- qualify campaign PID40798/PPID40782 실제 OS exit0, complete phase와 source131/contract/GPU 일치.
- 5080 Runtime은 Python3.12.13/Torch2.11.0+cu130/CUDA13.0/cuDNN91900, 고정 GPU `GPU-7500aa5a-f7b0-bf7c-3159-13852192cbc6`다.
- 실제 자료 interface/초기화, 3seed synthetic, causality가 통과했다. native/admission의 과학 optimizer update는0이며 실제 Validation 예측/Test 평가를 수행하지 않았다. 합성 업데이트는 학습 완료 건수로 세지 않는다.

training permit은 이 host의 receipt rawSHA와 실제 exit0 및 원래 start permit에만 결합됐다. 5090 receipt/승인은 없다. 18:28:26.126 KST train phase 시작 응답은 transport0이며, phase OS exit는 `not_yet_observed`다. 시작 명령만으로 실제 fit 또는 저장 완료를 주장하지 않고 후속 로컬 snapshot을 기다린다.

v2/v3/v4 실패 원문은 그대로 유지하며 각각 과학 fit0이다. pipeline 자체의 OS exit코드는 직접 관측하지 않았다는 구분을 유지한다.

## 완료 조건

- [x] native child 실제 exit0와 source/Runtime/GPU 검증
- [x] 세 동시 GPU owner 및 각 capacity child 실제 exit0
- [x] qualifier 실제 exit0 및 종료 후 GPU idle
- [x] 5080 단독 training permit/원래 절대lease 결합
- [x] 최초 scientific fit의 PID/argv/GPU와 저장 상태 확인
- [ ] 24개 scientific fit 종료 및 원본 회수

## 최초 scientific fit 관측: 2026-10-12 18:29:34.274 KST

로컬 snapshot `control/status/5080/6321bbd15c46451aa97ca8c90a466019.json` raw SHA `7d3ba1d323c67ec91ac18569d363ab8bc6c689deae21522e3c8a291c6dd22489`. embedded raw SHA, contract/source/GPU, 정확 PID/PPID/process group/argv, active leases를 독립 확인하고 `training_status`를 재실행했다. 결과 `training_running`, `identity_verified=true`, active3/완료0/전체24다.

| 자료·seed | 구조 | GPU PID / guard PID | 실제 저장 / best | 상태 |
|---|---|---|---|---|
| Taxi42 | 보정 없음 | 41390 / 41389 | 1 / 1 | 실제 학습 중 |
| Taxi42 | CNN만 | 41392 / 41391 | 1 / 1 | 실제 학습 중 |
| Taxi42 | GRU만 | 41394 / 41393 | 저장 전 / 미정 | 실제 학습 중 |
| Taxi42 | CNN+GRU | 없음 | 미시작 | 앞의3개 종료 후 대기 |

세 guard의 부모는 campaign41374이며 campaign의 부모는41361이다. NONE/CNN 저장 receipt의 `last_epoch_state.pt`, history, timing, exposure, train.log SHA 검증이 모두 true다. best1은 저장된 full Validation history의 strict earliest selector 값이며, 최종 selected binary는 아직 없다. 조회 시점의 저장 상태를 학습 완료 또는 최종 성과로 해석하지 않는다. 전체 queue ETA는 미확정이며 5090 조건12개는 별도 사용자 명령 대기다.

## 마지막 초기 확인: 2026-10-12 18:32:30 KST

추가 로컬 snapshot `control/status/5080/d5fd7d38c1604505a12a06b85628e85b.json` raw SHA `e2314d0f7f00be64801e33472d978efdf5b93f85ec6129c1af9c2ed660e11b47` 및 embedded raw SHA를 독립 확인했다. 동일3개 GPU PID/guard/argv/lease가 검증됐으며 세 조건 모두 checkpoint bundle의 state/history/timing/exposure/train.log SHA가 true다.

| Taxi seed42 구조 | 실제 저장 epoch | Validation history best epoch | 상태 |
|---|---:|---:|---|
| 보정 없음 | 5 | 4 | 학습 중 |
| CNN만 | 5 | 4 | 학습 중 |
| GRU만 | 2 | 2 | 학습 중 |
| CNN+GRU | 미시작 | 미정 | 앞의3개 자연종료 대기 |

최종 selected binary는 아직 없으며 자연종료 전의 저장 상태다. 완료0/진행3/전체24, queue ETA미확정이다. 본 회차의 초기 실행 확인은 여기서 종료하며 추가 SSH·지속 polling·자동화를 수행하지 않는다. 5090은 별도 사용자 명령 대기, 현재 원격0이다.
