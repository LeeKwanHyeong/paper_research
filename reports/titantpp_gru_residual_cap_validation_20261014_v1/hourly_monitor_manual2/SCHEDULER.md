# GRU 보정량 제한 후보 9개: manual2 종료 관측과 Validation 후속 집계

## 현재 상태와 승인 범위

- 사용자는 새 격리 root에서 5080·5090의 수동 qualification을 각각 한 번 승인했고, 통과하면 같은 9개 학습을 즉시 시작하도록 승인했다. 기존 실패 root·claim·실패 기록과 CNN·FULL 결과는 보존한다.
- 최초 관측 **2026-10-14 17:13:32 KST**에서 두 host의 `training_running`, `identity_verified=true`, `live_process_probes_verified=true`가 확인됐다. 고유 9개는 **완료0·진행2·대기7·미확정0**이며, 5080은 Taxi seed42, 5090은 Intermittent seed42가 진행 중이다. 최초 회차 `rounds/0d55db12ca4f40229a34c2d40a5d316c.json`의 rawSHA는 `de76861e8418509a2d37ddd5b39b380dfba4f2454d62afe44e71311a2178c877`이다. 이 원관측 시각을 이후 관측 시각으로 바꾸지 않는다.
- 대상은 고정 후보 `titantpp_cnn_gru54_residual_cap025`의 세 데이터셋×seed42·52·62, 총 9개뿐이다. 5080은 Taxi3·RAF3·Intermittent62의 7개, 5090은 Intermittent42·52의 2개이며 host마다 한 fit씩 진행한다.
- 목적은 손실·head·학습 예산·학습 파라미터 수를 고정한 상태에서 GRU 잔차 보정량 제한이 기존 FULL 대비 Validation 오류를 줄이는지 확인하는 것이다. CNN과의 비교는 별도로 유지한다. 이 작업은 Test 순위 반전 이후의 탐색적 후속 실험이며, Validation 개선을 Test 일반화 개선이나 반전 원인 입증으로 해석하지 않는다.

- 후속 자동화 **`gru-cap-9-validation`**은 ACTIVE로 등록됐다. 시작 기록 `STARTED_RECEIPT.json` rawSHA는 `21fb5197464bfeb78ff8e0a349952f29c51f8725fc36f1750c7453ef2235cb4b`이다.

## 고정 경로와 봉인

프로젝트 root는 `/Users/igwanhyeong/PycharmProjects/paper_research`다. 실행 전에 `TEST_SESSION_PROTOCOL.md` 0·1·9·12절과 이 문서를 확인한다.

- bundle: `search_artifacts/titantpp_gru_residual_cap_validation_20261014_v1/execution_manual2`
- 실행 계약 rawSHA: `274ff80ef771788dc4652749bd2875716c7ddeb59ba0a1a0e497358545fe1fdc`
- 실행 계약 canonical SHA: `9a7e553730e36879fa89f3732c5d4a5493ed60f6bfcb1a4f826cb0990693622f`
- execution integrity rawSHA: `7adc2f6a31a98e4dbe10809316711a67bc7ade2284583bed4a4699ba3d402516`
- `hourly_monitor_manual2/monitor_integrity.json` rawSHA: `5d798de08dd9c87c0ce3783230d88b86866b9492de74738808fdf240a54aeafa`
- `execution_manual2/collection_integrity.json` rawSHA: `99daf73bb1e8d895292adb5a7c7b234fe27b6f1e60af8b6c9a02e084be8b8ecd`
- 고정 원본 observer rawSHA: `89f5794e7e85a585cbdd1793b1320c08240245a519ef39836158743df5421c04`
- 고정 원본 collector rawSHA: `1dd9bcb96643999f0cff5f2531dcde78477440a2c45bb762e7a89f228bf6fa1e`

원격 root는 각각 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_gru_residual_cap_validation_20261014_v1_5080_manual2`와 `_5090_manual2`다. GPU·Runtime·입력·source134·고정 과학 계약·기존 18개 CNN/FULL Validation 기준선과 실패 증적은 helper가 기존 검증 경로로 확인한다. mismatch이면 추가 원격 호출·덮어쓰기를 하지 않고 사용자 조치를 보고한다.

## 회차당 한 번의 후속 실행

다음 명령을 프로젝트 root에서 회차당 정확히 한 번 실행한다. 별도 observer·SSH·반복 polling이나 helper 재실행을 추가하지 않는다.

```bash
PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_gru_cap_manual2 --monitor-integrity-sha 5d798de08dd9c87c0ce3783230d88b86866b9492de74738808fdf240a54aeafa --collection-integrity-sha 99daf73bb1e8d895292adb5a7c7b234fe27b6f1e60af8b6c9a02e084be8b8ecd
```

wrapper는 고정 원본 관측·회수 코드의 SHA를 확인하고 독립 namespace에서 새 launcher와 정확한 manual2 root만 연결한다. 두 봉인에는 wrapper와 manual2 launcher도 포함되며, 회수 봉인까지 검증한 뒤 관측에 들어간다. 원본 코드는 변경하지 않는다.

- live host의 status SSH는 회차당 최대 1회다. 검증된 종료 cache는 SSH0으로 원관측 시각을 유지한다. cache는 해당 root의 과거 종료 증적이며 현재 GPU 전역 유휴 증거가 아니다.
- 정확한 PID·PPID·argv·GPU UUID·허가·claim·lease·native qualification bridge·checkpoint/history/timing과 실제 supervised child exit를 검증한다. 조회 실패·신원 미확정을 학습 실패나 완료로 바꾸지 않는다. pipeline wrapper 자체 OS exit는 직접 관측하지 않았다는 사실을 유지한다.
- 회차 JSON·Markdown·latest의 rawSHA와 포인터를 확인한 뒤 같은 회차에 collector를 한 번 호출한다. 학습 전체 종료가 미검증이면 collector는 원격0·원본 회수0·Test0으로 대기한다.
- 실제 고유 9개의 worker exit0·terminal binary SHA와 두 host dispatcher exit0·train_complete·소유 PID/GPU process 부재가 모두 검증돼야 회수할 수 있다. 양쪽 종료 cache와 모든 로컬 검증 근거를 재검증한다.
- 원본 회수는 host별 읽기 전용 SSH tar stream 1회로 terminal manifest에 묶인 정확한 파일만 가져온다. 두 archive의 구성과 전체 SHA가 통과해야 새 로컬 경로에 배타적으로 추출한다. 기존 intent·부분 회수·실패가 있으면 자동 재시도하지 않는다.

## 보고와 완료 판정

기본 주기는 1시간이다. 상태가 그대로이거나 조치가 필요하지 않은 동안에는 조용히 유지한다. 의미 있는 진행 변화·완료·실패·사용자 조치 필요 시만 이 채팅에 알린다. 관측 결과에 실제 KST 시각, GPU·데이터셋·seed·완료/진행/대기/미확정과 저장/best epoch를 구분한다.

ETA가 필요하면 해당 fit의 최근 최대 10개 완료 epoch의 finite 양수 elapsed_seconds 중앙값만 사용한다. best 미갱신과 속도 유지 조건의 `min(300,max(40,best+40))`까지 잔여와 300epoch까지 잔여를 구분한다. 다른 fit·GPU 속도를 차용하지 않는다. 조건36시간과 기존 host168시간 절대 기한은 연장하지 않으며 ETA와 구분한다. 300epoch 대안이 남은 기한을 넘으면 현 기한 안에 보장되지 않는다고 표시한다.

로컬 회수 뒤 같은 고정 선택 checkpoint에서 전체 raw 수량 RMSE·MAE, 큰 수량 RMSE·MAE, 전체·큰 수량 recorded positive-integer TimeNLL의 여섯 Validation 지표를 집계한다. 선택 epoch는 전체 Validation raw 수량 RMSE의 첫 strict finite 최소이며 동률은 최초 epoch다. 세 seed의 평균·표본SD와 CAP−FULL·CAP−CNN의 seed별 차이를 따로 제시하고 악화·동률도 포함한다. Intermittent42·52는 5090/cuDNN92000, seed62는 5080/cuDNN91900이므로 Runtime 차이를 명시하고 시간은 host별로 분리한다.

최종 완료는 원격 학습 종료와 구분한다. `execution_manual2/retrieval/final_receipt.json`의 `status=complete`, candidate_fit_count9·seed_count3·baseline_reference_count18, archive·원본·선택 checkpoint·terminal manifest·전체 로컬 SHA와 `THREE_SEED_VALIDATION.json`/`.md`의 최종 SHA가 모두 검증되어야 한다. 최종 Validation 비교를 이 채팅에 보고한 뒤 이 후속 자동화 `gru-cap-9-validation` 하나만 삭제한다. 다른 자동화는 변경하지 않는다.

## 허용하지 않는 동작

후속 자동화는 관측·허가된 원본 회수·고정 Validation 집계와 별도 로컬 보고만 수행한다. deploy/qualification/train 재실행, 새 학습·새 seed·rho 탐색, 자동 retry/resume, checkpoint 재선택, Test 열람·평가, 기존 원본 덮어쓰기, claim/failure 삭제, kill, 공용 Runtime·driver·인증 변경, 원고 수정은 허용하지 않는다. 미확정·부분 실패이면 원문 증적과 필요한 사용자 조치를 알리고 기존 기록을 보존한다.
