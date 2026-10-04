# A100 SXM CNN·GRU seed42 탐색

**실험 범위를 9조건으로 고정한다 — 완료**

- 대상: 새 RunPod A100 SXM80GB 한 장에서 Taxi·Intermittent·RAF × CNN+MLP16·GRU54·CNN+GRU54 × seed42입니다. Instacart는 후속으로 유지합니다.
- 기존 MLP16은 다시 학습하지 않습니다. 각 데이터의 검증된 seed42 선택 checkpoint와 전체 Validation 결과를 비교 기준선으로 재사용합니다.
- Encoder1의 CNN은 현재 사건과 직전 두 관측 사건의 Q/K/V만 처리합니다. GRU54는 전체 중간 보정 모듈을 대체하며 과거 범위·활성화·잔차 크기도 바뀌므로 순환 연산 하나의 효과로 해석하지 않습니다.
- 수량·시간 출력부, log 수량 손실과 관측 정수 시간 확률질량 NLL, AdamW 설정, batch128, min40/max300/patience40, 최초 최소 전체 Validation raw RMSE checkpoint 선택을 유지합니다.
- 진행 중인 5080·5090의 폭8·12 동결 실행본은 변경하지 않습니다.

**원본 소스·자료·기준 checkpoint를 연결한다 — 완료**

- 실제 CPU 검증본117소스를 보존한 archive에서 복원하고 SHA를 대조했습니다. closure는 `40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd`입니다.
- 허용된 Train·Validation만 있는 기존 세 자료를 복사해 SHA·개체·수량·시간 모집단을 기존 계약과 대조했습니다. Test 행·예측·성능은 읽지 않았습니다.
- 기존 폭16 seed42의 원본 checkpoint3개와 선택 epoch 및 기록된 Validation replay SHA를 연결했습니다. 원본 재학습은 없습니다.
- 세 원본의 모델 경로·선택 epoch·초기화 출처·저장 텐서 SHA와 새 코드에서의 엄격한 복원을 CPU에서 확인했습니다. 이는 GPU 재추론 완료와 구분합니다.

| 데이터 | 기존 폭16 선택 epoch | 전체 Validation RMSE | MAE | Time NLL |
|---|---:|---:|---:|---:|
| Taxi | 27 | 82.499119 | 26.751941 | 0.679151 |
| Intermittent | 5 | 1.632470 | 0.713930 | 0.350936 |
| RAF | 9 | 33.983831 | 9.329766 | 3.555864 |

**실행 코드와 복원 가능한 증적을 검증한다 — 완료**

- 최신 Worker 29개와 임대·회수 제어 12개, 합계 41개 로컬 검증이 통과했습니다. 이 숫자는 네이티브 GPU 검증이나 실제 연구 학습의 통과 결과가 아닙니다.
- 준비 v1은 과학 소스117개, worker4개, 실행 제어 코드와 테스트·초안 계약을 148파일의 SHA archive로 보존한 기준선입니다. 당시 별도 임시 폴더로 복원한 코드에서도 40개 테스트가 모두 통과했습니다. 이 원본을 유지합니다.
- 배포 전 source import 경로 수정을 반영한 prelaunch v2는 150파일의 별도 SHA archive이며 유효성을 확인했습니다. 과학 소스117개와 worker4개를 유지하고 실행 제어 두 파일과 관련 테스트를 갱신했습니다. 자료·checkpoint·인증정보는 두 Git archive에 넣지 않았습니다.
- 로컬 Mac/Torch2.14에서 과거 폭16 소스와 새117소스가 만든 초기 텐서는 세 데이터 모두 동일했으나, 과거 Linux/Torch2.11 기록과는 SHA가 달랐습니다. 이후 A100/Torch2.11에서 과거 폭16 초기화 SHA가 세 데이터 모두 정확히 일치했고, 새 세 후보의 공통 텐서·난수 상태 일치 및 반복 초기화도 통과했습니다. 각 학습은 GPU에서 확인한 초기화 SHA를 그대로 사용합니다.
- 소스와 worker SHA, 정확한9조건·seed42·자료 범위 검사가 통과했습니다. 확정 GPU UUID·이번 비용 상한을 넣은 최종 실행 계약을 동결했습니다. 계약 SHA는 `46a0eb48d5207df800553f5d3508016ce9063d1dec9be2c5e84bfa87a1b1fdae`입니다.

**A100 결정성·메모리·속도를 검증한다 — 완료**

- 이번 캠페인의 총비용 상한 $100과 자동충전 금지가 승인됐습니다. 전용 Pod `okbl76hxfr7a9d`를 임대하고 146개 배포 파일의 SHA를 확인했습니다. native GPU 자격 검증을 통과해 동시 실행 수를 3으로 선택했습니다.
- 실제 배정한 A100 SXM Secure Cloud GPU 요금은 $1.59/h이며 저장료를 합친 실행 요금은 $1.596944/h입니다. CUDA13.0 가용성과 GPU 배정 확인을 실제 학습 검증 통과와 구분합니다.
- 조회한 계정 잔액 $53.0050935159는 이 Pod의 비용이 아닙니다. 과거 A100/PRO4500 예산과 합치지 않고 자동충전하지 않습니다. 총상한·현재 가용잔액에서 회수여유를 뺀 작업비·48시간 중 더 엄격한 마감으로 제한합니다. 계정 자동결제 설정 readback은 null이므로 미확인입니다.
- 승인 총상한 $100에서 회수·정리 여유 $20을 뺀 작업 상한은 $80이지만, 현재 잔액 $53.0050935159에서 같은 여유를 남겨 실제 작업 상한을 $33.0050935159로 줄였습니다. 실제 실행 요금으로 계산한 준비·학습 기한은 **2026-10-05 15:17:45 KST**이며 약20시간40분입니다. 현재 잔액의 다른 사용은 조기 중단을 유발할 수 있으며 이 기한은 9조건 완료 시간의 보장이 아닙니다.
- A100의 Torch2.11.0·CUDA13.0 환경을 엄격히 확인했습니다. 신규 세 후보의 전체 batch128, 최대 길이256/84에서 joint loss의 forward/backward, 반복 결정성, checkpoint 복원, 인과성·padding·관측 경계 검사가 통과했습니다.
- 같은 9개 합성 작업의 admission probe에서 단독 실행은 6.8141174316초, 3병렬은 3.5417716503초였습니다. 3병렬의 최대 기기 메모리 사용량은 7,662,993,408bytes이며 A100 메모리는 80GiB입니다. 이는 병렬 실행 허용을 위한 짧은 검증이며 논문 효율이나 실제 300epoch 학습 속도의 근거가 아닙니다.
- 기존 폭16 checkpoint3개의 전체 Validation replay도 같은 A100 환경에서 통과했습니다. 원래 GPU와 다른 환경에서 같은 자료·checkpoint를 확인한 것이며 새로운 독립 평가는 아닙니다.

**9조건의 학습과 결과 연결을 진행한다 — 진행 중**

- **2026-10-04 18:49:44 KST 관측**: 완료0, 실패0, 진행 중3(학습2·준비1), 대기6, 미확인0입니다. 실제 소유 worker와 GPU PID `1159/1161/1163`이 선언된 동시 실행 수3과 일치했습니다.
- CNN+MLP16의 Taxi·RAF는 저장 checkpoint로 학습을 확인했습니다. Intermittent는 소유 worker와 GPU 등록이 확인됐지만 아직 checkpoint가 없어 준비 중으로 구분합니다. GRU54와 CNN+GRU54의 세 데이터씩, 총6조건은 대기 중입니다.

| 데이터 | 후보 | 현재 상태 | 저장 / best epoch |
|---|---|---|---:|
| Taxi | CNN+MLP16 | 학습 중 | 1 / 1 |
| RAF | CNN+MLP16 | 학습 중 | 2 / 1 |
| Intermittent | CNN+MLP16 | 준비 중, checkpoint 미확인 | — / — |

- 전체 큐의 종료 예상 시간은 미확정입니다. 대기 중인 6조건의 실제 속도와 대기시간이 확인되지 않았으며, 비용·시간 기한을 완료 예상 시각으로 표시하지 않습니다. 자동 재시도·재개·추가 임대는 하지 않습니다.
- 전체 Validation RMSE·MAE·시간 NLL, 큰 수량 RMSE를 원래 선택 checkpoint의 동일 epoch에서 읽습니다. 큰 수량 기준은 원래 Train 경계 초과: Taxi3449, Intermittent187, RAF200입니다.
- Train은 기존 수량 구간별 고정 표본과 N/n 가중치를 사용한 진단이며 전체 Train 모집단 결과로 부르지 않습니다. 학습 과정에서 변하는 파라미터의 epoch 평균 손실과 고정 checkpoint 재추론도 구분합니다.
- 후보별 데이터마다 폭16 대비 개선·손해를 모두 표시합니다. 수량 개선과 시간 악화는 교환 관계로 보고하며, 단일seed 결과로 3seed 안정성·통계적 우월성·논문 효율을 주장하거나 기준선을 자동 변경하지 않습니다.
- 완료는 terminal success·소형 기록 및 원본 checkpoint SHA·selected/last 전체 Validation replay를 확인한 경우만 집계합니다. CPU binary 재추론 감사는 별도입니다.

**원본을 회수하고 소유 Pod의 과금을 종료한다 — 다음 작업**

- 회수 파일을 원격 manifest와 SHA 대조한 뒤 이번 캠페인이 새로 만든 Pod만 stop/delete하고 목록 부재를 확인합니다. 미회수 원본이 있으면 삭제하지 않습니다.
- 비용은 실제 배정 요금·경과시간에 의한 추정과 제공자 청구서를 구분합니다. 로컬 manager와 Mac 통신 장애 중 provider 종료는 보장되지 않으며 요청·readback·실제 적용을 각각 기록합니다.
- 긍정적인 후보의 5080 후속 seed 확대는 이번9조건 완료 결과와 5080의 기존 학습 종료를 확인한 뒤 별도 계약으로 정합니다.

과학 준비 원본은 `search_artifacts/titantpp_cnn_gru_a100_seed42_20261004_v1/execution_contract.draft.json`, `design.json`, `expected_initialization.json`, `verification/cpu_data_admission.json`, `preparation_science_receipt.json`입니다. 실행 허가와 GPU검증 완료 증거가 아닙니다.

준비 당시 소스·실행 코드 복원 증적은 `paper/reproducibility/snapshots/cnn_gru_a100_seed42_preparation_20261004_v1/manifest.json`이며 원본으로 보존합니다. 배포 전 최신 제어 코드와 41개 로컬 검증의 증적은 `paper/reproducibility/snapshots/cnn_gru_a100_seed42_prelaunch_20261004_v2/manifest.json`입니다. 다음 명령은 최신 v2의 로컬 소스 검증·복원과 합성/계약 테스트만 실행하며 Pod를 임대하거나 학습을 시작하지 않습니다. 복원 대상 폴더는 새 경로여야 합니다.

```sh
python3 paper/reproducibility/archive.py verify paper/reproducibility/snapshots/cnn_gru_a100_seed42_prelaunch_20261004_v2
python3 paper/reproducibility/archive.py restore paper/reproducibility/snapshots/cnn_gru_a100_seed42_prelaunch_20261004_v2 /private/tmp/titantpp-cnn-gru-a100-new-restore
python3 -m pytest /private/tmp/titantpp-cnn-gru-a100-new-restore/worker/tests/test_worker.py /private/tmp/titantpp-cnn-gru-a100-new-restore/control/tests/test_control.py -q
```

**시작 증적과 시간별 관측을 보존한다 — 완료**

- [native GPU qualification](native_qualification.json)은 원격 원본과 정확한 SHA가 일치합니다. 이 receipt의 `scientific_training_launched=false`는 자격 검증 완료 당시 기록이며, 이후 학습 진입은 [실제 PID·checkpoint 확인](launch_confirmation.json)에 별도로 보존했습니다.
- [초기 관측](initial_observation.json)은 2026-10-04 18:49:44 KST의 실제 관측입니다. [회수 관리자](initial_provider_manager_records.json)의 소유 PID·명령과 최근 기록을 함께 확인했습니다.
- [시간별 모니터](SCHEDULER.md)를 ACTIVE로 연결했습니다. 자동화 ID는 `titantpp-cnn-gru-a100-seed42`이며 생성 증적은 [scheduler receipt](scheduler_receipt.json)입니다.
