# v2 실제5080 진입 확인

상태: **5080_NATIVE_QUALIFICATION_FAILED_FITS_ZERO_V2_PRESERVED**. 실패한 v2 root·claim·lease·원본은 보존하며 자동 재시도하지 않는다.

## 확인된 사실

- 로컬에 보존된5080 health 호출은 helper OS exit1로 끝났다. 읽기 전용 진단은 `/var/lib/dpkg/lock`을 여는 단계에서 `PermissionError: [Errno 13]`을 기록했다.
- `HEALTH_DIAGNOSTIC_5080.json`은 mutation_attempted=false, training_failure_inferred=false다. 실제 native qualification 통과 증적은 아직 없다.
- 건강 판별이 끝나지 않았으므로 배포·qualification·실제 학습의 완료나 시작을 이 관측에서 추정하지 않는다.5090에 대한 QA 직접 호출은0이다.

진단 원문 SHA256: `4e5f00930a564ea5f958f633844b5efd5250015e86aa11ccb23edffd32ab735b`.

## 판정과 다음 확인

접근이 거절된 잠금 파일을 “잠금 없음”으로 처리하면 안 된다. 읽기 전용 잠금 상태 검증을 보완하고 원본 v2 코드·bundle·실패 증적은 유지해야 한다. 보완 후 현재5080 GPU/Runtime/유휴와 잠금 상태를 확인해야 native 단계로 진행할 수 있다.

이 문서는 전달된 로컬 증적의 QA다. QA는 SSH·GPU·학습을 직접 실행하지 않았고 Test 성과를 읽지 않았다.

## 권한 보완 후 실제 건강 조회 — 확인

관측 시각: 2026-10-12T17:31:06+09:00 (KST).5080 helper exit0 및 건강용 native probe 실제 child exit0를 확인했다. 이는 전체 실험 native qualification/admission이 아니다.

- 고정 GPU UUID 일치, GPU PID0, 기존 Runtime Python3.12.13/Torch2.11.0+cu130/CUDA13.0/cuDNN91900/NumPy2.4.4/Polars1.39.3 일치. driver595.99.02.
- reboot_required=false, package_locks=[], 새 owned root 없음.
- `unattended-upgr` PID1514가 감지돼 health status=waiting이다. 짧은 comm만으로 실제 updater 작업인지 상주 shutdown 대기 서비스인지 단정하지 않는다. exact argv 확인이 필요한 항목으로 root에 전달했다.
- 신규 배포·실험 qualification·fit 완료 증거는 아직 없다.

건강 원문 rawSHA `3a82f292d97d5605c88e3f14f8249cb01e1a92a694d5925851a0305b05ab6181`. 최초 실패 증적은 보존돼 있다.

## 실제 건강 통과와 배포·qualification 시작 — 확인

- 2026-10-12T17:37:37+09:00 (KST):5080 ready,GPU PID0/재부팅 요구0/실제 업데이트 프로세스0/패키지 락0. PID1514는 exact helper/argv/start_ticks469/S/PPID1 및 PreparingForShutdown=false 조건으로 idle shutdown observer 분류된 원문을 확인했다.
- 배포 reserve/upload/deploy 완료 원문이 생겼고, 2026-10-12T17:38:51+09:00 (KST)에 승인된5080의 qualify phase 시작을 확인했다. SSH helper exit0는 qualification subprocess 종료0와 다르며 phase_os_exit_code=not_yet_observed다.
- 실제 native qualification/admission 완료와 학습 시작은 아직 확인하지 않았다. QA 직접 SSH/GPU0,5090 승인·호출 증적0.
- ready health rawSHA `7e82312f5be486bd292b793173af185120cf78903488a94c431893dc8ebfdd95`, qualify-start rawSHA `1bd246c5957185fe0a5a13c654e923dbd5e2225e6dfec3f2ceb258f410673dd1`.

## 실제 qualification 실패 — 확인

- 원관측 시각: 2026-10-12T17:40:55+09:00 (KST).5080 snapshot의 모든 base64 원문 SHA를 대조했고 GPU PID=[] 및 qualify subprocess PID38442/PPID38435의 정확 argv·OS exit1, qualification/phase failure 원문을 확인했다.
- snapshot rawSHA: `ea6ea6715b0cd6474ba82f269feef0d91d60c6e59927fd0e2f538eb376f285d9`.
- root는 native.log의 `run_multilag_detail_execution.time_interface` line319에서 `KeyError: design_sha256`가 났다고 전달했다. 독립 로컬 코드 확인에서도 v2 계약에 그 key가 없고 봉인 source의 해당 함수가 필수로 접근하는 것을 확인했다. CPU synthetic/fake 검증이 실제 데이터 interface 통합을 충분히 포함하지 못했다.
- 이번 실패는 최초 fit 이전의 native 준비 계약 오류다. 과학 학습 결과나 numerical failure로 해석하지 않는다. 실제 fit0/합성 단계 전 중단은 root의 native.log 확인에 따른 것으로, 전체 qualification pass 증적은 없다.
- v2를 다시 실행하거나 기존 실패 파일을 고치지 않는다. root가 새 v3 계약/root에서 원래 frozen design provenance를 복원하고 실제 Train/Validation interface CPU 통합 검증을 추가한다. source131·모델·초기화·손실·데이터·selector를 바꾸는 작업으로 확대하지 않는다.5090 호출0 경계를 유지한다.
