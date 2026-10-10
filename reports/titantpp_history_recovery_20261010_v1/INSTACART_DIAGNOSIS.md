# Instacart 5080 관측 오류와 완료 원본 복구

상태: 완료. Instacart 학습 9/9와 두 서버의 원본 보존·Validation 집계가 완료됐다. GPU 관리 라이브러리 문제는 별도의 미해결 서버 운영 문제다.

## 확인된 원인

2026-10-10 11:06:19 KST 읽기 전용 진단에서 `nvidia-smi`가 exit18과 `Failed to initialize NVML: Driver/library version mismatch`를 반환했다. 로드된 NVIDIA 커널 모듈은 595.84, 사용자 공간 NVML은 595.99.02이다. `/proc/driver/nvidia/gpus/0000:01:00.0/information`의 UUID는 승인된 5080 GPU와 일치했다. 지정된 라이브러리 경로에는 맞는 595.84 NVML이 남아 있지 않았다. 버전이 바뀐 계기나 주체는 확인하지 않았다.

기존 observer는 GPU 조회를 먼저 수행하고 `check=True`에서 중단하므로, 그 뒤의 소유 프로세스·종료·checkpoint 읽기까지 도달하지 못했다. 학습이 실패한 증거가 아니다. 실제 5080 supervisor는 **2026-10-10 06:18:27 KST exit0**으로 종료됐고, 마지막 CNN+GRU54 seed62는 **E41 / best1**에서 끝났다. 기존 owner PID 52338/52337/41573은 모두 사라졌다. 5080의 6개 terminal manifest에 포함된 원본은 모두 SHA가 일치했다.

NVML 전체 compute-idle 상태는 확인할 수 없다. 대신 원본 회수 전에 승인된 GPU UUID, 해당 root argv를 가진 프로세스의 부재, fuser 장치 client에 이전 owner ID가 없는 점, supervisor exit0, 모든 terminal SHA를 다시 확인했다. `global_NVML_idle_verified=false`를 명시하며 GPU 전체 유휴로 위장하지 않았다. 다른 데스크톱 GPU client는 유지했다.

## 복구와 검증

새 collector는 기존 봉인 계약·운영 코드·원격 원본을 바꾸지 않는다. 5080/5090의 보존 archive와 manifest는 새 sibling 경로에 만들고, 로컬의 새 `titantpp_instacart_completion_recovery_20261010_v1` bundle로 회수했다. 각 archive, manifest, 614개 파일 및 selected/last binary를 stdlib SHA로 검증했다. Validation의 최초 strict finite 최소 전체 수량 RMSE epoch와 endpoint 일치도 확인했다. 추가 학습·재추론·Test 실행은 없다.

| 대상 | 상태 | 실제 회수 관측 KST | 완료 | 마지막 조건 저장/best |
|---|---|---|---:|---|
| 5080 Instacart | 완료 | 2026-10-10 11:09:08 | 6/6 | CNN+GRU54 seed62 E41/1 |
| 5090 Instacart | 완료 | 2026-10-10 11:09:11 | 3/3 | CNN+GRU54 seed52 E55/15 |

새 회수 관측시각은 과거 학습 관측시각을 대체하지 않는다. 학습 ETA는 종료로 해당하지 않는다. CPU binary 재추론 감사·Test·새 독립 평가는 미완료다.

3seed 평균(각 seed별 동일 선택 epoch): CNN+GRU54 전체 RMSE 5.87737, 큰 수량 RMSE 24.438, 전체 TimeNLL 2.8108. 폭12는 5.88006 / 24.6683 / 2.80721, 폭8은 5.88540 / 24.6355 / 2.80748이다. 평균 수량 오차 차이는 작고 CNN+GRU54 전체 TimeNLL은 두 MLP보다 소폭 높다. seed52의 전체 RMSE는 폭12가 더 낮으므로 모든 seed 일관 우월 주장은 하지 않는다. host의 계산시간을 합치지 않았다.

종료 시점 checkpoint receipt의 `exposure.json` hash는 6개 모두 terminal manifest와 다르다. 원인은 fit 종료 후 endpoint replay가 반영된 exposure를 다시 저장하는 기존 `titantpp_contribution_runtime.py`의 366행이다. history/epoch timing/last binary/train log는 checkpoint receipt와 terminal이 동일하다. 이 때문에 diagnostic의 `checkpoint_files_verified=false`가 terminal 원본 불일치를 의미하지 않는다. 최종 원본 판단은 terminal manifest를 기준으로 하며 이를 모두 검증했다.

## 변경 파일

- `paper/scripts/diagnose_titantpp_instacart_recovery.py`: GPU 오류에 종속되지 않는 지정 root 진단.
- `paper/scripts/recover_titantpp_instacart_completion.py`: 별도 원본 보존 및 Validation 집계.
- 본 문서와 진단 JSON, 새 recovery bundle의 archive/manifest/완료 receipt.

## 수용 조건

- [x] 현재 계약·source/input/operation·project control 봉인 SHA 확인.
- [x] nvidia-smi exit18 원문 재현 및 kernel/userspace 버전 불일치 확인.
- [x] 원본 9개와 selected/last binary 회수 및 SHA 검증.
- [x] 정확한 자기 소유 종료 확인과 supervisor exit0 확인.
- [x] Validation 9조건, 3seed 평균/표본SD 및 불리한 TimeNLL 보존.
- [x] 두 스크립트와 생성 원격 코드 Python 구문 검사.
- [x] 새 collector 실제 완료 경로 검증.
- [x] 기존 학습·driver·공용 Runtime·봉인 파일 변경 없음.
- [ ] NVIDIA 공용 드라이버/NVML 정상화: 이번 데이터 회수에 불필요하며 수정하지 않음.

## 다음 작업

Instacart는 재학습하지 않는다. 시간별 관측은 새 완료 receipt와 로컬 SHA를 사용한다. 후속 학습은 부모 세션의 A100 이력 보정 복구 계약을 따른다. 5080에서 새 GPU 학습을 시작하려면 별도 서버 운영 작업으로 드라이버/NVML 정합부터 해결해야 한다.

주요 증거: `INSTACART_DIAGNOSIS_20261010_110619KST.json` (raw SHA `aac9f862fb8598c6627cbe5b5c90cce53835a90406ac25c38a4c78210074ca24`), `INSTACART_GPU_DIAGNOSIS.json`, 새 recovery bundle `analysis/completion_receipt.json` 및 `analysis/VALIDATION_REPORT.md`.
