# Causal QKV Hard-LMM — Taxi·Intermittent 병렬 후속 평가

**진행 중:** 2026-09-08 08:26 KST 확인 기준, 두 서버 모두 seed42 학습 중이다.
이 디렉터리는 기존 Instacart 기준 미달 이후 사용자가 승인한 탐색적 후속
평가의 새 증적이다. 기존 screening 결과를 덮어쓰지 않는다.

## 현재 기준선과 동결 계약 — 완료

- 저장소/브랜치: `paper_research / codex/hard-lmm-causal-qkv`.
- 실행 소스 커밋: `84668e207d5121f211a6a93af12ca2f96068a25e`.
- 원래 모델 구현: `a7795355920d4bdc5fb63dc40deccbfcbd4daec0`.
  기존 Python 소스 405개를 byte/hash 동일하게 유지했다.
- 후보: `titantpp_hard_memory_causal_qkv`. 첫 encoder block의 event Q/K/V에
  kernel3 인과적 residual을 추가한 Backbone 후보이며, h64에서 추가 파라미터는 576개다.
- 기존 loss·head·raw-RMSE selector를 유지한다. seed42, 최대300 epoch,
  최소40 epoch, patience40이며, validation만 평가한다.
- 계약: `paper/contracts/hard_lmm_causal_qkv_followup_v1.json` 및 `.md`.
- 기존 Instacart 판정 `not_adopted_seed42_raw_rmse_gate_failed`와 원본
  `hard_lmm_causal_qkv_screening_20260907` 결과는 변경하지 않았다.

## 구현·로컬·CUDA 검증 — 완료

- 로컬 기존 관련 검사 71개 통과/CUDA 전용 1개 skip, 새 실행기 검사 11개 통과.
- 두 서버 각각 45개 검사 통과, skip 0개. 각 묶음에 실제 CUDA 초기 동일성,
  새 kernel gradient, 학습에 따른 상태·예측 변화 검사 3개가 포함된다.
- 두 서버의 L256 합성 학습에서 finite 모델·optimizer, 학습 경로와 상태 복원을 확인했다.
- 기존 Taxi·Intermittent full-data e1 통과 증적을 재사용했다.
- archive SHA-256:
  `d6e2b6af891f04104ab161d35d8eca0902ec5c5a7b454031b33e3dce7cf66ebc`.
  양쪽 전송·추출 후 archive와 source 405개, 데이터·split을 검증했다.
- `hosts/<host>/`에 보관한 JUnit·합성 학습·source receipt의 hash와 결과를
  로컬에서 재검증했다. `launch_receipt_verification.json` 참조.

## 실제 학습 — 진행 중

| 서버 | 데이터셋 | 본 학습 시작 KST | 08:26까지 완료 epoch | 현재 best epoch | 마지막 epoch train target |
| --- | --- | --- | ---: | ---: | ---: |
| RTX 5080 | Taxi | 08:22:41 | 17 | 11 | 38,393 |
| RTX 5090 | Intermittent | 08:22:52 | 3 | 2 | 393,824 |

두 tmux와 CUDA 학습 프로세스가 살아 있고 실행 오류가 없다.
이 표는 초기 상태 기록이며, 성능이나 최종 checkpoint의 판정이 아니다.
현재 상태와 확인 시각은 `monitor_latest.json`에 기록한다.

소스 경로는 서버별
`/home/leekwanhyeong/workspace/paper_research_causal_qkv_followup_84668e207d51_<host>`,
결과 경로는
`/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/hard_lmm_causal_qkv_followup_84668e207d51_<host>_20260908`이다.
tmux는 `causal_qkv_followup_84668e2_<host>`이며, 시작 명령어와 정확한 경로는
`launch_5080.json`, `launch_5090.json`에 보관했다.

## 남은 작업 순서

**1. 진행 중 / 5080·5090 — 각 데이터셋 학습 완료**
- 같은 구조·loss·selector와 기존 조기 종료 규칙으로 각 실험을 완료한다.
- 한 데이터셋의 성능 미달은 다른 서버의 승인된 평가를 중단시키지 않는다.
  실행·감사 오류는 해당 호스트에서 중단하고 자동 재시도하지 않는다.

**2. 다음 작업 / 로컬 — 최종 결과 감사와 탐색적 판정**
- 선택 checkpoint, 전체 처리 수, validation population/수량 구간,
  모델·optimizer 복원과 finite 상태를 감사하고 B 대비 다섯 지표를 비교한다.
- 기존 엄격한 데이터셋별 gate를 그대로 보고한다. 두 결과가 모두 감사되면
  고정한 후속 검토 기준도 계산한다. 세 데이터셋의 raw RMSE/전체 MAE 악화 각각
  1% 이하, body/>p99 MAE 각각2% 이하, legacy time loss 증가0.01 이하이며,
  Taxi·Intermittent 중 하나 이상의 raw RMSE 개선이1% 이상이어야 검토 자격을 부여한다.
- 이 기준은 추가 seed의 자원 사용을 검토하기 위한 것으로, 기존 Instacart 실패를
  통과로 바꾸거나 최종 채택·통계적 비열등성을 의미하지 않는다.

**3. 진행 중 / 이 작업 — 시간별 확인**
- Heartbeat `causal-qkv-taxi-intermittent`를 1시간 간격으로 설정했다.
- 완료·오류·필요 조치가 있을 때 알리고, 두 실험 종료 후 중지한다.
- 추가 seed·held-out 평가는 이번 실행 범위에 포함하지 않는다.
