# Causal QKV Hard-LMM — Taxi·Intermittent 병렬 후속 평가

**최종 완료:** Taxi는 2026-09-08 08:52:56 KST(epoch153, 선택113),
Intermittent는 09:16:26 KST(epoch47, 선택7)에 정상 조기 종료했다.
로컬·원격 감사는 통과했으나, 추가 seed 검토 기준은 미달했다. Taxi의 수량
개선과 함께 두 데이터셋의 시간 loss 악화, Intermittent의 RMSE 악화를 확인했다.
상세 결과는 [final_result.md](final_result.md)와 `final_decision.json`에 보관했다.
기존 Instacart 기준 미달 결과는 보존하며, 시간별 확인은 PAUSED로 변경했다.
아래의 08:26 표는 학습 시작 당시의 역사적 기록이다.

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

## 실제 학습 — 초기 확인 기록

| 서버 | 데이터셋 | 본 학습 시작 KST | 08:26까지 완료 epoch | 현재 best epoch | 마지막 epoch train target |
| --- | --- | --- | ---: | ---: | ---: |
| RTX 5080 | Taxi | 08:22:41 | 17 | 11 | 38,393 |
| RTX 5090 | Intermittent | 08:22:52 | 3 | 2 | 393,824 |

위 확인 시각에는 두 tmux와 CUDA 학습 프로세스가 살아 있고 실행 오류가 없었다.
이 표는 초기 상태 기록이며, 성능이나 최종 checkpoint의 판정이 아니다.
현재 상태와 확인 시각은 `monitor_latest.json`에 기록한다.

소스 경로는 서버별
`/home/leekwanhyeong/workspace/paper_research_causal_qkv_followup_84668e207d51_<host>`,
결과 경로는
`/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/hard_lmm_causal_qkv_followup_84668e207d51_<host>_20260908`이다.
tmux는 `causal_qkv_followup_84668e2_<host>`이며, 시작 명령어와 정확한 경로는
`launch_5080.json`, `launch_5090.json`에 보관했다.

## 남은 작업 순서

**1. 완료 / 5080·5090 — 두 데이터셋 학습과 감사**
- 같은 구조·loss·selector와 기존 조기 종료 규칙으로 두 실험을 완료했다.
- checkpoint·optimizer 복원, 전체 처리 건수·지표와 소스 무결성 감사를 통과했다.

**2. 완료 / 로컬 — 최종 판정과 증적 보관**
- 추가 seed 검토 기준에 미달하여 이 후보의 탐색을 종료했다.
- 기존 Instacart 실패를 보존하고, 새 결과와 독립 검산·복원 감사 증적을 기록했다.

**3. 다음 작업 / 로컬 — 코드·증적 통합 범위 검토**
- 결과는 `paper_research/codex/hard-lmm-causal-qkv`에 보존한다.
  `master` 병합과 기본 모델 전환은 수행하지 않았다.
- 추가 seed·held-out 평가를 실행하지 않았고, heartbeat는 종료 후 PAUSED로 변경했다.
