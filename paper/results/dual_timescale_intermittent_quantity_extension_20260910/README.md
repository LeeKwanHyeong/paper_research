# Intermittent 수량 중심 추가 평가

## 현재 상태

2026-09-10 **07:37:43 KST**에 5090에서 Intermittent seed42 학습을 시작했다. 초기 상태는 running, controller PID1884726, 학습 PID1884755, GPU 사용률84%, 사용 메모리 약2.9GB였다. 최신 진행 상태는 원격 `status.json`과 회수된 snapshot을 기준으로 확인한다.

**07:42:38 KST 확인:** epoch2 완료, 두 epoch 모두 train393,824개 및 finite 조건 정상. GPU85%, 메모리2,930MiB, controller 오류 로그 없음. 독립 초기 감사에서 epoch1 history가 기존 e1의 history SHA와 byte 단위로 동일함을 확인했다. 초기 검증은 통과했으며 성능 채택 여부는 아직 판단하지 않는다.

## 승인 범위와 원래 판정

사용자는 Taxi의 시간 guardrail 실패를 확인한 뒤, 변경하지 않은 후보의 Intermittent 평가를 별도로 승인했다. 원래 campaign의 `stopped_seed42_gate_failed`와 TitanTPP(B) 유지 기록은 변경하지 않는다. 이번 실행은 신규 후보의 세 번째 데이터셋에서 수량 성능을 확인하는 탐색적 추가 평가다.

- 소스: `paper_research`, `codex/hard-lmm-dual-timescale`, `ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c`.
- source/runtime/학습 코드는 변경하지 않았다. 별도 운영 wrapper가 기존 `job_command`, `verify_manifest`, `full_audit`만 재사용한다.
- Intermittent seed42 한 건, 최대300/minimum40/patience40.
- 기존 time+unweighted log1p-quantity loss, 최초 strict validation raw-RMSE 최솟값 selector를 유지한다.
- 원본 campaign과 같은 fresh seed42 초기화이며 e1 checkpoint를 이어 학습하지 않는다. e1의 CUDA·실행·복원 증적만 재사용한다.
- CUDA·비용·e1 반복 실행, 다른 데이터셋 재학습, 추가 seed, held-out, 모델 변경 또는 채택은 이번 실행에 포함되지 않는다.
- 시간 loss도 같은 정의로 보고하고, 원래 full gate와 수량 항목을 나눠 기록한다. 정상화된 Time NLL로 부르지 않는다.

## 실행 전 검증 완료

- 내부 SSH5090 =192.168.0.71, host=RTX5090-server, 계정과 소스/출력 경로 소유자=leekwanhyeong.
- GPU compute process 없음, 여유 메모리32,110MiB, 중복 Intermittent full fit 없음.
- Manifest에 등록된2,314개 source/input 파일과 계약·데이터·split·B reference SHA 일치.
- Runtime: Python3.12.13, PyTorch2.11.0+cu130, CUDA13.0.
- Intermittent e1: train393,824 / validation86,285, source와 대상 identity SHA 일치. 선택 checkpoint strict 복원 및 마지막 checkpoint AdamW46개 파라미터 복원 증적 재사용.
- 운영 wrapper syntax와 정상 scope/resume identity 통과, scope·resume identity 변조16개 거부 확인. `local_contract_check.json` 참조.
- 원격 `--plan-only`로 기존 source/proof SHA와 정확한 단일 학습 argv를 재확인했다. `remote_plan.json` 참조.
- 전송은 처음 자동 승인 검토에서 목적지 신뢰 증거 부족으로 거절됐다. SSH가 기존 사용자 내부 학습 서버이며 같은 계정 소유 경로임을 읽기 전용으로 증명한 뒤, 동일 scp 전송이 승인됐다. 약25KB의 운영 코드·계약·집계 비교표만 전송했고 raw 데이터·checkpoint·Secret은 전송하지 않았다.

## 경로와 복원

원격 source: `/home/leekwanhyeong/workspace/paper_research_dual_timescale_ca8823e/source`

원격 결과: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/dual_timescale_intermittent_quantity_extension_ca8823e_20260910`

학습 결과 하위 경로: `jobs/seed42_intermittent_frozen_5000/runs/titantpp_dual_timescale_memory/count_only_log_regression/seed_42`

tmux: `dual_timescale_inter_ca8823e`; 공유 GPU lock: `/tmp/paper_research_dual_timescale_gpu0.lock`.

운영 wrapper SHA: `86ad8d7f905eab67f4ed2dd6bae4c547a44488d006a62775ac52a8eb0fe7dc74`.

추가 실행 계약 SHA: `acbcf70ea17ab6449fdd8469c39a6ff7df8bc2fd4d0ffda075daf507f0813ed2`.

기술적 중단 시 실제 controller/child가 종료됐는지 확인하고 명시적 `--resume`으로만 재개한다. source·운영 코드·계약·argv·runtime·output identity가 같아야 하며 원본 trainer가 모델·optimizer·RNG를 복원한다. GPU lock은 학습 자식 프로세스에도 상속되므로 부모만 종료됐다고 중복 학습할 수 없다. 완료한 실행의 재개 요청은 audit SHA를 확인하고 no-op으로 종료한다.

## 시간 추정과 후속 처리

이전 e1의107.20초를 단순 환산하면100–150epoch에서 종료될 경우 시작 후 약3–4.5시간,300epoch를 모두 수행하면 약8.9시간이다. 이는 종료 epoch를 예측한 값이 아니며 실제 기록 속도에 따라 갱신한다.

기존 heartbeat `vnc-hard-lmm-5090-screening`을 **Intermittent 수량 평가 점검 / ACTIVE / 매1시간**으로 갱신했다. 정상 진행 중 의미 있는 변화가 없으면 알림을 반복하지 않는다. 완료·실패·조치 필요 때 알리고 이번 한 건이 terminal이면 PAUSED로 전환한다.

남은 순서:

1. **진행 중 / 5090:** 현재 한 건의 학습과 원본 `full_audit`를 완료한다.
2. **다음 작업 / 로컬:** 소형 summary/history/strata/audit/launch 증적을 회수하고, 기존 Taxi·Instacart 후보 및 B·RMTPP·THP를 재사용해 세 데이터셋 수량 비교를 완성한다.
3. **별도 검토:** Instacart 개선 진단과 신규 후보 변경은 이번 학습에서 자동 실행하지 않는다.

[실행 계약](execution_contract.json) · [원래 후보 평가](../final_backbone_and_baselines_20260909/final_campaign_report.md)
