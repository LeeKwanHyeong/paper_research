# Backbone 모니터링 인계와 최종 비교 준비

인계일: 2026-09-10 KST. 원래 작업: **Backbone 개선 방향 검토**
(`01a06e80-5507-7242-9b76-4c40e801fac7`).
대상 작업: `01a088c2-48c6-7950-be47-9f8ce7d3720e`.
로컬 저장소: `paper_research`, 현재 브랜치: `codex/hard-lmm-causal-qkv`.

## 현재 기준선 — 확인 완료

- 인계 대상은 heartbeat `vnc-hard-lmm-5090-screening`, 표시 이름
  **Intermittent 수량 평가 점검**이다. 주기는 매시간 한 번이며, 기존 자동화를
  현재 작업에 연결해 중복 모니터링을 방지한다. 이전 대상과 새 대상의 저장 확인은
  `automation_transfer.json`에 남긴다.
- 감시 대상은 사용자가 이미 승인한 5090 Intermittent seed42 추가 평가 **한 건**이다.
  신규 학습·추가 데이터셋·새 seed·모델 수정·최종 채택을 요청한 것이 아니다.
- B 기준선은 [검증된 seed42 raw-RMSE 비교](../../paper/results/final_backbone_and_baselines_20260909/seed42_validation_comparison.md)다.
  이전 v0.7 manuscript의 3-seed joint-objective selector 비교와 혼합하지 않는다.
- 후보는 dual-timescale `ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c`다.
  다른 key-value, bounded-QK, weighted retrieval, VNC 후보의 모니터를 재개하지 않는다.
- 원래 campaign의 Taxi 시간 guardrail 실패 및 B 유지 판정은 보존한다.
  Intermittent는 그 판정 이후 별도 승인된 quantity-focused extension이다.

## 마지막 저장 상태 — 실시간 상태와 구분

[monitor/latest.json](../../paper/results/dual_timescale_intermittent_quantity_extension_20260910/monitor/latest.json)의
마지막 관측 시각은 **09:42:28 KST**다. 72 epoch 완료, 최적 epoch37,
raw RMSE 1.5808443362, MAE 0.6676581532, legacy time loss −3.1682424993이다.
당시 controller·child·tmux·GPU 생존과 finite/학습 처리 건수 검증은 통과했고,
terminal full audit는 대기 중이었다. 이 값으로 현재 서버가 여전히 학습 중이라고
단정하지 않는다. 다음 scheduler 점검에서 원격 terminal 여부를 먼저 읽는다.

| validation 데이터셋 | B → 후보 raw RMSE | B → 후보 전체 MAE | 인계 시 증거 상태 |
| --- | --- | --- | --- |
| Taxi | 88.194997 → 78.255427 | 28.674020 → 25.048308 | 학습·실행 감사 완료, 시간 gate 실패 |
| Instacart | 5.872217 → 5.868059 | 3.993781 → 3.985497 | 학습·실행 감사 완료 |
| Intermittent | 1.499555 → 1.580844 | 0.604340 → 0.667658 | 09:42 저장된 중간 최적값, 최종 감사 대기 |

## 5090 기존 실행 확인 — 외부 작업 대기

- SSH alias: `5090` (사용자가 승인한 내부 서버).
- 소스: `/home/leekwanhyeong/workspace/paper_research_dual_timescale_ca8823e/source`.
- 결과: `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/dual_timescale_intermittent_quantity_extension_ca8823e_20260910`.
- job: `jobs/seed42_intermittent_frozen_5000`.
- run: 위 job의 `runs/titantpp_dual_timescale_memory/count_only_log_regression/seed_42`.
- tmux: `dual_timescale_inter_ca8823e`, 실행 파일
  `/home/leekwanhyeong/.conda/envs/ts_forecaster_ops/bin/tmux`.
- Python: `/opt/miniconda3/envs/ai_env/bin/python`.
- 상세 SHA·단일 실행·resume 조건은 기존
  [execution_contract.json](../../paper/results/dual_timescale_intermittent_quantity_extension_20260910/execution_contract.json)과
  보존한 scheduler prompt를 따른다. 현재 로컬 브랜치를 원격 frozen source로 대체하지 않는다.
- 매 실행에 한 번만 상태와 최근 history를 확인한다. 상태가 같거나 조치가 불필요한
  정상 진행이면 알리지 않는다. 완료·오류·사용자 조치 필요 시에만 알린다.
- history는 최상위 객체 안의 `history` 목록이다. 과거 모니터의 파싱 오류를
  학습 실패로 오인하지 않는다. SSH 연결 실패도 학습 실패와 구분한다.
- 기존 wrapper가 학습 후 `full_audit`를 실행한다. 학습 PID 종료만으로 전체 완료를
  판정하지 않는다. monitor는 checkpoint를 역직렬화하지 않는다.
- 기술적 중단의 복원은 기존 승인 조건을 모두 만족할 때만 동일 wrapper의
  `--resume`으로 수행할 수 있다. 성능 실패에는 resume하지 않는다.

## 최종 checkpoint와 증적 확인 — 다음 작업

- terminal이면 status·summary·history·strata·audit·launch·운영 계약의 소형 파일을
  `paper/results/dual_timescale_intermittent_quantity_extension_20260910/remote`로
  `--delete` 없이 회수한다. 대형 데이터나 checkpoint는 복사하지 않는다.
- source·runner·계약·runtime·데이터 SHA, validation target identity와 quantity SHA,
  train393,824 / validation86,285건, finite, earliest strict raw-RMSE selection,
  max300/min40/patience40 및 wrapper의 저장·복원 감사 결과를 확인한다.
- 모든 지표는 **학습 종료 후 확정된 raw-RMSE 최적 checkpoint 하나**에서 가져온다.
  last-epoch checkpoint나 지표별 최적 epoch를 섞지 않는다.
- summary 또는 감사가 없으면 누락 항목으로 기록한다. history에 validation 처리
  건수가 없다는 사실을 알고 있으므로 해당 건수는 terminal population 감사로 확인한다.
- 실행 감사 통과와 수량/시간 성능 gate 통과를 별도 항목으로 기록한다.

## 세 데이터셋 수량 비교 통합 — 감사 후 직렬 진행

- B·RMTPP·THP는 기존 [비교 JSON](../../paper/results/final_backbone_and_baselines_20260909/seed42_validation_comparison.json)을 재사용한다.
  Taxi·Instacart 후보는 기존 [최종 campaign 판정](../../paper/results/final_backbone_and_baselines_20260909/final_campaign_decision.json)의
  완료 결과와 원본 audit를 재사용한다. 원래 판정 파일을 덮어쓰지 않는다.
- 결과를 `paper/results/dual_timescale_intermittent_quantity_extension_20260910/integration/`에
  별도로 작성한다. 모델·dataset·seed·selected epoch·종료 epoch·RMSE·전체 MAE·
  Body MAE·>p99 MAE·legacy time loss·validation identity·출처를 포함한다.
- Body는 해당 데이터셋의 **train p95 이하**, tail은 **train p99 초과**라는 기존
  정의를 확인한다. 모델 간 경계·포함 조건·표본 membership가 같은지 확인한다.
- B 대비 개선율과 RMTPP·THP 대비 차이를 함께 표시한다. 잠정 Intermittent 행은
  감사된 최종값이 확보된 뒤에만 교체한다.
- Taxi의 기존 시간 loss 10.8258699686과 B+0.01 기준 미달을 유지하고,
  legacy clamped 점수를 정상화된 Time NLL로 표현하지 않는다.
- 완료 조건은 세 데이터셋의 동일 기준 표와 증적 출처가 갖춰지는 것이다.
  단일 seed validation 자료만으로 공통 우월성·통계적 유의성·최종 채택을 선언하지 않는다.

## 수정 여부 판단 준비 — 통합 후 검토

- [Instacart 진단](../../paper/results/instacart_dual_timescale_quantity_diagnostic_20260910/README.md)은
  원래 CUDA 집계값과 로컬 CPU 재현값의 작은 수치 차이 및 호환성 정책을 구분해 사용한다.
  주 비교표에는 원래 확정 지표를 사용하고 진단 재현값으로 몰래 대체하지 않는다.
- RMTPP 대비 수량8–35·이력8–15의 과소예측 차이와, B 대비 이력2–3의 약점을
  서로 다른 비교로 해석한다. 이 현상이 Intermittent에도 존재한다고 가정하지 않는다.
- 먼저 Intermittent의 전체·Body·tail 지표와 시간 점수를 확인하고, 필요할 때만
  후속 동일 표본 진단의 목적·검증 방법·완료 조건을 제안한다.
- 새 진단 실행, 모델·loss 변경, 재학습·seed 확대·held-out 평가는 자동 시작하지 않는다.
  해당 작업은 사용자의 후속 지시 범위에 따라 진행한다.
- Scheduler는 terminal 결과와 남은 작업을 이 작업에 기록한 뒤 PAUSED로 전환한다.
  원격 학습은 terminal이지만 소형 증적 회수·감사가 막힌 경우에는 완료라고 기록하지
  않고 필요한 조치를 구분해 알린다.

## 준비 작업의 범위

이번 인계에서는 scheduler 대상 연결과 로컬 인계 기록만 변경한다. 학습 코드,
DB, 원격 실행, frozen 계약, baseline·후보 결과는 수정하지 않는다. 커밋·Push·MR은
수행하지 않는다. 공통 checkpoint·모집단·selector에 의존하므로 최종 감사와 비교는
현재 작업에서 직렬로 진행한다.
