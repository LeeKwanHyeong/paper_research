# 인계 전 모니터링 지침

기존 자동화에서 인계한 운영 지침의 로컬 사본이다. 서버 접속 정보와 원격 경로는 자동화 프롬프트에 복사하지 않고 이 로컬 파일과 execution_contract.json에서 확인한다. 현재 작업 범위와 README.md의 인계 보완사항을 함께 적용한다.

사용자가 2026-09-10에 별도로 승인한 Intermittent 수량 중심 추가 평가 한 건만 한 시간마다 확인한다. 프로젝트는 /Users/igwanhyeong/PycharmProjects/paper_research이다. 과거 Taxi 시간 guardrail 실패와 final_backbone_and_baselines_20260909의 원래 종료 기록은 보존한다. 기존 5080 기준선 네 개는 모두 완료됐으므로 재개하지 않는다.

5090은 사용자의 내부 SSH alias이며 192.168.0.71, RTX5090-server, leekwanhyeong 계정이다. 학습 소스는 /home/leekwanhyeong/workspace/paper_research_dual_timescale_ca8823e/source, revision ca8823e688a510e64e7fb81c9dc9d158d4bdfc7c이다. source 상위 manifest.json의 SHA256은 a2807787fed399a8da14d9033677133964f795959280c1de3cb0a6f6d421ebd6이고 기존 계약 SHA256은 4fdba2b5d2fc458eae923b7628b9e3a0dfcd72f0de6e79bf5cafb86a0b07cc8e이다. Runtime Python은 /opt/miniconda3/envs/ai_env/bin/python, PyTorch2.11.0+cu130, CUDA13.0이다.

별도 결과 경로는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/dual_timescale_intermittent_quantity_extension_ca8823e_20260910이다. 이 경로의 run_extension.py와 execution_contract.json이 한 건만 실행하며 tmux 세션은 dual_timescale_inter_ca8823e이다. tmux 실행 파일은 /home/leekwanhyeong/.conda/envs/ts_forecaster_ops/bin/tmux이다. 운영 runner SHA256은 86ad8d7f905eab67f4ed2dd6bae4c547a44488d006a62775ac52a8eb0fe7dc74, 실행 계약 SHA256은 acbcf70ea17ab6449fdd8469c39a6ff7df8bc2fd4d0ffda075daf507f0813ed2이다. 학습 job은 jobs/seed42_intermittent_frozen_5000, run은 그 아래 runs/titantpp_dual_timescale_memory/count_only_log_regression/seed_42이다.

seed42, 최대300/minimum40/patience40, 기존 unweighted log1p quantity MSE와 legacy time loss, 가장 이른 strict validation raw-RMSE checkpoint 선택을 유지한다. 기존 CUDA·비용·세 데이터셋 e1 증적을 재사용하며 추가 e1이나 학습 후보를 만들지 않는다. 이 작업은 원래 Taxi 실패 후 사용자 승인으로 추가된 quantity-focused validation 평가다. 시간 loss를 계속 기록하고 원래 full gate 판정도 감사에 남기되, 이번 한 건 완료 뒤 추가 학습을 자동 실행하지 않는다. 추가 seed, held-out test, Instacart 재학습·모델 개선·loss/selector 변경·master/develop 병합·push는 승인 범위가 아니다.

매 점검에서 status.json, controller/학습 PID, tmux, GPU, history.json의 최근 epoch·best epoch·finite·train393824건과 validation86285건을 한 번 확인한다. 서버 wrapper가 학습 후 immutable source의 full_audit를 실행해 선택/마지막 checkpoint와 optimizer 복원, population SHA, finite와 earliest checkpoint/patience를 확인하고 audit.json/status.json을 작성한다. 모니터는 JSON/CSV/파일 byte hash만 사용하고 checkpoint를 새로 역직렬화하지 않는다. 실제 관측 속도로 조건부 ETA를 갱신한다. 정상 진행 중 조치가 필요 없거나 상태가 의미 있게 바뀌지 않았으면 조용히 종료하고, 완료·오류·사용자 조치 필요가 있을 때 알린다.

SSH 실패만으로 학습 실패로 판정하거나 중복 실행하지 않는다. 소스/runtime/데이터/계약을 변경하거나 다른 GPU 프로세스를 종료하지 않는다. 기술적 중단이면 실제 controller와 child가 모두 없고 동일 source·runner·계약·runtime·output identity가 유지됨을 확인한 경우에만 wrapper의 --resume으로 같은 작업을 복원할 수 있다. 동일 GPU lock은 /tmp/paper_research_dual_timescale_gpu0.lock이며 live child가 상속한다. 성능 결과를 보고 checkpoint 규칙을 변경하지 않는다.

완료 또는 실패 후 소형 status/summary/history/strata/audit/launch/운영 계약 증적을 --delete 없이 로컬 /Users/igwanhyeong/PycharmProjects/paper_research/paper/results/dual_timescale_intermittent_quantity_extension_20260910/remote 로 회수하고 checkpoint나 대형 데이터는 복사하지 않는다. 완료 시 기존 final_backbone_and_baselines_20260909의 B·RMTPP·THP와 현재 후보의 Taxi/Instacart 결과를 재사용해 같은 신규 후보의 세 데이터셋 수량 비교를 작성한다. 실행 감사 통과, 원래 시간 포함 gate, 수량 지표 성과를 구분하고 단일-seed validation 범위 및 Taxi 시간 악화를 명시한다. 최종 채택을 선언하지 않는다. 이번 한 건이 terminal 상태가 되면 결과와 남은 작업을 기록하고 이 heartbeat를 PAUSED로 변경한다.

