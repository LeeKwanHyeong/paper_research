# Taxi6 관측·회수·Validation 비교

사용자는 2026-10-15 Taxi 두 후보 ×3seed, 5080 최대3 병렬 학습과 종료 후 기존 CNN과 Validation RMSE·MAE·seed별 차이 비교를 승인했다. 이력 적응3개와 용량 대조군3개, 총6개만 포함한다. 학습과 tmux는 이미 시작했으므로 재설치·재시작하지 않는다.

프로젝트 TEST_SESSION_PROTOCOL.md 0·1·9·12와 이 문서를 읽는다. Main manuscript, Test, held-out 및 혼합 결과를 열지 않는다. 학습 source/계약/선택/기한과 기존 결과·실패 기록을 변경하지 않는다. 추가 학습·retry/resume·kill·Runtime/driver/auth변경0. 삭제된 A100은 접속하지 않는다.

프로젝트 root에서 회차당 한 번:
`PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.advance_titantpp_history_taxi6 --integrity-sha c34a26cce9b8656917e0c1dc10d67b53104b0fbd23ae02f96caed2d4b444661a`

이 helper는 전체 로컬 source·input·운영·baseline SHA를 검증한 다음, 5080 고정 root `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_cnn_history_taxi6_20261015_v1_5080`를 한 번 읽는다. 실제 process/GPU·lease가 확인된 상태만 진행으로 센다. 별도 SSH/반복polling은 하지 않는다. 실패 또는 ACTION_REQUIRED가 있으면 자동 재시도하지 않는다. 전체6 scientific worker exit0·모든 terminalbinarySHA·dispatcher 및 pipeline의 직접 관측 exit0·train_complete·소유PID/GPU부재를 확인한 뒤 같은 연결에서 원본을 회수한다. supervisor wrapper 자신의 OS exit는 직접 관측하지 않는다. 원 launcher의 pipeline_exit.json 마지막 literal backslash-n은 원문/SHA를 보존하고 알려진 suffix 하나만 엄격하게 해석하며 파일을 수정하지 않는다.

원본 archive/manifest는 새 retrieval 경로에 exclusive 보존한다. 기존 CNN3 기준선을 재사용해 여섯 Validation 지표의3seed 평균·표본SD(ddof1), seed별 Adaptive−CNN/Capacity−CNN/Adaptive−Capacity 차이를 기록한다. RMSE·MAE를 중심으로 보고하며 악화도 포함한다. 선택은 fullValidation rawRMSE 최초 strict finite 최소epoch다. Test 개선이나 원인 확정으로 확대하지 않는다.

평소 진행이 건강하고 사용자 조치가 없으면 조용히 유지한다. 한 후보의3seed 완료·다음 후보 시작, 전체완료, 실패, 사용자 조치 필요 시만 알린다. 종료 전 ETA가 필요하면 해당 fit의 최근최대10 finite양수 완료epoch 중앙값만 사용하며 best미갱신 조건과300epoch 시나리오를 구분한다. 전체큐 ETA는 미확정이다.

완료는 execution_v1/retrieval/final_receipt.json status=complete,new_fit_count=6,baseline_reference_count=3,seed_count=3,held_out_test_evaluated=false와 모든 raw SHA 및 THREE_SEED_VALIDATION.json/.md를 재검증한 때다. 최종3seed 결과 보고 후 이 Taxi6 자동화 하나만 삭제한다. 부분 완료에서는 유지한다.
