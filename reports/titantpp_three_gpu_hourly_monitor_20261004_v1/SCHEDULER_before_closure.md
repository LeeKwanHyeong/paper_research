# TitanTPP 학습 종료·RTX 원본 회수 시간별 관측

**9조건 학습·Validation·서버 종료 — 완료**

- [종료 관측 보고](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_training_terminal_20261005_v1/README.md)와 [summary](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_cnn_gru_training_terminal_20261005_v1/summary.json)를 현재 기준선으로 사용한다.
- 5080·5090의 원래 실제 terminal 관측시각을 유지하며 캐시로 반복SSH를 끝낸다.

**RTX checkpoint 원본 회수 확인 — 다음 작업**

- 로컬 회수와SHA검증이 미완료다. heartbeat는 추가 원격 회수나 평가를 실행하지 않고 보존 증거와 새로 생긴 로컬 회수 receipt만 확인한다.
- 회수·비용·종료 모든gate가 충족돼야 이 통합 모니터만삭제한다. CPU 재추론감사는 별도미완료다.

paper_research의 완료된 CNN·GRU seed42 실험과 RTX 원본 회수 상태를 매시간 읽기 전용으로 확인한다. 사용자가 시간별 GPU 상태 보고를 명시적으로 요청했으므로 매회 짧은 한국어 표로 알린다. 학습은 9/9조건 모두 완료됐으며 새 학습이나 평가를 시작하지 않는다.

현재 기준선
프로젝트는 /Users/igwanhyeong/PycharmProjects/paper_research, 이관 bundle은 search_artifacts/titantpp_cnn_gru_resume_dual_20261005_v1이다. TEST_SESSION_PROTOCOL.md 0·1·9·12절, reports/titantpp_three_gpu_hourly_monitor_20261004_v1/SCHEDULER.md, reports/titantpp_cnn_gru_training_terminal_20261005_v1/README.md와 summary.json, bundle의 README/current.json·current.host_contracts가 지정한 계약·approval/start/training permit·qualification receipt·recovery_inventory.json·hourly_monitor/latest를 먼저 읽는다.
5080 운영 canonical SHA는 6581c156867c7ad1b317d1af6fb3dfffc29bcf55def953f6264ff6b2de4f6a02, 5090 전용환경 운영 SHA는 6b85d3ea3b5b096653e2d5ba2ffbf35ceeefd5c58f3fe1699c0be29ae9eaf5fb이다. 부모 46a0eb48d5207df800553f5d3508016ce9063d1dec9be2c5e84bfa87a1b1fdae, 동결117소스 40184c7ff418f35b191ebfa309e462b9e88889443f36e4722cc5d06988d182cd, worker4 7d961867f612f86e1b78346630764d7624472532f596aba133a3bf3d24da2f26를 보존한다. current.json의 original_complete=7/remaining=2와 recovery_inventory의 remaining_jobs는 이관 당시 원본 분류이며 현재 진행 수가 아니다. 현재 완료 수는 original7+resume2의 고유 canonical ID로 집계한다. current/root/GPU/계약 또는 보존 terminal 증거가 달라지면 이전 대상으로 접속하거나 재시작하지 않고 변경 이유를 확인한다.

검증된 학습 종료
A100 원래7조건은 scientific_success·소형 원본 SHA·selected/last full Validation이 검증됐다. Taxi CNN+GRU54도166/best126으로 완료돼 재학습하지 않는다. 소유 Pod okbl76hxfr7a9d는 원본428개와 archive SHA 회수 후 정지·삭제·provider 목록부재가 확인됐다. A100 SSH/provider/manager를 다시 관측하지 않는다. 기존 recovery/ssh_info_attempt2/retrieval_receipt.json, recovery/cleanup_receipt.json, recovery/cost_estimate.json을 재사용한다. archive SHA는 21eb9b095b9ff62784235cbd7e5ae7a78423bf545f69c88a2fad3d6bfaa73f92, 보수 비용 약$25.82는 청구서나 계정잔액 차이가 아니다.
5080 Intermittent GRU54 seed42는 저장50/best10, 실제 완료2026-10-05 17:51:39 KST다. 실제 마지막 관측은18:41:17 KST, snapshot SHA d6f8f0bdad5093a1d5539b1dc4142d1268a758a3408d7a9b74a991354b6b0bf9다.
5090 Intermittent CNN+GRU54 seed42는 저장52/best12, 실제 완료2026-10-05 19:33:05 KST다. 실제 마지막 관측은19:36:28 KST, snapshot SHA 0ee8d0363e38685091cd973a583b99ee21b624335c2c3348f99855d48906cd8b다.
두 조건 모두 terminal scientific_success·소형 원본 SHA·selected/last full Validation·실제 소유 supervisor/worker/GPU PID 부재와 서버 terminal receipt가 검증됐다. 각 서버 학습 worker 수는0이다. 완료9, 실패/진행/대기/미확정/미시작 모두0이며 학습 ETA는 해당 없음이다. 종료 뒤 exposure가 최종 Validation 검증1회 추가로 중간 checkpoint receipt와 달라도 최종 terminal manifest의 SHA와 대조해 판정한다.

회차당 관측
프로젝트 root에서 아래 명령을 독립적으로 각각 정확히 한 번 호출한다.
/usr/local/bin/python3 paper/scripts/observe_titantpp_cnn_gru_resume_hourly.py --host 5080
/usr/local/bin/python3 paper/scripts/observe_titantpp_cnn_gru_resume_hourly.py --host 5090
두 서버 모두 terminal cache를 재사용하므로 SSH0회가 정상이다. 원래 실제 관측시각과 snapshot SHA를 유지하며 회차 실행시각을 새 관측시각으로 표시하지 않는다. terminal 캐시를 지우거나 반복 SSH, 다른 observer, polling, qualification, replay, 학습, resume/retry, manager, 임대, 프로세스 종료, claim 삭제를 추가하지 않는다. source/Runtime/data/loss/선택/예산/인증정보도 변경하지 않는다. 예상치 못한 캐시·계약 불일치는 미확정 및 필요한 사용자 조치를 보고하고 원격 작업을 시작하지 않는다.

결과와 후속 상태
Validation은 최초 최소 raw 수량 RMSE의 같은 선택 epoch에서 RMSE·MAE·Time NLL과 큰 수량 지표를 보고한다. A100 원본 history 값과 RTX 최종 Validation replay 값은 별도 열로 보존하며 checkpoint를 다시 선택하지 않는다. Train 큰 수량 경계는 Intermittent187, Taxi3449, RAF200 초과다. 단일seed·이종GPU 탐색이며 3seed 결과나 논문 효율 결론으로 쓰지 않는다. Test·예측·Validation/Test 혼합 파일을 읽거나 CNN/GRU Test 평가를 실행하지 않는다. Instacart는 후속이다.
기존 폭8/12의18조건과 Intermittent6 Validation gate·Test는 완료된 기준선이다. reports/titantpp_three_dataset_final_comparison_20261005_v1와 보존 receipt의 완료 메타데이터만 재사용하고 이전 capacity 관측기·finalize Test·새 평가를 실행하지 않는다.

RTX 원본 회수와 모니터 종료
RTX에서 새로 저장된 best/last checkpoint binary 원본의 로컬 회수 및 manifest SHA 검증은 학습 완료와 별개이며 아직 확인되지 않았다. 이번 모니터는 기존 로컬 회수 receipt와 원본 파일이 새로 생겼는지만 확인한다. 원격 manifest에 binary SHA가 기록됐다는 사실만으로 로컬 회수 완료라고 하지 않는다. 회수 파일이 있으면 해당 계약·canonical ID·최종 manifest와 로컬 파일 SHA를 대조한다. heartbeat에서 추가 SSH·원격 회수·재추론 감사·자동 retry를 시작하지 않는다.
새 RTX 원본 회수·두 서버 terminal·기존 A100 원본/비용/삭제목록부재·기존 Intermittent Test 완료가 모두 확인되면 최종 증거 보고 후 이 자동화 titantpp-cnn-gru-a100-seed42만 삭제한다. 원본 회수가 미확정이면 유지하고 회수 미완료를 명시한다. CPU binary 재추론 감사는 별도 미완료 항목이며 전체 연구 감사 완료라고 표현하지 않는다. 다른 자동화·PAUSED monitor·Pod·서버 작업·채팅 archive 상태를 변경하지 않는다.
