# 5090 MAC9 정확도 직접 대조

**현재 작업의 범위와 기준선을 보존한다 — 준비 완료**
- 사용자 2026-10-20 승인: “그러면 동일 조건의 MAC 정확도 실험이 필요 이 내용에 대해서 5090에서 진행하자”. 새 MAC9 학습·Validation9·고정 Test9·원본 회수·Shared15 저장 결과와의 비교를 진행한다.
- 프로젝트 TEST_SESSION_PROTOCOL.md0·1·9·12, SCIENCE_SPEC.json, FROZEN_CONDITIONS.json, EXPERIMENT_PLAN.md와 실제 receipt를 먼저 읽는다. 기존 MAC legacy 결과·실패·checkpoint와 Shared15/CNN/MLP16/외부6종의 결과를 보존한다. 5080 추가2seed 작업은 별개이며 변경하지 않는다.
- 정확한 모델은 titantpp_titans_mac, RAF/Taxi/Intermittent × seed42/52/62=9개뿐이다. source983과 MAC helper2는 동일 바이트로 유지하고 새 운영4파일만 추가한다. 이전 준비된 Taxi/Intermittent/Instacart MAC9 계약을 실행하지 않는다.

**공통 조건의 학습을 한 번 시작한다 — 다음 작업**
- 새 root는 /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titans_mac_accuracy9_20261020_v1_5090 이다. 기존 Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads/interop1/GPU-c9adc246-ef66-7906-bc07-6e152fa9952f를 유지한다. 공용 Runtime/driver/auth를 바꾸지 않는다.
- 기존 Shared15와 원본 입력·log1p 시간/수량·공통 수량/시간 head·logMSE+시간NLL가중치1·AdamW.001/wd.01/외부clip1·batch128·max300/min40/patience40을 맞춘다. MAC 내부 associative clip1과 오늘5090 비용 측정과 같은 의미 보존 실행 최적화를 유지한다.
- 새 로컬 코드·계약 검증/독립 검토/커밋/봉인 후 실제 반환 integritySHA만 사용한다. 새 경로·유휴·GPU/Runtime·source/inputSHA 검증과 최대 길이256/batch128의 MAC native 합성·실제3개 동시 admission을 통과해야 학습한다. 세 데이터 공통 head 초기값은 각 seed와 대조한다. 데이터별 nativeMAC 실행 범위는 오늘 완료한 비용 측정 증적도 함께 보존하며 새3동시 admission과 구분한다.
- RAF42/52/62→Taxi42/52/62→Intermittent42/52/62의 고정 순서, 최대3병렬이다. 실제 source·승인·계약과 exclusive attempt가 결속된 경우에만 처음 시작한다. 이미 시작한 root/claim/failure를 삭제·재시작하거나 자동 retry/resume하지 않는다.
- 조건36h·평가2h 및 상속 절대 host기한1792888885.815062를 연장하지 않는다. 최대300epoch를 남은 기한 내 보장하지 않는다.

**학습 상태와 종료를 구분해 관측한다 — 실행 중**
- 먼저 로컬 QUEUE_FAILURE/QUEUE_COMPLETE, retrieval/final_receipt, FINISHER_STARTED와 finish.stdout/stderr를 확인한다. 이미 있는 유한 finisher를 중복 시작하지 않는다. 무출력 대기는 실패가 아니다.
- 상태가 필요하면 봉인된 manage_titans_mac_accuracy9 --mode status --integrity-sha 실제SHA를 회차당 한 번만 실행한다. 원격 status 원문·transport exit/stdout/stderr/SHA와 actual PID/argv/start_ticks/lease/GPU를 연결한다. 조회 실패를 학습 실패로 바꾸지 않는다.
- 현재 관측 시각을 보존하고 과거 종료 cache를 현재 전역 GPU 유휴로 사용하지 않는다. ETA는 해당 fit 최근 최대10개 finite 양수 완료 epoch 시간 중앙값만 쓰고 best 미갱신 조건과300epoch 시나리오를 구분한다. 다른 모델·fit 속도를 빌리지 않는다.
- 실패/timeout/미완료 회수는 원문과 소유 child의 실제 exit를 보존하고 자동 반복하지 않는다. 소유가 확인된 새 child의 고정 timeout 정리만 허용한다. wrapper 자신의 OS exit를 부모가 직접 본 child exit와 구분한다.

**선택 checkpoint를 재현하고 고정 Test를 평가한다 — 학습 완료 후**
- 전체 fullValidation raw 수량 RMSE의 최초 strict finite 최소 epoch를 고정한다. 모든9 학습 종료 뒤 nativeValidation9에서 동일 target/truth/history와 전체/큰수량RMSE·MAE·TimeNLL 여섯지표를 검증한다. 전9 Validation receipt와 actualexit0 gate 뒤에만 Test9를 실행한다. Test로 재선택하거나 디버깅하지 않는다.
- train/Validation/Test27 scientific worker actualexit0, dispatcher actualexit0, 소유PID/GPU부재, 선택 binary/tensor/source/input SHA를 확인한 뒤 새 archive/manifest로 host1회 원본을 회수한다. 기존 모델의 학습·평가를 추가하지 않는다.
- 완료는 retrieval/final_receipt.status=complete,new_fit_count9,validation_replay_count9,candidate_test_count9,actual_scientific_worker_exit0_count27,actual_dispatcher_exit0_count1,held_out_test_evaluated=true와 archive/manifest/원본/결과 모든SHA가 확인된 때다.

**공유형과의 정확도 및 기존 비용 근거를 정리한다 — 회수 후 다음 작업**
- MAC9 결과와 기존 확정 Shared15 seed42/52/62만 비교한다. 데이터셋별 Validation/Test RMSE·MAE 평균±표본SD(ddof1), 각 seed Shared−MAC 절대·상대차이, 큰수량/TimeNLL·악화를 모두 기록한다. 시간반복3회와 학습3seed를 혼동하지 않는다.
- 과거 legacy시간head/joint선택 MAC를 새3seed에 합치지 않는다. 원본 MAC논문 native head의 재현으로 설명하지 않는다. 정확도 우세와 비용 우세는 실제 지표별로 판정한다. 평가 메모리 증가와 같은 반대 결과도 보존한다.
- Notion 논문 중심 page3ffbbe40-5613-8053-8078-db2860faa04c에 실제 시작·완료·결과·한계만 누적하고 읽어 확인한다. 원고 main.tex/main.reviewed.tex를 수정하거나 투고용 본문을 자동 작성하지 않는다.
