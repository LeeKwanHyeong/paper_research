# 공유형·외부 비교군 추가2seed와 논문 근거 준비

## 현재 기준선을 보존한다 — 완료

사용자 2026-10-20 원문: “ㅇㅇ 그럼 계획대로 5080에서 공유형과 외부 비교군의 추가 2seed 실험을 원본 세 데이터셋으로 시작하자. 최대 3개씩 병렬로 돌리고, 5090 학습은 그대로 둬. Intermittent 최종 best가 이번 고정본과 같은지 확인되면 논문 작성도 바로 들어가자.”

프로젝트 `/Users/igwanhyeong/PycharmProjects/paper_research`의 TEST_SESSION_PROTOCOL.md 0·1·9·12와 이 문서, SCIENCE_SPEC.json, FROZEN_CONDITIONS.json 및 실제 receipt를 먼저 읽는다. 신규42는 Shared15/RMTPP/THP/NHP/SAHP/S2P2 matched-head/AttNHP matched-head ×원본RAF/Taxi/Intermittent ×seed72/82다. 기존42/52/62, CNN, MLP16, checkpoint·source·실패·중단 작업·원고는 보존한다. Cap·새 백본탐색·추가학습으로 확대하지 않는다. 5090 원본9 작업과 기존 종료 대기를 변경하지 않는다. 사용자 승인 범위의 새5080 전용 경로 실행·평가·회수는 재질문하지 않는다.

## 5080에서 신규42를 마무리한다 — 진행 중

코드 commit `f2975cbd`, 준비 계약 rawSHA `4d88a5df41c41ed4076b766c4f1383263e29f2584c22cbab21306fdf126edac2`, canonical `5927e2c67b891e66b45de2e428ebc24165e5572e57d9c8ccb09f1e2057b164d8`, integrity `44794f5e4b9ff74e3ff903f964e619dbe565961ddc64c2117fa109b5bcc764ac`다. 기존983소스를 동일바이트 상속하고 신규 운영/과학 실행4파일만 더한987소스를 봉인했다. 로컬47tests/독립QA PASS. 실제 배포 요청은 2026-10-20 14:31:02KST이며 이는 qualification·학습 완료를 뜻하지 않는다. 실제 시작은 START_RECEIPT.json과 보존한 status 원문을 기준으로 한다.

신규 root는 SCIENCE_SPEC.native.root다. 동일5080 GPU/Runtime, 입력/log1p·logMSE+시간NLL가중치1·공통head·batch128·300/min40/patience40·최초 strictfinite Validation rawRMSE 선택·원래 기한을 유지한다. 이미 deploy/launch했으므로 prepare/seal/deploy/start를 다시하지 않는다. GPU를 점유한 옛 paused4는 그대로 둔다. 실시간 신원과 source/input/runtime·모델7 native합성검증·실제3동시 admission을 통과한 뒤만 학습한다. Train42→Val42의 전체 재현 gate→고정 Test42 순서다. 학습·평가 최대3병렬, 재선택0·기한연장0이다.

로컬 finite finisher PID35139가 이미 단 한 번 연결됐다. `manage_titantpp_shared_external_extra42 --mode finish` 또는 `start-finisher`를 중복 실행하지 않는다. finisher는 전체 종료 후 원본 archive/manifest/SHA 검증과 NEW_SEED_RESULTS.json을 만든다. 정상 대기 중 stdout0은 실패가 아니다. 실패·불완전 회수·기존 attempt/claim이 있으면 자동retry/resume/kill/삭제0, stdout/stderr·실제exit를 보존하고 한번 알린다. 기존 원격 작업을 반복하지 않는다.

회차마다 먼저 local QUEUE_FAILURE/QUEUE_COMPLETE, retrieval/final_receipt, finish.stdout/stderr를 읽는다. 진행 확인이 필요할 때만 프로젝트 root에서 아래 status를 회차당 한번 실행한다. 원격 반복polling/별도 observer를 추가하지 않는다.

`PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.manage_titantpp_shared_external_extra42 --mode status --integrity-sha 44794f5e4b9ff74e3ff903f964e619dbe565961ddc64c2117fa109b5bcc764ac`

완료cache가 있으면 SSH0으로 모든 raw SHA·원본·종료관측·선택·결과를 재검증한다. 상태조회 실패는 학습수치 실패가 아니다. 과거 관측시각을 현재로 바꾸지 않는다. 외부/기존CNN·MLP16을 다시 평가하지 않는다. 운영 완료와 성능 우세 판정을 구분한다.

## Intermittent 최종 선택을 고정본과 대조한다 — 외부 작업 대기

5090 원본9의 기존 finisher PID71592를 그대로 둔다. 해당 bundle execution_5090_v1/retrieval/final_receipt.json이 없으면 기다린다. 기존5090 finish/collect/start-finisher를 반복하지 않는다.

프로젝트 root에서 아래 local-only 도구를 한번 실행한다. 원본 전체 종료·회수·Validation9·모든 source/archive/manifest/원본/결과SHA 및 고정 Test3의 증거를 검증하고, Intermittent epoch5/5/11과 canonical 모델 tensor SHA가 같은지 비교한다. 직렬화 binary SHA 차이를 모델 차이로 보지 않는다.

`PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 reports/titantpp_shared_external_extra42_20261020_v1/verify_paper_gate.py`

없으면 waiting, 같으면 PAPER_START_GATE.json PASS, 다르면 MISMATCH다. MISMATCH를 PASS로 바꾸지 않고 기존 고정본·최종본 모두 보존해 보고한다. 새 최종 Test를 임의 반복해 덮어쓰지 않는다.

## 논문을 위한 연구자료를 준비한다 — gate 통과 후 다음 작업

PAPER_START_GATE PASS이면 추가42 종료를 기다리지 않고 별도 `reports/titantpp_shared_paper_preparation_20261020_v1/`에서 확정3seed 결과표, 구현에 대응하는 수식·방법 사실, 근거 목록, 재현 자료, 사람이 작성할 항목별 빈 구조를 만든다. 기존 원고 main.tex/main.reviewed.tex를 수정하지 않는다. PAPER_EVIDENCE_LEDGER.md와 VENUE_REQUIREMENTS.md를 먼저 읽는다.

공식 PAKDD2027 authors-kit은 생성형AI를 문법교정으로 제한한다. 투고용 초록·서론·방법·결론 본문을 자동 작성하지 않는다. 이 제한은 실험 검증 및 연구자료 준비와 구분한다. 사용자 원문이 준비되면 문법교정 범위로 돕는다. 성능·원리·속도·채택가능성을 과장하지 않는다. RAF 전체미우세, Taxi/Inter seed예외 및 TimeNLL악화, 반복Test참고, 공통head 외부구현, host/Runtime 차이를 모두 남긴다. 측정 전 속도기여 주장0.

## 완료된 추가2seed를 기존3seed와 비교한다 — 다음 작업

NEW_SEED_RESULTS가 complete이고 Train42/Val42/Test42 actualexit0 및 전체원본SHA를 확인한 뒤 새 별도5seed 보고서를 만든다. 기존3seed 결과를 수정하지 않는다. 데이터셋별 Shared+외부6종의 Validation/Test RMSE·MAE 평균±표본SD(ddof1), 모든seed Shared−각외부 절대/상대차이, 기존3/신규2 개별표를 작성한다. 서로 다른 데이터셋 raw오류를 평균내지 않는다. 큰수량/시간지표는 근거가 있는 만큼 보존하며 부족하면N/A와 이유를 적는다. 어떤seed·불리한결과도 제외하지 않는다. 새 로컬 비교도구가 필요하면 별도 파일로 구현·합성검증하되 봉인된4실행파일은 변경하지 않는다.

정상 진행·대기·변화없음은 조용히 유지하고, 주요단계전환·원고근거 gate판정·전체완료·실패·사용자조치가 있을 때만 알린다. 이 작업은 승인된42개와 조건부 논문 연구자료 준비의 유한 작업이다. 새 백본개선 루프를 만들지 않는다. 추가42 검증·5seed 보고 및 논문근거 준비 완료 또는 사용자 중단 때 이 후속 자동화 하나만 종료한다. 다른 자동화는 변경하지 않는다.

## 실제 시작 증거 — 2026-10-20 14:32:30 KST

START_RECEIPT.json은 보존한 원격 status stdout/SHA와 연결된다. native 모델7 검증과 실제3동시 admission의10worker exit0 뒤 첫3fit가 시작했다. 완료0·진행3·대기39·미확정0이다. Shared15 RAF72 PID616826 저장7/best6, RAF82 PID616827 저장7/best1, RMTPP RAF72 PID616828 저장11/best7이다. 실제worker의 PID/argv/start_ticks·lease와GPU를 확인했고 기존 paused4는 유지됐다. 이 과거시각을 새 관측시각으로 사용하지 않는다. 원 wrapper 자신의 OSexit는 직접 관측하지 않았다.

후속 heartbeat id는 `2seed`다. 15분 간격으로 이 유한 작업만 이어가며 정상 진행에는 알리지 않는다. 자동화 도구로 실제 등록했고 FINISHER_STARTED.json의 유한 종료 대기와 역할을 구분한다. 후속 heartbeat는 학습을 시작하거나 finish/collect를 중복 호출하지 않는다.


## 최신 단계 — 2026-10-20 14:48:43 KST 관측 및 논문 근거 준비 완료

이 단락이 위 과거 대기 설명을 대체한다. 5090 원본9 학습·Validation9·원본227개 회수가2026-10-20 14:34:44 KST 완료됐다. 원본 final receipt SHA는 `5f0deda74a3e606cd7a5819f3a15c55cf8c24105f5708abcf1dfc0f3ae7412d1`다. PAPER_START_GATE.json PASS이며 Intermittent 최종epoch5/5/11과 고정 Test3의canonical 모델가중치가 모두 같았다. 기존 임시 Test 원본은 그대로 두며 새추론을 실행하지 않았다. 이제 5090 종료를 기다리거나 finish/collect를 다시 실행하지 않는다. 과거 종료관측을 현재전역GPU유휴 증거로 사용하지 않는다.

논문 연구자료 `reports/titantpp_shared_paper_preparation_20261020_v1/`에 PREPARATION_RECEIPT.status=complete를 만들었다. 확정3seed144endpoint·288평균/표본SD(36N/A)·주지표216seed차이·전체648차이, 방법사실·재현·인간작성빈구조·미해결질문을 제공한다. 독립수치QA PASS 및 기존원본524개SHA 검증을 확인했다. 원고 본문은 자동작성하지 않았다. 후속회차는 이자료를 재생성하지 말고 receipt/manifest/SHA로 확인한다. 원래3seed자료는 추가5seed자료로 덮어쓰지 않는다.

5080 신규42는 실제관측14:48:43 KST에 학습완료6·진행3·대기33·미확정0, 과학실패0이다. RAF Shared15/RMTPP/THP 각각72/82의6fit 실제exit0·완료receipt가 있으며, NHP72/82·SAHP72가실제worker/GPU에서진행중이다. 전체42 nativeValidation/Test·최종회수는아직미완료다. 원본 status와 SHA는 `progress/20261020T054843Z/SUMMARY.json`에 결속된다. 이 관측시각을 새시각으로 사용하지 않는다.

5seed 집계기 `paper.scripts.compare_titantpp_shared_external_five_seed`를 별도신규파일로 준비했고 focused16tests/독립QA PASS다. sealed987/기존4실행소스는변경하지않았다. 실제42완료자료 전체로 실행한 검증은 아직없다. 추가42 finalreceipt와 모든증거가complete일때만 로컬에서 `PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m paper.scripts.compare_titantpp_shared_external_five_seed --check-only`를 실행한다. 실제ready 뒤에만 --check-only를빼고 한 번 산출한다. 기본출력 `reports/titantpp_shared_external_extra42_20261020_v1/five_seed_aggregation_v1/FIVE_SEED_RESULTS.json/.md`를 보존한다. 기존출력존재시 덮어쓰기나 재학습을하지않고 source/출력SHA를 검증한다. 실제자료에서 도구오류가 확인되면 원본·실패를 보존한 순수로컬 국소수정·합성검증·독립검토만 허용하며 nativeTest/학습을 재실행해 디버깅하지않는다.


후속 관측의 최신 포인터는 `LATEST_PROGRESS.json`이다. 해당 summary 경로·rawSHA·실제관측시각을 확인하고, 이 문서에 남은 과거 관측을 현재 상태로 사용하지 않는다. 정상 학습 중 완료 개수의 증가만 있고 데이터셋/단계 전환·실패·사용자조치가 없으면 알리지 않는다.


## 사용자 지적에 따른 해석 정정 — 2026-10-20

현재해석은 `reports/titantpp_shared_paper_interpretation_correction_20261020_v1/CORRECTION.md`를우선읽는다. 기존HistoryMLP의Titans-MAC대비5080효율실측은이미완료돼있으며 Shared15자체의배수와구분해재사용한다. Backbone전체의성능·비용기여에용량일치대조를필수로두지않는다. 현재공유형실험은Train학습/Validation선택/고정Test평가가확인되며, Test를봤다는이유만으로학습·구조·checkpoint튜닝에사용했다고단정하지않는다. 과거다른시도/관행의일반적한계를이번실험의확정사실로옮겨적지않는다. 기존frozen문서나집계기의구식해석문구는원문을보존하되정정기록과함께제시한다. 42개학습·평가경로/기존수치·checkpoint·source봉인은변경하지않는다.
