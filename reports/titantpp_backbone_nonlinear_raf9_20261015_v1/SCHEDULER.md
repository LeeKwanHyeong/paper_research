# CNN backbone 개선 반복 운영

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-10-15 Asia/Seoul
- Verification Status: PREPARING; 실제 학습/결과 검증은 각 receipt로 확인
- Version Label: backbone_loop_policy_v1

## 목적과 승인
사용자는 2026-10-15 RAF 우선 5090 최대3 병렬, 기준 통과 후 동일 backbone으로 Taxi5090와 Intermittent5080, 성능 미달 시 원인 분석과 새 backbone 후보의 반복 구현·검증·학습·Validation/Test·원본 회수·Notion 기록을 승인했다. 최신 원문은 “ㅇㅇ 진행하자. 승인할것들에 대해서는 미리 승인한다. 지금부터 진행시켜!”다. 이 범위의 정상 단계마다 재승인을 요구하지 않는다.

프로젝트 /Users/igwanhyeong/PycharmProjects/paper_research의 TEST_SESSION_PROTOCOL.md 0·1·9·12, 이 파일, LOOP_STATE.json, 해당 루프의 과학 계약·봉인·최신 receipt를 먼저 읽는다. 최신 승인은 이 신규 격리 캠페인의 학습과 고정 checkpoint Test를 포함한다. 과거 자동화의 관측전용 제한을 신규 캠페인의 금지로 확대하지 않으며, 과거 캠페인의 봉인·과학 조건은 변경하지 않는다.

목표는 CNN 내부의 인과적 국소 이력 처리 개선이다. 공통 입력/log1p 변환/손실/수량·시간 head/메모리/seed42·52·62/학습예산/선택기준은 유지한다. RevIN·GRU·head 변경은 이번 반복 범위에 넣지 않는다. 원고 main.tex/main.reviewed.tex, 기존 결과/실패/claim, 공용 Runtime/driver/auth, 다른 자동화, 삭제된 A100을 변경·접속하지 않는다. 외부 유료 GPU를 추가하지 않는다.

## 첫 루프 — RAF 5090
- schema: titantpp_backbone_nonlinear_raf9_20261015_v1
- bundle: search_artifacts/titantpp_backbone_nonlinear_raf9_20261015_v1
- report: reports/titantpp_backbone_nonlinear_raf9_20261015_v1
- remote root: /home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_nonlinear_raf9_20261015_v1_5090
- arms: titantpp_cnn / titantpp_cnn_channel_nonlinear_v1 / titantpp_cnn_joint_nonlinear_v1
- 3구조×3seed=9 fresh fits, 5090 전체 동시 scientific fit 최대3.
- 기존 CNN은 동결 원형의 별도 식별자이며, 동일 5090 기준선3을 새로 만든다. 기존5080 CNN 결과도 보존하여 원래 기준과 별도로 비교한다.
- 신규 두 구조는 첫 encoder block의 Q/K/V 각각에 같은 최근3관측 범위를 사용하는 비선형 경로를 둔다. 채널별 대조군과 시점·채널 결합 후보는 추가 파라미터 예산을 일치시키고 공통 가중치와 초기 경로 영향을 통제한다. 정확한 식과 차원은 실행 전 계약에 봉인한다.
- 새 격리 경로에 source/input/operation/approval을 결속하고 로컬 합성·계약검사, native GPU/Runtime/입력SHA, 실제3동시 qualification을 통과한 뒤 tmux 학습을 시작한다. qualification은 scientific fit으로 세지 않는다.
- 진행 중 root를 다시 설치하거나 시작하지 않는다. 기존 다른 작업이 점유하면 기다리며 종료하거나 빼앗지 않는다. 원본 덮어쓰기와 과거 claim 삭제는 하지 않는다.

## 검증과 판정
각 학습의 전체 fullValidation raw quantity RMSE 최초 strict finite 최소 epoch로 checkpoint를 선택한다. 동률은 첫 epoch다. terminal 원본과 선택 모델을 따로 보존하고 Test로 재선택하지 않는다. 실제 worker/dispatcher/pipeline 종료, owner/GPU 부재와 binary SHA 뒤 원본을 회수한다. wrapper 자신의 OS exit를 직접 보지 않았으면 직접 관측했다고 주장하지 않는다.

선택 모델의 native Validation 재현이 전부 통과한 후 같은 checkpoint로 Test를 평가한다. 결과는 dataset/split/campaign/source별로 구분한다. 판정은 반올림 전 3seed 평균을 사용하고 표본SD(ddof1)와 모든 seed별 절대·상대차이를 보고한다.

사용자 성능 기준: Validation과 Test 각각에서 RMSE와 MAE 모두 기존 TitanTPP CNN 및 동일 대상의 외부 비교군 지표별 최저 평균보다 엄격히 낮아야 한다. 한 split/지표라도 미달이면 pass=false. 원본CNN과 새 동일서버CNN 비교를 모두 공개한다. 외부 RMSE1위와 MAE1위가 다르면 지표별 최우수 기준이라고 부른다. baseline source/split/target count/지표 정의/선택 조건/SHA 검증이 없으면 미확정이며 통과로 승격하지 않는다. baseline registry는 후보 평가 전에 고정한다. 의미 없는 수치오차나 seed 불안정을 실질적·유의한 개선으로 과장하지 않는다.

주 방법론 후보는 joint nonlinear이고 channel nonlinear는 용량·비선형 대조군이다. 대조군 대비 이득은 별도 방법론 근거로 보고하며 사용자 성능 기준에 몰래 추가하거나 빼지 않는다. 대조군만 좋아지면 joint 방법론 성공으로 표시하지 않는다. 같은 checkpoint의 전체/큰수량 RMSE·MAE와 전체/큰수량 recorded positive-integer TimeNLL, 단순 기준선도 보존·보고한다.

이 Test는 이전 설계와 이후 반복에서 참고하므로 exploratory/test-informed 평가다. 독립적 일반화 입증·보편적 우월성·통계유의성·원인 확정을 주장하지 않는다. 독립 확인에는 아직 보지 않은 자료/기간이 필요하며 기존 Test 일부를 새 독립자료로 재명명하지 않는다.

## 반복과 데이터셋 확장
상태 순서: preparation → local_verified → native_qualified → training → terminal_verified → collected → validation_replay → test → compared → notion_recorded → next_round 또는 transfer.

RAF가 기준을 통과하면 동일 구조와 공통 과학 조건을 고정하여 Taxi5090와 Intermittent5080를 병렬 검증한다. 각 서버 최대3. 5080은 기존 taxi-history-cnn6-validation의 학습뿐 아니라 승인된 Validation/Test/회수까지 완료되고 실제 유휴임이 확인된 뒤만 사용한다. 완료 cache는 현재 전역유휴 증명이 아니다. 같은 host·Runtime CNN 기준선이 없으면 별도 재현 기준선을 만들고 기존 결과와 구분한다.

RAF 또는 확장 데이터셋에서 미달하면 실패 결과를 보존하고 Train/Validation 및 승인된 Test 오류를 분석하여 한 가지 backbone 가설과 필요한 대조군을 다음 버전 계약에 기록한다. source를 바꾼 새 후보는 새 root/ID/봉인으로 5090 RAF부터 검증한다. 이전 구조의 다른 데이터셋 통과를 새 구조에 이식하지 않는다. 데이터셋별 좋은 서로 다른 구조를 하나의 공통 모델 성과로 합치지 않는다. 같은 구조의 성능 미달만으로 같은 fit을 재실행하지 않는다.

새 루프의 구현·검증과 이후 학습은 승인되어 있다. 그러나 미확인 source/SHA/process identity·연결 실패·실행 오류·회수 불완전 상태를 성능 미달로 취급하여 새 학습을 중복 시작하지 않는다. 증적을 남기고 원인을 확인한다. 안전한 국소 수정은 새 버전에서 검증하되 기존 실행/실패/claim을 보존한다. 실패 원인 미확인 상태에서는 원격 재시도를 반복하지 않는다. 한 회차에는 한 캠페인의 상태 전환만 진행하며 동시에 서로 다른 과학 가설들을 무제한 시작하지 않는다. 새로운 가설의 근거가 없으면 이를 기록하고 무의미한 무작위 탐색을 하지 않는다.

## 기록과 스케줄
Notion https://app.notion.com/p/Backbone-Enhancement-History-3fabbe40561380e591e9e66ff02bb190 에 루프별로 가설·이전 결과와 변경 이유·정확한 구조·고정 조건·코드/계약/source/input SHA·host/GPU/Runtime/seed·상태/실제KST·Validation/Test 평균±SD·seed별차이·외부기준·통과/미달·불리한결과·해석한계·다음변경을 누적한다. 원본은 로컬 immutable archive/manifest로 보존하고 Notion에는 요약과 근거 위치를 기록한다. 각 루프 시작과 종료/실패를 별도 기록하고 과거 루프 내용을 덮어쓰지 않는다. Notion 장애 시 로컬 미전송 기록을 유지하고 누락을 알린다.

heartbeat는 매시간 현재 루프의 검증된 진입점으로 한 번 상태를 전진한다. 실행 명령·integrity SHA는 LOCAL_READY/START_RECEIPT와 LOOP_STATE에서 확인하며 아직 준비되지 않은 명령을 추측하여 실행하지 않는다. 준비 중이면 계약·구현·검증을 계속한다. 같은 미완료 claim/attempt에 중복 실행하지 않는다. 원격 조회는 live host별 한 회차 한 번, 종료 cache는 SHA 재검증 후 원시각을 유지한다. ETA는 해당 fit 최근 최대10개 finite양수 완료 epoch 중앙값만 사용하고 best 미갱신 조건과 e300을 구분한다. 기한은 계약에서 고정하고 실행 중 연장하지 않는다.

정상 진행·대기·변화없음은 조용히 유지한다. 루프 시작, 주요 단계 전환, 완료, 성능 판정, 실패/사용자 조치 때만 알린다. 기존 Taxi 자동화는 자체 범위만 담당하고 이 자동화가 대신 재실행하지 않는다. 첫 RAF 완료만으로 반복 자동화를 삭제하지 않는다. 동일 구조가 승인된 세 데이터셋의 기준을 충족하고 결과·증적·Notion 기록까지 끝나거나 사용자가 중단했을 때 종료를 보고한다.
