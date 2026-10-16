# RAF 수량 목표와 raw 오류를 같은 선택 모델에서 확인

**승인 범위와 현재 기준선을 고정한다 — 완료**

- 사용자의 “RAF 수량 목표 확인18개부터 진행하고 이어 Intermittent 시간 위치·분포 폭 분리18개도 진행하자. 새 학습 없이 Train·Validation만 평가하자”를 실행 승인으로 기록한다. 이 폴더는 첫 단계 RAF만 담당한다. 새 학습·Test 재평가·checkpoint 재선택·출력 변경·원고 편집은0이다.
- 같은 RAF full/보정 제거/current-only×seed42/52/62의 선택9checkpoint×Train/Validation=18평가다. 이전 Train27과 Validation 진단의 해당18개 원문은 재현 기준선이고, 기존 split의 평가 시간을 신규 시간으로 바꾸지 않는다.
- 과학 source129 closure `c165e40bdd48f7a6ae8e330d2e4b0f8eacb67eee0737cd859bfaf35876d4754a`, revision `57da14a2a7a5687e1bdef7d3164f3e8c0051da59`, 같은 전처리·B128·lookback/cap84·수량 경계>200·같은 선택 state를 유지한다. 학습 당시 이동하는 모델의 batch loss와 선택 모델의 평가를 구분한다.

**실제 log 목표와 raw 오류를 계측한다 — 진행 중**

- frozen forward가 반환하는 실제 per-target log1p 제곱오차와 같은 dtype의 signed log 오차를 전체·고정 수량 bin·tail에 집계한다. 원래 raw 전체/큰 수량 RMSE·MAE·TimeNLL 및 대상·truth/history digest와 state 불변을 먼저 재현한다.
- 원별 예측·목표 배열은 디스크에 저장하지 않는다. 집단 평균 예측/logit에 log를 취해 목표 오차를 만들어내지 않는다. model.eval·inference_mode·parameter update0을 확인한다.
- 같은 dataset/seed의 세 arm을 최대3개 동시 평가한다. native synthetic3route, 실제18worker exit0·pipeline OS exit0, immutable 원본 archive/manifest·로컬 SHA·독립 QA를 통과한 경우에만 결과를 사용한다. 기존 실패·claim을 지우거나 자동 retry/resume하지 않는다.

**원본과 종료를 확정한 뒤 두 번째 단계로 인계한다 — 다음 작업**

- RAF의 모든3seed·두 대조·raw/log 방향과 손해를 함께 분석한다. 목표 간 상충의 관측을 objective 변경 효과나 용량/최적화 원인의 증명으로 확대하지 않는다.
- 이 단계의 실제 종료·원본 회수·로컬 검증이 완료된 뒤에만 [Intermittent 단계](../titantpp_intermittent_time_cross_20261012_v1/README.md)를 launch한다. 원고·새 학습·검색은 자동 실행하지 않는다.
- 평가 코드와 독립 QA는 별도 파일로 병렬 준비하고, 계약 봉인·5080 GPU 실행·원본 회수·단계 인계는 현재 세션에서 직렬 진행한다.

**합성 CUDA 검증에서 중단된 실행을 보존한다 — 차단됨 / v2로 운영 수정 인계**

- 첫 native synthetic route의 eval-mode GRU backward 제약으로 실제 pipeline exit1이었다. 실데이터 Train/Validation 평가0개이며 기존 claim·실패·운영 seal은 보존했다. 이 root의18개 완료를 주장하지 않는다.
- 합성 gradient fixture만 일시 ATen 경로로 확인하는 [별도 v2](../titantpp_raf_log_raw_20261012_v2/README.md)로 수정 검증했다. 기존 v1 자동 retry·덮어쓰기0이다.
