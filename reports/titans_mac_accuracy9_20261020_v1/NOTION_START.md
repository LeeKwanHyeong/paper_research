## 동일 조건 MAC9 정확도 대조 — 실제 학습 시작
### 진행 중: RAF 3seed 첫 epoch 저장 확인
사용자 승인에 따라 MAC을 RAF·Taxi·Intermittent 원본 데이터에서 seed42·52·62, 총9개 새로 학습한다. 5090에서 <mention-date start="2026-10-20" startTime="16:16" timeZone="Asia/Seoul"/> scientific dispatcher가 실제로 시작했다. 실제 관측 <mention-date start="2026-10-20" startTime="16:19" timeZone="Asia/Seoul"/> 기준 RAF42·52·62 모두 저장 epoch1/best1이며 worker 신원·lease·GPU를 확인했다. 진행3·대기6·완료0·미확정0·실패0이다. best1은 중간 결과이며 아직 정확도 개선을 판단하지 않는다.
### 현재 기준선: 공유형과 입력·head·학습·선택 조건을 맞춘다
- 원본 uncapped 데이터, 시간·수량 log1p 입력, 같은 수량·시간 head, 수량 logMSE와 시간 TimeNLL 가중치1을 사용한다. batch128, AdamW lr0.001/weight decay0.01, max300/min40/patience40을 유지한다.
- 선택 checkpoint는 전체 fullValidation 실수량 RMSE의 최초 strict finite 최소 epoch다. MAE도 같은 checkpoint에서 계산한다. MAC 내부 associative gradient clip1과 오늘 완료한 5090 효율 측정의 의미 보존 실행 최적화를 유지한다.
- RTX5090 / Python3.12.13 / torch2.11.0+cu130 / cuDNN92000 / threads4다. native MAC 단독1개와 3개 동시 합성 검사, 총4 worker의 실제 exit0를 확인한 후 학습에 진입했다. 로컬 독립 구현55검사와 비교도구20검사도 통과했다.
- 실행은 RAF3 → Taxi3 → Intermittent3이며 최대3병렬이다. 이전 MAC legacy 실험과 기존 공유형·CNN·MLP16·외부 비교군 결과는 보존한다. 5080의 추가 seed72·82 학습은 그대로 둔다.
### 다음 작업: 전체 Validation 재현 뒤 고정 Test와 공유형 비교
- 전체9 학습 종료 → 선택 checkpoint의 native Validation9 전부 재현 → Test9 → 종료·원본 archive/manifest/SHA 검증 → 기존 공유형 3seed 저장 결과와 비교한다. 현재 Validation 재현0·Test0이며 원본 회수는 아직 하지 않았다.
- 데이터셋별 Validation/Test RMSE·MAE 평균±표본SD와 모든 seed의 Shared−MAC 절대·상대차이를 정리한다. 큰수량 RMSE·MAE와 전체/큰수량 TimeNLL, 악화도 함께 보존한다.
- 공유형의 기존 Test는 5080/cuDNN91900, 새 MAC은 5090/cuDNN92000이므로 정확도 평가 Runtime 차이를 명시한다. 공유형을 다시 학습하거나 평가하지 않는다. MAC 원논문 고유 head의 재현이 아니라 공통 조건의 event adapter 대조다.
- 종료를 기다리는 유한 finisher와 로컬 비교 후속 처리를 연결했다. 기존 실행을 중복 시작하거나 실패를 자동 재시도하지 않는다. 속도 개선과 정확도 개선은 각 측정 결과로 따로 판단한다.
### 시작 증적
- 보고 경로: `reports/titans_mac_accuracy9_20261020_v1/`
- 학습 코드 commit: `b2e667d7`; 로컬 비교 코드 commit: `e2875219`
- contract raw SHA: `05b646251dbf831f4a91034111fc7a7d8f66d659a681b789c59825bbd5cfc16f`
- integrity SHA: `1dd974eccccce4c9b338bfdcd6b3f3755ee96a1172ee46de6b63c6f19db0cd04`
- START_RECEIPT SHA: `860d008ea2611f61fcef3f83c06e7b3dee299646d54c443da43166e62a1048a2`
