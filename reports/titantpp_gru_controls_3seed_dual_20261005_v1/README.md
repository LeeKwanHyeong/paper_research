# GRU54 기여 분리: 3seed 대조 캠페인

상태: **두 서버 native 검증 통과, 신규33회 학습 시작과 첫 RTX 체크포인트 저장 확인 완료**입니다. 전체 학습 완료를 뜻하지 않습니다.

Taxi·Intermittent·RAF × seed42·52·62 ×4조건 총36개 중 기존 GRU54 seed42 세 결과를 원본 SHA로 재사용합니다. **신규33회**이며 기존 MLP16 기준선9개와 폭4 참조는 재학습하지 않습니다. Instacart는 후속입니다.

| 조건 | 보정 모듈이 직접 읽는 문맥 | 활성화 | 잔차 계수 | 보정 파라미터 |
|---|---|---|---:|---:|
| 기존 GRU54 | 관측 구간의 누적 문맥 | 두 관측 사건 이후 |1|24,732|
| 두 사건 GRU54 | 이전·현재 두 사건 |두 관측 사건 이후|1|24,732|
| 전체 분기 MLP16 | 이전·현재 두 사건 |두 사건 이후8분기 모두|1|24,576|
| 전체 분기 MLP16 /8 | 이전·현재 두 사건 |두 사건 이후8분기 모두|1/8|24,576|

공통 Encoder1 자체는 이전 문맥을 포함합니다. 대조는 보정 모듈이 직접 사용하는 문맥을 맞춥니다. GRU의 게이트·비선형성과 MLP의 차이를 순환 연산 하나의 효과로 단정하지 않습니다. 파라미터 차이는156개(보정 MLP 대비0.635%)입니다.

실제 학습된 잔차 크기를 강제로 같게 만들지는 않습니다. 고정 N/n 가중 Train 표본의 norm으로 진단하며 Validation 잔차 norm을 수집했다고 주장하지 않습니다. 재사용 GRU seed42의 추가 norm은 없는 값으로 보존합니다. 잔차 계수1 대1/8 효과는 동일 MLP의3seed 대조로 확인합니다.

## 실제 학습 진입 증거

| 서버 | 실제 관측(KST) | 신규 완료/실패/진행/대기/미확정 | 현재 조건 | 저장/best epoch |
|---|---|---|---|---|
|5080|2026-10-05 21:22:23|0/0/1/21/0|Taxi·두 사건 GRU54·seed42|12/9|
|5090|2026-10-05 21:22:28|0/0/1/10/0|Intermittent·두 사건 GRU54·seed42|1/1|

36개 canonical 조건 기준으로 기존3개 완료, 신규2개 진행,31개 대기입니다. 위 시각 이후 진행은 새 시간별 모니터 `titantpp-gru-3seed`로 갱신합니다. 모니터를 설정했고, 완료 후 원본 SHA 회수와 Validation 집계까지 확인합니다. 실제 supervisor/worker argv·PID와 GPU UUID/PID, 저장 checkpoint receipt·history·CPU에서 읽은 실제 epoch를 대조했습니다.

5080은 Taxi·RAF22조건,5090은 Intermittent11조건을 서버당 worker1개로 처리합니다. 각 데이터·seed의 신규 대조는 같은 GPU에 묶습니다. 기존 GRU42의 이종GPU 경로는 별도 표시합니다.

## 고정 계약과 검증

- 과학 계약 canonical SHA: `221e98d5f489842832a359d0c2d4d1ddeb4b22d0fbe08e5e564622a6c7f11b5f`.
- 과학 source123 closure: `4e94229fae002dc678025ecf056862f832c64459969f403b3cb5ac8cea8ba7b0`.
- 동결 구현 commit: `834c0f08e7cb4326401ee7f0df01b08061a9732d`, `paper_research/master`, Push 안 함.
- 부모117개 과학 파일은 원본 revision 그대로 보존하고 신규6개 파일을 추가했습니다. 부모commit에 신규구현이 이미 있었다고 주장하지 않습니다.
- CPU 구조 검증9개 통과. 두 서버에서4조건 결정성·인과성·활성 gradient·메모리, 기존 폭4 Validation9개와 GRU42 Validation3개 연결 검증을 통과했습니다. native 최대 할당 약1.94GiB입니다.
- 최소40/최대300/patience40, 원래 Train/Validation·AdamW·loss·loader를 유지합니다. 최초 최소 full Validation raw 수량 RMSE로 checkpoint를 고정하고, MAE·Time NLL은 같은epoch로 비교합니다.
- 큰 수량은 원래 Train 경계 Taxi3449·Intermittent187·RAF200 초과입니다. 실제 노출·batch prefix를 seed별 기존 원본과 대조합니다.
- 조건36시간/전체168시간, native timeout·배타claim·소유 lease를 적용합니다. 자동retry/resume는 없습니다. 첫2epoch300epoch 비용 projection은 경고이며 중단gate가 아닙니다.
- 공용 Runtime·인증·A100을 변경하지 않았습니다. 두 native receipt를 묶은 training permit으로 진입했습니다.

주 MLP16 기준선9개를 별도 `width16_validation_refs.json`으로 연결했습니다. 학습 중 생성되는 폭4 비교는 노출 및 보조 진단이고, 주 대조는 MLP16입니다. 독립 CLI의 route/상대입력경로 보완은 현재 로컬 코드에만 적용했으며 실행 중 동결 snapshot은 변경하지 않았습니다.

## 결과 해석과 후속 작업

`analysis_contract.json`은4개 paired 대조, 전체·큰 수량 RMSE/MAE/TimeNLL,3seed 평균·표본SD·seed별 차이와 차이의 표본SD를 정의합니다. 과학 계약의 loss/선택 기준을 바꾸지 않는 별도 집계 규칙입니다. 서로 다른 수량 단위의 raw RMSE를 합쳐 전체 순위를 만들지 않습니다. 일부 seed·시간·큰 수량 지표가 악화되면 그대로 보존합니다.

신규33조건의 selected/last full Validation·terminal scientific_success·소형 SHA·실제 소유 서버 종료를 확인한 뒤, checkpoint binary와 원본 manifest SHA를 별도로 회수하고36조건 및 MLP16 기준선을 집계합니다. 회수 실패·중단은 증거를 남기고 사용자 조치 필요로 보고하며 자동 재시도하지 않습니다. CPU binary 재추론 감사는 원본 SHA 검증과 별도입니다.

이번 신규3seed는 Train/Validation만 사용합니다. 기존 Test는 개발 과정에서 노출됐으며 이번 캠페인에서는 열지 않습니다. 이후 독립 평가와 논문 주장 검증은 별도 동결 계약으로 진행합니다.

**남은 작업**

1. **세 데이터의3seed 대조를 마친다 — 진행 중**: 5080·5090의 신규33fit를 관측하고, full Validation·원본 SHA와 서버 종료를 검증합니다.
2. **GRU54의 장점과 손해를 설명한다 — 다음 작업**: MLP16 및4조건의 seed별 전체·큰 수량·시간 지표와 Train 잔차 진단을 연결해 기여 후보를 정합니다.
3. **독립 평가로 논문 주장을 검증한다 — 다음 작업**: 위 결론 이후 별도 평가 계약과 분할을 동결합니다. 새 독립 평가와 논문 결론은 아직 확정하지 않았습니다.
