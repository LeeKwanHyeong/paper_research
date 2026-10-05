# GRU54 기여 분리: 3seed 대조 캠페인

상태: CPU9개 검증 완료, native GPU 검증·학습 진입 확인 전입니다.

Taxi·Intermittent·RAF × seed42·52·62 ×4조건 총36개 중 기존 GRU54 seed42 세 결과를 원본 SHA로 재사용합니다. **신규33회**입니다. 기존 폭16 기준선과 폭4 결과는 재학습하지 않습니다.

| 조건 | 보정 모듈이 직접 읽는 문맥 | 활성화 | 잔차 계수 | 보정 파라미터 |
|---|---|---|---:|---:|
| 기존 GRU54 | 관측 구간의 누적 문맥 | 두 사건 이후 |1|24,732|
| 두 사건 GRU54 | 이전·현재 두 사건 |두 사건 이후|1|24,732|
| 전체 분기 MLP16 | 이전·현재 두 사건 |8분기 모두|1|24,576|
| 전체 분기 MLP16 /8 | 이전·현재 두 사건 |8분기 모두|1/8|24,576|

공통 Encoder1 자체는 이전 문맥을 포함합니다. GRU의 게이트·비선형성과 MLP의 차이를 순환 연산 하나의 효과로 단정하지 않습니다. 파라미터 차이는156개(보정 MLP 대비0.635%)입니다. 학습 후 잔차 크기는 강제로 같게 만들지 않고 고정 Train 표본에서 진단합니다.

5080은 Taxi·RAF22조건, 5090은 Intermittent11조건을 단독 worker로 처리합니다. 각 데이터·seed의 신규 대조는 같은GPU에 묶습니다. 기존 GRU42의 이종GPU 경로는 별도 표시합니다.

최소40/최대300/patience40, 원래 Train/Validation·AdamW·loss·loader를 유지합니다. 체크포인트는 최초 최소 full Validation raw 수량 RMSE에서 고정합니다. MAE·Time NLL은 같은epoch로 비교하며 큰 수량 경계는 Taxi3449,Intermittent187,RAF200 초과입니다. 실제 학습 노출·batch 순서를 seed별 기존 원본과 대조합니다. 첫2epoch300epoch 비용 projection은 경고용이며 중단gate가 아닙니다.

각 조건36시간/전체168시간, native timeout·배타claim·서버 소유 lease·동일 native runtime 검증을 사용합니다. 자동retry/resume는 없습니다. 공용 Runtime·인증·A100을 변경하지 않습니다. 두 서버 native 검증 receipt를 묶은 training permit이 있어야 학습합니다.

분석은 데이터별3seed 차이·표본SD·전체와tail 수량·시간 지표를 함께 보존합니다. 우월성이 일관되지 않으면 손해를 포함한 tradeoff로 보고합니다. 기존 Test는 개발 과정에서 노출됐으므로 이번 신규 학습은 Test를 열지 않습니다. 이후 독립 평가와 논문 주장은 별도 동결 계약으로 진행합니다.

현재 소스는 부모 Git revision의117개 과학 파일을 그대로 보존하고 신규6개 파일의 SHA를 추가로 동결한 closure입니다. 부모commit에 신규구현이 이미 있었다고 주장하지 않습니다.
