# CNN+GRU54 3seed 검증

기존 대조 실험 중지와 checkpoint·이력 SHA 회수는 완료했습니다. 새 CNN+GRU54 계약·CPU 검증은 완료했으며 두 서버 native qualification 후 학습을 시작합니다.

|서버|신규 학습|재사용|
|---|---|---|
|5080|Taxi·RAF × seed52/62,4조건|동일 모델 seed42 Taxi·RAF|
|5090|Intermittent × seed52/62,2조건|동일 모델 seed42 Intermittent|

모델·손실·데이터·초기화·min40/max300/patience40·최초 최소 finite full Validation raw RMSE 선택 기준을 유지합니다. 부모 과학 소스117개는 byte/SHA가 같고, 기존123개 snapshot을 그대로 보존합니다. 새 운영 파일은 별도 SHA로 동결했습니다. 서버별 worker1과 native36h/전체168h guard·자동 retry 금지를 유지합니다. 공용 Runtime을 변경하지 않습니다.

**CNN+GRU54를 두 서버에서 검증하고 학습한다 — 진행 중**
- 계약의 exact GPU·Runtime, CNN/GRU 초기화·인과성·결정성·메모리와 seed42 selected/last full Validation replay를 확인합니다.
- 두 native qualification이 통과해야 host별 training permit으로 새6조건을 시작합니다.

**3seed 결과를 외부 비교군과 연결한다 — 다음 작업**
- 학습 완료와 selected/last full Validation 및 checkpoint 원본 SHA 회수를 확인합니다.
- 같은 seed의 S2P2·다른 외부5모델·MLP16·강한 내부 기준선에 전체/큰 수량 RMSE·MAE·TimeNLL과 seed별 변동을 연결합니다. Test3seed는 별도 동결 평가로 이어집니다.
- 이번 결합모델3seed 검증만으로 CNN/GRU 독립 기여나 interaction을 확정하지 않습니다. 기존 Test 개발 노출과 seed42 A100/RTX 계보를 보존합니다. CPU binary 재추론 감사와 새 독립 평가는 별도 미완료입니다.

원본 계약: execution_contract.json / 재사용 원본: reuse_seed42_registry.json / 이전 실험 중단: predecessor_user_stop_receipt.json
