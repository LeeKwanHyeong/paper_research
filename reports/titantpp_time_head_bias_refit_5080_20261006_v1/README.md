# CNN+GRU54 수량 동결·시간 bias 2개 재적합

**완료 원본과 5080 실행 환경을 확인한다 — 완료 / 현재 기준선**
- Taxi·RAF seed42/52/62의 원래 수량 선택 checkpoint 6개와 원본 terminal·selected full Validation·SHA를 연결했습니다. 2026-10-06 09:26 KST 현재 5080 GPU UUID와 compute PID 부재, 원래 dispatcher 종료코드0, 과학 소스123개 SHA를 확인했습니다.
- 5090의 Intermittent seed62 학습은 기존 계약으로 진행합니다. 새 작업은 5080의 별도 root에서 수행하며 기존 소스·checkpoint·이력·claim을 보존합니다.

**시간 bias만 Train에서 재적합하도록 구현한다 — 완료 / 로컬 검증**
- 학습 대상은 b_t와 w_raw 두 scalar입니다. CNN·GRU·Hard-LMM·수량 출력부·시간의 조건부 projection·모든 buffer를 고정하고 eval 동작을 유지합니다.
- 원래 시간 단위·양의 정수 간격 확률 질량·NLL 계산을 그대로 씁니다. Train의 frozen feature를 사용하며 Validation은 fitting에 넣지 않습니다.
- 새로운 head 단계는 Adam LR0.001, batch128, clip1, 최대40 epoch·patience10, 최초 최소 finite full Validation TimeNLL을 선택합니다. 원본 E0를 후보에 포함하며 동률·불개선이면 원본을 유지합니다. 원래 CNN+GRU54의 epoch/patience/수량 selector를 바꾸지 않습니다.
- 서버당 worker1, 조건1시간·전체6.5시간, CPU16GiB·GPU80% 상한, 자동 retry 없이 실행합니다. 130개 파라미터 재학습은 이번에 자동으로 이어서 실행하지 않습니다.
- CPU 집중 테스트41개를 통과했습니다. 두 scalar gradient/복원·원래 정수 시간 likelihood·전체 수량 byteSHA·동결 tensor/buffer·E0 선택·실패 재시도 거부·운영 허가 결합·실제 완료 epoch timing을 확인했습니다. Controller의 증적 성공 판정·허가 환경 전달·소유 child 종료 처리도 별도 코드 검토를 통과했습니다. 실제 GPU 검증과 학습 완료는 아직 확인하지 않았습니다.

**수량 보존과 시간 성능을 비교한다 — 다음 작업**
- 동일 타깃 순서에서 수량 예측 tensor byteSHA, 고정 weight·buffer SHA가 원본과 일치해야 합니다. 전체와 Train p99 초과 큰 수량 RMSE·MAE 및 같은 head epoch의 TimeNLL을 비교합니다. Taxi 경계3449, RAF200을 유지합니다.
- 시작 전 native qualification과 원본 Validation 재현을 확인합니다. 완료 시 selected/last head tensor·이력·단계별 timing·원본 SHA 연결을 회수합니다.
- 원래 Train을 본 encoder의 사후 재적합이며 독립 calibration이라고 표현하지 않습니다. Validation은 개발 선택 결과이고 Test는 접근하지 않습니다. S2P2의 동일 재적합 비교·Intermittent 후속·CPU binary 재추론 감사·독립 평가가 별도입니다.

계약과 실행 증적: search_artifacts/titantpp_time_head_bias_refit_5080_20261006_v1.
