# Key-only 과거 두 tap 고정 checkpoint 진단

현재 상태: 명세 고정, 구현·독립 검증·native 진단 대기. 학습0, Test0. 기존 CNN 모델/checkpoint/결과의 변경·재학습·재평가0. Query-only와 이전 후보 추론0.

프로젝트 `/Users/igwanhyeong/PycharmProjects/paper_research`의 TEST_SESSION_PROTOCOL.md0·1·9·12, 이 캠페인의 SCIENCE_SPEC.json(rawSHA `f15ff87dc3f643e86e99d61c5cb6f45eec61d774879edcbab295f03bfc6f535c`), NEXT_MECHANISM_ACCEPTANCE.json(rawSHA `b99147f6ac5cfd6ac02c66d1636848f3abdc93452e2d21e217d80ab3c54bfff9`), EXPERIMENT_PLAN.md 및 실제 현재 receipt를 먼저 읽는다. 기존 사용자 원인 분석 사전승인 범위의 새 격리 진단이며 새 학습이 아니다.

현재 Key-only seed42/52/62 epoch9/5/26만 사용한다. source145 closure `fc0668016b8eb89752e03d0751e317e77b06d8135c0b82f8b43625e345abf9a3`, 부모 학습 receipt `ac680e7665de247681a9e223b0f5e635217f342461104c31c10221e2e6c41342`, 평가 receipt `7325da1da450bbee0dc1c4bb5c50df1130a07d9988bbdae9392993e071ba992c`와 모든 선택 checkpoint/입력/metadata SHA를 검증한다. 기존 원본·과학source·실패·claim을 수정하지 않는다.

도구 준비 중이면 새 operation 파일의 구현·합성 검사·독립 검토를 끝낸다. 검증된 LOCAL_READY와 실제 prepare 반환의 진입점·integrity SHA만 사용하고 명령이나 SHA를 추측하지 않는다. 이 문서는 미검증 실행 명령을 제공하지 않는다. 명세 변경 또는 검증 실패를 자동 수정·재봉인으로 넘기지 않는다.

원격 진단 경로는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_key_history_probe_20261016_v1_5090`다. 같은5090 GPU `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`, Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads/interop1을 유지한다. 현재 유휴·소유 상태와 native Runtime/GPU/입력·합성3동시 qualification을 검증한 뒤 exclusive attempt로 한 번만 진행한다. 상속 절대 host deadline1792679647.915156은 연장하지 않는다. 승인 범위의 새 경로 쓰기·읽기와 회수 sandbox 승격은 재질문하지 않는다.

on은 원형 Key3tap, off는 현재tap 유지·과거2tap만 임시 사본에서 차단이다. 원함수 검증과 경계·finally 복구·모든 parameter/buffer/context 불변을 지킨다. 전체K 잔차 제거, 배율 탐색, 별도 과거tap 분리, 재선택, optimizer 및 학습은 금지한다.

원형 Validation3의 여섯 집계 지표와6690개 native 모집단 재현 gate가 모두 통과한 뒤 원형 Train3(25779개/seed), 차단 Train3, 차단 Validation3 순서로 실행한다. fresh worker 최대3병렬, 실제 데이터12개+합성3개다. 이전 개별 원형 행은 없으므로 새 원형 행을 참조한다. 신규 on/off target/truth/관측 이력은 정확히 결속하며 native parent-SHA/float32 digest와 pure-input-SHA/float64 exporter digest를 혼동하지 않는다.

시작된 경로를 재설치·재시작하지 않는다. livehost 관측은 회차당1회, 별도observer/반복polling0. 검증 terminal cache는SSH0·원관측시각 유지다. 조회 실패와 실제 worker 실패를 구분하고 미확정을 완료로 바꾸지 않는다. 오류·timeout·미완료 회수·기존attempt/claim/failure는 원문과 실제 transport exit를 보존하고 원격 자동 재시도를 반복하지 않는다. 새 진단 소유 process group에만 고정시간제한을 적용하며 다른 학습은 건드리지 않는다.

전12 native worker 및 단계pipeline actualexit0·명시적 소유PID/GPU 부재·원본회수 archive/manifest/선택/결과 모든 SHA가 검증된 뒤 저장 행 분석을 수행한다. supervisor wrapper 자체OSexit는 직접 관측한 경우만 기록한다. 고정15조건·Train고정 cut·공통개체 가중·모든 seed와 RMSE/MAE/큰수량/시간 손해를 남기고 새 학습 자동규칙을 만들지 않는다.

결과는 현재 공동학습 가중치의 과거K 경로 민감도로만 해석한다. CNN/Query/외부/Test 재평가, 원고 수정, 다른 자동화 변경, 삭제된 A100 접근0. 첫 진단 완료를 backbone 개선 성공으로 쓰지 않는다. 결과·판정·한계·다음 근거를 기존 승인된 Notion 루프 기록에 추가할 책임은 상위 운영 작업에 있으며 이 문서 작성은 실제 기록 완료를 의미하지 않는다. 정상 대기는 조용히 유지하고 시작·단계전환·완료·실패·사용자 조치만 알린다.
