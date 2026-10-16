# KeyOnly 과거 내용 대체 진단 후속 지침

프로젝트 `/Users/igwanhyeong/PycharmProjects/paper_research`의 TEST_SESSION_PROTOCOL.md 0·1·9·12와 이 경로의 SCIENCE_SPEC.json, NEXT_MECHANISM_ACCEPTANCE.json, EXPERIMENT_PLAN.md를 먼저 읽는다. 명세 SHA `f8ce822618bc0dcbacda3512215a9e368afb054a26466616cca1089b12d18678`, 검증 조건 SHA `6520d428c1cd3c4ac4f211c2c4e89a6b5b570d4efe22cf4c64ff73a948771d2e`를 검증한다. 사용자 사전승인 범위에서 새 진단 구현·합성 검증·독립검토·기록을 진행한다. 현재 이 문서 작성 시점에는 구현 봉인·실제 native 실행은 미완료다. 진입점과 integrity SHA를 추측하지 말고 실제 준비 반환값만 사용한다.

대상은 기존 KeyOnly3 seed42/52/62 선택epoch9/5/26, source145 closure `fc0668016b8eb89752e03d0751e317e77b06d8135c0b82f8b43625e345abf9a3`다. 부모 원본 receipt `42951c5ffb69639a438a002f0dd9e70eee1f3d8c234ad87b2ad42861c5041ff7`, 직전 집계 `953a87fa46e0d70b9c624d01a91df33c052c3a3158eecf8f1473c861c11cb2d7`, 집계 receipt `d90f3c67217fbe6eb414e1521fa6e63491f0161a302eb8227cf5b934c58a47eb`를 검증한다. 기존 부모 학습·평가·진단 helper를 재실행하지 않는다. 기존 CNN·QueryOnly·이전 후보 추론0, Test0, 새 학습0, optimizer0, checkpoint 재선택0이다.

on은 원형이다. off는 각 event 위치에서 실제로 존재하는 과거 K source만 현재 K로 대체하며 계수·가용 mask·원래 누적 순서를 유지한다. 원래 과거 내용과 kernel의 shape/index/nonfinite 검사를 먼저 실행한다. 임시 clone만 사용하고 원래 parameter/buffer/context/source/available 불변과 finally 함수 복원을 확인한다. 원본 w0는 불변이지만 off의 유효 현재계수는 w0+I1*w1+I2*w2로 달라진다. 실제 K 출력 norm·방향·attention/memory 배분은 보존하지 않는다. 과거 kernel을0으로 하는 직전 진단과 혼동하지 않는다. 계수합 보존은 잠재 상수 응답이며 실수량 예측 수준 보장을 뜻하지 않는다.

승인된 새 bundle은 `search_artifacts/titantpp_backbone_key_content_probe_20261016_v1`, 새 원격 root는 `/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/titantpp_backbone_key_content_probe_20261016_v1_5090`다. 같은5090 GPU `GPU-c9adc246-ef66-7906-bc07-6e152fa9952f`, Python3.12.13/torch2.11.0+cu130/cuDNN92000/4threads와 상속 절대 deadline1792679647.915156을 유지한다. fresh 소유/GPU유휴 검증과 모든 stdlib SHA/승인/native qualification 뒤 exclusive attempt로 한 번만 시작한다. 완료 과거 관측은 현재 유휴 증명이 아니다.

단계는 qualification3 → 새 onValidation3 → offTrain3 → offValidation3이다. 각 단계 최대3병렬이며 모든 worker는 freshprocess다. 새 onValidation3은 기존 onValidation 개별 예측·시간손실·여섯지표·exact target/truth/history를 rtol=atol=1e-5로 재현해야 한다. 기존 onTrain3은 원본 receipt/manifest/shards와 선택·source·input·state·Runtime·대상 검증 후 재사용하며 새 onTrain 실행0이다. condition/stage 정확일치는 새 onValidation 대 부모 onValidation에 적용한다. off/on은 각자 역할·split 일치를 검증하며 서로 같은 condition/stage를 요구하지 않는다. onValidation3 전체 gate와 재사용 onTrain3 검증 전 off 실행0이다. 새실데이터9+qualification3=12worker, 새pipeline1이다.

진행을 관측할 때 livehost는 회차당 한 번, 검증 완료cache는SSH0 및 원관측시각 유지다. PID/argv/startticks/lease/GPU 신원을 구분하고 조회 실패를 과학 실패로 바꾸지 않는다. 미완료/실패/timeout/기존attempt는 원문 stdout/stderr·크기·SHA·실제transport/worker/pipeline exit와 함께 보존하며 미확인 재시도/재시작/resume/kill/claim삭제/원본 덮어쓰기0이다. 공용Runtime/driver/auth·원고·다른 자동화·삭제된A100은 건드리지 않는다.

모든 신규worker12 실제exit0·pipeline실제exit0·명시적 ownerPID/GPU부재 뒤 원본 archive/manifest를 새 경로에 exclusive 보존하고 모든SHA를 검증한다. supervisor 자신의 OSexit와 부모가 직접 본 childexit는 구분한다. 최종 receipt에는 신규data9/qualification3/총native12/새onTrain0/재사용onTrain3/Val재현3/off6/seed3과 원본 lineage를 결속한다.

집계는 새 offTrain과 기존 onTrain, 새 offValidation과 재현된 새 onValidation만 짝짓는다. 고정15조건·TrainQ25/Q75·두 가중, 전체/큰수량q>200 RMSE·MAE와 recorded positive-integer TimeNLL, 모든seed차이·악화·평균±표본SD(ddof1)·가산검산·지지/제외를 보고한다. 기존 CNN/Query/Test 행을 추가로 읽지 않는다.

이 진단이 끝나면 같은 DC/삭제 변형을 계속 반복하지 않는다. 수준 배율·과거 내용 민감도의 근거와 한계를 정리하여 중복 없는 단일 CNN 가설/대조를 검토한다. 새 모델 성공·일반화·원인 확정·통계 유의성을 주장하지 않으며 근거 없는 학습 자동진입0이다. 완료 기록을 Notion 기존 Backbone Enhancement History에 누적하는 일은 부모 작업에서 수행한다. 정상 진행·대기·변화없음은 조용히 유지하고 시작·주요전환·완료·실패·사용자조치만 알린다. 이 문서는 다른 자동화를 수정하거나 기존 작업을 대신 실행할 권한을 추가하지 않는다.
