# 최종 FFN RAF6 운영 구현 — 로컬 검증 완료, 실제 실행 전

**5090에 새 후보 두 구조를 안전하게 실행한다 — 구현 완료**
- 신규 운영 파일은 prepare/deploy/advance/runtime 네 개다. 기존 qk_location 운영 도구를 새 파일로 복제한 뒤 FFN 계약으로 적응했으며 기존 파일을 수정하지 않았다.
- 기존 frozen source145와 입력을 부모 bundle에서 동일 바이트로 복사하고 신규 모델·runner 두 개만 더해 source147을 만든다. 현재 작업트리의 옛 source를 대신 복사하지 않는다.
- 기존 CNN3는 저장 결과 참조만이다. 새 학습은 Current42/52/62 전부 종료 뒤 Temporal42/52/62를 시작하며 최대 동시3이다.
- 기존 5090 GPU UUID, Python/torch/cuDNN/threads, 조건36h와 절대 host deadline1792679647.915156을 그대로 사용한다.

**실제 실행 전에 고정 근거와 합성검증을 확인한다 — 로컬 완료**
- `SCIENCE_SPEC.json` 및 DESIGN_MANIFEST의9개 설계 문서, EVIDENCE/VERIFIED_INPUTS, 부모 source/계약/종료 원본과 기존 CNN 참조 SHA를 쓰기/원격보다 먼저 확인한다.
- 운영 합성96개가 실제exit0로 통과했다. 실제원격0, 실데이터 학습0, Test0이다. 성공/실패 stdout·stderr를 `ops_tests/`에 별도로 보존한다.
- native synthetic 함수의 실제 모델·head·loss·gradient 경로는 부모 frozen147 overlay에서 CPU 합성으로 검증했다. CUDA 할당/장치 이동과 Linux start_ticks만 로컬 fixture로 대체했다. 이것은 GPU/Runtime/동시입장 검증이 아니다.
- RAF context84에서 두 후보 각각 active80579/registered105311, 기존 Q/K/V filter576과 FFN신규1152를 확인했다. c0 첫단계 a/b task-gradient0, 이후9 parameter×tap 그룹 gradient 활성, Q/K/V 공통투영 gradient, target 차단과 eval state 불변을 확인했다.
- 최초 운영 fixture1건은 science 파일 변조를검사하려다 새 design manifest 부재를먼저받았다. SCIENCE_SPEC를 첫검증으로 이동한 뒤 통과했다. CPU 통합 최초1건은 macOS에 /proc이 없어 신원fixture가 없었던 실패이며 새 fixture에서만 수정했다. 실제 실험 실패로 분류하지 않는다.

**부모가 독립검토와 커밋 후 실행한다 — 다음 작업**
- `LOCAL_INDEPENDENT_QA.json` status=PASS 및 `MODEL_SCIENCE_REVIEW.json` status=PASS_static_science_alignment가 필요하다. 이 문서는 독립검토를 대신하지 않는다.
- 모델/runner/운영 source를 커밋한 뒤에만 `paper.scripts.prepare_titantpp_post_attention_ffn_raf6 --mode prepare`를 실행한다. prepare는 새 bundle이 존재하면 거부한다.
- 이어 같은 모듈 `--mode seal --tests-receipt <OPS_TESTS.json 실제 경로> --model-tests-receipt <MODEL_TESTS.json 실제 경로>`를 실행한다. 성공 receipt actual_exit/failed/passed, 필수 source·stdout·stderr SHA까지 재검증하며 실제 반환 integrity SHA만 사용한다.
- `paper.scripts.deploy_titantpp_post_attention_ffn_raf6 --integrity-sha <실제 반환값>`이 새5090 root를단1회 설치한다. fresh GPU 유휴, native Runtime·입력·두 후보 합성 및3동시 입장검증 뒤에만 학습permit을 발급한다.
- `paper.scripts.advance_titantpp_post_attention_ffn_raf6 --integrity-sha <실제 반환값>`은 회차당1회 관측하며 모든6 worker·dispatcher·pipeline 실제exit0·terminal binary SHA·명시적owner/GPU부재 뒤만 회수한다. 완료cache는 원시각을 유지하고 모든 원본/종료/선택/집계를 로컬 재검증한다.
- 학습 종료 후 Test 도구는 별도 신규경로에서 준비하고 전6 native Validation gate 뒤에만 진행한다. 이 운영도구는 Test를 실행하지 않는다.

**실행하지 않은 범위를 구분한다 — 현재 미실행**
- 실제 prepare/seal/deploy/SSH/native qualification/학습/회수/커밋은 이 구현 worker가 실행하지 않았다.
- wrapper 자신 OS exit는 관측하지 않았다는 계약을 유지한다. 학습·관측 실패와 조회 실패를 구분하며 미확정 재시도·claim/failure 삭제·기한 연장은 없다.
