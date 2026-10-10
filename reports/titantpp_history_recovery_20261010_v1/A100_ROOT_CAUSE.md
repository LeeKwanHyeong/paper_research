# A100 이력 보정 운영 오류와 복구 준비 — 2026-10-10

상태: 로컬 수정·회귀 검증 완료. 새 GPU의 qualification, 3 worker admission, 실제 재개는 부모 세션 담당이며 이 문서 작성 시점에는 미실행.

## 확인된 사실과 원인 한계

- 부모 봉인 operation `campaign.py` rawSHA: `b4fddd5f50da00262d261beed8931bb564c95bde19e387cb6dcd3327f3b73ecd`.
- 실패 원본 `originals/partial/verified/originals/failure.json`은 2026-10-09 12:14:47 KST에 `Foreign GPU process group`을 기록했다. `check_active_gpu`가 `verify_gpu_membership`의 `pgrp == timeout PID and '--job' in argv` 조건을 통과하지 못했다.
- 같은 종료 증적에서 RAF seed42 보정 없음 worker는 exit0, full/current_only worker는 -15이다. 따라서 수치 NaN이나 비용 한도 종료로 분류하지 않는다.
- 원래 코드는 cmdline을 먼저 읽고 stat을 나중에 읽는다. 정상 worker가 종료되어 cmdline이 빈 문자열이 되는 경우에도 `Foreign GPU process group`으로 분류한다. 원래 함수를 AST로 추출한 회귀 테스트에서 이 오판 경로를 재현했다.
- 오류 순간의 PID/PPID/pgrp/argv snapshot이 보존되지 않아 **이번 사고가 실제 외부 프로세스였는지, 종료 경합이었는지는 확정할 수 없다**. 정상 종료 worker와 시간상 겹치는 종료 경합은 유력 가설이다. 코드상 재현 가능한 결함과 사고의 확정 원인은 구분한다.

## 최소 운영 수정

신규 `paper/scripts/titantpp_history_recovery_operations.py`:

1. stat → cmdline → stat으로 프로세스 start ticks와 ancestry 변화를 확인한다.
2. 사라진 PID·zombie·빈 cmdline은 최대 3회, 간격 0.1초의 제한된 관측 재확인을 수행한다. 학습 retry는 아니다.
3. 최신 GPU 목록에서 실제로 사라진 PID만 제외한다. 계속 존재하는 불명확 PID는 실패 처리한다.
4. 원래 pgrp/PPID/job/script/memory 검사를 유지하며 contract·host·mode·deadline을 포함한 전체 argv를 추가 대조한다.
5. 오류에 관측 snapshot을 포함해 같은 상황에서 검사값이 소실되지 않게 한다.

기존 봉인 운영/과학 파일은 변경하지 않았다. 새 복구 bundle만 이 helper를 사용한다. 유사 패턴은 `paper/scripts/prepare_titantpp_history_a100_parallel3.py`에도 있지만 기존 원본을 생성한 경로이므로 변경하지 않고 새 preparation으로 대체한다.

## checkpoint 실물 검사

부모 archive/manifest에 고정된 Validation-only 원본을 CPU에서 확인했다. 재추론은 하지 않았다.

| RAF seed42 | 저장 / best | optimizer 누적 step | 재개 시작 |
|---|---:|---:|---:|
| full | 23 / 9 | 4,646 | 24 |
| current_only | 46 / 9 | 9,292 | 47 |

두 실행 모두 다음을 확인했다.

- last checkpoint rawSHA와 server checkpoint receipt의 history/exposure/timing/train.log SHA 일치.
- frozen digest 함수로 current model와 best model tensor SHA 일치.
- schema2, Python/NumPy/Torch/CUDA RNG, train loader generator 상태, optimizer state 존재.
- history와 checkpoint history 일치, 최초 strict minimum best9 일치.
- exposure/timing은 각각 23·46 completed epoch와 일치하며 optimizer step이 exposure batch 합과 일치.

기존 trainer는 resume를 지원하지만 frozen `shared.audited_training`은 빈 exposure/timing으로 시작한다. 따라서 그대로 실행하면 E24/E47 저장 때 누적 epoch/optimizer 검사가 실패한다. 새 operation context는 검증된 prefix 기록을 이어 붙이고 원래 loader/save/optimizer 검사를 유지한다.

새 물리 A100으로 이관하는 것이므로 같은 GPU UUID나 완전한 무중단 수치 동등성으로 설명하지 않는다. Runtime/source/초기화/loss/optimizer/selector는 원래 계약을 유지하고 새 native qualification을 요구한다. 시간은 물리 GPU 구간별로 보존하며 복구 elapsed_seconds는 복구 구간 시간이다.

## 복구 준비물과 실행 경계

`paper/scripts/prepare_titantpp_history_recovery.py`는 실제 Pod ID, GPU UUID, 청구 시작, GPU 및 저장 포함 요율, 부모 실제 청구액을 필수 CLI 인자로 받는다. 부모 archive/393개 원본과 source129/input7 SHA를 검증한 뒤 새 bundle을 exclusive 생성한다.

- 원래 execution ID27개 유지: 완료10 재사용, 부분2 재개, 미시작15 fresh.
- 완료 원본은 `recovery_parent/run/<id>`에서 보존하며 새 `run/`에 완료 위장 복사하지 않는다.
- 첫 batch는 미완료 RAF42 2개, 이후 같은 dataset/seed 3개씩 동시 실행한다.
- 부모 청구액 + 신규 저장 포함 요율 + 회수 reserve5 ≤ $50로 종료시각을 계산한다.
- source129/input7 원본 bytes 유지. 새 operation closure만 생성한다.
- 부모 오류·claim·원본 보존, 자동 retry/resume 반복 없음.
- start permit의 lease 시작은 billing_started_at_unix로 고정하고 실제 발행 시각을 별도 기록한다.
- deployment/qualification/학습 실행은 부모 세션이 새 native Runtime·GPU·프로세스 검증 뒤 수행한다.

## 검증과 변경 파일

명령: `/usr/local/bin/python3 -m unittest paper.tests.test_titantpp_history_recovery_operations -v`

15개 테스트 PASS: 원래 오류 재현, 종료 PID 재관측, persistent 불명확 PID 거부, 실제 foreign group 거부, 신규 GPU PID 검증, zombie/stat 검사, exact argv 및 오류 증적, 10+2+15 실행 집계, 부모 원본 분리, resume exposure/timing prefix 및 누적 optimizer step 검증.

변경 파일:
- `paper/scripts/titantpp_history_recovery_operations.py`
- `paper/scripts/prepare_titantpp_history_recovery.py`
- `paper/tests/test_titantpp_history_recovery_operations.py`
- 이 보고서

완료 조건:
- [x] 봉인 원본·Test 경계 보존
- [x] 재현 가능한 오판 경로 테스트
- [x] 최소 운영 수정과 종료 경합 관측 증적
- [x] 두 resume checkpoint continuity와 SHA 검증
- [x] 새 preparation과 노출·시간 prefix 회귀 테스트
- [ ] 새 물리 GPU/Runtime native qualification·admission
- [ ] 실제 E24/E47 checkpoint 저장과 나머지 학습 완료

프로젝트 debug-investigator 역할 설정은 `.codex/agents/debug-investigator.toml`의 gpt-6-astra/high다. 도구가 제공하는 런타임 모델 identity 별도 조회 수단은 없으므로 설정값을 실제 런타임 측정처럼 주장하지 않는다. 요청된 실행 프로토콜 `.agents/skills/_shared/runtime/execution-protocols/codex.md`는 존재하지 않으며 `.agents/` 수정 금지에 따라 허용된 reports 경로에 기록했다.


## Prelaunch 검토에서 확인한 재개 연결 수정

상태: 로컬 수정·회귀 검증 완료. 새 native 실행 여부는 부모 세션에서 별도로 확인한다.

- 부모 input receipt와 새 native receipt의 전체 equality는 절대 경로가 달라 실패했다. 승인된 `resolved_inputs`의 두 파일만 부모·신규 source_root 사이에서 같은 상대 경로로 연결한다. 통계·데이터 SHA·모집단 등 나머지 필드는 완전히 같아야 하며, 부모 receipt bytes를 유지하고 `input_root_bridge.json`을 별도로 남긴다.
- 부모 체크포인트 identity에 포함된 `interface_meta.execution_contract_sha256`는 새 운영 계약과 다르다. 새 native qualification 검증을 먼저 완료한 뒤, 재개 2개에 한하여 이 필드만 정확한 부모 계약 digest로 연결한다. checkpoint rawSHA·부모 interface/resume identity·나머지 interface 전체 일치를 검증한다. frozen resume 검증, train_one, selected/last endpoint 검증이 모두 이 동일한 interface를 사용하며 `interface_contract_bridge.json`에 두 digest를 남긴다. fresh15와 qualification의 identity는 새 계약 그대로다.
- 부모 checkpoint `.pt`와 frozen science source 수정은 없다. endpoint replay의 기존 전체 identity equality도 유지한다.
- 실제 RAF parent receipt로 기존 실패를 재현하고 경로 연결 성공 및 통계/identity/모집단/경로 변조 거부를 검증했다. interface 과학 필드/부모 digest 변조 거부, fresh 미적용, binary SHA와 원본 bytes 불변도 검증했다.
- 예산은 부모 실제 청구와 신규 Pod 이전 network storage 상한을 별도 필드로 보존한다. `--prior-network-storage-usd`는 필수이며 1시간 보수 상한 0.0009722222222222222달러 이상이어야 한다. 학습·절대 종료시각 모두 이를 차감하고, 운영 cap 검사에서도 합산한다. 경계값 $50 통과·초과 거부 테스트를 추가했다.

완료 조건 추가:
- [x] 승인된 두 input 경로 외 변경 거부
- [x] 재개 두 조건의 운영 digest bridge와 과학 identity 전체 검증
- [x] fresh 조건 영향 없음·부모 원본 bytes 보존
- [x] 선행 network storage 포함 총 $50 cap 회귀 검증
