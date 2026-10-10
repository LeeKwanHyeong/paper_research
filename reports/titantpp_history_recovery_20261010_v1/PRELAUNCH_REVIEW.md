# Recovery prelaunch review

## Review Result: PASS

상태: 로컬 코드·생성 제어 파일 검토 완료. 발견한 HIGH 3개는 수정 후 재검증했다. 이 판정은 native GPU admission이나 실제 학습 완료를 뜻하지 않는다. 원격 호출 0, source 수정 0. `.agents/` 출력 요구와 수정 금지가 충돌하여 부모가 지정한 이 reports 파일만 작성했다. 지정 execution-protocol 파일은 로컬에 없었다.

### CRITICAL

없음.

### HIGH

미해결 없음.

### MEDIUM

없음.

### LOW

없음.

## 수정 확인

1. `paper/scripts/prepare_titantpp_history_recovery.py:90` — 부모 input receipt의 두 `resolved_inputs` 경로를 parent source root→새 source root로 정확히 매핑한다. 원본 receipt는 보존하며 input SHA·통계·populations 등 나머지 모든 필드는 strict equality로 검증한다. 기존 full equality가 새 root에서 실패하는 것을 실제 RAF receipt로 재현했고 정상 이동/변조 거부 테스트를 통과했다.
2. `paper/scripts/prepare_titantpp_history_recovery.py:105` — 새 qualification의 interface SHA를 먼저 확인한 뒤, resumed fit에서만 `execution_contract_sha256`를 parent digest로 bridge한다. 나머지 interface 필드는 원본 checkpoint와 정확히 일치해야 한다. bridge된 동일 interface가 native resume 검사, train_one, selected/last replay identity로 전달된다. 별도 bridge receipt에 양 계약 SHA를 기록하고 parent .pt bytes는 변경하지 않는다. parent digest/science 필드 변조 거부 및 fresh bypass 테스트를 통과했다.
3. `paper/scripts/prepare_titantpp_history_recovery_control.py:116` — 최종 원본과 부모 원본 SHA 재검증→해당 Pod 부재→전용 volume identity 확인→정확 ID 삭제→volume 부재 확인 뒤에만 cleanup 완료를 기록한다. Pod 생성 전 volume 비용도 누적 비용에 포함한다. mock 실행에서 순서/대상 ID 및 원본 변조 시 원격 삭제 차단을 확인했다. 실제 volume 삭제를 수행하지 않았다.

## Validation

- `/usr/local/bin/python3 -m unittest paper.tests.test_titantpp_history_recovery_operations -v`: **15/15 PASS**. 종료 경합·foreign process 차단·exact argv·2+3+3+3+3+3 잔여 batch·부모 manifest 구분·exposure/timing prefix·optimizer step·입력 root bridge·interface bridge 포함.
- `/usr/local/bin/python3 paper/scripts/prepare_titantpp_history_recovery_control.py --self-test`: **생성 control 11개, sealer 시나리오 8개 및 volume cleanup 시나리오 4개 PASS**, 원격 0. 부모 과금 누락·추가충전·기간 연장·GPU/Python/stale Runtime·원본 archive 변조 차단.
- 추가 stdlib mock: `delete_preserved_volume` 정확한 전용 ID/원본 검증/Pod 부재/삭제 후 부재/receipt 순서 PASS; current 또는 parent 원본 검증 실패 시 원격 호출 0 확인.
- frozen trainer `training.py:790`에서 `last_path.exists() and not force_rerun`을 통해 모델·optimizer·history·embedded best·Python/NumPy/torch/CUDA RNG·train loader generator를 복원하고 saved epoch+1에서 재개한다. 별도 args.resume 플래그는 필요하지 않다. base args `force_rerun=False` 및 원래 execution_role/source revision 유지 확인.
- 재개 prefix epoch, history, timing, exposure, optimizer step 검증이 실제 run_fit의 trainer 호출 전에 수행된다. 새 ETA에는 재개 후 epoch만 포함된다.
- GPU 경합 처리는 종료 중 cmdline/zombie를 bounded fresh inventory로 재검사한다. 살아 있는 foreign process와 전체 worker argv 불일치는 여전히 차단한다.
- 완료10은 부모 contract/terminal SHA로 재사용하고, 새17 terminal은 recovery contract로 검증한다. 새 archive는 부모 archive/manifest/원본을 포함하고 검증한다.
- all-in rate·parent bill·별도 prior storage·reserve5·총 cap50·절대 deadline 연결을 확인했다. 동일 GPU/무중단 bitwise 재현은 주장하지 않는다.

## 범위와 아직 남은 확인

실제 새 A100의 frozen Runtime, 두 .pt의 target-native `_validate_resume_payload` 통과, 신규 permit/package SHA 봉인, manager/service freshness, 첫 E24/E47 저장은 parent의 원격 admission 및 launch 확인 대상이다. 로컬 단위 테스트만으로 이 항목을 완료로 보지 않는다. Test/heldout 성과·예측, Val/Test 혼합 파일은 열지 않았다.

## Files changed

본 리뷰 파일만 작성. source/control/tests는 담당 구현 agent가 변경했다.

## Acceptance criteria

- [x] 운영 준비·소유권·native resume 경로 검토
- [x] 생성 control/sealer/cost/retrieval 검토
- [x] 집중 자동 테스트 실행 및 실제 실패 조건 재현
- [x] 발견한 HIGH 3개 수정 후 재검증
- [x] 원본·Test 경계 유지
- [ ] target Runtime 두 checkpoint admission
- [x] 실제 bundle 309파일 읽기 전용 sealer 검증
- [ ] 봉인 발행 및 manager/service 확인
- [ ] 첫 신규 저장 epoch 검증

## Remaining order

1. target native Runtime·checkpoint admission 및 seal 검증 — 다음 작업(parent).
2. 예산 guard/service freshness 확인 후 남은17 실행 시작 — 후속.
3. 첫 신규 저장 checkpoint 및 원본 prefix 연속성 확인 — 후속.
4. terminal+원본 회수+Validation 보고+전용 Pod/volume 부재 검증 — 후속.

## Actual bundle verification

- 실제 bundle canonical `da077e96e61255286c5cb35b54dc09ec8144e2d2f410e6eb46fea4681ee17899` 확인.
- operation closure `8d7a5bec2af2f7d746fdb38ff618a8aceda7ffff9391d9252f4e5b46cf70f8c0` 확인.
- 실제 campaign.py 및 control 11개가 검토한 생성기 출력과 byte-identical.
- 실제 `control/seal_launch.py` 기본 읽기검증 결과 `validated_not_sealed`, 309 files, remote_calls 0, writes 0.
- 부모실제청구 $30.124985997681506 + 선행volume $0.0009722222222222222 + 신규 all-in $1.79375/h, 회수예비 $5, 총 $50.
- 이 리뷰에서는 --seal, deploy, 원격 GPU/Pod API를 실행하지 않았다.

## Reviewed bytes

- `paper/scripts/prepare_titantpp_history_recovery.py`: `6e0366b662b7273670042d42dce27b98905d30c0d786e86e5e4bf5a25ccc3345`
- `paper/scripts/titantpp_history_recovery_operations.py`: `8df97122a42380c9129caa6eb3aad5c930fc2233fccf2bc8132a599958253ec5`
- `paper/scripts/prepare_titantpp_history_recovery_control.py`: `a620f54040be91bda39af85f2c4d316de18f9745254c42e07d65050b8ddcbd5a`
- `paper/tests/test_titantpp_history_recovery_operations.py`: `473d8847aa0b9c55a397a96d3aa52034d9891fad154c3d2e181de0439f824bb5`
