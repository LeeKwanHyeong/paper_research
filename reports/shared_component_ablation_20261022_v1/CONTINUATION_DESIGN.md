# A–D 비교의 짧은 확인·회수 제어

## 현재 학습을 식별해 이어받는다 — 구현 완료, 독립 검증 대기
- `advance_shared_component_ablation.py`는 새 학습을 시작하지 않는다. 사용자가 승인하고 root가 이미 시작한 두 `fit_validation` dispatcher와 parent의 PID·start_ticks·pgid·argv·cwd를 고정한다.
- 제어 파일은 이 보고서 아래 `continuation_v1/`에 둔다. `AUTHORIZATION.json`, `CONTROL_BINDINGS.json`, `LAUNCHED.json`은 실제 시작 증적을 확인한 root가 생성하며 이 구현 작업에서는 만들지 않았다.
- 파일 SHA 검증은 원격 호출보다 먼저 수행한다. 고정 대상은 제어·비교 코드, 계약 원문·integrity·배포 archive, source987, input9, 기존 A 사본36개, 과학 계약, 원래 두 학습의 시작 신원이다.
- MAC, 기존 외부 5seed, 기존 결과와 checkpoint, 정지된 선행 프로세스는 수정하지 않는다.

## 15분마다 짧게 확인한다 — 구현 완료
- `--mode advance --binding-sha RAW --authorization-sha RAW`를 프로젝트 root에서 한 번 호출한다.
- 정상 대기 확인은 host당 읽기 전용 SSH 1회, 최대45초다. 900초 미경과와 중복 실행은 SSH0이다. 장시간 SSH 대기나 별도 background finisher를 만들지 않는다.
- 매 회차의 stdout/stderr 실제 바이트·SHA·실제 OS 반환 값·timeout을 배타 경로에 보존한다. 조회 실패를 학습 실패로 바꾸지 않는다.
- 첫 transport 실패, 과학 실패, 신원·boot·GPU 소유권·기한 변경, 이미 존재하는 partial 작업이면 보존 후 차단한다. 이후 원격 호출0이며 자동 재접속·학습/평가/회수 retry·claim 삭제는 없다.
- read-only 확인은 원격 계약·source/input/legacy SHA, 현재 신원, 정지 프로세스 보존과 GPU 소유권을 확인한다. 실제 dispatcher exit0와 단계별 실제 child exit0가 있어야 회수를 허용한다.

## 종료 증적을 한 번 회수한다 — 구현 완료
- 단계가 끝나면 기존 원격 manager `--mode pack`을 한 번 실행한다. pack/전송은 45초 종료 확인과 구분하며 최대900초다.
- `BUNDLE/retrieval/{phase}/{host}/`에 pack.stdout/pack.stderr, transport.json, collection_receipt.json, collection archive, transfer.stdout/transfer.stderr/transfer.json을 보존한다.
- 전송 archive의 실제 크기·SHA, manifest SHA, 원격 source/input/legacy 확인 증적, 모든 파일 byte·SHA를 대조한다. 중복·절대/상위 경로·symlink/hardlink·특수 파일·누락·빈 과학 파일은 거부한다.
- `originals/`는 배타 생성하며 manager의 실제 단계 종료 검증을 재사용한다. 검증 완료 표식이 없는 partial 회수 경로는 재사용하거나 덮어쓰지 않는다.

## Validation36 gate 뒤에만 Test와 비용 측정을 시작한다 — 구현 완료
- 양 서버 fit/Validation 회수가 검증된 뒤 36개 각 모델의 원 train/reuse admission exit와 새 Validation receipt/exit를 byte 그대로 새 gate에 복사한다. A 재사용의 admission exit는 `reuse`, 새 학습은 `train`으로 구분한다.
- 144개 원 JSON과 모든 개별 Validation 재현을 `manager.verify_global_gate`로 검증한다. 같은 gate 바이트를 stdin으로 양 서버 새 root에 배타 전송하므로 큰 gate를 shell 인자에 넣지 않는다.
- 원래 source·기존 native qualification·fit/Validation의 실제 exit0·GPU·global gate를 재검증한 뒤에만 `test_cost`를 pinned 환경의 tmux에서 한 번 시작한다. 새 dispatcher/parent 실제 신원을 기록한다. test_cost 기존 claim/intent/partial이 있으면 재시작하지 않는다.
- Test36과 비용216 worker가 끝나면 같은 회수 검증을 거친다. 독립 comparator QA가 현재 비교 코드 SHA에 묶여 통과한 뒤에만 local-only check-only → 별도 보고 생성 → check-only로 마무리한다.
- `COLLECTION_COMPLETE`와 비교까지 끝난 `COMPLETE`를 구분한다. 완료 cache는 SSH0으로 모든 파일 inventory/SHA·종료·global gate·순수 집계 재계산을 다시 검증한다.

## 제어 파일 형식 — root가 실제 증적으로 생성
- `AUTHORIZATION.json`: `status=approved`, `campaign`, `matched_component_3seed`, `initial_fit_adoption_only`, `short_terminal_checks`, `collect_once`, `global_validation_then_test_cost`, `local_comparison`, `preserve_existing`, `mac_deferred`, `no_scientific_retry`가 모두 true. `maximum_check_seconds=45`, `minimum_check_interval_seconds=900`, `transfer_timeout_seconds=900`.
- `CONTROL_BINDINGS.json`: `campaign`, `report_root`, `bundle_root`, `authorization_sha256`, 원문 `contract_sha256`, `integrity_sha256`, `science_spec_sha256`, `initial_launch_sha256`, `files`(최소 local controller와 comparator 실제 SHA), `comparator_qa_sha256`(검토 대기면 null).
- `LAUNCHED.json`: canonical `contract_sha256`, `hosts.5080/5090.phase=fit_validation`, `identity`(claim의 실제 dispatcher), `supervisor_identity`(supervisor intent의 실제 parent). 두 identity에는 pid/start_ticks/pgid/argv/cwd를 포함한다.
- comparator QA: `reports/shared_component_ablation_20261022_v1/COMPARATOR_QA.json`, `status=PASS`, `files_reviewed_sha256['paper/scripts/compare_shared_component_ablation.py']`가 실제 SHA와 같아야 한다.

## 검증 범위 — 로컬 합성 증적
- 집중 검증은 36개 통과했다. 실제 36개 gate(원 JSON144개), 재사용 admission 단계 구분, 실제 tar 생성/추출/재해시, 손상된 receipt/pack byte/전송 반환 값/manifest/파일, symlink/hardlink/디렉터리 tar member 거부, cooldown/busy/실패 후 SSH0를 검증했다. 추출물은 그대로 둔 채 archive 내부 원 checkpoint 바이트만 같은 길이로 바꾸고 바깥 receipt SHA를 갱신한 경우도 cache 검증이 거부한다.
- 생성되는 read-only probe와 Test-cost tmux 시작 코드도 로컬 합성 경로에서 실행했다. 완료·대기·PID 신원 불일치, 300epoch 이력 중 최근5개만 반환, 144개 gate stdin 전달·재검증 뒤 Test-cost만 시작, 중복 시작 거부를 확인했다. tmux 경계는 mock으로 대체했다.
- 이 테스트는 실제 SSH·GPU·학습·추론을 실행하지 않았다. 실제 원격 제어는 root의 독립 검토와 고정된 승인·binding 파일 생성 뒤에만 사용한다.
