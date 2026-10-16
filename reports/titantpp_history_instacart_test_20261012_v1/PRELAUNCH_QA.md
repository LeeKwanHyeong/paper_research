# 기존 Test 36조건 평가 사전 QA

## Review Result: PASS

- Status: 사전 구현 검토 완료. 실제 5080 native qualification·전체 Validation36 replay·Test36·원본 회수 완료를 뜻하지 않는다.
- Scope: 이력 보정27 + Instacart9의 Validation 선택 checkpoint를 이용한 기존 Test 최종 비교.
- Authorization: 사용자의 “ㅇㅇ 그거 진행하자”에 따른 현재 36조건 Test 접근·추론 승인 범위. 새 독립 평가로 해석하지 않는다.
- Review changes: 이 보고서만 작성. 과학 코드·원본·checkpoint·계약·원고·Runtime·인증·`.agents/`는 검토자가 수정하지 않았다.
- Project QA configuration: `.codex/agents/qa-reviewer.toml`의 Astra/high 설정을 확인했다. 주입된 `.agents/skills/_shared/runtime/execution-protocols/codex.md`는 없으므로 설치된 `oma-qa` skill과 실행 절차를 읽었다.

### CRITICAL

- 남은 확인된 문제 없음.

### HIGH

- 남은 확인된 문제 없음.

### MEDIUM

- 남은 확인된 문제 없음.

### LOW

- 남은 확인된 문제 없음.

## 수정 확인한 발견 사항

| 최초 문제 | 수정 및 재검증 근거 |
|---|---|
| Runtime 계약에 없는 `python_version` 조회로 추론 전 KeyError 발생 | `paper/scripts/evaluate_titantpp_history_instacart_frozen_test.py:312`에서 기존 관측 Python 3.12.13 및 정확 executable을 검사한다. 계약 원본은 유지했다. |
| evaluator와 analyzer의 gate 경로·native receipt 이름·split 표기 불일치 | evaluator `:487`의 `{"split": split, **metrics.finish()}` 및 기준선 split, native job/source aliases를 확인했다. |
| Validation receipt 생성만 확인하고 실제 worker 종료를 evaluator gate가 검사하지 않음 | evaluator `:191`에서 모든 Validation receipt/actual exit0/시각·SHA를 검증하고 Test 전·후 gate 동일성을 검사한다. |
| 자식 시작 뒤 launch/status 쓰기 또는 두 번째 Popen 실패 시 시작한 자식이 남을 수 있음 | pipeline `:199`에서 즉시 추적 등록하고 전체 그룹 `finally`로 자기 process group만 종료·회수한다. TERM/INT도 같은 종료 경로를 사용한다. 실패 주입 테스트 세 종류 통과. |
| 준비된 빈 analysis 디렉터리를 거부하여 최종 집계 차단 | analyzer `:484`에서 빈 디렉터리를 허용하되 exclusive `analysis_claim.json`으로 중복·자동 retry를 차단한다. |
| 누락되거나 모두 동일하게 잘못된 target/raw/truth identity 또는 기준선이 집계를 통과할 수 있음 | analyzer `:258`에서 필수 digest, 계약 입력 SHA/대상 수, native 원본, 두 기준선의 정확한 일치를 요구한다. 계약 canonical SHA도 `:104`에서 검증한다. |
| 첫 operation seal에 변경 가능한 macOS runtime cache30개 포함 | 기존 seal `071c401e…`을 rejected 원본으로 보존했다. 수정 seal은 운영 코드4개와 `input_integrity.json`만 포함하고 배포 archive에서도 cache를 제외했다. v2 archive499개와 모든 SHA를 직접 검증했다. |

## 실행한 검증

- 마지막 실행: `PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m pytest -q -p no:cacheprovider paper/tests/test_titantpp_history_instacart_test_evaluator.py paper/tests/test_titantpp_history_instacart_test_analysis.py paper/tests/test_titantpp_history_instacart_test_pipeline.py`
- **80 passed in 2.13s**: evaluator36, analysis39, pipeline5.
- 세 운영 Python 파일 AST 파싱 통과.
- 실제 `input_integrity.json`의 **493개 파일 SHA 불일치 0**.
- 실제 canonical 계약 SHA `f7fe042869e239a369c4ca77aa0c519bad1cde40d10688f6538e51ef803c183c` 일치.
- 실제 36조건 선택 checkpoint의 비가중치 metadata를 검증했다: 선택 epoch, arm, seed, 초기화 SHA, Validation selector, source lineage, 저장 tensor SHA metadata 일치.
- 동결 과학 source129/history 및 source126/Instacart 전체 파일·closure 검증 통과.
- 여섯 source/arm 경로의 합성 CPU probe JSON을 이 report 디렉터리에서 직접 확인했다. 모두 PASS이고 qualification 전체 true다. 이 결과는 로컬 Torch 2.14 환경이며 `native_pinned_contract_checked:false`; checkpoint/실제 데이터 읽기 및 optimizer update는 0이다. QA는 해당 probe 구현을 검토했고 별도 전체 CPU 재추론을 수행하지 않았다.
- 실제 Test 추론·원격 접속·GPU 작업은 이 검토에서 수행하지 않았다. Python CLI 작업이므로 웹 접근성·npm audit는 적용 대상이 아니다.
- 배포 v2 seal SHA: `22db16310ed0b1af96b3fda32980f46095e387bc5c36e51531b2b964d14d3888`. 명시적 운영 파일5개 모두 현재 검토 SHA와 일치한다.
- v2 archive SHA: `2258906b00b409963e87f359abf347788058702c0d725c0ca1996a06498e7c5a`. 실제 archive SHA 및 구성499개 파일의 manifest/local SHA를 확인했고, runtime cache·비정규 member·절대경로·상위경로 member는 없다.
- `remote_setup_v2.py`를 추가 검토했다. 기존 root가 있으면 중단, transport SHA·고정 GPU·idle·reboot 검사, 정규 member499개 및 중복/탈출 경로 검사, exclusive root 생성 후 extracted input/operation SHA 검증 순서다. 실제 원격 실행은 주 세션에서 수행한다.

## Acceptance criteria

- [x] 원래 36조건의 고유 ID 및 3seed·arm 조합 고정.
- [x] 선택 epoch·checkpoint bytes·tensor state·metadata·과학 source를 검사하며 학습/재선택 경로 없음.
- [x] RAF 전체 부모 데이터의 Train/Validation 부분이 원래 export와 같은지 검사하는 경로 존재 (`evaluator:271`).
- [x] 시간순 전체 context, loader B128, shuffle false, target masking, 현재 target perturbation·단일/배치 일치 검사 유지 (`evaluator:292`).
- [x] 전체/큰 수량 RMSE·MAE·TimeNLL, 엄격한 tail 경계, 충분통계 재구성 및 finite 검사 유지.
- [x] 모든 Validation36의 여섯 지표·대상 수·실제 exit0 통과 전에는 Test를 시작할 수 없음.
- [x] 같은 dataset/seed의 세 arm, 최대 동시 3개, 조건/전체 timeout, claim·실패 보존, 자동 retry 없음.
- [x] 5080의 정확한 GPU UUID·기존 Runtime·유휴 확인 및 격리 root·source import 경로 검사.
- [x] 전역 gate·native 결과·actual exits·target/raw/truth identity의 집계 시 검증.
- [x] 3seed 평균·표본 SD, 모든 대조·여섯 지표 및 불리한 방향/Validation→Test 반전 보존 (`analyzer:372`).
- [x] 직전 수량·입력 이력 평균 기준선에 가짜 TimeNLL을 보고하지 않음.
- [x] 기존 Test 노출 이력과 `test_is_independent:false` 유지.
- [ ] 5080 실제 native probe·Validation36 gate 통과 — 실행 시점에 검사할 조건.
- [ ] Test36 실제 성공 종료·원본 회수 및 최종 집계 — 실행 후 별도 검증할 조건.

## 검토한 운영 파일 SHA

| 파일 | SHA256 |
|---|---|
| `paper/scripts/evaluate_titantpp_history_instacart_frozen_test.py` | `71aa18a719bdd1c1f735b6f78a5f7f50d93e696002d3561c35df22d1f7b0338c` |
| `paper/scripts/analyze_titantpp_history_instacart_frozen_test.py` | `f7f2538feb891169cedf9c12ce22581db9affc35a546e89b81258ce86d33047f` |
| `paper/scripts/run_titantpp_history_instacart_test_pipeline.py` | `18ad4e0ba38e286ac8bd49580840efb91e095650b6091b399c394f6d57797f64` |

운영 파일을 bundle에 복사한 뒤 이 SHA와 일치하는지 확인하고 operation seal을 고정한다. 실제 Runtime의 Validation 재현 실패는 Test 진입을 차단해야 하며, 사전 QA PASS를 근거로 허용오차나 모델을 변경해서는 안 된다.
