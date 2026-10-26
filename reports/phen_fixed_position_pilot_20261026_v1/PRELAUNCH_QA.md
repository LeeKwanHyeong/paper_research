# PHEN 고정 위치 인코딩 시험 — 독립 사전 QA

## Review Result: PASS

- 상태: 로컬 사전 검토 통과. 원격 synthetic/capacity 통과나 실제 학습 완료를 뜻하지 않는다.
- 검토일: 2026-10-26 (KST).
- 검토자: `/root/fixed_position_qa`. 프로젝트 역할 설정은 `gpt-6-astra` / `high`; 실제 로드된 모델은 별도로 확인할 수 없으므로 동일하다고 단정하지 않는다.
- 승인 범위: 5080의 RAF·Taxi PHEN seed 42 신규 학습 각각 1회와 Train/Validation 학습 곡선. Test·5090 변경·Instacart 재학습·MAC·기존 결과 덮어쓰기 제외.
- QA가 변경한 파일: `QA_NOTES.md`, 이 문서. 실행 소스 변경 및 SSH 수행 없음.

### CRITICAL

- 미해결 사항 없음.

### HIGH

- 미해결 사항 없음.

### MEDIUM

- 미해결 사항 없음.

### LOW

- 미해결 코드 결함 없음. 역할 지침의 `.agents/skills/_shared/runtime/execution-protocols/codex.md`는 현재 경로에 없었다. `.agents/` 수정 금지와 지정된 보고서 경로를 따라 본 문서에 결과를 기록했다.

## 검토 범위와 확인한 수정

- `paper/scripts/run_phen_fixed_position_pilot.py` 전 범위.
- `paper/tests/test_phen_fixed_position_pilot.py` 전 범위.
- 같은 보고서 폴더의 `prepare.py`, `plot_curves.py`, `remote_parent.py`, `transport.py`, `PLAN.md`, `EXECUTION_CONTRACT.json`, `PREPARED.json`.
- 동결 원본의 모델 생성·학습 hook·Train/Validation loader·joint loss tracker·checkpoint 선택·early stopping·평가 코드. 과거 Test 예측/성능은 열람하지 않았다.
- 초기 차단 결함: `transport.py:35`의 파일 전용 압축이 빈 `source/sample_data`를 누락해 runner의 root sentinel 검사에서 실패했다. 소유자가 `transport.py:51`에 `for name in ['models','utils','sample_data']:(r/'source'/name).mkdir(exist_ok=True)`를 추가했고 최종 코드에서 확인했다.
- 초기 봉인 불일치: live runner에 추가된 선택 epoch 곡선 SHA·수치 대조 6줄이 최초 bundle에 없었다. 소유자가 이전 계약을 보존하고 rev2를 봉인했다. 최종 live/bundle runner 바이트와 계약 바이트가 각각 일치한다.

## 검증 증거

1. 로컬 독립 실행: `PYTHONDONTWRITEBYTECODE=1 /usr/local/bin/python3 -m pytest paper/tests/test_phen_fixed_position_pilot.py -q -p no:cacheprovider` → **35 passed in 10.52s**, 실제 exit **0**. 실제 프로젝트 데이터 대신 작은 합성 Train/Validation을 사용한 CPU 검증이다.
2. 실제 PHEN 합성 2 epoch에서 추가 Train 평가 on/off의 최종 model tensor, optimizer, 전역 RNG, Train loader generator, 선택 epoch, 수치 이력, time-head fit proof가 동일함을 테스트한다. 최종 best 재평가는 epoch 곡선에 중복 추가되지 않는다.
3. 기존 Validation의 RNG 소비는 유지한다. 추가 Train 평가만 Python/NumPy/Torch/CUDA RNG 및 각 하위 모듈의 mode와 Train generator를 보존한다. 추가 평가는 별도 비셔플 loader를 사용하며 joint loss 관측 횟수도 기존 fit proof에 더하지 않는다.
4. 원래 모델을 먼저 생성한 뒤 positional Parameter를 base 10000, amplitude 1의 persistent sin/cos buffer로 교체한다. 나머지 초기 tensor와 RNG가 같고 optimizer가 고정 위치를 업데이트하지 않음을 검증한다. 정책 식별 buffer로 구형 모델에 실수로 엄격 로드하는 경우도 차단한다.
5. 독립 SHA 검사: 기준 소스 **987개 변경 0**, 최종 소스 **991개** 및 Train/Validation 입력 **4개** 모두 계약 SHA와 일치. 원래 training/common-condition 및 RAF·Taxi dataset 설정은 제외된 `full_data_identity` 외에 동일하다. 모든 검토 Python 파일의 AST 구문 검사 통과.
6. 학습은 RAF 다음 Taxi의 단독 child로 실행된다. protected process의 정지 상태·신원 및 GPU UUID·boot를 확인한다. synthetic/capacity가 같은 계약·runtime의 CUDA 결과여야 학습할 수 있다. 첫 child의 nonzero exit면 다음 dataset을 실행하지 않는다. 기존 시도/산출물은 exclusive 생성으로 보존한다.
7. 매 epoch 전체 Train과 Validation은 같은 epoch 종료 weights와 eval mode로 측정하며 대상 수를 대조한다. online Train 손실은 별도 표기한다. 선택 checkpoint의 tensor SHA를 해당 epoch 곡선과 대조하고 Validation 원소별 재현 및 파일 SHA를 확인한다.
8. child 실제 exit 및 부모가 관측한 dispatcher 실제 exit, stdout/stderr bytes와 SHA를 남긴다. parent 자신의 OS exit를 직접 관측했다고 주장하지 않는다. 결과 회수는 완료 manifest의 bytes/SHA와 archive SHA를 검증한다.
9. 곡선은 보간·평활화하지 않으며 실제 child exit 0과 fit receipt complete가 모두 있을 때만 완료로 표시한다. 미완료는 partial이다. 실제 데이터 곡선의 시각적 판독과 수치 해석은 결과 회수 후 수행해야 한다.

## 최종 검토 바이트

| 대상 | SHA256 |
|---|---|
| execution contract raw | `13d0c5352c494f520337ff4ae1d99d39a9f599e41a5445b78cfde70445aeb93e` |
| runner | `0db5e568702e293d41cc806424f1aead9c92dfc784deac1c7bf31a1207b507b9` |
| tests | `b801dd6ac4111f462839d02385619c7f3beb2fee5cac9e388590ad5dfcb4e73b` |
| prepare | `f0e6ea0d7ccce1d4522041f4e7c3f31f451ba2bbf0cbf291b3d8658884b65c38` |
| plot_curves | `c88051e51e813bf9ff89006a2f33a4d9558b1ad75410e25edd300f697f032433` |
| remote_parent | `8b9a55f5822555a25d771a4df8a793d8d74db5c52097c02b388b2673d22178eb` |
| transport | `b01bd681fa93aedf26f4ea8febcd5b4df04e0ef37bb6d1e63b1762c0cec1d2d8` |

## Acceptance criteria

- [x] 승인된 dataset/seed/host 및 Train/Validation 범위 고정
- [x] 기존 source/input/실험 자료 보존과 신규 경로 격리
- [x] 위치 인코딩 이외 초기 tensor와 기존 학습·선택 조건 유지
- [x] 추가 평가의 학습 RNG·optimizer·fit proof 비개입 검증
- [x] 같은 종료 weights의 전체 Train/Validation 곡선 및 대상 수 검사
- [x] 선택 checkpoint SHA·곡선·개별 Validation 재현 연결
- [x] 실패 시 후속 학습 중단 및 실제 child exit 증거 보존
- [x] 최종 source/contract 봉인과 검토 파일 일치
- [ ] 원격 5080 synthetic/capacity actual exit 0 — 실행 전 gate
- [ ] RAF·Taxi 실제 학습 종료, 회수, 학습 곡선 가독성 및 성능 해석 — 후속 작업

## 해석 및 남은 작업

- 이 실험은 위치 표현값·크기·학습 가능성을 함께 바꾸는 한 seed의 대조이다. 개선 여부를 Intermittent나 다른 seed로 일반화하거나, 미학습 positional row 하나의 단독 인과 효과로 주장하지 않는다.
- 추가 Train 평가를 포함한 elapsed/epoch를 과거 순수 학습 속도와 직접 비교하지 않는다. online 손실과 epoch-end eval 손실을 구분한다.
- 다음 순서: 승인된 5080 신규 경로 배포 → native synthetic/capacity gate → RAF/Taxi 순차 학습 → 실제 종료·회수 검증 → 같은 가중치 Train/Validation 곡선과 기존 Validation 기준선 비교. 이후 단계의 검증은 아직 완료되지 않았다.

## 실행 환경 전달 복구 검토 — attempt2

### Review Result: PASS (control repair, local-only)

첫 native qualification에서 아래 환경 전달 결함이 드러났다. 앞선 로컬 PASS는 native 실행 가능성을 증명하지 못했다. 최초 실패 기록을 유지한 상태에서 수정된 제어를 별도로 검토했다. 이 추가 판정도 CUDA qualification 통과를 대신하지 않는다.

### CRITICAL

- 미해결 사항 없음.

### HIGH

- 해결됨: 최초 `bundle/source/pilot_assets/remote_parent.py:23`은 계약의 `hosts.5080.environment`를 전달하지 않아 기준선 `LD_LIBRARY_PATH` 등이 누락됐다. RAF synthetic child가 `libnvrtc-builtins.so.13.0`을 찾지 못해 exit 1로 끝났다. attempt2 `child_environment()`는 `{**os.environ, **contract['hosts']['5080']['environment']}` 순서로 계약 값을 적용하고 source/cache 경로만 새 root에 맞춘다. 모든 child가 동일 helper를 통과한다.
- 미해결 HIGH 없음.

### MEDIUM

- 미해결 사항 없음.

### LOW

- 미해결 사항 없음.

### 독립 확인

- `NATIVE_FAILURE_DIAGNOSIS.stdout`에서 최초 synthetic 실제 exit 1, stderr 14,861 bytes 및 SHA256 `d91c158afbf2086b9d81ac31285040db52c709a168dbcdf189d7d61504cdf974` 일치를 직접 검산했다. source check exit 0 이후 첫 synthetic에서 중단됐고 당시 status의 두 jobs는 모두 빈 객체이며 dispatcher가 시작되지 않았다. 실제 데이터 fit 재시도가 아니다.
- attempt2의 runner/test 및 과학 소스는 첫 봉인과 동일하다. 소스 차이는 `pilot_assets/remote_parent.py` 하나다. datasets/training/position_encoding/architecture/input_files/jobs/common_condition 값은 모두 동일하다. host root와 source_root 및 PYTHONPATH/cache 경로만 새 시도로 분리하고, GPU·boot·protected process 조건은 유지한다.
- helper를 독립 import하여 로컬 child→grandchild를 실제 실행했다. 두 단계 모두 계약의 공개 환경 키 15개가 완전히 일치했고 실제 exit 0이었다. `LD_LIBRARY_PATH`, CUDA 장치, TF32, CUBLAS, 스레드, seed 조건은 원래 계약 값이다. 출력에 전체 ambient environment를 기록하지 않았다.
- 원래 실패 directory·계약·bundle은 보존된다. `transport.py --attempt 2`는 별도 local attempt2와 `PREPARED.json`에 고정된 별도 remote root를 사용하며 tmux 이름도 그 root 이름이다. 기존 시도를 재사용하거나 삭제하지 않는다.
- attempt2 bundle의 991 source + 4 input SHA, 최종 archive의 동일 995개 SHA 및 996개 member를 검증했다. 과학 runner 변경이 없어 앞서 수행한 35개 테스트 결과를 그 코드에 적용하며, 추가 변경은 환경 전파 테스트로 검증했다.
- attempt2 계약 raw SHA: `bd95bae0571ce393ec85c860e350272ffabbeb0b7e7dd8fef6179f6b8a357cfe`.
- attempt2 remote_parent SHA: `821917a8643cc5139f5105cdb3e914010b450c9cf3afacbaa85e9108775dee93`.
- 첫 독립 계약 비교에서 source_root까지 불변으로 요구한 QA assertion이 실패했다. 확인 결과 새 root로의 의도된 경로 정규화였으며, 검사를 `source_root == new_root/source`로 고쳐 재검증했다. 과학 조건 변화나 실행 결함으로 판정하지 않았다.

### Acceptance criteria 및 남은 순서

- [x] 최초 native 실패 원문·실제 exit·SHA 보존 확인
- [x] 실제 데이터 fit 진입 전 실패 확인
- [x] 기준선 환경의 child 및 grandchild 전파 검증
- [x] source/cache/receipt/tmux의 attempt2 격리 검토
- [x] 과학 코드·데이터·학습 설정 불변 및 archive SHA 확인
- [ ] attempt2 5080 native synthetic/capacity 실제 exit 0
- [ ] 승인된 최초 RAF/Taxi fit 두 건, 종료 및 결과 회수 검증

다음 작업은 attempt2의 native qualification이다. 다시 실패하면 해당 원인과 증거를 확인하기 전 실제 학습으로 넘어가면 안 된다. 이 QA는 SSH를 실행하지 않았다.
