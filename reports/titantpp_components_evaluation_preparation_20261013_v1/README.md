# CNN·GRU 구성요소36 평가 도구 준비

## 승인된 로컬 준비 작업 — 완료

원래 구성요소36개의 선택 checkpoint를 평가하는 운영 연결부를 준비한다. 대상은
Taxi·Intermittent·RAF의 NONE/CNN/GRU/FULL × seed42/52/62이다. 이력 보정27이나
Instacart9를 신규36개로 합산하지 않는다. 원고 편집, 원격 원본 회수, 실제 자료
추론, 새 학습, source131·기존 운영 봉인 변경은 이 작업의 실행 범위에 없다.

현재 설계 기준선은
`reports/titantpp_components_final_test_contract_20261013_v1/evaluation_design.json`
및 `EVALUATION_CONTRACT.md`다. 이 문서는 그 설계의 구현 상태를 별도로 기록한다.
실제 원본의 모든 SHA가 결속되기 전에는 평가 실행 준비 완료를 선언하지 않는다.

## checkpoint·원본 결속 — 구현 완료

`paper/scripts/prepare_titantpp_components_frozen_evaluation.py`는 stdlib만 사용해
회수 원본의 byte SHA, terminal manifest, 실제 scientific worker exit0, 선택
history와 원본 Validation endpoint를 확인한다. parquet를 해석하거나 저장 모델을
불러오지 않는다. 새 `contract.json`만 배타적으로 만들며 실행 permit을 발행하지 않는다.

- 원래 source131 closure와 training 계약 canonical을 고정한다.
- 전체 Validation raw 수량 RMSE의 최초 strict finite 최소 epoch를 선택한다.
  동률은 최초 epoch이며 Test를 선택에 사용하지 않는다. CSV/JSON 원본 history를
  모두 지원하고, 완료된 epoch1부터 마지막 epoch까지 빠짐없이 확인한다.
- 선택 binary, tensor SHA, metadata, 최초 초기화 SHA, source revision과 원본
  endpoint의 epoch·조건·tensor·여섯 지표를 연결한다. 선택 binary는
  `best_val_qty_rmse_model.pt`만 허용하며 종료 `last_epoch_state.pt`는 감사용이다.
- history와 endpoint 자체의 입력 SHA만 신뢰하지 않고 원본 terminal의 해당 파일
  SHA와도 대조한다. 원본 endpoint↔history의 RMSE는 기존 생산 경로의
  `rel_tol=1e-10, abs_tol=1e-8`을 유지하며 선택 epoch를 바꾸지 않는다.
- 원 학습 정의의 Train 통계·전처리·모델·data/split SHA를 고정한다. RAF는 전체
  parent 자료와 Train/Validation export의 SHA를 별도로 보존한다.
- 원5090 Intermittent42/52의8개와 5080의28개를 구분한다. seed62 이관에 따른
  물리 GPU·Runtime 차이를 숨기지 않는다.

합성 bindings packet의 검증 예시는 전용 preparation 테스트에 있다. 실제 packet은
원본 회수 후에만 구성한다. 각 job은 `checkpoint`, `checkpoint_metadata_*`,
`history_path/history_sha256/history_original_relative_path`, `last_checkpoint`,
`terminal_manifest_*`, `worker_exit_*`, `validation_endpoint_path`,
`validation_endpoint_sha256`, `validation_endpoint_original_relative_path`,
`validation_reference`를 가져야 한다. endpoint reference는 원본 selected의 전체·큰
수량 count와 RMSE/MAE/TimeNLL을 그대로 사용하며 `split=validation`을 명시한다.

## 신규36개 평가 연결부 — 구현 완료

`paper/scripts/evaluate_titantpp_components_frozen.py`는 frozen source131에서 각 구조를
생성하고 같은 선택 모델로 여섯 지표를 계산한다. 기존 평가기와 과학 코드는 수정하지 않는다.

- `check-contract`: stdlib 정적 검사. 모델·실자료를 불러오지 않는다.
- `qualify-synthetic`: fabricated CPU 입력과 임시 모델만 사용한다. 실제 checkpoint의
  재현이나 native GPU Validation 통과를 뜻하지 않는다.
- `worker`: 명시적인 해당 split/host/job permit과 byte SHA가 먼저 필요하다. 원래
  native CUDA Runtime만 허용하며, 정적 검사·permit 확인 전 데이터나 모델을 읽지 않는다.
- 실제 평가에서 `eval()`, `no_grad()`, strict state load를 사용한다. 평가 전후
  model tensor SHA와 checkpoint/data/source/도구 SHA, Runtime이 같아야 한다.
- Test는 모든36개 native Validation 영수증과 실제 child exit0를 묶은
  `validation_gate.json`을 먼저 검증한다. 5080의 기록을 5090의 현재 실행 경로로
  해석하거나 그 반대로 비교하지 않는다. 현재 worker의 실행 경로 검사는 유지한다.
- 기존 claim/receipt/failure가 있으면 자동 반복하지 않는다. 원본 회수·학습·GPU 작업을
  시작하는 기능은 없다. 평가 결과에 원시 예측 행을 내보내지 않는다.

Native Runtime packet 필수 필드는 `host`, `python`(절대 경로), `python_version`,
`torch`, `cuda`, `cudnn`, `numpy`, `polars`, `gpu_uuid`, `torch_num_threads`,
`interop_threads`, `dtype`, `deterministic_algorithms`, `TF32`다. threads1·interop1·
float32·determinism·TF32 off와 기존 cuDNN91900/92000을 유지한다.

평가의 전체·큰 수량 RMSE/MAE/recorded positive-integer TimeNLL은 동일 checkpoint에서
계산한다. 큰 수량은 Train 고정 경계 Taxi `q>3449`, Intermittent `q>187`, RAF
`q>200`이다. 직전 수량·동일 관측 이력 창 평균 기준선에는 시간 모델이 없어 TimeNLL을
보고하지 않는다. 통계 집계의 평균·표본SD·seed별 차이는 원 평가 설계를 유지한다.

Population packet은 count, target identity, truth, recorded-time SHA를 가진다.
target/truth는 기존 frozen 평가기의 행 순서와 필드를 사용한다. recorded-time의 추가
stdlib 정의는 아래와 같다. gap bytes는 목표 recorded gap을 평가 순서대로 배열한
float64 little-endian 값이다. 이 입력 SHA가 실제로 확보됐다는 의미는 아니며,
실제 자료 접근 전에 승인된 packet에 결속해야 한다.

```python
SHA256(b"titantpp_components_recorded_target_time_v1\0"
       + split.encode("utf-8") + b"\0" + gap_bytes)
```

## 파라미터 수 기록의 보완 — 확인 완료

실제 frozen 모델을 데이터셋별 입력 길이로 생성해 확인한 수는 다음과 같다.

| 데이터셋 | NONE 활성 | CNN 활성 | GRU 활성 | FULL 활성 | 등록 전체 |
|---|---:|---:|---:|---:|---:|
| Taxi·Intermittent | 89,859 | 90,435 | 114,591 | 115,167 | 115,167 |
| RAF | 78,851 | 79,427 | 103,583 | 104,159 | 104,159 |

RAF는 positional embedding 길이84, 다른 두 자료는256이다. 차이11,008은
`(256−84)×64`로 설명된다. 기존 선언값115,167을 모든 자료의 실측값이라고 부르지 않는다.
봉인된 선언 metadata·모델·학습 설정을 수정하지 않고, 평가 영수증에 데이터셋별 실측
`physical_parameter_counts`를 별도로 기록한다. 실측 활성 수는 평가용 gradient 차단 전
측정한다.

## 로컬 검증과 남은 실행 단계 — 로컬 검증 완료

전용 테스트로 선택·계약·permit·신원·원본 결속 실패를 확인하고, 실제 frozen 모델의
네 구조를 fabricated 입력에서 검증한다. 최종 실행 로그, 파일 SHA와 독립 검토 결과는
이 폴더의 QA 기록에 보존한다. CPU 합성 검증은 실제 native 수치 재현을 대신하지 않는다.

최종 통합 검증은 **68개 PASS, 실제 프로세스 exit0**이다: preparation46개,
evaluator21개, 봉인 source131 실제 모델의 합성 입력 검증1개(3자료×4구조의12모델).
계약 생성 결과를 실제 evaluator 정적 검사로 연결한 검증과 서로 다른 host의
28+8 영수증으로 전체 Validation gate를 검증한 회귀를 포함한다.
`INTEGRATION_TESTS.log`와 `LOCAL_QA.json`에 실행 명령·실제 종료·파일 SHA를 기록했다.
독립 코드 검토의 미해결 CRITICAL/HIGH/MEDIUM은0이다.

source131 파일131개, 봉인 운영파일4개, 기존 도구40개의 SHA는 실행 전후 모두
일치했다. 현재 CPU 검증 환경은 Python3.12.10/torch2.14.0/numpy2.3.1/polars1.31.0이며
학습 서버의 native Runtime과 다르다. 합성 모델의 single/batch 최대 차이는
`3.814697265625e-6`으로 정해 둔 `atol/rtol=1e-5` 안에 있었다. 새 학습·실자료 추론·
실제 checkpoint 불러오기·원격 호출·permit 생성은0이다.

1. **원36개 학습 종료와 immutable 원본을 결속한다 — 외부 작업 대기**
   남은 학습이 종료되고 원본 회수·manifest/archive 감사가 완료되어야 실제 contract를
   생성할 수 있다. 선택 binary와 원 GPU Runtime이 미확정이면 진행하지 않는다.
2. **원 GPU의36개 Validation 재현을 확인한다 — 다음 작업 / 실행 승인 필요**
   명시적 실행 permit 아래 여섯 지표 `rel_tol=1e-5, abs_tol=1e-5`, 정확한 모집단과
   입력 SHA, 실제 supervised exit0를 확인한다. 어느 한 조건이라도 실패하면 Test로 넘어가지 않는다.
3. **고정된 모델의 chronological Test를 평가한다 — 다음 작업 / 실행 승인 필요**
   전체 Validation gate가 완성된 뒤 5080의28개와5090의8개를 평가하고 원 설계의 비교군과
   집계한다. 새 학습·Test 재선택·통계 재추정은 없으며 기존 접근 이력을 보존한다.
   이를 새 독립 평가라고 부르지 않는다. 원격 supervisor 배포와 실제 exit/gate 생성은
   현재의 로컬 구현 검증과 구분한다.
