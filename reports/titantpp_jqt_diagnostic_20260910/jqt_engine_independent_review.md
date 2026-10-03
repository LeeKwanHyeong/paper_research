# J/Q/T engine 독립 검토 — 2026-09-10

검토 대상은 `paper/scripts/time_quantity_diagnostic.py`와
`simple_lab_test/search/tests/test_time_quantity_diagnostic_engine.py`이다.
검토자는 이 engine을 작성하지 않았으며 engine 파일을 수정하지 않았다.
실제 연구 데이터·원격 서버·held-out 성능에는 접근하지 않았다.

## 결론 — CPU 진단 범위에서 확인된 차단 문제 없음

처음 발견한 두 가지 재현성 문제는 engine 담당자가 수정했고, 수정 후
독립적으로 관련 테스트 13개를 실행해 통과를 확인했다. 현재 구현은
**고정된 legacy cap300 조건에서 J/Q/T를 비교하는 CPU 진단 도구**로
사용할 수 있다. 정규화된 관측 likelihood 비교나 CUDA 재현성 검증을
완료한 것으로 해석하면 안 된다.

## 발견 사항과 해결 — 완료

### 1. 유효한 다른 RNG 상태가 resume 검사에서 통과하던 문제

- 최초 구현은 model·optimizer tensor 해시는 검사했지만 global RNG와
  loader generator state는 존재 여부와 key 집합만 확인했다.
- 연구 데이터 대신 4개의 합성 표본으로 1 epoch를 만든 뒤, 저장된
  loader generator state 전부를 seed 999의 유효한 state tensor로
  교체했다. `_validate_resume`가 이를 거절하지 않는 것을 직접 재현했다.
- 이 상태는 다음 epoch의 순서를 바꾸므로, 저장된 데이터·초기값·예산이
  같아도 이전 실행을 그대로 복원했다는 보장이 깨진다.
- 수정 후 `_rng_hash`는 Python·NumPy·Torch RNG와 loader generator의
  컨테이너 타입·tensor/array 내용을 함께 지문화한다. resume는 이를
  복원하기 전에 검사한다. generator가 같은 객체를 공유하는 관계도
  계약에 저장하고, 저장 상태가 그 관계와 일치하는지 확인한다.
- 근거: `time_quantity_diagnostic.py:212`, `:366`, `:385`, `:421`.

### 2. 충분히 고정되지 않은 CUDA 환경에서 exact replay를 허용하던 문제

- 최초 구현은 `cuda`를 허용했지만 GPU 장치/compute capability, cuDNN
  버전, cuBLAS workspace 설정과 같은 실행 조건을 충분히 고정하지
  않았고 CUDA 재개 검증도 없었다.
- CPU synthetic 성공만으로 다른 GPU에서도 같은 수치와 checkpoint가
  재현된다고 주장할 수 없다.
- 수정 후 `run_arm`은 model을 장치로 옮기기 전에 CPU 전용 실행을
  강제한다. CUDA/MPS는 별도 재현성 검증 전까지 거절된다.
- 근거: `time_quantity_diagnostic.py:464` 및 accelerator 거절 테스트.

## 확인한 구현 성질 — 완료

- **기존 기준선 경로 보존:** 별도 engine이며 기존
  `core.target_outputs`와 qualified runner의 기본값을 변경하지 않는다.
  joint mode의 출력과 gradient는 동일 RNG 조건에서 원래 joint
  arithmetic과 비교하는 테스트가 있다.
- **비활성 head 보존:** quantity-only에서는 시간 head를, time-only에서는
  수량 head를 호출하지 않는다. 해당 parameter의 gradient를 지우고
  `requires_grad=False`로 두며 AdamW parameter 목록에서도 제외한다.
  여러 epoch 후에도 비활성 head tensor가 초기값과 동일함을 검사한다.
- **초기값·예산·순서:** 새 모델/loader를 arm마다 받으며 초기 state,
  loader 초기 generator 상태, batch 수, 목표 표본 수, optimizer 설정을
  계약에 고정한다. 고정 epoch 수를 수행하고 early stopping하지 않는다.
  합성 J/Q/T 실행에서 초기 hash, epoch별 batch-order hash 및 step 수의
  일치를 검증한다.
- **재개:** model·optimizer·RNG·loader generator·history·selector를
  epoch 단위의 atomic checkpoint에 함께 저장한다. 정상 재개와 중단 없는
  실행이 세 objective 각각 동일한 최종 state·history·selector를
  만드는 테스트가 있다. 다른 설정·초기값·source identity로의 재개를
  거절한다.
- **선택 기준:** raw quantity RMSE와 legacy time loss를 각각 저장하고
  strict decrease일 때만 교체한다. 동률이면 가장 이른 epoch를 유지한다.
  비활성 task의 metric/selector는 `None`이며 유효한 점수인 0으로
  대체하지 않는다.
- **누출 방지:** 기존 causal history 위치를 사용하고 최종 수량과 memory
  write를 가린다. 최종 target 값과 padding 값을 바꿔도 수량 예측이
  바뀌지 않는 테스트가 있다. validation target에 test label을 주는
  loader를 거절한다. 실제 split materialization과 digest 검사는 별도
  CLI wrapper 계약의 책임이다.
- **gradient 진단:** 두 loss는 한 번의 stochastic encoder realization을
  공유하고 진단 뒤 RNG, parameter flags, 기존 `.grad`, model state를
  보존한다. global clipping norm에는 encoder뿐 아니라 두 head도
  포함한다. 초기 수량 head가 0이면 cosine을 부적용 상태로 표현한다.

## 실행한 독립 확인 — 완료

```text
python3 -m pytest -q simple_lab_test/search/tests/test_time_quantity_diagnostic_engine.py \
  -k 'rng_state or unqualified_accelerator or epoch_resume'
13 passed, 11 deselected in 8.58s
```

전체 engine 24개 통과는 담당 agent가 보고했으며, 여기서는 위 13개를
독립적으로 재실행했다. 별도로 CLI 작성 범위에서 최신 wrapper 테스트
30개와 Ruff도 통과했다. 해당 검증에는 tiny synthetic J/Q/T 1 epoch와
완료 후 resume의 실제 engine 연동이 포함된다.

## 해석과 후속 범위

- 이 진단의 새 joint arm은 Q/T와 초기값·학습 RNG·관측 예산을 맞춘
  **새 비교 기준**이다. 과거 frozen B checkpoint의 실험 이력을 그대로
  재현했다고 보고하거나 기존 결과표에 덮어쓰지 않는다.
- J의 RMSE 최적 checkpoint와 시간 loss 최적 checkpoint가 다르면 두
  결과를 하나의 모델이 동시에 달성한 성적으로 합치지 않는다. 현재
  wrapper는 J(raw-RMSE) 대 Q(raw-RMSE), J(time) 대 T(time)를 별도 pair로
  비교하며 epoch별 순서·count·step 일치를 추가 검사한다.
- CPU 진단은 시간·수량 분리가 이득이라는 가설을 자동으로 증명하지
  않는다. 목적 함수, global clipping, representation sharing 효과의
  구분은 실제 사전 고정 비교 결과로 판단해야 한다.
- CUDA 학습, 다른 backbone으로의 확장, 정규화된 duration likelihood
  경로, 추가 seed/held-out 평가는 이번 구현의 검증 완료 범위가 아니다.
