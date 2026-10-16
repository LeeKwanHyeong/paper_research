# RAF log/raw v1 실제 실행 QA

## Review Result: FAIL

- 실제 종료 관측: 2026-10-12T12:45:04.964087+09:00; pipeline OS exit1. native qualification 첫 synthetic route에서 중단됐으며 선택 checkpoint Train/Validation 평가 시작0·완료0이다.
- 실행 전 로컬8/8 CPU 검증 PASS와 실제CUDA qualification 실패를 구분한다. 기존 PRELAUNCH_QA/operation seal을 수정하지 않는다.

### CRITICAL
- 없음.
### HIGH
- `search_artifacts/titantpp_raf_log_raw_20261012_v1/control/evaluate.py:702` — cuDNN GRU가 eval 상태인 synthetic fixture에서 backward를 호출해 `RuntimeError: cudnn RNN backward can only be called in training mode`가 발생했다. 원본 forward/수치 손실 실패가 아닌 운영 검증의 backend 호환성 오류다. 별도 v2에서 추가 synthetic gradient fixture만 다음과 같이 backend를 격리하고 종료 후 복원을 확인한다. 실제 선택 모델 forward는 원래 cuDNN/Runtime을 유지한다.

```python
with torch.backends.cudnn.flags(enabled=False), torch.enable_grad():
    # synthetic fixture의 plain/hooked forward와 gradient 동등성만 비교
    plain_grad = torch.autograd.grad(plain_loss, tuple(model.parameters()), allow_unused=True)
```

### MEDIUM
- 없음.
### LOW
- 없음.

## 수용 조건 및 보존

- [x] `FAILED_REMOTE_CONTROL/pipeline_exit.json` actual exit1, `pipeline_latest.json` failed/native_qualification, Train/Validation 완료 목록0 확인.
- [x] `NATIVE_PROBE_FAILURE.log` 원 traceback과 native worker exit1 확인.
- [x] v1의 source/contract/operation/permits/claims/failure를 보존한다. v1 completion 원문은 생성하지 않는다.
- [ ] 승인된18개 평가의 실제 완료는 새 운영 버전에서 별도로 검증한다.
- QA 변경 파일: 이 `FINAL_QA.md`만. scientific source·main.tex·.agents 수정0, 원격/GPU 실행0.

원 로그 raw SHA: `132b103381f30c334e0a50d7b85fc1a1efd691861d9e1d5290a65b820d4f7b0e`.
