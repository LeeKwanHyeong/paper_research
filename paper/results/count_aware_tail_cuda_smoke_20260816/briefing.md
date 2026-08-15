# Count-aware Tail-aware CUDA and Instacart Smoke

## 상태

- 최종 상태: **COMPLETE**
- 실행 서버: `5080`, NVIDIA GeForce RTX 5080
- Source revision: `a24788ff781bdcb9256e2a10e6eb6c90e21394c8`
- Frozen `lambda_tail`: `0.09111380335463036`
- Held-out test: 미사용

## 검증 결과

- Focused contract tests: `26 passed`
- Intermittent CUDA mini-run: T0/T1/T2 각 1 epoch, 총 3 runs 완료
- Instacart top-20 fixed-split smoke: T0/T1/T2 각 1 epoch, 총 3 runs 완료
- Forward, backward, checkpoint, history, summary, scale-wise artifact 생성 확인
- NaN, Inf, OOM 없음

## 초기 오류와 조치

첫 Instacart 실행은 validation에 표본이 없는 quantile bucket을 집계하면서 중단됐다.
모델 또는 loss 오류가 아니었으며, 빈 breakdown row를 생략하도록 수정한 뒤 동일 smoke를
재실행해 완료했다. 원본 로그에는 첫 traceback과 이후 성공 로그가 함께 남아 있다.

## 해석

이 결과는 실행 계약 검증용이며 모델 품질 근거로 사용하지 않는다. Intermittent
seed-42 e300 validation-only screening을 시작할 수 있는 상태다.
