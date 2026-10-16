# RAF log/raw v2 운영 수정 실행 전 독립 QA

## Review Result: PASS

- 관측시각: 2026-10-12T12:48:59.505376+09:00
- 범위: 승인된 선택 RAF9 × Train/Validation의18 pass. 이 판정은 로컬 코드·계약·봉인 준비 검토이며 실제 GPU 종료 판정이 아니다.
- 프로젝트 qa-reviewer 설정: gpt-6-astra / high. protocol0·1·9·12와 oma-qa 적용. `.agents/` 수정0; 요청된 전용 QA 보고서만 작성.

### CRITICAL
- 없음.
### HIGH
- 없음.
### MEDIUM
- 없음.
### LOW
- 없음.

## 검증 결과와 수용 조건

- [x] canonical `0c75bdd4d70adf8569ea2602c942e93f6b27b1206a4595711b2673997ee0758a`, input213개 raw SHA, scientific source129 closure, 선택9 checkpoint와 Train/Validation18 reference 계보를 stdlib로 재검증했다.
- [x] frozen `CountAwareTPP.quantity_outputs`의 reduction=none log MSE와 실제 hook logit의 softplus/log1p·raw decode를 같은 dtype에서 bitwise 확인한다. 평균 raw forecast로 log 손실을 재구성하지 않는다.
- [x] 실제 frozen loader/3route를 사용하는 fabricated CPU8tests 모두 PASS. Train3·Validation2의 worker 진입·완료, scalar loss/오염 loss 거부, 빈 tail·Test split/입력 SHA/대상 drift 거부, hook forward/gradient/state 비개입 검증 포함. 최초 signed-log fixture assertion은 첫 항 정확0을 반영해 수정했고8/8 재통과했다.
- [x] Python AST·bash -n 통과. Python batch 진단이므로 npm audit·web lint·WCAG는 적용 대상이 아니다. 인증 변경·외부 API·네트워크 호출은 evaluator에 없고, path confinement·정규 tar member·exclusive write·원본 SHA·실패 보존을 검토했다.
- [x] 최대3개 동일 dataset/seed arm, split은 Train/Validation, 학습·Test·selector 변경0. own process group에 한정한 timeout 정리와 actual child exit 기록을 검토했다.
- [x] finalizer의 exact worker launch argv vs sys.argv[1:] 정규화, PID·PPID, native launch/exit bridge, qualification 이후 실제 시작, pipeline OS exit0·소유 PID 부재·원본 archive/manifest 검증을 확인했다.
- [x] 준비·배포·회수는 격리 root와 봉인된 입력에 한정하고 재실행·재개·기존 claim 삭제를 허용하지 않는다.
- [x] 원고 SHA `26b3abb0f8dad81eccdbfff0e1f288d1032f8209ea545cc159627ddced7a4354` 유지 확인.
- [ ] 실제5080 native3 통과,18 OS worker/pipeline exit0, 원본 회수·로컬 SHA·최종 수치 독립 QA는 실행 후 FINAL_QA에서 확정한다.

## v1 실패와 이번 수정의 범위

- v1은 첫 synthetic native probe의 cuDNN GRU eval backward 제약으로 actual exit1; 실제 선택 checkpoint pass0이었다. 원래 v1 claim/failure/operation 및 FINAL_QA FAIL을 보존했다.
- v2는 **synthetic gradient hook 동등성 fixture에서만** `torch.backends.cudnn.flags(enabled=False)`로 ATen backward를 사용하고 원래 enabled 상태 복원을 확인한다. 그 앞의 native CUDA forward bitwise qualification 및 실제 선택 모델 추론은 기존 cuDNN 경로를 유지한다. 모델 eval 상태·원본 state·checkpoint·입력·selector는 동일하다.
- 수정 후 fabricated8/8 CPU tests 재통과. CPU만으로 CUDA 성공을 선언하지 않으며 v2 실제native3 exit0을 학습대상 읽기 전 필수 gate로 둔다.
- v1/v2의 jobs·datasets·sources·h1 객체가 정확히 동일함을 확인했다. 수정된 actual canonical은 `0c75bdd4d70adf8569ea2602c942e93f6b27b1206a4595711b2673997ee0758a`이다.

## 검토 파일과 raw SHA

- `paper/scripts/evaluate_titantpp_raf_log_raw.py` — `1fae4e7fa0f05adfbbac22ca1d28d76660cb0848209b165494db635e686af6e9`
- `paper/scripts/run_titantpp_raf_log_raw_pipeline.py` — `aa0450fa6ce41f99da85f66da287a7704fb6b7ebacfc8893e45eefc8353aa17f`
- `reports/titantpp_raf_log_raw_20261012_v2/local_verify_retrieval.py` — `d2e59561698428158f3210e99d7b29aebec4abf3bb1976e9a387ac23d10d2726`
- `reports/titantpp_raf_log_raw_20261012_v2/prepare_transport.py` — `096448c2b343ae5099e667d425ce16a14d41972a850774a0bc1e0feee3cbef52`
- `reports/titantpp_raf_log_raw_20261012_v2/remote_finalize.py` — `93ed517778d6ad2040ab7379cee5b789b58268131c11d1fec0195624e1955e85`
- `reports/titantpp_raf_log_raw_20261012_v2/remote_health.py` — `0b70f5985c85223dd00954655cfa7872b59931c4c365e1d34f3cd739bc2543d3`
- `reports/titantpp_raf_log_raw_20261012_v2/remote_launch.py` — `0440df486bb9794e718e92cae0f18ce11d97b5ae395c8a428da35bef67e9a150`
- `reports/titantpp_raf_log_raw_20261012_v2/remote_setup.py` — `1c06efb7c5b76f9f459c66ce5db2c1965a18a93dabbff505cfd532c2907259e0`
- `reports/titantpp_raf_log_raw_20261012_v2/remote_wait_and_finalize.py` — `ead2c7d50a89e06f485d12af45fbc980bcc5f2740d2a40d0d30bb0dc8e09b5a1`
- `reports/titantpp_raf_log_raw_20261012_v2/launch.sh` — `7e2cd981391b0d92f3c39f7ce0ed15c2087bb9cc0b13f3c3e04617cce5a5d364`
- `reports/titantpp_raf_log_raw_20261012_v1/test_evaluator.py` — `55a81784c775ea95d61855e655b997dd9da6d0160162d566824f0eb88ade6bd2`

## 변경 파일
- 이 검토자의 이번 변경: v2/PRELAUNCH_QA.md만. source·.agents·원고 수정0, 원격/GPU 호출0.
- 이 문서가 operation에 봉인되면 수정하지 않는다. operation/transport raw SHA는 봉인 후 별도 증적으로 대조한다.
