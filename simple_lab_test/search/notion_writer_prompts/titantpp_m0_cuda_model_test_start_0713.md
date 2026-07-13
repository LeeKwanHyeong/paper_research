다음 실험 시작 내용을 Notion에 정리해주세요.

위치:
- `5. Model Design Enhancement > Enhancement & Validation History`
- `2. Confirm and Refine Topic`에는 작성하지 않음
- 같은 제목의 페이지가 있으면 새 페이지를 만들지 말고 기존 페이지를 업데이트

페이지 제목:
- `TitanTPP M0 CUDA Model Test (2026-07-13)`

기준 시각:
- 실험 시작 시각: `2026-07-13 11:09:04 KST`
- 실행 서버: `5090` (`192.168.0.71`)
- tmux session: `titantpp_m0_cuda_modeltest_0713`
- conda env: `ai_env`

실험 목적:
- 새로 구현한 M0 direct log2-magnitude decoder가 5090 CUDA 환경에서 정상적으로 초기화되고 forward, NLL, magnitude loss, quantity reconstruction을 수행하는지 확인
- train-global normalization을 공유하는 `MagnitudeContext`와 Titan encoder 입력, direct magnitude head 사이의 CUDA device/shape 호환성 확인
- 이 단계는 실제 데이터 성능 비교가 아니라 Instacart e1 smoke 이전의 runtime contract 검증

실험 계획:
- dataset: synthetic intermittent-like batch, 실제 parquet 미사용
- model: TitanTPP
- candidate: `small_lmm`
- batch_size: `4`
- sequence length: `16`
- number of marks: `12` including PAD
- scale base: `2`
- quantity decoder: `direct_log_qty`
- normalization: `global` M0
- magnitude input embedding: `8`
- magnitude loss weight: `1.0`
- sigma floor: `0.0014535461338152059`
- exp2 clamp: `[-2, 15]`
- left padding: enabled
- device: CUDA

실행 명령어:

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d -s titantpp_m0_cuda_modeltest_0713 "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_m0_cuda_model_test_0713.sh"'
```

결과 작성란:
- 결과는 model-test 종료 후 업데이트 예정
- 종료 시각과 exit code를 `logs/run.log` 기준으로 기록
- `model_test_summary.json`, `model_test_summary.csv`, `MODEL_TEST_SUCCESS` 존재 여부 확인
- hidden shape, total NLL, marker/time NLL, magnitude loss, quantity prediction finite 여부 확인
- NaN, Inf, Traceback 여부 확인

초기 해석 가설:
- CUDA에서 모든 출력이 finite이고 success marker가 생성되면 M0 runtime contract를 통과한 것으로 판단
- 실패 시 성능 문제가 아니라 우선 device mismatch, shared context broadcasting, direct decoder shape 연결을 점검
- 이 결과만으로 M0가 V2보다 우수하다고 해석하지 않음

문체:
- 과장 없이 사실과 아직 검증하지 않은 내용을 구분
- 구현 완료와 모델 성능 검증 완료를 같은 의미로 쓰지 않음
