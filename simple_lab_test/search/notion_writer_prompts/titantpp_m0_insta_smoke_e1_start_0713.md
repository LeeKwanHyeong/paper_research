다음 실험 시작 내용을 Notion에 정리해주세요.

위치:
- `5. Model Design Enhancement > Enhancement & Validation History`
- `2. Confirm and Refine Topic`에는 작성하지 않음
- 같은 제목의 페이지가 있으면 새 페이지를 만들지 말고 기존 페이지를 업데이트

페이지 제목:
- `TitanTPP M0 Instacart Top-20 e1 Smoke (2026-07-13)`

기준 시각:
- 실험 시작 시각: `2026-07-13 11:16:45 KST`
- 실행 서버: `5090` (`192.168.0.71`)
- tmux session: `titantpp_m0_insta_e1_0713`
- conda env: `ai_env`

실험 목적:
- synthetic CUDA model-test를 통과한 M0 direct log2-magnitude decoder가 실제 Instacart fixed-split DataLoader에서 forward, backward, validation/test evaluation까지 수행되는지 확인
- top-20 subset의 train event만 사용해 global mean/population variance가 계산되고 manifest와 checkpoint config에 저장되는지 확인
- legacy predicted mark를 우회한 quantity evaluation과 `magnitude_loss`, `log_qty_mae/rmse` artifact schema 확인
- Intermittent e50 screening 전에 checkpoint, resume, scale-wise metric, plot 경로 문제를 차단

실험 계획:
- dataset: `insta_market_basket`, top `20` series
- model/candidate: `TitanTPP / small_lmm`
- epochs/seeds: `1 / 42`
- lr/batch size: `1e-3 / 16`
- lookback/max sequence length: `10 / 16`
- split mode: `fixed`
- train loss scope: `target_only`
- quantity decoder: `direct_log_qty`
- normalization: train-global M0
- value input: `none`
- marker objective: plain CE, `lambda_ordinal=0`
- magnitude/direct quantity objective: `lambda_magnitude=1.0`, `lambda_qty=0.25`
- sigma floor: `0.0014535461338152059`
- exp2 clamp: `[-2, 15]`
- eval selections: `best_val_nll,best_score,final`
- device: CUDA

실행 명령어:

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d -s titantpp_m0_insta_e1_0713 "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_m0_insta_smoke_e1_0713.sh"'
```

결과 작성란:
- 결과는 smoke 종료 후 업데이트 예정
- artifact read order: experiment manifest, run log, summary, test summary, histories, validation/test scale-wise metrics, plots
- 완료 여부, exit code, NaN/Inf/Traceback 여부, best epoch를 확인
- train-only global event count, mean, variance, std와 source split을 확인
- `qty_decoder_mode=direct_log_qty`, `magnitude_norm_mode=global`, legacy value-by-mark 비활성 여부를 확인
- validation/test NLL, magnitude loss, quantity MAE, log2 quantity MAE/RMSE를 확인
- empty scale bucket의 의도된 NaN과 runtime NaN을 구분

초기 해석 가설:
- e1이 정상 종료되고 M0 전용 artifact가 생성되면 actual-data integration gate를 통과한 것으로 판단
- e1 metric은 성능 우위 판단이나 held-out test 기반 튜닝에 사용하지 않음
- 실패 시 train-only stats 주입, direct evaluator routing, report aggregation 순으로 원인을 확인

문체:
- 구현 경로 확인과 모델 성능 검증을 구분
- smoke 결과를 V2 대비 우위로 과장하지 않음
- 확인된 artifact와 수치만 기록
