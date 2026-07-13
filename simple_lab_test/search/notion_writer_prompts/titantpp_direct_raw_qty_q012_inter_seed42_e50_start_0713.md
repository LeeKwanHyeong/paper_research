# TitanTPP Direct Raw Quantity Q0/Q1/Q2 Intermittent Seed-42 e50 Screening 시작

`5. Model Design Enhancement` 아래에 세부 페이지
`TitanTPP Direct Raw Quantity Q0 Q1 Q2 Intermittent Seed-42 e50 Screening`을
생성하거나 같은 제목의 기존 페이지를 갱신한다. 상위 history의 `2026-07-13` 아래에
제목 3 `Step 11. Q0/Q1/Q2 Intermittent Seed-42 e50 Validation-Only Screening`으로
연결한다. `2. Confirm and Refine Topic`에는 작성하지 않는다.

## 실험 시작 정보

- 실제 실행 시작 시각: `2026-07-13 17:14:05 KST`
- 실행 서버: `5090` (`192.168.0.71`, user `leekwanhyeong`)
- 프로젝트 경로: `/home/leekwanhyeong/workspace/paper_research`
- Conda 환경: `ai_env`
- tmux session: `inter_raw_q012_e50_0713`
- experiment artifact: `search_artifacts/model_enhancement_direct_raw_qty_q012_inter_seed42_e50_0713`
- V2 reference artifact: `search_artifacts/model_enhancement_v2_inter_validation_reference_raw_q012_0713`
- 상태: `in progress; V2 validation reference completed; Q0 training entered`

## 실험 목적

Instacart actual-data integration gate를 통과한 matched Q0/Q1/Q2
`direct_raw_qty`를 Intermittent fixed split에서 seed `42`, e50으로 비교한다.
Q0 raw/global은 raw-domain control, Q1 causal masked RevIN은 canonical diagnostic,
Q2 causal shrinkage RevIN `k=8`은 short-context primary candidate다.

이번 단계에서는 frozen V2 `best_val_nll` checkpoint를 validation target에만 다시
평가하고, Q0/Q1/Q2의 validation 지표와 비교한다. Candidate 판정 전에는
`test_*`, merged `runs.csv`, `paper_outputs/report.md`, test plot을 열지 않는다.

## 고정 조건

| 항목 | 값 |
| --- | ---: |
| dataset / split | `intermittent / fixed` |
| model / candidate | `TitanTPP / small_lmm` |
| epochs / seed | `50 / 42` |
| LR / batch | `1e-3 / 128` |
| lookback / max sequence | `52 weeks / 16` |
| decoder / domain | `direct_raw_qty / raw_qty` |
| train loss scope / mode | `target_only / hybrid` |
| marker objective | plain CE, `lambda_ordinal=0` |
| value/encoder gradient route | `coupled / coupled` |
| lambda magnitude / quantity | `1.0 / 0.25` |
| magnitude input embedding | `8` |
| RevIN epsilon / shrinkage k | `1e-5 / 8` |
| raw sigma floor | `0.0550124034288891` |
| center / affine / stat context | `mean / false / none` |
| checkpoint selections | `best_val_nll,best_score,final` |

Q0/Q1/Q2는 normalization mode만 `global`, `causal_revin`,
`causal_shrinkage_revin`으로 다르다. 한 변형의 실패가 다른 변형의 실행을 취소하지
않는다. 모든 normalization statistics는 fixed train event만 사용하며 appended target과
padding은 제외한다.

## V2 Validation-Only Reference

- source artifact: `model_enhancement_v2_inter_short_e50_0710`
- source selection / epoch: `best_val_nll / 19`
- source checkpoint는 실행 전 SHA와 strict state load를 확인
- evaluation split은 `validation`만 구성하며 manifest에
  `held_out_test_read=false` 기록
- overall NLL/marker/time, raw quantity MAE/RMSE/WAPE, log2 quantity MAE/RMSE,
  mark/time metric을 export
- history count `1`, `2-4`, `5-8`, `9+` quantity/log2 MAE를 export
- validation scale-wise `1-9`, `10-99`, `100-999`, `1000-9999`, `>=10000`
  metric을 별도 CSV/Parquet로 export

## Validation Acceptance Gate

V2 대비 candidate eligibility:

- overall raw quantity MAE `>=3%` 개선
- history count `<=4` raw quantity MAE `>=3%` 개선
- log2 quantity MAE regression `<=2%`
- validation share `>=5%`인 quantity bucket의 MAE regression `<=5%`
- marker NLL regression `<=1%`
- total NLL과 time NLL regression 각각 `<=0.5%`
- mark accuracy gap `>=-0.25%p`
- DT MAE regression `<=2%`
- 모든 loss/prediction/center/scale finite
- pre-clamp negative prediction share `<=1%`

Q1/Q2의 RevIN benefit은 위 V2 eligibility를 먼저 통과한 뒤 Q0 대비 다음 조건을
추가로 확인한다.

- overall raw quantity MAE `>=2%` 개선
- history count `<=4` raw quantity MAE `>=3%` 개선
- log2 quantity MAE regression `<=1%`
- mark/time/numeric safety 유지

Q1과 Q2가 모두 통과하고 Q2가 overall 또는 short-context에서 Q1보다 `1%` 이상
좋지 않으면 단순한 Q1을 선택한다. Q1 scale collapse는 Q2 취소 조건이 아니다.

## 실험 계획

1. 5090 GPU, Python, V2 checkpoint, fixed marked parquet, tmux 충돌을 확인한다.
2. Frozen V2 checkpoint를 validation-only로 재평가해 context/scale 기준선을 저장한다.
3. Q0, Q1, Q2를 동일 순서와 예산으로 실행한다.
4. 실행 시작 후 V2 reference 완료와 Q0 첫 학습 진입까지만 확인한다.
5. 지속 polling은 하지 않고 사용자가 결과 확인을 요청할 때 artifact를 동기화한다.
6. 결과는 manifest, log, summary, histories, validation scale-wise, validation plot
   순서로 읽고 gate를 기록한 뒤에만 held-out artifact unlock 여부를 결정한다.

## 실행 명령어

```bash
ssh 5090
/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s inter_raw_q012_e50_0713 \
  "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_direct_raw_qty_q012_inter_seed42_e50_0713.sh"
```

## Preflight 결과

- local focused tests: `38 passed`
- local/5090 script syntax: `bash -n` 통과
- evaluator, training utility, run script local/5090 SHA256 일치
- 5090 GPU: `NVIDIA GeForce RTX 5090`, memory used `41 MiB`, utilization `0%`
- frozen V2 checkpoint와 marked parquet 존재 확인
- 5090 CUDA strict-load dry-run: `state_dict=strict_ok`
- tmux `inter_raw_q012_e50_0713` 이름 충돌 없음

## 실행 진입 확인

- V2 validation-only reference는 `2026-07-13 17:14:09 KST`에 정상 완료
- validation samples: `41,901`
- V2 NLL / marker NLL: `5.666520 / 0.991274`
- V2 log2 quantity MAE / normalized RPS: `0.588742 / 0.035283`
- V2 mark accuracy / mark MAE: `0.572492 / 0.487411`
- history count `<=4`: `31,952`
- Q0 data split samples: train `136,256`, validation `41,901`, test `41,344`
- Q0 train-only raw quantity mean / std: `6.84585607 / 55.01240343`
- Q0 CUDA process와 첫 학습 진입을 확인한 뒤 지속 monitoring은 중단

## 초기 해석 가설

- Q1은 short/constant context에서 `sqrt(eps)` scale collapse와 큰 normalized loss를
  다시 보일 가능성이 높다.
- Q2는 Q1의 numeric tail을 안정화해야 하지만, Instacart e1에서 관찰된 low-quantity
  bucket 악화가 Intermittent의 `1-9` 다수 구간에서도 나타나는지 확인해야 한다.
- Quantity 개선이 있더라도 marker/time safety를 실패하면 모델을 승격하지 않는다.
- 이번 e50 결과만으로 held-out test 결론이나 multi-seed 승격을 자동 확정하지 않는다.
