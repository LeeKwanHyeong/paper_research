# TitanTPP Direct Raw Quantity Q0/Q1/Q2 Instacart Top-20 e1 Smoke 시작

`5. Model Design Enhancement` 아래에 세부 페이지
`TitanTPP Direct Raw Quantity Q0 Q1 Q2 Instacart Top-20 e1 Smoke`를 생성하거나
같은 제목의 기존 페이지를 갱신한다. 상위 history에는 제목 3
`Step 10. Q0/Q1/Q2 Instacart Top-20 e1 Integration Smoke`로 연결한다.

## 실험 시작 정보

- 시작 기록 시각: `2026-07-13 16:37 KST`
- 실행 서버: `5090` (`192.168.0.71`, user `leekwanhyeong`)
- 프로젝트 경로: `/home/leekwanhyeong/workspace/paper_research`
- Conda 환경: `ai_env`
- tmux session: `insta_raw_q012_e1_0713`
- artifact root: `search_artifacts/model_enhancement_direct_raw_qty_q012_insta_smoke_e1_0713`
- 상태: `prepared; 5090 preflight and launch pending`

## 실험 목적

5090 synthetic CUDA gate를 통과한 Q0/Q1/Q2 `direct_raw_qty`를 실제 Instacart
fixed-split DataLoader에 연결한다. 동일한 top-20 series, seed, e1 학습 예산에서
normalization mode만 바꾸고 train-only raw statistics, backward, checkpoint,
validation/test export와 scale-wise artifact 경로가 모두 동작하는지 확인한다.

이 단계는 e1 integration smoke다. Q0/Q1/Q2의 정확도 순위나 RevIN benefit을
판정하지 않는다. Held-out test artifact는 export/finite 여부만 확인하고 후보 선택에
사용하지 않는다.

## 고정 조건

| 항목 | 값 |
|---|---:|
| dataset | `insta_market_basket` top-20 |
| split | `fixed` |
| model / candidate | `TitanTPP / small_lmm` |
| epochs / seed | `1 / 42` |
| LR / batch | `1e-3 / 16` |
| lookback / max sequence | `10 weeks / 16` |
| decoder | `direct_raw_qty` |
| loss scope / loss mode | `target_only / hybrid` |
| marker objective | plain CE |
| magnitude embedding / weight | `8 / 1.0` |
| raw quantity loss weight | `0.25` |
| RevIN epsilon / shrinkage k | `1e-5 / 8` |
| center / affine / stat context | `mean / false / none` |
| selections | `best_val_nll,best_score,final` |

Variant는 Q0 `global`, Q1 `causal_revin`, Q2 `causal_shrinkage_revin`이다.
세 변형은 별도 artifact 디렉터리에 기록하며 한 변형 실패가 나머지 실행을 취소하지
않는다. Raw global mean/variance/std와 effective floor는 각 run에서 같은 Instacart
fixed train split만 사용해 계산하며 validation/test row를 통계에 포함하지 않는다.

## 실험 계획

1. 5090 GPU, `ai_env`, split parquet와 source 동기화 상태를 확인한다.
2. Q0, Q1, Q2를 같은 설정으로 순차 실행한다.
3. 각 run의 train-only raw statistics와 loader sample count가 동일한지 확인한다.
4. Epoch 1 backward, best/final checkpoint, validation/test metric export를 확인한다.
5. Manifest, log, summary, test summary, history, scale-wise, plot 순서로 artifact를 읽는다.
6. Integration gate 결과만 기록하고 e1 metric으로 모델을 선택하지 않는다.

## 실행 명령어

```bash
ssh 5090
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s insta_raw_q012_e1_0713 \
  "bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_direct_raw_qty_q012_insta_smoke_e1_0713.sh"
```

## Acceptance Gate

- Q0/Q1/Q2 모두 exit code 0과 `status=success`
- NaN, Inf, Traceback, CUDA runtime error 없음
- 세 run의 split, series/sample count, seed, batch, lookback, max sequence가 동일
- `magnitude_stats_source_split=train`이며 raw global moments/effective floor가 세 run에서 동일
- Epoch 1 train loss와 validation metric이 finite하고 raw magnitude diagnostics가 기록됨
- best validation NLL, best score, final checkpoint와 history가 생성됨
- validation/test summary, scale-wise metric, report/plot 경로가 생성됨

## 해석 제한

Top-20 e1은 데이터 경로와 artifact 계약 확인용이다. 수치가 finite하더라도 convergence,
Q0 대비 RevIN benefit, V2 대비 quantity 개선을 의미하지 않는다. 성능 판정은 이후
Intermittent seed-42 e50 validation-only screening의 사전 gate를 따른다.
