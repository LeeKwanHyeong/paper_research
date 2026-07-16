# TitanTPP V4 Taxi 2x2 Seed-42 e50 Screening

Notion의 `5. Model Design Enhancement` 아래 제목 2
`2026-07-16 | V4 Mark-Conditioned Time Head`, 제목 3
`Step 4. Taxi V2/V3b/V4a/V4b Validation-Only Screening`으로 정리한다.

## 상태

- 상태: `완료 - V4 미승격`
- 실행 서버 / tmux: `5090 / titantpp_v4_taxi_2x2_e50_0716`
- 실행 시작 시각: `2026-07-16 18:41:44 KST`
- 실행 종료 시각: `2026-07-16 19:29:29 KST`
- source revision: `c5e9cca4241a5579ba0af655c884d6692484ba5a`

## 목적

- V4 time head의 효과를 V2와 Taxi V3b 각각에서 분리해 확인한다.
- validation 기준으로 V4b의 multi-seed 승격 여부를 결정한다.
- 이 단계에서는 held-out test metric을 생성하거나 읽지 않는다.

## Factorial 계약

| Variant | Value head | Quantity-mark gradient | Time head | 역할 |
| --- | --- | --- | --- | --- |
| V2 | shared | coupled | shared | 공통 control |
| V3b | mark-conditioned experts | detached | shared | Taxi value-head control |
| V4a | shared | coupled | mark-conditioned | time-head 단독 효과 |
| V4b | mark-conditioned experts | detached | mark-conditioned | Taxi 승격 후보 |

## 고정 조건

- Taxi fixed split, `mid_lmm`, seed `42`, e50
- batch `128`, lookback `168`, max sequence `256`, learning rate `1e-3`
- residual input, hybrid loss, target-only, plain marker CE
- strict reproducibility, `best_val_nll` checkpoint
- `evaluation_scope=validation_only`
- source checksum `20/20` 일치, CUDA/Instacart integration gate `PASS`
- loader sample 계약: train/validation/test `38,393/8,268/8,327`
- V4 pair별 time NLL `0.5%` 이상 개선
- total NLL `0.5%`, DT MAE `1%`, marker NLL `2%`, mark accuracy `-0.25%p`,
  quantity MAE `5%` guardrail

## 실행 명령어

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s titantpp_v4_taxi_2x2_e50_0716 \
  "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  SOURCE_REVISION=<sync_commit_sha> \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_v4_taxi_2x2_seed42_e50_0716.sh"'
```

## 결과

- artifact 무결성: 네 Variant 모두 e50 완료, validation `8,268`건 기준
  history/scale/class/confusion 집계 일치. held-out test 산출물 없음.
- V4a vs V2: time NLL `0.415%` 개선으로 목표 `0.5%` 미달, DT MAE
  `4.172%` 악화로 guardrail 실패.
- V4b vs V3b: time NLL `0.321%` 개선으로 목표 미달. 나머지 guardrail은
  통과했지만 primary gate 실패.
- 50 epoch 전체에서 time NLL `0.5%` gate를 만족한 epoch는 두 pair 모두
  `10/50`뿐이며, 마지막 10 epoch에서는 V4a `2/10`, V4b `1/10`이었다.
  mark-conditioned time head 효과가 작고 checkpoint에 민감했다.
- V4a quantity MAE `26.66%` 개선의 `76.77%`, V4b `12.45%` 개선의
  `98.11%`가 `1000-9999` 구간에서 발생했다. 전 scale 공통 개선은 아니다.
- V4a의 mark accuracy `+0.351%p`는 mark 0 중심이며 balanced accuracy는
  `-0.055%p`였다. V4b는 mark accuracy `-0.169%p`, mark 2 recall
  `-3.565%p`로 이동했다.
- Plot에서도 late time NLL의 지속적 분리는 확인되지 않았다. V3b/V4b의
  낮은 quantity MAE는 time head보다 기존 value head 효과가 지배적이었다.
- 최종 결정: 공통 기준선은 V2, Taxi 확정 모델은 V3b 유지. V4a/V4b
  multi-seed와 held-out test는 진행하지 않고 V4를 미승격 종료한다.

## Notion 반영

- 반영 시각: `2026-07-17 06:21:24 KST`
- 대상: `5. Model Design Enhancement`
- URL: `https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`
- `2026-07-16 | V4 Mark-Conditioned Time Head` 아래 Step 1-4를 직접 추가했다.
- 재조회 결과 날짜 섹션 1개, Step 4개, Factorial/selected-checkpoint 표,
  V4a/V4b gate, history/scale/class 해석, held-out 잠금, 최종 판정이 모두
  확인됐다.
- selected-checkpoint 표의 Markdown 정렬 행이 데이터 행으로 표시된 문제는
  header table로 교체한 뒤 재조회했으며 전체 검증 항목이 통과했다.
