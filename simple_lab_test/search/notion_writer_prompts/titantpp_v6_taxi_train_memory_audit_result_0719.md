## 2026-07-17 | V6 Causal Series Memory

### Step 2. Taxi Train-Only Pre-Window Memory Audit

Notion location:

- parent: `5. Model Design Enhancement`
- page: `TitanTPP V6 Taxi Train-Only Pre-Window Memory Audit`
- related design: `TitanTPP V6 Causal Pre-Window Series Memory Hypothesis`

## 상태

완료. `2026-07-17 09:27:43 KST`부터 `09:28:24 KST`까지 5090에서 실행됐다.
Process exit code는 `0`이고 frozen audit gate는 `FAIL`이다. V6는 adapter 구현 전에
종료하며 Taxi incumbent는 V3b로 유지한다.

## 목적

현재 168시간 context보다 앞선 same-series event가 다음 mark, delta-time,
quantity에 안정적인 추가 신호를 제공하는지 fixed-split train row만으로 확인한다.

## Variant 계약

| 비교 | 입력 | 역할 |
| --- | --- | --- |
| Window-only probe | 현재 context summary | memory 없는 기준 |
| Memory-augmented probe | 동일 context + causal pre-window summary | 추가 이력의 증분 신호 |

Selection에서는 `M={16,32,64,128}`, `topk={4,8}`을 비교하고 strongest metric을
final primary로 동결한 뒤 마지막 train suffix를 한 번만 연다.

## 고정 조건

- Taxi fixed-split train parquet only
- lookback / max sequence: `168 / 256`
- train-internal chronological partition: `70% fit / 15% selection / 15% audit`
- metrics: marker CE, `log1p(delta_t)` MAE, `log2(quantity)` MAE
- bootstrap: series 단위 `2,000`회, seed `42`
- validation/test target read: `false`

## 실행 명령어

```bash
SOURCE_REVISION=6d7ed32ff65e91419a6d6ca6d7b28dbb8f73432c \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
TMUX_SESSION=titantpp_v6_taxi_memory_audit_0717 \
EXECUTION_SERVER=5090 \
bash simple_lab_test/search/scripts/run_titantpp_v6_taxi_train_memory_audit_0717.sh
```

## 결과

- source/loader/causal contract: PASS
- pre-window event 8개 이상 coverage: target `65.262%`, series `100%`
- selected candidate: `M=64`, `topk=4`, primary `marker_ce`
- final marker CE: `0.6235%` 개선, 95% CI `[-1.7265%, 2.9784%]`
- final time MAE: `2.4696%` 개선, 95% CI `[1.5236%, 3.5050%]`
- final quantity MAE: `0.2875%` 악화, 95% CI `[-0.8713%, 0.2630%]`
- failed gate: primary improvement `>=1%`, primary CI lower bound `>0`
- artifact 재계산: candidate ranking, coverage, target means, 2,000-series
  bootstrap와 gate가 저장값에 최대 오차 `4.3e-14`로 일치
- numerical caveat: marker logistic probe `max_iter=1000` convergence warning
- final decision: `close_v6_before_model_implementation`

`M64/topk4`는 model constant로 동결하지 않는다. V6a/V6b, CUDA/e1/e50,
multi-seed, held-out 단계는 열지 않고 Taxi V3b를 유지한다.

## Local Audit Trail

- artifact:
  `search_artifacts/model_enhancement_titantpp_v6_taxi_train_memory_audit_0717`
- completion check: `2026-07-19 09:19:17 KST`, 5090 단회 확인
- local sync: `19` regular files, `886,450` bytes, checksum mode
- independent validation: `PASS`, maximum absolute delta `4.3e-14`
- focused audit tests: `8 passed`
- Notion direct update/refetch: `2026-07-19 09:26:39 KST`, `4/4 PASS`
- 5080 sync: commit `e61c692`, tracked source `215` files and V6 artifact synced;
  source `7/7`, artifact `2/2`, dataset SHA checks passed
