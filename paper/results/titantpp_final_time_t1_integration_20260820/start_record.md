# TitanTPP Final Time Head + T1 통합 검증 시작 기록

## 상태

- 상태: 5080 tmux 실행 중
- 준비 시각: 2026-08-20 13:24:33 KST
- 실험 시작 시각: 2026-08-20 13:25:37 KST
- 실행 서버: 5080
- tmux: `titan_final_time_t1_e300_0820`
- source revision: `387b1f515e23a75cb1402ffb14e4673a34ffb740`
- artifact: `search_artifacts/count_aware_final_time_t1_integration_e300_20260820`
- evaluation scope: validation only
- held-out test: 사용하지 않음

## 목적

Train-only audit에서 선택한 slope-free H3 time head를 Hard-LMM 및 T1 quantity loss와
결합한다. Fresh H0 control과 비교해 time likelihood를 보존하면서 quantity body와
tail을 손상하지 않는지 확인한다.

## Factorial 계약

| Variant | Titan memory | Quantity loss | Time head | 역할 |
| --- | --- | --- | --- | --- |
| F0 | Hard-LMM | T1 tail-shared | H0 scaled exact RMTPP | matched control |
| F1 | Hard-LMM | T1 tail-shared | H3 log-normal duration | candidate |

두 run은 time-head family 외 dataset, split, source revision, seed, model dimension,
optimizer, batch order, early stopping과 checkpoint selection을 동일하게 사용한다.

## 고정 조건

- dataset: Intermittent frozen 5,000 fixed split
- seed: 42
- epochs: maximum 300, minimum 40, patience 40
- checkpoint: minimum validation joint objective
- learning rate / batch size: `1e-3` / `128`
- lookback / maximum sequence length: `520주` / `256`
- hidden dimension: 64
- T1 lambda: `0.09111380335463036`
- H3 time scale: train median `3.0`
- H3 initial location / sigma: train log-scaled mean/std
- gradient route: time과 quantity 모두 shared Hard-LMM encoder 학습

## 사전 검증

- 5080 CUDA focused tests: `42 passed`
- Intermittent CUDA 2-batch forward/backward 및 checkpoint: 통과
- Instacart top-20 e1 actual-data artifact: 통과
- H3 density, Jacobian, survival/median, extreme-duration gradient: finite
- held-out test artifact: 없음

초기 gradient clipping 비율은 Intermittent 2-batch에서 `100%`, Instacart e1에서
`97.70%`였다. 고정 상수 선택에는 사용하지 않으며, e300 history에서 clipping 비율이
감소하는지 함께 확인한다.

## 실행 명령어

```bash
SOURCE_REVISION=387b1f515e23a75cb1402ffb14e4673a34ffb740 \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/home/leekwanhyeong/miniconda3/envs/ai_env/bin/python \
bash simple_lab_test/search/scripts/run_count_aware_final_time_t1_integration_e300_20260820.sh
```

## 결과
