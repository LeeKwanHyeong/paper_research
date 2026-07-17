## 2026-07-17 | V6 Causal Series Memory

### Step 2. Taxi Train-Only Pre-Window Memory Audit

Notion location:

- parent: `5. Model Design Enhancement`
- page title: `TitanTPP V6 Taxi Train-Only Pre-Window Memory Audit`
- related design: `TitanTPP V6 Causal Pre-Window Series Memory Hypothesis`

## 상태

준비 중. 시작 기록 시각은 `2026-07-17 09:17:13 KST`이며 실행 위치는
`5090 / titantpp_v6_taxi_memory_audit_0717`이다. Audit 구현과 5090 source
동기화는 완료했지만 tmux 실행은 아직 시작하지 않았다.

## 목적

Taxi의 현재 168시간 context보다 앞선 same-series event가 다음 mark, delta-time,
quantity 예측에 추가 정보를 제공하는지 fixed-split train row만으로 확인한다.
Audit이 통과할 때만 V6 memory budget과 top-k를 고정하고 adapter 구현을 연다.

## Variant 계약

| 비교 | 입력 | 역할 |
| --- | --- | --- |
| Window-only probe | 현재 168시간 context summary | 추가 memory가 없는 기준 |
| Memory-augmented probe | 동일 context summary + causal pre-window retrieval | 추가 과거 정보의 증분 신호 확인 |

Memory 후보는 `M={16,32,64,128}`과 `topk={4,8}`이다. 두 probe는 같은 target과
plain linear estimator를 사용하며 memory 입력만 다르다.

## 고정 조건

- dataset: `yellow_trip_hourly_train.parquet`, fixed-split train row only
- server / tmux: `5090 / titantpp_v6_taxi_memory_audit_0717`
- lookback / max sequence: `168 / 256`
- train-internal partition: series별 시간순 `70% fit / 15% selection / 15% audit`
- metrics: marker CE, `log1p(delta_t)` MAE, `log2(quantity)` MAE
- memory ordering: `memory index < context start <= context end < target index`
- artifact: `search_artifacts/model_enhancement_titantpp_v6_taxi_train_memory_audit_0717`
- validation/test row와 metric은 읽지 않음

## 실행 명령어

```bash
SOURCE_REVISION=<synced_commit> \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
TMUX_SESSION=titantpp_v6_taxi_memory_audit_0717 \
bash simple_lab_test/search/scripts/run_titantpp_v6_taxi_train_memory_audit_0717.sh
```

## 결과

## Local Audit Trail

### Source Sync Record

- source commit: `6d7ed32ff65e91419a6d6ca6d7b28dbb8f73432c`
- sync target: `5090:/home/leekwanhyeong/workspace/paper_research`
- synced files: `7`
- local/remote SHA-256 match: `7/7 PASS`
- tmux launched: `false`
- Notion direct update: `completed and refetched`

| File | SHA-256 |
| --- | --- |
| `.agents/results/architecture/adr-titantpp-v6-causal-pre-window-series-memory.md` | `e56b05aff0c7827defeda95f75e84595a6dea614602db3af2789c64556266b07` |
| `.agents/results/architecture/titantpp-model-status-baseline-registry.md` | `de9ce2ac9e2b2e2e38e07e97187d72002a05b0327e10d6c12703f9fe373f19db` |
| `simple_lab_test/search/analyze_taxi_pre_window_memory_audit.py` | `bf1c4b33c2e785da612d91e0bb16e0927dddc1681a99a244ce6bc0ff7dfd7ced` |
| `simple_lab_test/search/model_enhancement_strategy.md` | `ea79fab05f2db03b6e29900a4220f26a9d099c300770d91cdc52369f61baf87d` |
| `simple_lab_test/search/scripts/run_titantpp_v6_taxi_train_memory_audit_0717.sh` | `a2a1d94baaa2a176b9da0163aec2aaf8484e49ba279062f7d2ce62d85571109b` |
| `simple_lab_test/search/search_experiment_guide.md` | `d4bb1a41a90afc0a69b8cc1bfc7ff3452c51f5897d1424c3b23936e9e6805a8a` |
| `simple_lab_test/search/tests/test_taxi_pre_window_memory_audit.py` | `9839ff742497b8d1290e9672aec3534718e629b09fa6a90a4b4e2e67d94e5ca0` |

### Notion Direct Update Result

- reflected at: `2026-07-17 09:19:05 KST`
- parent: `5. Model Design Enhancement`
  (`https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`)
- detail page: `TitanTPP V6 Taxi Train-Only Pre-Window Memory Audit`
  (`https://app.notion.com/p/3a0bbe4056138182bae2c5241cb4cea8`)
- related design: `TitanTPP V6 Causal Pre-Window Series Memory Hypothesis`
  (`https://app.notion.com/p/39fbbe4056138118871fcd18c6b31174`)
- strategy: `TitanTPP Model Enhancement Strategy`
  (`https://app.notion.com/p/394bbe4056138046bf3bfbc6f4c8c31a`)
- refetch verification:
  `5. Model Design Enhancement > 2026-07-17 > Step 2` placement,
  `준비 중 / 5090 sync 완료 / tmux 미실행`, empty result body,
  V2 and Taxi V3b incumbent unchanged, and adapter implementation still gated.
