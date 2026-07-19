## 2026-07-17 | V6 Causal Series Memory

### Step 2. Taxi Train-Only Pre-Window Memory Audit

Notion location:

- parent: `5. Model Design Enhancement`
- page title: `TitanTPP V6 Taxi Train-Only Pre-Window Memory Audit`
- related design: `TitanTPP V6 Causal Pre-Window Series Memory Hypothesis`

## 상태

완료. `2026-07-17 09:27:43 KST`에 시작해 `09:28:24 KST`에 process exit code
`0`으로 종료됐다. Frozen final gate는 `FAIL`이며 V6는 adapter 구현 전에 종료한다.
Taxi incumbent는 V3b로 유지한다.

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
SOURCE_REVISION=6d7ed32ff65e91419a6d6ca6d7b28dbb8f73432c \
PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
TMUX_SESSION=titantpp_v6_taxi_memory_audit_0717 \
EXECUTION_SERVER=5090 \
bash simple_lab_test/search/scripts/run_titantpp_v6_taxi_train_memory_audit_0717.sh
```

## 결과

- artifact 상태: completed, finite target errors, NaN/Traceback 없음
- data/loader: source quality와 exact loader target·causal ordering gate 통과
- held-out: validation/test target read `false`
- coverage: pre-window event 8개 이상 target `65.262%`, series `100%`
- selection: `M=64`, `topk=4`, primary `marker_ce`
- final marker CE: `0.6235%` 개선, 95% CI `[-1.7265%, 2.9784%]`
- final time MAE: `2.4696%` 개선, 95% CI `[1.5236%, 3.5050%]`
- final quantity MAE: `0.2875%` 악화, 95% CI `[-0.8713%, 0.2630%]`
- failed checks: primary improvement `>=1%`, primary bootstrap CI lower bound `>0`
- 판정: `M64/topk4`를 model constant로 동결하지 않고 V6a/V6b 구현과
  CUDA/e1/e50/multi-seed/held-out을 열지 않는다. Taxi V3b를 유지한다.
- caveat: marker logistic probe에서 `max_iter=1000` convergence warning이
  발생했다. Final suffix를 본 뒤 solver를 바꾸는 rerun은 열지 않는다.

## Local Audit Trail

### Source Sync Record

- source commit: `6d7ed32ff65e91419a6d6ca6d7b28dbb8f73432c`
- sync target: `5090:/home/leekwanhyeong/workspace/paper_research`
- synced files: `7`
- local/remote SHA-256 match: `7/7 PASS`
- tmux launched: `true`
- actual start: `2026-07-17 09:27:41 KST`
- preflight status: `PASS`
- source manifest SHA-256:
  `85365fc8435b4eda3dd11e059f3e5312eea90dd244e6fbee34dd8651ab74e107`
- Notion direct update: `start and final states completed and refetched`

| File | SHA-256 |
| --- | --- |
| `.agents/results/architecture/adr-titantpp-v6-causal-pre-window-series-memory.md` | `e56b05aff0c7827defeda95f75e84595a6dea614602db3af2789c64556266b07` |
| `.agents/results/architecture/titantpp-model-status-baseline-registry.md` | `de9ce2ac9e2b2e2e38e07e97187d72002a05b0327e10d6c12703f9fe373f19db` |
| `simple_lab_test/search/analyze_taxi_pre_window_memory_audit.py` | `bf1c4b33c2e785da612d91e0bb16e0927dddc1681a99a244ce6bc0ff7dfd7ced` |
| `simple_lab_test/search/model_enhancement_strategy.md` | `ea79fab05f2db03b6e29900a4220f26a9d099c300770d91cdc52369f61baf87d` |
| `simple_lab_test/search/scripts/run_titantpp_v6_taxi_train_memory_audit_0717.sh` | `a2a1d94baaa2a176b9da0163aec2aaf8484e49ba279062f7d2ce62d85571109b` |
| `simple_lab_test/search/search_experiment_guide.md` | `d4bb1a41a90afc0a69b8cc1bfc7ff3452c51f5897d1424c3b23936e9e6805a8a` |
| `simple_lab_test/search/tests/test_taxi_pre_window_memory_audit.py` | `9839ff742497b8d1290e9672aec3534718e629b09fa6a90a4b4e2e67d94e5ca0` |

### Preflight And Initial Entry

- dependency: Python `3.12.13`, matplotlib `3.10.8`, numpy `2.1.3`, polars
  `1.39.3`, sklearn `1.8.0`
- runner: executable and `bash -n PASS`
- dataset: train-only `38,524` rows, `131` series, marks `[0,1,2,3]`, SHA-256
  `0055229740f3f5b612ff8a2a256b1726008918a2ccb453c6bd66909c48ab2cb3`
- dataset quality: required null `0`, duplicate part/seq `0`, non-train row `0`,
  invalid mark `0`, non-positive/non-finite quantity `0`, held-out read `false`
- launch guard before start: tmux session absent, output directory empty
- source revision gate: remote `.git`은 없는 rsync workspace이므로 Git HEAD가
  아니라 동기화 파일 `7/7` SHA-256 일치로 판정
- source manifest:
  `search_artifacts/model_enhancement_titantpp_v6_taxi_train_memory_audit_0717/source_sync_manifest.json`
- initial check: tmux active, `logs/audit.log` 생성, `status.json` 생성 전
- initial log: targets `38,393`, pre-window count `>=8` target share `65.26%`,
  eligible series share `100%`까지 coverage decoding 진입
- sklearn logistic probe의 `max_iter=1000` convergence warning이 초기 화면에
  관찰됐으나 프로세스는 계속 실행 중이다. 최종 artifact 분석에서 수렴 상태와
  acceptance 영향 여부를 확인한다.

### Notion Start-State Update Result

- reflected at: `2026-07-17 09:30:49 KST`
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
  `실험 중 / 2026-07-17 09:27:41 KST / 5090 tmux`, empty result body,
  V2 and Taxi V3b incumbent unchanged, and adapter implementation still gated.

### Completion And Artifact Validation

- completion check: `2026-07-19 09:19:17 KST`, 단회 확인
- remote completion: `2026-07-17 09:28:24 KST`, exit code `0`
- local sync: artifact `19` regular files, `886,450` bytes, rsync checksum mode
- expected audit artifact: manifest, log, summary, quality, coverage, candidate,
  final target/series metrics, gate, report, plots 모두 존재
- independent recomputation: candidate `M64/topk4`, target means, coverage,
  2,000-series bootstrap, gate 재현 `PASS`
- maximum saved/recomputed absolute delta: `4.3e-14`
- focused audit tests: `8 passed`
- final decision: `close_v6_before_model_implementation`

### Final Notion Result Update

- reflected at: `2026-07-19 09:26:39 KST`
- detail page: completed, gate `FAIL`, final metrics and V6 closure verified
- parent history: Step 2 status/result verified under `5. Model Design Enhancement`
- related design: V6 closed before adapter implementation
- strategy registry: V6 `종료`, Taxi V3b retained
- refetch verification: `4/4 PASS`

### 5080 Source And Artifact Sync

- synced at: `2026-07-19 09:33:08 KST`
- source commit: `e61c6925288953908de81a999de3f1fdbc7931f0`
- destination: `5080:/home/leekwanhyeong/workspace/paper_research`
- tracked source: `215` files, checksum mode, no delete
- changed source checksum verification: `7/7 PASS`
- V6 artifact: `19` regular files, `886,450` bytes
- artifact manifest/summary checksum verification: `2/2 PASS`
- Taxi train parquet SHA-256:
  `0055229740f3f5b612ff8a2a256b1726008918a2ccb453c6bd66909c48ab2cb3`
- runtime: Python `3.12.13`, PyTorch `2.11.0+cu130`, CUDA `true`, tmux `3.4`
- remote workspace has no Git metadata; future source gate uses synced commit and
  file SHA-256 manifest
