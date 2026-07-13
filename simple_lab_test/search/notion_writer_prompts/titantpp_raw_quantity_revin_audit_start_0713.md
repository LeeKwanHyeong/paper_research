# TitanTPP Raw-Quantity RevIN Train-Only Audit And Q2 Constants Freeze

## Notion 위치

- 상위 페이지: `5. Model Design Enhancement`
- 날짜 섹션: `2026-07-13`
- 세부 단계: `Intermittent train-only raw history, variance, tail audit and Q2 constants freeze`
- 같은 제목의 상세 페이지가 있으면 새로 만들지 않고 기존 페이지를 업데이트한다.

## 기준 시각

- 실험 시작 시각: `2026-07-13 15:10:00 KST`
- 실행 서버: `5090` (`192.168.0.71`)
- tmux session: `titantpp_raw_revin_audit_0713`
- conda env: `ai_env`
- project root: `/home/leekwanhyeong/workspace/paper_research`

## 상태

- `in progress`
- 로컬 focused formula test: `5 passed`
- 5090 `ai_env`에는 `pytest`가 없어 원격 focused test는 미실행했다. 새 dependency는 설치하지 않고 실제 audit entrypoint로 런타임을 검증한다.

## 실험 목적

- 기존 log-domain audit 결과를 raw-quantity RevIN 실패 근거로 사용하지 않는다.
- Intermittent fixed-split train 데이터에서 raw 수량의 history length, local variance, level shift, tail concentration을 다시 측정한다.
- Q0 raw/global control과 Q1 causal masked RevIN의 정규화 안정성을 참조값으로 계산한다.
- Q2 causal moment shrinkage의 `k` 후보를 train-only gate로 비교하고 구현에 사용할 상수를 고정하거나, 통과 후보가 없으면 Q2 구현을 차단한다.
- 이 단계는 모델 정확도나 RevIN 효과를 주장하는 실험이 아니라 normalization feasibility와 상수 선택을 위한 audit다.

## 실험 계획

- dataset: `sample_data/head_office/marked_target_with_split.parquet`
- split scope: fixed `train` only
- held-out validation/test read: `false`
- context contract: `RMTPPWeekLookbackDataset`과 동일한 train-target count 및 context-length distribution
- lookback: `52 weeks`
- max sequence length: `16` including target, maximum context length `15`
- magnitude domain: raw `demand_qty`; log transform 없음
- Q0: train-global raw mean/std control
- Q1: causal masked history mean/std, `eps=1e-5`
- Q2: causal history/global second-moment shrinkage
- Q2 `k` candidates: `{1, 2, 4, 8, 16}`
- raw scale floor: `max(0.001 * train_global_raw_std, 1e-4)`
- Q2 eligibility: finite values, one-event median scale at least `0.5 * global std`, median alpha at least `0.25`, normalized target p99 and `|u| > 3` share no worse than Q0
- selection rule: eligible 후보 중 normalized target absolute p99가 가장 낮은 값, 동률이면 더 작은 `k`

## 실행 명령어

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d -s titantpp_raw_revin_audit_0713 "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python TMUX_SESSION=titantpp_raw_revin_audit_0713 bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_raw_quantity_revin_audit_0713.sh"'
```

## 결과 artifact

- server: `/home/leekwanhyeong/workspace/paper_research/search_artifacts/model_enhancement_raw_quantity_revin_audit_0713`
- local: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/model_enhancement_raw_quantity_revin_audit_0713`
- reading order: `audit_manifest.json` -> `logs/audit.log` -> `audit_summary.json` -> detailed CSV/parquet -> `report.md` -> plots

## 결과 작성란

- audit 완료 후 시작·종료 시각과 Q2 freeze/block 결정을 업데이트한다.
- raw tail, short-history/zero-variance 비율, Q0/Q1 normalized-target tail, Q2 candidate gate를 구분해 기록한다.
- 상수 선택은 train-only evidence로만 표현하고 predictive-performance evidence로 확대 해석하지 않는다.

## 다음 판단

- Q2 constants가 `frozen`이면 동일 split, initialization, budget, loss coefficient를 유지한 Q0/Q1/Q2 모델 구현과 CUDA model-test로 이동한다.
- Q2 constants가 `blocked`이면 gate 실패 원인을 먼저 분석하고 임의의 `k`로 모델 실험을 시작하지 않는다.
