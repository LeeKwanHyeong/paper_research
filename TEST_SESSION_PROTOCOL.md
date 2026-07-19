# TPP Experiment Session Protocol

이 문서는 `paper_research` 프로젝트에서 다른 Codex 세션, Data Analyst 세션,
Notion Writer 세션이 같은 기준으로 실험을 실행하고 기록하기 위한 공통 기준 문서다.

핵심 목적은 세 가지다.

- GPU 서버에서 실험을 같은 방식으로 시작한다.
- 실험 시작 시점, 목적, 계획, 결과 업데이트를 Notion에 빠르게 남긴다.
- artifact를 로컬로 내려받은 뒤 같은 순서로 결과를 읽는다.

## 0. Session Operating Rules

모든 `paper_research` 세션은 작고 정확한 변경을 기본값으로 둔다. 요청 범위를
넓히지 말고, 실험 실행/분석/문서화에 필요한 최소 변경만 수행한다.

공통 원칙:

- 작업 시작 전 objective, constraints, assumptions를 짧게 확인한다.
- 관련 파일, 로그, artifact를 먼저 읽고 실제 change point 또는 분석 기준을 찾는다.
- 비자명한 작업은 편집이나 실행 전에 짧은 단계별 계획을 남긴다.
- 확인된 사실과 추론을 구분한다. 코드, 로그, CSV, test 결과로 확인한 내용만
  confirmed로 다루고, 해석은 inferred로 분리한다.
- dirty worktree에서는 기존 사용자 변경을 되돌리지 않는다. 관련 없는 변경은 무시한다.
- 코드 변경은 최소 diff로 제한하고, 기존 architecture, naming, CLI option,
  artifact layout을 유지한다.
- 새 dependency, config, env var, service, abstraction은 현재 요청을 해결하는 데
  필요할 때만 추가한다.
- 검증은 focused test 또는 targeted command로 수행한다. 실행하지 못한 검증은
  Notion/최종 보고에 명시한다.

세션별 역할:

| Session | Primary responsibility | Required behavior |
| --- | --- | --- |
| Main Developer | runner, model, data path 구현 | entrypoint에서 실제 execution path를 추적하고 최소 코드 변경 후 smoke/targeted test를 수행 |
| Data Analyst | artifact 분석, metric 해석 | manifest, logs, summary/test/scale-wise 순서로 읽고 validation/test 방향과 NLL split을 분리 |
| Model Enhancement | TitanTPP 구조 강화 | 현재 구현과 target architecture를 구분하고 R0/L0/S0 baseline 및 decision rule을 고정 |
| Notion Writer | 실험 계획/결과 문서화 | 10번/11번 템플릿을 따르고 과장 없이 완료/진행/위험/다음 액션을 정리 |
| Concept Explainer / Paper Writer | 교수님 보고/논문화 | 실험 근거와 해석을 분리하고 TitanTPP 단독 우월성을 과도하게 주장하지 않음 |

상태 보고는 아래 분류를 사용한다.

| Status | Meaning |
| --- | --- |
| implemented | 코드/실험/문서가 실제로 완료되고 artifact 또는 테스트로 확인됨 |
| partially implemented | 일부 경로는 동작하지만 전체 protocol 또는 모든 dataset/seed가 끝나지 않음 |
| scaffolded | 구조나 문서만 준비되고 실제 실험/통합 검증은 아직 없음 |
| in progress | 서버/tmux에서 실행 중이거나 분석이 진행 중 |
| blocked | 사용자 입력, 서버 상태, missing artifact 등으로 다음 단계 진행 불가 |
| not implemented | 코드 또는 실험이 아직 존재하지 않음 |

보고 형식:

- 완료한 일, 확인 근거, 남은 위험, 다음 액션을 짧게 정리한다.
- 디버깅은 구체적 hypothesis에서 시작하고 실제 execution path와 로그로 확인한다.
- review는 findings를 먼저 쓰고 severity 순서로 정리한다.
- Notion 결과 페이지에는 실험 종료 시각, artifact path, 완료 여부, NaN/Traceback,
  best setting, validation/test 일치 여부, NLL split, quantity/scale-wise 해석을 포함한다.

## 1. Server Priority

기본 실행 서버는 `5090`이다. `5090`이 사용 중이거나 보조 실험을 병렬로 돌릴 때만
`5080`을 사용한다.

Current user override (2026-07-19, supersedes 2026-07-10):

- 별도 지시가 있기 전까지 Model Enhancement 작업의 source sync, smoke,
  screening, long-epoch는 `5080`에서 실행한다.
- `5090`은 다른 작업에 사용 중이므로 사용자가 명시적으로 요청하기 전에는 새
  명령이나 실험을 시작하지 않는다.
- 로컬 커밋을 먼저 만든 뒤 필요한 tracked source와 실험 artifact만
  `5080:~/workspace/paper_research`에 checksum 동기화한다. `--delete`는 사용하지
  않는다.
- 실험 시작 후 초기 설정, GPU process, 첫 학습 진입까지만 확인하고 지속 polling은
  하지 않는다. 결과 확인은 사용자가 요청했을 때 수행한다.

| Priority | Server | Purpose |
| --- | --- | --- |
| 1 (temporary override) | `ssh 5080` | Model Enhancement source sync와 후속 실험 |
| 2 (explicit request only) | `ssh 5090` | 기존 본실험 서버; 현재 다른 작업 사용 중 |

공통 원칙:

- conda environment는 `ai_env`를 사용한다.
- 장시간 학습은 반드시 `tmux` 안에서 실행한다.
- artifact는 서버의 `~/workspace/paper_research/search_artifacts/` 아래에 저장한다.
- 로컬 분석은 `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/`로 내려받은 뒤 진행한다.

## 2. Standard Server Entry

### 2.1 Primary: 5090

확인된 endpoint와 실행 경로:

| Item | Value |
| --- | --- |
| SSH alias | `5090` |
| Host/IP | `192.168.0.71` |
| SSH port | `22` |
| User | `leekwanhyeong` |
| Project | `/home/leekwanhyeong/workspace/paper_research` |
| conda env | `/opt/miniconda3/envs/ai_env` |
| Python | `/opt/miniconda3/envs/ai_env/bin/python` |
| tmux | `/opt/miniconda3/envs/ai_env/bin/tmux` |

비대화형 SSH에서는 conda와 tmux가 `PATH`에 없을 수 있으므로 장시간 실험 시작 시
위 절대 경로를 사용한다.

```bash
ssh 5090
conda activate ai_env
cd ~/workspace/paper_research
tmux new-session -s <session_name>
```

로컬에서 비대화형으로 시작하는 표준 형식:

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d -s <session_name> \
  "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  bash /home/leekwanhyeong/workspace/paper_research/<script_path>"'
```

이미 세션이 있으면:

```bash
ssh 5090
conda activate ai_env
tmux attach -t <session_name>
```

tmux 안에서도 환경이 풀려 있으면 다시 실행한다.

```bash
conda activate ai_env
cd ~/workspace/paper_research
```

### 2.2 Secondary: 5080

```bash
ssh 5080
conda activate ai_env
cd ~/workspace/paper_research
tmux new-session -s <session_name>
```

이미 세션이 있으면:

```bash
ssh 5080
conda activate ai_env
tmux attach -t <session_name>
```

## 3. Session Naming Rule

세션 이름은 나중에 `tmux ls`만 봐도 목적을 알 수 있게 만든다.

```text
<dataset_or_scope>_<experiment_keyword>_<epochs>_<date>
```

예시:

```text
insta_lr_sensitivity_e50_0705
inter_yellow_lr_sensitivity_e50_0704
titantpp_stabilization_e100_0705
fixed_split_all_models_e800_0706
```

## 4. Long-Running Command Template

실험은 가능하면 `logs/run.log`에 tee로 남긴다.

```bash
mkdir -p ~/workspace/paper_research/search_artifacts/<experiment_name>/logs

python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/<experiment_name> \
  --datasets <dataset_list> \
  --models <model_list> \
  --epochs <epochs> \
  --seeds <seed_list> \
  --lr <lr> \
  --batch-size <batch_size> \
  --max-seq-len <max_seq_len> \
  --eval-selections best_val_nll,best_score,final \
  --split-mode fixed \
  --value-head-activation identity \
  --device cuda \
  2>&1 | tee -a ~/workspace/paper_research/search_artifacts/<experiment_name>/logs/run.log
```

실험 종료 후 tmux shell을 유지하고 싶으면 마지막에 아래를 붙여도 된다.

```bash
exec bash
```

단, `exec bash`는 파이프라인 뒤에 바로 붙이면 의도와 다르게 동작할 수 있다.
필요하면 아래처럼 subshell로 묶어서 쓴다.

```bash
(
  python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
    --base-dir ~/workspace/paper_research/search_artifacts/<experiment_name> \
    --datasets <dataset_list> \
    --models <model_list> \
    --epochs <epochs> \
    --seeds <seed_list> \
    --lr <lr> \
    --batch-size <batch_size> \
    --max-seq-len <max_seq_len> \
    --eval-selections best_val_nll,best_score,final \
    --split-mode fixed \
    --value-head-activation identity \
    --device cuda \
    2>&1 | tee -a ~/workspace/paper_research/search_artifacts/<experiment_name>/logs/run.log
  exec bash
)
```

### 4.1 Strict Deterministic Launch

동일 seed 구조 비교나 exact-reproduction gate에는 long-epoch의 명시적 strict
모드를 사용한다. 세 환경 변수는 Python 프로세스 안에서 설정하지 말고 launcher의
`env`에 넣어야 한다. `SOURCE_REVISION`은 선택한 실행 서버에 checksum 동기화한
로컬 full commit SHA를 사용한다.

```bash
env \
  PYTHONHASHSEED=42 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  SOURCE_REVISION=<full_commit_sha> \
  /opt/miniconda3/envs/ai_env/bin/python \
  ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/<experiment_name> \
  --datasets intermittent \
  --models titantpp \
  --epochs <epochs> \
  --seeds 42 \
  --split-mode fixed \
  --reproducibility-mode strict \
  --device cuda
```

strict 모드 계약:

- `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`, `SOURCE_REVISION` 누락은 실행 전
  failure로 처리한다.
- Torch deterministic algorithms와 deterministic cuDNN을 켜고 cuDNN benchmark를
  끈다. deterministic operation 오류는 warning으로 낮추지 않는다.
- train shuffle은 run seed 전용 `torch.Generator`와 `num_workers=0`을 사용하며,
  resume checkpoint에 loader generator state를 함께 저장한다.
- grouped series는 `oper_part_no` 오름차순으로 고정한다.
- root/run manifest에는 source revision, PyTorch/CUDA/GPU, deterministic flags,
  loader seed와 source dataset SHA-256을 기록한다.
- summary와 checkpoint에는 selection별 canonical tensor-state SHA-256을 기록한다.
- strict artifact는 기존 standard cache와 섞이지 않도록 run path의
  `repro_strict/` 아래에 저장한다.

## 5. Current Dataset Names

CLI에서 사용하는 dataset 이름은 아래 값으로 통일한다.

| CLI dataset | File contract |
| --- | --- |
| `intermittent` | `sample_data/head_office/marked_target_with_split.parquet` or marked target source |
| `yellow_trip_hourly` | `sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet` |
| `insta_market_basket` | `sample_data/insta_market_basket/instacart_marked_target_with_split.parquet` |

주의:

- `insta_market_basket` 원본 `instacart_marked_target_df.parquet`는 department mark라서 직접 학습에 쓰지 않는다.
- Instacart 학습에는 `instacart_marked_target_with_split.parquet`을 사용해야 한다.
- `yellow_trip_hourly`는 raw `yellow_trip.parquet`를 내부 변환하지 않는다. preprocessing notebook에서 만든 hourly parquet를 사용한다.

## 6. Recommended Experiment Flow

### 6.1 Smoke Test

새 코드나 새 옵션을 추가한 뒤 먼저 짧게 확인한다.

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py model-test \
  --models rmtpp,titantpp,thp \
  --titan-candidates small_lmm \
  --thp-candidates small \
  --device cuda \
  --left-pad
```

### 6.2 Short Screening

교수님 피드백 대응, LR sensitivity, NaN 확인, convergence 속도 확인은 `e50` 또는
`e100`으로 먼저 본다.

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/<experiment_name> \
  --datasets <dataset_list> \
  --models rmtpp,titantpp \
  --titan-candidates small_lmm,mid_lmm \
  --epochs 50 \
  --seeds 42 \
  --lr <1e-3_or_5e-3_or_1e-2> \
  --batch-size 512 \
  --max-seq-len 64 \
  --eval-selections best_val_nll,best_score,final \
  --split-mode fixed \
  --value-head-activation identity \
  --value-input-mode residual \
  --train-loss-scope target_only \
  --device cuda
```

### 6.3 Main Comparison

short screening에서 안정적인 후보를 고른 뒤 long-epoch 본실험으로 확장한다.

```bash
python ~/workspace/paper_research/simple_lab_test/search/tpp_experiment.py long-epoch \
  --base-dir ~/workspace/paper_research/search_artifacts/<experiment_name> \
  --datasets intermittent,yellow_trip_hourly,insta_market_basket \
  --models rmtpp,titantpp,thp \
  --titan-candidates <selected_titan_candidates> \
  --thp-candidates small,base \
  --epochs 800 \
  --seeds 42,52,62 \
  --lr 1e-3 \
  --batch-size 512 \
  --max-seq-len 64 \
  --eval-selections best_val_nll,best_score,final \
  --split-mode fixed \
  --value-head-activation identity \
  --device cuda
```

## 7. Monitoring Checklist

tmux 안에서 확인할 것:

```bash
tmux ls
tmux attach -t <session_name>
```

서버 리소스:

```bash
nvidia-smi
```

로그 tail:

```bash
tail -f ~/workspace/paper_research/search_artifacts/<experiment_name>/logs/run.log
```

run별 train log:

```bash
find ~/workspace/paper_research/search_artifacts/<experiment_name>/runs -name train.log | sort
```

NaN 또는 error 확인:

```bash
grep -RniE "nan|NaN|ERROR|Traceback" \
  ~/workspace/paper_research/search_artifacts/<experiment_name>/logs \
  ~/workspace/paper_research/search_artifacts/<experiment_name>/runs
```

## 8. Artifact Sync To Local

서버 결과를 로컬로 내려받는다.

```bash
rsync -avz 5090:~/workspace/paper_research/search_artifacts/<experiment_name>/ \
  /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/<experiment_name>/
```

5080에서 돌린 경우:

```bash
rsync -avz 5080:~/workspace/paper_research/search_artifacts/<experiment_name>/ \
  /Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/<experiment_name>/
```

## 9. Artifact Reading Order

결과 분석은 아래 순서로 읽는다.

```text
1. experiment_manifest.json
2. logs/run.log
3. leaderboard/summary.csv
4. leaderboard/test_summary.csv
5. leaderboard/histories.csv
6. leaderboard/scale_wise_summary.csv
7. leaderboard/test_scale_wise_summary.csv
8. paper_outputs/report.md
9. paper_outputs/plots/
```

Validation-only candidate gate에서는 held-out lock이 위의 일반 reading order보다
우선한다. 현재 unified artifact의 `leaderboard/runs.csv`와
`paper_outputs/report.md`는 validation과 test metric을 같은 파일에 포함할 수 있으므로,
gate를 기록하기 전에는 열지 않는다. 이 단계에서는 `experiment_manifest.json`,
`logs/run.log`, `leaderboard/summary.csv`, `leaderboard/histories.csv`,
validation `scale_wise_summary.csv`, validation confusion/class metric, validation plot만
읽는다. `leaderboard/test_*`, run-local `test_*`, test plot도 같은 시점까지 잠근다.

핵심 질문:

- 모든 run이 완료됐는가?
- NaN 또는 Traceback이 있었는가?
- best validation NLL epoch은 언제인가?
- final epoch에서 성능이 악화됐는가?
- total NLL 개선이 marker NLL 때문인가, time NLL 때문인가?
- quantity MAE가 전체와 scale-wise에서 함께 개선됐는가?
- test result가 validation selection과 같은 방향인가?

## 10. Notion Writer Prompt Template

아래 템플릿을 Notion Writer 세션에 그대로 전달하면 된다.

### 10.1 Notion Location Routing

Current user override (2026-07-12):

- Model Enhancement Session에서 생성하는 model design, architecture decision,
  ablation, smoke, screening, matched comparison, multi-seed 결과는 모두
  `5. Model Design Enhancement` 하위에 작성한다.
- 위 작업은 validation metric을 포함하더라도 `2. Confirm and Refine Topic` 아래에
  생성하지 않는다.
- `2. Confirm and Refine Topic > Model Validation`은 연구 주제 확인, contribution
  framing, baseline 타당성 확인처럼 topic refinement가 primary scope일 때만 사용한다.
- 페이지 생성 전에 동일 제목을 검색하고, 기존 페이지가 있으면 업데이트한다.
- Model Enhancement 페이지가 다른 위치에 있으면 중복 생성하지 않고
  `5. Model Design Enhancement` 아래로 이동한 뒤 업데이트한다.

Current user override (2026-07-13):

- `notion_writer_prompts/` 파일은 Notion Writer Session 전달물이 아니라 실험별
  source draft와 audit trail로 사용한다.
- Model Enhancement Session은 prompt를 만든 뒤 작업을 멈추지 않고, 연결된 Notion
  workspace의 `5. Model Design Enhancement`에 직접 생성/업데이트한다.
- 작성 전 동일 제목을 검색하고 기존 페이지를 fetch한 뒤 중복 없이 수정한다.
- 작성 후 상위 경로, 제목 2/3 구조, 상세 페이지 링크를 다시 fetch해 검증한다.
- 실험 기록은 시작 날짜를 제목 2로, 세부 목표와 Step을 제목 3으로 정리한다.

Current user override (2026-07-15):

- Model Enhancement 실험 페이지는 교수님이 빠르게 읽을 수 있도록 `상태`, `목적`,
  `Factorial 계약`, `고정 조건`, `실행 명령어`, `결과` 순서로만 작성한다.
- Factorial 실험이 아니면 `Factorial 계약`을 `Variant 계약`으로 바꾸고 비교 대상의
  차이만 간단히 적는다.
- `Frozen Reference`, SHA/checksum, source manifest, preflight 상세, acceptance contract
  내부값, held-out lock 세부 규칙과 디버깅 로그는 Notion 본문에 적지 않는다. 이 정보는
  local source draft, manifest, ADR과 artifact에 보존한다.
- 실험 중에는 `결과` 제목만 만들고 본문은 비워 둔다. 실험 완료 후 artifact 분석을
  마친 시점에만 결과와 해석을 추가한다.
- 실행 중 오류나 복구가 있었더라도 현재 상태를 이해하는 데 필요한 한 줄만 `상태`에
  남기고, 상세 원인과 검증 내역은 개발 기록에 둔다.

```text
다음 실험 내용을 Notion에 정리해주세요.

위치:
- Model Enhancement 작업: 5. Model Design Enhancement
- Topic refinement 작업: 2. Confirm and Refine Topic > Model Validation
- 현재 작업의 primary scope에 해당하는 위치 하나만 선택
- 관련 상위 페이지가 있으면 그 하위 페이지로 만들고, 없으면 새 페이지로 생성

페이지 제목:
- <experiment_title>

상태:
- <준비 중 | 실험 중 | 완료 | 중단>
- 실험 시작 시각: <YYYY-MM-DD HH:MM:SS KST>
- 실행 서버 / tmux: <server> / <session_name>

목적:
- <이 실험을 왜 하는지>
- <이번 비교로 확인할 가설>

Factorial 계약:
- <Variant별로 달라지는 축과 역할을 표로 작성>
- <비교 축 외 조건은 같다는 점을 한 줄로 명시>

고정 조건:
- dataset: <dataset_list>
- model: <model_list>
- epochs / seeds: <epochs> / <seed_list>
- lr / batch_size: <lr_list> / <batch_size>
- lookback / max_seq_len: <lookback> / <max_seq_len>
- split_mode: fixed
- 주요 model/loss 옵션: <핵심 옵션만 작성>
- artifact: <artifact_path>

실행 명령어:
- <shell_command>

결과:
- 실험 중에는 제목만 생성하고 본문을 작성하지 않음
- 완료 후 핵심 결과, 해석과 다음 결정을 간단히 추가
```

## 11. Notion Result Update Prompt Template

실험 완료 후에는 아래 템플릿을 사용한다.

```text
다음 실험 결과를 기존 Notion 페이지에 업데이트해주세요.

대상 페이지:
- <notion_page_title_or_url>

분석 기준:
- experiment_manifest.json으로 설정 확인
- logs/run.log에서 완료 여부와 NaN/Traceback 확인
- leaderboard/summary.csv에서 validation 성능 확인
- leaderboard/test_summary.csv에서 held-out test 성능 확인
- histories.csv에서 learning curve와 best epoch 확인
- scale_wise_summary.csv와 test_scale_wise_summary.csv에서 scale-wise quantity MAE 확인
- paper_outputs/plots가 있으면 learning curve와 scale-wise plot을 함께 참고

업데이트 규칙:
- 기존 페이지의 `결과` 제목 아래에만 내용을 추가
- 완료 여부와 실험 종료 시각
- Variant별 핵심 metric 비교 표
- Factorial main effect와 interaction에 대한 짧은 해석
- acceptance 결과와 다음 결정
- `Frozen Reference`, checksum, manifest, preflight와 디버깅 상세를 다시 추가하지 않음
```

## 12. Current Interpretation Rules

현재까지의 실험 히스토리 기준 해석 규칙:

- RMTPP가 고LR에서 안정적이고 TitanTPP가 NaN이면, value-conditioned objective 자체보다 Titan encoder/LMM/value-head 결합부의 optimization sensitivity로 본다.
- `5e-3`에서 best epoch이 빨라지면 교수님 피드백의 “learning rate가 작아 convergence가 느릴 가능성”에 대한 부분 근거가 된다.
- `1e-2`에서 NaN 또는 성능 악화가 발생하면 blind LR increase가 아니라 warmup/scheduler가 필요하다고 본다.
- total NLL만 낮아지고 marker NLL 또는 quantity MAE가 나빠지면 time likelihood가 지배한 착시일 수 있다.
- Instacart는 scale 0/1이 대부분이고 tail scale은 매우 희소하므로, 전체 MAE와 tail MAE를 분리해서 해석한다.
- RevIN 해석에서는 target/input transform domain과 normalization method를 분리한다.
  Train-global normalization은 RevIN이 아니며, log-domain 실패를 raw-domain RevIN
  실패로 일반화하지 않는다.
- TitanTPP가 standalone Titan wrapper 대신 `MemoryEncoder`를 직접 호출하는 경우,
  `TitanConfig.use_revin` 값만으로 RevIN 활성화를 판단하지 않고 실제 forward path를
  확인한다.
- TitanTPP 단독 우월성은 현재 단계에서 강하게 주장하지 않는다. 더 안전한 contribution은 value-conditioned marked TPP framework와 안정화된 encoder variant 비교다.

## 13. Related Documents

- `simple_lab_test/search/README.md`
- `simple_lab_test/search/search_experiment_guide.md`
- `simple_lab_test/search/search_experiment_info.md`
- `simple_lab_test/search/model_enhancement_strategy.md`
- `dataset_analysis_report.md`
