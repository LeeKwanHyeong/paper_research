# TitanTPP Q3 Factorial Intermittent Seed-42 e50 Validation-Only Screening 시작

`5. Model Design Enhancement` 아래에 세부 페이지
`TitanTPP Q3 Factorial Intermittent Seed-42 e50 Validation-Only Screening`을
생성한다. 상위 history의 제목 2 `2026-07-14 | Q3 Actual-Data Integration` 아래
제목 3 `Step 2. Q3 Intermittent Seed-42 e50 Validation-Only Screening`으로 연결한다.
`2. Confirm and Refine Topic`에는 작성하지 않는다.

## 준비 상태

- 준비 기록 시각: `2026-07-14 16:22 KST`
- 첫 실행 시도: `2026-07-15 08:11:06 KST` (V2 reference 전용 metric 계약 오류로 학습 전 중단)
- 실제 실행 시작 시각: `2026-07-15 08:15:07 KST`
- 실행 서버: `5090` (`192.168.0.71`, user `leekwanhyeong`)
- 프로젝트 경로: `/home/leekwanhyeong/workspace/paper_research`
- Conda 환경: `ai_env`
- tmux session: `titantpp_q3_inter_e50_0714`
- runner: `simple_lab_test/search/scripts/run_titantpp_q3_inter_seed42_e50_0714.sh`
- experiment artifact:
  `search_artifacts/model_enhancement_titantpp_q3_inter_seed42_e50_0714`
- V2 validation reference:
  `search_artifacts/model_enhancement_v2_inter_validation_reference_q3_0714`
- Notion page:
  `https://app.notion.com/p/39dbbe405613814ab90acb3f61406daf`
- 실행 준비 commit: `a0a65e59b4c66e0f9a03cf28c2928ab7f421127c`
- 실행 source revision: `f5851ff8476f2d01f3fc0a51e02c8f1eef27a418`
- source checksum 검증 시각: `2026-07-15 08:06:34 KST`
- evaluator fix checksum 재검증 시각: `2026-07-15 08:13:49 KST`
- CUDA·data·reference preflight 시각: `2026-07-15 08:08:58 KST`
- source sync manifest:
  `search_artifacts/model_enhancement_titantpp_q3_inter_seed42_e50_0714/source_sync_manifest.json`
- preflight manifest:
  `search_artifacts/model_enhancement_titantpp_q3_inter_seed42_e50_0714/preflight_manifest.json`
- 상태: `in progress; V2 reference PASS; fresh Q2 epoch 1 PASS; monitoring stopped`

Source revision, source-sync manifest와 CUDA/data/reference preflight를 고정했다.
V2 validation reference와 fresh Q2 epoch 1 진입을 확인한 뒤 지속 monitoring을 중단했다.

## 실험 목적

Instacart actual-data integration gate를 통과한 Q2/Q3a/Q3b/Q3c를 Intermittent
fixed split에서 동일 e50·seed-42 예산으로 비교한다. Q2의 raw shrinkage
normalization과 decoder는 고정하고 다음 두 요인만 분리한다.

1. Magnitude loss를 Titan encoder에서 detach하면 marker representation이 회복되는가
2. Raw loss에 log2 Huber auxiliary를 추가하면 low-quantity/log2 error가 보호되는가

Fresh Q2는 이전 Q2 결과의 재현 control이다. Q3a/Q3b의 중간 결과와 관계없이 Q3c까지
모두 실행한 뒤 `2 x 2` factorial interaction을 계산한다. 이 단계는 validation-only
candidate gate이며 multi-seed와 held-out 결론을 포함하지 않는다.

## Factorial 계약

| Variant | Magnitude encoder gradient | Log2 auxiliary | 역할 |
| --- | --- | --- | --- |
| Q2 control | `coupled` | `none` | 같은 revision의 fresh reproduction control |
| Q3a | `detached` | `none` | gradient-isolation main effect |
| Q3b | `coupled` | `log_huber` | low-quantity auxiliary main effect |
| Q3c | `detached` | `log_huber` | combined interaction |

네 variant는 gradient/auxiliary 두 축 외 설정, parameter count, data order와 seed를
동일하게 유지한다. Detached mode도 observed raw quantity encoder input은 유지하며,
marker/time NLL은 encoder와 magnitude input projection을 계속 학습한다.

## 고정 실행 조건

| 항목 | 값 |
| --- | ---: |
| dataset / split | `intermittent / fixed` |
| expected train / validation / test samples | `136256 / 41901 / 41344` |
| train event count / series | `159643 / 23387` |
| model / candidate | `TitanTPP / small_lmm` |
| expected marks / parameters | `12 / 78111` |
| epochs / seed | `50 / 42` |
| LR / batch | `1e-3 / 128` |
| lookback / max sequence | `52 weeks / 16` |
| decoder / normalization | `direct_raw_qty / causal_shrinkage_revin` |
| train loss scope / mode | `target_only / hybrid` |
| marker objective | plain CE, `lambda_ordinal=0` |
| lambda time / magnitude / raw quantity | `1.0 / 1.0 / 0.25` |
| lambda log quantity / Huber delta / floor | `0.25 / 1.0 / 1.0` |
| magnitude input embedding | `8` |
| RevIN epsilon / shrinkage k | `1e-5 / 8` |
| requested and effective raw sigma floor | `0.0550124034288891` |
| train raw mean / variance / std | `6.8458560663 / 3026.3645310228 / 55.0124034289` |
| center / affine / stat context | `mean / false / none` |
| analysis scale / tail order | `10 / 4` |
| checkpoint selections | `best_val_nll,best_score,final` |
| decision checkpoint | `best_val_nll` |

## Frozen Reference Identity

V2는 기존 checkpoint를 현재 evaluator로 validation target에만 다시 평가한다.

- V2 source artifact: `model_enhancement_v2_inter_short_e50_0710`
- V2 checkpoint selection / epoch: `best_val_nll / 19`
- V2 checkpoint SHA256:
  `1a901eb2ac912537e25b6c798978870a6f650857b41642f2a0b773030cc103c0`
- Fixed marked parquet SHA256:
  `dab4d8a7217f9c14d1c2336f649aef9ddaf2ba440d074e446d8fd5cc41506a30`
- Fixed train parquet SHA256:
  `3d66e0dc2ef671f652427b5f4756604b29efd10301ec42f9f5b9a7631eb8c242`
- Fixed validation parquet SHA256:
  `10c4811d02db5e4bff50af230e068754901bb0cf106f7ca29a5f8b694294ac72`
- Fixed test parquet SHA256:
  `191d675819db63647f34446bb9fae79f0822d2c09ef05387aeae3877b6fe8263`
- Fixed split manifest SHA256:
  `49752a1bd4ccaf1c2b8e37321e3657cb098d303c51a093eecf1c91ea3ef9bdfe`
- Frozen Q2 source artifact:
  `model_enhancement_direct_raw_qty_q012_inter_seed42_e50_0713`
- Frozen Q2 summary SHA256:
  `256fbd69a4a63bbb1b6e2cb97f3a223067990b2a639a8bc4a1428e61ae8066f2`

Runner는 V2/Q2 reference와 다섯 fixed-split source SHA, prior
`SCREENING_SUCCESS`를 학습 전에 확인한다. 하나라도 다르면 exit code `2`로
중단하며 다른 reference로 자동 대체하지 않는다. Test parquet SHA는 data identity만
검증하며 target metric이나 row 내용을 분석하지 않는다.

## 5090 Source Sync와 Preflight

- `a0a65e5` changed-file set 5개를 5090 비-Git 작업 복사본에 rsync했다.
- 로컬·원격 SHA256 5개가 모두 일치하고 checksum dry-run 변경 파일은 `0`개다.
- Runner 권한은 `755`, recovery evaluator를 포함한 runtime dependency 9개 SHA는
  로컬과 일치한다.
- Source manifest는 full revision, Q3 implementation revision, 파일별 SHA와 검증
  시각을 기록하며 로컬·원격에서 동일하다.
- RTX 5090은 `41 MiB / 0%`, PyTorch `2.11.0+cu130`, CUDA `13.0`이며 실제 CUDA
  tensor allocation을 통과했다.
- 다섯 fixed-split source와 V2 checkpoint/marked parquet, frozen Q2 summary SHA가
  모두 frozen 값과 일치하고 Q2 `SCREENING_SUCCESS`가 존재한다.
- Q2/Q3a/Q3b/Q3c remote CLI parser와 두 축 외 common config identity가 통과했다.
- Target tmux `titantpp_q3_inter_e50_0714`는 비어 있다.
- 원격 `ai_env`에는 `pytest`가 없어 focused test 재실행은 collection 전에 중단됐다.
  Dependency를 추가하지 않았고, 로컬 focused `25/25`, search 전체 `110/110` 통과본과
  remote runtime source SHA가 일치하며 remote parser/config identity가 직접 통과한
  것으로 대체했다.
- Test parquet은 SHA identity만 확인했으며 row, metric, report, plot은 읽지 않았다.

## 초기 실행과 Evaluator Recovery

- 첫 시도는 `08:11:06 KST`에 시작했으나 legacy V2의 구조적 N/A인
  `val_log_qty_aux_loss=NaN`을 active finite metric으로 오판해 `08:11:11`에 학습 전
  중단됐다. CUDA, data, model training failure는 아니며 held-out은 읽지 않았다.
- 실패 root log는 `logs/launch_attempt_1_failed.log`로 보존했다.
- `mark_residual`에서 `val_magnitude_loss`와 `val_log_qty_aux_loss`만 명시적 N/A로
  처리하고 active metric finite check는 유지했다. Fix는 `f5851ff`로 커밋했고 focused
  `25/25`, search 전체 `110/110`을 통과했다.
- Fix 2개 파일을 5090에 checksum 동기화하고 source/preflight manifest를 갱신한 뒤
  legacy V2 N/A contract를 원격에서 직접 재검증했다.
- 두 번째 시도는 `08:15:07 KST`에 시작했고 V2 reference가 `08:15:11`에
  `exit_code=0`, samples `41901`, NLL `5.666520`, marker NLL `0.991274`, log2 MAE
  `0.588742`, mark accuracy `0.572492`로 완료됐다.
- Fresh Q2는 fixed split `136256/41901/41344`, train event `159643`, raw mean/std
  `6.84585607/55.01240343`으로 진입했다.
- Q2 epoch 1은 train loss `49.851103`, validation NLL `10.523623`, accuracy
  `0.541133`, DT MAE `36.215743`, quantity MAE `3.142503`으로 완료됐다. Q2의
  train/validation log auxiliary는 계약대로 모두 `0`이다.
- `08:21:16 KST` 기준 tmux는 active, Python GPU process는 `712 MiB`이며 지속
  monitoring을 중단했다. 다음 확인은 사용자 요청 시 한 번만 수행한다.

## Runtime Acceptance Contract

Runner는 artifact root에 `acceptance_contract.json`을 생성한다. Machine-readable
contract의 unrounded 값을 실제 판정에 사용하고, 아래 표는 6자리 요약이다.

### Fresh Q2 reproduction

- frozen Q2 대비 total NLL, raw quantity MAE, log2 quantity MAE 상대 차이 `<=1%`
- mark accuracy 절대 차이 `<=0.25%p`
- sample count, train event count, raw moments, model/optimizer config,
  checkpoint policy exact match
- 재현 실패 시 Q3 효과를 해석하지 않고 code/data drift부터 조사

### Full candidate gate

| Metric | Gate |
| --- | ---: |
| Overall raw quantity MAE | `<=2.736781` |
| History count `<=4` raw MAE | `<=2.053191` |
| Log2 quantity MAE | `<=0.600517` |
| `1-9` raw MAE | `<=0.999348` |
| `10-99` raw MAE | `<=9.784525` |
| Marker NLL | `<=1.001186` |
| Total NLL | `<=5.694853` |
| Time NLL | `<=4.698623` |
| Mark accuracy | `>=56.999%` |
| DT MAE | `<=42.905873` |
| Predicted mark-0 absolute share error | `<=5.850%p` |
| Mark-1 recall | `>=44.616%` |
| Pre-clamp negative prediction share | `<=1%` |
| Normalized-target non-finite count | `0` |

모든 적용 가능한 loss, prediction, center와 scale이 finite여야 한다. Validation share
`>=5%`인 bucket은 `1-9`, `10-99`이며 둘 다 gate를 통과해야 한다.

## Mechanism과 선택 규칙

- Q3a mark-accuracy recovery ratio
  `(acc_Q3a-acc_Q2)/(acc_V2-acc_Q2)`가 `>=0.50`이면 shared-gradient interference
  근거로 기록한다.
- Q3a의 overall/short raw MAE는 fresh Q2보다 `5%` 넘게 악화되면 안 된다.
- Q3b log2 MAE는 fresh Q2보다 `>=5%` 개선하고, overall/short raw MAE는 fresh
  Q2보다 `5%` 넘게 악화되면 안 된다.
- 주요 metric마다 interaction
  `(Q3c-Q3a)-(Q3b-Q2)`를 기록하되 promotion threshold로 사용하지 않는다.
- Full gate를 통과한 single intervention을 Q3c보다 우선한다.
- Q3a와 Q3b가 모두 통과하면 새 loss hyperparameter가 없는 Q3a를 우선한다.
- 결합이 필요한 경우에만 Q3c를 선택한다. 모두 실패하면 V2를 유지한다.

## Held-Out Lock

Unified long-epoch runner는 fixed-split test artifact를 자동 생성할 수 있지만 seed-42
validation 판정이 기록되기 전에는 아래 파일을 열지 않는다.

- `leaderboard/runs.csv`
- `leaderboard/test_*`
- run-local `test_*`
- `paper_outputs/report.md`
- test plot

판정 전 읽을 수 있는 범위는 manifest, acceptance contract, root/variant log,
variant status, `summary.csv`, `histories.csv`, validation scale-wise,
validation confusion/class metric과 validation plot뿐이다. Seed-42 candidate가 full gate를
통과해도 바로 held-out을 열지 않고, 먼저 strict matched seeds `42,52,62`를 통과해야 한다.

## 실험 계획

1. 준비 commit을 5090 비-Git 작업 복사본에 checksum 동기화하고 source manifest를 남긴다.
2. RTX 5090, PyTorch/CUDA, tmux 충돌, entrypoint, reference와 다섯 source SHA를
   preflight한다.
3. Frozen V2 checkpoint를 validation-only로 재평가한다.
4. Fresh Q2, Q3a, Q3b, Q3c를 같은 예산으로 순차 실행한다.
5. 실행 시작 후 V2 reference 완료, Q2 split/raw statistics와 첫 epoch 진입까지만 확인한다.
6. 지속 polling을 중단하고 사용자 요청 시 완료 여부를 한 번 확인한다.
7. Artifact를 로컬로 동기화한 뒤 held-out lock 범위 안에서 gate를 판정한다.

## 실행 명령어

```bash
ssh 5090
/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s titantpp_q3_inter_e50_0714 \
  "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_q3_inter_seed42_e50_0714.sh"
```

## 로컬 준비 검증

- runner `bash -n`: `PASS`
- runtime `acceptance_contract.json` heredoc JSON parse: `PASS`
- Q2/Q3a/Q3b/Q3c CLI parser/validation: `PASS`
- 허용된 gradient/auxiliary 두 축 외 `ExperimentConfig` identity: `PASS`
- 로컬 V2/Q2 reference와 다섯 fixed-split source SHA match: `PASS`
  (test parquet은 identity-only hash)
- focused Q3 regression: `19 passed`
- evaluator recovery focused / search full: `25 passed / 110 passed`
- `git diff --check`: `PASS`
- 5090 source checksum/CUDA/data/reference/CLI preflight: `PASS`
- V2 validation reference: `PASS`
- fresh Q2 epoch 1: `PASS`
- tmux launch/training: `in progress; monitoring stopped`

첫 focused test 명령은 project root가 `PYTHONPATH`에 없어 collection 단계에서
`ModuleNotFoundError: models`가 발생했다. 동일 코드를 `PYTHONPATH=.`로 다시 실행해
`19/19` 통과했으며 model 또는 runner failure로 분류하지 않는다.

## 다음 단계

- 완료: source sync/preflight, evaluator recovery, V2 reference와 fresh Q2 epoch 1 확인
- 진행 중: 5090 tmux에서 Q2/Q3a/Q3b/Q3c e50 순차 실행
- 다음: 사용자 요청 시 완료 여부 단회 확인 및 artifact 로컬 동기화
- 그다음: validation-only lock 범위에서 acceptance gate와 factorial interaction 분석
