# TitanTPP Q3 Factorial Instacart Top-20 e1 Smoke 시작

`5. Model Design Enhancement` 아래에 세부 페이지
`TitanTPP Q3 Factorial Instacart Top-20 e1 Smoke`를 생성한다. 상위 history에는
제목 2 `2026-07-14 | Q3 Actual-Data Integration` 아래 제목 3
`Step 1. Q3 Factorial Instacart Top-20 e1 Smoke`로 연결한다.

## 실험 시작 정보

- 시작 기록 시각: `2026-07-14 08:33 KST`
- 실행 서버: `5090` (`192.168.0.71`, user `leekwanhyeong`)
- 프로젝트 경로: `/home/leekwanhyeong/workspace/paper_research`
- Conda 환경: `ai_env`
- tmux session: `titantpp_q3_insta_e1_0714`
- runner: `simple_lab_test/search/scripts/run_titantpp_q3_insta_smoke_e1_0714.sh`
- artifact root: `search_artifacts/model_enhancement_titantpp_q3_insta_smoke_e1_0714`
- 실행 준비 commit: `d552b7749c0e3836c277338dc44d82de50589e82`
- source 검증 시각: `2026-07-14 08:41:54 KST`
- CUDA·데이터 preflight 시각: `2026-07-14 08:44:52 KST`
- 실제 실행 시작 시각: `2026-07-14 08:45:33 KST`
- 실행 종료 시각: `2026-07-14 08:45:53 KST`
- 완료 단회 확인 시각: `2026-07-14 12:57:58 KST`
- 상태: `completed and synced; integration analysis pending`

## 실험 목적

5090 synthetic CUDA gate를 통과한 Q2/Q3a/Q3b/Q3c를 실제 Instacart fixed-split
DataLoader에 연결한다. 이전 Q0/Q1/Q2 top-20 e1 smoke와 같은 split, series, seed,
학습 예산을 유지하고 Q3의 두 요인만 바꿔 actual-data backward, checkpoint,
cache/resume identity, loss logging, summary, scale-wise export가 모두 동작하는지 확인한다.

이 단계는 실제 데이터 통합 smoke다. 한 epoch 수치로 Q3 성능, gradient detachment 효과,
log2 auxiliary 효과를 판정하지 않으며 Intermittent 데이터도 열지 않는다.

## Factorial 계약

| Variant | Magnitude encoder gradient | Log2 auxiliary | 역할 |
|---|---|---|---|
| Q2 control | `coupled` | `none` | 같은 revision의 fresh control |
| Q3a | `detached` | `none` | gradient-routing integration |
| Q3b | `coupled` | `log_huber` | dual-domain loss integration |
| Q3c | `detached` | `log_huber` | combined interaction integration |

네 변형은 같은 `direct_raw_qty` decoder, Q2 `causal_shrinkage_revin`, state-dict
구조와 parameter count를 사용한다. Q2/Q3a 및 Q3b/Q3c는 같은 초기 가중치에서 forward
scalar가 같지만, 한 번의 backward 이후에는 gradient route가 달라지므로 epoch 1 metric
동일성을 요구하지 않는다.

## 고정 조건

| 항목 | 값 |
|---|---:|
| dataset / split | `insta_market_basket` top-20 / `fixed` |
| expected train / validation / test samples | `1380 / 300 / 300` |
| model / candidate | `TitanTPP / small_lmm` |
| epochs / seed | `1 / 42` |
| LR / batch | `1e-3 / 16` |
| lookback / max sequence | `10 weeks / 16` |
| decoder / normalization | `direct_raw_qty / causal_shrinkage_revin` |
| loss scope / mode | `target_only / hybrid` |
| marker objective | plain CE |
| lambda magnitude / quantity | `1.0 / 0.25` |
| lambda log quantity / Huber delta / floor | `0.25 / 1.0 / 1.0` |
| RevIN epsilon / shrinkage k | `1e-5 / 8` |
| sigma floor constant | `0.0550124034288891` |
| center / affine / stat context | `mean / false / none` |
| selections | `best_val_nll,best_score,final` |

Normalization mode와 raw context statistics는 네 변형에서 동일하다. Global raw moments와
effective floor는 같은 Instacart fixed train split으로만 계산하며 validation/test row를
통계에 포함하지 않는다.

## 실험 계획

1. 준비 커밋을 5090 비-Git 작업 복사본에 동기화하고 checksum/source manifest를 남긴다.
2. RTX 5090, `ai_env`, CUDA linker, split parquet, CLI 계약을 preflight한다.
3. Q2, Q3a, Q3b, Q3c를 같은 설정으로 순차 실행한다.
4. 각 변형의 sample count, raw statistics, parameter count와 variant identity를 대조한다.
5. Epoch 1 backward, checkpoint, cache/resume identity와 새 loss logging을 확인한다.
6. Manifest, log, summary, test summary, histories, scale-wise metrics, plots 순서로 읽는다.
7. Integration gate만 판정하고 e1 metric으로 후보를 선택하지 않는다.

## 실행 명령어

```bash
ssh 5090
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate ai_env
/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s titantpp_q3_insta_e1_0714 \
  "bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_q3_insta_smoke_e1_0714.sh"
```

## 실행 시작 확인

- `d552b77` 변경 파일 4개를 5090 비-Git 작업 복사본에 rsync했고 checksum
  dry-run에서 변경 파일이 없음을 확인했다.
- 로컬·원격 SHA256 4개가 모두 일치하고 runner 권한은 `755`다.
- `source_sync_manifest.json`에 full revision, Q3 구현 revision, 파일별 SHA256과
  검증 시각을 기록했다.
- Preflight는 RTX 5090 idle `41 MiB / 0%`, PyTorch `2.11.0+cu130`, CUDA `13.0`,
  실제 CUDA tensor allocation을 통과했다.
- Instacart fixed-split 파일 5개, top-20 `2,000 rows / 20 series`, quantity
  reconstruction max error `1.42e-14`, loader samples `1380/300/300`을 확인했다.
- Q2/Q3a/Q3b/Q3c CLI contract가 모두 통과했다.
- 초기 확인에서 Q2는 같은 split과 train-only raw statistics를 사용해 epoch 1을
  완료했고 Python GPU process `676 MiB`를 확인했다.
- Conda `libtinfo.so.6` version warning이 출력됐지만 tmux, runner, CUDA 계산과 Q2
  epoch는 정상 동작했다. Runtime failure 근거로 보지 않는다.
- 초기 진입 확인 뒤 지속 polling을 중단했다. Q3a/Q3b/Q3c 완료 여부와 전체 gate는
  사용자 요청 시 단회 확인한다.

## 완료 및 Artifact 동기화 확인

- 요청 시 한 차례만 원격 상태를 확인했다. Tmux session은 종료됐고 실행 중인 GPU
  process는 없었다.
- Root `SMOKE_SUCCESS` sentinel과 aggregate exit code `0`을 확인했다. Q2, Q3a,
  Q3b, Q3c의 개별 exit code도 모두 `0`이며 failure sentinel은 없다.
- 원격 artifact root를 로컬 같은 경로로 checksum rsync했다. 로컬에는 파일 `388`개,
  약 `18M`가 있으며 checksum dry-run에서 변경 항목이 없었다.
- Root manifest, source sync manifest, root log와 variant status가 파싱되고 서로
  일치했다.
- 네 변형 모두 experiment manifest, summary, test summary, history, validation/test
  scale-wise metrics, report, plot, `best_val_nll` checkpoint를 포함한다.
- 이 단계에서는 생성·동기화 계약만 확인했다. Metric 해석과 integration gate 판정은
  protocol artifact reading order에 따른 다음 작업으로 남긴다.

## Acceptance Gate

- Q2/Q3a/Q3b/Q3c 모두 exit code 0과 `SMOKE_SUCCESS`
- NaN, Inf loss, Traceback, CUDA runtime error/OOM 없음
- 네 run의 fixed split, series/sample count, seed, batch, lookback, max sequence가 동일
- `magnitude_stats_source_split=train`이며 raw global moments/effective floor가 동일
- parameter count와 state-dict 구조가 동일하고 variant path/config는 두 실험 요인만 다름
- Q2/Q3a의 log auxiliary는 0, Q3b/Q3c의 log auxiliary는 positive finite
- Epoch 1 train/validation loss와 raw magnitude diagnostics가 finite
- best validation NLL, best score, final checkpoint와 history가 생성됨
- validation/test summary, scale-wise metrics, report와 plots가 생성됨

## 해석 제한

Top-20 e1은 actual-data 경로와 artifact 계약 확인용이다. Q2와 Q3a 또는 Q3b와 Q3c의
epoch 1 metric이 달라도 gradient-routing 학습 효과의 증거로 사용하지 않는다. Held-out
test output은 export/finite 확인에만 사용하고 모델 선택에 사용하지 않는다. 이 gate가
통과한 뒤에만 Intermittent seed-42 e50 validation-only screening을 준비한다.
