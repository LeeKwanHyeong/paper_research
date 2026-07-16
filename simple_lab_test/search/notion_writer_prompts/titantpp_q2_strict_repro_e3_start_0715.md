# TitanTPP Q2 Strict Deterministic e3 Exact-Reproduction Probe

Notion의 `5. Model Design Enhancement` 아래에 작성한다. 상위 history에는 제목 2
`2026-07-15 | Strict Reproducibility Gate`, 제목 3
`Step 1. Q2 e3 A/B Exact-Reproduction Probe`로 연결한다.

## 상태

- 상태: `완료`
- 실험 시작 시각: `2026-07-15 22:53:50 KST`
- 실험 종료 시각: `2026-07-15 22:56:00 KST`
- 실행 서버 / tmux: `5090 / titantpp_q2_strict_e3_0715`
- 완료 여부: Run A, Run B, exact comparator 모두 `exit_code=0`

## 목적

같은 Q2 설정과 seed를 사용한 두 독립 Python process가 epoch history, checkpoint 선택
epoch, model tensor state까지 정확히 재현하는지 확인한다. 이 실험은 Q2 성능을 다시
평가하는 실험이 아니라 향후 구조 비교에 사용할 deterministic 실행 기반을 검증하는
실험이다.

## Variant 계약

| 실행 | 모델 설정 | 실행 단위 | 역할 |
| --- | --- | --- | --- |
| Run A | Q2 `coupled + no log auxiliary` | fresh Python process | exact 비교 기준 |
| Run B | Q2 `coupled + no log auxiliary` | fresh Python process | 독립 재실행 |

두 실행은 process와 artifact directory만 분리한다. model, initialization seed, data,
shuffle seed, optimizer, epoch budget과 checkpoint policy는 모두 같다.

## 고정 조건

| 항목 | 값 |
| --- | --- |
| dataset / split | `intermittent / fixed` |
| model / candidate | `TitanTPP / small_lmm` |
| epochs / seed | `3 / 42` |
| learning rate / batch size | `1e-3 / 128` |
| lookback / max sequence | `52 weeks / 16` |
| decoder | `direct_raw_qty` |
| normalization | `causal_shrinkage_revin`, `k=8` |
| magnitude route / auxiliary | `coupled / none` |
| train loss scope / mode | `target_only / hybrid` |
| reproducibility | `strict`, dedicated loader generator, `num_workers=0` |
| process environment | `PYTHONHASHSEED=42`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| exact gate | history JSON, selected epochs, best-score/best-NLL/final state digest |
| artifact | `search_artifacts/model_enhancement_titantpp_q2_strict_repro_e3_0715` |

이 단계에서는 성능 우열을 판단하지 않는다.

## 실행 명령어

```bash
ssh 5090 '/opt/miniconda3/envs/ai_env/bin/tmux new-session -d \
  -s titantpp_q2_strict_e3_0715 \
  "env PROJECT_ROOT=/home/leekwanhyeong/workspace/paper_research \
  PYTHON_BIN=/opt/miniconda3/envs/ai_env/bin/python \
  SOURCE_REVISION=f6da9af9193f6f5bcd6dd60a711b9e8921593829 \
  bash /home/leekwanhyeong/workspace/paper_research/simple_lab_test/search/scripts/run_titantpp_q2_strict_repro_e3_0715.sh"'
```

## 결과

- 두 독립 Q2 실행의 3개 epoch history가 완전히 같았고 comparator의 전체 `22/22`
  검사를 통과했다. 불일치는 `0`건이었다.
- Best score와 best validation NLL은 두 실행 모두 epoch 2, final은 epoch 3으로
  동일하게 선택됐다.
- Best score, best validation NLL, final checkpoint의 model tensor state가 모두
  정확히 일치했다.
- Strict deterministic 실행 기반은 검증됐다. 이 결과는 Q2의 성능 승격을 의미하지
  않으며, V2 유지와 Q3 미승격 결정도 변경하지 않는다.
- Q3는 자동으로 재개하지 않는다. 명시적으로 다시 열 경우에만 strict 기반의 matched
  e50 비교를 준비한다.
