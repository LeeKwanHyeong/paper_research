# Notion Update: TitanTPP V5b Bounded Class-Prior Contract

## 작성 위치

- `5. Model Design Enhancement > Enhancement & Validation History`
- 날짜 구역: `2026-07-19 | Intermittent V5b Class-Prior Enhancement`
- 세부 Step: `Step 1. Bounded Class-Prior Marker Loss Contract`
- Model Enhancement 내용이므로 `2. Confirm and Refine Topic`에는 작성하지 않는다.
- 같은 제목의 상세 페이지가 있으면 업데이트하고, 없을 때만 새 페이지를 만든다.

## 페이지 제목

`TitanTPP V5b Bounded Class-Prior Marker Loss Contract`

## 작성 원칙

- `상태`, `목적`, `Variant 계약`, `고정 조건`, `실행 명령어`, `결과` 순서만 사용한다.
- 모델 성능 결과가 아니라 구현 전 설계 동결 기록임을 분명히 한다.
- weight 계산 내부 과정, acceptance threshold 전체, checksum, preflight, held-out
  lock 세부 규칙은 Notion 본문에 넣지 않는다.
- V5b가 확정 모델이거나 성능이 개선됐다고 쓰지 않는다.

## 상태

- 날짜: `2026-07-19 KST`
- 상태: `설계 완료, 구현 전`
- 실행 서버: 후속 실험은 `5080`
- tmux: 아직 없음
- Intermittent incumbent: V2 `small_lmm`
- V5b registry: `SELECTED_HYPOTHESIS`

## 목적

Intermittent의 mark 분포 편향에서 rare tail 전체를 키우지 않고, high-support
mark `0/1` 경계의 예측 경쟁만 완만하게 조정할 수 있는지 확인한다.

Train target의 `86.60%`가 marks `0-2`에 집중돼 있고, V5a에서는 mark-0 예측
비중이 `58.750%`까지 증가하면서 mark-1 recall이 `24.664%`로 하락했다. V5b는
이 문제를 별도 marker-objective ablation으로 다룬다.

## Variant 계약

| Variant | Marker train objective | 적용 class | Inference logits | 역할 |
| --- | --- | --- | --- | --- |
| V2 | ordinary CE | 전체 real mark | 원본 logits | active baseline |
| V5b | bounded class-prior weighted CE | train support `>=1%`인 marks `0-5` | 원본 logits | validation candidate |

- marks `6-10`은 support가 낮아 weight `1.0`으로 유지한다.
- reported `nll_marker`는 V2와 동일한 ordinary CE로 유지한다.
- logit adjustment, V5a RPS, V3 value expert, resampling은 사용하지 않는다.

## 고정 조건

- dataset: `intermittent`
- model/candidate: TitanTPP V2/V5b, `small_lmm`
- first quality budget: e50 / seed `42`
- follow-up: seed-42 validation 통과 시 seeds `42,52,62`
- lr / batch size: `1e-3` / `128`
- lookback / max sequence length: `52` / `16`
- split / reproducibility: fixed / strict
- value input / quantity loss / train scope: residual / hybrid / target-only
- checkpoint: ordinary `best_val_nll`
- prior source: fixed-split train next-event targets only
- evaluation: mark-1 recall, `1 -> 0` confusion, accuracy, balanced/macro metrics,
  ordinary marker NLL, calibration, quantity/time guardrail
- held-out test: strict multi-seed validation 통과 후 한 번만 확인

## 실행 명령어

아직 없음. 이번 단계는 objective와 gate 설계 동결이며, 구현·focused test가 끝난 뒤
5080용 runner와 tmux 시작 기록을 별도로 작성한다.

## 결과

- train-only Laplace prior와 bounded square-root weighting을 V5b 방식으로 선택했다.
- marks `0-5`만 완만하게 보정하고 marks `6-10`은 중립으로 유지한다.
- train·inference 모두 logits를 조정하지 않고, reported marker NLL도 ordinary CE로
  유지한다.
- train/validation/test prior 차이가 작아 logit adjustment는 첫 V5b에서 제외했다.
- V5b는 아직 구현·검증·승격되지 않았으며 Intermittent baseline은 V2다.

## 다음 작업

`V5b train-prior helper, weighted marker loss, calibration metrics, artifact identity와 focused test 구현`

## Local Source

```text
.agents/results/architecture/adr-titantpp-v5b-bounded-class-prior-marker-loss.md
.agents/results/architecture/titantpp-model-status-baseline-registry.md
simple_lab_test/search/model_enhancement_strategy.md
search_artifacts/model_enhancement_inter_mark_diagnostics_0712
search_artifacts/model_enhancement_v5a_inter_short_e50_0712
```

## Direct Notion Update

- detail page:
  `https://app.notion.com/p/3a2bbe40561381c3b087c53a945c0a63`
- parent history:
  `https://app.notion.com/p/2e8bbe40561380a88b5eef94e834892e`
- strategy page:
  `https://app.notion.com/p/394bbe4056138046bf3bfbc6f4c8c31a`
- refetch verification: `3/3 PASS`
