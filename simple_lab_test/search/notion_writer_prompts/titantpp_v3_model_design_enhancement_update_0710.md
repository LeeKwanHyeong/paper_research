다음 내용을 기존 Notion의 `5. Model Design Enhancement` 페이지에 업데이트해주세요.

위치:
- `5. Model Design Enhancement`
- 기존 TitanTPP 모델 강화 문서 안에서 V1/V2 다음 단계로 연결해주세요.

대상 섹션:
- `TitanTPP V3: Mark-Conditioned Value Head`
- 같은 제목 또는 같은 설계 내용을 다루는 섹션이 있으면 해당 섹션을 업데이트하고, 중복 페이지나 중복 섹션은 만들지 마세요.

문서 상태:
- 설계: `confirmed`
- 구현: `implemented`
- focused verification: `passed`
- short screening: `in progress`

기준 시각 및 실행 환경:
- 문서 업데이트 기준 시각: `2026-07-10 10:54:55 KST`
- screening 시작 시각: `2026-07-10 10:33:38 KST`
- 실행 서버: `5080`
- tmux session: `titantpp_v3_smoke_screen_e50_0710`
- conda env: `ai_env`

## 1. Enhancement Context

V1/V2에서 확인한 내용:
- V1은 shared residual value head와 `residual_only` objective를 사용합니다.
- V2는 같은 shared residual value head를 유지하면서 differentiable expected quantity loss를 추가한 `hybrid` objective입니다.
- V2는 Intermittent와 Yellow Trip에서 quantity MAE 및 score를 개선했지만, Yellow Trip marker NLL 악화와 seed variance가 남았습니다.
- V3의 목적은 Titan backbone이나 RMTPP time head를 바꾸는 것이 아니라, next mark와 quantity residual의 의존성을 value head에 명시적으로 반영하는 것입니다.

설계 질문:
- 서로 다른 mark scale에 속한 event가 동일한 residual predictor 하나를 공유하는 것이 충분한가?
- 예측된 mark에 따라 quantity residual의 조건부 분포가 달라지는 구조가 V2의 quantity 개선을 유지하거나 강화할 수 있는가?

## 2. V1/V2/V3 Design Comparison

| Version | Value head | Training objective | 핵심 역할 |
| --- | --- | --- | --- |
| V1 | shared residual head | marker NLL + time NLL + residual loss | pre-enhancement reference |
| V2 | shared residual head | V1 + expected quantity loss | quantity reconstruction 강화 |
| V3 | shared residual + mark-specific delta experts | V2와 동일한 hybrid objective | residual prediction을 next mark에 condition |

중요한 ablation 원칙:
- V2와 V3의 첫 비교에서는 value head 구조만 변경합니다.
- `loss_mode=hybrid`, `lambda_qty=0.25`, value input, Titan memory, time head, lookback, max sequence length, fixed split은 동일하게 유지합니다.
- 따라서 V2/V3 차이는 `value_head_mode=shared`와 `value_head_mode=mark_conditioned_experts`로 제한합니다.

## 3. V3 Architecture

real mark 수를 `K`라고 할 때, PAD mark는 expert 대상에서 제외합니다.

Shared residual:

```text
r_shared(h_j) = W_shared h_j + b_shared
```

Mark-specific residual:

```text
r_k(h_j) = activation(r_shared(h_j) + delta_k(h_j))
delta(h_j) = W_delta h_j + b_delta
```

각 hidden state는 하나의 residual만 출력하는 대신 real mark별 residual vector를 출력합니다.

```text
value_by_mark shape = [batch, transition, K_real]
```

설계 의도:
- shared head는 모든 mark가 공유하는 quantity residual pattern을 학습합니다.
- delta expert는 해당 mark에서만 필요한 보정값을 학습합니다.
- 완전히 독립된 expert 여러 개보다 shared representation을 유지해 rare mark의 데이터 부족 위험을 줄입니다.

## 4. Training Contract

Residual supervision은 ground-truth next mark에 해당하는 expert만 선택합니다.

```text
value_hat_j = r_{y_(j+1)}(h_j)
L_value = Huber(value_hat_j, value_target_(j+1))
```

Hybrid quantity supervision은 모든 real mark expert를 predicted mark probability로 가중합니다.

```text
E[q_(j+1) | h_j]
  = sum_k p(y_(j+1)=k | h_j) * base^(k + r_k(h_j))

L_total
  = L_marker + L_time + L_value + lambda_qty * L_qty
```

첫 V3a 실험에서는 quantity loss에서 mark probability gradient를 detach하지 않습니다.
- V2와 동일한 quantity-to-mark coupling을 유지해야 architecture-only ablation이 됩니다.
- Yellow Trip marker NLL trade-off가 계속 확인될 때만 V3b `detached-gate`를 후속 설계합니다.

## 5. Inference And Evaluation Contract

- residual/value metric은 true next mark expert를 사용해 조건부 residual 자체의 학습 품질을 측정합니다.
- 실제 reconstructed quantity는 predicted next mark와 그 mark의 residual expert를 사용합니다.
- hybrid training의 expected quantity는 argmax가 아니라 전체 mark probability와 전체 expert를 사용합니다.
- 이 구분을 통해 mark 분류 오류와 mark 내부 residual 오류를 혼합하지 않고 해석할 수 있습니다.

## 6. Initialization And Fair Comparison

- `value_mark_delta_head`의 weight와 bias는 모두 zero initialization합니다.
- 초기 상태에서 모든 `r_k(h)`는 기존 V2 `r_shared(h)`와 정확히 같습니다.
- delta head 생성 전후의 PyTorch RNG state를 보존해 같은 seed에서 V2/V3 공통 parameter가 byte-level로 동일하게 초기화되도록 했습니다.
- 따라서 학습 전 V2/V3 NLL과 expected quantity가 같고, 학습 중에만 mark-specific delta가 분화됩니다.

이 초기화는 V3의 추가 parameter 때문에 공통 backbone/head 초기값까지 달라지는 confounder를 제거하기 위한 장치입니다.

## 7. Scope Kept Unchanged

이번 V3에서 변경하지 않은 항목:
- Titan `MemoryEncoder`
- LMM 및 memory mode
- causal mask와 teacher-forcing transition 구성
- RMTPP marker head
- RMTPP continuous-time likelihood head
- value input masking
- temporal/event-count lookback 정책
- `max_seq_len`
- dataset preprocessing와 mark scale
- fixed chronological split
- checkpoint selection rule

## 8. Implementation Summary

핵심 구현:
- `RMTPPConfig.value_head_mode`: `shared | mark_conditioned_experts`
- `TitanTPP.predict_value_by_mark(h)`: real mark별 residual 반환
- `TitanTPP.predict_value(h, marks)`: 명시된 mark 또는 predicted mark expert 선택
- `TitanTPP.expected_qty_from_logits(...)`: shared residual과 mark-conditioned residual shape 모두 지원
- `TitanTPP.nll(...)`: true-mark expert residual loss와 all-expert expected quantity loss 계산
- `predict_value_for_marks(...)`: TitanTPP, RMTPP, THP 평가 경로의 backward-compatible adapter
- CLI, manifest, leaderboard, history, scale-wise metadata에 `value_head_mode` 기록
- V3 run directory에 `valuehead_mark_conditioned_experts` segment를 추가해 V2 artifact와 충돌 방지
- legacy `shared` mode는 기존 run path를 유지해 이전 artifact resume를 보호

주요 코드:
- `/Users/igwanhyeong/PycharmProjects/paper_research/models/RMTPPs/config.py`
- `/Users/igwanhyeong/PycharmProjects/paper_research/models/RMTPPs/TitanTPP.py`
- `/Users/igwanhyeong/PycharmProjects/paper_research/models/RMTPPs/value_conditioning.py`
- `/Users/igwanhyeong/PycharmProjects/paper_research/utils/training.py`
- `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/common/models.py`
- `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/common/runner.py`
- `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/common/modes/model_test.py`
- `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/tpp_experiment.py`

설계 근거 문서:
- `/Users/igwanhyeong/PycharmProjects/paper_research/.agents/results/architecture/adr-titantpp-v3-mark-conditioned-value-head.md`
- `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/model_enhancement_strategy.md`

## 9. Focused Verification

Focused pytest `6/6` passed:
1. 같은 seed에서 V2/V3 공통 state가 동일하고 zero-init output과 expected quantity가 동일한지 확인
2. V3 NLL이 finite이고 `value_by_mark` shape가 올바른지 확인
3. residual loss gradient가 ground-truth로 선택된 expert에만 전달되는지 확인
4. explicit mark가 해당 expert를 정확히 선택하는지 확인
5. appended target mark가 이전 hidden state와 logits에 영향을 주지 않는지 확인
6. V2/V3 run directory와 artifact identity가 분리되는지 확인

추가 검증:
- local `py_compile` passed
- local shared RMTPP/TitanTPP/THP model-test passed
- local V3 `small_lmm`, `mid_lmm` model-test passed
- shared/V3 zero-init exact equivalence: NLL `3.890020`, predicted quantity mean `90.139053`
- local Instacart 20-series e1 official long-epoch integration passed
- 5080 `py_compile` passed
- 5080 V3 small/mid model-test passed
- 5080 `ai_env`에는 pytest가 없어 원격 pytest는 실행하지 않았고 새 dependency는 설치하지 않음

Focused test 경로:
- `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/tests/test_titantpp_mark_conditioned_value_head.py`

## 10. Instacart Mini Smoke Gate

상태:
- `completed`
- dataset: Instacart top 20 series
- candidate: `small_lmm`
- epoch: `1`
- seed: `42`
- value head: `mark_conditioned_experts`
- runtime ERROR, NaN, Traceback 없음
- manifest, log, leaderboard, histories, scale-wise metrics, report, plots 생성 확인

Smoke test metric:
- validation NLL: `3.278132`
- validation score: `0.447405`
- validation quantity MAE: `5.883943`
- held-out test NLL: `3.175295`
- held-out test score: `0.506049`
- held-out test quantity MAE: `5.517786`
- held-out test mark accuracy: `0.523333`

해석 제한:
- 이 결과는 top 20 series, 1 epoch의 execution gate입니다.
- V3의 성능 우월성 또는 일반화 성능을 주장하는 근거로 사용하지 마세요.

## 11. Short Screening Status

`2026-07-10 10:54:55 KST` 기준:

| Order | Dataset | Version | Candidate | Epochs | Seed | Status |
| --- | --- | --- | --- | ---: | ---: | --- |
| 1 | Instacart top 20 | V3 | `small_lmm` | 1 | 42 | completed smoke gate |
| 2 | Intermittent | V2 shared | `small_lmm` | 50 | 42 | completed at `10:42:40 KST` |
| 3 | Intermittent | V3 experts | `small_lmm` | 50 | 42 | completed at `10:49:10 KST` |
| 4 | Yellow Trip Hourly | V2 shared | `mid_lmm` | 50 | 42 | in progress |
| 5 | Yellow Trip Hourly | V3 experts | `mid_lmm` | 50 | 42 | queued |

Intermittent와 Taxi 결과는 artifact reading order에 따른 분석이 끝나기 전까지 performance conclusion으로 작성하지 마세요.

실험 페이지 연결 정보:
- 실험 제목: `TitanTPP V3 Mark-Conditioned Value Head Smoke And Short Screening e50`
- 실행 스크립트: `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/scripts/run_v3_smoke_short_screening_0710.sh`
- 실험 시작 프롬프트: `/Users/igwanhyeong/PycharmProjects/paper_research/simple_lab_test/search/notion_writer_prompts/titantpp_v3_smoke_short_screening_start_0710.md`

Artifact 경로:
- Instacart V3 smoke: `~/workspace/paper_research/search_artifacts/model_enhancement_v3_insta_smoke_e1_0710`
- Intermittent V2: `~/workspace/paper_research/search_artifacts/model_enhancement_v2_inter_short_e50_0710`
- Intermittent V3: `~/workspace/paper_research/search_artifacts/model_enhancement_v3_inter_short_e50_0710`
- Taxi V2: `~/workspace/paper_research/search_artifacts/model_enhancement_v2_taxi_short_e50_0710`
- Taxi V3: `~/workspace/paper_research/search_artifacts/model_enhancement_v3_taxi_short_e50_0710`

## 12. Risks And Decision Rules

Risks:
- rare mark는 해당 expert가 받는 residual supervision이 적어 충분히 분화되지 않을 수 있습니다.
- predicted mark가 틀리면 quantity inference에서 잘못된 residual expert가 선택됩니다.
- V3a는 quantity-to-mark gradient coupling을 유지하므로 V2의 marker NLL trade-off가 그대로 남을 수 있습니다.
- single-seed e50 screening은 방향성 판단용이며 최종 모델 선택 근거가 아닙니다.

Decision rules:
- Intermittent와 Taxi에서 quantity MAE, total NLL, marker NLL, mark accuracy를 V2와 함께 비교합니다.
- quantity 개선만 있고 marker NLL이 악화되면 V3를 즉시 채택하지 않습니다.
- Taxi marker NLL trade-off가 지속되면 V3b detached-gate를 다음 ablation으로 진행합니다.
- V3a가 screening gate를 통과하면 Instacart, Intermittent, Taxi 전체 fixed-split multi-seed 비교로 확장합니다.
- V3a가 quantity 및 marker 양쪽에서 명확히 열세이면 expert 수 축소 또는 mark-grouped expert를 검토합니다.

## 13. Page Structure

페이지는 아래 순서로 구성해주세요.
1. 상단 status callout: `Design confirmed / Implemented / Focused tests passed / Screening in progress`
2. Enhancement Context
3. V1/V2/V3 Comparison
4. Architecture And Equations
5. Training Contract
6. Inference And Evaluation Contract
7. Initialization And Fair Comparison
8. Unchanged Scope
9. Implementation Map
10. Verification Evidence
11. Instacart Smoke Gate
12. Short Screening Status
13. Risks And Decision Rules
14. Next Action

작성 주의사항:
- confirmed implementation, verified behavior, experiment hypothesis를 구분해서 작성해주세요.
- V3가 marker NLL을 개선했다고 아직 쓰지 마세요. 이는 screening에서 검증할 가설입니다.
- Instacart e1 smoke metric을 성능 비교 결과처럼 해석하지 마세요.
- Intermittent 완료 로그만으로 결과 결론을 내리지 말고 artifact 분석 완료 전에는 `analysis pending`으로 표시하세요.
- V3는 Titan backbone 전체 교체가 아니라 V2 value head의 조건부 확장이라는 점을 명확히 써주세요.
- 코드 inventory만 나열하지 말고, 왜 zero initialization과 RNG preservation이 공정한 V2/V3 비교에 필요한지 설명해주세요.
- 실험 결과 상세 표는 Model Validation 실험 페이지에 두고, 이 페이지에는 설계 결정과 현재 gate 상태를 중심으로 요약해주세요.

Next Action:
- Taxi V2/V3 e50 완료 대기
- protocol artifact reading order에 따른 Intermittent/Taxi V2-vs-V3 분석
- screening 결과로 V3a 유지, V3b detached-gate, 또는 grouped expert 중 다음 구조 결정
