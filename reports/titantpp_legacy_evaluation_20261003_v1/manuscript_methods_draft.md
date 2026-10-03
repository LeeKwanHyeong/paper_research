# 기존 test 재평가의 방법론·한계 삽입 문안

## 범위와 현재 상태

요청의 적용 범위는 “결과수치 없이 원고삽입용 영문 방법론/한계 문장과 수정위치”이며, “main.tex와봉인파일편집금지”를 유지한다. 아래 영문은 고정된 방법과 해석 한계의 초안이다. 성능 수치, 순위, 개선 주장, 전체 실행 성공 주장은 포함하지 않는다. 데이터셋 수·조건 수·시드·재표집 설정은 결과가 아닌 계약의 정의다.

현재 부모 세션에서 전체 validation 검증을 진행하고 있다. 완료된 test 결과는 이 문안의 근거로 사용하지 않았다. 원고에 붙일 때 부모 세션이 최종 실행 증적을 확인하고, 미완료·실패 여부와 실제 결과를 별도로 반영한다. 기존 validation 표와 v0.7 기준선은 보존한다.

## 삽입 위치

행 번호는 이 파일 작성 시점의 [main.tex](/Users/igwanhyeong/PycharmProjects/paper_research/paper/titantpp_pakdd_2027_draft/main.tex) 기준이다. 편집 후에는 절 이름과 LaTeX label로 위치를 확인한다.

| 문안 | 원고 위치 | 적용 방법 |
| --- | --- | --- |
| A. 평가 범위 | `Data and Evaluation Scope`, `sec:data-scope`, 348행. 기존 범위 문단은 376–381행 | 모든 비교가 validation이라는 문장을 개발 비교와 후속 기존 test 재평가로 구분한다. 날짜 이동 설명은 유지한다. |
| B. 고정 비교군과 선택 | `Comparators and Controls`, 383행 및 `Metrics and Repeated Runs`, 421행 | 비교군 설명 뒤에 고정된 108조건의 범위를 추가한다. 기존 84조건 validation 설명을 삭제하거나 108조건 test 설명으로 덮어쓰지 않는다. |
| C. 순차 예측과 허용 이력 | `Data and Evaluation Scope`의 평가 범위 문단 뒤 | 예측 시점에 허용되는 관측과 고정 전처리를 명시한다. 길이 제한·단위 정의는 기존 데이터 설명에 연결한다. |
| D–F. 집계와 불확실성 | `Metrics and Repeated Runs`, 421행의 현재 설명 뒤 | 시드별 집계, 재표집 단위, 고정 비교 기준을 설명한다. 분량이 길면 상세 재표집 문단은 새로운 test 평가 부록으로 옮기고 본문에서 참조한다. |
| G. Taxi 기간 한계 | D–F 다음 및 `Discussion and Limitations`, `sec:limitations` | 기본 구간 미산출을 명시한다. 민감도 분석을 대체 기본 구간으로 승격하지 않는다. |
| H–I. 해석·비교군 한계 | `Discussion and Limitations`, 731행. 기존 평가 대기 문단은 740–749행 | 원본 checkpoint 연결이 아직 대기라는 낡은 문장을 교체한다. 전체 test 실행 완료 여부는 부모 세션이 확인한 상태만 적는다. |
| 결과와 연결 | 현재 `Computational Efficiency`, 649행 앞 | 별도 `Retrospective reevaluation on existing test splits` 절과 `sec:legacy-test` label을 추가할 위치다. 실제 표·성과 문장은 전체 실측 후 작성한다. |

기존 `tab:main`, `tab:naive`, `tab:ablation`, `tab:extension`, `tab:strata-quantity`, `tab:strata-history`의 validation 범위와 수치는 유지한다. 새 test 표는 `tab:legacy-test-main`, `tab:legacy-test-simple`, `tab:legacy-test-anchors`, `tab:legacy-test-structural`로 구분한다. 기존 train 분포와 연산 효율 표를 이번 재평가의 결과로 재분류하지 않는다.

## A. 평가 범위

```latex
The development comparisons concern validation partitions that informed
architecture choices. A separate, frozen protocol specifies a retrospective
reevaluation of the existing test splits. These splits are part of the
legacy research record, and their untouched status has not been established.
Accordingly, the reevaluation addresses the fixed models on the existing
test populations without providing an independent confirmation. Development
validation results remain separately identified throughout the paper.
```

## B. 고정 비교군과 checkpoint 선택

```latex
The reevaluation panel comprises nine learned models across four datasets
and three seeds, yielding 108 selected-checkpoint conditions. TitanTPP is
the fixed representative. Six external encoders share the numerical input
and prediction-head interface, while two structural controls assess the
current-only and all-available-history alternatives. For each condition,
the selected checkpoint is the earliest epoch attaining the minimum
validation raw-quantity RMSE. Selection remains fixed across all reported
metrics and is not repeated on test outcomes.

Two deterministic quantity rules accompany the learned models. Each rule
contributes one prediction vector per dataset from the same admitted
history, rather than three seed replicates. Because these rules are
deterministic, seed dispersion and duration NLL are undefined for them.
```

편집 메모: 논문에 이미 제시된 6개 외부 encoder 이름을 다시 나열할 필요는 없다. 두 구조 비교는 `titantpp_current_only_param_matched`와 `titantpp_all_available_history_mlp`를 가리킨다. “selected-checkpoint conditions”는 새 학습 108회를 뜻하지 않는다.

## C. 순차 예측과 허용 이력

```latex
Prediction follows chronological, one-step evaluation of the next observed
event within each sequence. At a forecast origin, the admitted history
contains only earlier observations within the frozen time window and
sequence-length limit. An earlier test event may enter a later history
only after its own prediction and observation. The current target and
future events cannot condition the forecast. Model parameters and
train-derived transformations remain fixed, with no retraining, joint
train--validation refit, or preprocessing fitted to test outcomes.
```

## D. 지표 집계와 pairing

```latex
For each seed, MAE and RMSE aggregate raw-quantity errors over the same
eligible targets with equal event weights. Duration NLL evaluates the
probability assigned to the recorded integer observation under the frozen
heteroscedastic lognormal head. After each seed's metrics are computed,
the reported result is their arithmetic mean with the sample standard
deviation. Predictions are not ensembled. Every comparison requires the
same target identities and observed-history definitions across models and
seeds. Missing or nonfinite predictions invalidate the affected comparison,
and the protocol does not replace its population with a favorable
intersection of successful predictions.
```

## E. 고정 학습 시드에 조건부인 재표집

```latex
Uncertainty is estimated by paired resampling of evaluation units,
conditional on the three fixed trained fits per learned model. The same
bootstrap weights apply to every model and seed, while the seeds
themselves are not resampled. Each draw first aggregates error sums and
target counts, then computes the seed-specific metrics and their mean
difference. The difference is TitanTPP minus the comparator. When a
reference is deterministic, its single prediction vector contributes to
each paired comparison without introducing additional independent replicates.

Intermittent resampling treats an entire site as one unit so that its
constituent series remain together. Instacart resamples whole users, and
RAF resamples whole parts. For Taxi, the primary analysis resamples
circular blocks of 168 shared calendar hours across all spatial cells.
The calendar grid includes hours without eligible targets, but no targets
are imputed into those hours. Each eligible analysis comprises 10,000
draws with fixed random-number settings and linear percentile interpolation.
Invalid empty draws are recorded without silent replacement, and their
occurrence prevents interval finalization.
```

## F. 고정 비교 기준과 다중 비교 범위

```latex
The anchor comparisons remain those selected from development validation.
RMTPP is the fixed anchor for Taxi and Intermittent, whereas S2P2 is the
fixed anchor for Instacart and RAF. For these four RMSE comparisons,
nominal 98.75\% percentile intervals apply a Bonferroni adjustment to
the four-comparison family. Although pointwise 95\% percentile intervals
accompany individual comparisons, the multiplicity adjustment covers only
the four anchor RMSE contrasts. Other metrics, structural contrasts, and
sensitivity analyses remain descriptive. These intervals characterize evaluation-sample
uncertainty conditional on the trained fits, not variability under repeated
model training or an independent assessment of generalization.
```

편집 메모: 네 anchor는 “이번 재평가 전에 고정”되었지만 연구 개발 이전에 등록된 비교군은 아니다. `preregistered`, `pre-development`, `confirmatory family`로 강화하지 않는다. Bonferroni 범위 밖의 비교에도 같은 보정이 적용됐다고 쓰지 않는다.

## G. Taxi 기본 구간 미산출과 민감도 분석

```latex
The frozen Taxi test period is shorter than the minimum of two 168-hour
blocks required by the primary uncertainty protocol. Its primary temporal
interval is therefore not estimable under that protocol, although the
point estimates remain reportable. Separate sensitivity analyses consider
24-hour and 336-hour shared blocks and whole-cell resampling. They do not
replace the unavailable primary interval, and each retains its own
eligibility and dependence assumptions. Intermittent similarly reports
whole-series resampling only as a sensitivity to its primary whole-site
analysis. Intervals based on fewer than 20 nominal units are interpreted
descriptively rather than as evidence of reliable asymptotic coverage.
```

편집 메모: Taxi 문장은 성능값이 아닌 고정 모집단의 날짜 메타데이터와 `2L` 기간 조건에서 확인했다. 모든 민감도 구간이 산출됐다는 주장은 포함하지 않는다. whole-cell 구간은 공간 셀에 조건부인 분석이며 공통 시간 충격을 반영하는 기본 구간과 동등하지 않다.

## H. retrospective 재평가의 한계

```latex
The evaluation freeze follows architecture development and earlier
exploratory analyses. It is a local specification for this reevaluation,
not a public or pre-development registration. Freezing checkpoint
identities and the analysis protocol constrains subsequent choices, but
it cannot restore an untouched evaluation population. The results must
therefore be interpreted as retrospective evidence on existing test splits.
Neither paired bootstrap intervals nor agreement across the three seeds
establishes an independent replication.
```

## I. Deep Renewal 제외에 관한 한계

```latex
Deep Renewal was excluded from the manuscript comparison after its
validation results were available, following an explicit scope decision.
This exclusion was not a pre-result scientific eligibility criterion.
Although its experiments remain in the research record, the reported
comparison is restricted to numerical-history encoders with common prediction heads,
the two simple quantity references, and the specified structural controls.
That scope does not establish superiority over native distributional
demand predictors. Retrospective comparator selection remains a limitation
even when the subsequent reevaluation panel is fixed.
```

편집 메모: 이 제외는 validation 결과가 존재한 뒤의 사용자 요청이었다. 원고에서는 “explicit scope decision”으로 기술하고, 내부 근거에는 그 시점을 보존한다. 불리한 결과 때문에 native 방법이 과학적으로 부적격이었다는 이유를 새로 만들지 않는다. Deep Renewal의 실제 성능과 비교 방향은 이 초안에서 읽거나 추론하지 않았다.

## 선택적으로 붙일 초록·결론의 범위 문장

전체 실행 증적 확인 후에만 초록 44–66행의 평가 설명에 다음 한 문장을 검토한다. 결과 문장과 결합할 때 개발 validation과 기존 test 재평가의 수치를 혼합하지 않는다.

```latex
The evaluation distinguishes development validation from a frozen
retrospective reevaluation of existing test splits, whose untouched status
has not been established.
```

결론 782행 이후에는 실제 결과를 설명한 다음 다음 범위 문장을 검토할 수 있다. 이 문장은 개선 방향을 미리 가정하지 않는다.

```latex
The resulting claims remain specific to the declared comparison panel
and legacy test populations, with uncertainty conditional on the fixed
trained models.
```

## Claim--evidence map

| 문안의 주장 | 원본 근거 | 상태 |
| --- | --- | --- |
| 기존 test retrospective 재평가, validation/test 승인, 재학습 없음 | [execution_contract.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/execution_contract.json)의 `route`, `approval`, `prediction_protocol` | 고정 계약. 전체 실행 성공 주장 아님. |
| 4개 데이터셋·9개 학습 모델·3개 시드·108개 선택 checkpoint | 같은 계약의 `approval` 및 [verification.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_final_eval_checkpoint_binding_20261003_v1/verification.json)의 `selected_conditions` | 선택 및 연결 완료 기준. |
| earliest validation RMSE 선택, 대표 모델·구조 비교의 고정 범위 | [effective_protocol.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_final_eval_runner_validation_20261003_v1/effective_protocol.json)의 `selection`, `representative`, `structural_secondary_panel` | 기존 freeze를 이번 계약이 계승. |
| 순차 관측 이력, target/future 금지, train-fitted 전처리 고정 | 실행 계약의 `prediction_protocol` | 허용 규칙. 실행 적합성은 부모 세션에서 검증. |
| event-weighted seed별 지표, 평균·표본 SD, 결정적 기준 한 벡터 | 실행 계약의 `aggregation` 및 [analyze.py](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/analyze.py)의 `_model_metrics`, `bootstrap_sums` | 코드·계약 확인. 실제 지표 미열람. |
| site/user/part 단위, Taxi 168시간·민감도, 10,000회, 고정 seed 조건부 구간 | 실행 계약의 `resampling` 및 집계 코드의 `analysis_specs`, `draw_weights`, `bootstrap_sums` | 고정 방법 확인. 구간 산출 성공은 별도 결과. |
| 고정 네 anchor와 98.75%/95% 범위 | 실행 계약의 `fixed_validation_selected_anchors`, `resampling.fixed_four_RMSE_anchor_interval_quantiles`, `multiplicity_scope` | 고정 비교 범위. 유의성 결론 없음. |
| Taxi 기본 기간 조건 미충족 | [dataset_manifest.json](/Users/igwanhyeong/PycharmProjects/paper_research/reports/titantpp_legacy_evaluation_20261003_v1/dataset_manifest.json)의 Taxi `populations.test.recorded_date_min/max`와 실행 계약의 `minimum_taxi_grid_hours_to_compute` | 날짜 메타데이터로 확인. 예측·성능과 무관. |
| 연구 개발 이후 freeze, 독립 미접근 평가 아님 | 기존 protocol의 `registration`, `eligibility` 및 현재 dataset manifest의 `not_independent_untouched_data` | 해석 한계 유지. |
| Deep Renewal의 사후 사용자 요청 제외와 연구 기록 보존 | 실행 계약의 `comparison_scope.deep_renewal`, `selection_transparency` | 승인된 범위 결정. 사전 과학적 제외로 재서술하지 않음. |

## Writing notes

- `oma-academic-writer`의 sentence-structure, academic-verb, hedging, anti-AI 리소스를 적용했다. 짧은 범위 문장과 조건절을 섞고, `specifies`, `comprises`, `retains`, `estimates`, `constrains`처럼 행위를 설명하는 동사를 중심으로 작성했다.
- 확정된 계약은 직접 서술하고, 실행 성공이나 성능 방향은 주장하지 않았다. 독립 일반화·새 학습의 불확실성·전체 native 수요 예측 방법에 대한 우월성은 주장 범위에 포함하지 않았다.
- 이 파일은 원고 삽입 후보 모음이다. 전부 본문에 이어 붙이지 말고 위치표에 따라 중복을 줄인다. 상세 재표집 설정은 부록으로 이동할 수 있다.
- 실제 결과 문장, 표 수치, 최종 caption과 완료 상태는 부모 세션이 전체 실측 뒤 확정한다. 이 작업은 `main.tex`, 봉인 코드 및 계약을 변경하지 않았다.
