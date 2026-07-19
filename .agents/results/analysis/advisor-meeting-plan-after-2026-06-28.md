# Advisor Meeting Plan After 2026-06-28

- Prepared: 2026-07-19
- Reporting boundary: the advisor's 2026-06-28 reply
- Evidence cutoff: 2026-07-19
- Status: meeting scope and evidence plan fixed; slides and figures not yet produced

## 1. Meeting Outcome

The meeting should not be a chronological report of every experiment. Its purpose is
to obtain the advisor's decision on the paper's claim and the final comparison scope.
The main deck should answer three questions.

1. Did increasing the learning rate resolve the slow convergence concern?
2. What did the model changes after 2026-06-28 actually improve, under controlled comparisons?
3. Should the paper describe the method as history-conditioned demand modeling, or should it
   open a new experiment with genuine exogenous covariates?

The recommended deliverable is an eight-slide main deck with a short appendix. A separate
one-page email summary can be derived from the deck after the advisor meeting is scheduled.

### In scope

- A direct answer to the advisor's `5x` and `10x` learning-rate suggestion
- Loss decomposition and the reason checkpoint selection is necessary
- The architectural path from RMTPP to TitanTPP V2 and Taxi V3b
- Confirmatory results from matched seeds and settings
- Failed branches only when they explain why the retained design was selected
- Two explicit paper-scope choices for the advisor

### Out of scope

- A diary of all V1-V7 and M/Q runs
- Treating smoke tests or single-seed screening as final evidence
- Claiming that RevIN is generally ineffective
- Presenting V5b as an empirical result; only its design has been fixed
- Calling observed event quantity an exogenous variable

## 2. Working Thesis

The claim supported by the current implementation is:

> RMTPP-style event likelihoods are retained while the recurrent history encoder and the
> quantity prediction path are redesigned to represent history-dependent demand scale.

This wording matches the code and the completed comparisons. TitanTPP reads past marks,
inter-event times, and observed event quantities. V3b further predicts a mark-conditional
quantity residual and blocks the quantity objective from updating the mark probabilities
through the expected-quantity path.

The phrase "improvement for exogenous effects" is not yet supported. No external signal such
as calendar, weather, promotion, or price is supplied to TitanTPP in the completed experiments.
If the advisor intends `exogenous` literally, that becomes a new data and model track rather
than a relabeling of V2 or V3b.

## 3. Questions And Current Answers

| Advisor question or likely challenge | Evidence-based answer | Limit |
| --- | --- | --- |
| Was `1000+` epochs necessary because the learning rate was too small? | Partly. In the Instacart RMTPP run, `5e-3` moved the best validation NLL from epoch `26` to `5`, but `1e-2` did not move it further. | This is a 50-epoch, single-seed diagnostic. |
| Did `5x` or `10x` break training? | It depended on model and data. Instacart RMTPP remained finite at all three rates, while several TitanTPP configurations failed from epoch `3-28` at `5e-3` or `1e-2`. Taxi and demand runs stayed finite, but high learning rates could still damage marker quality. | Stability and final quality are different criteria. |
| Was the model simply too small? | The long Taxi run produced late validation minima and post-minimum degradation for many RMTPP and TitanTPP configurations, so the models can reach an overfit regime. Capacity alone does not explain all slow convergence. | The long run is Taxi, seed `42`, and `1e-3`; it is diagnostic rather than a final benchmark. |
| Why not compare only total NLL? | Time NLL can dominate total NLL and can become strongly negative in the long run while marker NLL worsens. Marker NLL, time NLL, mark accuracy, quantity error, and time error must therefore be reported separately. | NLL components have different scales and should not be compared by raw magnitude. |
| What improvement is confirmed after 2026-06-28? | V2 improved quantity MAE over V1 across all three datasets. Taxi V3b then improved every matched seed over V2 in NLL, marker NLL, quantity MAE, and mark accuracy, apart from a small permitted time-NLL regression. | V3b is confirmed for Taxi only and is not a universal replacement for V2. |
| Does the model already reflect exogenous shocks? | It reflects past event history and quantity variation, including large observed quantities. It does not yet identify or ingest an external cause of a shock. | A causal or exogenous-effect claim needs new covariates and controls. |

## 4. Evidence Tiers

### Tier A: confirmatory model evidence

These results may support the main conclusion.

- V1 versus V2: three datasets, epochs `200`, seeds `42,52,62`, matched candidate selection
  protocol and completed runs.
- Taxi V2 versus V3b: epochs `50`, seeds `42,52,62`, matched dataset, candidate, lookback,
  maximum sequence length, objective, and evaluation selection.

### Tier B: diagnostic evidence

These results answer the advisor's questions but should not rank final models.

- Learning-rate diagnostics at `1e-3`, `5e-3`, and `1e-2`
- Taxi decomposition run to epoch `1000`
- Determinism A/B checks

### Tier C: design-selection evidence

These results explain why branches were stopped.

- V3a/V3c, V4a/V4b, V5a
- M0 and Q0-Q3 RevIN/magnitude branches
- V6 and V7 feasibility analyses
- V5b design, clearly labeled as not yet executed

The main deck should use Tier A for model claims, Tier B for training interpretation, and one
summary slide for Tier C. Detailed Tier C numbers belong in the appendix.

## 5. Main Deck

### Slide 1. Boundary And Executive Answer

- Start from the advisor's 2026-06-28 reply, not from the beginning of TitanTPP development.
- State the answer in three lines: higher learning rates accelerated some runs but were not
  universally stable; decomposed validation metrics changed the training interpretation; V2
  and Taxi V3b are the retained models.

### Slide 2. Learning Rate And Convergence

- Show the direct `1x/5x/10x` response.
- Contrast the finite RMTPP runs with candidate-dependent TitanTPP failures.
- Conclude that final training should use early stopping and a validated learning-rate schedule,
  not default to epoch `1000` or blindly use `10x`.

### Slide 3. Why Decomposed Metrics Were Required

- Show validation total NLL, marker NLL, time NLL, and quantity MAE on aligned epochs.
- Mark the best validation checkpoint and the final epoch.
- Explain that a lower total NLL alone can hide marker degradation.

### Slide 4. What Changed In The Model

- RMTPP: recurrent encoder with the existing mark/time likelihood heads.
- V2: Titan history encoder, observed quantity input, and quantity-aware loss.
- V3b: mark-conditional residual heads and detached quantity-to-mark gradient path.
- Explicitly mark unchanged modules and the absence of external covariates.

### Slide 5. V2 As The Common Enhancement Baseline

- Report V2 relative to V1 across Instacart, demand, and Taxi.
- Emphasize the quantity improvements, while preserving the `+1.23%` Taxi NLL trade-off.
- Use V2 as the common baseline, not as proof that every Titan configuration dominates RMTPP.

### Slide 6. Taxi V3b Confirmatory Result

- Present the matched V2/V3b seed-wise comparison.
- Highlight that the improvement is marker-led rather than time-NLL-led.
- State that all three seeds passed and that V3b remains Taxi-specific.

### Slide 7. What Was Stopped And What Was Learned

- Collapse stopped branches into four causes: gradient interference, marker imbalance,
  quantity normalization trade-off, and insufficient auxiliary history signal.
- Keep V2 for demand and Instacart, and V3b for Taxi.
- Mention V5b as a planned marker-imbalance test, not as a result.

### Slide 8. Advisor Decisions

Ask for decisions on:

1. Paper framing: current history-conditioned scale model or a new exogenous-covariate track
2. Main empirical focus: multi-dataset V2 or Taxi-focused V3b mechanism
3. Final comparison matrix and stopping rule
4. Whether V5b should be run before freezing the manuscript experiments

## 6. Figure Plan

| ID | Visual | Message | Source and construction | Caveat |
| --- | --- | --- | --- | --- |
| F1 | Learning-rate response: best-epoch dot plot plus finite/failure heatmap | `5x` can accelerate convergence, but `10x` is not generally safe for TitanTPP. | Join the RMTPP and TitanTPP LR summaries. Show RMTPP `26/5/26` best epochs and annotate Titan failure epochs. | Label `e50`, seed `42`, diagnostic. |
| F2 | Decomposed learning curves with checkpoint markers | Total NLL cannot be interpreted without marker and time components. | Recreate a clean two-model plot from the epoch-1000 histories; do not reuse the crowded six-line plot unchanged. | Explain negative time NLL and keep the y-axis labels explicit. |
| F3 | Architecture diagram: RMTPP to V2 to V3b | The likelihood heads stay RMTPP-style; the history encoder and quantity path change. | Draw from `TitanTPP.forward`, `TitanTPP.nll`, V3, and V3b ADRs. Use solid arrows for forward flow and a stopped-gradient symbol for V3b. | Do not label quantity as exogenous. |
| F4 | Three-dataset slope graph for V1 to V2 | V2 improved quantity MAE by `1.04%`, `9.94%`, and `25.86%`. | Use best-validation-NLL test summaries after per-dataset candidate selection. Add NLL change as a small secondary label. | The baseline is V1, not RMTPP. |
| F5 | Paired seed plot for Taxi V2 versus V3b | V3b's Taxi gain is reproduced for every matched seed. | Plot seed-wise NLL, marker NLL, quantity MAE, and mark accuracy; add the mean changes `-2.335%`, `-16.448%`, `-49.086%`, and `+0.729%p`. | Show time NLL `+0.181%` rather than hiding it. |
| F6 | Enhancement decision funnel | The final models resulted from explicit acceptance gates rather than trying every variant until one won. | Summarize V2 retained, V3b Taxi-only, and stopped V3c/V4/V5a/M/Q/V6/V7 branches. | Avoid dense version-by-version implementation detail. |

### Appendix tables

- Dataset profile: modeling rows, series, sequence-length distribution, quantity range
- Comparison controls: epochs, seeds, candidate, lookback, maximum sequence length, learning rate
- Metric definitions: total NLL, marker NLL, time NLL, quantity MAE, mark accuracy, time MAE
- Full stopped-branch table with the reason each branch did not pass
- Reproducibility note and artifact identifiers

Raw-source counts from earlier emails and modeling-table row counts must be labeled separately.
The deck should not present `7,891,189` raw demand rows beside `242,888` modeling events as if
they described the same processing stage.

## 7. Numbers Approved For The Main Deck

### V1 to V2

| Dataset | Score change | Quantity MAE change | Test NLL change |
| --- | ---: | ---: | ---: |
| Instacart | `+0.000377` | `-1.04%` | `+0.04%` |
| Demand | `+0.007727` | `-9.94%` | `-0.32%` |
| Taxi | `+0.016058` | `-25.86%` | `+1.23%` |

### Taxi V2 to V3b

| Metric | Change |
| --- | ---: |
| Total NLL | `-2.335%` |
| Marker NLL | `-16.448%` |
| Time NLL | `+0.181%` |
| Quantity MAE | `-49.086%` |
| Value MAE | `-27.303%` |
| Mark accuracy | `+0.729%p` |

The V3b row may be described as a confirmatory Taxi result because all matched seeds
`42,52,62` passed. It may not be generalized to demand or Instacart.

## 8. Claim Guardrails

### Supported

- Increasing the learning rate shortened convergence in some configurations.
- A larger learning rate did not guarantee stable or better TitanTPP training.
- Decomposing total NLL changed the interpretation of long training.
- V2 improved quantity prediction over V1 across the three evaluated datasets.
- V3b improved Taxi prediction over matched V2 runs and reduced seed variability.
- The completed model reflects observed history-dependent quantity variation.

### Not supported

- TitanTPP is inherently or universally superior to RMTPP.
- Epoch `1000` is generally required.
- V3b is the best model for all datasets.
- RevIN is unsuitable for point-process demand forecasting in general.
- The completed model estimates causal effects of exogenous variables.
- V5b improves marker imbalance.

## 9. Paper-Scope Decision

### Option A: current evidence, recommended

Frame the paper around preserving RMTPP likelihoods while improving history and quantity-scale
representation. Use V2 as the common enhancement baseline and V3b as the Taxi-specific
conditional mechanism. Keep the statement about external shocks as future work.

### Option B: literal exogenous-effect extension

Add timestamp-aligned external covariates, define which effect is expected, add an ablation that
removes each covariate, and compare against an RMTPP model receiving the same inputs. This option
requires a new dataset contract and a new experimental schedule before manuscript freeze.

The advisor should choose between these options before more architecture branches are opened.

## 10. Evidence Map

- Protocol and interpretation rules: `TEST_SESSION_PROTOCOL.md`
- Model status: `.agents/results/architecture/titantpp-model-status.md`
- LR summary: `search_artifacts/inter_yellow_lr_sensitivity_e50/lr_sensitivity_summary.csv`
- Instacart Titan LR summary: `search_artifacts/insta_lr_sensitivity_e50/lr_sensitivity_summary.csv`
- Instacart RMTPP LR summary: `search_artifacts/insta_rmtpp_lr_sensitivity_e50/rmtpp_lr_sensitivity_summary.csv`
- Long-run decomposition: `search_artifacts/nll_decomposition_yellow_overfit_e1000`
- V1: `search_artifacts/model_enhancement_v1_residual_e200_0705`
- V2: `search_artifacts/model_enhancement_v2_hybrid_e200_0705`
- Taxi V2 matched comparator: `search_artifacts/model_enhancement_v2_taxi_multiseed_e50_0710`
- Taxi V3b: `search_artifacts/model_enhancement_v3b_taxi_multiseed_e50_0710`
- Architecture: `models/RMTPPs/TitanTPP.py`
- V3/V3b decisions: `.agents/results/architecture/adr-titantpp-v3-mark-conditioned-value-head.md`,
  `.agents/results/architecture/adr-titantpp-v3b-detached-quantity-gradient.md`

Every source path must be checked before figure generation. If a path has been renamed, the deck
must reference the verified replacement rather than copying the placeholder above.

## 11. Acceptance Criteria For The Meeting Material

- Every headline number resolves to a manifest and summary artifact.
- Tier A and Tier B evidence are visually labeled and never merged into one ranking.
- V2/V3b controls list epochs, seeds, candidate, lookback, and maximum sequence length.
- Every relative change states its baseline.
- The small Taxi time-NLL regression is visible.
- `external`, `causal`, and `shock effect` are absent from the confirmed-contribution wording.
- The main deck ends with explicit decisions, not another experiment list.

## 12. Next Work Order

1. Verify and correct every evidence path in this plan.
2. Draft the eight-slide Korean narrative and speaker notes.
3. Generate F1-F6 from the frozen artifacts.
4. Assemble the deck and appendix, then audit every number against its source.
5. Prepare the one-page email summary and likely advisor Q&A.
