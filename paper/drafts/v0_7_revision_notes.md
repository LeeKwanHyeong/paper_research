# TitanTPP v0.7 revision notes

## 1. Revision purpose

This revision abandons the v0.6 argument that TitanTPP generally improves
long-sequence modeling through an exponent-residual quantity representation.
The controlled validation experiments do not support that claim. The revised
paper instead asks a narrower question:

> Can a Titan-inspired history encoder improve continuous demand-count
> prediction over the recurrent RMTPP backbone when both models share the same
> mark-free count interface?

RMTPP becomes the **primary baseline** because the architectural contrast of
interest is recurrent state compression versus causal attention with learned
memory. THP remains a **secondary strong attention baseline** and must still be
reported. Its result defines the boundary of the contribution: TitanTPP may
improve substantially over recurrence without being uniformly better than a
standard Transformer encoder.

## 2. Current evidence status

### Completed

- The mark-free count-aware contract removes the quantity-derived categorical
  mark, residual decoder, mark loss, and mark accuracy.
- RMTPP, THP, and TitanTPP receive the same continuous event token:

  $$
  x_i=[\log(1+\Delta t_i),\log(1+q_i)].
  $$

- Every backbone uses the same event-time density and direct log-quantity
  regression head.
- RMTPP and THP have completed seeds 42, 52, and 62 on the fixed Intermittent
  split. TitanTPP seeds 42 and 52 have completed; seed 62 is still running.
- All reported values are validation results. The held-out test split has not
  been evaluated.

### Provisional validation signal

The following comparison uses the two seeds currently completed for all three
models (42 and 52). It is evidence of direction, not the final table.

| Model | Quantity MAE | Quantity RMSE | Time NLL |
|---|---:|---:|---:|
| Count-aware RMTPP | 2.8692 | 10.4910 | -3.59949 |
| Count-aware THP | 0.6196 | 1.8316 | -3.59948 |
| Count-aware TitanTPP | 0.7171 | 1.7507 | -3.59334 |

Relative to the matched RMTPP runs, TitanTPP reduces quantity MAE by about
75.0% and quantity RMSE by about 83.3%. Its time NLL is worse by approximately
0.0062, which remains within the pre-registered tolerance of 0.01.

THP changes the interpretation. TitanTPP has lower RMSE but higher MAE than THP
on both completed seeds. The current evidence therefore supports a reduction
in large quantity errors relative to RMTPP, but it does not support universal
superiority over attention-based TPP encoders.

### Evidence that must not be claimed

- TitanTPP does not qualify for a general count-prediction advantage over both
  RMTPP and THP because it loses to THP on MAE in seeds 42 and 52.
- TitanTPP does not qualify for a long-history advantage. In the
  `history > 128` stratum, it does not improve both MAE and RMSE over THP, and
  the gain over RMTPP does not increase consistently with history length.
- The exponent-residual representation cannot remain a main contribution.
  Controlled Taxi experiments found that direct log-scale regression performs
  better over most of the quantity distribution.
- Mark accuracy is not a meaningful endpoint for the revised count-prediction
  task and must not appear as a principal metric.

## 3. Revised core contributions

### Contribution 1: Mark-free count-aware event formulation

The revised formulation models each positive-demand observation as an event
with a waiting time and a continuous count. Quantity is no longer converted
into a categorical magnitude mark for the main model. The history encoder
receives transformed event time and count, while separate heads predict the
next inter-event time and the next log-transformed quantity.

The contribution is not the `log1p` transformation by itself. The defensible
point is the reformulation of a marked TPP demand interface into a count-aware
event model that predicts a concrete quantity without selecting and tuning
quantity bins.

**Evidence required**

- Direct log-quantity regression versus log-transformed categorical binning
  under the same RMTPP backbone, split, seeds, and training budget.
- Raw-scale quantity RMSE as the primary common metric.
- Raw-scale MAE and log-quantity MSE as supporting metrics.
- Train-derived p50, p90, p95, and p99 quantity-stratum errors to identify
  where binning loses within-bin resolution.

This experiment has not yet been completed. Until it is available, the paper
may describe the formulation but cannot claim that it is empirically superior
to log-binned quantity marks.

### Contribution 2: Titan-inspired encoder against recurrent history compression

The main architectural comparison fixes the count interface and changes only
the history encoder. RMTPP is the primary baseline because it represents the
event prefix through a recurrent hidden state. TitanTPP replaces that path with
causal attention and learned memory retrieval.

The current bounded claim is:

> Under an identical mark-free count-regression interface, the Titan-inspired
> encoder reduces quantity prediction error relative to the recurrent RMTPP
> backbone while maintaining comparable event-time likelihood.

The wording must remain conditional on the final three-seed result. It must not
be expanded into a claim about long histories because the history-stratified
analysis does not support that mechanism.

**Evidence required**

- RMTPP and TitanTPP seed-level and three-seed mean $\pm$ sample standard
  deviation for time NLL, log-quantity MSE, quantity MAE, and quantity RMSE.
- Paired seed improvements, not only differences between aggregate means.
- Quantity-stratified RMSE to determine whether the gain comes from ordinary
  observations or from a small number of extreme errors.
- THP results in a secondary table or analysis paragraph to disclose that a
  standard attention encoder remains competitive.

### Contribution 3: Error-regime analysis rather than a long-sequence claim

The third contribution is an empirical analysis of where each encoder helps.
The current two-seed signal suggests that TitanTPP may reduce the largest
quantity errors, particularly above the train-derived p99 threshold, even when
its average MAE is not lower than THP.

This contribution qualifies only if seed 62 reproduces the extreme-tail result.
If it does, the paper may discuss **extreme-count error reduction**. It must not
call the result broad long-tail superiority because TitanTPP is not better than
THP in every upper-quantity stratum.

## 4. Revised paper argument

### Introduction

1. Intermittent demand contains two targets: the time of the next positive
   event and its count.
2. Conventional marked TPPs naturally predict event time and categorical event
   type, but a quantity bin is not the quantity itself. Its definition also
   introduces bin-boundary and resolution choices.
3. Direct log-count prediction provides a common continuous output interface.
   Once this interface is fixed, the effect of the history backbone can be
   evaluated independently.
4. TitanTPP is introduced as a Titan-inspired count-aware TPP whose principal
   comparison is against recurrent RMTPP. The contribution is bounded to the
   observed quantity-error reduction, not asserted as a general long-memory
   advantage.

### Related work

- Classical and neural temporal point processes.
- RMTPP as the principal recurrent reference.
- THP as a strong attention-based reference and an important boundary case.
- Intermittent-demand methods that separate occurrence time and demand size.
- Target transformations and discretization for skewed count prediction.

### Method

1. Define events $e_i=(t_i,q_i)$, history $\mathcal H_i$, and targets
   $\Delta t_{i+1}$ and $q_{i+1}$.
2. Define continuous event tokens using `log1p` time and quantity.
3. Describe the Titan-inspired causal encoder without claiming unbounded or
   test-time memory.
4. Define the time head and direct log-quantity head.
5. Define the joint training and checkpoint objective:

   $$
   \mathcal L_{\mathrm{joint}}
   =\mathcal L_{\mathrm{time\text{-}NLL}}
   +\lambda_q\operatorname{MSE}(\hat z,\log(1+q)).
   $$

6. Report time NLL and quantity regression loss separately. The joint objective
   must not be labelled as NLL.

### Experiments

The experiments answer two controlled questions.

1. **Quantity-interface comparison:** With RMTPP fixed, does direct
   log-quantity regression improve raw-scale RMSE over log-transformed
   categorical binning?
2. **Backbone comparison:** With direct regression fixed, does TitanTPP improve
   count prediction over RMTPP?

THP remains a secondary strong baseline. It is not removed from the evidence;
instead, the manuscript explains that it tests whether any improvement is due
only to replacing recurrence with attention.

## 5. Experiment and evidence plan

### Main table: RMTPP versus TitanTPP

Report the following on the frozen held-out test split after validation
selection and configuration freeze:

| Metric | Role |
|---|---|
| Time NLL | Common event-time likelihood comparison |
| Log-quantity MSE | Quantity training-space error |
| Raw quantity RMSE | Primary count metric; sensitive to large errors |
| Raw quantity MAE | Secondary count metric; typical absolute error |

The main text may emphasize RMTPP as the primary comparison, but the caption
must state that both models share the same count head and differ only in their
history encoder.

### Secondary table: THP boundary comparison

Report THP with the same metrics. The interpretation should state whether
TitanTPP improves RMSE, MAE, both, or neither relative to a standard causal
Transformer. This prevents the RMTPP comparison from being read as evidence
that learned memory is necessarily better than attention.

### Quantity-interface ablation

At minimum, compare:

| Variant | Prediction target | Outputs |
|---|---|---|
| Log-binned mark | Category derived from transformed quantity | Time density + categorical quantity distribution |
| Direct log-quantity regression | $\log(1+q)$ | Time density + continuous quantity |

For log-binning, reconstruct a numeric quantity using a train-fitted bin
representative so that both variants can be compared using the same raw-scale
RMSE and MAE. Mark accuracy may be recorded for diagnostics but is not a main
paper metric.

The number of bins and reconstruction rule must be selected using training and
validation data only, then frozen before held-out evaluation.

### Figure plan

- **Figure 1:** Revised mark-free TitanTPP schematic. Remove magnitude-mark and
  residual paths. Show event sequence, continuous time/count token, Titan
  history encoder, time head, and log-count regression head.
- **Figure 2:** Quantity RMSE by train-derived quantity stratum for RMTPP, THP,
  and TitanTPP. Use p50, p90, p95, p99, and p99+ strata.
- **Optional Figure 3:** Direct log regression versus log-binning, emphasizing
  bin-resolution error rather than mark accuracy.

## 6. Held-out test protocol

The held-out test remains locked until the following are complete:

1. TitanTPP seed 62 finishes and the three-seed validation result is qualified.
2. The log-binning variant and its bin reconstruction rule are frozen.
3. The final list of models, metrics, and figures is fixed.

Each validation-selected checkpoint is then evaluated once on the held-out
test split. Results must be decomposed by objective type:

- Direct regression models: time NLL, log-quantity MSE, raw MAE, and raw RMSE.
- Log-binned model: time NLL, quantity-mark NLL, reconstructed raw MAE, and
  reconstructed raw RMSE.

No combined NLL should compare a categorical model with a regression model,
because the quantity objectives are defined on different sample spaces. Time
NLL and reconstructed quantity error provide the common comparison.

## 7. Claim-evidence map

| Candidate claim | Evidence | Current status |
|---|---|---|
| Direct count prediction avoids quantity-bin tuning | Mark-free formulation | Formulation established; empirical comparison pending |
| TitanTPP improves quantity prediction over RMTPP | Matched count head, fixed split, paired seeds | Strong provisional signal; seed 62 pending |
| TitanTPP preserves event-time modeling quality | Time NLL degradation within 0.01 | Supported on completed seeds; final aggregation pending |
| TitanTPP improves over THP | Lower RMSE but higher MAE on seeds 42 and 52 | Trade-off only; no general superiority |
| TitanTPP improves extreme-count prediction | p99+ MAE/RMSE on seeds 42 and 52 | Promising; seed 62 required |
| TitanTPP captures long histories better | History-stratified metrics | Not supported; remove from manuscript |
| Exponent-residual improves long-tail prediction | Quantity-interface controls | Not supported; remove as main contribution |

## 8. Wording rules for the next manuscript

### Permitted after final validation confirmation

- “TitanTPP substantially reduces count-prediction error relative to the
  recurrent RMTPP backbone under an identical direct-regression interface.”
- “The improvement is concentrated in RMSE and extreme-count errors, whereas
  comparison with THP reveals a trade-off in average absolute error.”
- “Event-time likelihood remains comparable within the pre-specified
  tolerance.”

### Prohibited

- “TitanTPP is superior to existing TPP models.”
- “TitanTPP captures long-term dependencies better.”
- “Exponent-residual modeling solves long-tail quantity prediction.”
- “TitanTPP achieves the best performance on every metric.”
- Any claim based only on validation results after held-out results become
  available.

## 9. Remaining work order

### In progress: Complete the mark-free backbone experiment

- Finish TitanTPP seed 62 on the 5080 server.
- Verify all nine checkpoints, split hashes, source revision, and absence of
  held-out test artifacts.
- Aggregate paired three-seed results and reapply the RMTPP-primary,
  THP-boundary interpretation.
- Completion condition: a frozen validation qualification report.

### Next task: Implement the log-binning control

- Fix the RMTPP backbone and compare log-binned categorical prediction with
  direct log-quantity regression.
- Freeze bin boundaries and numeric reconstruction values from training data.
- Use raw quantity RMSE as the primary common metric.
- Completion condition: three-seed validation results with identical event and
  stratum membership.

### Next task: Freeze the revised manuscript contract

- Select the final model rows, metrics, claims, tables, and figures.
- Decide whether the p99+ result qualifies as a reproducible analysis claim.
- Completion condition: no unresolved model or metric choice remains.

### Final evaluation: One-time held-out test

- Evaluate only the frozen validation-selected checkpoints.
- Report time and quantity components separately.
- Completion condition: final test tables, stratified figure, and artifact
  manifest are ready for manuscript use.

### Manuscript revision

- Rewrite the abstract, introduction, method, experiments, and conclusion from
  the frozen evidence.
- Replace the exponent-residual architecture figure with the mark-free model.
- Do not target the abandoned ICTC deadline; use the revised contribution to
  select a realistic venue and schedule.
