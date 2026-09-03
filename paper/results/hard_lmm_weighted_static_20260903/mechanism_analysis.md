# Post-stop diagnosis and next backbone decision

## Answer and scope

The weighted-retrieval change is a weak intervention on the final quantity
prediction, especially on RAF. Taxi's small overall improvement concentrates in
the upper quantity tail, not the common body. Separately, the official joint
selector strongly disagrees with raw quantity metrics. These are observed
mechanisms/tradeoffs, not proof of a unique cause or a universal architecture fix.

Only Taxi and RAF have completed candidate runs. Intermittent stopped at e91;
Instacart's main run never started. Neither receives a final performance verdict.
No model, loss, time head, selector, installed runtime or server was changed for
this analysis. No training, optimizer step, validation inference or held-out
evaluation ran. The candidate and monitoring remain stopped.

Sources: original and candidate summaries, complete histories, quantity-stratum
CSVs, immutable checkpoints, the actual model code, and prior frozen diagnostics.
`mechanism_audit.json` contains the exact calculations and input/state digests.

## 1. What weighting actually changes

The bank remains 64 static vectors; the same cosine top-4 mechanism selects them.
Only the selected vectors' mean becomes a tau-1 softmax weighted sum. There is no
new addressing projection, history feature, interaction layer or output head.
Top-k membership is discrete; selected scores, unlike membership indices, remain
in autograd. The implementation is not a dead-gradient/no-op bug.

The following are **train-only mechanism measurements**, not benchmark metrics:
1,024 targets per dataset, uniformly sampled without replacement using fixed
seed 20260903; 16 batches of 64, eval mode, original weights never updated.

| Selected candidate checkpoint | Taxi e2 | RAF e20 |
| --- | ---: | ---: |
| Mean largest of four weights; uniform reference = 25% | 27.3846% | 25.3921% |
| Mean entropy / log(4); uniform reference = 1 | 0.996914 | 0.999895 |
| Mean residual-vector change from uniform, relative norm | 5.4198% | 0.3385% |
| Mean absolute relative quantity prediction change from uniform | 0.073576% | 0.003638% |

This counterfactual compares weighted versus uniform aggregation **inside the
same frozen candidate checkpoint**, not independently trained models. Fresh-run
differences also reflect changed optimization trajectories and cannot all be
attributed to this immediate inference effect. At the final, unselected Taxi e42
checkpoint, the direct quantity change is still only 0.219811% on this sample.

The decoder explains why a vector change need not be a useful quantity change.
For local state h, memory residual r and linear quantity head z = w(h+r)+b,
the implemented prediction is expm1(softplus(z)), mathematically exp(z).
Thus the memory-to-quantity effect is through the scalar projection w*r,
and changing r adds a multiplicative factor exp(w*delta_r) to the quantity.
Changing the vector without materially changing that projection does little to
quantity prediction. This is a structural observation, not proof that all
residuals or linear heads are inadequate.

## 2. The observed gain is mostly tail, not body

Full validation, matched seed 42 and official joint checkpoints:

| Dataset | Overall MAE change | RMSE change | Body <= train p95 MAE change |
| --- | ---: | ---: | ---: |
| Taxi | -1.8800% | -2.0855% | -0.5877% |
| RAF | +0.0151% | -0.0322% | +0.0751% |

For Taxi, the disjoint >p95 bins contain 380/8,268 targets (4.5960%) but explain
86.6526% of the net absolute-error reduction and 95.5467% of the squared-error
reduction. The <=p50 MAE worsens 0.6655%, p50-p90 worsens 0.4598%, p90-p95 improves
1.7928%, and >p99 improves 2.8605%. Count-weighted bins reconcile with the official
overall MAE and RMSE to absolute 1e-6. This is not a solved body-prediction problem.

RAF changes are very small: <=p50 MAE worsens 0.7610%, while higher-bin improvements
are slight. Its selected candidate's retrieval remains nearly uniform. No broad
improvement is supported by these two completed results.

## 3. Better raw quantity epochs already existed

The unchanged official criterion is Time NLL + log1p-quantity MSE, not raw MAE,
body MAE or RMSE. The selector correctly executed that contract.

| Taxi candidate history view | Epoch | MAE | RMSE | Time NLL | Log-MSE | Joint |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Official min joint | 2 | 50.7945 | 177.7517 | 1.366480 | 0.191182 | 1.557661 |
| Post-hoc min log-MSE | 3 | 37.3385 | 121.2552 | 1.379998 | 0.185840 | 1.565838 |
| Post-hoc min MAE | 36 | 29.9128 | 93.5509 | 1.394218 | 0.294847 | 1.689065 |

At e3, log-MSE improves 0.005341 but Time NLL worsens 0.013518 relative to e2:
the time term more than offsets the quantity gain. At e36, **both terms worsen**:
time +0.027738 and log-MSE +0.103665. Therefore this is not solely a time-loss
problem. Thirty-seven of 42 epochs have lower raw MAE and RMSE than the selected
e2, but are not better according to the official joint objective.

This predates the modification: original Hard-LMM also selects e2 (MAE 51.7677,
RMSE 181.5376) although its post-hoc min-MAE e41 has 29.7997 / 94.0659. THP also
has a different raw-MAE optimum, so this mismatch is not a Titan-exclusive bug.
The post-hoc values are **not replacement results, accepted candidates, or fair
comparisons with another model's official checkpoint**. Per-epoch body metrics
and e36 checkpoint weights are not available; overall improvement cannot be
called body improvement at that epoch.

RAF likewise selects e20, despite e25's slightly lower raw MAE (9.01884 instead
of 9.08786). Time NLL worsens 0.161182 and log-MSE 0.002648. The same pattern
appears in original Hard-LMM and THP histories.

Squared error in log space targets conditional expected log-quantity; its inverse
is not generally the raw conditional mean or median. The metadata string
`distribution_median_expm1_location` must not be treated as a theorem for direct
log-MSE. Raw squared/absolute losses target different point functionals. Matching
the evaluation metric to the intended forecast is a methodological requirement,
not permission to reselect favorable epochs retrospectively.
[Gneiting, Making and Evaluating Point Forecasts](https://arxiv.org/abs/0912.0902).

## 4. Task interference: present, but not a proven dominant cause

Gradient cosine was measured with no updates, using time loss versus log-MSE
on shared encoder plus prototype parameters, in the same 16 train batches.

| Selected candidate | Mean cosine | Negative batches | Memory-bank-only mean cosine |
| --- | ---: | ---: | ---: |
| Taxi | -0.10277 | 12/16 | -0.03064 |
| RAF | -0.00717 | 9/16 | -0.01554 |

Original Hard-LMM gives similar shared-gradient means, -0.09555 and -0.00294.
The bank-only conflict is weak; simply blocking time gradients to the bank is
not strongly justified as the main cure. Joint versus body-MAE gradients are
positive in 15/16 selected-candidate batches on each dataset. Training direction
is therefore not universally opposed to body improvement.

In a separate **same-checkpoint train counterfactual**, removing r only from
the time path changes mean Time NLL by +0.001518 (original Taxi) and -0.015701
(original RAF). Candidate values are +0.001604 and -0.014782. Quantity is unchanged
by construction when its h+r path is retained. These are small-sample inference
effects, not new validation results or a guarantee about retraining. Signs can
change at later checkpoints; complete distributions are in the JSON.

Training gradient clipping also predates the candidate: average epoch clip rates
are 57.89% versus original 58.44% on Taxi, and 90.69% versus 90.72% on RAF.
Clipping is not evidence of a new failure here, nor permission to disable safety
clipping. PCGrad addresses task-gradient conflict, but these measurements do not
justify adding it automatically, and that would alter the optimization contract.
[Yu et al., Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html).

## Recommended order, not implemented or launched

### Next: one routing-only candidate, not more weighted-memory tuning

Keep original Hard-LMM uniform top-4 and the entire quantity route h+r. Send h
without the prototype residual to the existing time head. Keep inputs, persistent
memory, quantity/time head formulas, losses, joint selection and guardrails fixed.
This adds zero parameters and tests whether coupling the same prototype residual
to time prediction is useful. It is not memory removal from quantity, a new gate,
or a claim that the existing bank is harmful to body events.

The rationale is the measured time/quantity selection tradeoff and a bounded
train-only routing counterfactual, not strong evidence of gradient cancellation.
The local encoder still serves both tasks, so this will not remove all conflict
and cannot fix log-MSE versus raw-MAE mismatch by itself. It is a **low-cost
diagnostic candidate**, not a promised solution or adopted main model.

Freeze one configuration before results; first verify unchanged quantity outputs
at identical weights, causal behavior, finite time gradients and time-only route
differences. A short, declared matched screening budget should precede another
four-dataset e300 launch. New training requires agreement; nothing was started.

### Conditional: test the local encoder rather than keep changing retrieval

If routing does not help, use the already strong THP-style local encoder as an
explicit controlled local-encoder replacement while retaining the original
Hard-LMM residual and common heads/losses. Compare against THP without that bank
and original Hard-LMM; record parameter and compute differences. This is a
component-attribution control, not automatically a novel Titans contribution.

The code differs in more than memory: Titan uses pre-norm blocks, FF width 128,
learned positional embeddings and persistent tokens; the controlled THP uses
post-norm, FF width 256 and its own attention block. Existing THP superiority
cannot isolate which difference matters. Do not silently change them all and
call it a single-component improvement. First define the local-encoder factor
explicitly and keep memory as the separate comparison factor.

Learned query/key addressing is a later possibility, not the first follow-up:
key/value separation is an established design, not by itself a new contribution.
A learned key behind only hard indices and uniform averaging has no useful
score-gradient path; an actual proposal must retain differentiable addressing
and show meaningful quantity projection changes before costly screening.
[Miller et al., Key-Value Memory Networks](https://aclanthology.org/D16-1147/).

### Separate objective study, only if the official objective is reconsidered

If raw body accuracy is the primary claim rather than joint TPP fit, freeze a
new objective/selection contract and apply it consistently to all comparators.
Do not compare a body-selected Titan to old joint-selected THP/RMTPP/NHP/SAHP.
This is an objective ablation, not evidence of backbone improvement. Existing
body/tail thresholds are not lowered to rescue this candidate.

Prior evidence argues against repeating naive shrinkage or calibration: working
smooth shrinkage passed 0/4 datasets; Taxi scalar calibration reproduced almost
all MLP gains; frozen readouts did not establish one accepted cross-dataset cell.
Intermittent's prior body gains show that improving body is possible, but seed
replication did not finish and cannot be claimed confirmed. No frozen readout
fits or full feature-cache extraction need to be repeated for this decision.

## Verification and limitations

CPU runtime: existing torch 2.7.1, one thread, no package changes. The final audit
took 26.6 seconds after an initial 26.9-second pass; the second pass adds separate
encoder/bank gradients and time-path attribution. All common measurements agree.
No GPU training was performed in either pass. Original/candidate selected and
candidate final checkpoints were checked on the same targets, with target
quantity and duration masked from history. First-batch predictions/time losses
match the official evaluator within rtol/atol 1e-5. All six state digests are
unchanged and parameter .grad fields remain unset. Outputs are finite.

This CPU diagnostic does not assert bitwise equality with historical CUDA.
The train sample is not a population validation result; 16 ordered batches and
one seed do not establish generalization or a training-trajectory causal effect.
Repeated validation inspection has informed these hypotheses. The held-out test
remains locked and is not available for choosing this or any next candidate.

Reproduction: `/usr/local/bin/python3 -s paper/scripts/diagnose_hard_lmm_weighted_static.py`.
The command refuses to overwrite an existing audit. No optimizer exists in the
script. Focused tests cover selector distinction, finite checks, gradient cosine,
softmax behavior and the decoder identity; memory/causal regression tests are
also included. Detailed verification is in `mechanism_verification.json`.

The first Notion publication attempt was rejected by the external-transfer
approval review; no alternate upload was attempted. After explicit user approval,
the analysis and proposed next work were published to the existing experiment
page on September 3. Readback verified the content, preservation of the earlier
stop/results record, and the unimplemented/unlaunched status of the next candidate.
Publication evidence is in `mechanism_notion_publication.json`. The approval was
limited to this documentation update, not model changes, training, or server work.
