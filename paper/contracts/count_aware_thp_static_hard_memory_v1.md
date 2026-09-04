# Count-aware THP + Static Hard Memory: Prospective Contract

## Status: Local Contract Only

Frozen on 2026-09-04 in local `paper_research/master`, against reviewed code
`af14bcfba59fef6f5b9f1d7c4682fba804beddaa`. The JSON companion is authoritative.
This task permits document tests and local baseline identity audits, NOT candidate
implementation, server access/sync, CUDA/e1, training, scheduling or publication.
No candidate performance exists. Earlier Local Time and weighted-retrieval hold
decisions stay intact; nothing is resumed. No push is authorized.

## Decision and Hypothesis

Test one original static prototype bank added to the existing project THP. Ask
whether it improves overall RMSE and extreme-tail MAE while preserving THP body
and overall MAE. Do not assume the encoder is the proven bottleneck, that memory
is necessarily useful, or that combining separately good components will work.

| Model | Local representation | Static prototype residual | Role |
| --- | --- | --- | --- |
| Existing CountAwareTHP | original THP h | none | direct control |
| New Count-aware THP + Static Hard Memory | identical THP h | original uniform hard top-4 r | single-factor candidate |
| Existing Hard-LMM (`titantpp`) | Titan encoder including persistent tokens | original hard top-4 | secondary compatibility reference |

Direct attribution is THP versus THP+bank. Comparing the candidate with original
Hard-LMM changes the local block, FF width, normalization, positional embedding
and persistent-token configuration as well; it is NOT a pure memory comparison.
Even the direct comparison adds 4,096 parameters, so it tests the memory-module
package, not retrieval versus a parameter-matched alternative. A later novelty
or efficiency claim would need additional controls. This is not original Titans
LMM/MAC, and not an adopted TitanTPP replacement.

## Exact Single Change

Reserved ID `thp_static_hard_memory`, role `t0_thp_static_hard_memory`. Preserve
legacy IDs and caches; do not register this candidate in the contract-only phase.

Keep `CountAwareTHP` exactly: Linear(2,64) on log1p inter-event time and quantity,
two THPEncoderLayer blocks, four heads, FF width 256, post-norm, GELU, dropout 0.1,
LayerNorm epsilon 1e-6, original causal/padding masks. This specific count-aware
adapter has NO learned positional embedding, sinusoidal timestamp encoder,
persistent tokens or RNN. Do not import `THPTemporalEncoder` instead or add
temporal encodings because the original THP paper uses them.

After the final masked THP state h, append one `HardLocalMemoryMatcher` with a
trainable [1,64,64] bank, initialized by randn*0.02. Normalize queries and keys
for cosine top-4 selection, but average the four ORIGINAL unnormalized values.
Return `(h+r)*valid_mask` to BOTH existing heads, without another normalization,
activation, gate, projection, weighting or time-only bypass. Compute the base
encoder and matcher once per forward. `encode_task_states` returns time first,
quantity second; both share the same undetached state.

The bank has 4,096 parameters; baseline THP has 100,291, hence expected candidate
104,387 on both datasets. This is an arithmetic expectation, not an executed
candidate parameter-count test. Never reduce FF width to hide the added capacity.
Static prototypes are outer-optimizer parameters shared across series, NOT
attention persistent tokens or series history. There are no online writes,
momentum, forgetting or series-specific buffers to reset. Evaluation freezes
all parameters. Hard selection does not provide a differentiable score-index
path; gradients reach selected values and the direct h path.

Construct the original THP parameters first, retaining names, shapes and RNG
order. Initialize only the new bank inside a CPU `torch.random.fork_rng(devices=[])`
context to preserve the caller RNG. Shared parameters must initially match THP
exactly for the same seed. All parameters then train fresh; no old or smoke
checkpoint supplies training weights. Zero-residual parity is a diagnostic-only
test with copied shared weights; nonzero memory is not required to reproduce THP
predictions. Test nonzero head fixtures to avoid vacuous zero-initialized gradients.

## Baseline Reuse: Verified with Boundaries

The dedicated baseline registry pins original Taxi/RAF seed42 THP and Hard-LMM
summary, history, best/last checkpoint file hashes, canonical best-state hashes,
source revision, data/split identities, contexts and strata. It does not inherit
calibration/shrinkage training from the old frozen-probe registry. Four local
reference audits passed without model construction, inference or data materialization.

Reuse only if full data/split/cutoff/target counts, mark-free log-MSE interface,
legacy time head, optimizer, batch/lr, e300/min40/patience40 and strict joint
selection still match at launch. The historical and reviewed CountAwareTHP and
THPEncoderLayer class ASTs match. Source revisions can differ only with reviewed
unrelated or declared single-factor changes; do not infer semantic equality from
matching option names alone. A missing or mismatched identity blocks comparison,
not permission for automatic reruns or substitution with a 3-seed average.

Historical `split_rows` includes test counts. The references report no held-out
evaluation and have no run-local test result artifacts; this does NOT prove that
test rows were never materialized historically. The NEW candidate must exclude
test before materialization and reproduce the pinned train/validation target
counts and strata. No test rows or test metrics are opened in this task; files
are hashed as bytes, not parsed as dataset rows. Historical runtime differences
are disclosed; validation reuse is not bitwise or matched-runtime CUDA replay.

| Dataset | Lookback / max sequence | Train / validation targets | THP best / completed | Hard-LMM best / completed |
| --- | --- | --- | --- | --- |
| Taxi | 168 hours / 256 | 38,393 / 8,268 | 25 / 65 | 2 / 42 |
| RAF | 84 months / 84 | 25,779 / 6,690 | 20 / 60 | 20 / 60 |

THP body MAE is better than Hard-LMM on Taxi (20.435580 versus 23.167407), but
worse on RAF (4.349096 versus 4.225942). This is why no single baseline is silently
used to weaken the goal. RMTPP/NHP/SAHP and seeds52/62 are not rerun or inspected
for the current seed42 gate. Existing Hard-LMM weights remain unchanged.

## Frozen Learning and Prediction Contract

Fresh all-parameter training, seed42, batch128, lr0.001, e300/min40/patience40.
AdamW: one group, betas(0.9,0.999), eps1e-8, weight decay0.01, time LR multiplier1.
Global L2 clipping1 across all trainable parameters. Direct log1p quantity MSE
weight1, tail weight0, original quantity head/softplus/expm1 and legacy time head.
Launch time scale3, slope-cap argument10/3, intercept limit30, safety40, initial
b_t0 and w_raw-3. Train-derived scale statistics are separate, not launch overrides.
No new loss, PCGrad, scheduler, AMP or compile policy. Keep existing seeded
warn-only deterministic loaders, zero workers, shuffle train only, no drop-last.

Use the earliest strict minimum of full-validation Time NLL+log-MSE. Ties retain
the earlier epoch; epoch0 is ineligible. No post-hoc raw-MAE/RMSE choice, headwise
selection or checkpoint continuation from smoke. A different backbone does not
by itself resolve the objective's mismatch with raw MAE/RMSE.

Prediction uses position length-2 after right-padding. The target quantity is
masked to zero and the causal mask blocks target duration/future tokens from
the history representation; true next duration is used only for density scoring.
Padding must remain zero after retrieval. Test prefix, batch and series isolation,
both-task gradients, immutable evaluation parameters and full checkpoint replay.
Reject incomplete/wrong-route caches or checkpoints rather than `strict=False`
loading them into a seemingly compatible model.

## Prospective Acceptance: Both Comparisons Must Pass

The new direct-control gate below is frozen BEFORE candidate implementation or
results. The 2% minimum effect is an engineering screening criterion, not a
statistical threshold or a previously inherited rule. The prior Hard-LMM gate
is preserved separately and is NOT relaxed or retroactively applied to old results.

| Metric | Versus original THP | Versus original Hard-LMM |
| --- | --- | --- |
| Body <=train p95 MAE | no worsening | >=5% improvement |
| Overall MAE | no worsening | report |
| Overall RMSE | >=2% improvement | <=2% worsening |
| >train p99 MAE | >=2% improvement | <=2% worsening |
| Time NLL | increase <=0.01 | increase <=0.01 |

Body = sum(count*MAE)/sum(count) over le_p50,p50_p90,p90_p95; tail = gt_p99.
Use unrounded finite values and inclusive boundaries. Empty/missing strata,
nonfinite data or nonpositive quantity reference denominators are not_evaluable,
never pass. Time uses an absolute difference even when NLL is negative.

| Dataset | Body MAE max | Overall MAE max | RMSE max | >p99 MAE max | Time NLL max |
| --- | --- | --- | --- | --- | --- |
| Taxi | 20.435580 | 39.108153 | 137.161078 | 789.616462 | 1.370333 |
| RAF | 4.014645 | 9.092432 | 35.497627 | 309.425391 | 3.285820 |

Rounded limits here are explanatory; registry and formulas retain full precision.
Every condition must pass on BOTH datasets, never an average that hides one
failure. A finite but failing result is hold_no_expansion, not a technical error.
Passing only makes the candidate eligible for separately approved follow-up;
one validation seed cannot establish significance, generality or model adoption.
Report p95/p99/history metrics, joint/log-MSE/time, selected epochs, parameter
delta, epoch/startup times and peak VRAM. No cost ratio without matched evidence.

## Remaining Work: Serial, Not Authorized Yet

1. Separate implementation approval: implement the distinct candidate and pass
   the 16 model requirements in the JSON, including zero-residual THP parity,
   gradients, causality and checkpoint/cache identity. Current document tests
   are NOT those model tests; status is `not_run_candidate_not_implemented`.
2. Separate server approval: source checksum sync without --delete to 5080;
   local tests precede CUDA contracts, which precede full Taxi/RAF e1. No live
   server readiness or candidate runtime performance is claimed now.
3. Separate performance approval: two fresh seed42 candidate runs only, same
   epoch contract. Run Taxi then RAF independently, even if Taxi fails the
   performance gate; technical failure records an atomic failed status and stops.
4. Audit results against both baselines and stop on any gate failure. Further
   datasets/seeds need both gates passing and a separate decision/contract.

Future launch requires inactive GDM/graphics, no competing compute, free VRAM
>=12,000 MiB and readable kernel logs without recent Xid/OOM. Service changes
require explicit approval; no automatic restart, resume, retry or overwrite.
No commands, training revision, scheduler or finish-time estimate are invented
before implementation. This contract does not start steps 2 or 3.

Alternatives rejected for this phase: combining the held Local Time path,
retuning retrieval temperature/gates, changing loss/selection, replacing all
Titan block details at once and calling the result a pure memory improvement,
or claiming a novel original-Titans mechanism. The control tests added value
of this specific static memory under an otherwise unchanged project THP.
