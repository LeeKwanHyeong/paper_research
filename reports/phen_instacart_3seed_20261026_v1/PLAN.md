# PHEN Instacart 3-seed extension

## Scope — fixed before training

The user authorized Instacart experiments on both 5080 and 5090, then explicitly
selected **PHEN only, three seeds, three fresh fits**. No component variants,
external baselines, MAC, tuning, warm starts, or scientific retries are included.
Existing results, checkpoints, manuscripts and stopped predecessor processes are
preserved. This phase ends with selected-checkpoint Validation and replay;
Test is not loaded or evaluated in this phase.

## Current baseline — verified

- The previous RAF/Taxi/Intermittent component campaign completed separately.
- Both GPUs were idle at the 2026-10-26 08:10 KST observation. The 5080 still
  holds the original stopped predecessor processes and their reserved memory.
- Instacart's frozen Train/Validation export is byte-bound and matches the
  corresponding original rows. It contains 2,197,401 Train rows and 503,733
  Validation rows; the next-event target counts are 1,991,192 and 503,733.
- This dataset has historical held-out evaluation exposure. A later Test result
  must not be described as evaluation on previously untouched data.

## Prepare an isolated execution — in progress

- Reuse the exact PHEN full-model factory and joint objective used in the paper.
  Do not modify the frozen four-component campaign or its control records.
- Use the existing 52-day input window and max sequence length 64, including
  the target slot. History-availability gates remain 1,2,4,8,16,32,64,128;
  all gates reuse the current/previous-state relation.
- Fix seeds 42,52,62; batch 128; AdamW lr 0.001, weight decay 0.01; gradient
  clipping 1; maximum 300 epochs, minimum 40, patience 40. Select the earliest
  strict finite minimum of full-Validation quantity RMSE. No result-dependent
  schedule or hyperparameter changes.
- Preserve Train-derived quantity and recorded-duration statistics. Use
  positive-integer day observations with top code 30. Large quantity is strict
  q > 35, not the historical tail-loss threshold 25. Tail loss remains zero.

## Verify and launch — next

- Run focused local contract and scientific-path tests and independent review.
- Commit only new task code and documentation. Bundle immutable tracked source
  and Train/Validation inputs into new directories under each server's
  paper_research/search_artifacts. Verify every bundled file after transfer.
- Qualify the full model on each native GPU using synthetic forward/backward
  checks and batch128 × sequence64 capacity. These checks are not experiment
  results and do not update any scientific checkpoint.
- Run seed42 on 5080; seed52 then seed62 on 5090. The hosts run concurrently;
  each GPU admits at most one new fit. A GPU/boot/process guard precedes each
  fit. Use tmux and a parent that records actual child exit codes.
- Verify initial configuration, GPU child identity and first training entry.
  Do not add a recurring automation or continuous polling for this request.

## Evidence and follow-up

Keep source/input/contract hashes, actual stdout/stderr/exit records, full epoch
history, selected checkpoint binary/tensor hashes, time-head training proof,
full Validation predictions and an independently reconstructed-model replay.
No completion claim is made until actual exits and replay checks pass.

After the three runs finish, aggregate Validation metrics as mean and sample SD
(ddof=1), retaining each seed and adverse outcomes. Any Test/cost stage is a
separate follow-up to these fixed checkpoints; no additional fitting is needed.

ARS experiment-agent run workflow is used for execution/provenance review.
The explicit execution request supplies launch authorization; no redundant
confirmation is required for the same three fits.
