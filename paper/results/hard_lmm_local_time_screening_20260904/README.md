# Hard-LMM Local Time: Approved Fresh Screening

## Scope and Baseline

On 2026-09-04 the user explicitly approved Taxi and RAF seed-42 fresh e300
screening on 5080. This extends the authorization after the completed CUDA/e1
phase, not the model or performance contract. Historical authorization fields in
`paper/contracts/hard_lmm_local_time_v1.json` describe the original freeze and
are preserved unchanged.

- Candidate: `titantpp_hard_memory_local_time`, role `t0_hard_memory_local_time`.
- Training revision: `b9d0ac0b5052102d32a7bce57bff294d75576bdb`.
- Frozen server source: `/home/leekwanhyeong/workspace/paper_research_local_time_b9d0ac0`.
- Passed smoke audit: `paper/results/hard_lmm_local_time_20260903/smoke_verification.json`.
- Taxi then RAF, each in a fresh independent Python process; no smoke weights,
  resume, retry, benchmark retraining, or extra CUDA/e1 runs.
- All original parameters train from fresh initialization; quantity remains
  uniform Hard-LMM `h+r`, time receives local `h` including persistent attention.
- e300/min40/patience40, batch128, AdamW lr0.001, direct log-MSE, tail loss zero,
  legacy time head, fixed split, full train/validation and joint checkpoint
  selection are unchanged. Held-out test stays closed.
- Each dataset must improve body MAE by at least 5%, with overall RMSE and
  gt_p99 MAE deterioration at most 2% and Time NLL increase at most 0.01.
  Both datasets are evaluated even if Taxi fails the performance gate.
  Missing/empty/zero-reference evidence is not evaluable, never a pass.

## Execution Preparation

`paper/scripts/run_hard_lmm_local_time_screening.py` is separate orchestration.
It verifies the pinned training manifest and every frozen source file, checks
the exact smoke-tested Python/PyTorch/CUDA versions, baseline/data identities,
and re-audits existing e1 artifacts without modifying them or running inference.
The child imports all trainer modules from the immutable training snapshot;
the orchestration snapshot must have its own committed source manifest.

Before each dataset it requires inactive GDM, no desktop or CUDA workload,
at least 12,000 MiB free VRAM, and readable kernel logs without recent Xid/OOM.
It records child CUDA peak allocated/reserved bytes and observed epoch timing
without changing the trainer, RNG, gradients, compilation or optimizer.
Existing output roots are rejected before any write. Failures are recorded
atomically and do not start the next dataset. No service or runtime changes
are authorized by this launch approval.

Planned artifact (not created on 5080):
`search_artifacts/hard_lmm_local_time_seed42_e300_20260904`.
Planned tmux (not started): `hard_lmm_local_time_seed42_0904`.

## Local Verification

`local_tests.xml`: **155 passed, 1 conditional CUDA skip**, using existing local
Python 3.12.10 / PyTorch 2.7.1, without installing or changing dependencies.
Coverage includes the ten prior model/contract modules plus the fresh-screening
module: exact approved command differences, fresh-output protection, no retry
or next-run execution after failures, fixed gate boundaries and missing evidence,
sequential evaluation despite performance rejection, and an isolated child import
test proving that training resolves modules from the frozen source root.

The extended artifact audit accepts the real history envelope for both e1 and
e300, requires earliest joint-minimum checkpoint selection and the original
patience rule, checks full target counts, and reconciles the final checkpoint
against the saved best state rather than the last epoch's current weights.
Synthetic tests include drift in optimizer, route, digest, scope and finiteness.

`prelaunch_verification.json`: the two existing full-e1 artifacts pass read-only
re-audit, including pinned dataset/baseline identities and checkpoint hashes.
Every file hash in both artifact directories is unchanged before/after the check.
There were no forward passes, optimizer steps or dataset inference in this audit.
Model, trainer, common training code and frozen contract have no diff against
the smoke-tested `b9d0ac0` revision. `git diff --check` passed.

## Current Blocker

The initial SSH preflight and one bounded connection retry both timed out at
`5080` port 22 on 2026-09-04. No source was transferred and no training was
launched in this session. Current GPU, GDM and kernel health are unknown;
yesterday's healthy smoke postflight is not a current observation.

No training monitor has been created because no training job is running.
The user has been asked to check 5080 power/Tailscale connectivity. There is
no automatic connection polling, service restart, VPN change or remote retry.

The attempted update to the existing Notion experiment page was rejected by
the permission reviewer because the proposed content included internal server
paths, revision identifiers and experiment/connectivity status. No Notion write
succeeded, and no alternate publication route was attempted. Explicit approval
for that destination/content is required before retrying publication. This does
not invalidate the user's separate approval to run the two experiments on 5080.

After connection recovery: verify live preflight and immutable source, transfer
only the committed orchestration package without `--delete`, validate package
checksums, start the two-run tmux job, confirm its first epoch, and create one
hourly monitor. Notify on meaningful progress/completion/failure, not unchanged
state. Do not revive the cancelled weighted-retrieval monitor.

After completion: sync artifacts without `--delete`, audit manifest, log,
summary, held-out absence, histories, quantity/history metrics, checkpoint
digests and plots; report fixed-gate decisions and computational cost; update
and re-read the existing Notion result page; commit only related evidence to
local `paper_research/master`. No push or seed/dataset expansion is authorized.

Prior e1 measurements were 12.235 seconds (Taxi) and 2.662 seconds (RAF).
Linear e300 extrapolation is approximately 74.5 minutes total, excluding
startup and audit; early stopping can shorten it substantially. This is an
estimate from one epoch, not a measured e300 runtime or an absolute finish time.
