# Hard-LMM Local Time: Approved Fresh Screening

## Latest Status: Fresh Screening Running

2026-09-04 **13:16:00 KST**: started the approved Taxi-to-RAF seed-42 fresh
screening on 5080, tmux `hard_lmm_local_time_seed42_0904`, artifact
`search_artifacts/hard_lmm_local_time_seed42_e300_20260904`.
The user completed GDM shutdown. Preflight confirmed GDM/graphics inactive,
no CUDA compute jobs, 15,825 MiB free VRAM, no recent readable kernel Xid/OOM,
unchanged source manifests, and a nonexistent output directory before launch.
No GDM/sudo retry, new source transfer, CUDA/e1 rerun or model edit was needed.

`launch_5080.json` preserves the exact command and source revisions. Initial
observation at **13:16:42 KST** (`initial_observation_5080.json`) confirms the
parent tmux and CUDA child PID 4675, Taxi epoch 3 complete / joint-best epoch 2,
zero of two completed runs, GDM inactive, GPU 2,252 MiB used / 76% utilization.
The actual contract is e300/min40/patience40, seed42, full train/validation,
dedicated local-time backbone/role, frozen training revision and validation-only.
The observed epoch had 38,393 train events and finite train/validation metrics.
These are startup checks, not a completed performance result or gate decision.

An hourly thread heartbeat **`hard-lmm-local-time-5080`** is ACTIVE. It performs
one check per run, reports meaningful changes and required actions, avoids
unchanged-state noise and extra polling, and never retries/resumes training.
It audits/synchronizes completed artifacts without `--delete`, compares the
fixed gates, updates and re-reads the existing Notion page, commits related
results to local `paper_research/master`, and pauses after completion or a
clear failure. No benchmark retraining, 5090 work, push or held-out evaluation.

Initial ETA, conditional on unchanged speed: approximately **13:30-14:35 KST**.
Taxi epochs 2-3 averaged 11.372 seconds; RAF still uses the prior full-e1
measurement of 2.662 seconds, not a current measured epoch. A no-further-best
Taxi stop at e42 followed by minimum-length RAF would reach about 13:26 before
startup/audit overhead. Both running to e300 extrapolate to about 14:26 before
overhead. Best-checkpoint improvements, contention and speed changes can extend
these estimates; the monitor should replace them with actual RAF speed later.

Next: leave both approved runs uninterrupted. At completion, validate the full
evidence and report official joint-selected quantity/body/tail/time metrics and
costs against the fixed Hard-LMM baseline. Do not infer success from early epochs.
Keep GDM inactive during training. If the user needs their desktop after the
job, they can run `ssh -t 5080 'sudo systemctl start gdm'`; do not request the
sudo password in chat or automatically restart the graphical service.

## Earlier State: Awaiting User Sudo Authentication

2026-09-04 12:51 KST: the user explicitly approved temporarily stopping 5080
GDM after the graphical-session termination risk was explained. The authorized
`sudo -n systemctl stop gdm` returned `sudo: a password is required`; GDM was
not stopped. Approval is no longer missing. The remaining action is for the user
to authenticate directly in their terminal; do not request a password in chat,
search for stored credentials, kill the desktop instead, or bypass preflight.

```bash
ssh -t 5080 'sudo systemctl stop gdm'
```

Non-GPU preparation is now complete: the committed 727,698-byte package was
transferred to 5080, verified against its archive digest, and unpacked into the
new `/home/leekwanhyeong/workspace/paper_research_local_time_screening_484a052`
snapshot without overwriting either the frozen model tree or any artifacts.
All 253 source/contract/report hashes and the immutable training manifest pass.
The screening readiness function also validated the exact runtime, pinned
baseline/data identities and both existing e1 artifacts, without training or
inference. Evidence: `server_readiness_1253.json`, checked at 12:53 KST.
GDM was still active; training and its hourly monitor have not started.

Next: after the user completes sudo authentication, confirm GDM/graphics are
inactive and the unchanged GPU/kernel preflight passes. Then launch the prepared
Taxi-to-RAF job, confirm the first epoch, and create its hourly monitor. Do not
ask for GDM or training approval again, and do not rerun CUDA/e1 or repackage
the already verified source unless its integrity check fails.

## Earlier State: GDM Approval Required

2026-09-04 12:43 KST: SSH connectivity to 5080 is restored. The frozen training
manifest and all source checksums match; Python 3.12.13, PyTorch 2.11.0+cu130
and CUDA 13.0 match the successful smoke environment. No CUDA compute job or
tmux session is running. NVIDIA-SMI reports 184/16,303 MiB used, utilization 0%,
with gnome-shell (149 MiB) and Xwayland (6 MiB). The accessible kernel log has
no Xid/OOM in the last hour. These are read-only observations, not a stress test.

GDM is active, so the original headless preflight correctly blocks launch.
The user has been asked to approve stopping GDM temporarily because doing so
can terminate the graphical session and its applications. GDM has not been
stopped; no process or service has been changed. Source transfer, fresh training
and its scheduler have not started. The intended output artifact does not exist.

The committed orchestration revision is `484a052cc820c111dcd514e90ee294bcba59cacb`;
the separate training snapshot remains `b9d0ac0`. A 727,698-byte local package
contains 253 committed source/contract/smoke-report files, plus its manifest and
an empty root-discovery sentinel. No dataset, checkpoint or secret is packaged.
`package.json` records its archive and manifest digests. Isolated package tests
passed **55/55** (`isolated_package_tests.xml`); they do not rerun CUDA/e1. Tests and
`server_preflight_1243.json` preserve readiness evidence without GPU training.

Next: obtain explicit GDM-stop approval, confirm the headless preflight again,
transfer and checksum the separate orchestration snapshot, launch Taxi then RAF,
confirm the first epoch and create one hourly monitor. Training approval is
already granted; do not weaken the preflight or repeat completed CUDA/e1 runs.
The user's follow-up also approves the previously requested Notion publication
of internal paths, revision identifiers and experiment status to the existing
experiment page. Publication and readback succeeded at 12:45 KST; the prior
weighted-candidate cancellation remains intact. Evidence is in
`notion_publication_resumed.json`; the earlier permission denial is preserved
in `notion_publication.json` as historical evidence.

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

## Earlier SSH Blocker

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
