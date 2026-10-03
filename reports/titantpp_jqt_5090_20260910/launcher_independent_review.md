# 5090 J/Q/T launcher independent review

Reviewed snapshot file: `/Users/igwanhyeong/PycharmProjects/paper_research/search_artifacts/jqt_cuda_5090_20260910/source/paper/scripts/launch_time_quantity_suite.py`.

Reviewed SHA256: `47d7525a2210329178995d43331dd731417869450fb8917a9447b2c4baa61b05`. The writer was still editing/tests were ongoing; line numbers refer to this reviewed version. No source edits or remote commands were performed.

## Findings

1. **P2 — Completion audit accepts a mutually equal but impossible training budget (lines 193–205).** `verify_job_completion` trusts `equal_training_exposure_verified` and only compares the three summaries' step/hash values with each other. It never requires the dataset's actual `ceil(train_count / 128) * 120` step count, verifies per-epoch exposure, or connects the paired artifact to those summaries. A synthetic fixture declaring 120 completed epochs but `global_step=1` in all three arms, no history at all, and a three-field paired JSON was accepted as complete. Reproduction fixtures: `/tmp/jqt-launcher-review-7vbzxqhc`. Recommended fix: use the pinned dataset contract to check expected train/validation counts and steps, rerun `audit_arm_comparison` over current summaries, compare its result to the saved paired artifact, and bind/check the arm contracts or checkpoint receipts as appropriate. This is a final-audit gap; the correctly pinned current runner itself does not intentionally generate such truncated summaries.

2. **P2 — The monotonic ceiling is reset for each dataset child (lines 162–174, 228–242).** Absolute wall-clock deadline is reused, but `monotonic_deadline` is recomputed inside each `run_child`. A backward clock adjustment between children therefore grants extra remaining time to the next child. Example: a fixed absolute wall deadline initially has 259,200 seconds remaining; after 100 seconds of work and a 3,600-second clock rollback, the next child's freshly computed monotonic deadline is 3,600 seconds later than the original suite monotonic ceiling. Recommended fix: establish one suite-wide monotonic deadline once and pass the same value to all children and preflight timeout calculations. The qualification-start wall timestamp remains the intended cumulative budget origin; if robustness across clock changes during qualification is required, retain a monotonic qualification origin/boot identity in its receipt too.

3. **P2 hardening — The GPU lock is held only by the launcher (lines 129–147, 170, 214).** The lock descriptor is not yielded or passed via `pass_fds`, and Python `Popen` closes descriptors by default. A SIGKILL/crash of the launcher releases the cooperating lock while its `start_new_session` training child can remain alive. `assert_gpu_idle` normally detects an already active CUDA child, but there is a pre-CUDA-initialization window where another suite could acquire the lock and launch. Recommended fix: let the owned training child inherit the lock descriptor. This does not by itself enforce the deadline after an uncatchable launcher death; independent child/watchdog deadline handling would be a separate resilience mechanism. Normal caught SIGINT/SIGTERM/timeout paths do kill only the newly created child's process group.

## Confirmed protections

- Qualification status, `qualifies_cuda`, explicit `cuda:0`, required check flags, source revision/file map/source SHA, runtime SHA, and pinned receipt-file SHA are checked before launch.
- Dataset order/count, seed42,120epochs,J/Q/T objectives,batch128,static-B cap300,train-only+validation population set, and the declared suite step ceiling6,816,240 are checked.
- Source path traversal outside the pinned root is rejected. Each dataset launch rechecks suite/source/qualification/contracts, and the child CLI independently verifies its full source/runtime/data contract.
- Existing suite output is rejected; child commands omit `--resume` and contain no retry loop. A nonzero child exit aborts the remaining suite.
- The cooperating lock is acquired nonblockingly; GPU occupancy aborts rather than preempts another process.
- Timeout/interruption cleanup targets only the process group created for the current child (`start_new_session=True`); it does not signal an unrelated GPU PID.
- Per-child waits check both wall-clock and monotonic remaining time once per second. Atomic engine epoch artifacts survive mid-epoch termination; unfinished progress is not promoted to a completed suite.

## Validation performed

- Read launcher, corresponding CLI paired-comparison audit, and qualification receipt-writing/check fields.
- Executed only the synthetic completion-audit counterexample described above. No CUDA, model training, existing research data, checkpoints, or held-out results were opened.
- This review is not actual CUDA qualification and does not approve the scientific hypothesis.
