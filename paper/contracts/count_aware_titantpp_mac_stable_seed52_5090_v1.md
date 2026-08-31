# TitanTPP-MAC Stable Seed52: 5090 Companion Stage

## Scope And Baseline

The user's 2026-08-31 request authorizes adding the 5090 while the existing
5080 seed62 stage continues. Retain the earlier seed assignment: four fresh
seed52 runs on 5090. No seed62 duplication and no seed42 job in this stage.
Seed42 remains necessary under the new policy before a final three-seed table;
historical unbounded-memory results cannot substitute for it.

Training source stays c4dbf856c32e6502acc660ffac23c3e2f68e5375 and explicit
inner gradient clipping remains 1. The completed 5090 Instacart seed52 full-e1
report is reused only as qualification evidence, never as a training checkpoint.
No model, loss, optimizer, training implementation or performance gate changes.

## Execution And Qualification

1. Verify the 24 frozen training hashes, four data/split digests and the
   completed Instacart seed52 full-epoch proof in a new isolated 5090 snapshot.
2. Run RAF, Taxi and Intermittent seed52 full train+validation e1, each followed
   by the same coverage/finite/checkpoint digest/prediction/memory replay gate.
   No series or batch subset may substitute for this qualification.
3. Only after all three gates pass, run fresh seed52 Instacart, RAF, Taxi and
   Intermittent validation: max300/min40/patience40, batch128, lr0.001, h64,
   direct log-MSE, no tail loss, and minimum validation joint objective.
4. Validate each result before moving on; preserve errors with no retry or
   checkpoint resume. Existing nonempty output directories are rejected.

Time head and context match the 5080 contract exactly: legacy_clamped_rmtpp,
time scale3, slope maximum10/3, intercept limit30, outer/inner clipping1/1.
Dynamo limits64/512, independent child process groups, GDM inactive,
no CUDA compute process and at least12000 MiB free are required before a stage.

The 5090 entrypoint intentionally freezes a separate copy of the qualified
5080 orchestration so the active 5080 snapshot and its authorization cannot
change. Split-parity tests compare every training argument apart from
server/seed provenance and confirm identical datasets/source/policies.

## Runtime And Service Boundary

Use the verified Python environment and absolute tmux path on 5090, not a
PATH-based assumption from 5080. Do not install packages or modify services.
At 09:38 KST the GPU was idle (2 MiB), GDM inactive, and port8011 was not
listening. Existing CPU services are outside this scope and remain untouched.
Kernel journal access is restricted; an empty query is not evidence of no Xid.

A dedicated hourly 5090 monitor records context gates (0-3) separately from
e300 runs (0-4). Leave the 5080 monitor intact. No automated service restore,
restart, retry, or change to the frozen training policy is authorized here.

## Closeout

Sync without deletion and validate source, logs, contracts, histories, finite
quantity/history metrics and checkpoint evidence. Report seed52 only, update
the existing Notion launch page, and commit related results on
paper_research/master. Do not push or pool old unbounded results with these
runs. Held-out test remains locked.
