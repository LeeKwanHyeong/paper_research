# TitanTPP-MAC Stable Seed62: 5080-Only First Stage

## Authorization And Baseline

The user's 2026-08-31 instruction is to proceed on the 5080 first. Preserve
the prior server assignment: seed 62 on four datasets. Do not connect to,
start jobs on, or change services on the 5090. Seeds 42 and 52 remain deferred.
This authorizes this new explicit launch, not automatic resurrection of a
historical failed launcher.

Training remains pinned to `c4dbf856c32e6502acc660ffac23c3e2f68e5375`, with
`--titans-memory-gradient-clip 1`. The completed Instacart full-epoch gates
for seeds 42, 52 and 62 are the current baseline. The 5080 seed62 proof is
verified and reused as qualification evidence only, never as an e300
checkpoint. Old unbounded MAC results cannot enter the new three-seed mean.
No model, loss, optimizer or training implementation is changed in this stage.

## Serial Execution

1. Verify all 24 frozen training hashes, all four data/split checksums, and the
   completed Instacart qualification proof in the new isolated snapshot.
2. Run a complete train AND validation epoch for RAF (context 84), Taxi
   (context 256), and Intermittent (context 256), each in a separate process.
   No batch limit or series subset is allowed. Check full coverage, finite
   history/metrics, source, inner policy, checkpoint digest, strict restore,
   exact predictions and observed-history memory replay after each run.
3. Only after all three gates pass, run fresh seed62 validation in this order:
   Instacart, RAF, Taxi, Intermittent. Maximum 300 epochs, minimum 40,
   patience 40, batch 128, learning rate 0.001, hidden size 64, direct log-MSE,
   no tail loss, and minimum validation joint objective selection.
4. Retain each complete run only after the same artifact gate checks it.
   Do not present four seed62 runs as a final three-seed comparison.

The legacy time-head arguments are explicit, preserving the preceding MAC
execution: time scale 3, slope maximum 10/3, intercept limit 30. Dynamo limits
remain 64 and 512. Dataset context is the only per-dataset model configuration.
The original performance thresholds and compute-cost amendment are not tuned
using these preflights. Held-out evaluation remains locked.

## Failure And Monitoring

Before each training or CUDA validation process, require the correct RTX 5080,
at least 12,000 MiB free VRAM, no CUDA compute process, inactive GDM and no
gnome-shell/Xwayland. A project lock prevents duplicate launchers. Reject an
existing nonempty output root rather than overwriting or silently resuming it.
On child failure or a handled termination signal, preserve logs/fixtures,
terminate only this launcher's child process group and atomically write failed
status. Do not retry, skip a batch, or modify services automatically.

`status.json` records phase, dataset, seed, child PID, current command/log and
validated counts. The hourly monitor follows only this 5080 artifact. First
report context qualification separately from e300 progress and estimate time
from observed per-dataset epochs, not from GPU branding alone.

After completion, sync without deletion, validate all four run reports, update
the existing Notion launch page and commit only related results on
`paper_research/master`. Further seeds, a cross-seed decision and held-out test
are outside this first-stage authorization.
