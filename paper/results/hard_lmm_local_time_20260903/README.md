# Hard-LMM Quantity Memory / Local Time

## Implementation and Local Verification

2026-09-03, local `paper_research/master`. The subsequent user instruction
authorizes implementing the frozen route, local verification, then 5080 CUDA
contracts and full Taxi/RAF e1 smoke. It does not authorize e300 screening,
seed/dataset expansion, service changes, or push. The original contract's
authorization/status fields describe its earlier freeze phase and are preserved.

- Candidate: `titantpp_hard_memory_local_time`, role `t0_hard_memory_local_time`.
- Only route change: time receives masked local encoder output `h`, including
  persistent-token attention; quantity still receives original uniform `h+r`.
- No new parameters, loss, time head, optimizer, selector, or memory updates.
- Separate route metadata prevents silently relabeling a same-shaped checkpoint.
- Full train/validation data filtering occurs before materialization; test rows
  are excluded. The smoke wrapper rejects existing artifacts, runs each dataset
  in an isolated process, and records failures atomically without retry/resume.

Local runtime: Python 3.12.10, PyTorch 2.7.1; existing installation unchanged.
`local_tests.xml`: **128 passed, 1 skipped** across ten focused contract/regression
modules. The first regression pass found an exact supported-backbone list that
needed the new opt-in identifier; it was updated without changing legacy behavior.
`reference_route_parity.json`: pinned Taxi/RAF original checkpoint file and state
digests verified; synthetic-history quantity states/predictions match exactly.
No dataset rows, including held-out rows, were used for those checkpoint tests.

The tests cover parameter/RNG identity; single encoder execution; quantity output
and quantity-only gradient equality in train/eval; live quantity-to-prototype
gradients; absent time-to-prototype gradients; shared persistent gradients;
causal target/future/padding masking; series isolation; finite extreme-input
AdamW steps with global clipping; strict checkpoint inference replay; wrong-route
rejection; and fail-closed smoke orchestration/artifact audits. Optimizer state
serialization is checked, but exact stochastic training resumption is not claimed.

At identical weights and RNG, quantity is unchanged. After joint optimization,
time gradients alter the shared encoder and global gradient clipping, so unchanged
quantity predictions after training are **not** promised. e1 is a feasibility
check, not evidence that the original body/tail performance gates are passed.

## Next Boundary

Implementation, local validation and 5080 CUDA/full-e1 checks are complete.
The dedicated smoke entry point has no screening option and cannot launch e300.
Main Taxi/RAF seed-42 screening remains a separate decision. If approved, use
fresh artifacts/initialization with the frozen e300/min40/patience40 contract,
not either e1 checkpoint; keep all original performance thresholds. Do not resume
the cancelled weighted candidate or retrain the benchmark models.

## Server Smoke Audit Correction

The frozen `b9d0ac0` snapshot passed all 250 source checksums, isolated-package
tests (28 passed), and 5080 CUDA contracts (14 passed, 2.06 seconds). Taxi full e1
finished successfully. The first post-run validator incorrectly indexed
`history.json` as a list; the actual runner writes `{"history": [...]}`. Its
`KeyError: 0` correctly stopped the launcher before RAF; it was not a model, GPU,
or training failure. The historical failed status is preserved unchanged.

The local-only validator was corrected to require the real envelope; its fixture
now uses that envelope and rejects a bare list. Twenty-nine focused model/validator
tests pass. The completed Taxi artifact passes the corrected audit, including full
38,393/8,268 targets, strata, finite metrics, optimizer, routing and checkpoint
digests. Taxi and CUDA were not rerun. RAF alone ran in a fresh directory using
the exact original `b9d0ac0` training snapshot and generated e1 command, followed
by corrected local validation. No source patch was applied to the server.

## Final CUDA and E1 Evidence

`smoke_verification.json` is the authoritative combined local audit; it does not
rewrite the historical launcher failure. Final local suite: **129 passed, 1
skipped** (`final_local_tests.xml`). CUDA: **14 passed**, no skips. Isolated package:
**28 passed** before transfer. Training source remains `b9d0ac0`; corrected local
validator is `5d43bf6`. Remote Python 3.12.13/PyTorch 2.11.0+cu130 was unchanged.

| Dataset | Full Train / Validation Targets | e1 Train+Validation Seconds | Audit |
| --- | ---: | ---: | --- |
| Taxi | 38,393 / 8,268 | 12.235 | Passed |
| RAF | 25,779 / 6,690 | 2.662 | Passed |

RAF completed at **2026-09-03 19:47:15 KST**, followed by successful local audit.
Timing is the runner's e1 elapsed time, not an e300 forecast. Every summary,
history and CSV numeric metric is finite. Quantity/history CSV values reconcile
with summaries; data/source hashes, original counts/strata, AdamW defaults,
head/route metadata and last/best checkpoint digests match. Test data were not
materialized or evaluated. The e1 runner does not generate convergence plots;
none are claimed as verified. Postflight: no CUDA processes, GDM inactive,
15,798 MiB free, no recent accessible kernel Xid/OOM.

Raw checkpoints remain in the local `search_artifacts` directories identified in
`smoke_verification.json`. Curated manifests, logs, summaries, histories, CSVs and
CUDA evidence are committed under `source_5080_taxi` and `source_5080_raf`; large
checkpoint binaries are not committed. Transfers used no `--delete`. The short
checks finished in this session; no new scheduler, installation, GDM/service
change, 5090 operation, push or e300 training occurred.

Peak allocated VRAM was not instrumented in these short e1 runs; pre/postflight
free VRAM must not be presented as peak allocation. Record it separately if
performance screening is authorized. Notion publication/readback is verified in
`notion_publication.json`, including preservation of the prior cancellation.
