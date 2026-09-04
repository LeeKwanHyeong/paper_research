# Separate-Key Retrieval: 5090 CUDA and Full e1 Amendment

Date: 2026-09-04. Repository/branch: local `paper_research/master`.

## Scope

The user selected 5090 for the next preflight. The original model contract is
unchanged. This execution amendment permits only isolated source transfer, CUDA
model contracts and full Taxi/Instacart seed42 e1 train/validation. It does not
authorize e300, baseline retraining, held-out evaluation, runtime installation,
service changes or automatic retries. No THP model work is included.

Frozen model revision: `c5fc6bc6b271f52324ea3090a6edf15c265b9251`.
The companion JSON pins model/contract files and both full target counts.
The separate execution commit is recorded by the exported source manifest.

## Execution and Audit

`package_hard_lmm_key_value_smoke.py` exports committed source and six hash-pinned
reference files (launch, summary, checkpoint for each dataset). References are
audit-only: the candidate starts fresh. Existing remote datasets are read by
absolute path; no existing checkout or artifact is overwritten.

`run_hard_lmm_key_value_smoke.py` checks free VRAM, GDM, GPU processes and recent
Xid/OOM, then runs CUDA tests and each dataset in separate Python processes. It
has no screening or budget override option. It rejects existing output roots
and records failed status on handled failures. It does not restart failed runs.

Audit order: source manifest, launcher/run logs, launch/summary, held-out
absence, history, quantity/history strata, checkpoint and generated outputs.
The audit checks full train/validation counts, exact original quantile bins,
finite checkpoint/optimizer tensors and metrics, common time-head launch
arguments, direct log-MSE, one AdamW group and exact CPU best/last replay.
Key/value divergence is only operation evidence, not a performance result.

Before CUDA execution, next-optimizer-step CUDA replay tolerance is explicitly
set to rtol/atol 1e-6 for reduction-order differences. CPU replay remains exact.
All other existing model contract checks remain unchanged.

## Local Verification

Python 3.12.10, torch 2.7.1 CPU, polars 1.31.0. Focused model, contract and smoke
orchestration tests: **45 passed** in 8.30 seconds. Tests include a synthetic
full runner invocation, held-out sentinel exclusion, audit drift rejection,
failure status, package sentinel/reference digests and overwrite rejection.

## Next Boundary

After CUDA and both full e1 audits pass, report the runtime evidence and request
separate approval for any seed42 e300 screening. e1 metrics do not support a
model adoption or superiority claim.

Notion record: https://app.notion.com/p/3d1bbe40561381c2a67afd4e5da20c67
