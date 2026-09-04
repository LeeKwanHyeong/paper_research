# 5090 Separate-Key Retrieval Preflight Preparation

Status: **CUDA and both full e1 audits passed** after explicit transfer approval.
Executed 2026-09-04 21:19:31 to 21:22:27 KST. e300 was separately authorized and
launched afterward; see `../hard_lmm_key_value_screening_5090_20260904/README.md`.

- Model revision: `c5fc6bc6b271f52324ea3090a6edf15c265b9251`.
- Prepared execution revision: `db7d70684b81f83380acfe08307854546e9ea75f`.
- Local tests: 45 passed in 8.30s, Python 3.12.10 / torch 2.7.1 CPU.
- 5090 read-only preflight: RTX 5090 idle, 2 MiB used, 0% utilization;
  GDM inactive and no recent Xid/OOM match. Existing services were untouched.
- Existing 5090 runtime: Python 3.12.13, torch 2.11.0+cu130, polars 1.39.3,
  pytest 9.0.3. CUDA available. No installation or environment changes.
- Taxi and Instacart dataset/split SHA256 matched the pinned registry.
- Local package: `/private/tmp/hard_lmm_key_value_smoke_db7d706_5090.tar.gz`.
- Package: 1,692,726 bytes; 479 manifest files including six audit-only baseline
  files. No raw dataset or credentials are bundled.
- Package SHA256: `3551d01c17aa0040ea231ae87763102b61498cf67d0a4ef571cd3f25d7ba0690`.

The first transfer was blocked by the permission reviewer. The user then
explicitly approved the source/reference payload and conditional e300. Transfer
proceeded through the same reviewed rsync path; no bypass was used.

## Confirmed Results

- CUDA model contracts: 29 passed, zero skipped/failed; see CUDA XML/log.
- Taxi: full 38,393 train / 8,268 validation targets, 10.65s wall time.
- Instacart: full 1,991,192 train / 503,733 validation targets, 160.14s wall time.
- Source manifest, data/split hashes, contract and checkpoint digests matched.
- Remote and local audits passed: finite metrics/optimizer, original quantile
  and history strata/counts, exact best/last checkpoint CPU replay.
- Ten generated CSVs contained only finite numeric values. No traceback, OOM or
  held-out artifact was found. This runner does not generate e1 plots.
- Keys and values diverged after training; this is operation evidence only.
  No performance gate or superiority conclusion was inferred from e1.
- Full raw evidence is under local `search_artifacts/hard_lmm_key_value_smoke_5090_20260904`.

## Remaining Order

1. Monitor the separately authorized two-run e300 screening on 5090.
2. Audit final artifacts and compare with the original Hard-LMM seed42 using the
   unchanged body/tail/time criteria.
3. Decide any further datasets/seeds only after results and separate approval.

No held-out evaluation, baseline retraining, model changes or service changes
were performed. The e300 hourly scheduler is recorded in the linked launch note.

Notion: https://app.notion.com/p/3d1bbe40561381c2a67afd4e5da20c67
