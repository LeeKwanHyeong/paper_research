# 5090 Separate-Key Retrieval Preflight Preparation

Status: **blocked at source-transfer permission**, not a model/runtime failure.
CUDA tests and actual-data e1 training have **not started**.

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

The transfer tool rejected egress because selecting server 5090 was not treated
as sufficiently explicit authorization to transfer the private source and
reference checkpoint payload. No alternative transfer was attempted.

## Remaining Order

1. Obtain explicit approval for this package transfer to the user's 5090 server.
2. Transfer without deletion/overwrite; verify checksum in a new isolated source
   directory and recheck GPU availability.
3. Run CUDA contracts, then full Taxi and Instacart seed42 e1 in separate processes.
4. Audit source/log/summary/scope/history/strata/checkpoint and sync evidence.
5. Only after a passing runtime gate consider separately approved e300 screening.

No performance claims, held-out evaluation, baseline retraining or scheduler
were made. No training was launched, so there is nothing to monitor yet.

Notion: https://app.notion.com/p/3d1bbe40561381c2a67afd4e5da20c67
