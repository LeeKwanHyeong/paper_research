# Weighted Static Retrieval: Implementation Handoff

## Completed

- Repository/branch: local `paper_research/master`.
- Implementation and prospective contract: `ed80d04e033202585cd4e6609510afdb0e477245`.
- Only backbone change: the original hard top-4 static prototype average becomes
  a cosine-softmax weighted sum, temperature 1.0. No added parameters or head.
- Legacy initialization/state keys and the original mean formula/gradients pass
  regression tests. Weighted retrieval changes the residual for different scores
  within the same selected set; causal masks and checkpoint replay are tested.
- The entire candidate model will train fresh, not a frozen readout. Existing
  original Hard-LMM and external benchmark results will be reused.
- Local verification: `/usr/local/bin/python3 -s`, PyTorch 2.7.1; 93 tests passed,
  one CUDA-only legacy test skipped. No dependency installation was performed.

The first pytest group (weighted memory/runner, legacy memory backbones,
B0 diagnostics, persistent/dual memory) passed 69 tests with one skip. The second
group (B012 screening, count-aware TPP, dataset, model-role contracts) passed 24.
`git diff --check` passed before the implementation commit.

## Server and Approval Boundary

Read-only 5080 inspection found RTX 5080, 34 MiB used, 15,801 MiB free, utilization
0%, no CUDA compute process, GDM inactive and no Xid/OOM entry in the last hour
of accessible boot kernel logs. Original four dataset launch contracts and
seed-42 summaries were inspected without evaluating any model or held-out data.

The 5080 source transfer was rejected by the execution permission reviewer,
which required explicit approval of private source egress to that destination.
No workaround or indirect transfer was attempted. CUDA tests, e1 smoke and e300
have NOT started. No new monitoring automation was created for an absent run.

Pending transfer is the committed source-only package
`/private/tmp/hard_lmm_weighted_static_source_20260903.tar` (275 files, 3,112,960
bytes), SHA-256 `444ca993e66490c31aebd41554cbfc848001c298c525b103b34931b698084daa`.
It contains source, tests and contracts, not datasets, checkpoints or credentials.
Destination archive:
`5080:/home/leekwanhyeong/workspace/hard_lmm_weighted_static_source_ed80d04.tar`.
Planned new snapshot:
`/home/leekwanhyeong/workspace/paper_research_weighted_static_ed80d04`.
Do not overwrite the active project mirror or change GDM/services/5090.

## Next Work

1. After explicit transfer approval, transfer without `--delete`, verify archive
   and source-manifest digests, extract into the new snapshot and use the existing
   5080 `ai_env/bin/python -s` runtime.
2. Run candidate CUDA contracts and full-data e1 on all four datasets. Validate
   manifest, logs, launch contracts, summary/history, checkpoint digest,
   quantity/history strata, train/validation target counts and held-out absence.
3. Only after all four pass, launch four candidate-only seed-42 e300 runs in tmux
   using `run_hard_lmm_weighted_static.py --phase screening`. Record actual e1
   speed for ETA and create hourly monitoring. Never auto-resume a failed run.
4. Compare against original frozen baseline identities using the unchanged body,
   RMSE, tail and time gates. Record runtime differences and single-seed limits.

The prospective JSON/Markdown contract is in `paper/contracts/hard_lmm_weighted_static_v1.*`.
Notion: [2026-09-03 Hard-LMM Similarity-Weighted Static Retrieval](https://app.notion.com/p/3d0bbe40561381d98efecd94ec3976a8).
The Notion result section remains empty because there are no performance results.
