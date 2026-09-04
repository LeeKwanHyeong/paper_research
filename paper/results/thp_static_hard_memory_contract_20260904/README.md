# THP + Static Hard Memory: Contract Freeze Evidence

## Completed: Local Contract, Not a Candidate Experiment

Repository/branch: local `paper_research/master`.
Reviewed source: `af14bcfba59fef6f5b9f1d7c4682fba804beddaa`.
Contract: `paper/contracts/count_aware_thp_static_hard_memory_v1.json` and its
Markdown companion. Dedicated baseline registry:
`paper/contracts/count_aware_thp_static_hard_memory_baselines_v1.json`.

The candidate is the unchanged project CountAwareTHP plus one original static
hard top-4 prototype residual, supplied to both original heads. It is NOT the
previous local-time candidate, a calibration probe or Titans-MAC. No candidate
implementation, forward pass, inference, training or performance result exists.
The expected extra 4,096 parameters are disclosed; this is not a capacity-matched
test of retrieval itself. No model, loss or training source was changed.

## Verification

- `audit_baselines.py`: four original Taxi/RAF seed42 THP/Hard-LMM references
  passed. This is local metadata, byte hashes and CPU checkpoint tensor loading
  only; no model construction, optimizer step or dataset row materialization.
- `baseline_verification.json`: source/file/state digests, finite metrics and
  checkpoint/optimizer tensors, fixed target counts, first strict joint minimum,
  early stopping, optimizer/head arguments, weighted strata totals and matching
  THP/Hard-LMM quantity/history stratum labels/counts were checked.
- Historical and reviewed CountAwareTHP/THPEncoderLayer class ASTs match. Reviewed
  source identities are pinned in the contract; current source files also remain
  unchanged from the reviewed revision.
- `document_tests.xml`: 31 document tests passed (21 new and 10 existing local-time
  contract regression checks). These validate contract consistency and frozen
  identities, NOT the 16 pending candidate model implementation requirements.
- Python 3.12.10 / Torch 2.7.1 were used without changing installed packages.
  Pytest emitted an unrelated installed `fs/pkg_resources` deprecation warning.

Reproduction commands, from the repository root:

```sh
env MPLCONFIGDIR=/private/tmp/weighted-mpl /usr/local/bin/python3 -s paper/results/thp_static_hard_memory_contract_20260904/audit_baselines.py
env MPLCONFIGDIR=/private/tmp/weighted-mpl /usr/local/bin/python3 -s -m pytest -q simple_lab_test/search/tests/test_thp_static_hard_memory_contract.py simple_lab_test/search/tests/test_hard_lmm_local_time_contract.py --junitxml=paper/results/thp_static_hard_memory_contract_20260904/document_tests.xml
```

The audit refuses to overwrite a mismatched frozen registry. Existing reference
files were hash-checked before and after loading and remain unchanged.
Historical split metadata contains test counts; no test evaluation is recorded,
but historical absence of test-row materialization is not established. New
candidate loading must exclude test before materialization. Historical runtime
reuse is not a bitwise or matched-runtime replay claim.

## Prospective Decision, No Outcome Yet

Direct THP control: preserve body and overall MAE, improve overall RMSE and >p99
MAE by at least 2% each, and allow at most 0.01 absolute Time NLL increase.
The 2% improvement thresholds are NEW prospective engineering screening rules,
not statistical significance requirements or retroactive criteria.

Original Hard-LMM compatibility criteria remain unchanged: body MAE improvement
at least 5%; overall RMSE and >p99 MAE regression at most 2%; Time NLL increase
at most 0.01. Both reference sets must pass on BOTH Taxi and RAF individually.
Full-precision limits are in the registry. Missing/empty/nonfinite evidence is
not evaluable. A finite performance rejection means no expansion, not a retry.
Existing weighted/local-time hold decisions are unchanged.

## Next Work: Serial and Separately Approved

1. Approval required: implement only the distinct candidate locally in
   `paper_research/master`, then run identity/gradient/causality/checkpoint tests.
2. Approval required after local tests: checksum source sync to 5080, CUDA tests
   and full Taxi/RAF e1 feasibility checks. Do not change services implicitly.
3. Approval required after feasibility: two fresh Taxi/RAF seed42 e300 runs,
   original epoch/selection contract and independent processes. No benchmark
   reruns, other datasets or seeds are authorized by this contract.
4. Audit results and apply all gates. Only a pass on both datasets permits a
   separately agreed follow-up, not adoption or a general superiority claim.

No server access, service changes, scheduler, Notion write or push occurred in
this contract-only task. No completion-time estimate is possible before actual
candidate runtime measurements. Baseline verification does not certify future
5080 availability or memory consumption.
