# Completed A100 candidates: bounded technical inference attempt 2

## Current status

- Preparation complete: twelve selected conditions, full Validation before Test, no training.
- Original `../a100` attempt is preserved. Its first Validation CUDA call could not locate an already installed NVRTC builtins library; zero prediction rows and zero Test rows were produced.
- This is an explicitly authorized environment-only inference attempt, not a new scientific condition or a training retry. GPU execution waits for the parent to release the single 5080 GPU after Deep Renewal.

## Exact change and invariant evidence

`evaluate.py`, `run_all.py`, `verify.py`, `prepare.py`, and the synthetic tests are byte-identical copies from the first attempt. The registry, source closures, selected checkpoint/history/terminal/replay bindings, data SHA, loader, target population, prediction protocol, and all metrics stay unchanged. The original CPU checkpoint identity/strict-load audit (12/12) is reused by SHA.

The operational contract changes the dedicated remote workspace to `a100_workspace_attempt2`, records its parent contract/failure and existing runtime-library SHA, and requires this per-process environment:

```
LD_LIBRARY_PATH=/home/leekwanhyeong/miniconda3/envs/ai_env/lib/python3.12/site-packages/nvidia/cu13/lib
```

This is the existing library directory already used by width16 evaluation. No library is installed, no shared Runtime or scientific source is changed. `prepare_attempt2.py` creates the operational overlay exclusively; the copied historical `prepare.py` is not the attempt2 preparation entrypoint.

All twelve full Validation gates must pass before any of the twelve Test runs. Existing MLP seed42 predictions are reused only for target/truth identity checks; their original whole condition folders remain read-only. Four full data files reuse approved 5080 bytes by SHA. Instacart was not scheduled in either A100 campaign. These are seed42 exploratory results evaluated on RTX 5080, not native A100 performance/efficiency measurements or a three-seed result.

## Local verification

- Static source/checkpoint/baseline gate: passed, 12 conditions, no inference.
- Synthetic metric, identity, receipt, and selection tests: 22 passed.
- Selected binary CPU identity/strict-load check: original 12/12 receipt reused with identical selected/source bindings.

`execution_contract.json` binds the first attempt and its failure evidence. `code_seal.json` binds the new operational files. A subsequent launch receipt must record the required LD_LIBRARY_PATH and recheck existing runtime-library SHA before any GPU execution. Automatic retries remain prohibited; any failure is preserved as a separate outcome.
