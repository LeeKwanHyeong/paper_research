# Intermittent frozen readout seed replication

Freeze this contract and its runner/tests on `paper_research/master` before new
fits. Seed 42 is discovery evidence from the existing readout factorial and is
reused by checksum, not rerun. Seeds 52 and 62 refer to the original Hard-LMM
backbone checkpoints, NOT new adapter initialization seeds or TitanTPP-MAC.

## Fixed experiment

Replicate constant/log-MSE and linear/raw-MAE, retaining constant/raw-MAE and
linear/log-MSE as matched controls. No MLP, new hyperparameter, loss mixture, or
dataset expansion. Two seeds times four cells means eight new fits. Each reports
both predeclared joint and body checkpoint selections; the selector is not chosen
after seeing the new results. The body selector and MAE objective remain explicitly
diagnostic departures from official T0. The legacy model stays frozen.

All readout architecture, normalization, optimizer, loss definitions, 40-epoch
budget, batch 1024, lr .001, clip norm 1, initialization/shuffle seed 42, and strict
earliest-checkpoint tie rule are inherited exactly from
`hard_lmm_readout_factorial_v1.json`. Thus the changed factor is the backbone
checkpoint seed. Each checkpoint gets its own train-only feature standardization.
We do not copy trained seed-42 readout weights to another backbone representation.

## Cache and safety

Copy only the two existing best-joint checkpoints from 5080 to local input files;
verify the historical file/state hashes and summaries. No server compute, GDM,
service, runtime installation, original checkpoint, or source change.

Extract the identical normalized hidden/residual/statistics features on local CPU
torch 2.7.1, using the original extractor and batch 128. A discarded first-256-train
inference preflight verifies official-output equivalence and estimates extraction
cost. Use the exact seed-42 uniform 65,536-train indices and full 86,285 validation
targets. Check target IDs, series IDs, context ends, history lengths and quantities
against the old cache. Features and logits differ by checkpoint, naturally.

All eight one-epoch train-only readout preflights must pass before new validation
caches are extracted. Discard preflight weights and start every main fit at exact
identity. The parquet scan excludes held-out rows before dataframe materialization;
validation labels never enter head inputs, normalization, or optimizer updates.
Baseline predictions must replay each original summary within absolute 1e-5.
Fail closed on checksum, nonfinite, data alignment, or baseline mismatch. Do not
silently relax tolerances, retry a failed fit, or change a setting after results.

## Interpretation and decision

The unchanged gate is body MAE improvement >=5%, overall RMSE and above-p99 MAE
regression <=2%, Time NLL regression <=.01, finite metrics, and selected epoch >0.
Replication needs the SAME cell/selector to pass seeds 52 and 62 individually.
Three-seed support additionally requires its existing seed-42 result to pass.
Do not average away a failing seed or select different cells for different seeds.

Report each seed, paired changes, and unweighted means/sample standard deviations
(ddof=1) for original/candidate overall/body/tail/history metrics. Report seed 42
as discovery and seeds 52/62 as replication on the SAME previously inspected
validation set, not independent-data confirmation. Failure does not prove no
better encoder/readout exists. Success does not authorize e300, model promotion,
loss/selection contract amendment, or held-out test evaluation.

## Commands

```bash
/usr/local/bin/python3 -s paper/scripts/run_hard_lmm_readout_seed_replication.py --output search_artifacts/hard_lmm_readout_seed_replication_20260903
/usr/local/bin/python3 -s paper/scripts/validate_hard_lmm_readout_seed_replication.py --artifact search_artifacts/hard_lmm_readout_seed_replication_20260903 --output paper/results/hard_lmm_readout_seed_replication_20260903
```

If feature extraction cannot finish promptly, leave the immutable job running with
a progress log and a user-authorized monitor. Do not start another training job.
