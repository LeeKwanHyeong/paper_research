# Experiment artifact manifests

`final_fair_artifact_manifest.*` records one row for every dataset, model, and seed in the frozen
T2 comparison contract. Regenerate it on the machine that stores the experiment artifacts:

```bash
python paper/scripts/build_artifact_manifest.py \
  --project-root . \
  --active-run-root search_artifacts/final_fair_matched_rmtpp_thp_e300_20260805
```

The JSON file retains check-level evidence, the CSV file is intended for analysis, and the Markdown
file provides a readable qualification report.

