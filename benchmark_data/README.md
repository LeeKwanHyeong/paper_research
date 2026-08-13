# Benchmark Data Registry

This directory is the reproducible data boundary for the count-aware TPP paper.
Large raw and derived data files live under `data/` and are intentionally ignored
by Git. Contracts, manifests, audit reports, and preprocessing code are tracked.

## Qualified configuration

| Role | Dataset | Quantity provenance | Status |
|---|---|---|---|
| Main | Intermittent v2 | Native order quantity | Frozen and experiment-ready |
| Main | Online Retail II | Native transaction quantity | Frozen and experiment-ready |
| Main candidate | RAF Spare Parts | Native monthly demand | Frozen and experiment-ready; raw-data redistribution not cleared |
| Auxiliary | NYC Taxi Hourly | Derived pickup count per grid-hour | Existing frozen split |
| Optional auxiliary | Instacart | Derived basket size | Existing frozen split |

Datasets with no native or defensible derived count target, including the EasyTPP
Amazon, Retweet, StackOverflow, Taobao, and Volcano splits, are not part of this
benchmark. GlucoBench is also excluded because glucose is a continuous measurement,
not a demand count.

## Layout

```text
benchmark_data/
├── contracts/            # preprocessing and split contracts
├── data/                 # ignored large raw/derived files
│   ├── main/
│   ├── candidates/
│   └── auxiliary/
├── manifests/            # hashes, profiles, and lineage
├── reports/              # dataset qualification reports
├── scripts/              # deterministic preparation and audit code
└── tests/                # contract tests
```

## Rebuild

```bash
python3 benchmark_data/scripts/audit_and_prepare_raf.py
python3 benchmark_data/scripts/verify_raf_model_input.py
python3 benchmark_data/scripts/prepare_online_retail_ii.py
python3 benchmark_data/scripts/audit_auxiliary_frozen.py
python3 benchmark_data/scripts/verify_benchmark_model_inputs.py
python3 benchmark_data/scripts/build_data_inventory.py
python3 benchmark_data/tests/test_benchmark_contracts.py
```

RAF preparation emits a common event table, fixed chronological train/validation/test
files, and a split manifest under `data/candidates/raf_spare_parts/`. The model-facing
time unit is one month, and `seq` is the month index from January 1996. The original
workbook remains excluded from redistribution.

Online Retail II is aggregated by SKU and event hour, then converted to the common
count-aware schema. Its eligibility rule is fitted on the training interval only.
The frozen table contains 800,330 events from 3,125 eligible SKUs, with 567,063
train, 94,947 validation, and 138,320 held-out test events. Held-out rows never
determine SKU inclusion, filtering thresholds, aggregation rules, or model choices.

The Taxi and Instacart auxiliary artifacts retain their existing frozen splits.
`audit_auxiliary_frozen.py` verifies hashes, split counts, positive quantities, and
unique entity/sequence keys. `verify_benchmark_model_inputs.py` performs a sampled
validation-loader smoke test for Online Retail II, Taxi, and Instacart without
evaluating held-out test predictions. M5 Walmart is outside the active benchmark
scope and is not tracked as pending work.
