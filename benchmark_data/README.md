# Benchmark Data Registry

This directory is the reproducible data boundary for the count-aware TPP paper.
Large raw and derived data files live under `data/` and are intentionally ignored
by Git. Contracts, manifests, audit reports, and preprocessing code are tracked.

## Qualified configuration

| Role | Dataset | Quantity provenance | Status |
|---|---|---|---|
| Main | Intermittent v2 | Native order quantity | Frozen and experiment-ready |
| Main | Online Retail II | Native transaction quantity | Prepared under contract v1 |
| Main candidate | RAF Spare Parts | Native monthly demand | Structurally qualified; usage permission must be checked |
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
python3 benchmark_data/scripts/prepare_online_retail_ii.py
python3 benchmark_data/scripts/build_data_inventory.py
python3 benchmark_data/tests/test_benchmark_contracts.py
```

The Online Retail II eligibility rule is fitted on the training interval only.
Held-out rows never determine SKU inclusion, filtering thresholds, or aggregation
rules.
