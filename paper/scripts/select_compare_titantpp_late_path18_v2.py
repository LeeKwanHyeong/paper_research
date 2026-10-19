"""Select one common log-time loss from the completed quantity-loss27 only.

The sealed three-arm aggregation/reporting implementation is reused unchanged,
with its manager import bound to the separate v2 campaign. No missing raw-time
results are synthesized and no inference, remote status or training is run here.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import statistics

_BASE_PATH = Path(__file__).with_name("select_compare_titantpp_late_path18.py")
_BASE_SHA = "a3553f9cfde5aa1c3822cb012c8a657d91abdcf93628ee7826f20727caebda76"
if hashlib.sha256(_BASE_PATH.read_bytes()).hexdigest() != _BASE_SHA:
    raise ValueError("Sealed late-path18 comparator changed")
_spec = importlib.util.spec_from_file_location("paper.scripts._late_path18_v2_comparison_private", _BASE_PATH)
_impl = importlib.util.module_from_spec(_spec)
_source = _BASE_PATH.read_text()
_old_import = "from paper.scripts import manage_titantpp_late_path18 as manager"
if _source.count(_old_import) != 1:
    raise ValueError("Sealed comparator manager import differs")
_source = _source.replace(_old_import, "from paper.scripts import manage_titantpp_late_path18_v2 as manager")
exec(compile(_source, str(_BASE_PATH), "exec"), _impl.__dict__)
_impl.TRANSFORMS = ("log1p",)
_ORIGINAL_VALIDATE = _impl._validate_row


def _validate_row_v2(row):
    _ORIGINAL_VALIDATE(row)
    _impl.require(row.get("time_input_transform", "log1p_nonnegative_float32") == "log1p_nonnegative_float32"
                  and row.get("quantity_input_transform", "log1p_nonnegative_float32") == "log1p_nonnegative_float32",
                  "Observed time and quantity input must remain log1p")


_impl._validate_row = _validate_row_v2


def select_common_condition(records):
    """Choose a loss using exact 27 completed log-time MLP16 endpoints.

    Per-dataset three-seed means are divided by the same dataset's fixed log-MSE
    mean, then the three ratios receive equal weight. RMSE is primary, MAE breaks
    exact ties, and the fixed log_mse/raw_mse/raw_mae order breaks remaining ties.
    """
    require = _impl.require
    require(isinstance(records, (list, tuple)), "Endpoints must be a sequence")
    indexed = {}
    for row in records:
        _impl._validate_row(row)
        require(row["time_input"] == "log1p", "Only observed log1p-time endpoints are admitted")
        require(row.get("quantity_input_transform", "log1p_nonnegative_float32") == "log1p_nonnegative_float32",
                "Quantity input must remain log1p")
        if any(key in row for key in ("model", "model_name")) or row.get("arm") in _impl.MODELS:
            require(_impl._model(row) == _impl.BASELINE, "Common-setting selection must use stored MLP16 endpoints only")
        key = tuple(row[key] for key in ("dataset", "seed", "loss_mode"))
        require(key not in indexed, "Expected exact completed log-time27 matrix")
        indexed[key] = row
    expected = {(dataset, seed, loss) for dataset in _impl.DATASETS for seed in _impl.SEEDS for loss in _impl.LOSSES}
    require(len(records) == len(indexed) == 27 and set(indexed) == expected,
            "Expected exact completed log-time27 matrix")
    for dataset in _impl.DATASETS:
        for seed in _impl.SEEDS:
            reference = indexed[dataset, seed, "log_mse"]
            for loss in _impl.LOSSES:
                _impl._validate_pair(reference, indexed[dataset, seed, loss])
        for cohort in _impl.COHORTS:
            require(len({indexed[dataset, seed, "log_mse"]["metrics"][cohort]["count"]
                         for seed in _impl.SEEDS}) == 1, "Seed cohort counts differ")
    fixed_means = {dataset: {metric: statistics.mean(
        indexed[dataset, seed, "log_mse"]["metrics"]["overall"][metric] for seed in _impl.SEEDS)
        for metric in _impl.METRICS[:2]} for dataset in _impl.DATASETS}
    require(all(value > 0. for means in fixed_means.values() for value in means.values()),
            "Fixed reference quantity means must be positive for ratio selection")
    scores = []
    for loss in _impl.LOSSES:
        datasets = []
        for dataset in _impl.DATASETS:
            group = [indexed[dataset, seed, loss] for seed in _impl.SEEDS]
            stats = {metric: _impl._summary([row["metrics"]["overall"][metric] for row in group])
                     for metric in _impl.METRICS[:2]}
            means = {metric: stats[metric][0] for metric in _impl.METRICS[:2]}
            ratios = {metric: means[metric] / fixed_means[dataset][metric] for metric in _impl.METRICS[:2]}
            require(all(_impl.math.isfinite(value) for value in ratios.values()), "Nonfinite selection ratio")
            datasets.append({"dataset": dataset, "means": means,
                "sample_sd_ddof1": {metric: stats[metric][1] for metric in _impl.METRICS[:2]},
                "fixed_reference_means": deepcopy(fixed_means[dataset]), "ratios": ratios,
                "seed_records": deepcopy(group)})
        scores.append({"condition": {"time_input": "log1p", "loss_mode": loss, "time_alpha": 0.},
            "macro_rmse_ratio": statistics.mean(row["ratios"]["qty_rmse"] for row in datasets),
            "macro_mae_ratio": statistics.mean(row["ratios"]["qty_mae"] for row in datasets),
            "datasets": datasets})
    selected = min(scores, key=lambda row: (row["macro_rmse_ratio"], row["macro_mae_ratio"],
                                           _impl.LOSSES.index(row["condition"]["loss_mode"])))
    return {"schema": "titantpp_late_path_common_condition_selection_v2", "scope": "validation_only",
        "selection_rule": "min macro three-seed mean RMSE/log-MSE reference ratio; then MAE ratio; then log_mse/raw_mse/raw_mae order; exact comparisons; log1p-time fixed",
        "fixed_reference_condition": deepcopy(_impl.REFERENCE_CONDITION),
        "selected_condition": deepcopy(selected["condition"]),
        "selected_scores": {key: selected[key] for key in ("macro_rmse_ratio", "macro_mae_ratio")},
        "condition_scores": scores, "records": deepcopy(records), "input_endpoint_count": 27,
        "dataset_count": 3, "seed_count": 3, "condition_count": 3,
        "held_out_test_evaluated": False, "raw_errors_pooled": False, "raw_time_endpoints_used": 0,
        "interpretation": "Validation-based common training setting, not a backbone contribution or independent evaluation"}


_impl.select_common_condition = select_common_condition
for _name, _value in tuple(_impl.__dict__.items()):
    if not _name.startswith("_"):
        globals()[_name] = _value


if __name__ == "__main__":
    main()
