"""Pure Cap MLP comparison checks, with no model/GPU/remote execution."""
from copy import deepcopy
import math
from pathlib import Path
import statistics

import pytest

from paper.scripts import compare_titantpp_external_cap27 as c


@pytest.fixture
def example():
    definitions, rows = [], []
    names = ("yellow_trip_hourly", "raf_spare_parts", "intermittent_frozen_5000")
    for di, dataset in enumerate(names):
        identities = {s: {"target_count": 10, "target_identity_sha256": dataset + s,
                          "target_quantity_sha256": "capped-" + dataset + s} for s in ("validation", "test")}
        definitions.append({"dataset_id": dataset, "cap_value": 20.,
            "inherited_data_identity": {"populations": {"validation": identities["validation"]}},
            "parent_data_identity": {"populations": {"test": identities["test"]}}})
        for split in ("validation", "test"):
            for ai, arm in enumerate(c.ARMS):
                for si, seed in enumerate(c.SEEDS):
                    baseline = 10. + di * 100 + si + (1 if split == "test" else 0)
                    delta = (0, .5, 1., -.5, (-1, 2, 0)[si], 1.5, -2.)[ai]
                    rmse, mae, temporal = baseline + delta, (baseline + delta) / 2, (baseline + delta) / 10
                    def group(n):
                        return {"count": n, "qty_rmse": rmse if n else None, "qty_mae": mae if n else None,
                                "time_nll": temporal if n else None, "qty_sse": n * rmse ** 2,
                                "qty_absolute_error_sum": n * mae, "time_nll_sum": n * temporal}
                    metrics = {"overall": group(10), "tail": group(0), "cap_hit": group(2), "cap_value": 20.,
                        "tail_rule": "q > cap_value", "cap_hit_rule": "q == cap_value",
                        "time_metric": "recorded_positive_integer_time_nll"}
                    rows.append({"dataset": dataset, "arm": arm, "seed": seed, "host": "5080" if seed == 42 else "5090",
                        "split": split, "selected_epoch": si + 1, "checkpoint_sha256": "binary", "state_sha256": "state",
                        "target_identity": identities[split], "metrics": metrics})
    jobs = [{"id": f"{r['dataset']}__{r['arm']}__seed{r['seed']}",
             **{k: r[k] for k in ("dataset", "arm", "seed", "host")}} for r in rows if r["split"] == "validation"]
    return {"datasets": definitions, "jobs": jobs}, rows


def test_three_seed_mean_sample_sd_and_signed_differences(example):
    contract, rows = example
    result = c.aggregate(rows, contract, "validation")
    data = result["datasets"][0]
    rmse = data["arms"][0]["groups"]["overall"]["metrics"]["qty_rmse"]
    assert rmse["mean"] == 11 and rmse["sample_sd"] == 1
    comparison = data["comparisons"][0]["groups"]["overall"]
    assert [s["metrics"]["qty_rmse"]["absolute"] for s in comparison["seed_differences"]] == [-1, 2, 0]
    summary = comparison["summary"]["qty_rmse"]
    assert summary["absolute"]["mean"] == pytest.approx(1 / 3)
    assert summary["absolute"]["sample_sd"] == pytest.approx(statistics.stdev([-1, 2, 0]))
    assert (summary["improved_seed_count"], summary["worsened_seed_count"], summary["tied_seed_count"]) == (1, 1, 1)
    assert comparison["seed_differences"][0]["metrics"]["qty_rmse"]["relative_percent"] == -10
    assert comparison["seed_differences"][1]["metrics"]["qty_rmse"]["relative_percent"] == pytest.approx(200 / 11)


def test_empty_tail_stays_unavailable_and_dataset_units_are_separate(example):
    contract, rows = example
    result = c.aggregate(rows, contract, "test")
    assert len(result["datasets"]) == 3 and result["cross_dataset_raw_metric_average_computed"] is False
    assert [d["arms"][0]["groups"]["overall"]["metrics"]["qty_rmse"]["mean"] for d in result["datasets"]] == [12, 112, 212]
    for data in result["datasets"]:
        for arm in data["arms"]:
            assert arm["groups"]["tail"]["count"] == 0
            assert arm["groups"]["tail"]["metrics"]["qty_rmse"]["mean"] is None
        for comparison in data["comparisons"]:
            summary = comparison["groups"]["tail"]["summary"]["qty_mae"]
            assert summary["worsened_seed_count"] == 0 and summary["unavailable_seed_count"] == 3
    text = c.markdown(result)
    assert "N/A" in text and "None" not in text and "nan" not in text.lower()
    assert "RMTPP−MLP4" in text and "S2P2−CurrentOnly" in text


@pytest.mark.parametrize("kind", ["missing", "duplicate", "identity", "group_count"])
def test_incompatible_population_or_seed_matrix_is_rejected(example, kind):
    contract, original = example
    rows = deepcopy(original)
    if kind == "missing":
        rows.pop(0)
    elif kind == "duplicate":
        rows[0] = deepcopy(rows[1])
    elif kind == "identity":
        rows[0]["target_identity"]["target_quantity_sha256"] = "other quantities"
        # The source fixture shares identities; break just this row explicitly.
        rows[0]["target_identity"] = {**original[0]["target_identity"], "target_quantity_sha256": "different"}
    else:
        rows[0]["metrics"]["cap_hit"]["count"] = 1
    with pytest.raises(ValueError):
        c.aggregate(rows, contract, "validation")


def test_zero_reference_relative_difference_and_partial_missing():
    assert c.differences(2., 0.) == {"absolute": 2., "relative_percent": None, "status": "relative_N/A_zero_reference"}
    assert c.differences(-2., -1.)["relative_percent"] == -100
    with pytest.raises(ValueError):
        c.differences(None, 1.)
    with pytest.raises(ValueError):
        c.summarize([1., None, 2.])


def test_metric_support_and_additive_sums_are_checked(example):
    metrics = example[1][0]["metrics"]
    c.verify_metric_groups(metrics, 10, 20.)
    for update in ({"tail_rule": "other"}, {"cap_value": 21.}, {"time_metric": "other"}):
        with pytest.raises(ValueError):
            c.verify_metric_groups({**metrics, **update}, 10, 20.)
    bad = deepcopy(metrics)
    bad["overall"]["qty_sse"] += 10
    with pytest.raises(ValueError, match="sum"):
        c.verify_metric_groups(bad, 10, 20.)
    bad = deepcopy(metrics)
    bad["tail"]["qty_mae"] = 0.
    with pytest.raises(ValueError, match="empty"):
        c.verify_metric_groups(bad, 10, 20.)




def test_all_seven_models_and_external_mlp_contrasts(example):
    contract, rows = example
    result = c.aggregate(rows, contract, "validation")
    assert result["model_fit_count"] == 63
    assert result["new_external_fit_count"] == 27
    assert result["stored_mlp_reference_count"] == 36
    assert all(len(d["arms"]) == 7 and len(d["comparisons"]) == 15 for d in result["datasets"])
    assert len(c.CONTRASTS) == len(set(c.CONTRASTS)) == 15
    assert all((a, b) in c.CONTRASTS for a in c.EXTERNAL_ARMS for b in c.MLP_ARMS)
