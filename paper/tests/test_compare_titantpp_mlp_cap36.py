"""Pure Cap MLP comparison checks, with no model/GPU/remote execution."""
from copy import deepcopy
import math
from pathlib import Path
import statistics

import pytest

from paper.scripts import compare_titantpp_mlp_cap36 as c


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
                    delta = (0, (-1, 2, 0)[si], .5, -.5)[ai]
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
    assert "MLP16−MLP4" in text and "CurrentOnly−MLP16" in text


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


def collected_fixture(example, tmp_path, monkeypatch):
    contract, rows = example
    m = c.manager
    monkeypatch.setattr(m, "BUNDLE", tmp_path / "bundle")
    monkeypatch.setattr(m, "REPORT", tmp_path / "report")
    monkeypatch.setattr(m, "verify_local", lambda: contract)
    monkeypatch.setattr(m, "verify_collected", lambda *args: {"status": "complete"})
    m.write(m.BUNDLE / "execution_contract.json", contract)
    m.write(m.REPORT / "PREPARED.json", {"status": "prepared"})
    for host in ("5080", "5090"):
        dest = m.BUNDLE / "retrieval" / host
        originals = dest / "originals"
        originals.mkdir(parents=True)
        assigned = [j for j in contract["jobs"] if j["host"] == host]
        for job in assigned:
            folder = originals / "run" / job["id"]
            selected = {"selected_checkpoint": "selected.pt", "selected_epoch": c.SEEDS.index(job["seed"]) + 1,
                        "checkpoint_sha256": "binary", "state_sha256": "state"}
            m.write(folder / "train_receipt.json", selected)
            for split in ("validation", "test"):
                row = next(r for r in rows if all(r[k] == job[k] for k in ("dataset", "arm", "seed")) and r["split"] == split)
                m.write(folder / (split + "_receipt.json"), {**row, **selected,
                    "parameters_unchanged": True, "prediction_rows": 10})
        inventory = {str(p.relative_to(originals)): c.sha(p) for p in originals.rglob("*") if p.is_file()}
        m.write(dest / "manifest.json", {"files": inventory})
        (dest / "original.tar.gz").write_bytes(b"opaque original archive")
        m.write(dest / "transport.json", {"actual_transport_exit": 0})
        m.write(dest / "receipt.json", {"status": "complete", "host": host, "new_fit_count": len(assigned),
            "validation_replay_count": len(assigned), "test_count": len(assigned), "held_out_test_evaluated": True,
            "archive_sha256": c.sha(dest / "original.tar.gz"), "manifest_sha256": c.sha(dest / "manifest.json"),
            "file_count": len(inventory)})
    return contract, rows


def test_both_collection_archives_and_all_originals_are_rehashed(example, tmp_path, monkeypatch):
    contract, rows = collected_fixture(example, tmp_path, monkeypatch)
    loaded, files = c.load_verified(contract)
    assert len(loaded) == 72 and files
    archive = c.manager.BUNDLE / "retrieval/5090/original.tar.gz"
    archive.write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive"):
        c.load_verified(contract)


def test_comparison_writes_four_separate_reports_and_immutable_final_cache(example, tmp_path, monkeypatch):
    collected_fixture(example, tmp_path, monkeypatch)
    final = c.compare()
    assert final["status"] == "complete" and final["new_fit_count"] == final["candidate_test_count"] == final["validation_replay_count"] == 36
    assert final["actual_evaluation_worker_exit0_count"] == 72 and final["baseline_reference_count"] == 0
    assert len(final["report_files"]) == 4
    assert c.compare() == final
    validation = c.manager.REPORT / "THREE_SEED_VALIDATION.json"
    value = c.read(validation)
    assert value["split"] == "validation"
    validation.write_text("changed")
    with pytest.raises(ValueError, match="evidence changed"):
        c.compare()
