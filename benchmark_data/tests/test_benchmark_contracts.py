from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def main() -> None:
    registry = load_json("manifests/dataset_registry.json")
    retail_contract = load_json("contracts/online_retail_ii_v1.json")
    retail_manifest = load_json("manifests/online_retail_ii_v1.json")
    retail_split = load_json(
        "data/main/online_retail_ii/online_retail_ii_split_manifest.json"
    )
    raf_contract = load_json("contracts/raf_spare_parts_v1.json")
    raf_manifest = load_json("manifests/raf_spare_parts_v1.json")
    auxiliary_manifest = load_json("manifests/auxiliary_frozen_v1.json")

    assert retail_contract["series_eligibility"]["future_events_used"] is False
    assert retail_contract["held_out_policy"]["test_used_for_filtering"] is False
    assert retail_manifest["split"]["eligibility_fit_scope"] == "train only"
    assert retail_manifest["qualification"]["tpp_convertible"] is True
    assert retail_manifest["event_profile"]["eligible_skus"] > 0
    assert sum(retail_manifest["split"]["counts"].values()) == retail_manifest["event_profile"]["events"]
    assert retail_split["split_counts"] == {
        "train": 567063,
        "validation": 94947,
        "test": 138320,
    }
    assert retail_split["next_event_target_counts"] == {
        "train": 563938,
        "validation": 94947,
        "test": 138320,
    }
    assert retail_split["eligibility"]["eligible_skus"] == 3125
    assert retail_split["held_out_policy"]["test_used_for_model_selection"] is False

    retail_frame = pd.read_parquet(
        ROOT / "data/main/online_retail_ii/online_retail_ii_with_split.parquet"
    )
    required = {
        "oper_part_no",
        "demand_dt",
        "seq",
        "demand_qty",
        "chronological_split",
        "delta_t",
    }
    assert required <= set(retail_frame.columns)
    assert len(retail_frame) == sum(retail_split["split_counts"].values())
    assert retail_frame["oper_part_no"].nunique() == 3125
    assert (retail_frame["demand_qty"] > 0).all()
    assert not retail_frame.duplicated(["oper_part_no", "seq"]).any()

    assert raf_contract["held_out_policy"]["test_used_for_filtering"] is False
    assert raf_manifest["audit"]["items"] == 5000
    assert raf_manifest["audit"]["months"] == 84
    assert raf_manifest["audit"]["missing_monthly_values"] == 0
    assert raf_manifest["audit"]["negative_monthly_values"] == 0
    assert raf_manifest["qualification"]["tpp_convertible"] is True
    assert raf_manifest["qualification"]["publication_status"] == "research_use_with_citation"
    assert raf_manifest["qualification"]["redistribution_status"] == "not_cleared"

    raf_split_path = ROOT / "data/candidates/raf_spare_parts/raf_spare_parts_split_manifest.json"
    raf_split = json.loads(raf_split_path.read_text())
    assert raf_split["split_counts"] == {
        "train": 30779,
        "validation": 6690,
        "test": 5226,
    }
    assert raf_split["next_event_target_counts"] == {
        "train": 25779,
        "validation": 6690,
        "test": 5226,
    }
    assert raf_split["held_out_policy"]["test_used_for_model_selection"] is False
    raf_frame = pd.read_parquet(
        ROOT / "data/candidates/raf_spare_parts/raf_spare_parts_with_split.parquet"
    )
    required = {
        "oper_part_no",
        "demand_dt",
        "seq",
        "demand_qty",
        "chronological_split",
        "delta_t",
    }
    assert required <= set(raf_frame.columns)
    assert len(raf_frame) == sum(raf_split["split_counts"].values())
    assert raf_frame["oper_part_no"].nunique() == 5000
    assert (raf_frame["demand_qty"] > 0).all()
    assert not raf_frame.duplicated(["oper_part_no", "seq"]).any()
    assert set(raf_frame["chronological_split"].unique()) == {
        "train",
        "validation",
        "test",
    }

    for manifest in (retail_manifest, raf_manifest):
        for artifact in manifest["artifacts"].values():
            path = Path(artifact["path"])
            assert path.exists(), path
            assert path.stat().st_size == artifact["size_bytes"]

    auxiliary_ids = {item["dataset_id"] for item in auxiliary_manifest["datasets"]}
    assert auxiliary_ids == {"yellow_trip_hourly", "instacart"}
    for item in auxiliary_manifest["datasets"]:
        assert item["status"] == "frozen_experiment_ready"
        assert item["checks"] == {
            "held_out_test_evaluated": False,
            "positive_quantity": True,
            "split_artifacts_match": True,
            "unique_entity_sequence_key": True,
        }
        assert sum(item["split_counts"].values()) == item["rows"]
        for artifact in item["artifacts"].values():
            path = Path(artifact["path"])
            assert path.exists(), path
            assert path.stat().st_size == artifact["size_bytes"]

    dataset_ids = {item["dataset_id"] for item in registry["datasets"]}
    assert {"intermittent_v2", "online_retail_ii", "raf_spare_parts"} <= dataset_ids
    assert "m5_walmart" not in dataset_ids
    retail_registry = next(
        item for item in registry["datasets"] if item["dataset_id"] == "online_retail_ii"
    )
    assert retail_registry["status"] == "frozen_experiment_ready"
    for item in registry["datasets"]:
        data_path = item.get("data_path")
        if data_path:
            assert (ROOT.parent / data_path).exists(), data_path

    print("benchmark contracts: PASS")


if __name__ == "__main__":
    main()
