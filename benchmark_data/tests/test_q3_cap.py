"""Small synthetic data tests; no research inference or source-data writes."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_q3_cap.py"
spec = importlib.util.spec_from_file_location("build_q3_cap", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def sample():
    table = pa.table({
        "oper_part_no": pa.array(["a"] * 6, type=pa.large_string()),
        "seq": pa.array([1, 2, 3, 4, 5, 6], type=pa.int64()),
        "demand_dt": pa.array([10, 11, 15, 20, 50, 60], type=pa.int64()),
        "demand_qty": [1., 2., 3., 10., 100000., 999999.],
        "chronological_split": pa.array(["train"] * 4 + ["validation", "test"], type=pa.large_string()),
        "delta_t": pa.array([0, 1, 4, 5, 30, 10], type=pa.int32()),
        "log_qty": [0.] * 6, "log10_qty": [0.] * 6,
        "raw_order": pa.array([0] * 6, type=pa.int32()),
        "mark": pa.array([0] * 6, type=pa.int32()),
        "scale_residual": [0.] * 6, "z": [0.] * 6,
        "unrelated": ["keep"] * 6,
    })
    return table.replace_schema_metadata({b"note": b"preserve me"})


def replace(table, name, values):
    field = table.schema.field(name)
    return table.set_column(table.schema.get_field_index(name), field, pa.array(values, type=field.type))


def test_train_quantile_only_and_preserve_exact_structure():
    original = sample()
    capped, report = module.transform(original, 2., {})
    assert report["cap_value"] == 4.75
    assert report["target_train_q3_reference_only"] == 6.5
    assert capped["demand_qty"].to_pylist() == [1., 2., 3., 4.75, 4.75, 4.75]
    assert capped.schema.equals(original.schema, check_metadata=True)
    for name in original.column_names:
        if name not in module.QUANTITY_COLUMNS:
            assert original[name].equals(capped[name])
    assert report["splits"]["train"]["next_event_target_count"] == 3
    assert report["splits"]["test"]["next_event_target_count"] == 1
    assert "before" not in report["splits"]["test"]
    assert report["splits"]["train"]["after"]["log1p_mean"] == np.log1p([1., 2., 3., 4.75]).mean()
    assert report["splits"]["train"]["next_event_targets_after"]["raw_rms"] == np.sqrt(np.mean(np.square([2., 3., 4.75])))
    changed = replace(original, "demand_qty", [1., 2., 3., 10., 1e10, 1e11])
    assert module.transform(changed, 2., {})[1]["cap_value"] == 4.75


@pytest.mark.parametrize("base", [2., 10.])
def test_derived_columns_recomputed_without_stale_quantities(base):
    capped, report = module.transform(sample(), base, {})
    q = np.array(capped["demand_qty"])
    logs = np.log2(q) if base == 2 else np.log10(q)
    np.testing.assert_array_equal(np.array(capped["log_qty"]), logs)
    np.testing.assert_array_equal(np.array(capped["log10_qty"]), np.log10(q))
    np.testing.assert_array_equal(np.array(capped["raw_order"]), np.floor(logs))
    np.testing.assert_array_equal(np.array(capped["z"]), logs)
    np.testing.assert_array_equal(np.array(capped["scale_residual"]), logs - np.array(capped["mark"]))
    assert report["magnitude_rule"]["max_order"] == int(np.floor(logs[:4]).max())


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1., 0.])
def test_invalid_quantity_rejected(bad):
    with pytest.raises(ValueError):
        module.transform(replace(sample(), "demand_qty", [1., 2., bad, 4., 5., 6.]), 2., {})


def test_unknown_split_duplicate_key_missing_train_rejected():
    with pytest.raises(ValueError, match="Unknown split"):
        module.transform(replace(sample(), "chronological_split", ["oops"] * 6), 2., {})
    with pytest.raises(ValueError, match="Duplicate"):
        module.transform(replace(sample(), "seq", [1, 1, 3, 4, 5, 6]), 2., {})
    with pytest.raises(ValueError, match="No Train"):
        module.transform(replace(sample(), "chronological_split", ["test"] * 6), 2., {})


def test_missing_derived_columns_not_added():
    original = sample().drop(["log_qty", "log10_qty", "raw_order", "mark", "scale_residual", "z"])
    capped, _ = module.transform(original, 2., {})
    assert capped.column_names == original.column_names


def test_parent_mark_rule():
    values = np.array([0] * 999 + [4])
    assert module.fit_max_order(values, {"config": {"min_count": 100, "min_coverage": .999}}) == 4
    assert module.fit_max_order(values, {}) == 4
    assert module.fit_max_order(np.array([0] * 995 + [2] * 4 + [4]), {"config": {"min_count": 100, "min_coverage": .999}}) == 2


def test_build_roundtrip_sources_unchanged_and_overwrite_guard(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    original = source / "fake_with_split.parquet"
    parent = source / "fake_split_manifest.json"
    pq.write_table(sample(), original)
    parent.write_text(json.dumps({"magnitude_rule": {"scale_base": 2.}}))
    old_bytes = original.read_bytes(), parent.read_bytes()
    output = tmp_path / "Cap"
    output.mkdir()
    (output / "README.md").write_text("root-owned documentation")
    result = module.build(source, output, {"fake": (".", "fake", 2.)})
    assert result["status"] == "complete"
    assert old_bytes == (original.read_bytes(), parent.read_bytes())
    assert (output / "README.md").read_text() == "root-owned documentation"
    full = pq.read_table(output / "data/fake/fake_with_split.parquet")
    assert full.schema.equals(sample().schema, check_metadata=True)
    for record in result["files"]:
        assert module.sha256(Path(record["path"])) == record["sha256"]
    assert pq.read_table(output / "data/fake/fake_train_validation.parquet").num_rows == 5
    with pytest.raises(FileExistsError):
        module.build(source, output, {"fake": (".", "fake", 2.)})
    with pytest.raises(ValueError, match="outside"):
        module.build(source, source / "nested", {"fake": (".", "fake", 2.)})


def test_all_inputs_validated_before_any_outputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for prefix, table in (("good", sample()), ("bad", replace(sample(), "demand_qty", [1., 2., 3., 4., 0., 6.]))):
        pq.write_table(table, source / f"{prefix}_with_split.parquet")
        (source / f"{prefix}_split_manifest.json").write_text("{}")
    output = tmp_path / "Cap"
    with pytest.raises(ValueError):
        module.build(source, output, {"first": (".", "good", 2.), "second": (".", "bad", 2.)})
    assert not output.exists()
