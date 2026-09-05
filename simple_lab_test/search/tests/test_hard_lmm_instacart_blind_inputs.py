"""Synthetic parquet contracts for blind context selection and explicit labels."""

import hashlib

import numpy as np
import polars as pl
import pytest
import torch

from data_loader.event_seq_data_module import RMTPPWeekLookbackDataset
from paper.scripts.count_aware_tpp_backbone.core import prepare_count_frame
from paper.scripts.hard_lmm_instacart_blind_inputs import select_contexts, read_observed, read_body_labels


def write_source(tmp_path, lengths=(1, 9, 12, 7), *, values=True):
    records = [{"oper_part_no": 999, "seq": 0, "chronological_split": "test", "delta_t": -99., "demand_qty": -99.}]
    # Int keys expose accidental lexical sorting (2 must precede 10).
    for key, count in zip((1, 2, 10, 20), lengths):
        for end in range(count):
            records.append({"oper_part_no": key, "seq": end,
                "chronological_split": "train", "delta_t": 0.8 if end == 0 else end % 4 + 1.9,
                "demand_qty": float(end * 2 + key)})
        records.append({"oper_part_no": key, "seq": count, "chronological_split": "validation",
                        "delta_t": -99., "demand_qty": -99.})
    frame = pl.DataFrame(records)
    if not values:
        frame = frame.drop("delta_t", "demand_qty")
    path = tmp_path / "source.parquet"
    frame.write_parquet(path)
    return path


def test_metadata_only_native_ids_prior_offsets_and_determinism(tmp_path):
    path = write_source(tmp_path, values=False)
    # The singleton consumes series index 0 but contributes no target offsets.
    prior = {"series_index": torch.tensor([2]), "target_index": torch.tensor([12]),
             "context_end": torch.tensor([4]), "history_length": torch.tensor([5])}
    meta, selection = select_contexts(path, prior)
    assert meta["train_series"] == 4 and meta["train_targets"] == 25
    assert meta["prior_identity_rows_verified"] == 1
    assert selection["series_index"].tolist() == [1, 3]
    assert selection["native_series_key"] == [2, 20]
    other_meta, other = select_contexts(path, prior)
    assert meta == other_meta
    for name in selection:
        np.testing.assert_array_equal(selection[name], other[name])
    for s, end, target in zip(selection["series_index"], selection["context_end"], selection["target_index"]):
        assert target == {1: 0, 3: 19}[s] + end
        limit = {1: 9, 3: 7}[s]
        expected = min(range(2, limit - 1), key=lambda e: hashlib.sha256(
            f"instacart_balanced_temporal_v1:context:20260905:{s}:{e}".encode()).digest())
        assert end == expected
    bad = {**prior, "target_index": torch.tensor([11])}
    with pytest.raises(ValueError, match="offset"):
        select_contexts(path, bad)


def test_input_features_never_materialize_selected_target_values(tmp_path):
    path = write_source(tmp_path)
    meta, selection = select_contexts(path, [])
    features, histories = read_observed(path, selection)
    frame = pl.read_parquet(path).with_row_index("rid")
    target_ids = selection["target_physical_row_id"].tolist()
    poisoned = frame.with_columns(
        pl.when(pl.col("rid").is_in(target_ids)).then(pl.lit("TARGET_POISON")).otherwise(pl.col("demand_qty").cast(pl.String)).alias("demand_qty"),
        pl.when(pl.col("rid").is_in(target_ids)).then(pl.lit("TARGET_GAP_POISON")).otherwise(pl.col("delta_t").cast(pl.String)).alias("delta_t"),
    ).drop("rid")
    # Observed numeric gaps are integer strings here, matching canonical clipping.
    poisoned = poisoned.with_columns(pl.col("delta_t").str.replace(r"\.\d+$", ""))
    poisoned.write_parquet(path)
    same_meta, same_selection = select_contexts(path, [])
    assert meta == same_meta
    for name in selection:
        np.testing.assert_array_equal(selection[name], same_selection[name])
    new_features, new_histories = read_observed(path, selection)
    for name in features:
        np.testing.assert_allclose(features[name], new_features[name])
    for name in histories:
        torch.testing.assert_close(histories[name], new_histories[name])
    with pytest.raises(pl.exceptions.InvalidOperationError):
        read_body_labels(path, selection, [0])


def test_full_dataset_context_and_float32_input_parity(tmp_path):
    path = write_source(tmp_path)
    _, selection = select_contexts(path, [], maximum_series=2)
    features, histories = read_observed(path, selection)
    frame = pl.read_parquet(path).filter(pl.col("chronological_split") == "train")
    dataset = RMTPPWeekLookbackDataset(prepare_count_frame(frame), lookback_weeks=52,
        max_seq_len=64, mode="all", target_splits={"train"})
    for position, target in enumerate(selection["target_index"]):
        sample = dataset[int(target)]
        gap = sample["dts"][sample["mask"]][:-1]
        qty = sample["values"][sample["mask"]][:-1]
        length = len(gap)
        assert dataset.index[int(target)] == (selection["series_index"][position], selection["context_end"][position])
        assert features["history_length"][position] == length
        torch.testing.assert_close(histories["dts"][position, :length], gap, rtol=0, atol=0)
        torch.testing.assert_close(histories["quantities"][position, :length], qty, rtol=0, atol=0)
        assert not histories["mask"][position, length:].any()
    labels = read_body_labels(path, selection, [1, 0])
    assert labels["selection_position"].tolist() == [1, 0]
    for pos, q in zip(labels["selection_position"], labels["quantity"]):
        sample = dataset[int(selection["target_index"][pos])]
        assert q == sample["values"][sample["mask"]][-1].item()


def test_seq_window_time_boundary_and_h3_eligibility(tmp_path):
    records = []
    for key, seqs in ((2, [0, 1, 2, 100]), (10, list(range(110))), (20, [0, 100, 200, 300])):
        records.extend({"oper_part_no": key, "seq": seq, "chronological_split": "train"} for seq in seqs)
    path = tmp_path / "seq.parquet"
    pl.DataFrame(records).write_parquet(path)
    _, selection = select_contexts(path, [])
    assert selection["series_index"].tolist() == [0, 1]
    for key, start, end, length in zip(selection["native_series_key"], selection["context_start"], selection["context_end"], selection["history_length"]):
        seqs = np.asarray([0, 1, 2, 100] if key == 2 else list(range(110)))
        assert start == max(np.searchsorted(seqs, seqs[end] - 51), end - 62)
        assert length == end - start + 1 and 3 <= length <= 63


def test_label_reader_reads_only_explicit_positions_and_does_not_require_gap(tmp_path):
    path = write_source(tmp_path)
    _, selection = select_contexts(path, [])
    expected = read_body_labels(path, selection, [0])
    disallowed = selection["target_physical_row_id"][1:].tolist()
    frame = pl.read_parquet(path).with_row_index("rid").with_columns(
        pl.when(pl.col("rid").is_in(disallowed)).then(pl.lit("UNREQUESTED_LABEL"))
        .otherwise(pl.col("demand_qty").cast(pl.String)).alias("demand_qty")
    ).drop("rid", "delta_t")
    frame.write_parquet(path)
    actual = read_body_labels(path, selection, [0])
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    for positions in ([], [-1], [len(selection["series_index"])], [0, 0]):
        with pytest.raises(ValueError, match="Allowed"):
            read_body_labels(path, selection, positions)


def test_duplicate_and_invalid_identity_contracts_are_rejected(tmp_path):
    path = write_source(tmp_path)
    for excluded in ([99], [-1], [1.5]):
        with pytest.raises(ValueError):
            select_contexts(path, excluded)
    with pytest.raises(ValueError):
        select_contexts(path, [], maximum_series=0)
    frame = pl.read_parquet(path)
    duplicate = frame.filter(pl.col("chronological_split") == "train").head(1)
    pl.concat((frame, duplicate)).write_parquet(path)
    with pytest.raises(ValueError, match="Duplicate"):
        select_contexts(path, [])
