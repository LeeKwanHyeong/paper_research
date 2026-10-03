import numpy as np
import polars as pl
import pytest

from paper.scripts import probe_time_quantity_gradients as probe
from paper.scripts.run_time_quantity_diagnostic import exact_target_population
from paper.scripts.run_taxi_quantity_interface_ablation import make_loader


def test_sample_is_paired_reproducible_without_replacement():
    a = probe.sample_indices(5000, 20260910, 16, 128)
    b = probe.sample_indices(5000, 20260910, 16, 128)
    assert np.array_equal(a, b)
    assert len(a) == len(set(a.tolist())) == 2048
    assert a.min() >= 0 and a.max() < 5000
    assert not np.array_equal(a, probe.sample_indices(5000, 42, 16, 128))


@pytest.mark.parametrize('args', [(10, 1, 2, 8), (10, 1, True, 2), (10, -1, 1, 2), (10, 1, 0, 2)])
def test_sample_refuses_invalid_or_expanded_budget(args):
    with pytest.raises(ValueError):
        probe.sample_indices(*args)


def test_probe_materializes_only_train_and_preserves_canonical_population(tmp_path):
    path = tmp_path/'mixed.parquet'
    rows = []
    for part in ['A', 'B']:
        for seq in range(1, 11):
            split = 'train' if seq <= 6 else 'validation' if seq <= 9 else 'test'
            rows.append(dict(oper_part_no=part, seq=seq, delta_t=1, demand_qty=float(seq) if split=='train' else float('nan'), chronological_split=split))
    pl.DataFrame(rows).write_parquet(path)
    frame = probe.train_frame(path)
    assert frame.height == 12
    assert frame['chronological_split'].unique().to_list() == ['train']
    loader = make_loader(frame, target_split='train', batch_size=2, lookback_weeks=520, max_seq_len=256, shuffle=False, generator=None)
    _, expected = exact_target_population(frame, target_split='train', lookback_weeks=520, max_seq_len=256)
    assert probe.population(loader.dataset, 'train') == {key: expected[key] for key in ['target_count', 'target_identity_sha256', 'target_quantity_sha256']}
    assert len(loader.dataset)==10
