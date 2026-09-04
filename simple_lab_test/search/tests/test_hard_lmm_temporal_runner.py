"""The history extraction boundary must remove target and padding first."""
import numpy as np
import torch

from paper.scripts.hard_lmm_temporal_features import observed_features
from paper.scripts.run_hard_lmm_temporal_diagnostic import extract_histories


class _Dataset:
    index = [(0, 2)]
    seq_lists = [[1, 3, 7, 9]]
    W = 52
    max_len = 6

    def __init__(self, target_dt, target_qty, padding):
        self.target_dt, self.target_qty, self.padding = target_dt, target_qty, padding

    def __getitem__(self, _):
        return {"dts": torch.tensor([self.padding, self.padding, 77., 2., 4., self.target_dt]),
                "values": torch.tensor([self.padding, self.padding, 2., 5., 3., self.target_qty]),
                "mask": torch.tensor([False, False, True, True, True, True])}


def test_extraction_excludes_target_and_padding_even_if_poisoned():
    observed = observed_features(torch.tensor([77., 2., 4.]), torch.tensor([2., 5., 3.]))
    cache = {"target_index": torch.tensor([0]), "series_index": torch.tensor([0]),
             "context_end": torch.tensor([2]), "history_length": torch.tensor([3]),
             "fold": torch.tensor([1]), "quantity": torch.tensor([9.]),
             "stats": torch.tensor([[observed["mean_log_quantity"], observed["latest_deviation"]]])}
    baseline, histories, _ = extract_histories(_Dataset(2., 9., 0.), cache)
    poisoned_cache = {**cache, "quantity": torch.tensor([9999.])}
    changed, altered_histories, _ = extract_histories(_Dataset(float("nan"), 9999., float("nan")), poisoned_cache)
    for key in baseline:
        np.testing.assert_equal(baseline[key], changed[key])
    for before, after in zip(histories[0], altered_histories[0]):
        torch.testing.assert_close(before, after)
    assert baseline["internal_span"].item() == 6.
