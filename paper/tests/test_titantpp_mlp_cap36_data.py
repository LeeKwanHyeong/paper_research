"""Small fabricated-data checks for capped Train initialization."""
import unittest
import numpy as np
from paper.scripts.prepare_titantpp_mlp_cap36_data import train_statistics


class CapDataTests(unittest.TestCase):
    def test_all_train_rows_initialize_but_only_targets_define_raw_scale(self):
        # First event contributes to log initialization, not next-event RMS.
        s = train_statistics([1., 2., 10.], [2., 10.])
        self.assertAlmostEqual(s["train_log_mean"], np.log1p([1., 2., 10.]).mean())
        self.assertAlmostEqual(s["train_log_std"], np.log1p([1., 2., 10.]).std())
        self.assertAlmostEqual(s["raw_scale"], 52. ** .5)
        changed = train_statistics([1., 2., 9.], [2., 9.])
        self.assertNotEqual(s["all_train_quantity_sha256"], changed["all_train_quantity_sha256"])

    def test_invalid_or_constant_values_are_rejected(self):
        for rows, targets in (([], [1]), ([1, float("nan")], [1]), ([-1, 2], [2]),
                              ([1, 2], []), ([1, 2], [-1]), ([2, 2], [2])):
            with self.subTest(rows=rows, targets=targets), self.assertRaises(ValueError):
                train_statistics(rows, targets)


if __name__ == "__main__":
    unittest.main()
