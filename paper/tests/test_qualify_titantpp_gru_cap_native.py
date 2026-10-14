import unittest
import torch
from paper.scripts import run_titantpp_gru_residual_cap as runner
from paper.scripts.qualify_titantpp_gru_cap_native import synthetic_check


class NativeCheckLogic(unittest.TestCase):
    def test_three_seeds_exercise_cap_and_causality_on_fabricated_cpu_inputs(self):
        torch.set_num_threads(4)
        data = runner.fabricated_data()
        data['loader']['max_seq_len'] = 12
        for seed in (42, 52, 62):
            with self.subTest(seed=seed):
                state = torch.get_rng_state().clone()
                result = synthetic_check(runner, data, seed, device='cpu', batch_size=3)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(result['synthetic_optimizer_updates'], 4)
                self.assertEqual(result['scientific_optimizer_updates'], 0)
                self.assertFalse(result['real_data_loaded'])
                self.assertEqual(result['withheld_target'], 'invariant')

    def test_foreign_seed_and_native_small_batch_rejected(self):
        data = runner.fabricated_data()
        with self.assertRaises(ValueError): synthetic_check(runner, data, 43, device='cpu')
        with self.assertRaises(ValueError): synthetic_check(runner, data, 42, device='cuda:0', batch_size=3)


if __name__ == '__main__':
    unittest.main()
