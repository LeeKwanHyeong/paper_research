"""Synthetic frozen-memory contracts only; no datasets, SSH, training or Test."""
import importlib.util
import inspect
from pathlib import Path
import unittest

import torch

from paper.scripts.titantpp_cnn_memory_control import (
    ALLOWED_ALPHAS, frozen_memory_attenuation, memory_state_sha256,
)


SOURCE = (Path(__file__).resolve().parents[2] /
          "search_artifacts/titantpp_components_taxi_raf24_test_20261014_v1/source/"
          "models/Titan/common/memory.py")
spec = importlib.util.spec_from_file_location("frozen_memory_for_synthetic_contract", SOURCE)
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


class FrozenMemoryControlTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.matcher = frozen.HardLocalMemoryMatcher(8, mem_size=11, topk=4).eval()
        self.encoded = torch.randn(5, 7, 8)

    def test_exact_alpha_one_and_state_preservation(self):
        before = memory_state_sha256(self.matcher)
        with torch.no_grad():
            expected = self.matcher(self.encoded)
            with frozen_memory_attenuation(self.matcher, 1) as audit:
                actual = self.matcher(self.encoded)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(before, memory_state_sha256(self.matcher))
        self.assertEqual(audit.state_sha_before, audit.state_sha_after)
        self.assertTrue(audit.restored)
        self.assertNotIn("forward", self.matcher.__dict__)
        self.assertNotIn("retrieve", self.matcher.__dict__)

    def test_all_attenuations_use_exact_original_selected_residual(self):
        with torch.no_grad():
            residual, info = self.matcher.retrieve(self.encoded)
            for alpha in ALLOWED_ALPHAS:
                with self.subTest(alpha=alpha):
                    with frozen_memory_attenuation(self.matcher, alpha) as audit:
                        actual = self.matcher(self.encoded)
                    self.assertTrue(torch.equal(actual, self.encoded + alpha * residual))
                    record = audit.records[0]
                    self.assertTrue(torch.equal(record["prototype_indices"], info["prototype_indices"]))
                    self.assertTrue(torch.equal(record["topk_similarity"], info["topk_similarity"]))
                    if alpha == 0:
                        self.assertIs(actual, self.encoded)

    def test_capture_avoids_subtracting_rounded_output(self):
        # Small residual disappears when first added to a large fp32 input.
        encoded = torch.full((1, 1, 8), 2 ** 25, dtype=torch.float32)
        with torch.no_grad():
            self.matcher.mem.fill_(3.0)
            residual, _ = self.matcher.retrieve(encoded)
            self.assertFalse(torch.equal(self.matcher(encoded) - encoded, residual))
            with frozen_memory_attenuation(self.matcher, .25) as audit:
                actual = self.matcher(encoded)
        self.assertTrue(torch.equal(actual, encoded + .25 * residual))
        torch.testing.assert_close(audit.records[0]["memory_norm"],
                                   torch.linalg.vector_norm(residual.double(), dim=-1), rtol=0, atol=0)

    def test_telemetry_matches_hand_computed_vectors(self):
        matcher = frozen.HardLocalMemoryMatcher(2, mem_size=1, topk=1).eval()
        with torch.no_grad():
            matcher.mem.copy_(torch.tensor([[[3., 4.]]]))
            with frozen_memory_attenuation(matcher, .5) as audit:
                matcher(torch.tensor([[[0., 2.], [0., 0.]]]))
        rec = audit.records[0]
        torch.testing.assert_close(rec["encoded_norm"], torch.tensor([[2., 0.]], dtype=torch.float64))
        torch.testing.assert_close(rec["memory_norm"], torch.tensor([[5., 5.]], dtype=torch.float64))
        torch.testing.assert_close(rec["scaled_memory_norm"], torch.tensor([[2.5, 2.5]], dtype=torch.float64))
        torch.testing.assert_close(rec["cosine_h_memory"], torch.tensor([[.8, 0.]], dtype=torch.float64))

    def test_batch_partition_and_retrieval_consistency(self):
        with torch.no_grad():
            for alpha in ALLOWED_ALPHAS:
                with frozen_memory_attenuation(self.matcher, alpha) as full:
                    combined = self.matcher(self.encoded)
                with frozen_memory_attenuation(self.matcher, alpha) as split:
                    chunks = [self.matcher(part) for part in self.encoded.split(2)]
                torch.testing.assert_close(combined, torch.cat(chunks), rtol=0, atol=0)
                for key in ("encoded_norm", "memory_norm", "prototype_indices", "topk_similarity"):
                    torch.testing.assert_close(full.records[0][key],
                        torch.cat([r[key] for r in split.records]), rtol=1e-6, atol=1e-7)

    def test_sink_streams_without_retaining_records(self):
        received = []
        with torch.no_grad(), frozen_memory_attenuation(self.matcher, .5, received.append) as audit:
            self.matcher(self.encoded)
        self.assertEqual(audit.records, [])
        self.assertEqual(audit.call_count, 1)
        self.assertEqual(len(received), 1)

    def test_nested_rejection_does_not_remove_outer_interception(self):
        with torch.no_grad():
            with frozen_memory_attenuation(self.matcher, 0) as audit:
                with self.assertRaisesRegex(ValueError, "nested"):
                    with frozen_memory_attenuation(self.matcher, 1):
                        pass
                self.assertIs(self.matcher(self.encoded), self.encoded)
            self.assertTrue(audit.restored)
            with frozen_memory_attenuation(self.matcher, 1):
                self.matcher(self.encoded)

    def test_sink_and_inference_error_cleanup(self):
        def fail(_):
            raise RuntimeError("sink failed")
        with torch.no_grad():
            with self.assertRaisesRegex(RuntimeError, "sink failed"):
                with frozen_memory_attenuation(self.matcher, .5, fail):
                    self.matcher(self.encoded)
            with self.assertRaises(ValueError):
                with frozen_memory_attenuation(self.matcher, .5):
                    self.matcher(self.encoded[0])
            self.assertNotIn("forward", self.matcher.__dict__)
            self.assertNotIn("retrieve", self.matcher.__dict__)
            with frozen_memory_attenuation(self.matcher, 1):
                self.matcher(self.encoded)

    def test_target_is_not_an_accepted_api_argument(self):
        self.assertEqual(set(inspect.signature(frozen_memory_attenuation).parameters),
                         {"matcher", "alpha", "telemetry_sink"})
        with torch.no_grad():
            with self.assertRaises(TypeError):
                with frozen_memory_attenuation(self.matcher, .5, target=torch.ones(5)):
                    pass
            with frozen_memory_attenuation(self.matcher, .5):
                with self.assertRaises(TypeError):
                    self.matcher(self.encoded, target=torch.ones(5))

    def test_invalid_alpha_eval_and_gradient_guards(self):
        for alpha in (float("nan"), float("inf"), -.25, 1.1, .75, True, torch.tensor(.5)):
            with self.subTest(alpha=alpha), torch.no_grad(), self.assertRaises(ValueError):
                with frozen_memory_attenuation(self.matcher, alpha):
                    pass
        with self.assertRaisesRegex(ValueError, "disabled gradients"):
            with frozen_memory_attenuation(self.matcher, 1):
                pass
        self.matcher.train()
        with torch.no_grad(), self.assertRaisesRegex(ValueError, "eval mode"):
            with frozen_memory_attenuation(self.matcher, 1):
                pass

    def test_preexisting_hooks_and_subclasses_are_rejected(self):
        hook = self.matcher.register_forward_hook(lambda _m, _a, out: out)
        try:
            with torch.no_grad(), self.assertRaisesRegex(ValueError, "hooks"):
                with frozen_memory_attenuation(self.matcher, 1):
                    pass
        finally:
            hook.remove()
        weighted = frozen.SimilarityWeightedLocalMemoryMatcher(8).eval()
        with torch.no_grad(), self.assertRaises(TypeError):
            with frozen_memory_attenuation(weighted, 1):
                pass

    def test_parameter_mutation_is_detected_without_rewriting_original(self):
        before = self.matcher.mem.detach().clone()
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "parameters/buffers changed"):
            with frozen_memory_attenuation(self.matcher, .5):
                self.matcher.mem.add_(1)
        self.assertTrue(torch.equal(self.matcher.mem, before + 1))
        self.assertNotIn("forward", self.matcher.__dict__)
        self.assertNotIn("retrieve", self.matcher.__dict__)

    def test_empty_bank_and_explicit_memory_are_supported(self):
        empty = frozen.HardLocalMemoryMatcher(8, mem_size=0, topk=4).eval()
        explicit = torch.randn(3, 8)
        with torch.no_grad():
            with frozen_memory_attenuation(empty, .25) as audit:
                self.assertTrue(torch.equal(empty(self.encoded), self.encoded))
                residual, _ = self.matcher.retrieve(self.encoded, explicit)
                actual = empty(self.encoded, explicit)
                torch.testing.assert_close(actual, self.encoded + .25 * residual, rtol=0, atol=0)
        self.assertEqual(audit.records[0]["prototype_indices"].shape, (5, 7, 0))

    def test_inference_mode_and_nonpersistent_scalar_buffer_hash(self):
        self.matcher.register_buffer("audit_counter", torch.tensor(0), persistent=False)
        with torch.inference_mode(), frozen_memory_attenuation(self.matcher, .5) as audit:
            self.matcher(self.encoded)
        self.assertEqual(audit.state_sha_before, audit.state_sha_after)
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "parameters/buffers changed"):
            with frozen_memory_attenuation(self.matcher, .5):
                self.matcher.audit_counter.add_(1)

    def test_configuration_mutation_is_reported_and_methods_restored(self):
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, "mode/configuration changed"):
            with frozen_memory_attenuation(self.matcher, .5):
                self.matcher.topk = 1
        self.assertNotIn("forward", self.matcher.__dict__)
        self.assertNotIn("retrieve", self.matcher.__dict__)


if __name__ == "__main__":
    unittest.main()
