"""Synthetic CPU contracts; no datasets, checkpoints or GPU operations."""
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from models.Titan.common.titans_mac_paper import (
    CausalDepthwiseProjection, PaperMACAttentionBlock, PaperMACBlock, PaperMemoryState, PaperNeuralMemory,
    TitansMACPaperEncoder, apply_rope,
)


class PaperMACTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(192)

    def model(self, **kwargs):
        options = dict(input_dim=2, d_model=8, n_heads=2, n_layers=2, d_ff=16,
                       persistent_memory_size=2, segment_size=4, dropout=0,
                       concat_order="persistent_memory_current")
        options.update(kwargs)
        return TitansMACPaperEncoder(**options).double().eval()

    def test_constructor_requires_explicit_concat_contract(self):
        with self.assertRaises(TypeError):
            TitansMACPaperEncoder()
        with self.assertRaises(ValueError):
            self.model(concat_order="guessed")

    def test_memory_forward_is_residual_gelu_layernorm(self):
        memory = PaperNeuralMemory(4, 16).double()
        state = memory.initial_state(2)
        keys = torch.randn(2, 3, 4, dtype=torch.double)
        expected = torch.stack([
            keys[b] + F.layer_norm(F.linear(F.gelu(F.linear(keys[b], state.weights[0][b])), state.weights[1][b]), (4,))
            for b in range(2)
        ])
        torch.testing.assert_close(memory.read(state, keys), expected)

    def test_associative_gradients_match_independent_autograd(self):
        memory = PaperNeuralMemory(4, 16).double()
        state = memory.initial_state(3)
        keys = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
        values = torch.randn(3, 4, dtype=torch.double, requires_grad=True)
        gradients, loss = memory.associative_gradients(state, keys, values)
        reference_losses, reference_gradients = [], [[], []]
        for b in range(3):
            weights = [w[b].detach().clone().requires_grad_() for w in state.weights]
            predicted = keys[b] + F.layer_norm(F.linear(F.gelu(F.linear(keys[b], weights[0])), weights[1]), (4,))
            reference = (predicted - values[b]).square().sum()
            reference_losses.append(reference)
            row = torch.autograd.grad(reference, weights)
            for layer in range(2):
                reference_gradients[layer].append(row[layer])
        torch.testing.assert_close(loss, torch.stack(reference_losses), rtol=1e-10, atol=1e-10)
        for layer in range(2):
            torch.testing.assert_close(gradients[layer], torch.stack(reference_gradients[layer]), rtol=1e-9, atol=1e-9)

    def test_mixed_outer_gradient_matches_autograd_reference(self):
        memory = PaperNeuralMemory(4, 16).double()
        state = memory.initial_state(1)
        keys = torch.randn(1, 4, dtype=torch.double, requires_grad=True)
        values = torch.randn(1, 4, dtype=torch.double, requires_grad=True)
        analytic, _ = memory.associative_gradients(state, keys, values)
        prediction = keys + F.layer_norm(torch.bmm(state.weights[1], F.gelu(torch.bmm(state.weights[0], keys[..., None])).reshape(1, 16, 1)).squeeze(-1), (4,))
        reference = torch.autograd.grad((prediction - values).square().sum(), state.weights, create_graph=True)
        probes = [torch.randn_like(g) for g in analytic]
        a = torch.autograd.grad(sum((g * p).sum() for g, p in zip(analytic, probes)), (keys, values, *memory.initial_weights), retain_graph=True)
        r = torch.autograd.grad(sum((g * p).sum() for g, p in zip(reference, probes)), (keys, values, *memory.initial_weights))
        for actual, expected in zip(a, r):
            torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-8)

    def test_sequential_write_matches_autograd_recurrence(self):
        memory = PaperNeuralMemory(4, 16, gradient_chunk_size=1).double()
        state = memory.initial_state(2)
        inputs = torch.randn(2, 3, 4, dtype=torch.double)
        mask = torch.tensor([[True, True, True], [True, False, False]])
        observed, _ = memory.write(state, inputs, mask)
        normalized = memory.norm(inputs)
        keys, values = memory.key(normalized), memory.value(normalized)
        theta, eta, alpha = [[torch.sigmoid(layer(normalized)) for layer in layers] for layers in (memory.update_rate, memory.momentum_rate, memory.forget_rate)]
        expected = []
        for b in range(2):
            weights = [w[b].detach().clone().requires_grad_() for w in state.weights]
            momenta = [torch.zeros_like(w) for w in weights]
            for i in range(3):
                if not mask[b, i]:
                    continue
                predicted = keys[b, i] + F.layer_norm(F.linear(F.gelu(F.linear(keys[b, i], weights[0])), weights[1]), (4,))
                gradients = torch.autograd.grad((predicted - values[b, i]).square().sum(), weights, create_graph=True)
                momenta = [eta[j][b, i, :, None] * m - theta[j][b, i, :, None] * g for j, (m, g) in enumerate(zip(momenta, gradients))]
                weights = [(1 - alpha[j][b, i, :, None]) * w + momenta[j] for j, w in enumerate(weights)]
            expected.append(weights)
        for layer in range(2):
            torch.testing.assert_close(observed.weights[layer], torch.stack([x[layer] for x in expected]), rtol=1e-9, atol=1e-9)

    def test_chunk_start_gradient_oracle_with_vector_gates(self):
        memory = PaperNeuralMemory(4, 16, gradient_chunk_size=16).double()
        with torch.no_grad():
            for layers in (memory.update_rate, memory.momentum_rate, memory.forget_rate):
                for layer in layers:
                    layer.weight.normal_(0, .1)
        inputs = torch.randn(1, 18, 4, dtype=torch.double)
        mask = torch.ones(1, 18, dtype=torch.bool)
        state = memory.initial_state(1)
        observed, _ = memory.write(state, inputs, mask)
        normalized = memory.norm(inputs)
        keys, values = memory.key(normalized), memory.value(normalized)
        theta, eta, alpha = [[torch.sigmoid(layer(normalized)) for layer in layers] for layers in (memory.update_rate, memory.momentum_rate, memory.forget_rate)]
        weights = [w[0].detach().clone().requires_grad_() for w in state.weights]
        momenta = [torch.zeros_like(w) for w in weights]
        for i in range(18):
            if i % 16 == 0:
                chunk_weights = weights
            prediction = keys[0, i] + F.layer_norm(F.linear(F.gelu(F.linear(keys[0, i], chunk_weights[0])), chunk_weights[1]), (4,))
            gradients = torch.autograd.grad((prediction - values[0, i]).square().sum(), chunk_weights, create_graph=True)
            momenta = [eta[j][0, i, :, None] * m - theta[j][0, i, :, None] * g for j, (m, g) in enumerate(zip(momenta, gradients))]
            weights = [(1 - alpha[j][0, i, :, None]) * w + momenta[j] for j, w in enumerate(weights)]
        for layer in range(2):
            torch.testing.assert_close(observed.weights[layer][0], weights[layer], rtol=1e-8, atol=1e-8)
        memory.gradient_chunk_size = 1
        sequential, _ = memory.write(state, inputs, mask)
        self.assertGreater((sequential.weights[0] - observed.weights[0]).abs().max().item(), 1e-5)

    def test_b16_full_outer_gradient_matches_reference(self):
        memory = PaperNeuralMemory(4, 16, gradient_chunk_size=16).double()
        inputs = torch.randn(1, 18, 4, dtype=torch.double, requires_grad=True)
        state = memory.initial_state(1)
        actual, _ = memory.write(state, inputs, torch.ones(1, 18, dtype=torch.bool))
        normalized = memory.norm(inputs)
        keys, values = memory.key(normalized), memory.value(normalized)
        theta, eta, alpha = [[torch.sigmoid(layer(normalized)) for layer in layers] for layers in (memory.update_rate, memory.momentum_rate, memory.forget_rate)]
        weights = [w[0] for w in state.weights]
        momenta = [torch.zeros_like(w) for w in weights]
        for i in range(18):
            if i % 16 == 0:
                chunk_weights = weights
            prediction = keys[0, i] + F.layer_norm(F.linear(F.gelu(F.linear(keys[0, i], chunk_weights[0])), chunk_weights[1]), (4,))
            gradients = torch.autograd.grad((prediction - values[0, i]).square().sum(), chunk_weights, create_graph=True)
            momenta = [eta[j][0, i, :, None] * m - theta[j][0, i, :, None] * g for j, (m, g) in enumerate(zip(momenta, gradients))]
            weights = [(1 - alpha[j][0, i, :, None]) * w + momenta[j] for j, w in enumerate(weights)]
        probes = [torch.randn_like(w) for w in weights]
        parameters = (inputs, *memory.parameters())
        actual_grad = torch.autograd.grad(sum((w[0] * p).sum() for w, p in zip(actual.weights, probes)), parameters, allow_unused=True, retain_graph=True)
        reference_grad = torch.autograd.grad(sum((w * p).sum() for w, p in zip(weights, probes)), parameters, allow_unused=True)
        for a, r in zip(actual_grad, reference_grad):
            if a is None or r is None:
                self.assertIs(a, r)
            else:
                torch.testing.assert_close(a, r, rtol=2e-7, atol=2e-7)

    def test_no_grad_and_inference_mode_keep_online_writes(self):
        model = self.model()
        inputs = torch.randn(2, 5, 2, dtype=torch.double)
        regular, state, _ = model.forward_with_state(inputs)
        with torch.no_grad():
            no_grad, no_grad_state, _ = model.forward_with_state(inputs)
        with torch.inference_mode():
            inference, inference_state, _ = model.forward_with_state(inputs)
        torch.testing.assert_close(regular, no_grad)
        torch.testing.assert_close(regular, inference)
        for i, initial in enumerate(model.layers[0].neural_memory.initial_state(2).weights):
            self.assertGreater((state[0].weights[i] - initial).abs().max().item(), 1e-7)
            torch.testing.assert_close(state[0].weights[i], no_grad_state[0].weights[i])
            torch.testing.assert_close(state[0].weights[i], inference_state[0].weights[i])

    def test_outer_loss_reaches_inner_write_and_projection_parameters(self):
        model = self.model()
        inputs = torch.randn(2, 5, 2, dtype=torch.double, requires_grad=True)
        output = model.encode_prefix(inputs)
        (output * torch.randn_like(output)).sum().backward()
        for name, parameter in model.named_parameters():
            if any(token in name for token in ("initial_weights", "update_rate", "momentum_rate", "forget_rate", "neural_memory.key", "neural_memory.value")):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(parameter.grad.abs().max().item(), 0, name)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_padding_and_batch_isolation(self):
        model = self.model()
        inputs = torch.randn(3, 7, 2, dtype=torch.double)
        lengths = [2, 5, 7]
        mask = torch.arange(7)[None] < torch.tensor(lengths)[:, None]
        batched = model.encode_prefix(inputs, mask)
        for b, length in enumerate(lengths):
            single = model.encode_prefix(inputs[b:b + 1, :length])
            torch.testing.assert_close(batched[b], single[0], rtol=1e-9, atol=1e-9)
        altered = inputs.clone(); altered[1:] *= 1000
        torch.testing.assert_close(batched[0], model.encode_prefix(altered, mask)[0])

    def test_excluded_target_cannot_enter_projection_or_gradient(self):
        model = self.model()
        inputs = torch.randn(2, 6, 2, dtype=torch.double, requires_grad=True)
        mask = torch.tensor([[True] * 3 + [False] * 3, [True] * 5 + [False]])
        expected = model.encode_prefix(inputs, mask)
        changed = inputs.detach().clone()
        changed[~mask] = float("nan")
        torch.testing.assert_close(expected, model.encode_prefix(changed, mask))
        expected.sum().backward()
        self.assertEqual(torch.count_nonzero(inputs.grad[~mask]).item(), 0)

    def test_new_forward_resets_state_and_never_changes_parameters(self):
        model = self.model()
        original = {name: value.clone() for name, value in model.state_dict().items()}
        a, b = torch.randn(1, 6, 2, dtype=torch.double), torch.randn(1, 3, 2, dtype=torch.double)
        first = model.encode_prefix(a)
        model.encode_prefix(b)
        torch.testing.assert_close(first, model.encode_prefix(a), rtol=0, atol=0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)

    def test_read_write_read_uses_updated_state(self):
        model = self.model()
        inputs = torch.randn(1, 3, 2, dtype=torch.double)
        reads, writes = [], []
        original_read, original_write = model.layers[0].neural_memory.read, model.layers[0].neural_memory.write
        def read(state, queries):
            reads.append((state, queries))
            return original_read(state, queries)
        def write(state, values, mask):
            result = original_write(state, values, mask)
            writes.append((state, values, result[0]))
            return result
        with patch.object(model.layers[0].neural_memory, "read", side_effect=read), patch.object(model.layers[0].neural_memory, "write", side_effect=write):
            model(inputs)
        self.assertEqual(len(reads), 2)
        self.assertIs(reads[0][0], writes[0][0])
        self.assertIs(reads[1][0], writes[0][2])
        self.assertIs(reads[1][1], writes[0][1])

    def test_no_update_ablation_changes_output_but_not_module_state(self):
        model = self.model()
        inputs = torch.randn(1, 4, 2, dtype=torch.double)
        actual = model.encode_prefix(inputs)
        def no_write(state, values, mask):
            return state, {"associative_loss": values.new_zeros(mask.shape), "write_applied": torch.zeros_like(mask)}
        with patch.object(model.layers[0].neural_memory, "write", side_effect=no_write):
            unchanged = model.encode_prefix(inputs)
        self.assertGreater((actual - unchanged).abs().max().item(), 1e-7)

    def test_causal_projection_no_future_and_l2_queries(self):
        projection = CausalDepthwiseProjection(8, 4, normalize=True).double()
        inputs = torch.randn(2, 7, 8, dtype=torch.double)
        original = projection(inputs)
        changed = inputs.clone(); changed[:, 4:] += 100
        torch.testing.assert_close(original[:, :4], projection(changed)[:, :4], rtol=0, atol=0)
        torch.testing.assert_close(original.norm(dim=-1), torch.ones(2, 7, dtype=torch.double))

    def test_rope_preserves_norm(self):
        inputs = torch.randn(2, 3, 7, 4, dtype=torch.double)
        torch.testing.assert_close(apply_rope(inputs).norm(dim=-1), inputs.norm(dim=-1))

    def test_attention_is_causal_including_persistent_positions(self):
        block = PaperMACAttentionBlock(8, 2, 16, 0, 4).double().eval()
        inputs = torch.randn(2, 8, 8, dtype=torch.double)
        valid = torch.ones(2, 8, dtype=torch.bool)
        actual = block(inputs, valid)
        changed = inputs.clone(); changed[:, 1:] += 100
        # Even the first persistent token cannot see a later persistent token.
        torch.testing.assert_close(actual[:, 0], block(changed, valid)[:, 0], rtol=0, atol=0)
        changed = inputs.clone(); changed[:, 5:] -= 100
        torch.testing.assert_close(actual[:, :5], block(changed, valid)[:, :5], rtol=0, atol=0)

    def test_masked_projection_is_zero_before_and_after_convolution(self):
        projection = CausalDepthwiseProjection(8, 4, normalize=True).double()
        inputs = torch.randn(1, 4, 8, dtype=torch.double)
        mask = torch.tensor([[True, True, False, False]])
        actual = projection(inputs, mask)
        inputs[:, 2:] = float("nan")
        torch.testing.assert_close(actual, projection(inputs, mask))
        self.assertEqual(torch.count_nonzero(actual[:, 2:]).item(), 0)

    def test_metadata_distinguishes_local_sizes_from_paper_experiment(self):
        choices = self.model().implementation_choices()
        self.assertEqual(choices["paper_experiment_reference"]["segment_size"], 512)
        self.assertEqual(choices["segment_size"], 4)
        self.assertEqual(choices["inner_gradient_chunk_size"], 16)
        self.assertIn("requires explicit", choices["concat_equation_figure_conflict"])

    def test_invalid_mask_fails_closed(self):
        inputs = torch.zeros(1, 3, 2, dtype=torch.double)
        for mask in (torch.tensor([[False, False, False]]), torch.tensor([[True, False, True]])):
            with self.assertRaises(ValueError):
                self.model().encode_prefix(inputs, mask)

    def test_literal_concat_variants_are_distinct(self):
        prefix = self.model()
        equation = self.model(concat_order="persistent_current_memory")
        equation.load_state_dict(prefix.state_dict())
        inputs = torch.randn(1, 4, 2, dtype=torch.double)
        self.assertGreater((prefix(inputs) - equation(inputs)).abs().max().item(), 1e-6)

    def test_complete_mac_layers_own_distinct_memory_and_write_all_valid_tokens(self):
        model = self.model()
        inputs = torch.randn(2, 4, 2, dtype=torch.double)
        mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
        _, states, diagnostics = model.forward_with_state(inputs, mask=mask)
        self.assertEqual(len(states), 2)
        self.assertNotEqual(model.layers[0].neural_memory.initial_weights[0].data_ptr(), model.layers[1].neural_memory.initial_weights[0].data_ptr())
        expected = torch.tensor([2 + 2 * 3, 2 + 2 * 2])
        for layer in range(2):
            torch.testing.assert_close(diagnostics["write_applied"][:, layer, 0].sum(1), expected)
            initial = model.layers[layer].neural_memory.initial_state(2)
            self.assertGreater((states[layer].weights[0] - initial.weights[0]).abs().max().item(), 1e-7)

    def test_compact_valid_write_matches_unpadded_at_chunk_boundaries(self):
        memory = PaperNeuralMemory(4, 16, gradient_chunk_size=16).double()
        for length in (1, 15, 16, 17, 255):
            with self.subTest(length=length):
                valid_inputs = torch.randn(1, length, 4, dtype=torch.double)
                padded = torch.full((1, length * 2, 4), float("nan"), dtype=torch.double)
                mask = torch.zeros(1, length * 2, dtype=torch.bool)
                padded[:, ::2] = valid_inputs
                mask[:, ::2] = True
                start = memory.initial_state(1)
                actual, _ = memory.write(start, padded, mask)
                expected, _ = memory.write(start, valid_inputs, torch.ones(1, length, dtype=torch.bool))
                for a, e in zip(actual.weights, expected.weights):
                    torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-9)

    def test_empty_later_segments_do_not_write_persistent_tokens(self):
        model = self.model()
        inputs = torch.randn(2, 7, 2, dtype=torch.double)
        mask = torch.tensor([[True, True, False, False, False, False, False], [True] * 7])
        _, batched_states, diagnostics = model.forward_with_state(inputs, mask=mask)
        _, single_states, _ = model.forward_with_state(inputs[:1, :2])
        self.assertEqual(diagnostics["write_applied"][0, :, 1].sum().item(), 0)
        for batched, single in zip(batched_states, single_states):
            for a, e in zip(batched.weights, single.weights):
                torch.testing.assert_close(a[:1], e, rtol=1e-9, atol=1e-9)

    def test_aggregated_write_matches_reference_state_and_outer_gradient(self):
        memory = PaperNeuralMemory(4, 16, gradient_chunk_size=16).double()
        with torch.no_grad():
            for layers in (memory.update_rate, memory.momentum_rate, memory.forget_rate):
                for layer in layers:
                    layer.weight.normal_(0, .1)
        initial = memory.initial_state(2)
        momentum = tuple((torch.randn_like(w) * .02).requires_grad_() for w in initial.weights)
        state = PaperMemoryState(initial.weights, momentum)
        inputs = torch.randn(2, 35, 4, dtype=torch.double, requires_grad=True)
        mask = torch.rand(2, 35) > .3
        actual, actual_diagnostics = memory.write(state, inputs, mask)
        expected, reference_diagnostics = memory.write_reference(state, inputs, mask)
        for a, e in zip((*actual.weights, *actual.weight_momentum), (*expected.weights, *expected.weight_momentum)):
            torch.testing.assert_close(a, e, rtol=2e-8, atol=2e-8)
        torch.testing.assert_close(actual_diagnostics["associative_loss"], reference_diagnostics["associative_loss"], rtol=2e-8, atol=2e-8)
        probes = [torch.randn_like(w) for w in (*actual.weights, *actual.weight_momentum)]
        parameters = (inputs, *momentum, *memory.parameters())
        a_grads = torch.autograd.grad(sum((w * p).sum() for w, p in zip((*actual.weights, *actual.weight_momentum), probes)), parameters, allow_unused=True, retain_graph=True)
        r_grads = torch.autograd.grad(sum((w * p).sum() for w, p in zip((*expected.weights, *expected.weight_momentum), probes)), parameters, allow_unused=True)
        for a, r in zip(a_grads, r_grads):
            if a is None or r is None:
                self.assertIs(a, r)
            else:
                torch.testing.assert_close(a, r, rtol=3e-7, atol=3e-7)

    def test_float32_aggregated_write_matches_reference(self):
        memory = PaperNeuralMemory(8, 32, gradient_chunk_size=16)
        inputs = torch.randn(3, 35, 8)
        mask = torch.rand(3, 35) > .25
        state = memory.initial_state(3)
        actual, _ = memory.write(state, inputs, mask)
        expected, _ = memory.write_reference(state, inputs, mask)
        for a, e in zip((*actual.weights, *actual.weight_momentum), (*expected.weights, *expected.weight_momentum)):
            torch.testing.assert_close(a, e, rtol=2e-4, atol=2e-4)

    def test_compacted_mac_block_matches_each_unpadded_sequence(self):
        block = PaperMACBlock(8, 2, 16, 0, 2, 32, 2, 4, 16, "persistent_memory_current").double().eval()
        inputs = torch.randn(2, 17, 8, dtype=torch.double)
        lengths = [1, 15]
        mask = torch.arange(17)[None] < torch.tensor(lengths)[:, None]
        batched, state, _ = block(inputs, mask, block.neural_memory.initial_state(2))
        for b, length in enumerate(lengths):
            individual, individual_state, _ = block(inputs[b:b + 1, :length], torch.ones(1, length, dtype=torch.bool), block.neural_memory.initial_state(1))
            torch.testing.assert_close(batched[b, :length], individual[0], rtol=1e-8, atol=1e-8)
            for actual, expected in zip(state.weights, individual_state.weights):
                torch.testing.assert_close(actual[b:b + 1], expected, rtol=1e-8, atol=1e-8)

    def test_selected_width64_config_cpu_forward_backward_is_finite(self):
        model = TitansMACPaperEncoder(concat_order="persistent_memory_current", dropout=0)
        self.assertEqual(model.persistent_memory_size, 128)
        self.assertEqual(model.segment_size, 512)
        inputs = torch.randn(1, 256, 2, requires_grad=True)
        mask = torch.arange(256)[None] < 255
        output = model.encode_prefix(inputs, mask)
        self.assertEqual(output.shape, (1, 64))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertEqual(inputs.grad[:, 255].abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)


if __name__ == "__main__":
    unittest.main()
