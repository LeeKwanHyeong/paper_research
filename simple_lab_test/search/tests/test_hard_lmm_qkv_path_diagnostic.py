from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from paper.scripts import run_hard_lmm_qkv_path_diagnostic as diagnostic


ROOT = Path(__file__).resolve().parents[3]
CONTRACT = ROOT / "paper/contracts/hard_lmm_qkv_path_diagnostic_v1.json"


def test_real_contract_is_exact():
    contract = diagnostic.read_json(CONTRACT)
    diagnostic.validate_contract(contract, CONTRACT)


@pytest.mark.skipif(not os.environ.get("QKV_DIAGNOSTIC_FROZEN_ROOT"),
                    reason="QKV_DIAGNOSTIC_FROZEN_ROOT is not configured")
def test_frozen_405_source_and_all_six_real_checkpoint_routes_and_masks():
    frozen = Path(os.environ["QKV_DIAGNOSTIC_FROZEN_ROOT"])
    evidence = diagnostic.verify_frozen_source(diagnostic.read_json(CONTRACT), frozen)
    assert evidence["verified_python_count"] == 405
    assert evidence["frozen_implementation_commit"].startswith("84668e2")
    runner = ROOT / "paper/scripts/run_hard_lmm_qkv_path_diagnostic.py"
    code = """
import json, runpy, sys
ns = runpy.run_path(sys.argv[1])
repo = ns['Path'](sys.argv[2]).resolve()
frozen_root = ns['Path'](sys.argv[3]).resolve()
contract_path = ns['Path'](sys.argv[4]).resolve()
contract = ns['read_json'](contract_path)
modules = ns['import_frozen_modules'](repo, frozen_root)
result = ns['actual_model_preflight'](repo, contract, modules)
print(json.dumps(result, sort_keys=True))
"""
    process = subprocess.run(
        [sys.executable, "-s", "-c", code, str(runner), str(ROOT), str(frozen), str(CONTRACT)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    result = json.loads(process.stdout.splitlines()[-1])
    assert result["status"] == "passed"
    assert set(result["datasets"]) == {
        "yellow_trip_hourly", "intermittent_frozen_5000", "insta_market_basket",
    }
    assert all(
        len(row["candidate"]["variants"]) == 8
        and row["candidate"]["state_restored_after_all_masks"]
        for row in result["datasets"].values()
    )


def test_fold_rule_uses_canonical_series_id_and_exact_first8_big_endian():
    for value in (7, np.int64(7), "part-A", 12.5):
        canonical = diagnostic.canonical_series_id(value)
        expected = int.from_bytes(
            hashlib.sha256(("qkv-path-v1:" + canonical).encode()).digest()[:8], "big"
        ) % 2
        assert diagnostic.fold_for_series(value) == expected
    assert diagnostic.canonical_series_id(7.0) == "7"


class _Dataset:
    def __init__(self):
        self.parts = [f"series-{index}" for index in range(80)]
        self.index = [(part, context) for part in range(80) for context in range(3)]
        self.seq_lists = [[0, 1, 2, 3] for _ in self.parts]

    def __len__(self):
        return len(self.index)


def test_sampling_is_deterministic_balanced_and_freezes_ids_before_outputs():
    dataset = _Dataset()
    sampling = {
        "seed": 20260908, "targets_per_fold": 24,
        "minimum_unique_series_per_fold": 8,
    }
    first, manifest = diagnostic.sample_train_targets(dataset, sampling)
    second, second_manifest = diagnostic.sample_train_targets(dataset, sampling)
    assert np.array_equal(first, second)
    assert manifest == second_manifest
    assert manifest["status"] == "sample_ids_frozen_before_model_outputs"
    assert manifest["fold_counts"] == {"0": 24, "1": 24}
    assert len(set(first.tolist())) == 48


class _Attention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.causal_q_kernel = nn.Parameter(torch.full((3, dim), 0.1))
        self.causal_k_kernel = nn.Parameter(torch.full((3, dim), 0.2))
        self.causal_v_kernel = nn.Parameter(torch.full((3, dim), 0.3))


class _Layer(nn.Module):
    def __init__(self, dim: int, *, causal: bool):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        self.attn = _Attention(dim) if causal else nn.Identity()

    def forward(self, x, mask=None):
        output = self.linear(x)
        if isinstance(self.attn, _Attention):
            kernel = (
                self.attn.causal_q_kernel + self.attn.causal_k_kernel
                + self.attn.causal_v_kernel
            ).sum(0)
            output = output + x * kernel
        return output * mask.unsqueeze(-1).to(output.dtype)


class _Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.input_proj = nn.Linear(2, dim, bias=False)
        self.layers = nn.ModuleList([_Layer(dim, causal=True), _Layer(dim, causal=False)])
        self.use_pos_emb = False


class _LMM(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mem = nn.Parameter(torch.arange(16 * dim, dtype=torch.float32).reshape(1, 16, dim) / 100)
        self.topk = 4

    def retrieve(self, hidden):
        residual = self.mem[:, :4].mean(1).unsqueeze(1).expand_as(hidden)
        indices = torch.arange(4).view(1, 1, 4).expand(hidden.size(0), hidden.size(1), 4)
        return residual, {"prototype_indices": indices}


class _Model(nn.Module):
    def __init__(self, dim: int = 4):
        super().__init__()
        torch.manual_seed(9)
        self.encoder = _Encoder(dim)
        self.lmm = _LMM(dim)
        self.quantity_head = nn.Linear(dim, 1)
        self.v_t = nn.Linear(dim, 1, bias=False)
        self.b_t = nn.Parameter(torch.tensor(-1.0))
        self.w_raw = nn.Parameter(torch.tensor(-2.0))
        self.time_intercept_limit = 300.0

    @staticmethod
    def continuous_features(dts, quantities, mask):
        values = torch.stack((torch.log1p(dts.clamp_min(0)), torch.log1p(quantities.clamp_min(0))), -1)
        return values * mask.unsqueeze(-1)

    def predict_quantity(self, hidden):
        location = F.softplus(self.quantity_head(hidden).squeeze(-1))
        return location, torch.expm1(location)

    def log_f_dt(self, hidden, duration):
        w = F.softplus(self.w_raw) + 1e-3
        intercept = torch.clamp(self.v_t(hidden).squeeze(-1) + self.b_t, max=self.time_intercept_limit)
        wd = torch.clamp(w * duration, max=10.0)
        return intercept + wd - (torch.exp(intercept) / w) * torch.expm1(wd)

    def encode_task_states(self, dts, quantities, mask, memory_write_mask=None):
        del memory_write_mask
        hidden = self.encoder.input_proj(self.continuous_features(dts, quantities, mask))
        for layer in self.encoder.layers:
            hidden = layer(hidden, mask=mask)
        residual, _ = self.lmm.retrieve(hidden)
        hidden = (hidden + residual) * mask.unsqueeze(-1)
        return hidden, hidden


def _right_pad(dts, quantities, mask):
    positions = torch.arange(mask.size(1)).expand_as(mask)
    order = torch.argsort((~mask).long() * mask.size(1) + positions, dim=1)
    gather = lambda value: torch.gather(value, 1, order)
    right_mask = gather(mask)
    return gather(dts), gather(quantities), right_mask, right_mask.sum(1)


def _official(model, dts, mask, quantities, *, lambda_log_qty):
    del lambda_log_qty
    dts, quantities, mask, lengths = _right_pad(dts, quantities, mask)
    rows = torch.arange(dts.size(0))
    targets, history = lengths - 1, lengths - 2
    visible = quantities.clone()
    visible[rows, targets] = 0
    writes = mask.clone()
    writes[rows, targets] = False
    time_state, quantity_state = model.encode_task_states(dts, visible, mask, memory_write_mask=writes)
    location, prediction = model.predict_quantity(quantity_state[rows, history])
    true = quantities[rows, targets]
    return {
        "pred_qty": prediction,
        "log_qty_loss": torch.square(location - torch.log1p(true)),
        "time_loss": -model.log_f_dt(time_state[rows, history], dts[rows, targets]),
    }


FROZEN_FAKE = diagnostic.FrozenModules(
    prepare_count_frame=None, right_pad_batch=_right_pad, target_outputs=_official,
    Dataset=None, collate=None, checkpoint_load=None, restore_model=None,
)


def _batch():
    dts = torch.tensor([[0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.5, 2.5, 4.0]])
    quantities = torch.tensor([[0.0, 0.0, 2.0, 3.0, 5.0], [1.0, 2.0, 3.0, 4.0, 7.0]])
    mask = torch.tensor([[False, False, True, True, True], [True, True, True, True, True]])
    return dts, mask, quantities


def test_all_three_rows_are_zeroed_only_for_disabled_kernels_and_state_restores():
    model = _Model()
    state_before = diagnostic.canonical_state_sha256(model.state_dict())
    common_before = model.encoder.layers[0].linear.weight.detach().clone()
    with diagnostic.causal_kernel_mask(model, [1, 1, 0]):
        parameters = dict(model.named_parameters())
        assert parameters[diagnostic.KERNEL_KEYS["V"]].count_nonzero() == 0
        assert parameters[diagnostic.KERNEL_KEYS["Q"]].count_nonzero() > 0
        assert torch.equal(model.encoder.layers[0].linear.weight, common_before)
    assert diagnostic.canonical_state_sha256(model.state_dict()) == state_before
    with pytest.raises(RuntimeError):
        with diagnostic.causal_kernel_mask(model, [0, 0, 0]):
            raise RuntimeError("interrupted")
    assert diagnostic.canonical_state_sha256(model.state_dict()) == state_before


def test_float32_legacy_slope_is_promoted_then_float64_terms_are_clamped_exactly():
    intercept = torch.tensor([301.0, -2.0], dtype=torch.float32)
    raw_w = torch.tensor(-2.0, dtype=torch.float32)
    duration = torch.tensor([100.0, 2.0], dtype=torch.float32)
    result = diagnostic.legacy_time_telemetry(intercept, raw_w, duration, intercept_limit=300.0)
    expected_w = (F.softplus(raw_w) + 1e-3).double()
    assert result["w"].dtype == torch.float64
    assert torch.equal(result["w"], expected_w.expand(2))
    assert result["time_intercept_clamped"][0] == 300.0
    assert result["wd_clamped"][0] == 10.0
    assert result["intercept_saturated"].tolist() == [True, False]
    assert result["wd_saturated"].tolist() == [True, False]


def test_synthetic_extraction_matches_official_and_blocks_target_duration_quantity_padding():
    model = _Model().eval()
    dts, mask, quantities = _batch()
    with torch.no_grad():
        extracted = diagnostic.extract_variant_batch(model, dts, mask, quantities, FROZEN_FAKE)
        parity = diagnostic.assert_official_parity(
            model, dts, mask, quantities, extracted, FROZEN_FAKE,
        )
        leakage = diagnostic.assert_target_padding_invariance(
            model, dts, mask, quantities, extracted, FROZEN_FAKE,
        )
    assert max(parity.values()) == 0.0
    assert set(leakage) == set(diagnostic.INVARIANT_FIELDS)
    assert extracted["top4_indices"].shape == (2, 4)
    assert extracted["legacy_time_loss"].dtype == torch.float32
    assert extracted["integral_term"].dtype == torch.float64


def test_gradient_records_use_requested_flat_schema_without_optimizer_or_grad_residue():
    model = _Model().eval()
    dts, mask, quantities = _batch()
    batch = (torch.zeros_like(dts, dtype=torch.long), dts, mask, torch.arange(2), quantities)
    before = diagnostic.canonical_state_sha256(model.state_dict())
    rows = diagnostic.gradient_records(model, {0: [batch], 1: [batch]}, variant="FULL", frozen=FROZEN_FAKE)
    assert len(rows) == 2 * 6
    assert set(rows[0]) == {
        "variant", "fold", "batch_index", "group", "dot_product", "cosine",
        "time_norm", "quantity_norm", "unused_or_zero_gradient",
    }
    assert {row["group"] for row in rows} == {
        "encoder_layer1", "encoder_layer2", "hard_lmm_bank", "new_Q", "new_K", "new_V",
    }
    assert diagnostic.canonical_state_sha256(model.state_dict()) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_npz_is_written_atomically_with_flat_non_object_arrays(tmp_path):
    output = tmp_path / "extraction.npz"
    arrays = {
        "fold": np.asarray([0, 1], dtype=np.int8),
        "series_id": np.asarray(["a", "b"], dtype=str),
        "FULL__prediction": np.asarray([1.0, 2.0], dtype=np.float32),
    }
    diagnostic.save_npz(output, arrays)
    with np.load(output, allow_pickle=False) as loaded:
        assert set(loaded.files) == set(arrays)
        assert loaded["series_id"].dtype.kind == "U"
    assert not output.with_suffix(".npz.tmp").exists()
