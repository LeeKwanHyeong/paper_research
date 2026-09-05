"""Runner boundaries for the frozen transition-error diagnostic."""

from __future__ import annotations

from unittest.mock import Mock

import polars as pl
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from paper.scripts import hard_lmm_transition_error_analysis as analysis
from paper.scripts import run_hard_lmm_transition_error_diagnostic as run


class _TinyHardMemory(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mem_size = 6
        self.topk = 4
        values = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4) / 100.0
        self.mem = nn.Parameter(values)

    def retrieve(self, encoded):
        batch, length, dim = encoded.shape
        starts = torch.arange(length, device=encoded.device) % self.mem_size
        indices = torch.stack(
            tuple((starts + shift) % self.mem_size for shift in range(4)), dim=-1
        ).expand(batch, -1, -1)
        memory = self.mem.expand(batch, -1, -1)
        selected = torch.gather(
            memory.unsqueeze(1).expand(-1, length, -1, -1),
            2,
            indices.unsqueeze(-1).expand(-1, -1, -1, dim),
        )
        residual = selected.mean(dim=2)
        similarity = torch.linspace(1.0, 0.4, 4, device=encoded.device)
        similarity = similarity.expand(batch, length, -1)
        return residual, {
            "prototype_indices": indices,
            "topk_similarity": similarity,
        }


class _TinyModel(nn.Module):
    """A causal toy with the same runner-facing interface as frozen T0."""

    def __init__(self) -> None:
        super().__init__()
        self.lmm = _TinyHardMemory()
        self.quantity_head = nn.Linear(4, 1)
        with torch.no_grad():
            self.quantity_head.weight.copy_(torch.tensor([[0.2, -0.1, 0.05, 0.1]]))
            self.quantity_head.bias.fill_(0.3)

    def _encode_base(self, dts, history_quantities, mask, *, memory_write_mask=None):
        del memory_write_mask
        token = torch.stack(
            (
                dts,
                history_quantities,
                torch.ones_like(dts),
                dts + 0.5 * history_quantities,
            ),
            dim=-1,
        )
        token = token * mask.unsqueeze(-1)
        encoded = token.cumsum(dim=1)
        return encoded * mask.unsqueeze(-1)

    def encode_task_states(self, dts, history_quantities, mask, *, memory_write_mask=None):
        local = self._encode_base(
            dts,
            history_quantities,
            mask,
            memory_write_mask=memory_write_mask,
        )
        residual, _ = self.lmm.retrieve(local)
        fused = (local + residual) * mask.unsqueeze(-1)
        return fused, fused

    def quantity_outputs(self, hidden, true_quantity):
        location = F.softplus(self.quantity_head(hidden).squeeze(-1))
        prediction = torch.expm1(location)
        target = true_quantity.clamp_min(0).log1p()
        loss = (location - target).square()
        zeros = torch.zeros_like(loss)
        return {
            "train_loss": loss,
            "log_mse": loss,
            "distribution_nll": zeros,
            "location_huber": zeros,
            "scale": zeros,
            "tail_aux_loss": zeros,
            "tail_indicator": zeros,
            "point_prediction": prediction,
        }

    def log_f_dt(self, hidden, dt):
        return torch.zeros_like(dt) + hidden[:, 0] * 0.0


def _toy_batch():
    dts = torch.tensor(
        [[0.0, 0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0, 4.0]]
    )
    quantities = torch.tensor(
        [[0.0, 0.0, 2.0, 4.0, 8.0], [0.0, 1.0, 3.0, 7.0, 9.0]]
    )
    mask = torch.tensor(
        [[False, False, True, True, True], [False, True, True, True, True]]
    )
    return dts, mask, quantities


def test_real_contract_pins_scope_fold_gate_and_deferred_work():
    contract = run.load_contract()
    assert tuple(row["dataset"] for row in contract["datasets"]) == run.DATASETS
    assert contract["diagnostic_scope"]["allowed_split"] == "train_only"
    assert contract["diagnostic_scope"]["held_out_test_evaluated"] is False
    assert contract["signal_gate"]["failure_action"] == "do_not_implement_or_train_the_candidate"
    assert contract["deferred_follow_up"]["name"] == "quantile-adaptive loss and checkpoint alignment"
    assert analysis.BOOTSTRAP_SEED == contract["signal_gate"]["paired_series_bootstrap"]["seed"]


def test_fold_hash_is_fixed_and_series_disjoint():
    expected = {
        series: int.from_bytes(
            __import__("hashlib").sha256(f"20260906:{series}".encode()).digest()[:8],
            "big",
        )
        % 2
        for series in range(100)
    }
    assert {run.fold_for_series(series) for series in expected} == {0, 1}
    assert all(run.fold_for_series(series) == fold for series, fold in expected.items())
    assert run.fold_for_series(torch.tensor(17)) == expected[17]


def test_lazy_train_filter_never_materializes_bad_other_splits(tmp_path):
    path = tmp_path / "split.parquet"
    pl.DataFrame(
        {
            "oper_part_no": [1, 1, 1],
            "seq": [1, 2, 3],
            "chronological_split": ["train", "validation", "test"],
            "demand_qty": [3.0, float("nan"), float("inf")],
        }
    ).write_parquet(path)
    frame = run.load_train_frame(path)
    assert frame.height == 1
    assert frame["demand_qty"].item() == 3.0
    assert set(frame["chronological_split"].to_list()) == {"train"}


def test_sham_rotates_only_legal_transition_sources():
    indices = torch.arange(5 * 4).reshape(1, 5, 4)
    observed = torch.tensor([[False, True, True, True, True]])
    writable = torch.tensor([[False, True, True, True, False]])
    sham = run.cyclic_transition_sham_indices(indices, observed, writable)

    # Legal transitions end at positions 2 and 3, hence sources 1 and 2 swap.
    assert torch.equal(sham[0, 1], indices[0, 2])
    assert torch.equal(sham[0, 2], indices[0, 1])
    assert torch.equal(sham[0, 0], indices[0, 0])
    assert torch.equal(sham[0, 3:], indices[0, 3:])


def test_batch_cache_matches_official_prediction_and_required_schema():
    model = _TinyModel().requires_grad_(False).eval()
    dts, mask, quantities = _toy_batch()
    extracted = run.extract_batch_features(model, dts, mask, quantities)
    assert set(run.REQUIRED_CACHE_FIELDS) - {"fold", "series_id"} <= set(extracted)
    assert extracted["final_h"].shape == (2, 4)
    assert extracted["final_top4_indices"].shape == (2, 4)
    assert torch.equal(extracted["history_length"], torch.tensor([2, 3]))
    assert torch.equal(extracted["transition_count"], torch.tensor([1, 2]))
    assert run.assert_official_prediction_parity(
        model, dts, mask, quantities, extracted
    ) == pytest.approx(0.0, abs=1e-6)


def test_target_and_padding_values_cannot_change_transition_features():
    model = _TinyModel().requires_grad_(False).eval()
    dts, mask, quantities = _toy_batch()
    extracted = run.extract_batch_features(model, dts, mask, quantities)
    differences = run.assert_target_padding_feature_invariance(
        model, dts, mask, quantities, extracted
    )
    assert set(differences) == set(run.FEATURE_INVARIANCE_FIELDS)
    assert max(differences.values()) == pytest.approx(0.0, abs=1e-6)


def test_one_event_history_has_exact_zero_transition_summary():
    model = _TinyModel().requires_grad_(False).eval()
    dts = torch.tensor([[0.0, 0.0, 0.0, 2.0, 3.0]])
    quantities = torch.tensor([[0.0, 0.0, 0.0, 4.0, 8.0]])
    mask = torch.tensor([[False, False, False, True, True]])
    extracted = run.extract_batch_features(model, dts, mask, quantities)
    assert extracted["history_length"].item() == 1
    assert extracted["transition_count"].item() == 0
    for name in (
        "selected_occupied_fraction",
        "selected_count_mean",
        "selected_count_std",
        "last_error",
        "unconditioned_mean_error",
        "prototype_conditioned_error",
        "sham_prototype_conditioned_error",
    ):
        assert extracted[name].item() == 0.0


def test_existing_output_refuses_before_dataset_access(monkeypatch, tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(run, "RAW", raw)
    monkeypatch.setattr(run, "RESULT", tmp_path / "result")
    forbidden = Mock(side_effect=AssertionError("dataset access is forbidden"))
    monkeypatch.setattr(run, "load_train_frame", forbidden)
    with pytest.raises(ValueError, match="overwrite"):
        run.extract_phase()
    forbidden.assert_not_called()


def test_failed_or_incomplete_extraction_cannot_fit(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "RESULT", tmp_path / "result")
    monkeypatch.setattr(run, "RAW", tmp_path / "raw")
    run.RESULT.mkdir()
    run.save_json(run.RESULT / "execution_manifest.json", {"status": "failed_extraction"})
    forbidden = Mock(side_effect=AssertionError("fitting is forbidden"))
    monkeypatch.setattr(analysis, "analyze_cache", forbidden)
    with pytest.raises(ValueError, match="incomplete"):
        run.analyze_phase()
    forbidden.assert_not_called()


def test_intermittent_override_resolves_exact_registry_state_and_hashes():
    contract = run.load_contract()
    registry = run.read_json(run.ROOT / contract["reference"]["registry"])
    dataset_contract = contract["datasets"][0]
    row = run._registry_row(registry, "intermittent_v2")
    checkpoint, launch, summary, hashes, used_override = run.resolve_frozen_checkpoint(
        dataset_contract, row
    )
    assert used_override
    assert checkpoint.is_file() and launch.is_file() and summary.is_file()
    assert dataset_contract["local_checkpoint_override"]["checkpoint_state_sha256"] == row[
        "checkpoint_state_sha256"
    ]
    run.verify_hashes(hashes, label="Intermittent override")
