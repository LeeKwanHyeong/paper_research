import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.Titan.common.memory import HardLocalMemoryMatcher
from paper.scripts.analyze_count_aware_b0_retrieval import (
    LEGACY_INTERMITTENT_SOURCE_REVISION,
    EventParquetSink,
    PrototypeUsageAccumulator,
    aggregate_event_shards,
    b0_counterfactual_outputs,
    build_event_frame,
    git_revision,
    restore_b0,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


def test_hard_lmm_retrieval_trace_preserves_historical_forward() -> None:
    torch.manual_seed(17)
    matcher = HardLocalMemoryMatcher(d_model=8, mem_size=7, topk=3).eval()
    encoded = torch.randn(2, 5, 8)

    enc_n = F.normalize(encoded, p=2, dim=-1)
    memory = matcher.mem.expand(encoded.size(0), -1, -1)
    mem_n = F.normalize(memory, p=2, dim=-1)
    similarity = torch.matmul(enc_n, mem_n.transpose(-2, -1))
    _, expected_indices = torch.topk(similarity, 3, dim=-1)
    expanded_memory = memory.unsqueeze(1).expand(-1, encoded.size(1), -1, -1)
    expected_residual = torch.gather(
        expanded_memory,
        2,
        expected_indices.unsqueeze(-1).expand(-1, -1, -1, encoded.size(-1)),
    ).mean(dim=2)

    residual, trace = matcher.retrieve(encoded)
    observed = matcher(encoded)

    assert torch.equal(trace["prototype_indices"], expected_indices)
    assert torch.allclose(
        trace["topk_similarity"],
        torch.topk(similarity, 3, dim=-1).values,
    )
    assert torch.allclose(residual, expected_residual)
    assert torch.allclose(observed, encoded + expected_residual)


def test_hard_lmm_empty_memory_has_zero_residual() -> None:
    matcher = HardLocalMemoryMatcher(d_model=4, mem_size=0, topk=2)
    encoded = torch.randn(2, 3, 4)

    residual, trace = matcher.retrieve(encoded)

    assert torch.count_nonzero(residual) == 0
    assert trace["prototype_indices"].shape == (2, 3, 0)
    assert trace["topk_similarity"].shape == (2, 3, 0)
    assert torch.equal(matcher(encoded), encoded)


def test_source_revision_override_supports_execution_copies_without_git() -> None:
    assert git_revision("abc123") == "abc123"


def test_legacy_intermittent_checkpoint_is_inferred_only_from_pinned_source(
    tmp_path,
) -> None:
    model, _ = build_count_aware_model(
        "titantpp",
        hidden_dim=16,
        train_log_mean=1.0,
        max_seq_len=6,
    )
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "backbone": "titantpp",
            "encoder_config": {"d_model": 16, "max_len": 6},
            "evaluation_scope": "validation_only",
            "held_out_test_evaluated": False,
            "interface_meta": {
                "mode": "mark_free_count_aware_log_regression",
                "quantity_loss": "mse_on_log1p_quantity",
                "train_target_mean": 1.0,
            },
            "model_state_dict": state,
            "model_state_sha256": canonical_state_dict_sha256(state),
            "source_revision": LEGACY_INTERMITTENT_SOURCE_REVISION,
        },
        checkpoint,
    )

    restored, audit = restore_b0(checkpoint, {"lambda_tail": 0.0}, "cpu")

    assert restored.memory_mode == "static_hard_lmm"
    assert audit["legacy_contract_inferred"] is True
    assert all(audit["checkpoint_checks"].values())


def _b0_fixture() -> tuple[
    CountAwareTitanTPP,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    torch.manual_seed(23)
    model, _ = build_count_aware_model(
        "titantpp",
        hidden_dim=16,
        train_log_mean=1.0,
        train_log_std=0.5,
        max_seq_len=6,
    )
    assert isinstance(model, CountAwareTitanTPP)
    model.eval()
    dts = torch.tensor(
        [
            [0.0, 0.0, 1.0, 2.0, 3.0, 1.0],
            [0.0, 1.0, 1.0, 2.0, 1.0, 4.0],
        ]
    )
    mask = torch.tensor(
        [
            [False, False, True, True, True, True],
            [False, True, True, True, True, True],
        ]
    )
    quantities = torch.tensor(
        [
            [0.0, 0.0, 1.0, 2.0, 4.0, 5.0],
            [0.0, 3.0, 2.0, 7.0, 1.0, 14.0],
        ]
    )
    return model, dts, mask, quantities


def test_b0_counterfactual_matches_official_prediction_and_masks_target() -> None:
    model, dts, mask, quantities = _b0_fixture()
    changed = quantities.clone()
    changed[:, -1] = torch.tensor([50.0, 140.0])

    diagnostic = b0_counterfactual_outputs(model, dts, mask, quantities)
    mutated = b0_counterfactual_outputs(model, dts, mask, changed)
    with torch.no_grad():
        official = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )

    assert torch.allclose(
        diagnostic["pred_memory_on"],
        official["pred_qty"],
        atol=1e-7,
    )
    for key in (
        "pred_memory_on",
        "pred_memory_off",
        "memory_residual_norm",
        "topk_similarity",
        "prototype_indices",
    ):
        assert torch.allclose(diagnostic[key], mutated[key], atol=1e-7)
    assert torch.isfinite(diagnostic["squared_error_delta_on_minus_off"]).all()
    assert torch.allclose(
        diagnostic["log_location_memory_on"] - diagnostic["log_location_memory_off"],
        diagnostic["log_location_shift"],
    )


def test_event_breakdowns_and_prototype_usage_are_accounted(tmp_path) -> None:
    model, dts, mask, quantities = _b0_fixture()
    diagnostic = b0_counterfactual_outputs(model, dts, mask, quantities)
    quantity_contract = {
        "boundaries": [3.0, 6.0, 10.0, 12.0],
        "strata": [
            {"stratum": "le_p50", "stratum_order": 0},
            {"stratum": "p50_p90", "stratum_order": 1},
            {"stratum": "p90_p95", "stratum_order": 2},
            {"stratum": "p95_p99", "stratum_order": 3},
            {"stratum": "gt_p99", "stratum_order": 4},
        ],
    }
    history_contract = {
        "boundaries": [3, 4],
        "strata": [
            {"stratum": "short", "stratum_order": 0},
            {"stratum": "medium", "stratum_order": 1},
            {"stratum": "long", "stratum_order": 2},
        ],
    }
    region_contract = {
        "body_le_p95": ["le_p50", "p50_p90", "p90_p95"],
        "tail_p95_p99": ["p95_p99"],
        "extreme_tail_gt_p99": ["gt_p99"],
    }
    frame, scopes = build_event_frame(
        dataset="synthetic",
        seed=42,
        event_offset=0,
        dataset_index=[(0, 2), (1, 3)],
        part_indices=torch.tensor([0, 1]),
        diagnostic=diagnostic,
        quantity_contract=quantity_contract,
        history_contract=history_contract,
        region_contract=region_contract,
    )
    sink = EventParquetSink(tmp_path / "events", chunk_rows=1)
    sink.append(frame)
    sink.close()

    metrics = aggregate_event_shards(tmp_path / "events")
    overall = metrics.filter(pl.col("scope_type") == "overall").row(
        0,
        named=True,
    )
    assert overall["event_count"] == 2
    assert np.isclose(
        overall["mae_delta_on_minus_off"],
        frame["abs_error_delta_on_minus_off"].mean(),
    )
    assert set(frame["quantity_region"].to_list()) == {
        "body_le_p95",
        "extreme_tail_gt_p99",
    }

    usage = PrototypeUsageAccumulator(
        "synthetic",
        42,
        model.lmm.mem_size,
        model.lmm.topk,
    )
    usage.update(
        diagnostic["prototype_indices"].numpy(),
        diagnostic["topk_similarity"].numpy(),
        scopes,
    )
    overall_rows = [
        row
        for row in usage.rows()
        if row["scope_type"] == "overall" and row["scope"] == "overall"
    ]
    assert np.isclose(sum(row["selection_share"] for row in overall_rows), 1.0)
    assert np.isclose(sum(row["rank1_share"] for row in overall_rows), 1.0)
    assert sum(row["selection_count"] for row in overall_rows) == 8
