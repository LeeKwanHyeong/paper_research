import polars as pl
import pytest
import torch

from models.Titan.common.memory import MemoryAttention
from paper.scripts.run_count_aware_tpp_backbone_control import (
    BACKBONES,
    build_model,
    prepare_count_frame,
    target_outputs,
)

def test_count_frame_removes_quantity_marks_and_keeps_raw_counts() -> None:
    frame = pl.DataFrame({
        "mark": [0, 4],
        "scale_residual": [0.1, 0.9],
        "demand_qty": [1.0, 17.0],
    })

    transformed = prepare_count_frame(frame)

    assert transformed["mark"].to_list() == [0, 0]
    assert transformed["scale_residual"].to_list() == [1.0, 17.0]


@pytest.mark.parametrize("backbone", BACKBONES)
def test_count_aware_models_have_only_time_and_quantity_outputs(backbone: str) -> None:
    model, _ = build_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.0,
        max_seq_len=8,
    )

    assert not hasattr(model, "mark_head")
    assert hasattr(model, "quantity_head")
    assert hasattr(model, "v_t")


def test_target_quantity_is_not_visible_to_history_encoder() -> None:
    torch.manual_seed(7)
    model, _ = build_model(
        "rmtpp",
        hidden_dim=16,
        train_log_mean=1.0,
        max_seq_len=8,
    )
    model.eval()
    dts = torch.tensor([[0.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 3.0]])
    mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    quantities = torch.tensor([[0.0, 0.0, 2.0, 5.0], [0.0, 3.0, 4.0, 6.0]])
    changed = quantities.clone()
    changed[0, -1] = 50.0
    changed[1, -1] = 60.0

    with torch.no_grad():
        original = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
        mutated = target_outputs(
            model,
            dts,
            mask,
            changed,
            lambda_log_qty=1.0,
        )

    assert torch.allclose(original["pred_qty"], mutated["pred_qty"], atol=1e-7)
    assert torch.allclose(
        original["joint_loss"],
        original["time_loss"] + original["log_qty_loss"],
    )


def test_titan_attention_ignores_masked_padding_keys() -> None:
    torch.manual_seed(11)
    attention = MemoryAttention(
        d_model=8,
        n_heads=2,
        contextual_mem_size=0,
        persistent_mem_size=0,
        dropout=0.0,
        use_causal=True,
    ).eval()
    mask = torch.tensor([[True, True, False, False]])
    original = torch.randn(1, 4, 8)
    changed = original.clone()
    changed[:, 2:, :] = 1000.0

    with torch.no_grad():
        left = attention(original, mask=mask)
        right = attention(changed, mask=mask)

    assert torch.isfinite(left).all()
    assert torch.allclose(left[:, :2], right[:, :2], atol=1e-6)
    assert torch.count_nonzero(left[:, 2:]) == 0
