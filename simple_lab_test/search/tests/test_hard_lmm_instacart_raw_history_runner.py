import copy
from unittest.mock import Mock

import pytest
import torch

from paper.scripts import run_hard_lmm_instacart_raw_history as run


def test_real_contract_freezes_local_scope_and_matched_capacity():
    contract = run.load_contract()
    assert contract["authorization"]["held_out_test"] is False
    assert contract["authorization"]["server_or_5090_access"] is False
    assert contract["input_boundary"]["raw_dimension"] == contract["input_boundary"]["h_dimension"] == 64
    assert contract["cohort"]["expected_kept_rows"] == 65525
    assert contract["primary_decision"]["paired_series_bootstrap"]["lower_quantile"] == pytest.approx(.05 / 24)


def test_prepare_refuses_existing_paths_before_loading_inputs(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "RAW", tmp_path / "raw")
    monkeypatch.setattr(run, "RESULT", tmp_path / "result")
    run.RAW.mkdir()
    forbidden = Mock(side_effect=AssertionError("input access forbidden"))
    monkeypatch.setattr(run, "_load_frozen_inputs", forbidden)
    with pytest.raises(ValueError, match="overwrite"):
        run.prepare_phase()
    forbidden.assert_not_called()


def test_prepare_success_freezes_input_before_any_target_or_checkpoint_access(monkeypatch, tmp_path):
    contract = copy.deepcopy(run.load_contract())
    contract["cohort"].update(expected_source_rows=4, expected_kept_rows=3,
                              expected_excluded_history_gt32=1)
    contract["frozen_inputs"] = {}
    lengths = torch.tensor([3, 4, 32, 33])
    mask = torch.arange(64)[None, :] < lengths[:, None]
    dts = mask.float()
    quantities = 2 * mask.float()
    histories = {"dts": dts, "quantities": quantities, "mask": mask,
                 "history_length": lengths}
    selection = {
        "history_length": lengths.numpy(),
        "series_index": torch.tensor([10, 11, 12, 13]).numpy(),
        "fold": torch.tensor([0, 1, 0, 1]).numpy(),
        "target_physical_row_id": torch.tensor([100, 101, 102, 103]).numpy(),
    }
    monkeypatch.setattr(run, "RAW", tmp_path / "raw")
    monkeypatch.setattr(run, "RESULT", tmp_path / "result")
    monkeypatch.setattr(run, "load_contract", lambda: contract)
    monkeypatch.setattr(run, "_load_frozen_inputs", lambda _contract: (histories, selection))
    monkeypatch.setattr(run, "source_hashes", lambda: {})
    monkeypatch.setattr(run, "relative", lambda path: str(path))
    monkeypatch.setattr(run.subprocess, "check_output", lambda *args, **kwargs: "deadbeef\n")
    monkeypatch.setattr(run.platform, "platform", lambda: "test-platform")
    forbidden = Mock(side_effect=AssertionError("outcome access forbidden during prepare"))
    monkeypatch.setattr(run, "read_body_labels", forbidden)
    run.prepare_phase()
    marker = run.read(run.RESULT / "input_manifest.json")
    manifest = run.read(run.RESULT / "execution_manifest.json")
    assert marker["target_quantities_read"] is False
    assert marker["checkpoints_restored"] is False
    assert manifest["status"] == "prepared"
    assert manifest["cohort"]["kept_rows"] == 3
    forbidden.assert_not_called()


class _Memory:
    def retrieve(self, local):
        return torch.zeros_like(local), None


class _Frozen(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.lmm = _Memory()

    def _encode_base(self, dts, quantities, mask, memory_write_mask):
        assert torch.equal(mask, memory_write_mask)
        result = torch.zeros((*dts.shape, 64))
        result[..., 0] = dts
        result[..., 1] = quantities
        return result

    def predict_quantity(self, fused):
        log = fused[:, 0] + fused[:, 1]
        return log, torch.expm1(log)


def test_extract_h_uses_last_observed_token_and_never_needs_target():
    histories = {
        "dts": torch.tensor([[2., 3., 0., 0.], [5., 7., 11., 0.]]),
        "quantities": torch.tensor([[13., 17., 0., 0.], [19., 23., 29., 0.]]),
        "mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool),
    }
    model = _Frozen().eval()
    h, base = run.extract_h_and_base(model, histories, [0, 1], batch_size=1)
    assert h.shape == (2, 64)
    torch.testing.assert_close(h[:, :2], torch.tensor([[3., 17.], [11., 29.]]))
    torch.testing.assert_close(base, torch.tensor([20., 40.]))


def test_extract_rejects_non_prefix_padding():
    histories = {
        "dts": torch.tensor([[2., 0., 3., 0.]]),
        "quantities": torch.tensor([[13., 0., 17., 0.]]),
        "mask": torch.tensor([[1, 0, 1, 0]], dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="prefix-valid"):
        run.extract_h_and_base(_Frozen().eval(), histories, [0])


def test_require_stage_rejects_before_any_probe_cache_read(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "RESULT", tmp_path)
    run.save(tmp_path / "execution_manifest.json", {"status": "prepared"})
    with pytest.raises(ValueError, match="Expected execution stage extracted"):
        run._require_stage("extracted")
