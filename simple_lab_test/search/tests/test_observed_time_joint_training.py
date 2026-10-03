"""Synthetic CPU integration of the opt-in recorded-duration joint objective."""
from __future__ import annotations

import copy
import hashlib
import json
import sys

import numpy as np
import polars as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import TIME_HEAD_MODE_HETEROSCEDASTIC_LOGNORMAL_DURATION as HETERO
from paper.scripts import run_count_aware_tpp_backbone_control as cli
from paper.scripts.count_aware_tpp_backbone import observed_time as obs, training
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from simple_lab_test.search.common.runner import torch_load_checkpoint

VARIANT = "count_only_log_regression"


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def args_for(monkeypatch, tmp_path, backbone="titantpp", name="run"):
    monkeypatch.setattr(sys, "argv", ["runner", "--data", str(tmp_path / "synthetic.parquet"),
        "--split-manifest", str(tmp_path / "synthetic.json"), "--output-dir", str(tmp_path / name),
        "--source-revision", "1" * 40, "--execution-role", "synthetic_cpu_only", "--device", "cpu",
        "--model-role", obs.ROLE, "--backbones", backbone, "--quantity-variants", VARIANT,
        "--time-head-mode", HETERO, "--time-scale", "3", "--dataset-contract", "insta_market_basket",
        "--checkpoint-monitor", "validation_raw_quantity_rmse", "--epochs", "3", "--min-epochs", "3",
        "--early-stopping-patience", "3", "--batch-size", "4", "--hidden-dim", "16", "--max-seq-len", "8",
        "--allow-partial-contract"])
    return cli.parse_args()


def interface():
    return {"train_target_mean": 1.5, "train_target_std": 1.0, "data_scope": "synthetic_cpu_only",
        "time_head": {"mode": HETERO, "time_scale": 3.0, "time_initial_location": 0.1,
        "time_initial_scale": 0.7, "statistics_source_split": "train",
        "train_time_statistics": {"time_scale": 3.0, "target_log_scaled_mean": 0.1,
                                  "target_log_scaled_std": 0.7},
        "observation_likelihood": obs.observation_contract("insta_market_basket")}}


def grid_frame():
    return pl.DataFrame({"delta_t": [0, 1, 2, 30], "chronological_split": ["train"] * 3 + ["validation"]})


def dataset(split):
    n = 9 if split == "train" else 7
    dts = torch.tensor([1., 2., 3., 7., 30., 1.]).repeat(n, 1)
    qty = 1 + torch.arange(n * 6).reshape(n, 6).float() / 3
    mask = torch.arange(6)[None] < (torch.arange(n) % 5 + 2)[:, None]
    dts[~mask] = 0; qty[~mask] = 0
    return TensorDataset(torch.zeros_like(dts), dts, mask, torch.zeros_like(dts), qty)


def install_loader(monkeypatch, traces):
    class Recorded:
        def __init__(self, loader, split):
            self.loader, self.split = loader, split
        def __getattr__(self, name):
            return getattr(self.loader, name)
        def __len__(self):
            return len(self.loader)
        def __iter__(self):
            h = hashlib.sha256()
            for batch in self.loader:
                for t in batch:
                    h.update(t.contiguous().numpy().tobytes())
                yield batch
            traces[self.split].append(h.hexdigest())
    def loader(_frame, *, target_split, batch_size, shuffle, generator, **_):
        return Recorded(DataLoader(dataset(target_split), batch_size=batch_size, shuffle=shuffle,
                                   generator=generator), target_split)
    monkeypatch.setattr(training, "make_loader", loader)


def run(args, backbone, meta=None, frame=None):
    return training.train_one(args=args, frame=grid_frame() if frame is None else frame,
        quantity_contract={"boundaries": [], "strata": [{"label": "all"}]},
        interface_meta=interface() if meta is None else meta, backbone=backbone,
        quantity_variant=VARIANT, seed=42)


def directory(args, backbone="titantpp"):
    return args.output_dir / "runs" / backbone / VARIANT / "seed_42"


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: same(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b, strict=True): same(x, y)
    else: assert a == b


def build(backbone, dataset_id="insta_market_basket", observation=True):
    return build_count_aware_model(backbone, hidden_dim=16, train_log_mean=1.5,
        max_seq_len=8, time_head_mode=HETERO, time_scale=3, time_initial_location=.1,
        time_initial_scale=.7, time_observation_contract=obs.observation_contract(dataset_id) if observation else None)


@pytest.mark.parametrize("backbone", obs.BACKBONES)
def test_observation_changes_objective_not_initial_parameters_or_rng_and_reaches_encoder(backbone):
    torch.manual_seed(9); density, _ = build(backbone, observation=False)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(9); model, meta = build(backbone)
    assert torch.equal(old_rng, torch.get_rng_state())
    same(density.state_dict(), model.state_dict())
    assert meta["time_head"]["observation_likelihood"] == obs.observation_contract("insta_market_basket")
    model.eval()
    _, dts, mask, _, qty = next(iter(DataLoader(dataset("train"), batch_size=9)))
    out = target_outputs(model, dts, mask, qty, lambda_log_qty=1)
    assert out["time_loss"].dtype == torch.float64
    assert torch.isfinite(out["joint_loss"]).all() and (out["time_loss"] >= 0).all()
    out["time_loss"].mean().backward()
    head_names = {n for n, _ in model.time_head_named_parameters()}
    encoder_grads = [p.grad for n,p in model.named_parameters() if n not in head_names and not n.startswith("quantity_head") and p.grad is not None]
    assert encoder_grads and all(torch.isfinite(g).all() for g in encoder_grads)
    assert sum(g.abs().sum().item() for g in encoder_grads) > 0
    model.zero_grad()
    # Quantity head starts at zero weights; activate it to test its encoder path.
    with torch.no_grad(): model.quantity_head.weight.fill_(.02)
    out = target_outputs(model, dts, mask, qty, lambda_log_qty=1)
    out["quantity_train_loss"].mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n,p in model.named_parameters()
               if n not in head_names and not n.startswith("quantity_head"))


def test_five_backbones_real_trainer_equal_batches_selectors_and_cache(monkeypatch, tmp_path):
    all_traces = []
    for backbone in obs.BACKBONES:
        traces = {"train": [], "validation": []}; install_loader(monkeypatch, traces)
        args = args_for(monkeypatch, tmp_path, backbone)
        summary, _, _ = run(args, backbone)
        last = torch_load_checkpoint(directory(args, backbone) / "last_epoch_state.pt", map_location="cpu")
        selected = torch_load_checkpoint(directory(args, backbone) / "best_val_qty_rmse_model.pt", map_location="cpu")
        assert summary["status"] == "success" and summary["completed_epochs"] == 3
        assert [r["epoch"] for r in last["history"]] == [1,2,3]
        earliest = min(last["history"], key=lambda r: r["val_qty_rmse"])
        assert selected["best_epoch"] == summary["best_epoch"] == earliest["epoch"]
        assert all(r["train_all_finite"] for r in last["history"])
        assert last["resume_identity"]["interface_meta"]["time_head"]["observation_likelihood"]["top_code"] == 30
        assert all(v["step"].item() == 9 for v in last["optimizer_state_dict"]["state"].values())
        before = copy.deepcopy(traces); cached, _, _ = run(args, backbone)
        assert cached["checkpoint_state_sha256"] == summary["checkpoint_state_sha256"]
        assert before == traces
        all_traces.append(traces)
    assert all(t == all_traces[0] for t in all_traces)


@pytest.mark.parametrize("backbone", obs.BACKBONES)
def test_prediction_state_excludes_target_and_padding_values(monkeypatch, backbone):
    torch.manual_seed(12)
    model, _ = build(backbone)
    model.eval()
    _, dts, mask, _, qty = next(iter(DataLoader(dataset("train"), batch_size=9)))
    captured = []
    original = model.log_observation_dt
    def capture(hidden, target):
        captured.append(hidden.detach().clone())
        return original(hidden, target)
    monkeypatch.setattr(model, "log_observation_dt", capture)
    with torch.no_grad():
        before = target_outputs(model, dts, mask, qty, lambda_log_qty=1)
        changed_dt, changed_qty = dts.clone(), qty.clone()
        rows, targets = torch.arange(dts.size(0)), mask.sum(1) - 1
        changed_dt[rows, targets] = 2
        changed_qty[rows, targets] += 100
        changed_dt[~mask] = 999
        changed_qty[~mask] = 999
        after = target_outputs(model, changed_dt, mask, changed_qty, lambda_log_qty=1)
    torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
    torch.testing.assert_close(before["pred_qty"], after["pred_qty"], rtol=0, atol=0)
    assert not torch.equal(before["true_qty"], after["true_qty"])


def test_full_and_resumed_training_match_and_observation_identity_cannot_change(monkeypatch, tmp_path):
    traces = {"train": [], "validation": []}; install_loader(monkeypatch, traces)
    full_args = args_for(monkeypatch, tmp_path, name="full")
    full,_,_ = run(full_args, "titantpp")
    resumed_args = args_for(monkeypatch, tmp_path, name="resumed")
    real_save = training.atomic_torch_save
    def interrupt(payload, path):
        real_save(payload, path)
        if path.name == "last_epoch_state.pt" and payload["epoch"] == 1:
            raise RuntimeError("synthetic interruption")
    monkeypatch.setattr(training, "atomic_torch_save", interrupt)
    with pytest.raises(RuntimeError, match="synthetic interruption"): run(resumed_args, "titantpp")
    monkeypatch.setattr(training, "atomic_torch_save", real_save)
    resumed,_,_ = run(resumed_args, "titantpp")
    assert full["checkpoint_state_sha256"] == resumed["checkpoint_state_sha256"]
    a = torch_load_checkpoint(directory(full_args) / "last_epoch_state.pt", map_location="cpu")
    b = torch_load_checkpoint(directory(resumed_args) / "last_epoch_state.pt", map_location="cpu")
    for key in ("model_state_dict","best_state_dict","optimizer_state_dict","rng_state","train_loader_generator_state","history"):
        same(a[key], b[key])
    wrong = interface(); wrong["time_head"]["observation_likelihood"]["top_code"] = None
    with pytest.raises(ValueError, match="identity mismatch"): run(resumed_args, "titantpp", wrong)
    wrong = interface(); del wrong["time_head"]["observation_likelihood"]
    with pytest.raises(ValueError, match="identity mismatch"): run(resumed_args, "titantpp", wrong)
    wrong = interface(); wrong["time_head"]["time_scale"] = 9
    with pytest.raises(ValueError, match="initialization"): run(resumed_args, "titantpp", wrong)


@pytest.mark.parametrize("value", [-1., 1.5, 31., float("nan")])
def test_direct_trainer_rejects_invalid_recordings_before_cast_or_cache(monkeypatch, tmp_path, value):
    args = args_for(monkeypatch, tmp_path)
    frame = pl.DataFrame({"delta_t": [value], "chronological_split": ["train"]})
    with pytest.raises(ValueError, match="Recorded"): run(args, "titantpp", frame=frame)


def test_train_only_initialization_ignores_validation_values():
    frame = pl.DataFrame({"oper_part_no": ["a"] * 8, "seq": list(range(8)),
        "delta_t": [1,2,3,4,2,1,2,3], "mark": [0]*8, "scale_residual": [1.]*8,
        "chronological_split": ["train"]*5 + ["validation"]*3})
    changed = frame.with_columns(pl.when(pl.col("chronological_split") == "validation")
        .then(30).otherwise(pl.col("delta_t")).alias("delta_t"))
    kwargs = {"lookback_weeks": 52, "max_seq_len": 8}
    assert cli.derive_train_time_contract(frame, **kwargs) == cli.derive_train_time_contract(changed, **kwargs)


def test_historical_role_cannot_enable_new_objective(monkeypatch, tmp_path):
    args = args_for(monkeypatch, tmp_path)
    args.model_role = "experimental"
    with pytest.raises(ValueError, match="dedicated runner"): cli.run(args)


@pytest.mark.parametrize("dataset_id", list(obs.DATASETS))
def test_production_cli_on_synthetic_parquet_uses_one_observation_law(monkeypatch, tmp_path, dataset_id):
    # Every row is fabricated. The deliberately invalid excluded row proves
    # filtering happens before materialization/grid validation in this route.
    rows = []
    for part in range(3):
        for i in range(12):
            rows.append({"oper_part_no": str(part), "seq": i, "delta_t": [0,1,2,3,4,7][i % 6],
                         "demand_qty": float(1 + i + part * 20),
                         "chronological_split": "train" if i < 8 else "validation"})
    rows.append({"oper_part_no": "synthetic_excluded", "seq": 0, "delta_t": -999,
                 "demand_qty": -999., "chronological_split": "test"})
    raw = pl.DataFrame(rows)
    path = tmp_path / "synthetic.parquet"; raw.write_parquet(path)
    manifest = tmp_path / "synthetic.json"; manifest.write_text('{"synthetic": true}')
    entry = dict(cli.DATASET_CONTRACTS[dataset_id])
    entry.update(data_sha256=cli.sha256_file(path), split_manifest_sha256=cli.sha256_file(manifest))
    monkeypatch.setitem(cli.DATASET_CONTRACTS, dataset_id, entry)
    args = args_for(monkeypatch, tmp_path)
    args.dataset_contract = dataset_id
    args.hidden_dim = 64; args.max_seq_len = entry["max_seq_len"]; args.lookback_weeks = entry["lookback"]
    args.epochs = args.min_epochs = args.early_stopping_patience = 1
    args.seeds = "42"
    admitted = raw.filter(pl.col("chronological_split").is_in(["train", "validation"]))
    stats = cli.derive_train_time_contract(cli.prepare_count_frame(admitted),
        lookback_weeks=args.lookback_weeks, max_seq_len=args.max_seq_len)
    args.time_scale = stats["time_scale"]
    cli.run(args)
    launch = json.loads((args.output_dir / "launch_contract.json").read_text())
    summary = json.loads((directory(args) / "summary.json").read_text())
    assert launch["split_rows"] == {"train": 24, "validation": 12}
    assert launch["time_head"]["reported_metric"] == "recorded_positive_integer_time_nll"
    assert launch["time_head"]["observation_likelihood"] == obs.observation_contract(dataset_id)
    assert launch["time_head"]["train_time_statistics"] == stats
    assert summary["completed_epochs"] == 1 and summary["status"] == "success"
    assert summary["encoder_config"]["time_head"]["observation_likelihood"] == obs.observation_contract(dataset_id)
    assert summary["best_val_time_nll"] >= 0


def test_strict_tie_retains_earliest_observation_checkpoint(monkeypatch, tmp_path):
    traces = {"train": [], "validation": []}; install_loader(monkeypatch, traces)
    args = args_for(monkeypatch, tmp_path)
    actual = training.evaluate
    def tied(**kwargs):
        result = actual(**kwargs)
        return {**result, "qty_rmse": 42.0}
    monkeypatch.setattr(training, "evaluate", tied)
    summary, _, _ = run(args, "titantpp")
    assert summary["best_epoch"] == 1
    last = torch_load_checkpoint(directory(args) / "last_epoch_state.pt", map_location="cpu")
    assert last["model_state_sha256"] != summary["checkpoint_state_sha256"]
