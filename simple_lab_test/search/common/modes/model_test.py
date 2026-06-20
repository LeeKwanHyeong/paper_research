from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import polars as pl
import torch

from simple_lab_test.search.common.configs import ExperimentConfig, RunConfig
from simple_lab_test.search.common.models import (
    build_model,
    canonical_model_name,
    default_thp_candidates,
    default_titan_candidates,
    find_candidate_by_name,
    make_rmtpp_proxy_candidate,
)
from simple_lab_test.search.common.runner import forward_model, set_global_seed
from simple_lab_test.search.common.experiment_utils import (
    ensure_dir,
    save_json,
    to_jsonable,
)


def parse_csv(text: str) -> tuple[str, ...]:
    """
    Parse comma-separated CLI values.
    """
    return tuple(token.strip() for token in text.split(",") if token.strip())


def build_synthetic_batch(
    *,
    batch_size: int,
    seq_len: int,
    num_marks: int,
    device: str,
    left_pad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a small marked TPP batch with optional left padding.

    The final mark id is reserved as padding, matching the project loaders.
    Residual values live in [0, 1), which matches the magnitude-factorized
    value-head target used by RMTPP/TitanTPP/THP.
    """
    if num_marks < 3:
        raise ValueError("num_marks must be at least 3 because the final id is reserved as padding.")
    if seq_len < 3:
        raise ValueError("seq_len must be at least 3 to test next-event prediction.")

    pad_id = int(num_marks - 1)
    real_mark_count = pad_id
    marks = torch.randint(0, real_mark_count, (batch_size, seq_len), device=device)
    dts = torch.rand(batch_size, seq_len, device=device).mul(4.0).add(0.1)
    values = torch.rand(batch_size, seq_len, device=device)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)

    if left_pad:
        max_pad = max(1, min(3, seq_len - 2))
        for row_idx in range(batch_size):
            pad_len = row_idx % (max_pad + 1)
            if pad_len == 0:
                continue
            marks[row_idx, :pad_len] = pad_id
            dts[row_idx, :pad_len] = 0.0
            values[row_idx, :pad_len] = 0.0
            mask[row_idx, :pad_len] = False

    return marks, dts, values, mask


def selected_model_jobs(args: Any) -> list[tuple[str, str, Any]]:
    """
    Resolve model/candidate pairs for the synthetic model test.
    """
    model_names = tuple(canonical_model_name(name) for name in parse_csv(args.models))
    titan_candidates = default_titan_candidates()
    thp_candidates = default_thp_candidates()
    jobs: list[tuple[str, str, Any]] = []

    if "rmtpp" in model_names:
        hidden_dim = int(args.rmtpp_hidden_dim)
        candidate = make_rmtpp_proxy_candidate(hidden_dim, args.rmtpp_rnn_type)
        jobs.append(("rmtpp", candidate.name, candidate))

    if "titantpp" in model_names:
        titan_names = parse_csv(args.titan_candidates) or ("small_lmm",)
        for name in titan_names:
            candidate = find_candidate_by_name(titan_candidates, name)
            jobs.append(("titantpp", candidate.name, candidate))

    if "thp" in model_names:
        thp_names = parse_csv(args.thp_candidates) or ("small",)
        for name in thp_names:
            candidate = find_candidate_by_name(thp_candidates, name)
            jobs.append(("thp", candidate.name, candidate))

    unsupported = sorted(set(model_names) - {"rmtpp", "titantpp", "thp"})
    if unsupported:
        raise ValueError(f"Unsupported model names: {unsupported}")
    if not jobs:
        raise ValueError("No model jobs selected.")
    return jobs


def assert_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    """
    Raise a readable error when a model emits NaN/Inf.
    """
    if not torch.isfinite(tensor).all().item():
        raise FloatingPointError(f"{name} contains NaN or Inf.")


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    """
    Convert nested config fields to compact JSON strings for CSV export.
    """
    flat: dict[str, Any] = {}
    for key, value in row.items():
        jsonable = to_jsonable(value)
        if isinstance(jsonable, (dict, list)):
            flat[key] = json.dumps(jsonable, ensure_ascii=True, sort_keys=True)
        else:
            flat[key] = jsonable
    return flat


@torch.no_grad()
def run_one_model_test(
    *,
    args: Any,
    model_name: str,
    candidate_name: str,
    candidate: Any,
    marks: torch.Tensor,
    dts: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, Any]:
    """
    Instantiate one model and validate the project-standard TPP interface.
    """
    cfg = ExperimentConfig(
        base_dir=str(args.output_dir),
        device=args.device,
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        rmtpp_rnn_type=args.rmtpp_rnn_type,
        rmtpp_mark_emb_dim=args.rmtpp_mark_emb_dim,
        rmtpp_hidden_dim=int(args.rmtpp_hidden_dim),
        loss_mode="residual_only",
    )
    run_cfg = RunConfig(
        dataset_name="synthetic",
        dataset_kind=None,
        model_name=model_name,
        candidate_name=candidate_name,
        candidate=candidate,
        seed=int(args.seed),
        epochs=0,
        scale_base=float(args.scale_base),
        titan_profile="model_test",
    )
    marked_meta = {
        "num_marks": int(args.num_marks),
        "max_order": int(args.num_marks - 2),
        "series_count": int(args.batch_size),
    }

    model, rmtpp_cfg, encoder_cfg = build_model(
        cfg=cfg,
        run_cfg=run_cfg,
        marked_meta=marked_meta,
    )
    model.eval()

    h = forward_model(model, marks, dts, mask)
    out = model.nll(marks, dts, values=values, mask=mask)
    assert_finite_tensor("hidden_state", h)
    for key in ("nll", "nll_marker", "nll_time", "value_loss"):
        assert_finite_tensor(key, out[key])

    h_prev = h[:, -2, :]
    pad_id = int(model.cfg.num_marks - 1)
    logits = model.mark_head(h_prev)[:, :pad_id]
    value_hat = model.predict_value(h_prev)
    pred_mark = torch.argmax(logits, dim=-1)
    qty_hat = model.reconstruct_qty(pred_mark, value_hat)
    dt_hat = model.sample_next_dt(
        h_prev,
        u=torch.full((h_prev.size(0),), 0.5, dtype=h_prev.dtype, device=h_prev.device),
    )

    for name, tensor in {
        "mark_logits": logits,
        "value_hat": value_hat,
        "qty_hat": qty_hat,
        "dt_hat": dt_hat,
    }.items():
        assert_finite_tensor(name, tensor)

    param_count = sum(param.numel() for param in model.parameters())
    return {
        "status": "success",
        "model_name": canonical_model_name(model_name),
        "candidate_name": candidate_name,
        "device": args.device,
        "batch_size": int(args.batch_size),
        "seq_len": int(args.seq_len),
        "num_marks": int(args.num_marks),
        "hidden_shape": list(h.shape),
        "nll": float(out["nll"].item()),
        "nll_marker": float(out["nll_marker"].item()),
        "nll_time": float(out["nll_time"].item()),
        "value_loss": float(out["value_loss"].item()),
        "steps": int(out["steps"].item()),
        "qty_hat_mean": float(qty_hat.mean().item()),
        "dt_hat_mean": float(dt_hat.mean().item()),
        "parameter_count": int(param_count),
        "rmtpp_config": to_jsonable(rmtpp_cfg),
        "encoder_config": to_jsonable(encoder_cfg),
    }


def run_model_test(args: Any) -> list[dict[str, Any]]:
    """
    Run synthetic interface tests and persist a small report table.
    """
    set_global_seed(int(args.seed))
    output_dir = ensure_dir(Path(args.output_dir))
    marks, dts, values, mask = build_synthetic_batch(
        batch_size=int(args.batch_size),
        seq_len=int(args.seq_len),
        num_marks=int(args.num_marks),
        device=args.device,
        left_pad=bool(args.left_pad),
    )

    rows: list[dict[str, Any]] = []
    for model_name, candidate_name, candidate in selected_model_jobs(args):
        try:
            row = run_one_model_test(
                args=args,
                model_name=model_name,
                candidate_name=candidate_name,
                candidate=candidate,
                marks=marks,
                dts=dts,
                values=values,
                mask=mask,
            )
        except Exception as exc:
            row = {
                "status": "failed",
                "model_name": canonical_model_name(model_name),
                "candidate_name": candidate_name,
                "error": repr(exc),
            }
            if args.stop_on_error:
                raise
        rows.append(row)

    save_json(
        {
            "model_test_args": vars(args),
            "rows": rows,
        },
        output_dir / "model_test_summary.json",
    )
    pl.DataFrame([flatten_for_csv(row) for row in rows]).write_csv(output_dir / "model_test_summary.csv")

    return rows
