from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import polars as pl
import torch

from models.RMTPPs.value_conditioning import predict_value_for_marks
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
    if args.qty_mark_gradient_mode == "detached" and set(model_names) != {"titantpp"}:
        raise ValueError(
            "qty_mark_gradient_mode='detached' currently supports TitanTPP-only model tests."
        )
    if args.value_encoder_gradient_mode == "detached":
        if set(model_names) != {"titantpp"}:
            raise ValueError(
                "value_encoder_gradient_mode='detached' currently supports "
                "TitanTPP-only model tests."
            )
        if (
            args.value_head_mode != "mark_conditioned_experts"
            or args.qty_mark_gradient_mode != "detached"
        ):
            raise ValueError(
                "value_encoder_gradient_mode='detached' requires "
                "value_head_mode='mark_conditioned_experts' and "
                "qty_mark_gradient_mode='detached'."
            )
    lambda_ordinal = float(args.lambda_ordinal)
    if not math.isfinite(lambda_ordinal) or lambda_ordinal < 0.0:
        raise ValueError("lambda_ordinal must be finite and non-negative.")
    if args.marker_loss_mode == "ce" and lambda_ordinal != 0.0:
        raise ValueError("marker_loss_mode='ce' requires lambda_ordinal=0.")
    if args.marker_loss_mode == "ce_rps":
        if set(model_names) != {"titantpp"}:
            raise ValueError(
                "marker_loss_mode='ce_rps' currently supports TitanTPP-only model tests."
            )
        if lambda_ordinal <= 0.0:
            raise ValueError("marker_loss_mode='ce_rps' requires lambda_ordinal>0.")
    if args.qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}:
        decoder_mode = str(args.qty_decoder_mode)
        if set(model_names) != {"titantpp"}:
            raise ValueError(f"{decoder_mode} supports TitanTPP-only model tests.")
        if not math.isclose(float(args.scale_base), 2.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{decoder_mode} model-test requires scale_base=2.0.")
        if args.marker_loss_mode != "ce" or lambda_ordinal != 0.0:
            raise ValueError(f"{decoder_mode} model-test requires plain marker CE.")
        if decoder_mode == "direct_log_qty" and args.magnitude_norm_mode != "global":
            raise ValueError("direct_log_qty model-test supports only global normalization.")
    q3_active = (
        args.magnitude_encoder_gradient_mode != "coupled"
        or args.magnitude_aux_loss_mode != "none"
    )
    if q3_active:
        if set(model_names) != {"titantpp"}:
            raise ValueError("Q3 magnitude modes support TitanTPP-only model tests.")
        if args.qty_decoder_mode != "direct_raw_qty":
            raise ValueError("Q3 magnitude modes require direct_raw_qty.")
        if args.magnitude_norm_mode != "causal_shrinkage_revin":
            raise ValueError("Q3 magnitude modes require causal_shrinkage_revin.")
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
        value_head_mode=str(args.value_head_mode),
        qty_mark_gradient_mode=str(args.qty_mark_gradient_mode),
        value_encoder_gradient_mode=str(args.value_encoder_gradient_mode),
        marker_loss_mode=str(args.marker_loss_mode),
        lambda_ordinal=float(args.lambda_ordinal),
        qty_decoder_mode=str(args.qty_decoder_mode),
        magnitude_norm_mode=str(args.magnitude_norm_mode),
        magnitude_input_emb_dim=int(args.magnitude_input_emb_dim),
        lambda_magnitude=float(args.lambda_magnitude),
        magnitude_encoder_gradient_mode=str(args.magnitude_encoder_gradient_mode),
        magnitude_aux_loss_mode=str(args.magnitude_aux_loss_mode),
        lambda_log_qty=float(args.lambda_log_qty),
        log_qty_huber_delta=float(args.log_qty_huber_delta),
        log_qty_floor=float(args.log_qty_floor),
        magnitude_sigma_floor=float(args.magnitude_sigma_floor),
        magnitude_revin_eps=float(args.magnitude_revin_eps),
        magnitude_shrinkage_k=float(args.magnitude_shrinkage_k),
        magnitude_center_mode=str(args.magnitude_center_mode),
        magnitude_revin_affine=bool(args.magnitude_revin_affine),
        magnitude_stat_context_mode=str(args.magnitude_stat_context_mode),
        magnitude_exp_clamp_min=float(args.magnitude_exp_clamp_min),
        magnitude_exp_clamp_max=float(args.magnitude_exp_clamp_max),
        loss_mode=(
            "hybrid"
            if args.qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}
            else "residual_only"
        ),
        train_loss_scope=(
            "target_only"
            if args.qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}
            else "all"
        ),
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
    history_mask = mask.clone()
    history_mask[:, -1] = False
    synthetic_z = marks.clamp(max=int(args.num_marks - 2)).float() + values
    train_z = synthetic_z[history_mask]
    train_magnitude = torch.exp2(train_z) if args.qty_decoder_mode == "direct_raw_qty" else train_z
    global_var = train_magnitude.var(unbiased=False)
    global_std = global_var.sqrt().clamp_min(0.1)
    sigma_floor = float(args.magnitude_sigma_floor)
    if args.qty_decoder_mode == "direct_raw_qty":
        sigma_floor = max(0.001 * float(global_std.item()), 1e-4)
    marked_meta = {
        "num_marks": int(args.num_marks),
        "max_order": int(args.num_marks - 2),
        "series_count": int(args.batch_size),
        "magnitude_global_mean": float(train_magnitude.mean().item()),
        "magnitude_global_var": float(global_var.item()),
        "magnitude_global_std": float(global_std.item()),
        "magnitude_sigma_floor": sigma_floor,
    }

    model, rmtpp_cfg, encoder_cfg = build_model(
        cfg=cfg,
        run_cfg=run_cfg,
        marked_meta=marked_meta,
    )
    model.eval()

    h = forward_model(model, marks, dts, mask, values)
    out = model.nll(
        marks,
        dts,
        values=values,
        mask=mask,
        loss_scope=(
            "target_only"
            if args.qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}
            else "all"
        ),
    )
    assert_finite_tensor("hidden_state", h)
    for key in ("nll", "nll_marker", "nll_time", "value_loss"):
        assert_finite_tensor(key, out[key])
    for key in ("ordinal_marker_loss", "marker_train_loss"):
        if key in out:
            assert_finite_tensor(key, out[key])
    if args.qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}:
        assert_finite_tensor("magnitude_loss", out["magnitude_loss"])
        assert_finite_tensor("log_qty_aux_loss", out["log_qty_aux_loss"])

    h_prev = h[:, -2, :]
    pad_id = int(model.cfg.num_marks - 1)
    logits = model.mark_head(h_prev)[:, :pad_id]
    pred_mark = torch.argmax(logits, dim=-1)
    if args.qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}:
        direct = model.predict_direct_magnitude(
            h_prev,
            marks=marks,
            values=values,
            mask=mask,
        )
        value_hat = direct["denormalized"]
        qty_hat = direct["qty"]
    else:
        value_hat = predict_value_for_marks(model, h_prev, pred_mark)
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
        "magnitude_loss": float(out.get("magnitude_loss", torch.tensor(float("nan"))).item()),
        "log_qty_aux_loss": float(
            out.get("log_qty_aux_loss", torch.tensor(float("nan"))).item()
        ),
        "ordinal_marker_loss": (
            float(out["ordinal_marker_loss"].item())
            if "ordinal_marker_loss" in out
            else float("nan")
        ),
        "marker_train_loss": (
            float(out["marker_train_loss"].item())
            if "marker_train_loss" in out
            else float(out["nll_marker"].item())
        ),
        "value_head_mode": str(getattr(model.cfg, "value_head_mode", "shared")),
        "qty_mark_gradient_mode": str(
            getattr(model.cfg, "qty_mark_gradient_mode", "coupled")
        ),
        "value_encoder_gradient_mode": str(
            getattr(model.cfg, "value_encoder_gradient_mode", "coupled")
        ),
        "marker_loss_mode": str(getattr(model.cfg, "marker_loss_mode", "ce")),
        "lambda_ordinal": float(getattr(model.cfg, "lambda_ordinal", 0.0)),
        "qty_decoder_mode": str(getattr(model.cfg, "qty_decoder_mode", "mark_residual")),
        "magnitude_norm_mode": str(getattr(model.cfg, "magnitude_norm_mode", "global")),
        "lambda_magnitude": float(getattr(model.cfg, "lambda_magnitude", 1.0)),
        "magnitude_encoder_gradient_mode": str(
            getattr(model.cfg, "magnitude_encoder_gradient_mode", "coupled")
        ),
        "magnitude_aux_loss_mode": str(
            getattr(model.cfg, "magnitude_aux_loss_mode", "none")
        ),
        "lambda_log_qty": float(getattr(model.cfg, "lambda_log_qty", 0.25)),
        "log_qty_huber_delta": float(
            getattr(model.cfg, "log_qty_huber_delta", 1.0)
        ),
        "log_qty_floor": float(getattr(model.cfg, "log_qty_floor", 1.0)),
        "magnitude_revin_eps": float(getattr(model.cfg, "magnitude_revin_eps", 1e-5)),
        "magnitude_shrinkage_k": float(getattr(model.cfg, "magnitude_shrinkage_k", 8.0)),
        "magnitude_center_mode": str(getattr(model.cfg, "magnitude_center_mode", "mean")),
        "magnitude_revin_affine": bool(getattr(model.cfg, "magnitude_revin_affine", False)),
        "magnitude_stat_context_mode": str(
            getattr(model.cfg, "magnitude_stat_context_mode", "none")
        ),
        "value_by_mark_shape": (
            list(model.predict_value_by_mark(h_prev).shape)
            if args.qty_decoder_mode not in {"direct_log_qty", "direct_raw_qty"}
            and callable(getattr(model, "predict_value_by_mark", None))
            else None
        ),
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
