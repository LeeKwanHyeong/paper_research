"""
Unified CLI for TPP benchmark/search experiments.

This file is the new single entrypoint for experiment families that used to be
split across several scripts. The first fully migrated mode is `long-epoch`,
which covers:
1. RMTPP / TitanTPP / THP model comparison
2. Titan and THP candidate sweeps
3. best-validation-NLL checkpointing
4. scale-wise quantity error reporting

Auxiliary modes live under `simple_lab_test.search.common.modes`, so the
project exposes one user-facing experiment command while shared internals stay
easy to reuse.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_paper_research")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_paper_research")


def _configure_stdio_utf8() -> None:
    """
    Keep remote Linux logs readable even when the shell locale is not UTF-8.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_utf8()

THIS_FILE = Path(__file__).resolve()
BOOTSTRAP_ROOT = THIS_FILE.parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from simple_lab_test.common.pathing import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(THIS_FILE)

import torch

from simple_lab_test.search.common.configs import ExperimentConfig
from simple_lab_test.search.common.modes.model_test import run_model_test
from simple_lab_test.search.common.models import canonical_model_name
from simple_lab_test.search.common.runner import run_long_epoch_experiment


DEFAULT_BASE_DIR = PROJECT_ROOT / "search_artifacts" / "tpp_experiment_long_epoch"
ALLOWED_EVAL_SELECTIONS = {"best_val_nll", "best_score", "final"}
ALLOWED_MODELS = {"rmtpp", "titantpp", "thp"}
LEGACY_MODULE_BY_COMMAND = {
    "overfit": "simple_lab_test.search.common.modes.overfit",
    "qty-ablation": "simple_lab_test.search.common.modes.qty_loss_ablation",
}


def _csv_tuple(value: str, *, cast=str) -> tuple:
    """
    Parse comma-separated CLI values while ignoring accidental empty tokens.
    """
    if value is None:
        return tuple()
    return tuple(cast(token.strip()) for token in value.split(",") if token.strip())


def _parse_optional_positive_int(value: int | None) -> int | None:
    """
    Treat 0 as "disabled" for max-series and optional hidden-size arguments.
    """
    if value is None:
        return None
    return None if int(value) <= 0 else int(value)


def _validate_long_epoch_args(args: argparse.Namespace) -> None:
    """
    Fail fast for misspelled model names or checkpoint selection names.
    """
    models = {canonical_model_name(model) for model in _csv_tuple(args.models)}
    unsupported_models = sorted(models - ALLOWED_MODELS)
    if unsupported_models:
        raise ValueError(f"Unsupported models: {unsupported_models}. Allowed: {sorted(ALLOWED_MODELS)}")

    eval_selections = set(_csv_tuple(args.eval_selections))
    unsupported_selections = sorted(eval_selections - ALLOWED_EVAL_SELECTIONS)
    if unsupported_selections:
        raise ValueError(
            "Unsupported eval selections: "
            f"{unsupported_selections}. Allowed: {sorted(ALLOWED_EVAL_SELECTIONS)}"
        )
    if float(args.lambda_dt) < 0.0:
        raise ValueError("--lambda-dt must be non-negative.")
    if args.evaluation_scope == "validation_only" and args.split_mode != "fixed":
        raise ValueError("--evaluation-scope validation_only requires --split-mode fixed.")
    if int(args.value_input_emb_dim) <= 0:
        raise ValueError("--value-input-emb-dim must be positive.")
    lambda_ordinal = float(args.lambda_ordinal)
    if not math.isfinite(lambda_ordinal) or lambda_ordinal < 0.0:
        raise ValueError("--lambda-ordinal must be finite and non-negative.")
    if args.marker_loss_mode == "ce" and lambda_ordinal != 0.0:
        raise ValueError("--marker-loss-mode ce requires --lambda-ordinal 0.")
    if args.marker_loss_mode == "ce_rps":
        if models != {"titantpp"}:
            raise ValueError(
                "--marker-loss-mode ce_rps currently supports TitanTPP-only runs. "
                "Use --models titantpp."
            )
        if lambda_ordinal <= 0.0:
            raise ValueError("--marker-loss-mode ce_rps requires --lambda-ordinal > 0.")
    if args.time_head_mode == "mark_conditioned":
        if models != {"titantpp"}:
            raise ValueError(
                "--time-head-mode mark_conditioned currently supports TitanTPP-only runs."
            )
        if args.qty_decoder_mode != "mark_residual":
            raise ValueError(
                "--time-head-mode mark_conditioned requires --qty-decoder-mode mark_residual."
            )
        if args.marker_loss_mode != "ce" or lambda_ordinal != 0.0:
            raise ValueError(
                "--time-head-mode mark_conditioned requires plain marker CE."
            )
        supported_value_route = (
            args.value_head_mode == "shared"
            and args.qty_mark_gradient_mode == "coupled"
            and args.value_encoder_gradient_mode == "coupled"
        ) or (
            args.value_head_mode == "mark_conditioned_experts"
            and args.qty_mark_gradient_mode == "detached"
            and args.value_encoder_gradient_mode == "coupled"
        )
        if not supported_value_route:
            raise ValueError(
                "--time-head-mode mark_conditioned supports only V4a "
                "shared/coupled/coupled or V4b "
                "mark_conditioned_experts/detached/coupled."
            )
        if args.test_time_memory != "none":
            raise ValueError(
                "--time-head-mode mark_conditioned requires --test-time-memory none."
            )
    if args.qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}:
        decoder_mode = str(args.qty_decoder_mode)
        datasets = set(_csv_tuple(args.datasets))
        if models != {"titantpp"}:
            raise ValueError(f"--qty-decoder-mode {decoder_mode} supports TitanTPP-only runs.")
        unsupported_datasets = sorted(datasets - {"intermittent", "insta_market_basket"})
        if unsupported_datasets:
            raise ValueError(
                f"{decoder_mode} currently supports log2-factorized datasets only: "
                f"intermittent,insta_market_basket. Unsupported: {unsupported_datasets}"
            )
        if args.split_mode != "fixed":
            raise ValueError(f"{decoder_mode} requires --split-mode fixed.")
        if args.train_loss_scope != "target_only":
            raise ValueError(f"{decoder_mode} requires --train-loss-scope target_only.")
        if args.loss_mode != "hybrid":
            raise ValueError(f"{decoder_mode} requires --loss-mode hybrid.")
        if args.value_input_mode != "none":
            raise ValueError(f"{decoder_mode} requires --value-input-mode none.")
        if args.marker_loss_mode != "ce" or lambda_ordinal != 0.0:
            raise ValueError(f"{decoder_mode} requires plain marker CE.")
        if args.value_head_mode != "shared":
            raise ValueError(f"{decoder_mode} does not combine with mark-conditioned experts.")
        if args.qty_mark_gradient_mode != "coupled" or args.value_encoder_gradient_mode != "coupled":
            raise ValueError(f"{decoder_mode} does not combine with detached V3 routes.")
        if args.test_time_memory != "none":
            raise ValueError(f"{decoder_mode} does not support contextual test-time memory.")
        if decoder_mode == "direct_log_qty" and args.magnitude_norm_mode != "global":
            raise ValueError("The first direct_log_qty activation supports only global M0 normalization.")
        if decoder_mode == "direct_raw_qty":
            if args.magnitude_center_mode != "mean":
                raise ValueError("direct_raw_qty requires --magnitude-center-mode mean.")
            if args.magnitude_revin_affine:
                raise ValueError("direct_raw_qty requires --no-magnitude-revin-affine.")
            if args.magnitude_stat_context_mode != "none":
                raise ValueError("direct_raw_qty requires --magnitude-stat-context-mode none.")
    q3_active = (
        args.magnitude_encoder_gradient_mode != "coupled"
        or args.magnitude_aux_loss_mode != "none"
    )
    if q3_active:
        if models != {"titantpp"}:
            raise ValueError("Q3 magnitude modes require --models titantpp.")
        if args.qty_decoder_mode != "direct_raw_qty":
            raise ValueError("Q3 magnitude modes require --qty-decoder-mode direct_raw_qty.")
        if args.magnitude_norm_mode != "causal_shrinkage_revin":
            raise ValueError(
                "Q3 magnitude modes require --magnitude-norm-mode "
                "causal_shrinkage_revin."
            )
        if not (
            math.isclose(float(args.lambda_log_qty), 0.25, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(
                float(args.log_qty_huber_delta), 1.0, rel_tol=0.0, abs_tol=1e-12
            )
            and math.isclose(float(args.log_qty_floor), 1.0, rel_tol=0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "Q3 modes require --lambda-log-qty 0.25, "
                "--log-qty-huber-delta 1.0, and --log-qty-floor 1.0."
            )
    if int(args.magnitude_input_emb_dim) <= 0:
        raise ValueError("--magnitude-input-emb-dim must be positive.")
    for name in (
        "lambda_magnitude",
        "lambda_log_qty",
        "log_qty_huber_delta",
        "log_qty_floor",
        "magnitude_sigma_floor",
        "magnitude_revin_eps",
        "magnitude_shrinkage_k",
        "magnitude_exp_clamp_min",
        "magnitude_exp_clamp_max",
    ):
        if not math.isfinite(float(getattr(args, name))):
            raise ValueError(f"--{name.replace('_', '-')} must be finite.")
    if (
        float(args.lambda_magnitude) <= 0.0
        or float(args.lambda_log_qty) <= 0.0
        or float(args.log_qty_huber_delta) <= 0.0
        or float(args.log_qty_floor) <= 0.0
        or float(args.magnitude_sigma_floor) <= 0.0
        or float(args.magnitude_revin_eps) <= 0.0
        or float(args.magnitude_shrinkage_k) <= 0.0
    ):
        raise ValueError(
            "Magnitude lambdas, log-loss constants, scale constants, and shrinkage k "
            "must be positive."
        )
    if float(args.magnitude_exp_clamp_min) >= float(args.magnitude_exp_clamp_max):
        raise ValueError("Magnitude exp2 clamp min must be smaller than max.")
    if args.qty_mark_gradient_mode == "detached" and models != {"titantpp"}:
        raise ValueError(
            "--qty-mark-gradient-mode detached currently supports TitanTPP-only runs. "
            "Use --models titantpp."
        )
    if args.value_encoder_gradient_mode == "detached":
        if models != {"titantpp"}:
            raise ValueError(
                "--value-encoder-gradient-mode detached currently supports "
                "TitanTPP-only runs. Use --models titantpp."
            )
        if (
            args.value_head_mode != "mark_conditioned_experts"
            or args.qty_mark_gradient_mode != "detached"
        ):
            raise ValueError(
                "--value-encoder-gradient-mode detached requires "
                "--value-head-mode mark_conditioned_experts and "
                "--qty-mark-gradient-mode detached."
            )


def add_shared_long_epoch_args(parser: argparse.ArgumentParser) -> None:
    """
    Keep long-epoch options in one place so future subcommands can reuse them.
    """
    defaults = ExperimentConfig(base_dir=str(DEFAULT_BASE_DIR))
    parser.add_argument(
        "--base-dir",
        default=str(DEFAULT_BASE_DIR),
        help="Directory where experiment artifacts will be written.",
    )
    parser.add_argument(
        "--datasets",
        default="intermittent,yellow_trip_hourly",
        help=(
            "Comma-separated dataset names. Supported in long-epoch mode: "
            "intermittent,yellow_trip_hourly,insta_market_basket. "
            "Run simple_lab_test/notebooks/preprocessing/yellow_trip.ipynb before using yellow_trip_hourly."
        ),
    )
    parser.add_argument(
        "--models",
        default="rmtpp,titantpp",
        help="Comma-separated model names. Supported: rmtpp,titantpp,thp.",
    )
    parser.add_argument(
        "--titan-profile",
        default=defaults.titan_profile,
        choices=["dataset_best", "overall", "score_priority"],
        help="Report-derived scale-base/default-Titan profile.",
    )
    parser.add_argument(
        "--titan-candidates",
        default="",
        help="Optional comma-separated TitanCandidate names to sweep.",
    )
    parser.add_argument(
        "--thp-candidates",
        default="",
        help="Optional comma-separated THP preset names: small,base,deep,wide.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--reproducibility-mode",
        default=defaults.reproducibility_mode,
        choices=["standard", "strict"],
        help=(
            "standard preserves the legacy runtime. strict requires launcher-provided "
            "PYTHONHASHSEED, CUBLAS_WORKSPACE_CONFIG, and SOURCE_REVISION and enables "
            "deterministic Torch/CUDA and train-loader controls."
        ),
    )
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument(
        "--lambda-dt",
        type=float,
        default=defaults.lambda_dt,
        help="Weight for the continuous-time likelihood term in the training objective.",
    )
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seeds", default="42,52,62", help="Comma-separated random seeds.")
    parser.add_argument("--lookback-weeks", type=int, default=defaults.lookback_weeks)
    parser.add_argument("--max-seq-len", type=int, default=defaults.max_seq_len)
    parser.add_argument("--intermittent-max-series", type=int, default=None)
    parser.add_argument("--yellow-max-series", type=int, default=None)
    parser.add_argument(
        "--insta-max-series",
        type=int,
        default=None,
        help="Optional number of top Instacart user series to keep. Omit for all eligible users.",
    )
    parser.add_argument("--rmtpp-rnn-type", default=defaults.rmtpp_rnn_type, choices=["rnn", "gru", "lstm"])
    parser.add_argument("--rmtpp-mark-emb-dim", type=int, default=defaults.rmtpp_mark_emb_dim)
    parser.add_argument(
        "--rmtpp-hidden-dim",
        type=int,
        default=0,
        help="Optional fixed RMTPP decoder hidden size. Use 0 to mirror the active candidate d_model.",
    )
    parser.add_argument(
        "--split-mode",
        default=defaults.split_mode,
        choices=["internal", "fixed"],
        help=(
            "internal: rebuild the legacy train/validation split from one event table. "
            "fixed: consume *_with_split.parquet and keep test held out for final evaluation."
        ),
    )
    parser.add_argument(
        "--evaluation-scope",
        default=defaults.evaluation_scope,
        choices=["validation_and_test", "validation_only"],
        help=(
            "Whether a fixed-split run exports held-out test metrics after training. "
            "Use validation_only while a model-selection gate is still locked."
        ),
    )
    parser.add_argument(
        "--value-head-activation",
        default=defaults.value_head_activation,
        choices=["sigmoid", "identity"],
        help=(
            "Activation for residual/scale value head. Use identity when tail-order "
            "merging can produce scale_residual values above 1."
        ),
    )
    parser.add_argument(
        "--value-head-mode",
        default=defaults.value_head_mode,
        choices=["shared", "mark_conditioned_experts"],
        help=(
            "Value residual head architecture. 'shared' preserves V2; "
            "'mark_conditioned_experts' enables the V3 shared-plus-delta head."
        ),
    )
    parser.add_argument(
        "--time-head-mode",
        default=defaults.time_head_mode,
        choices=["shared", "mark_conditioned"],
        help=(
            "RMTPP time-head architecture. 'shared' preserves V2/V3; "
            "'mark_conditioned' enables the V4 shared-plus-delta intercept."
        ),
    )
    parser.add_argument(
        "--qty-mark-gradient-mode",
        default=defaults.qty_mark_gradient_mode,
        choices=["coupled", "detached"],
        help=(
            "Gradient policy for the expected-quantity mark gate. 'coupled' preserves "
            "V2/V3a; 'detached' enables V3b without changing forward values."
        ),
    )
    parser.add_argument(
        "--value-encoder-gradient-mode",
        default=defaults.value_encoder_gradient_mode,
        choices=["coupled", "detached"],
        help=(
            "Gradient policy at the value-to-encoder boundary. 'coupled' preserves "
            "V2/V3a/V3b; 'detached' enables V3c without changing forward values."
        ),
    )
    parser.add_argument(
        "--marker-loss-mode",
        default=defaults.marker_loss_mode,
        choices=["ce", "ce_rps"],
        help="Marker objective. 'ce_rps' keeps CE and adds normalized ordinal RPS.",
    )
    parser.add_argument(
        "--lambda-ordinal",
        type=float,
        default=defaults.lambda_ordinal,
        help="Weight of normalized RPS when --marker-loss-mode ce_rps is active.",
    )
    parser.add_argument(
        "--qty-decoder-mode",
        default=defaults.qty_decoder_mode,
        choices=["mark_residual", "direct_log_qty", "direct_raw_qty"],
        help="Exclusive quantity decoder for legacy, log-domain, or raw-domain quantity.",
    )
    parser.add_argument(
        "--magnitude-norm-mode",
        default=defaults.magnitude_norm_mode,
        choices=["global", "causal_revin", "causal_shrinkage_revin"],
        help="Stateless magnitude normalization used by the direct decoder.",
    )
    parser.add_argument(
        "--magnitude-input-emb-dim",
        type=int,
        default=defaults.magnitude_input_emb_dim,
    )
    parser.add_argument("--lambda-magnitude", type=float, default=defaults.lambda_magnitude)
    parser.add_argument(
        "--magnitude-encoder-gradient-mode",
        default=defaults.magnitude_encoder_gradient_mode,
        choices=["coupled", "detached"],
        help="Gradient route from direct magnitude losses into the Titan encoder.",
    )
    parser.add_argument(
        "--magnitude-aux-loss-mode",
        default=defaults.magnitude_aux_loss_mode,
        choices=["none", "log_huber"],
        help="Optional Q3 low-quantity auxiliary objective.",
    )
    parser.add_argument("--lambda-log-qty", type=float, default=defaults.lambda_log_qty)
    parser.add_argument(
        "--log-qty-huber-delta",
        type=float,
        default=defaults.log_qty_huber_delta,
    )
    parser.add_argument("--log-qty-floor", type=float, default=defaults.log_qty_floor)
    parser.add_argument(
        "--magnitude-sigma-floor",
        type=float,
        default=defaults.magnitude_sigma_floor,
    )
    parser.add_argument("--magnitude-revin-eps", type=float, default=defaults.magnitude_revin_eps)
    parser.add_argument(
        "--magnitude-shrinkage-k",
        type=float,
        default=defaults.magnitude_shrinkage_k,
    )
    parser.add_argument(
        "--magnitude-center-mode",
        default=defaults.magnitude_center_mode,
        choices=["mean"],
    )
    parser.add_argument(
        "--magnitude-revin-affine",
        action=argparse.BooleanOptionalAction,
        default=defaults.magnitude_revin_affine,
    )
    parser.add_argument(
        "--magnitude-stat-context-mode",
        default=defaults.magnitude_stat_context_mode,
        choices=["none"],
    )
    parser.add_argument(
        "--magnitude-exp-clamp-min",
        type=float,
        default=defaults.magnitude_exp_clamp_min,
    )
    parser.add_argument(
        "--magnitude-exp-clamp-max",
        type=float,
        default=defaults.magnitude_exp_clamp_max,
    )
    parser.add_argument(
        "--loss-mode",
        default=defaults.loss_mode,
        choices=["residual_only", "hybrid", "qty_only"],
        help="TPP quantity objective. Main paper runs should usually keep residual_only.",
    )
    parser.add_argument(
        "--value-input-mode",
        default=defaults.value_input_mode,
        choices=["none", "residual", "log_qty"],
        help=(
            "Optional value-conditioned marked TPP input. 'none' keeps the baseline; "
            "'residual' adds past scale_residual; 'log_qty' adds past log_base(qty)."
        ),
    )
    parser.add_argument(
        "--value-input-emb-dim",
        type=int,
        default=defaults.value_input_emb_dim,
        help="Projection dimension for optional value-conditioned input features.",
    )
    parser.add_argument(
        "--train-loss-scope",
        default=defaults.train_loss_scope,
        choices=["all", "target_only"],
        help=(
            "all: train on every transition in the window. target_only: train only "
            "on the final next-event transition, matching fixed validation/test."
        ),
    )
    parser.add_argument(
        "--test-time-memory",
        default=defaults.test_time_memory,
        choices=["none", "contextual"],
        help=(
            "Optional TitanTPP TTM-Lite evaluation. 'contextual' keeps a "
            "series-wise online contextual memory during validation/test metric export."
        ),
    )
    parser.add_argument("--analysis-scale-base", type=float, default=defaults.analysis_scale_base)
    parser.add_argument(
        "--analysis-tail-order",
        type=int,
        default=defaults.analysis_tail_order,
        help="Final scale bucket is >= analysis_scale_base ** analysis_tail_order.",
    )
    parser.add_argument(
        "--eval-selections",
        default="best_val_nll",
        help="Comma-separated checkpoints for scale-wise metrics: best_val_nll,best_score,final.",
    )
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")


def add_model_test_args(parser: argparse.ArgumentParser) -> None:
    """
    Add synthetic interface-test options for RMTPP/TitanTPP/THP models.
    """
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "search_artifacts" / "tpp_model_test"),
        help="Directory where model-test summary files will be written.",
    )
    parser.add_argument(
        "--models",
        default="thp",
        help="Comma-separated model names. Supported: rmtpp,titantpp,thp.",
    )
    parser.add_argument(
        "--titan-candidates",
        default="small_lmm",
        help="Comma-separated Titan candidates used when --models includes titantpp.",
    )
    parser.add_argument(
        "--thp-candidates",
        default="small",
        help="Comma-separated THP candidates. Supported defaults: small,base,deep,wide.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--num-marks", type=int, default=6)
    parser.add_argument("--scale-base", type=float, default=10.0)
    parser.add_argument(
        "--value-head-mode",
        default="shared",
        choices=["shared", "mark_conditioned_experts"],
    )
    parser.add_argument(
        "--time-head-mode",
        default="shared",
        choices=["shared", "mark_conditioned"],
    )
    parser.add_argument(
        "--qty-mark-gradient-mode",
        default="coupled",
        choices=["coupled", "detached"],
    )
    parser.add_argument(
        "--value-encoder-gradient-mode",
        default="coupled",
        choices=["coupled", "detached"],
    )
    parser.add_argument(
        "--marker-loss-mode",
        default="ce",
        choices=["ce", "ce_rps"],
    )
    parser.add_argument("--lambda-ordinal", type=float, default=0.0)
    parser.add_argument(
        "--qty-decoder-mode",
        default="mark_residual",
        choices=["mark_residual", "direct_log_qty", "direct_raw_qty"],
    )
    parser.add_argument(
        "--magnitude-norm-mode",
        default="global",
        choices=["global", "causal_revin", "causal_shrinkage_revin"],
    )
    parser.add_argument("--magnitude-input-emb-dim", type=int, default=8)
    parser.add_argument("--lambda-magnitude", type=float, default=1.0)
    parser.add_argument(
        "--magnitude-encoder-gradient-mode",
        default="coupled",
        choices=["coupled", "detached"],
    )
    parser.add_argument(
        "--magnitude-aux-loss-mode",
        default="none",
        choices=["none", "log_huber"],
    )
    parser.add_argument("--lambda-log-qty", type=float, default=0.25)
    parser.add_argument("--log-qty-huber-delta", type=float, default=1.0)
    parser.add_argument("--log-qty-floor", type=float, default=1.0)
    parser.add_argument("--magnitude-sigma-floor", type=float, default=0.0014535461338152059)
    parser.add_argument("--magnitude-revin-eps", type=float, default=1e-5)
    parser.add_argument("--magnitude-shrinkage-k", type=float, default=8.0)
    parser.add_argument("--magnitude-center-mode", default="mean", choices=["mean"])
    parser.add_argument(
        "--magnitude-revin-affine",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--magnitude-stat-context-mode", default="none", choices=["none"])
    parser.add_argument("--magnitude-exp-clamp-min", type=float, default=-2.0)
    parser.add_argument("--magnitude-exp-clamp-max", type=float, default=15.0)
    parser.add_argument("--rmtpp-rnn-type", default="gru", choices=["rnn", "gru", "lstm"])
    parser.add_argument("--rmtpp-hidden-dim", type=int, default=64)
    parser.add_argument("--rmtpp-mark-emb-dim", type=int, default=32)
    parser.add_argument(
        "--left-pad",
        action="store_true",
        help="Include left-padded rows in the synthetic batch to stress THP attention masks.",
    )
    parser.add_argument("--stop-on-error", action="store_true")


def build_long_epoch_config(args: argparse.Namespace) -> ExperimentConfig:
    """
    Convert validated CLI args into the dataclass used by shared runners.
    """
    _validate_long_epoch_args(args)
    return ExperimentConfig(
        base_dir=args.base_dir,
        experiment_mode="long-epoch",
        device=args.device,
        reproducibility_mode=args.reproducibility_mode,
        datasets=_csv_tuple(args.datasets),
        models=tuple(canonical_model_name(model) for model in _csv_tuple(args.models)),
        titan_profile=args.titan_profile,
        titan_candidates=_csv_tuple(args.titan_candidates),
        thp_candidates=_csv_tuple(args.thp_candidates),
        epochs=int(args.epochs),
        lr=float(args.lr),
        lambda_dt=float(args.lambda_dt),
        batch_size=int(args.batch_size),
        seeds=_csv_tuple(args.seeds, cast=int),
        lookback_weeks=int(args.lookback_weeks),
        max_seq_len=int(args.max_seq_len),
        intermittent_max_series=_parse_optional_positive_int(args.intermittent_max_series),
        yellow_max_series=_parse_optional_positive_int(args.yellow_max_series),
        insta_max_series=_parse_optional_positive_int(args.insta_max_series),
        split_mode=args.split_mode,
        evaluation_scope=args.evaluation_scope,
        rmtpp_rnn_type=args.rmtpp_rnn_type,
        rmtpp_mark_emb_dim=int(args.rmtpp_mark_emb_dim),
        rmtpp_hidden_dim=_parse_optional_positive_int(args.rmtpp_hidden_dim),
        value_head_activation=args.value_head_activation,
        value_head_mode=args.value_head_mode,
        time_head_mode=args.time_head_mode,
        qty_mark_gradient_mode=args.qty_mark_gradient_mode,
        value_encoder_gradient_mode=args.value_encoder_gradient_mode,
        marker_loss_mode=args.marker_loss_mode,
        lambda_ordinal=float(args.lambda_ordinal),
        qty_decoder_mode=args.qty_decoder_mode,
        magnitude_norm_mode=args.magnitude_norm_mode,
        magnitude_input_emb_dim=int(args.magnitude_input_emb_dim),
        lambda_magnitude=float(args.lambda_magnitude),
        magnitude_encoder_gradient_mode=args.magnitude_encoder_gradient_mode,
        magnitude_aux_loss_mode=args.magnitude_aux_loss_mode,
        lambda_log_qty=float(args.lambda_log_qty),
        log_qty_huber_delta=float(args.log_qty_huber_delta),
        log_qty_floor=float(args.log_qty_floor),
        magnitude_sigma_floor=float(args.magnitude_sigma_floor),
        magnitude_revin_eps=float(args.magnitude_revin_eps),
        magnitude_shrinkage_k=float(args.magnitude_shrinkage_k),
        magnitude_center_mode=args.magnitude_center_mode,
        magnitude_revin_affine=bool(args.magnitude_revin_affine),
        magnitude_stat_context_mode=args.magnitude_stat_context_mode,
        magnitude_exp_clamp_min=float(args.magnitude_exp_clamp_min),
        magnitude_exp_clamp_max=float(args.magnitude_exp_clamp_max),
        loss_mode=args.loss_mode,
        value_input_mode=args.value_input_mode,
        value_input_emb_dim=int(args.value_input_emb_dim),
        train_loss_scope=args.train_loss_scope,
        test_time_memory=args.test_time_memory,
        analysis_scale_base=float(args.analysis_scale_base),
        analysis_tail_order=int(args.analysis_tail_order),
        eval_selections=_csv_tuple(args.eval_selections),
        force_rerun=bool(args.force_rerun),
        stop_on_error=bool(args.stop_on_error),
    )


def build_parser() -> argparse.ArgumentParser:
    """
    Build the top-level experiment CLI.
    """
    parser = argparse.ArgumentParser(
        description="Unified TPP experiment runner for RMTPP, TitanTPP, and THP baselines."
    )
    subparsers = parser.add_subparsers(dest="command")

    long_parser = subparsers.add_parser(
        "long-epoch",
        help="Run long-epoch model comparison with best-NLL checkpoints and scale-wise errors.",
    )
    add_shared_long_epoch_args(long_parser)

    model_test_parser = subparsers.add_parser(
        "model-test",
        help="Run a fast synthetic interface test for RMTPP, TitanTPP, and TransformerHawkesTPP.",
    )
    add_model_test_args(model_test_parser)

    # Auxiliary experiment modes live under common/modes and are dispatched
    # through the unified CLI so users do not need separate root scripts.
    for command in ("overfit", "qty-ablation"):
        mode_parser = subparsers.add_parser(
            command,
            add_help=False,
            help=f"Run the {command} experiment mode.",
        )
        mode_parser.add_argument("mode_args", nargs=argparse.REMAINDER)

    return parser


def main() -> None:
    """
    Dispatch the requested experiment mode.
    """
    parser = build_parser()
    args, unknown_args = parser.parse_known_args()
    if args.command is None:
        parser.print_help()
        return

    if args.command == "long-epoch":
        if unknown_args:
            parser.error(f"unrecognized arguments for long-epoch: {' '.join(unknown_args)}")
        cfg = build_long_epoch_config(args)
        print(f"Unified TPP Experiment Configuration:: {cfg}")
        run_long_epoch_experiment(cfg)
        return

    if args.command == "model-test":
        if unknown_args:
            parser.error(f"unrecognized arguments for model-test: {' '.join(unknown_args)}")
        rows = run_model_test(args)
        for row in rows:
            if row.get("status") == "success":
                print(
                    "[model-test] "
                    f"{row['model_name']}:{row['candidate_name']} "
                    f"hidden={row['hidden_shape']} "
                    f"nll={row['nll']:.6f} "
                    f"qty_hat_mean={row['qty_hat_mean']:.6f}"
                )
            else:
                print(f"[model-test][FAILED] {row['model_name']}:{row['candidate_name']} {row.get('error')}")
        return

    if args.command in LEGACY_MODULE_BY_COMMAND:
        module_name = LEGACY_MODULE_BY_COMMAND[args.command]
        # Keep delegation explicit in logs so mode execution is easy to
        # distinguish when reviewing remote terminal output later.
        print(f"[tpp_experiment] Delegating '{args.command}' to {module_name}", flush=True)
        # `argparse.REMAINDER` and `parse_known_args` can reorder unknown
        # option/value pairs. The mode parsers expect original argv order, so
        # slice it directly from the selected subcommand.
        command_idx = sys.argv.index(args.command)
        passthrough_args = sys.argv[command_idx + 1:]
        module = importlib.import_module(module_name)
        old_argv = sys.argv[:]
        try:
            sys.argv = [module_name, *passthrough_args]
            module.main()
        finally:
            sys.argv = old_argv
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
