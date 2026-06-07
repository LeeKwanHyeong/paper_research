"""
Unified CLI for TPP benchmark/search experiments.

This file is the new single entrypoint for experiment families that used to be
split across several scripts. The first fully migrated mode is `long-epoch`,
which covers:
1. RMTPP / TitanTPP / THP model comparison
2. Titan and THP candidate sweeps
3. best-validation-NLL checkpointing
4. scale-wise quantity error reporting

The legacy mode implementations now live under `simple_lab_test.search.common`
so the project exposes one user-facing experiment command while we continue to
deduplicate internals safely.
"""

from __future__ import annotations

import argparse
import importlib
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
from simple_lab_test.search.common.models import canonical_model_name
from simple_lab_test.search.common.runner import run_long_epoch_experiment


DEFAULT_BASE_DIR = PROJECT_ROOT / "search_artifacts" / "tpp_experiment_long_epoch"
ALLOWED_EVAL_SELECTIONS = {"best_val_nll", "best_score", "final"}
ALLOWED_MODELS = {"rmtpp", "titantpp", "thp"}
LEGACY_MODULE_BY_COMMAND = {
    "overfit": "simple_lab_test.search.common.modes.overfit",
    "qty-ablation": "simple_lab_test.search.common.modes.qty_loss_ablation",
    "yellow-resolution": "simple_lab_test.search.common.modes.yellow_trip_resolution",
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
        default="intermittent,yellow_trip",
        help="Comma-separated dataset names. Currently: intermittent,yellow_trip.",
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
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--seeds", default="42,52,62", help="Comma-separated random seeds.")
    parser.add_argument("--lookback-weeks", type=int, default=defaults.lookback_weeks)
    parser.add_argument("--max-seq-len", type=int, default=defaults.max_seq_len)
    parser.add_argument("--intermittent-max-series", type=int, default=None)
    parser.add_argument("--yellow-max-series", type=int, default=None)
    parser.add_argument("--rmtpp-rnn-type", default=defaults.rmtpp_rnn_type, choices=["rnn", "gru", "lstm"])
    parser.add_argument("--rmtpp-mark-emb-dim", type=int, default=defaults.rmtpp_mark_emb_dim)
    parser.add_argument(
        "--rmtpp-hidden-dim",
        type=int,
        default=0,
        help="Optional fixed RMTPP decoder hidden size. Use 0 to mirror the active candidate d_model.",
    )
    parser.add_argument(
        "--loss-mode",
        default=defaults.loss_mode,
        choices=["residual_only", "hybrid", "qty_only"],
        help="TPP quantity objective. Main paper runs should usually keep residual_only.",
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


def build_long_epoch_config(args: argparse.Namespace) -> ExperimentConfig:
    """
    Convert validated CLI args into the dataclass used by shared runners.
    """
    _validate_long_epoch_args(args)
    return ExperimentConfig(
        base_dir=args.base_dir,
        experiment_mode="long-epoch",
        device=args.device,
        datasets=_csv_tuple(args.datasets),
        models=tuple(canonical_model_name(model) for model in _csv_tuple(args.models)),
        titan_profile=args.titan_profile,
        titan_candidates=_csv_tuple(args.titan_candidates),
        thp_candidates=_csv_tuple(args.thp_candidates),
        epochs=int(args.epochs),
        lr=float(args.lr),
        batch_size=int(args.batch_size),
        seeds=_csv_tuple(args.seeds, cast=int),
        lookback_weeks=int(args.lookback_weeks),
        max_seq_len=int(args.max_seq_len),
        intermittent_max_series=_parse_optional_positive_int(args.intermittent_max_series),
        yellow_max_series=_parse_optional_positive_int(args.yellow_max_series),
        rmtpp_rnn_type=args.rmtpp_rnn_type,
        rmtpp_mark_emb_dim=int(args.rmtpp_mark_emb_dim),
        rmtpp_hidden_dim=_parse_optional_positive_int(args.rmtpp_hidden_dim),
        loss_mode=args.loss_mode,
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

    # Temporary migration bridge: these commands now live under common/modes
    # while we move their internals into shared utilities one mode at a time.
    for command in ("overfit", "qty-ablation", "yellow-resolution"):
        legacy_parser = subparsers.add_parser(
            command,
            add_help=False,
            help=f"Temporary passthrough to the legacy {command} runner.",
        )
        legacy_parser.add_argument("legacy_args", nargs=argparse.REMAINDER)

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

    if args.command in LEGACY_MODULE_BY_COMMAND:
        module_name = LEGACY_MODULE_BY_COMMAND[args.command]
        # Keep passthrough explicit in logs so unified-vs-mode execution is easy
        # to distinguish when reviewing remote terminal output later.
        print(f"[tpp_experiment] Delegating '{args.command}' to {module_name}", flush=True)
        # `argparse.REMAINDER` and `parse_known_args` can reorder unknown
        # option/value pairs. The legacy mode parsers expect original argv
        # order, so slice it directly from the selected subcommand.
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
