"""Re-evaluate one TitanTPP checkpoint on validation targets only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_loader.event_seq_data_module import (  # noqa: E402
    RMTPPWeekLookbackDataset,
    collate_week_lookback,
)
from models.RMTPPs.TitanTPP import TitanTPP  # noqa: E402
from models.RMTPPs.config import RMTPPConfig  # noqa: E402
from models.Titan import TitanConfig  # noqa: E402
from simple_lab_test.search.common.runner import evaluate_scale_wise_qty  # noqa: E402
from utils.training import TrainingConfig, eval_next_event_week_lookback  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved TitanTPP checkpoint and export validation-only metrics "
            "without constructing or reading held-out test samples."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--marked-parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--analysis-scale-base", type=float, default=10.0)
    parser.add_argument("--analysis-tail-order", type=int, default=4)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate checkpoint/config compatibility without reading the dataset.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataclass_kwargs(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in payload.items() if key in allowed}


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False),
        encoding="utf-8",
    )


def json_safe_metrics(
    metrics: dict[str, Any],
    undefined_metrics: set[str],
) -> dict[str, Any]:
    """Represent decoder-inapplicable metrics as JSON null, not non-standard NaN."""
    safe_metrics = dict(metrics)
    for name in undefined_metrics:
        value = safe_metrics.get(name)
        if value is not None and not math.isfinite(float(value)):
            safe_metrics[name] = None
    return safe_metrics


def undefined_validation_metrics(qty_decoder_mode: str) -> set[str]:
    """Return metrics that are structurally inapplicable to one decoder family."""
    if qty_decoder_mode == "mark_residual":
        return {"val_magnitude_loss", "val_log_qty_aux_loss"}
    if qty_decoder_mode in {"direct_log_qty", "direct_raw_qty"}:
        return {"value_mae", "val_value_loss"}
    raise ValueError(f"Unsupported quantity decoder mode: {qty_decoder_mode}")


def assert_finite_applicable_metrics(
    metrics: dict[str, Any],
    undefined_metrics: set[str],
) -> None:
    """Reject non-finite active metrics while allowing explicit N/A fields."""
    for name, value in metrics.items():
        if name.startswith("_") or name in undefined_metrics:
            continue
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"Validation metric is not finite: {name}={value}")


def validation_loader(
    marked_df: pl.DataFrame,
    training_cfg: TrainingConfig,
) -> DataLoader:
    dataset = RMTPPWeekLookbackDataset(
        marked_df,
        lookback_weeks=training_cfg.lookback,
        max_seq_len=training_cfg.max_seq_len,
        val_ratio=training_cfg.val_ratio,
        mode="all",
        split_col="chronological_split",
        target_splits={"validation"},
    )
    return DataLoader(
        dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_week_lookback,
    )


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    marked_path = Path(args.marked_parquet).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    payload = load_checkpoint(checkpoint_path)
    rmtpp_payload = dict(payload["rmtpp_config"])
    rmtpp_payload.setdefault("value_head_mode", "shared")
    rmtpp_payload.setdefault("qty_mark_gradient_mode", "coupled")
    rmtpp_payload.setdefault("value_encoder_gradient_mode", "coupled")
    rmtpp_payload.setdefault("marker_loss_mode", "ce")
    rmtpp_payload.setdefault("lambda_ordinal", 0.0)
    rmtpp_cfg = RMTPPConfig(**dataclass_kwargs(RMTPPConfig, rmtpp_payload))
    titan_cfg = TitanConfig(**dataclass_kwargs(TitanConfig, payload["encoder_config"]))

    if (
        rmtpp_cfg.value_head_mode != "shared"
        or rmtpp_cfg.qty_mark_gradient_mode != "coupled"
        or rmtpp_cfg.value_encoder_gradient_mode != "coupled"
        or rmtpp_cfg.marker_loss_mode != "ce"
        or float(rmtpp_cfg.lambda_ordinal) != 0.0
    ):
        raise ValueError(
            "Validation reference must be a V2 shared/coupled/coupled CE checkpoint."
        )

    device = str(args.device)
    model = TitanTPP(rmtpp_cfg, titan_cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    training_payload = dataclass_kwargs(TrainingConfig, payload["training_config"])
    training_payload["device"] = device
    training_cfg = TrainingConfig(**training_payload)
    manifest = {
        "status": "dry_run" if args.dry_run else "completed",
        "evaluation_split": "validation",
        "held_out_test_read": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "checkpoint_selection": payload.get("selection"),
        "checkpoint_epoch": payload.get("summary", {}).get("best_val_nll_epoch"),
        "marked_parquet": str(marked_path),
        "device": device,
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "rmtpp_config": rmtpp_payload,
        "encoder_config": payload["encoder_config"],
        "training_config": training_payload,
        "analysis_scale_base": float(args.analysis_scale_base),
        "analysis_tail_order": int(args.analysis_tail_order),
    }

    if args.dry_run:
        write_json(manifest, output_dir / "validation_reference_manifest.json")
        print(
            "[validation-reference][dry-run] "
            f"checkpoint={checkpoint_path} device={device} state_dict=strict_ok"
        )
        return

    if not marked_path.exists():
        raise FileNotFoundError(f"Marked validation source not found: {marked_path}")
    marked_df = pl.read_parquet(marked_path)
    loader = validation_loader(marked_df, training_cfg)
    metrics = eval_next_event_week_lookback(
        model,
        loader,
        device,
        target_only_nll=True,
    )
    scale_df = evaluate_scale_wise_qty(
        model=model,
        val_loader=loader,
        device=device,
        analysis_scale_base=float(args.analysis_scale_base),
        analysis_tail_order=int(args.analysis_tail_order),
    )
    metrics["score"] = float(
        metrics["mark_acc"]
        - 0.01 * metrics["dt_mae"]
        - 0.001 * metrics["qty_mae"]
    )

    qty_decoder_mode = str(getattr(rmtpp_cfg, "qty_decoder_mode", "mark_residual"))
    undefined_metrics = undefined_validation_metrics(qty_decoder_mode)
    assert_finite_applicable_metrics(metrics, undefined_metrics)
    exported_metrics = json_safe_metrics(metrics, undefined_metrics)
    populated_scale_df = scale_df.filter(pl.col("count") > 0)
    for row in populated_scale_df.iter_rows(named=True):
        for name in ("qty_mae", "qty_rmse", "qty_wape", "log_abs_error"):
            if not math.isfinite(float(row[name])):
                raise FloatingPointError(
                    f"Validation scale-wise metric is not finite: "
                    f"bucket={row['scale_label']} {name}={row[name]}"
                )

    manifest["marked_rows"] = int(marked_df.height)
    manifest["validation_samples"] = int(len(loader.dataset))
    write_json(manifest, output_dir / "validation_reference_manifest.json")

    result = {
        "source_checkpoint": str(checkpoint_path),
        "source_selection": payload.get("selection"),
        "source_epoch": payload.get("summary", {}).get("best_val_nll_epoch"),
        "evaluation_split": "validation",
        "held_out_test_read": False,
        "metrics": exported_metrics,
    }
    write_json(result, output_dir / "validation_reference.json")
    scale_df.write_csv(output_dir / "validation_scale_wise.csv")
    scale_df.write_parquet(output_dir / "validation_scale_wise.parquet")
    csv_row = {
        "source_checkpoint": str(checkpoint_path),
        "source_selection": payload.get("selection"),
        "source_epoch": payload.get("summary", {}).get("best_val_nll_epoch"),
        "evaluation_split": "validation",
        "held_out_test_read": False,
        **exported_metrics,
    }
    with (output_dir / "validation_reference.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_row))
        writer.writeheader()
        writer.writerow(csv_row)

    print(
        "[validation-reference] "
        f"samples={len(loader.dataset)} "
        f"nll={metrics['val_nll']:.6f} "
        f"marker_nll={metrics['val_nll_marker']:.6f} "
        f"log_qty_mae={metrics['log_qty_mae']:.6f} "
        f"rps={metrics['val_ordinal_marker_loss']:.6f} "
        f"mark_acc={metrics['mark_acc']:.6f} "
        f"mark_mae={metrics['mark_mae']:.6f} "
        f"context_n_le_4={int(metrics['context_1_count'] + metrics['context_2_4_count'])}"
    )


if __name__ == "__main__":
    main()
