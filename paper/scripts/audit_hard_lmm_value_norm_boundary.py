"""Audit whether the VNC-Hard-LMM contrast is non-degenerate in frozen B banks.

This script reads checkpoint state only. It never constructs a dataset loader,
fits a parameter, evaluates a prediction, or accesses held-out targets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


DEFAULT_OUTPUT = (
    ROOT / "paper/results/hard_lmm_value_norm_boundary_20260909/analysis.json"
)
CHECKPOINT_ROOT = ROOT / (
    "search_artifacts/"
    "hard_lmm_raw_rmse_checkpoint_alignment_seed42_5090_20260906_f752434/"
    "seed42_e300"
)
RUN_SUFFIX = Path(
    "quantile_checkpoint_alignment/runs/titantpp/"
    "count_only_log_regression/seed_42/best_val_qty_rmse_model.pt"
)
BASELINES = {
    "intermittent_frozen_5000": {
        "checkpoint_file_sha256": (
            "63d38ee4401c208dcfa24206682edaca56e2d2a5ca1037b09decdc2ffa567c08"
        ),
        "checkpoint_state_sha256": (
            "4b984d415d83a479fdacb49b12d964afa668adad136511590a54293df763c0d0"
        ),
    },
    "yellow_trip_hourly": {
        "checkpoint_file_sha256": (
            "46b456816dbc3c36e4f9c115c05b480fc030a8c2004646d179416d528bd897d5"
        ),
        "checkpoint_state_sha256": (
            "eddc8b4a11b233606bf74508f0ef721e05e55bbcb047d18edab4aa61f2f43c41"
        ),
    },
    "insta_market_basket": {
        "checkpoint_file_sha256": (
            "e594f8df0dcd66249c3724063f44eee3de213bc83a780ca81fdf39a094bfed07"
        ),
        "checkpoint_state_sha256": (
            "a74e1f3055b03db5ac85adf2fdb63711890ea9d06449965e3ca749aefccc2f02"
        ),
    },
}
EPSILON = 1e-8
COMBINATION_SIZE = 4
CHUNK_SIZE = 8192


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    _require(values.ndim == 1 and values.size > 0, "quantile input is empty")
    _require(bool(np.isfinite(values).all()), "quantile input is non-finite")
    return {
        "min": float(values.min()),
        "p01": float(np.quantile(values, 0.01)),
        "p10": float(np.quantile(values, 0.10)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
    }


def _all_quartet_geometry(memory: np.ndarray) -> dict[str, Any]:
    """Compare raw and equalized means for every four-row bank subset."""
    _require(memory.shape == (64, 64), "expected a [64,64] Hard-LMM bank")
    row_norms = np.linalg.norm(memory, axis=1)
    _require(bool(np.isfinite(row_norms).all()), "bank row norms are non-finite")
    mean_norm = float(row_norms.mean())
    equalized = memory * (
        mean_norm / np.maximum(row_norms, EPSILON)
    )[:, None]
    combinations = np.asarray(
        list(itertools.combinations(range(memory.shape[0]), COMBINATION_SIZE)),
        dtype=np.int16,
    )

    cosines: list[np.ndarray] = []
    relative_differences: list[np.ndarray] = []
    orthogonal_fractions: list[np.ndarray] = []
    for start in range(0, combinations.shape[0], CHUNK_SIZE):
        index = combinations[start : start + CHUNK_SIZE]
        raw = memory[index].mean(axis=1)
        normalized = equalized[index].mean(axis=1)
        raw_norm = np.linalg.norm(raw, axis=1)
        normalized_norm = np.linalg.norm(normalized, axis=1)
        denominator = np.maximum(raw_norm * normalized_norm, EPSILON)
        cosine = np.sum(raw * normalized, axis=1) / denominator
        delta = np.linalg.norm(normalized - raw, axis=1) / np.maximum(
            raw_norm, EPSILON
        )
        projection_scale = np.sum(raw * normalized, axis=1) / np.maximum(
            raw_norm**2, EPSILON
        )
        orthogonal = np.linalg.norm(
            normalized - projection_scale[:, None] * raw, axis=1
        ) / np.maximum(normalized_norm, EPSILON)
        cosines.append(np.clip(cosine, -1.0, 1.0))
        relative_differences.append(delta)
        orthogonal_fractions.append(orthogonal)

    cosine = np.concatenate(cosines)
    relative_difference = np.concatenate(relative_differences)
    orthogonal_fraction = np.concatenate(orthogonal_fractions)
    non_collinear = orthogonal_fraction > 1e-10
    return {
        "combination_count": int(combinations.shape[0]),
        "combination_size": COMBINATION_SIZE,
        "raw_vs_equalized_cosine": _quantiles(cosine),
        "relative_l2_difference": _quantiles(relative_difference),
        "orthogonal_fraction_after_best_scalar_fit": _quantiles(
            orthogonal_fraction
        ),
        "non_collinear_tolerance": 1e-10,
        "non_collinear_count": int(non_collinear.sum()),
        "non_collinear_fraction": float(non_collinear.mean()),
    }


def _audit_checkpoint(dataset: str, expected: dict[str, str]) -> dict[str, Any]:
    path = CHECKPOINT_ROOT / dataset / RUN_SUFFIX
    _require(path.is_file(), f"missing B checkpoint: {path}")
    file_sha = _sha256(path)
    _require(
        file_sha == expected["checkpoint_file_sha256"],
        f"B checkpoint file digest drift: {dataset}",
    )
    payload = torch_load_checkpoint(path, map_location="cpu")
    _require(isinstance(payload, dict), f"invalid B checkpoint payload: {dataset}")
    _require(payload.get("backbone") == "titantpp", f"B backbone drift: {dataset}")
    _require(payload.get("seed") == 42, f"B seed drift: {dataset}")
    _require(
        payload.get("variant") == "count_only_log_regression",
        f"B objective drift: {dataset}",
    )
    _require(
        payload.get("checkpoint_monitor") == "validation_raw_quantity_rmse",
        f"B selector drift: {dataset}",
    )
    _require(
        payload.get("evaluation_scope") == "validation_only"
        and payload.get("held_out_test_evaluated") is False,
        f"B evaluation scope drift: {dataset}",
    )
    state = payload.get("model_state_dict")
    _require(isinstance(state, dict), f"missing B model state: {dataset}")
    state_sha = canonical_state_dict_sha256(state)
    _require(
        state_sha == payload.get("model_state_sha256")
        == expected["checkpoint_state_sha256"],
        f"B checkpoint state digest drift: {dataset}",
    )
    memory = state.get("lmm.mem")
    _require(
        isinstance(memory, torch.Tensor)
        and tuple(memory.shape) == (1, 64, 64)
        and bool(torch.isfinite(memory).all()),
        f"invalid B prototype bank: {dataset}",
    )
    array = memory.detach().cpu().to(torch.float64).numpy()[0]
    row_norms = np.linalg.norm(array, axis=1)
    mean = float(row_norms.mean())
    return {
        "dataset": dataset,
        "checkpoint_path": str(path.relative_to(ROOT)),
        "checkpoint_file_sha256": file_sha,
        "checkpoint_state_sha256": state_sha,
        "best_epoch": int(payload["best_epoch"]),
        "row_norm": {
            **_quantiles(row_norms),
            "mean": mean,
            "standard_deviation": float(row_norms.std()),
            "coefficient_of_variation": float(row_norms.std() / mean),
            "zero_row_count": int((row_norms == 0).sum()),
        },
        "all_top4_subset_geometry": _all_quartet_geometry(array),
    }


def build_analysis() -> dict[str, Any]:
    datasets = [
        _audit_checkpoint(dataset, expected)
        for dataset, expected in BASELINES.items()
    ]
    all_non_degenerate = all(
        row["all_top4_subset_geometry"]["non_collinear_count"]
        == row["all_top4_subset_geometry"]["combination_count"]
        for row in datasets
    )
    return {
        "schema_version": 1,
        "analysis_id": "hard_lmm_value_norm_boundary_checkpoint_audit_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "input": "three pinned seed42 validation-raw-RMSE-selected B checkpoint states",
            "checkpoint_state_only": True,
            "train_rows_loaded": False,
            "validation_rows_loaded": False,
            "held_out_rows_loaded": False,
            "predictions_generated": False,
            "parameter_fit": False,
            "performance_selection": False,
        },
        "candidate": {
            "contract_id": "hard_lmm_value_norm_consistent_v1",
            "epsilon": EPSILON,
            "formula": (
                "r_base+tanh(alpha_raw)*(mean(bank_mean_norm*m_j/"
                "max(||m_j||,epsilon))-r_base)"
            ),
            "search_indices_changed_for_fixed_state": False,
            "search_similarities_changed_for_fixed_state": False,
            "relation_to_weighted_static": (
                "same prototype-aggregation boundary, but inverse-row-norm value "
                "coefficients rather than cosine-score softmax coefficients"
            ),
        },
        "datasets": datasets,
        "decision": {
            "all_real_bank_top4_subsets_non_collinear": all_non_degenerate,
            "candidate_is_scalar_post_retrieval_shrinkage": False,
            "candidate_is_mathematically_non_degenerate": all_non_degenerate,
            "implementation_gate": "pass" if all_non_degenerate else "fail",
            "performance_claim_supported": False,
            "reason": (
                "The contrast changes value direction for every four-row subset in "
                "the three pinned banks, but no event targets or predictions were used."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    analysis = build_analysis()
    _require(
        analysis["decision"]["implementation_gate"] == "pass",
        "value-norm contrast is degenerate",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(output),
        "datasets": len(analysis["datasets"]),
        "status": analysis["decision"]["implementation_gate"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
