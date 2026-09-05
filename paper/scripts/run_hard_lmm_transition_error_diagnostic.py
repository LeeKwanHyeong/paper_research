#!/usr/bin/env python3
"""Run the frozen, train-only Hard-LMM transition-error signal diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from data_loader.event_seq_data_module import (
    RMTPPWeekLookbackDataset,
    collate_week_lookback,
)
from paper.scripts.analyze_count_aware_b0_retrieval import (
    checkpoint_path,
    restore_b0,
)
from paper.scripts.count_aware_tpp_backbone.core import (
    prepare_count_frame,
    right_pad_batch,
    target_outputs,
)
from paper.scripts.hard_lmm_frozen_probe import sample_indices
from paper.scripts.hard_lmm_transition_error_features import (
    transition_error_features,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


CONTRACT = ROOT / "paper/contracts/hard_lmm_transition_error_v1.json"
RAW = ROOT / "search_artifacts/hard_lmm_transition_error_probe_20260906"
RESULT = ROOT / "paper/results/hard_lmm_transition_error_probe_20260906"
DATASETS = (
    "intermittent_v2",
    "yellow_trip_hourly",
    "insta_market_basket",
)
BATCH_SIZE = 64
TORCH_THREADS = 4

REQUIRED_CACHE_FIELDS = (
    "quantity",
    "base_pre_activation",
    "history_length",
    "transition_count",
    "selected_occupied_fraction",
    "selected_count_mean",
    "selected_count_std",
    "last_error",
    "unconditioned_mean_error",
    "prototype_conditioned_error",
    "sham_prototype_conditioned_error",
    "fold",
    "series_id",
)
FEATURE_INVARIANCE_FIELDS = (
    "base_pre_activation",
    "history_length",
    "transition_count",
    "selected_occupied_fraction",
    "selected_count_mean",
    "selected_count_std",
    "last_error",
    "unconditioned_mean_error",
    "prototype_conditioned_error",
    "sham_prototype_conditioned_error",
    "final_h",
    "final_top4_indices",
    "final_top4_similarity",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically save strict JSON so interrupted stages retain prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fold_for_series(series_index: int | torch.Tensor) -> int:
    series = int(series_index)
    payload = f"20260906:{series}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2


def load_train_frame(path: Path) -> pl.DataFrame:
    """Filter at lazy-scan level, before any validation or test row is collected."""
    frame = (
        pl.scan_parquet(path)
        .filter(pl.col("chronological_split") == "train")
        .collect()
        .sort(["oper_part_no", "seq"])
    )
    require(frame.height > 0, "Train population is empty")
    require(
        set(frame["chronological_split"].unique().to_list()) == {"train"},
        "A non-train row was materialized",
    )
    return frame


def source_hashes() -> dict[str, str]:
    tracked = subprocess.check_output(
        ["git", "ls-files", "models", "data_loader"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    explicit = [
        "paper/contracts/hard_lmm_transition_error_v1.json",
        "paper/scripts/analyze_count_aware_b0_retrieval.py",
        "paper/scripts/count_aware_tpp_backbone/core.py",
        "paper/scripts/hard_lmm_frozen_probe.py",
        "paper/scripts/hard_lmm_transition_error_analysis.py",
        "paper/scripts/hard_lmm_transition_error_features.py",
        "paper/scripts/run_hard_lmm_transition_error_diagnostic.py",
        "simple_lab_test/search/common/runner.py",
    ]
    paths = sorted(set(tracked + explicit))
    missing = [name for name in paths if not (ROOT / name).is_file()]
    require(not missing, f"Required source files are missing: {missing}")
    return {name: sha256_file(ROOT / name) for name in paths}


def verify_hashes(hashes: Mapping[str, str], *, label: str) -> None:
    for name, expected in hashes.items():
        path = ROOT / name
        require(path.is_file(), f"Missing {label}: {name}")
        require(
            sha256_file(path) == expected,
            f"Changed {label}: {name}",
        )


def load_contract() -> dict[str, Any]:
    contract = read_json(CONTRACT)
    require(
        contract.get("contract_id") == "hard_lmm_transition_error_v1",
        "Unexpected transition-error contract",
    )
    reference = contract["reference"]
    registry_path = ROOT / reference["registry"]
    require(registry_path.is_file(), "Frozen T0 registry is missing")
    require(
        sha256_file(registry_path) == reference["registry_sha256"],
        "Frozen T0 registry hash changed",
    )
    require(
        tuple(row["dataset"] for row in contract["datasets"]) == DATASETS,
        "Dataset order or scope changed",
    )
    scope = contract["diagnostic_scope"]
    require(scope["allowed_split"] == "train_only", "Diagnostic is not train-only")
    require(scope["validation_rows_materialized"] is False, "Validation access enabled")
    require(scope["held_out_test_evaluated"] is False, "Held-out access enabled")
    require(scope["maximum_targets_per_dataset"] == 8192, "Sample cap changed")
    require(scope["sample_seed"] == 42, "Sample seed changed")
    require(scope["minimum_targets_per_fold"] == 512, "Fold target minimum changed")
    require(scope["minimum_unique_series_per_fold"] == 30, "Fold series minimum changed")
    require(
        scope["series_fold_rule"]
        == "sha256('20260906:'+series_index)_first8_big_endian_mod_2",
        "Fold assignment changed",
    )
    probe = contract["probe"]
    require(probe["l2_penalty"] == 1.0, "Probe penalty changed")
    require(probe["hyperparameter_search"] is False, "Hyperparameter search enabled")
    gate = contract["signal_gate"]
    require(
        gate["paired_series_bootstrap"]["repeats"] == 10000,
        "Bootstrap count changed",
    )
    require(
        gate["minimum_pooled_relative_improvement_each_comparison"] == 0.01,
        "Signal threshold changed",
    )
    require(
        gate["failure_action"] == "do_not_implement_or_train_the_candidate",
        "Failure action changed",
    )
    require(contract["authorization"]["held_out_test"] is False, "Held-out authorized")
    return contract


def _registry_row(registry: Mapping[str, Any], dataset: str) -> dict[str, Any]:
    rows = [row for row in registry["datasets"] if row["dataset"] == dataset]
    require(len(rows) == 1, f"Registry row is missing or duplicated: {dataset}")
    return rows[0]


def resolve_frozen_checkpoint(
    dataset_contract: Mapping[str, Any],
    registry_row: Mapping[str, Any],
) -> tuple[Path, Path, Path, dict[str, str], bool]:
    """Resolve the exact canonical state, including the audited local override."""
    override = dataset_contract.get("local_checkpoint_override")
    if override is None:
        artifact = ROOT / registry_row["artifact_dir"]
        checkpoint = checkpoint_path(artifact, 42)
        launch = artifact / "launch_contract.json"
        summary = checkpoint.with_name("summary.json")
        expected = {
            str(launch.relative_to(ROOT)): registry_row["contract_sha256"],
            str(checkpoint.relative_to(ROOT)): registry_row["checkpoint_file_sha256"],
            str(summary.relative_to(ROOT)): registry_row["summary_sha256"],
        }
        return checkpoint, launch, summary, expected, False

    require(
        dataset_contract["dataset"] == "intermittent_v2",
        "A local checkpoint override is allowed only for Intermittent",
    )
    require(
        override["checkpoint_state_sha256"]
        == registry_row["checkpoint_state_sha256"],
        "Local override state does not equal the registry-pinned T0 state",
    )
    artifact = ROOT / override["artifact_dir"]
    checkpoint = checkpoint_path(artifact, 42)
    launch = artifact / "launch_contract.json"
    summary = checkpoint.with_name("summary.json")
    expected = {
        str(launch.relative_to(ROOT)): override["launch_contract_sha256"],
        str(checkpoint.relative_to(ROOT)): override["checkpoint_file_sha256"],
        str(summary.relative_to(ROOT)): override["summary_sha256"],
    }
    return checkpoint, launch, summary, expected, True


def cyclic_transition_sham_indices(
    top4_indices: torch.Tensor,
    observed_mask: torch.Tensor,
    memory_write_mask: torch.Tensor,
) -> torch.Tensor:
    """Rotate only legal transition-source assignments within each row."""
    require(top4_indices.ndim == 3 and top4_indices.size(-1) == 4, "Bad top-4 shape")
    require(observed_mask.shape == top4_indices.shape[:2], "Bad observed mask shape")
    require(memory_write_mask.shape == observed_mask.shape, "Bad write mask shape")
    sham = top4_indices.clone()
    valid_transition = torch.zeros_like(observed_mask)
    valid_transition[:, 1:] = (
        observed_mask[:, :-1]
        & observed_mask[:, 1:]
        & memory_write_mask[:, :-1]
        & memory_write_mask[:, 1:]
    )
    for row in range(top4_indices.size(0)):
        # A transition ending at t uses the source assignment at t-1.
        source_positions = torch.nonzero(valid_transition[row, 1:], as_tuple=False).flatten()
        if source_positions.numel() > 1:
            assignments = top4_indices[row, source_positions].clone()
            sham[row, source_positions] = assignments.roll(shifts=1, dims=0)
    return sham


@torch.no_grad()
def extract_batch_features(
    model: torch.nn.Module,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Extract row-level final-history features from one legal target window."""
    require(model.lmm is not None, "Transition error requires static Hard-LMM")
    require(model.lmm.topk == 4, "Transition error contract requires top-4 retrieval")
    dts, quantities, mask, lengths = right_pad_batch(dts, quantities, mask)
    rows = torch.arange(dts.size(0), device=dts.device)
    target_positions = lengths - 1
    history_positions = lengths - 2

    history_quantities = quantities.masked_fill(~mask, 0.0).clone()
    history_quantities[rows, target_positions] = 0.0
    memory_write_mask = mask.clone()
    memory_write_mask[rows, target_positions] = False

    local = model._encode_base(
        dts,
        history_quantities,
        mask,
        memory_write_mask=memory_write_mask,
    )
    residual, trace = model.lmm.retrieve(local)
    valid = mask.unsqueeze(-1).to(dtype=local.dtype)
    local = local * valid
    residual = residual * valid
    fused = (local + residual) * valid
    top4 = trace["prototype_indices"]
    similarities = trace["topk_similarity"]
    require(top4.shape[:2] == mask.shape and top4.size(-1) == 4, "Bad retrieval trace")
    prototype_count = int(model.lmm.mem.size(1))

    base_pre_activation = model.quantity_head(fused).squeeze(-1)
    observed_log1p = quantities.clamp_min(0.0).log1p()
    actual = transition_error_features(
        base_pre_activation,
        observed_log1p,
        top4,
        mask,
        prototype_count,
        memory_write_mask=memory_write_mask,
    )
    sham_write_indices = cyclic_transition_sham_indices(
        top4, mask, memory_write_mask
    )
    sham = transition_error_features(
        base_pre_activation,
        observed_log1p,
        top4,
        mask,
        prototype_count,
        memory_write_mask=memory_write_mask,
        write_top4_prototype_indices=sham_write_indices,
    )

    final_top4 = top4[rows, history_positions]
    final_counts = actual["prototype_error_count"][rows, history_positions]
    selected_counts = final_counts.gather(1, final_top4).to(base_pre_activation.dtype)
    final_activation = base_pre_activation[rows, history_positions]
    output = {
        "quantity": quantities[rows, target_positions].float(),
        "base_pre_activation": final_activation,
        "base_log_prediction": F.softplus(final_activation),
        "base_raw_prediction": torch.expm1(F.softplus(final_activation)),
        "history_length": lengths - 1,
        "transition_count": actual["valid_transition_count"][rows, history_positions],
        "selected_occupied_fraction": (selected_counts > 0).to(local.dtype).mean(-1),
        "selected_count_mean": selected_counts.mean(-1),
        "selected_count_std": selected_counts.std(-1, unbiased=False),
        "last_error": actual["last_error"][rows, history_positions],
        "unconditioned_mean_error": actual["unconditioned_prefix_mean"][
            rows, history_positions
        ],
        "prototype_conditioned_error": actual[
            "prototype_conditioned_top4_mean"
        ][rows, history_positions],
        "sham_prototype_conditioned_error": sham[
            "prototype_conditioned_top4_mean"
        ][rows, history_positions],
        "final_h": local[rows, history_positions],
        "final_residual": residual[rows, history_positions],
        "final_top4_indices": final_top4,
        "final_top4_similarity": similarities[rows, history_positions],
        "target_position": target_positions,
        "history_position": history_positions,
    }
    for name, tensor in output.items():
        require(bool(torch.isfinite(tensor).all()), f"Nonfinite batch field: {name}")
    return {name: tensor.detach().cpu() for name, tensor in output.items()}


def assert_official_prediction_parity(
    model: torch.nn.Module,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
    extracted: Mapping[str, torch.Tensor],
) -> float:
    with torch.no_grad():
        official = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )["pred_qty"].detach().cpu()
    observed = extracted["base_raw_prediction"]
    torch.testing.assert_close(observed, official, rtol=1e-5, atol=1e-6)
    return float((observed.double() - official.double()).abs().max())


def assert_target_padding_feature_invariance(
    model: torch.nn.Module,
    dts: torch.Tensor,
    mask: torch.Tensor,
    quantities: torch.Tensor,
    extracted: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Change forbidden target/padding values and require identical features."""
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    row_ids = torch.arange(mask.size(0), device=mask.device)
    positions = torch.arange(mask.size(1), device=mask.device).expand_as(mask)
    target_positions = torch.where(mask, positions, -1).max(dim=1).values
    changed_dts[row_ids, target_positions] += 10007.0
    changed_quantities[row_ids, target_positions] += 20011.0
    changed_dts[~mask] = 30013.0
    changed_quantities[~mask] = 40009.0
    changed = extract_batch_features(model, changed_dts, mask, changed_quantities)
    differences: dict[str, float] = {}
    for name in FEATURE_INVARIANCE_FIELDS:
        before = extracted[name]
        after = changed[name]
        if before.is_floating_point():
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
            differences[name] = float((before.double() - after.double()).abs().max())
        else:
            require(torch.equal(before, after), f"Forbidden input changed {name}")
            differences[name] = 0.0
    return differences


def _identity_fields(
    dataset: RMTPPWeekLookbackDataset,
    target_indices: torch.Tensor,
    part_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    contexts: list[int] = []
    target_sequences: list[int] = []
    for target_index, part_index in zip(target_indices.tolist(), part_indices.tolist()):
        indexed_part, context_end = dataset.index[int(target_index)]
        require(indexed_part == int(part_index), "Dataset target/series identity changed")
        contexts.append(int(context_end))
        target_sequences.append(int(dataset.seq_lists[indexed_part][context_end + 1]))
    return {
        "target_index": target_indices.clone().to(torch.long),
        "series_id": part_indices.clone().to(torch.long),
        "fold": torch.tensor(
            [fold_for_series(series) for series in part_indices], dtype=torch.long
        ),
        "context_end": torch.tensor(contexts, dtype=torch.long),
        "target_seq": torch.tensor(target_sequences, dtype=torch.long),
    }


def extract_dataset_cache(
    model: torch.nn.Module,
    dataset: RMTPPWeekLookbackDataset,
    indices: torch.Tensor,
    *,
    dataset_name: str,
    body_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.requires_grad_(False).eval()
    state_before = canonical_state_dict_sha256(model.state_dict())
    memory_before = model.lmm.mem.detach().cpu().clone()
    loader = DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_week_lookback,
    )
    arrays: dict[str, list[torch.Tensor]] = {}
    offset = 0
    started = time.monotonic()
    parity_max_abs: float | None = None
    invariance_max_abs: dict[str, float] | None = None
    for batch_number, (_, dts, mask, part_indices, quantities) in enumerate(loader):
        require(quantities is not None, "Quantity observations are missing")
        batch = extract_batch_features(model, dts, mask, quantities)
        count = len(part_indices)
        selected = indices[offset : offset + count]
        batch.update(_identity_fields(dataset, selected, part_indices.cpu()))
        if batch_number == 0:
            parity_max_abs = assert_official_prediction_parity(
                model, dts, mask, quantities, batch
            )
            invariance_max_abs = assert_target_padding_feature_invariance(
                model, dts, mask, quantities, batch
            )
        for name, tensor in batch.items():
            require(bool(torch.isfinite(tensor).all()), f"Nonfinite cache field: {name}")
            arrays.setdefault(name, []).append(tensor)
        offset += count
        if batch_number % 32 == 0 or offset == len(indices):
            print(
                json.dumps(
                    {
                        "stage": "extract",
                        "dataset": dataset_name,
                        "completed": offset,
                        "total": len(indices),
                        "seconds": round(time.monotonic() - started, 2),
                    }
                ),
                flush=True,
            )
    require(offset == len(indices), "Incomplete diagnostic sample")
    state_after = canonical_state_dict_sha256(model.state_dict())
    require(state_after == state_before, "Frozen model state changed during extraction")
    require(torch.equal(memory_before, model.lmm.mem.detach().cpu()), "Memory state changed")
    cache: dict[str, Any] = {
        name: torch.cat(chunks, dim=0) for name, chunks in arrays.items()
    }
    cache["body_threshold"] = float(body_threshold)
    require(set(REQUIRED_CACHE_FIELDS).issubset(cache), "Analysis cache schema is incomplete")
    n = len(indices)
    require(
        all(len(cache[name]) == n for name in REQUIRED_CACHE_FIELDS),
        "Analysis cache fields are not row-aligned",
    )
    require(torch.equal(cache["target_index"], indices.cpu()), "Sample order changed")
    for fold in (0, 1):
        selected = cache["fold"] == fold
        require(
            int(selected.sum()) >= 512,
            f"{dataset_name} fold {fold} has too few targets",
        )
        require(
            int(cache["series_id"][selected].unique().numel()) >= 30,
            f"{dataset_name} fold {fold} has too few series",
        )
    audit = {
        "state_sha256_before_and_after": state_before,
        "memory_state_unchanged": True,
        "official_prediction_parity_first_batch": True,
        "official_prediction_max_abs": parity_max_abs,
        "target_padding_feature_invariance_first_batch": True,
        "target_padding_feature_max_abs": invariance_max_abs,
        "sample_targets": n,
        "sample_series": int(cache["series_id"].unique().numel()),
        "fold_targets": {
            str(fold): int((cache["fold"] == fold).sum()) for fold in (0, 1)
        },
        "fold_series": {
            str(fold): int(
                cache["series_id"][cache["fold"] == fold].unique().numel()
            )
            for fold in (0, 1)
        },
    }
    return cache, audit


def _runtime_manifest() -> dict[str, Any]:
    return {
        "python": sys.version,
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "polars": pl.__version__,
        "platform": platform.platform(),
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "batch_size": BATCH_SIZE,
    }


def extract_phase() -> None:
    contract = load_contract()
    # Contract reads are safe; refuse before registry, dataset, or checkpoint access.
    require(not RAW.exists() and not RESULT.exists(), "Refusing to overwrite diagnostic output")
    registry = read_json(ROOT / contract["reference"]["registry"])
    torch.set_num_threads(TORCH_THREADS)
    torch.manual_seed(contract["diagnostic_scope"]["sample_seed"])
    RAW.mkdir(parents=True)
    RESULT.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "status": "extracting",
        "started_at": utc_now(),
        "contract_sha256": sha256_file(CONTRACT),
        "registry": contract["reference"]["registry"],
        "registry_sha256": contract["reference"]["registry_sha256"],
        "datasets": {},
        "validation_rows_materialized": False,
        "held_out_rows_materialized": False,
        "model_parameter_updates": False,
        "server_accessed": False,
        "deferred_follow_up": "quantile-adaptive loss and checkpoint alignment",
    }
    save_json(RESULT / "execution_contract.json", contract)
    save_json(RESULT / "execution_manifest.json", manifest)
    try:
        manifest.update(
            source_revision=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            source_hashes=source_hashes(),
            runtime=_runtime_manifest(),
        )
        save_json(RESULT / "execution_manifest.json", manifest)
        for dataset_contract in contract["datasets"]:
            name = dataset_contract["dataset"]
            row = _registry_row(registry, name)
            checkpoint, launch_path, _, checkpoint_hashes, used_override = (
                resolve_frozen_checkpoint(dataset_contract, row)
            )
            input_hashes = {
                row["data_path"]: row["data_sha256"],
                row["split_manifest_path"]: row["split_manifest_sha256"],
                **checkpoint_hashes,
            }
            verify_hashes(input_hashes, label=f"{name} frozen input")
            frame = load_train_frame(ROOT / row["data_path"])
            require(
                frame.height == int(dataset_contract["train_rows"]),
                f"{name} train row count changed",
            )
            require(
                frame["oper_part_no"].n_unique() == int(dataset_contract["train_series"]),
                f"{name} train series count changed",
            )
            prepared = prepare_count_frame(frame)
            dataset = RMTPPWeekLookbackDataset(
                prepared,
                lookback_weeks=row["lookback"],
                max_seq_len=row["max_seq_len"],
                mode="all",
                split_col="chronological_split",
                target_splits={"train"},
            )
            require(
                len(dataset) == frame.height - frame["oper_part_no"].n_unique(),
                f"{name} train target population changed",
            )
            indices = sample_indices(
                len(dataset),
                contract["diagnostic_scope"]["maximum_targets_per_dataset"],
                seed=contract["diagnostic_scope"]["sample_seed"],
            )
            launch = read_json(launch_path)
            model, restore_audit = restore_b0(checkpoint, launch, "cpu")
            require(
                restore_audit["model_state_sha256"] == row["checkpoint_state_sha256"],
                f"{name} restored state differs from registry-pinned T0",
            )
            if used_override:
                require(
                    restore_audit["model_state_sha256"]
                    == dataset_contract["local_checkpoint_override"][
                        "checkpoint_state_sha256"
                    ],
                    "Intermittent override state audit failed",
                )
            cache, extraction_audit = extract_dataset_cache(
                model,
                dataset,
                indices,
                dataset_name=name,
                body_threshold=float(dataset_contract["body_threshold_train_p95"]),
            )
            cache_path = RAW / name / "train_cache.pt"
            require(not cache_path.exists(), "Refusing to overwrite a train cache")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(cache, cache_path)
            manifest["datasets"][name] = {
                "input_hashes": input_hashes,
                "checkpoint_override_used": used_override,
                "registry_checkpoint_state_sha256": row["checkpoint_state_sha256"],
                "restore_audit": restore_audit,
                "train_rows": frame.height,
                "train_series": frame["oper_part_no"].n_unique(),
                "train_targets": len(dataset),
                "sample_index_sha256": hashlib.sha256(
                    indices.cpu().numpy().tobytes()
                ).hexdigest(),
                "cache_path": str(cache_path.relative_to(ROOT)),
                "cache_sha256": sha256_file(cache_path),
                "extraction_audit": extraction_audit,
            }
            save_json(RESULT / "execution_manifest.json", manifest)
            del model, dataset, prepared, frame, cache
        require(
            sha256_file(CONTRACT) == manifest["contract_sha256"],
            "Contract changed during extraction",
        )
        require(
            sha256_file(ROOT / manifest["registry"]) == manifest["registry_sha256"],
            "Registry changed during extraction",
        )
        verify_hashes(manifest["source_hashes"], label="diagnostic source")
        for name, metadata in manifest["datasets"].items():
            verify_hashes(metadata["input_hashes"], label=f"{name} frozen input")
        manifest.update(status="extracted", extraction_completed_at=utc_now())
        save_json(RESULT / "execution_manifest.json", manifest)
    except BaseException as exc:
        manifest.update(
            status="failed_extraction",
            failed_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        save_json(RESULT / "execution_manifest.json", manifest)
        raise


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def analyze_phase() -> None:
    from paper.scripts import hard_lmm_transition_error_analysis as analysis

    contract = load_contract()
    require(RESULT.is_dir(), "Extraction result is missing")
    manifest_path = RESULT / "execution_manifest.json"
    require(manifest_path.is_file(), "Extraction manifest is missing")
    manifest = read_json(manifest_path)
    require(manifest.get("status") == "extracted", "Extraction incomplete or already analyzed")
    analysis_path = RESULT / "analysis.json"
    decision_path = RESULT / "evidence_decision.json"
    prediction_path = RAW / "oof_predictions.npz"
    reader_path = RESULT / "reader_evidence.json"
    require(
        not any(path.exists() for path in (analysis_path, decision_path, prediction_path, reader_path)),
        "Refusing to overwrite analysis output",
    )
    torch.set_num_threads(TORCH_THREADS)
    manifest.update(status="analyzing", analysis_started_at=utc_now())
    save_json(manifest_path, manifest)
    try:
        require(
            sha256_file(CONTRACT) == manifest["contract_sha256"],
            "Contract changed after extraction",
        )
        require(
            sha256_file(ROOT / manifest["registry"]) == manifest["registry_sha256"],
            "Registry changed after extraction",
        )
        verify_hashes(manifest["source_hashes"], label="diagnostic source")
        for name, metadata in manifest["datasets"].items():
            verify_hashes(metadata["input_hashes"], label=f"{name} frozen input")
            cache_path = ROOT / metadata["cache_path"]
            require(cache_path.is_file(), f"Missing frozen cache: {name}")
            require(
                sha256_file(cache_path) == metadata["cache_sha256"],
                f"Changed frozen cache: {name}",
            )
        results: dict[str, Any] = {}
        prediction_arrays: dict[str, np.ndarray] = {}
        repeats = contract["signal_gate"]["paired_series_bootstrap"]["repeats"]
        for name in DATASETS:
            metadata = manifest["datasets"][name]
            cache_path = ROOT / metadata["cache_path"]
            require(
                sha256_file(cache_path) == metadata["cache_sha256"],
                f"Changed frozen cache before fit: {name}",
            )
            cache = torch.load(cache_path, map_location="cpu", weights_only=False)
            require(
                set(REQUIRED_CACHE_FIELDS).issubset(cache),
                f"Incomplete analysis cache: {name}",
            )
            print(
                json.dumps({"stage": "analyze", "dataset": name, "started_at": utc_now()}),
                flush=True,
            )
            summary, arrays = analysis.analyze_cache(
                cache, bootstrap_repeats=repeats
            )
            results[name] = summary
            for key, value in arrays.items():
                prediction_arrays[f"{name}__{key}"] = _to_numpy(value)
            save_json(analysis_path, results)
        decision = analysis.aggregate_common_gate(results)
        save_json(decision_path, decision)
        np.savez_compressed(prediction_path, **prediction_arrays)
        reader_evidence = {
            "status": "complete",
            "contract_id": contract["contract_id"],
            "research_question": contract["research_question"],
            "scope": "frozen T0 train-only signal diagnostic",
            "validation_rows_materialized": False,
            "held_out_test_evaluated": False,
            "datasets": list(DATASETS),
            "analysis": results,
            "common_gate": decision,
            "interpretation": (
                "A pass is only a necessary train-internal signal for implementation; "
                "it is not validation performance or model adoption."
            ),
            "failure_action": contract["signal_gate"]["failure_action"],
            "deferred_follow_up": "quantile-adaptive loss and checkpoint alignment",
        }
        save_json(reader_path, reader_evidence)
        require(
            sha256_file(CONTRACT) == manifest["contract_sha256"],
            "Contract changed during analysis",
        )
        verify_hashes(manifest["source_hashes"], label="diagnostic source")
        for name, metadata in manifest["datasets"].items():
            verify_hashes(metadata["input_hashes"], label=f"{name} frozen input")
            require(
                sha256_file(ROOT / metadata["cache_path"]) == metadata["cache_sha256"],
                f"Changed frozen cache during fit: {name}",
            )
        manifest.update(
            status="complete",
            completed_at=utc_now(),
            analysis_sha256=sha256_file(analysis_path),
            evidence_decision_sha256=sha256_file(decision_path),
            oof_predictions_sha256=sha256_file(prediction_path),
            reader_evidence_sha256=sha256_file(reader_path),
        )
        save_json(manifest_path, manifest)
    except BaseException as exc:
        manifest.update(
            status="failed_analysis",
            failed_at=utc_now(),
            error=f"{type(exc).__name__}: {exc}",
        )
        save_json(manifest_path, manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("extract", "analyze", "all"),
        default="all",
    )
    args = parser.parse_args()
    if args.phase in ("extract", "all"):
        extract_phase()
    if args.phase in ("analyze", "all"):
        analyze_phase()


if __name__ == "__main__":
    main()
