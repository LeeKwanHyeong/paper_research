"""Exact artifact comparator for independent strict TitanTPP reproduction runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from simple_lab_test.search.common.runner import (
    canonical_state_dict_sha256,
    torch_load_checkpoint,
)


SELECTIONS = ("best_score", "best_val_nll", "final")
STRICT_CUBLAS_CONFIGS = {":4096:8", ":16:8"}
RUNTIME_IDENTITY_KEYS = (
    "source_revision",
    "python_hash_seed",
    "cublas_workspace_config",
    "python_version",
    "numpy_version",
    "polars_version",
    "torch_version",
    "torch_cuda_version",
    "cudnn_version",
    "torch_deterministic_algorithms",
    "torch_deterministic_warn_only",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "cuda_available",
    "cuda_devices",
    "train_loader_num_workers",
    "grouped_series_order",
)


@dataclass(frozen=True)
class RunArtifacts:
    base_dir: Path
    root_manifest_path: Path
    run_dir: Path
    run_manifest_path: Path
    summary_path: Path
    history_path: Path


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        value = json.load(file_obj)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _find_one(base_dir: Path, pattern: str) -> Path:
    matches = sorted(base_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one artifact for pattern={pattern!r} under {base_dir}; "
            f"found={len(matches)}"
        )
    return matches[0]


def discover_run_artifacts(base_dir: str | Path) -> RunArtifacts:
    base_path = Path(base_dir).expanduser().resolve()
    root_manifest_path = base_path / "experiment_manifest.json"
    if not root_manifest_path.is_file():
        raise FileNotFoundError(f"Missing root experiment manifest: {root_manifest_path}")
    summary_path = _find_one(base_path, "runs/**/metrics/summary.json")
    run_dir = summary_path.parent.parent
    run_manifest_path = run_dir / "manifest" / "run_config.json"
    history_path = run_dir / "metrics" / "history.json"
    for path in (run_manifest_path, history_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing strict reproduction artifact: {path}")
    return RunArtifacts(
        base_dir=base_path,
        root_manifest_path=root_manifest_path,
        run_dir=run_dir,
        run_manifest_path=run_manifest_path,
        summary_path=summary_path,
        history_path=history_path,
    )


def _without_base_dir(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    normalized.pop("base_dir", None)
    return normalized


def _runtime_identity(runtime: dict[str, Any]) -> dict[str, Any]:
    return {key: runtime.get(key) for key in RUNTIME_IDENTITY_KEYS}


def _dataset_hashes_complete(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value) and all(
            _dataset_hashes_complete(item) for item in value.values()
        )
    if isinstance(value, list):
        return bool(value) and all(_dataset_hashes_complete(item) for item in value)
    if isinstance(value, str):
        return len(value) == 64 and all(
            character in "0123456789abcdef" for character in value.lower()
        )
    return False


def _source_hashes(runtime: dict[str, Any]) -> Any:
    dataset_sources = runtime.get("dataset_sources")
    if isinstance(dataset_sources, dict):
        return {
            dataset: [row.get("sha256") for row in rows]
            for dataset, rows in dataset_sources.items()
        }
    if isinstance(dataset_sources, list):
        return [row.get("sha256") for row in dataset_sources]
    return None


def _strict_runtime_is_valid(
    runtime: dict[str, Any],
    *,
    expected_seed: int,
    require_loader_seed: bool,
) -> bool:
    source_revision = str(runtime.get("source_revision", ""))
    valid = (
        runtime.get("mode") == "strict"
        and len(source_revision) in {40, 64}
        and all(character in "0123456789abcdef" for character in source_revision.lower())
        and runtime.get("python_hash_seed") == str(expected_seed)
        and runtime.get("cublas_workspace_config") in STRICT_CUBLAS_CONFIGS
        and runtime.get("torch_deterministic_algorithms") is True
        and runtime.get("torch_deterministic_warn_only") is False
        and runtime.get("cudnn_deterministic") is True
        and runtime.get("cudnn_benchmark") is False
        and runtime.get("cuda_available") is True
        and runtime.get("train_loader_num_workers") == 0
        and runtime.get("grouped_series_order") == "oper_part_no_ascending"
    )
    if require_loader_seed:
        valid = (
            valid
            and runtime.get("train_loader_seed") == expected_seed
            and runtime.get("train_loader_generator") == "dedicated_torch_generator"
        )
    return bool(valid)


def compare_strict_runs(
    run_a_dir: str | Path,
    run_b_dir: str | Path,
    *,
    expected_epochs: int = 3,
    expected_seed: int = 42,
    expected_sample_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Compare only training/validation identity artifacts; held-out files stay unread."""
    expected_counts = expected_sample_counts or {
        "train": 136256,
        "validation": 41901,
        "test": 41344,
    }
    artifacts_a = discover_run_artifacts(run_a_dir)
    artifacts_b = discover_run_artifacts(run_b_dir)
    root_a = _load_json(artifacts_a.root_manifest_path)
    root_b = _load_json(artifacts_b.root_manifest_path)
    manifest_a = _load_json(artifacts_a.run_manifest_path)
    manifest_b = _load_json(artifacts_b.run_manifest_path)
    summary_a = _load_json(artifacts_a.summary_path)
    summary_b = _load_json(artifacts_b.summary_path)
    history_bytes_a = artifacts_a.history_path.read_bytes()
    history_bytes_b = artifacts_b.history_path.read_bytes()
    history_a = _load_json(artifacts_a.history_path).get("history")
    history_b = _load_json(artifacts_b.history_path).get("history")

    checks: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, **details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **details})

    root_cfg_a = root_a.get("experiment_config", {})
    root_cfg_b = root_b.get("experiment_config", {})
    root_runtime_a = root_a.get("reproducibility", {})
    root_runtime_b = root_b.get("reproducibility", {})
    run_runtime_a = manifest_a.get("reproducibility", {})
    run_runtime_b = manifest_b.get("reproducibility", {})

    add_check(
        "matched_experiment_config",
        _without_base_dir(root_cfg_a) == _without_base_dir(root_cfg_b),
        run_a_sha256=_canonical_json_sha256(_without_base_dir(root_cfg_a)),
        run_b_sha256=_canonical_json_sha256(_without_base_dir(root_cfg_b)),
    )
    add_check(
        "strict_experiment_contract",
        all(
            config.get("reproducibility_mode") == "strict"
            and config.get("epochs") == expected_epochs
            and config.get("seeds") == [expected_seed]
            and config.get("split_mode") == "fixed"
            and config.get("datasets") == ["intermittent"]
            and config.get("models") == ["titantpp"]
            for config in (root_cfg_a, root_cfg_b)
        ),
    )
    add_check(
        "strict_root_runtime",
        _strict_runtime_is_valid(
            root_runtime_a,
            expected_seed=expected_seed,
            require_loader_seed=False,
        )
        and _strict_runtime_is_valid(
            root_runtime_b,
            expected_seed=expected_seed,
            require_loader_seed=False,
        ),
    )
    runtime_identity_a = _runtime_identity(root_runtime_a)
    runtime_identity_b = _runtime_identity(root_runtime_b)
    add_check(
        "matched_runtime_identity",
        runtime_identity_a == runtime_identity_b,
        run_a_sha256=_canonical_json_sha256(runtime_identity_a),
        run_b_sha256=_canonical_json_sha256(runtime_identity_b),
    )
    source_hashes_a = _source_hashes(root_runtime_a)
    source_hashes_b = _source_hashes(root_runtime_b)
    add_check(
        "matched_complete_dataset_hashes",
        source_hashes_a == source_hashes_b and _dataset_hashes_complete(source_hashes_a),
        run_a=source_hashes_a,
        run_b=source_hashes_b,
    )
    add_check(
        "strict_run_loader_runtime",
        _strict_runtime_is_valid(
            run_runtime_a,
            expected_seed=expected_seed,
            require_loader_seed=True,
        )
        and _strict_runtime_is_valid(
            run_runtime_b,
            expected_seed=expected_seed,
            require_loader_seed=True,
        ),
    )
    add_check(
        "matched_run_runtime_identity",
        _runtime_identity(run_runtime_a) == _runtime_identity(run_runtime_b),
        run_a_sha256=_canonical_json_sha256(_runtime_identity(run_runtime_a)),
        run_b_sha256=_canonical_json_sha256(_runtime_identity(run_runtime_b)),
    )
    run_source_hashes_a = _source_hashes(run_runtime_a)
    run_source_hashes_b = _source_hashes(run_runtime_b)
    root_inter_hashes = (
        source_hashes_a.get("intermittent")
        if isinstance(source_hashes_a, dict)
        else None
    )
    add_check(
        "matched_complete_run_dataset_hashes",
        run_source_hashes_a == run_source_hashes_b
        and _dataset_hashes_complete(run_source_hashes_a)
        and run_source_hashes_a == root_inter_hashes,
        run_a=run_source_hashes_a,
        run_b=run_source_hashes_b,
    )

    for config_name in ("run_config", "training_config", "rmtpp_config", "encoder_config"):
        value_a = manifest_a.get(config_name)
        value_b = manifest_b.get(config_name)
        add_check(
            f"matched_{config_name}",
            value_a == value_b,
            run_a_sha256=_canonical_json_sha256(value_a),
            run_b_sha256=_canonical_json_sha256(value_b),
        )

    q2_contract_a = manifest_a.get("rmtpp_config", {})
    q2_contract_b = manifest_b.get("rmtpp_config", {})
    add_check(
        "q2_model_contract",
        all(
            config.get("qty_decoder_mode") == "direct_raw_qty"
            and config.get("magnitude_norm_mode") == "causal_shrinkage_revin"
            and config.get("magnitude_encoder_gradient_mode") == "coupled"
            and config.get("magnitude_aux_loss_mode") == "none"
            and config.get("train_loss_scope") == "target_only"
            and config.get("loss_mode") == "hybrid"
            for config in (q2_contract_a, q2_contract_b)
        ),
    )
    add_check(
        "loader_sample_counts",
        manifest_a.get("loader_sample_counts") == expected_counts
        and manifest_b.get("loader_sample_counts") == expected_counts,
        expected=expected_counts,
        run_a=manifest_a.get("loader_sample_counts"),
        run_b=manifest_b.get("loader_sample_counts"),
    )

    summary_identity_valid = all(
        summary.get("status") == "success"
        and summary.get("reproducibility_mode") == "strict"
        and summary.get("train_loader_seed") == expected_seed
        and summary.get("source_revision") == root_runtime_a.get("source_revision")
        and summary.get("epochs") == expected_epochs
        for summary in (summary_a, summary_b)
    )
    add_check("summary_identity", summary_identity_valid)

    selected_epochs_a = {
        "best_score": summary_a.get("best_score_epoch"),
        "best_val_nll": summary_a.get("best_val_nll_epoch"),
        "final": summary_a.get("final_epoch"),
    }
    selected_epochs_b = {
        "best_score": summary_b.get("best_score_epoch"),
        "best_val_nll": summary_b.get("best_val_nll_epoch"),
        "final": summary_b.get("final_epoch"),
    }
    epochs_in_range = all(
        isinstance(epoch, int) and 1 <= epoch <= expected_epochs
        for epoch in selected_epochs_a.values()
    ) and selected_epochs_a["final"] == expected_epochs
    add_check(
        "exact_selected_epochs",
        selected_epochs_a == selected_epochs_b and epochs_in_range,
        run_a=selected_epochs_a,
        run_b=selected_epochs_b,
    )

    history_sha_a = _sha256_bytes(history_bytes_a)
    history_sha_b = _sha256_bytes(history_bytes_b)
    add_check(
        "exact_history_json",
        history_bytes_a == history_bytes_b,
        run_a_sha256=history_sha_a,
        run_b_sha256=history_sha_b,
    )
    expected_epoch_sequence = list(range(1, expected_epochs + 1))
    parsed_epochs_a = (
        [row.get("epoch") for row in history_a]
        if isinstance(history_a, list)
        else None
    )
    parsed_epochs_b = (
        [row.get("epoch") for row in history_b]
        if isinstance(history_b, list)
        else None
    )
    add_check(
        "complete_epoch_history",
        parsed_epochs_a == expected_epoch_sequence
        and parsed_epochs_b == expected_epoch_sequence,
        expected=expected_epoch_sequence,
        run_a=parsed_epochs_a,
        run_b=parsed_epochs_b,
    )

    state_digests_a: dict[str, str | None] = {}
    state_digests_b: dict[str, str | None] = {}
    for selection in SELECTIONS:
        checkpoint_a = artifacts_a.run_dir / "checkpoints" / f"{selection}_model.pt"
        checkpoint_b = artifacts_b.run_dir / "checkpoints" / f"{selection}_model.pt"
        payload_a = torch_load_checkpoint(checkpoint_a, map_location="cpu")
        payload_b = torch_load_checkpoint(checkpoint_b, map_location="cpu")
        stored_a = payload_a.get("model_state_sha256")
        stored_b = payload_b.get("model_state_sha256")
        recomputed_a = canonical_state_dict_sha256(payload_a["model_state_dict"])
        recomputed_b = canonical_state_dict_sha256(payload_b["model_state_dict"])
        summary_digest_a = summary_a.get(f"{selection}_state_sha256")
        summary_digest_b = summary_b.get(f"{selection}_state_sha256")
        state_digests_a[selection] = stored_a
        state_digests_b[selection] = stored_b
        add_check(
            f"{selection}_checkpoint_digest_integrity",
            stored_a == recomputed_a == summary_digest_a
            and stored_b == recomputed_b == summary_digest_b,
            run_a=stored_a,
            run_b=stored_b,
        )

    add_check(
        "exact_selection_state_digests",
        state_digests_a == state_digests_b
        and all(isinstance(value, str) and len(value) == 64 for value in state_digests_a.values()),
        run_a=state_digests_a,
        run_b=state_digests_b,
    )

    mismatch_names = [check["name"] for check in checks if not check["passed"]]
    return {
        "schema_version": 1,
        "compared_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if not mismatch_names else "fail",
        "scope": "strict training/validation reproducibility; held-out artifacts were not read",
        "run_a": str(artifacts_a.base_dir),
        "run_b": str(artifacts_b.base_dir),
        "expected_epochs": int(expected_epochs),
        "expected_seed": int(expected_seed),
        "history_sha256": {"run_a": history_sha_a, "run_b": history_sha_b},
        "selected_epochs": {"run_a": selected_epochs_a, "run_b": selected_epochs_b},
        "state_digests": {"run_a": state_digests_a, "run_b": state_digests_b},
        "checks": checks,
        "mismatch_count": len(mismatch_names),
        "mismatch_names": mismatch_names,
        "decision": (
            "exact reproduction gate passed"
            if not mismatch_names
            else "exact reproduction gate failed; do not run e50"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two independent strict TitanTPP Q2 runs exactly."
    )
    parser.add_argument("--run-a", required=True)
    parser.add_argument("--run-b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-epochs", type=int, default=3)
    parser.add_argument("--expected-seed", type=int, default=42)
    parser.add_argument("--expected-train-samples", type=int, default=136256)
    parser.add_argument("--expected-validation-samples", type=int, default=41901)
    parser.add_argument("--expected-test-samples", type=int, default=41344)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = compare_strict_runs(
            args.run_a,
            args.run_b,
            expected_epochs=int(args.expected_epochs),
            expected_seed=int(args.expected_seed),
            expected_sample_counts={
                "train": int(args.expected_train_samples),
                "validation": int(args.expected_validation_samples),
                "test": int(args.expected_test_samples),
            },
        )
        exit_code = 0 if report["status"] == "pass" else 1
    except Exception as exc:
        report = {
            "schema_version": 1,
            "compared_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "error",
            "scope": "strict training/validation reproducibility; held-out artifacts were not read",
            "error": repr(exc),
            "decision": "artifact validation failed; do not run e50",
        }
        exit_code = 2
    with output_path.open("w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, ensure_ascii=True, indent=2)
        file_obj.write("\n")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
