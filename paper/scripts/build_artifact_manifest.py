#!/usr/bin/env python3
"""Build the paper comparison artifact manifest from experiment leaderboards."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


QUAL_READY = "final_comparison_ready"
QUAL_DRAFT = "draft_only"
QUAL_PENDING = "pending_active_run"
QUAL_MISSING = "rerun_required"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--contract", type=Path, default=Path("paper/contracts/final_fair_comparison.json"))
    parser.add_argument("--dataset-contract", type=Path, default=Path("paper/contracts/datasets.json"))
    parser.add_argument("--artifact-root", type=Path, default=Path("search_artifacts"))
    parser.add_argument("--active-run-root", type=Path)
    parser.add_argument("--merge-manifest", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("paper/manifests"))
    return parser.parse_args()


def rooted(project_root: Path, value: Path) -> Path:
    return value if value.is_absolute() else project_root / value


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalized(value: Any, default: str = "") -> str:
    text = "" if value is None else str(value).strip().lower()
    return text or default


def as_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool | None = None) -> bool | None:
    if isinstance(value, bool):
        return value
    text = normalized(value)
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return default


def resolve_artifact_path(project_root: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def load_rows(artifact_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for leaderboard in sorted(artifact_root.glob("**/leaderboard/runs.csv")):
        try:
            with leaderboard.open(newline="", encoding="utf-8") as handle:
                source_rows = list(csv.DictReader(handle))
        except (OSError, csv.Error, UnicodeDecodeError):
            continue
        for row in source_rows:
            row["_leaderboard_path"] = str(leaderboard)
            row["_artifact_root"] = str(leaderboard.parent.parent)
            rows.append(row)
    return rows


def expected_runs(contract: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = [int(seed) for seed in contract["common_protocol"]["seeds"]]
    return [
        {"dataset": dataset, "model": model, "seed": seed}
        for dataset in contract["datasets"]
        for model in dataset["models"]
        for seed in seeds
    ]


def default_value_head(row: dict[str, Any]) -> str:
    return normalized(row.get("value_head_mode"), "shared")


def default_gradient_mode(row: dict[str, Any]) -> str:
    return normalized(row.get("qty_mark_gradient_mode"), "coupled")


def default_value_input(row: dict[str, Any]) -> str:
    return normalized(row.get("value_input_mode"), "none")


def candidate_matches(row: dict[str, Any], expected: dict[str, Any], protocol: dict[str, Any]) -> bool:
    dataset = expected["dataset"]
    model = expected["model"]
    if normalized(row.get("status")) != "success":
        return False
    if normalized(row.get("dataset_name")) != normalized(dataset["dataset_id"]):
        return False
    if normalized(row.get("model_name")) != normalized(model["family"]):
        return False
    if as_int(row.get("seed")) != expected["seed"]:
        return False

    family = normalized(model["family"])
    if family in {"titantpp", "thp"}:
        if normalized(row.get("candidate_name")) != normalized(model["candidate"]):
            return False
    elif family == "rmtpp" and as_int(row.get("rmtpp_hidden_dim")) != int(model["rmtpp_hidden_dim"]):
        return False

    if normalized(row.get("loss_mode"), "residual_only") != normalized(model["quantity_objective"]):
        return False
    if default_value_input(row) != normalized(model["quantity_input"]):
        return False
    if default_value_head(row) != normalized(model["value_head_mode"]):
        return False
    if default_gradient_mode(row) != normalized(model["qty_mark_gradient_mode"]):
        return False
    if as_int(row.get("batch_size")) != int(protocol["batch_size"]):
        return False
    if abs((as_float(row.get("lr"), -1.0) or -1.0) - float(protocol["learning_rate"])) > 1e-12:
        return False
    if as_int(row.get("lookback_weeks")) != int(dataset["lookback"]):
        return False
    if as_int(row.get("max_seq_len")) != int(dataset["max_seq_len"]):
        return False
    if abs((as_float(row.get("scale_base"), -1.0) or -1.0) - float(dataset["scale_base"])) > 1e-12:
        return False
    return normalized(row.get("split_mode")) == "fixed"


def load_run_config(project_root: Path, row: dict[str, Any]) -> tuple[Path | None, dict[str, Any] | None]:
    run_dir = resolve_artifact_path(project_root, str(row.get("run_dir") or ""))
    if run_dir is None:
        return None, None
    path = run_dir / "manifest" / "run_config.json"
    if not path.is_file():
        return run_dir, None
    try:
        return run_dir, read_json(path)
    except (OSError, json.JSONDecodeError):
        return run_dir, None


def dataset_hash_check(run_config: dict[str, Any] | None, expected_hashes: dict[str, str]) -> tuple[bool, str]:
    if not run_config:
        return False, "run_config_missing"
    sources = run_config.get("reproducibility", {}).get("dataset_sources")
    if not sources:
        sources = run_config.get("marked_meta", {}).get("dataset_sources")
    if not isinstance(sources, list):
        return False, "dataset_hashes_missing"
    actual = {
        str(item.get("role")): str(item.get("sha256"))
        for item in sources
        if isinstance(item, dict)
    }
    mismatches = [role for role, digest in expected_hashes.items() if actual.get(role) != digest]
    if mismatches:
        return False, "dataset_hash_mismatch:" + ",".join(mismatches)
    return True, ""


def test_evidence(row: dict[str, Any], run_dir: Path | None) -> bool:
    if as_bool(row.get("held_out_test_evaluated"), False):
        return True
    if any(
        key.startswith("test_")
        and key != "test_time_memory"
        and str(value or "").strip()
        for key, value in row.items()
    ):
        return True
    if run_dir and run_dir.is_dir():
        for folder in (run_dir / "metrics", run_dir / "paper_outputs"):
            if folder.is_dir() and any(path.name.startswith("test_") for path in folder.iterdir()):
                return True
    return False


def qualification_checks(
    project_root: Path,
    row: dict[str, Any],
    expected_revision: str,
    expected_hashes: dict[str, str],
) -> dict[str, Any]:
    run_dir, run_config = load_run_config(project_root, row)
    checkpoint = resolve_artifact_path(project_root, str(row.get("best_val_nll_checkpoint_path") or ""))
    hash_ok, hash_reason = dataset_hash_check(run_config, expected_hashes)
    test_used = test_evidence(row, run_dir)
    checks = {
        "status_success": normalized(row.get("status")) == "success",
        "initial_budget_met": as_int(row.get("epochs")) in {300, 800},
        "strict": normalized(row.get("reproducibility_mode")) == "strict",
        "source_revision": str(row.get("source_revision") or "") == expected_revision,
        "fixed_split": normalized(row.get("split_mode")) == "fixed",
        "validation_only": normalized(row.get("evaluation_scope")) == "validation_only",
        "held_out_test_locked": not test_used,
        "best_val_nll_checkpoint": bool(checkpoint and checkpoint.is_file()),
        "best_val_nll_state_hash": bool(str(row.get("best_val_nll_state_sha256") or "").strip()),
        "dataset_hashes": hash_ok,
        "run_config": run_config is not None,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    if hash_reason and "dataset_hashes" in reasons:
        reasons[reasons.index("dataset_hashes")] = hash_reason
    return {
        "checks": checks,
        "reasons": reasons,
        "run_dir": str(run_dir) if run_dir else "",
        "checkpoint": str(checkpoint) if checkpoint else "",
        "test_evidence": test_used,
    }


def candidate_score(row: dict[str, Any], expected_revision: str, details: dict[str, Any]) -> tuple[int, ...]:
    checks = details["checks"]
    return (
        int(all(checks.values())),
        int(str(row.get("source_revision") or "") == expected_revision),
        int(normalized(row.get("reproducibility_mode")) == "strict"),
        int(normalized(row.get("evaluation_scope")) == "validation_only"),
        int(not details["test_evidence"]),
        int(checks["dataset_hashes"]),
        int(checks["best_val_nll_checkpoint"]),
        as_int(row.get("epochs"), 0),
    )


def load_active_contract(path: Path | None) -> dict[str, Any] | None:
    if path is None or not (path / "launch_contract.json").is_file():
        return None
    try:
        return read_json(path / "launch_contract.json")
    except (OSError, json.JSONDecodeError):
        return None


def active_expected(active: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    if not active or normalized(active.get("status")) != "running":
        return False
    if normalized(expected["model"]["model_id"]) not in {"rmtpp_matched", "thp_matched"}:
        return False
    dataset_id = expected["dataset"]["dataset_id"]
    return any(item.get("dataset") == dataset_id for item in active.get("queue", []))


def build_manifest(
    project_root: Path,
    contract: dict[str, Any],
    dataset_contract: dict[str, Any],
    rows: list[dict[str, Any]],
    active: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    protocol = contract["common_protocol"]
    expected_revision = contract["benchmark_source_revision"]
    hashes = {item["dataset_id"]: item["expected_hashes"] for item in dataset_contract["datasets"]}
    manifest: list[dict[str, Any]] = []

    for item in expected_runs(contract):
        candidates = [row for row in rows if candidate_matches(row, item, protocol)]
        checked = [
            (
                row,
                qualification_checks(
                    project_root,
                    row,
                    expected_revision,
                    hashes[item["dataset"]["dataset_id"]],
                ),
            )
            for row in candidates
        ]
        checked.sort(key=lambda pair: candidate_score(pair[0], expected_revision, pair[1]), reverse=True)

        dataset = item["dataset"]
        model = item["model"]
        result: dict[str, Any] = {
            "dataset_id": dataset["dataset_id"],
            "dataset": dataset["paper_name"],
            "model_id": model["model_id"],
            "model": model["paper_label"],
            "comparison_role": model["comparison_role"],
            "seed": item["seed"],
        }

        if checked:
            row, details = checked[0]
            ready = all(details["checks"].values())
            result.update(
                {
                    "qualification": QUAL_READY if ready else QUAL_DRAFT,
                    "next_action": "retain" if ready else "rerun_under_frozen_contract",
                    "reasons": details["reasons"],
                    "artifact_root": row.get("_artifact_root", ""),
                    "leaderboard_path": row.get("_leaderboard_path", ""),
                    "run_dir": details["run_dir"],
                    "best_val_nll_checkpoint_path": details["checkpoint"],
                    "epochs": as_int(row.get("epochs")),
                    "best_val_nll_epoch": as_int(row.get("best_val_nll_epoch")),
                    "best_val_nll": as_float(row.get("best_val_nll")),
                    "source_revision": row.get("source_revision", ""),
                    "reproducibility_mode": row.get("reproducibility_mode", ""),
                    "split_mode": row.get("split_mode", ""),
                    "evaluation_scope": row.get("evaluation_scope", ""),
                    "held_out_test_evaluated": details["test_evidence"],
                    "checks": details["checks"],
                    "candidate_count": len(checked),
                }
            )
        elif active_expected(active, item):
            result.update(
                {
                    "qualification": QUAL_PENDING,
                    "next_action": "wait_for_active_run",
                    "reasons": ["active_run_not_complete"],
                    "artifact_root": "",
                    "leaderboard_path": "",
                    "run_dir": "",
                    "best_val_nll_checkpoint_path": "",
                    "epochs": 300,
                    "best_val_nll_epoch": -1,
                    "best_val_nll": None,
                    "source_revision": active.get("source_revision", "") if active else "",
                    "reproducibility_mode": "strict",
                    "split_mode": "fixed",
                    "evaluation_scope": "validation_only",
                    "held_out_test_evaluated": False,
                    "checks": {},
                    "candidate_count": 0,
                }
            )
        else:
            result.update(
                {
                    "qualification": QUAL_MISSING,
                    "next_action": "run_under_frozen_contract",
                    "reasons": ["no_matching_artifact"],
                    "artifact_root": "",
                    "leaderboard_path": "",
                    "run_dir": "",
                    "best_val_nll_checkpoint_path": "",
                    "epochs": -1,
                    "best_val_nll_epoch": -1,
                    "best_val_nll": None,
                    "source_revision": "",
                    "reproducibility_mode": "",
                    "split_mode": "",
                    "evaluation_scope": "",
                    "held_out_test_evaluated": False,
                    "checks": {},
                    "candidate_count": 0,
                }
            )
        manifest.append(result)
    return manifest


def merge_manifests(
    current: list[dict[str, Any]],
    imported: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    rank = {
        QUAL_MISSING: 1,
        QUAL_DRAFT: 2,
        QUAL_PENDING: 3,
        QUAL_READY: 4,
    }

    def key(item: dict[str, Any]) -> tuple[str, str, int]:
        return item["dataset_id"], item["model_id"], int(item["seed"])

    def score(item: dict[str, Any]) -> tuple[int, int, int]:
        return (
            rank.get(item.get("qualification", ""), 0),
            int(item.get("epochs") or -1),
            -len(item.get("reasons") or []),
        )

    selected = {key(item): item for item in current}
    for manifest in imported:
        for item in manifest:
            item_key = key(item)
            if item_key not in selected or score(item) > score(selected[item_key]):
                selected[item_key] = item
    return [selected[key(item)] for item in current]


def write_csv(path: Path, manifest: list[dict[str, Any]]) -> None:
    fields = [
        "dataset_id", "dataset", "model_id", "model", "comparison_role", "seed",
        "qualification", "next_action", "epochs", "best_val_nll_epoch", "best_val_nll",
        "source_revision", "reproducibility_mode", "split_mode", "evaluation_scope",
        "held_out_test_evaluated", "candidate_count", "reasons", "artifact_root", "run_dir",
        "best_val_nll_checkpoint_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in manifest:
            row = {key: item.get(key, "") for key in fields}
            row["reasons"] = ";".join(item["reasons"])
            writer.writerow(row)


def write_markdown(path: Path, manifest: list[dict[str, Any]], revision: str) -> None:
    counts = Counter(item["qualification"] for item in manifest)
    by_model = Counter((item["model"], item["qualification"]) for item in manifest)
    lines = [
        "# Artifact qualification manifest", "",
        f"> Generated against frozen source revision `{revision}`.",
        "> This report contains validation provenance only. It does not qualify held-out test results.", "",
        "## Current status", "",
        f"- Expected run contracts: {len(manifest)}",
        f"- Final-comparison ready: {counts[QUAL_READY]}",
        f"- Draft only: {counts[QUAL_DRAFT]}",
        f"- Pending active run: {counts[QUAL_PENDING]}",
        f"- No matching artifact: {counts[QUAL_MISSING]}", "",
        "## Qualification by model", "",
        "| Model | Ready | Draft only | Pending | Missing |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for model in sorted({item["model"] for item in manifest}):
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                model,
                by_model[(model, QUAL_READY)], by_model[(model, QUAL_DRAFT)],
                by_model[(model, QUAL_PENDING)], by_model[(model, QUAL_MISSING)],
            )
        )
    lines.extend([
        "", "## Run-level decision", "",
        "| Dataset | Model | Seed | Qualification | Epochs | Best epoch | Reasons |",
        "| :--- | :--- | ---: | :--- | ---: | ---: | :--- |",
    ])
    for item in manifest:
        reason = ", ".join(item["reasons"]) or "-"
        lines.append(
            f"| {item['dataset']} | {item['model']} | {item['seed']} | {item['qualification']} | "
            f"{item['epochs']} | {item['best_val_nll_epoch']} | {reason} |"
        )
    lines.extend([
        "", "## Decision rules", "",
        "A run is final-comparison ready only when its model contract matches, the epoch budget is 300 or an approved continuation to 800, strict reproducibility and validation-only evaluation are recorded, the source revision and fixed-split hashes match, no held-out test evidence exists, and the best-validation-NLL checkpoint and state hash are present.",
        "",
        "A draft-only artifact may support preliminary discussion, but it must be rerun before entering the final comparison table. Pending rows belong to the active frozen launcher and are reclassified after completion.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    contract_path = rooted(project_root, args.contract)
    dataset_contract_path = rooted(project_root, args.dataset_contract)
    artifact_root = rooted(project_root, args.artifact_root)
    active_root = rooted(project_root, args.active_run_root) if args.active_run_root else None
    output_dir = rooted(project_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = read_json(contract_path)
    dataset_contract = read_json(dataset_contract_path)
    manifest = build_manifest(
        project_root,
        contract,
        dataset_contract,
        load_rows(artifact_root),
        load_active_contract(active_root),
    )
    merge_paths = [rooted(project_root, path) for path in args.merge_manifest]
    imported_manifests = [read_json(path)["runs"] for path in merge_paths]
    manifest = merge_manifests(manifest, imported_manifests)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "contract": str(contract_path),
        "artifact_root": str(artifact_root),
        "active_run_root": str(active_root) if active_root else None,
        "merged_manifests": [str(path) for path in merge_paths],
        "benchmark_source_revision": contract["benchmark_source_revision"],
        "summary": dict(Counter(item["qualification"] for item in manifest)),
        "runs": manifest,
    }
    (output_dir / "final_fair_artifact_manifest.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    write_csv(output_dir / "final_fair_artifact_manifest.csv", manifest)
    write_markdown(
        output_dir / "final_fair_artifact_manifest.md",
        manifest,
        contract["benchmark_source_revision"],
    )


if __name__ == "__main__":
    main()
