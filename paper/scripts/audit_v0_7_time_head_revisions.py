#!/usr/bin/env python3
"""Audit the source-level time-head contract behind frozen v0.7 results.

The audit reads committed blobs directly from the local Git object database. It
does not import historical modules or use the network. A failed source or
contract assertion is written to the report and produces a nonzero exit code.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = (
    REPOSITORY_ROOT
    / "paper/results/titantpp_v0_7_validation_freeze_20260905"
    / "time_head_revision_audit.json"
)

EXECUTION_SOURCES = (
    {
        "dataset": "Intermittent-5000",
        "revision": "044add1f3de768d804d9f0269fd0013bd9658a35",
        "path": "paper/scripts/run_count_aware_tpp_backbone_control.py",
    },
    {
        "dataset": "Taxi",
        "revision": "6a01aea9024db9e3ef6cfdd2c3d0219ceb320856",
        "path": "models/TPPs/CountAwareTPP.py",
    },
    {
        "dataset": "Instacart",
        "revision": "28293c43521615be2ed8fad5b043dc9df8e5e457",
        "path": "models/TPPs/CountAwareTPP.py",
    },
)

REPAIR_REVISION = "b1d9e638c68bc7bab9ace6b5c8fa9d0af989c8f7"
MODEL_PATH = "models/TPPs/CountAwareTPP.py"
MODEL_CLASS = "SharedTimeCountModel"
EXPECTED_INTERCEPT_CAP = 300.0
EXPECTED_WD_CAP = 10.0
CURRENT_DEFAULT_INTERCEPT_LIMIT = 30.0


def git_bytes(*arguments: str) -> bytes:
    """Run a read-only Git command and return its exact stdout bytes."""
    completed = subprocess.run(
        ["git", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout


def git_text(*arguments: str) -> str:
    return git_bytes(*arguments).decode("utf-8").strip()


def committed_source(revision: str, path: str) -> tuple[bytes, str, str]:
    source = git_bytes("show", f"{revision}:{path}")
    blob_sha1 = git_text("rev-parse", f"{revision}:{path}")
    source_sha256 = hashlib.sha256(source).hexdigest()
    return source, blob_sha1, source_sha256


def class_definition(tree: ast.Module, class_name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return node
    raise AssertionError(f"class {class_name!r} is missing")


def method_definition(class_node: ast.ClassDef, method_name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == method_name:
            return node
    raise AssertionError(f"method {class_node.name}.{method_name} is missing")


def keyword_default(method: ast.FunctionDef, argument_name: str) -> float | None:
    for argument, default in zip(method.args.kwonlyargs, method.args.kw_defaults):
        if argument.arg == argument_name:
            if isinstance(default, ast.Constant) and isinstance(
                default.value, (int, float)
            ):
                return float(default.value)
            raise AssertionError(
                f"{argument_name} has a non-numeric default: {ast.unparse(default)}"
            )
    return None


def assigned_clamp_max(method: ast.FunctionDef, variable_name: str) -> str:
    """Return the source expression passed as max= in an assigned torch.clamp."""
    for node in ast.walk(method):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == variable_name
            for target in node.targets
        ):
            continue
        for candidate in ast.walk(node.value):
            if not isinstance(candidate, ast.Call):
                continue
            function = candidate.func
            if not isinstance(function, ast.Attribute) or function.attr != "clamp":
                continue
            for keyword in candidate.keywords:
                if keyword.arg == "max":
                    return ast.unparse(keyword.value)
    raise AssertionError(
        f"no assigned clamp max found for {method.name}.{variable_name}"
    )


def function_arguments(method: ast.FunctionDef) -> list[str]:
    arguments = [*method.args.posonlyargs, *method.args.args]
    return [argument.arg for argument in arguments if argument.arg != "self"]


def inspect_model_source(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source.decode("utf-8"))
    model_class = class_definition(tree, MODEL_CLASS)
    initializer = method_definition(model_class, "__init__")
    features = method_definition(model_class, "continuous_features")
    log_f_dt = method_definition(model_class, "log_f_dt")

    feature_identifiers = sorted(
        {
            node.id
            for node in ast.walk(features)
            if isinstance(node, ast.Name)
        }
    )
    feature_source = ast.unparse(features)
    mark_identifiers = [
        identifier
        for identifier in feature_identifiers
        if "mark" in identifier.lower()
    ]

    return {
        "model_class": MODEL_CLASS,
        "mark_free": (
            function_arguments(features) == ["dts", "history_quantities", "mask"]
            and "torch.log1p(dts.float().clamp_min(0.0))" in feature_source
            and "torch.log1p(history_quantities.float().clamp_min(0.0))"
            in feature_source
            and not mark_identifiers
        ),
        "mark_free_evidence": {
            "continuous_feature_arguments": function_arguments(features),
            "feature_terms": ["log1p(delta_t)", "log1p(history_quantity)"],
            "mark_identifiers": mark_identifiers,
        },
        "configured_time_intercept_limit_default": keyword_default(
            initializer, "time_intercept_limit"
        ),
        "log_f_dt": {
            "intercept_clamp_max_expression": assigned_clamp_max(
                log_f_dt, "intercept"
            ),
            "wd_clamp_max_expression": assigned_clamp_max(log_f_dt, "wd"),
        },
    }


def inspect_legacy_intercept_sites(source: bytes) -> dict[str, str]:
    tree = ast.parse(source.decode("utf-8"))
    model_class = class_definition(tree, MODEL_CLASS)
    sites: dict[str, str] = {}
    for method_name in ("log_f_dt", "log_survival_dt", "predict_time_median"):
        method = method_definition(model_class, method_name)
        sites[method_name] = assigned_clamp_max(method, "intercept")
    return sites


def append_check(
    checks: list[dict[str, Any]],
    check_id: str,
    expected: Any,
    observed: Any,
) -> None:
    checks.append(
        {
            "id": check_id,
            "expected": expected,
            "observed": observed,
            "passed": observed == expected,
        }
    )


def build_report() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    execution_entries: list[dict[str, Any]] = []

    for specification in EXECUTION_SOURCES:
        revision = specification["revision"]
        path = specification["path"]
        resolved_revision = git_text("rev-parse", f"{revision}^{{commit}}")
        source, blob_sha1, source_sha256 = committed_source(revision, path)
        inspected = inspect_model_source(source)
        dataset_key = specification["dataset"].lower().replace("-", "_")

        append_check(
            checks,
            f"{dataset_key}.revision",
            revision,
            resolved_revision,
        )
        append_check(checks, f"{dataset_key}.mark_free", True, inspected["mark_free"])
        append_check(
            checks,
            f"{dataset_key}.log_f_dt.intercept_cap",
            repr(EXPECTED_INTERCEPT_CAP),
            inspected["log_f_dt"]["intercept_clamp_max_expression"],
        )
        append_check(
            checks,
            f"{dataset_key}.log_f_dt.wd_cap",
            repr(EXPECTED_WD_CAP),
            inspected["log_f_dt"]["wd_clamp_max_expression"],
        )

        execution_entries.append(
            {
                **specification,
                "git_blob_sha1": blob_sha1,
                "source_sha256": source_sha256,
                **inspected,
                "executed_legacy_caps": {
                    "intercept_max": EXPECTED_INTERCEPT_CAP,
                    "wd_max": EXPECTED_WD_CAP,
                },
            }
        )

    repair_parent = git_text("rev-parse", f"{REPAIR_REVISION}^")
    repair_resolved = git_text("rev-parse", f"{REPAIR_REVISION}^{{commit}}")
    parent_source, parent_blob, parent_sha256 = committed_source(
        repair_parent, MODEL_PATH
    )
    repair_source, repair_blob, repair_sha256 = committed_source(
        REPAIR_REVISION, MODEL_PATH
    )
    parent_sites = inspect_legacy_intercept_sites(parent_source)
    repair_sites = inspect_legacy_intercept_sites(repair_source)

    append_check(checks, "repair.revision", REPAIR_REVISION, repair_resolved)
    for method_name in ("log_f_dt", "log_survival_dt", "predict_time_median"):
        append_check(
            checks,
            f"repair.parent.{method_name}.intercept_cap",
            repr(EXPECTED_INTERCEPT_CAP),
            parent_sites[method_name],
        )
        append_check(
            checks,
            f"repair.commit.{method_name}.intercept_cap",
            "self.time_intercept_limit",
            repair_sites[method_name],
        )

    current_revision = git_text(
        "log", "-1", "--format=%H", "HEAD", "--", MODEL_PATH
    )
    current_source, current_blob, current_sha256 = committed_source(
        current_revision, MODEL_PATH
    )
    current_inspected = inspect_model_source(current_source)
    current_worktree_source = (REPOSITORY_ROOT / MODEL_PATH).read_bytes()
    current_worktree_sha256 = hashlib.sha256(current_worktree_source).hexdigest()

    append_check(
        checks,
        "current.default_time_intercept_limit",
        CURRENT_DEFAULT_INTERCEPT_LIMIT,
        current_inspected["configured_time_intercept_limit_default"],
    )
    append_check(
        checks,
        "current.log_f_dt.intercept_cap_expression",
        "self.time_intercept_limit",
        current_inspected["log_f_dt"]["intercept_clamp_max_expression"],
    )
    append_check(
        checks,
        "current.log_f_dt.wd_cap",
        repr(EXPECTED_WD_CAP),
        current_inspected["log_f_dt"]["wd_clamp_max_expression"],
    )
    append_check(
        checks,
        "current.worktree_matches_head",
        current_sha256,
        current_worktree_sha256,
    )

    drift_entries = []
    for entry in execution_entries:
        current_path = entry["path"]
        path_exists_at_head = (
            subprocess.run(
                ["git", "cat-file", "-e", f"HEAD:{current_path}"],
                cwd=REPOSITORY_ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )
        if path_exists_at_head:
            head_path_source, head_path_blob, head_path_sha256 = committed_source(
                "HEAD", current_path
            )
            del head_path_source
        else:
            head_path_blob = None
            head_path_sha256 = None
        drift_entries.append(
            {
                "dataset": entry["dataset"],
                "execution_path": current_path,
                "execution_git_blob_sha1": entry["git_blob_sha1"],
                "head_path_exists": path_exists_at_head,
                "head_git_blob_sha1": head_path_blob,
                "head_source_sha256": head_path_sha256,
                "source_changed_since_execution": (
                    head_path_sha256 != entry["source_sha256"]
                ),
                "semantic_drift": {
                    "execution_intercept_cap": EXPECTED_INTERCEPT_CAP,
                    "current_default_intercept_cap": CURRENT_DEFAULT_INTERCEPT_LIMIT,
                    "wd_cap_unchanged": EXPECTED_WD_CAP,
                },
            }
        )

    all_passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "audit_scope": "TitanTPP v0.7 frozen validation time-head execution contract",
        "status": "PASS" if all_passed else "FAIL",
        "execution_sources": execution_entries,
        "repair_commit": {
            "revision": REPAIR_REVISION,
            "parent_revision": repair_parent,
            "path": MODEL_PATH,
            "parent_git_blob_sha1": parent_blob,
            "parent_source_sha256": parent_sha256,
            "commit_git_blob_sha1": repair_blob,
            "commit_source_sha256": repair_sha256,
            "legacy_intercept_clamp_before": parent_sites,
            "legacy_intercept_clamp_after": repair_sites,
            "change": (
                "Replaced the hard-coded legacy intercept maximum 300.0 with "
                "self.time_intercept_limit in log density, survival, and median paths."
            ),
        },
        "current_code": {
            "revision": current_revision,
            "path": MODEL_PATH,
            "git_blob_sha1": current_blob,
            "source_sha256": current_sha256,
            "worktree_source_sha256": current_worktree_sha256,
            "worktree_matches_head": current_worktree_sha256 == current_sha256,
            "configured_time_intercept_limit_default": (
                current_inspected["configured_time_intercept_limit_default"]
            ),
            "log_f_dt": current_inspected["log_f_dt"],
        },
        "current_code_drift": drift_entries,
        "held_out_execution_consequence": (
            "Held-out evaluation must run each frozen model row from its original "
            "source revision, or use a separately verified output-equivalent "
            "compatibility path that forces legacy intercept cap 300.0 and wd cap "
            "10.0. The current default time_intercept_limit=30.0 changes the legacy "
            "time score and cannot be mixed with the frozen validation rows."
        ),
        "checks": checks,
    }


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    temporary_path = REPORT_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(serialized, encoding="utf-8")
    temporary_path.replace(REPORT_PATH)


def main() -> int:
    try:
        report = build_report()
    except Exception as error:  # Preserve a machine-readable failure artifact.
        report = {
            "schema_version": 1,
            "audit_scope": "TitanTPP v0.7 frozen validation time-head execution contract",
            "status": "ERROR",
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
        }
        write_report(report)
        print(f"time-head revision audit ERROR: {error}", file=sys.stderr)
        return 2

    write_report(report)
    failed_checks = [check for check in report["checks"] if not check["passed"]]
    if failed_checks:
        print(
            f"time-head revision audit FAIL: {len(failed_checks)} check(s) failed",
            file=sys.stderr,
        )
        return 1
    print(
        "time-head revision audit PASS: "
        f"{len(report['checks'])} checks; report={REPORT_PATH.relative_to(REPOSITORY_ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
