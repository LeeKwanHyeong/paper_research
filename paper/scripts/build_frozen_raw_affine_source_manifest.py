#!/usr/bin/env python3
"""Build a full-file manifest for a committed raw-affine source snapshot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def git_output(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def committed_file_sha256(revision: str, path: str) -> str:
    content = subprocess.run(
        ["git", "show", f"{revision}:{path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def build_manifest(revision: str) -> dict:
    full_revision = git_output("rev-parse", f"{revision}^{{commit}}")
    if len(full_revision) != 40:
        raise ValueError("A full Git commit is required")
    paths = git_output("ls-tree", "-r", "--name-only", full_revision).splitlines()
    files = {path: committed_file_sha256(full_revision, path) for path in paths}
    required = {
        "paper/contracts/frozen_raw_affine_calibration_v1.json",
        "paper/scripts/frozen_raw_affine_calibration.py",
        "paper/scripts/build_frozen_raw_affine_source_manifest.py",
        "paper/scripts/control_frozen_raw_affine_5080.py",
        "paper/scripts/run_frozen_raw_affine_calibration.py",
        "paper/scripts/summarize_frozen_raw_affine_calibration.py",
        "paper/scripts/count_aware_tpp_backbone/core.py",
        "paper/scripts/run_hard_lmm_time_head_refit.py",
        "paper/scripts/run_intermittent_log_backbone_control.py",
        "models/TPPs/CountAwareTPP.py",
        "models/TPPs/CountAwareFactory.py",
        "data_loader/event_seq_data_module.py",
        "simple_lab_test/search/common/runner.py",
    }
    if not required <= set(files):
        raise ValueError(f"Committed source is incomplete: {sorted(required - set(files))}")
    return {
        "schema": "frozen_raw_affine_source_manifest_v1",
        "source_revision": full_revision,
        "git_tree": git_output("rev-parse", f"{full_revision}^{{tree}}"),
        "file_count": len(files),
        "files": files,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite source manifest: {args.output}")
    manifest = build_manifest(args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({key: manifest[key] for key in ("source_revision", "git_tree", "file_count")}))


if __name__ == "__main__":
    main()
