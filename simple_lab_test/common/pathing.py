"""
Project-root bootstrap helpers for scripts and notebooks.

The `simple_lab_test` directory now contains multiple nested folders. Scripts
that run from those folders should not guess the repository root from a fixed
number of `..` hops. Instead, they can search upward until they find the core
project directories used by this repo.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_SENTINELS = ("models", "utils", "sample_data")


def _iter_candidate_roots(start: Path) -> list[Path]:
    """
    Generate a stable list of candidate directories to inspect.

    We keep the search simple and deterministic: only the current path and its
    parents are considered. This is enough because all scripts/notebooks live
    inside the repository tree.
    """
    resolved = start.resolve()
    base = resolved if resolved.is_dir() else resolved.parent
    return [base, *base.parents]


def resolve_project_root(start: str | Path | None = None) -> Path:
    """
    Find the repository root by searching for the required top-level folders.
    """
    anchor = Path(start) if start is not None else Path.cwd()
    for candidate in _iter_candidate_roots(anchor):
        if all((candidate / name).exists() for name in ROOT_SENTINELS):
            return candidate

    checked = "\n".join(str(path) for path in _iter_candidate_roots(anchor))
    raise RuntimeError(
        "Could not locate the paper_research project root. Checked:\n"
        f"{checked}"
    )


def ensure_project_root_on_path(start: str | Path | None = None) -> Path:
    """
    Resolve the project root and register it on `sys.path` for imports.
    """
    project_root = resolve_project_root(start)
    root_str = str(project_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return project_root
